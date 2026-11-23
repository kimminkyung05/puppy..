"""UF 초안: 믹스견별 커버리지와 정답 비율 (truth_UF.csv).

입력 ({dir} = data/evalset_uf):
  dna_UF.csv, dogs_UF.csv            parse_uf_dna.py 가 만든 것
  breed_map_UF_draft.csv             draft_uf_breed_map.py 가 만들고 사람이 고친 것
출력:
  truth_UF.csv  긴 형식. 믹스견(is_purebred=False)만. 한 줄 = 개 한 마리의 매핑된 품종 하나.
    dog_id, coverage, coverage_with_similar, pct_sum, target_classes, status, breed_raw,
    pct, share_norm, share_norm_with_similar, notes

뜻
  - coverage              status=정확 으로 학습 클래스에 매핑된 DNA 비율의 합 / 100
  - coverage_with_similar status=정확 + 유사 로 매핑된 비율의 합 / 100
  - pct_sum               페이지에 적힌 비율의 합(%). 페이지는 개마다 품종을 4개까지만 적어서 100 보다 작은 개가 많다.
                          적히지 않은 나머지는 "모름"이며 coverage 의 분모(100)에는 들어간다.
  - target_classes        매핑된 학습 클래스(; 구분). 묶음(Poodle 계열 등)은 여러 클래스가 한 정답 품종이다.
  - pct                   원래 비율(%). 같은 개에서 여러 표기가 같은 target 으로 가면 더한다.
  - share_norm            status=정확 인 줄만으로 합 1 이 되게 다시 나눈 비율 (유사 줄은 빈칸)
  - share_norm_with_similar  정확 + 유사 줄로 합 1 이 되게 다시 나눈 비율
  - 매핑된 품종이 하나도 없는 믹스견은 target_classes 가 빈 줄 하나로 남긴다 (coverage 0).

주의
  - flag 가 붙은 DNA 줄(퍼센트 기호 없음 등)은 읽힌 숫자를 그대로 쓰고 notes 에 "flag 미확정"을 적는다.
  - 순종(한 품종 100%)은 여기에 넣지 않고 요약에만 보인다.

사용 (프로젝트 루트에서, PowerShell):
  $env:DOGMIX_PATHS = "data\\paths_local.yaml"
  python scripts/build_uf_truth.py --dir data/evalset_uf
"""
import argparse
import collections
import csv
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import tsinghua  # noqa: E402
from src.common.config import load_yaml  # noqa: E402

EXACT, SIMILAR, NONE = "정확", "유사", "없음"


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def train_classes() -> list[str]:
    paths = load_yaml("paths")
    items = tsinghua.parse_split(paths["tsinghua"]["train_list"])
    items = tsinghua.filter_classes(items, paths["classes"]["exclude"], paths["classes"]["exclude_dingo"])
    return tsinghua.class_list(items)


def load_map(path: Path, classes: list[str]) -> dict:
    """breed_raw → (target_classes 문자열, status). 클래스 이름과 status 를 검사한다."""
    out, errors = {}, []
    for r in read_csv(path):
        names = [c.strip() for c in r["tsinghua_classes"].split(";") if c.strip()]
        st = r["status"].strip()
        if st not in (EXACT, SIMILAR, NONE):
            errors.append(f"{r['breed_raw']}: status '{st}'")
        if st != NONE and not names:
            errors.append(f"{r['breed_raw']}: status={st} 인데 클래스가 비어 있음")
        errors += [f"{r['breed_raw']}: 학습 클래스에 없는 이름 '{c}'" for c in names if c not in classes]
        out[r["breed_raw"]] = ("; ".join(names) if st != NONE else "", st)
    if errors:
        raise SystemExit("breed_map_UF_draft.csv 오류:\n  " + "\n  ".join(errors))
    return out


def bins(values: list[float]) -> str:
    lo = sum(v < 0.5 for v in values)
    mid = sum(0.5 <= v < 0.8 for v in values)
    hi = sum(v >= 0.8 for v in values)
    return f"0.5 미만 {lo:>3} | 0.5~0.8 {mid:>3} | 0.8 이상 {hi:>3}"


def num(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if x else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/evalset_uf")
    args = ap.parse_args()
    d = Path(args.dir)

    classes = train_classes()
    bmap = load_map(d / "breed_map_UF_draft.csv", classes)
    dogs = read_csv(d / "dogs_UF.csv")
    dna = collections.defaultdict(list)
    for r in read_csv(d / "dna_UF.csv"):
        dna[r["dog_id"]].append(r)
    unmapped = sorted({r["breed_raw"] for rs in dna.values() for r in rs} - set(bmap))
    if unmapped:
        raise SystemExit(f"매핑 표에 없는 품종 표기: {unmapped}")

    rows, stats, pure = [], [], []
    for dog in dogs:
        did = dog["dog_id"]
        if dog["is_purebred"] == "True":
            b = dna[did][0]["breed_raw"]
            pure.append((did, b, bmap[b][1], bmap[b][0]))
            continue
        notes = []
        if any(r["flag"] for r in dna[did]):
            notes.append("flag 미확정")
        if any(r["pct"] == "" for r in dna[did]):
            notes.append("숫자 못 읽은 줄 제외")
        if float(dog["pct_sum"]) > 100.005:
            notes.append("합 100 초과")
        # 같은 target 으로 가는 표기는 더한다
        agg = collections.OrderedDict()
        for r in dna[did]:
            target, st = bmap[r["breed_raw"]]
            if st == NONE or r["pct"] == "":
                continue
            a = agg.setdefault(target, {"pct": 0.0, "status": set(), "raw": []})
            a["pct"] += float(r["pct"])
            a["status"].add(st)
            a["raw"].append(r["breed_raw"])
        for a in agg.values():                       # 정확과 유사가 섞이면 유사로 본다
            a["st"] = SIMILAR if SIMILAR in a["status"] else EXACT
        exact = sum(a["pct"] for a in agg.values() if a["st"] == EXACT)
        both = sum(a["pct"] for a in agg.values())
        cov, cov_s = exact / 100, both / 100
        stats.append({"dog_id": did, "cov": cov, "cov_s": cov_s, "sum": float(dog["pct_sum"]),
                      "n_exact": sum(a["st"] == EXACT for a in agg.values()), "n_both": len(agg)})
        head = [did, num(cov), num(cov_s), dog["pct_sum"]]
        if not agg:
            rows.append(head + ["", "", "", "", "", "", " | ".join(notes)])
        for target, a in agg.items():
            rows.append(head + [
                target, a["st"], "; ".join(a["raw"]), num(a["pct"], 2),
                num(a["pct"] / exact) if a["st"] == EXACT and exact > 0 else "",
                num(a["pct"] / both) if both > 0 else "",
                " | ".join(notes)])

    target_file = d / "truth_UF.csv"
    tmp = target_file.with_name(target_file.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dog_id", "coverage", "coverage_with_similar", "pct_sum", "target_classes", "status",
                    "breed_raw", "pct", "share_norm", "share_norm_with_similar", "notes"])
        w.writerows(rows)
    os.replace(tmp, target_file)

    # ---- 요약 ----
    cov = [s["cov"] for s in stats]
    cov_s = [s["cov_s"] for s in stats]
    listed = [s["sum"] / 100 for s in stats]
    print(f"학습 클래스 {len(classes)}개 기준 | 믹스견 {len(stats)}마리, 순종 {len(pure)}마리")
    print("\ncoverage 분포 (믹스견, 분모 100%)")
    print(f"  정확만        : {bins(cov)}   평균 {statistics.mean(cov):.3f}  중앙 {statistics.median(cov):.3f}")
    print(f"  정확 + 유사   : {bins(cov_s)}   평균 {statistics.mean(cov_s):.3f}  중앙 {statistics.median(cov_s):.3f}")
    print(f"  (상한) 적힌 합: {bins(listed)}   평균 {statistics.mean(listed):.3f}  중앙 {statistics.median(listed):.3f}"
          "   ← 모든 품종이 매핑돼도 넘을 수 없는 값")
    print("\ncoverage 0.5 이상인 믹스견 가운데 매핑된 품종 수 (비율 평가에는 2개 이상이 필요)")
    for label, ck, nk in (("정확만", "cov", "n_exact"), ("정확 + 유사", "cov_s", "n_both")):
        ok = [s for s in stats if s[ck] >= 0.5]
        c = collections.Counter(min(s[nk], 4) for s in ok)
        print(f"  {label:<10}: {len(ok)}마리 = 1개 {c[1]} / 2개 {c[2]} / 3개 {c[3]} / 4개 {c[4]}")
    zero = [s["dog_id"] for s in stats if s["cov_s"] == 0]
    print(f"\n매핑된 품종이 하나도 없는 믹스견: {len(zero)}마리 {zero}")
    low = sorted(stats, key=lambda s: s["cov"])[:8]
    print("coverage(정확만) 가장 낮은 8마리:", ", ".join(f"{s['dog_id']}={s['cov']:.2f}" for s in low))
    print("\n순종 (truth_UF.csv 에는 넣지 않음):")
    for did, b, st, target in pure:
        print(f"  {did}  100% {b}  →  {target or '(없음)'}  [{st}]")
    flagged = sorted({r[0] for r in rows if r[10]})
    print(f"\nnotes 가 붙은 개: {flagged}")
    print(f"저장 → {target_file}  ({len(rows)}줄)")


if __name__ == "__main__":
    main()
