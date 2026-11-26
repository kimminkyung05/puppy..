"""UF 사람 기준선 임시 채점 (dev 만). 정식 채점이 아니다.

지표 정의는 docs/METRICS_FINAL.md (정확 매핑이 공식, 정확 + 유사 행은 참고값).

사람 기준 = UF 페이지 설문 칸 'Top Responses' (human_UF.csv). 개마다 응답자가 많이 고른 품종 5개가 순서대로 있고
비율은 없다. 그래서 포착 혈통 비율@3 과 Top-1 만 낸다 (분포 겹침은 비율이 없어 내지 않는다).

채점 공간 두 가지
  (1) 모델과 같은 공간: 응답 → DNA 표기 → breed_map_UF_draft.csv → Tsinghua 클래스 묶음. provisional_score_uf.py 와
      같은 정답(truth_UF.csv) · 같은 상한 · 같은 찍기(학습셋 클래스 빈도)로 채점한다. A · B 표에 바로 붙일 수 있는 값.
      매핑 밖 품종(예: Russell Terrier)을 맞혀도 0 이다. 모델도 그 품종을 낼 수 없으므로 같은 조건이다.
  (2) DNA 표기 공간: 매핑 없이 응답과 DNA 품종 이름을 바로 맞춘다 (MuttMix 식 사람 점수). 참고값.
      정답은 DNA 원래 비율(pct / 100), 상한은 DNA 상위 3품종 비율 합이다.

응답 고르기
  - 기본(breeds): '기타'(No Predominant Breed) 를 건너뛰고 앞에서부터 품종 응답 3개.
  - as_given: 적힌 순서 그대로 앞 3개 ('기타' 도 한 자리를 차지하고 아무것도 맞히지 못한다).
  - 응답 하나가 DNA 품종 여럿을 가리키면(예: 'Dachshund (including ...)') 그 품종들을 모두 가리킨 것으로 본다.

응답 표기 → DNA 표기 별칭(ALIAS)은 B 가 두 표기 목록을 보고 정했다 (팀 확인 필요). 별칭에 없고 DNA 표기와 대소문자 무시로
같지 않은 응답은 아무 DNA 품종도 가리키지 않는다(그 품종이 평가셋 DNA 에 없다).

사용 (B/ 안에서. 찍기 기준을 위해 Tsinghua 분할 파일 경로가 필요하다)
  $env:DOGMIX_PATHS = "..\\data\\paths_local.yaml"
  python scripts/score_human_uf.py --uf-dir ..\\data\\evalset_uf --out results/dev_human_baseline.csv
출력 CSV 에는 집계값만 있다 (개별 개의 DNA · 응답은 넣지 않는다).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common.config import load_yaml  # noqa: E402
from src.common.tsinghua import class_list, filter_classes, parse_split  # noqa: E402
from provisional_score_uf import (EXACT, SIMILAR, TOPK, breed_of_class, captured, ceiling,  # noqa: E402
                                  load_truth, merge_to_breeds, paired, split_classes, top1_answers)

NOT_BREED_KIND = "기타"

# 설문 응답 표기 → DNA 표기 (B 판단, 팀 확인 필요)
ALIAS = {
    "American Eskimo Dog": ["American Eskimo"],
    "Anatolian Shepherd Dog": ["Anatolian Shepherd"],
    "Argentine Dogo": ["Dogo Argentino"],
    "Brittany": ["Brittany Spaniel"],
    "Bull Terrier (including miniature)": ["Bull Terrier", "Miniature Bull Terrier"],
    "Bulldog (English)": ["Bulldog"],
    "Corgi (including Cardigan, Pembroke)": ["Pembroke Welsh Corgi"],
    "Dachshund (including miniature, standard, smooth coat, wirehair, longhair)": [
        "Long Haired Dachshund", "Miniature Short Haired Dachshund", "Short Haired Dachshund", "Wire Haired Dachshund"],
    "Dobermann Pinscher": ["Doberman Pinscher"],
    "Fox Terrier (including smooth, wire, toy)": ["Fox Terrier", "Wire Fox Terrier", "Toy Fox Terrier"],
    "Foxhound (including American, English, Treeing Walker Coonhound)": ["American Foxhound"],
    "German Shepherd Dog": ["German Shepherd"],
    "Ibizan Hound": ["Ibizian Hound"],
    "Jack Russell Terrier (includes Russell Terrier)": ["Russell Terrier"],
    "Malamute, Alaskan": ["Alaskan Malamute"],
    "Pointer (includes English Pointer)": ["Pointer"],
    "Poodle (including toy, miniature, standard)": ["Poodle", "Miniature Poodle"],
    "Redbone Coonhound": ["RedBone Coonhound"],
    "Schnauzer (including miniature, standard, giant)": ["Miniature Schnauzer", "Standard Schnauzer"],
    "Shar-Pei, Chinese": ["Chinese Sharpei"],
    "Springer Spaniel (including English, Welsh)": ["English Springer Spaniel"],
    "Weimaraner": ["Weimeraner"],
    "Wheaten Terrier, Soft Coated": ["Soft Coated Wheaton Terrier", "Wheaten Terrier"],
}


def norm_key(s) -> str:
    return "; ".join(split_classes(s))


def dna_names(resp: str, dna_lower: set[str]) -> set[str]:
    """응답 하나 → 가리키는 DNA 표기(소문자) 집합."""
    return {n.lower() for n in ALIAS.get(resp, [resp]) if n.lower() in dna_lower}


def picks(h: pd.DataFrame, mode: str) -> dict[str, list[str]]:
    """개 → 채점에 쓸 응답 3개."""
    out = {}
    for dog, g in h.sort_values("order").groupby("dog_id"):
        if mode == "breeds":
            g = g[g["kind"] != NOT_BREED_KIND]
        out[dog] = g["response_raw"].tolist()[:TOPK]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uf-dir", default="../data/evalset_uf")
    ap.add_argument("--split", default="dev", choices=["dev", "hold"], help="hold 는 최종 채점 때 한 번만")
    ap.add_argument("--out", default=None, help="기본 results/{split}_human_baseline.csv")
    args = ap.parse_args()
    args.out = args.out or f"results/{args.split}_human_baseline.csv"
    uf = Path(args.uf_dir)

    idx = pd.read_csv(uf / "eval_index_UF.csv")
    dev = sorted(set(idx.loc[idx["split"] == args.split, "dog_id"]))       # 기본 dev 에서는 hold 를 읽지도 않는다
    human = pd.read_csv(uf / "human_UF.csv", encoding="utf-8-sig")
    human = human[human["dog_id"].isin(dev)]
    dna = pd.read_csv(uf / "dna_UF.csv", encoding="utf-8-sig")
    dna = dna[dna["dog_id"].isin(dev)]
    bmap = pd.read_csv(uf / "breed_map_UF_draft.csv", encoding="utf-8-sig")
    all_dna_lower = {b.lower() for b in pd.read_csv(uf / "dna_UF.csv", encoding="utf-8-sig")["breed_raw"]}
    name_map = {r["breed_raw"].lower(): (r["status"], norm_key(r["tsinghua_classes"])) for _, r in bmap.iterrows()}

    # 찍기 기준: 학습셋 클래스 빈도 (provisional_score_uf.py 와 같은 정의)
    paths = load_yaml("paths")
    ccfg = paths.get("classes", {})
    train = filter_classes(parse_split(paths["tsinghua"]["train_list"]),
                           ccfg.get("exclude", []), ccfg.get("exclude_dingo", False))
    classes = class_list(train)
    freq = pd.Series([it.class_name for it in train]).value_counts(normalize=True).reindex(classes).fillna(0)
    print(f"찍기 기준: 학습셋 {len(train)}장, {len(classes)}클래스 | {args.split} {len(dev)}마리")

    unknown = sorted({r for r in human.loc[human["kind"] != NOT_BREED_KIND, "response_raw"]
                      if not dna_names(r, all_dna_lower)})
    print(f"DNA 표기로 옮길 수 없는 dev 응답 {len(unknown)}종 (평가셋 DNA 에 없는 품종): {', '.join(unknown)}")

    rows = []
    for policy, allowed in [("exact", {EXACT}), ("exact+similar", {EXACT, SIMILAR})]:
        orig, norm, _ = load_truth(uf / "truth_UF.csv", with_similar=(policy != "exact"))
        orig = {d: {norm_key(k): v for k, v in t.items()} for d, t in orig.items() if d in dev}
        norm = {d: {norm_key(k): v for k, v in t.items()} for d, t in norm.items() if d in dev}
        c2b = {c: norm_key(b) for c, b in breed_of_class(uf / "breed_map_UF_draft.csv", allowed).items()}
        chance = merge_to_breeds(pd.DataFrame([freq.to_numpy()] * len(dev), columns=classes, index=dev), c2b)

        def targets(resp):
            out = set()
            for n in dna_names(resp, all_dna_lower):
                st, key = name_map.get(n, (None, ""))
                if st in allowed and key:
                    out.add(key)
            return out

        for mode in ["breeds", "as_given"]:
            pk = picks(human, mode)
            for kind, truth in [("orig", orig), ("norm", norm)]:
                use = [d for d in dev if d in truth and (kind == "orig" or truth[d])]
                hum = pd.Series({d: sum(truth[d].get(t, 0.0) for t in set().union(*[targets(r) for r in pk[d]] or [set()]))
                                 for d in use}, dtype=float)
                ch = captured(chance, truth).reindex(use)
                diff, se2, verdict = paired(hum, ch)
                verdict = {"앞쪽이 높음": "찍기보다 높음", "뒤쪽이 높음": "찍기보다 낮음"}.get(verdict, verdict)
                rows.append(dict(space="model", mapping=policy, picks=mode, truth=kind, metric=f"captured@{TOPK}",
                                 n=len(use), human=hum.mean(), chance=ch.mean(), ceiling=ceiling(truth, use).mean(),
                                 diff_vs_chance=diff, two_se=se2, judgement=verdict))

        # Top-1: METRICS_FINAL.md 3.4절 (DNA 1위 품종, 동률이면 그중 하나). 사람은 첫 품종 응답.
        # 공식 값은 정확 매핑 행, 정확 + 유사 행은 참고값
        ans = top1_answers(dna, bmap, allowed, set(classes), dev)
        first = picks(human, "breeds")
        use = [d for d in dev if ans[d]]
        hit = 0
        for d in use:
            cls = set()
            for key in targets(first[d][0]) if first[d] else set():
                cls.update(split_classes(key))
            hit += bool(cls & ans[d])
        rows.append(dict(space="model", mapping=policy, picks="first_breed", truth="dominant", metric="top1",
                         n=len(use), human=hit / len(use) if use else float("nan"), hits=hit,
                         coverage=len(use) / len(dev)))

    # (2) DNA 표기 공간 (매핑 없음, 원래 비율)
    dtruth = {}
    for d, g in dna[dna["pct"].notna()].groupby("dog_id"):
        dtruth[d] = (g.assign(b=g["breed_raw"].str.lower()).groupby("b")["pct"].sum().astype(float) / 100).to_dict()
    for mode in ["breeds", "as_given"]:
        pk = picks(human, mode)
        use = [d for d in dev if d in dtruth]
        hum = pd.Series({d: sum(dtruth[d].get(n, 0.0) for n in set().union(*[dna_names(r, all_dna_lower) for r in pk[d]] or [set()]))
                         for d in use}, dtype=float)
        rows.append(dict(space="dna_name", mapping="none", picks=mode, truth="orig", metric=f"captured@{TOPK}",
                         n=len(use), human=hum.mean(), ceiling=ceiling(dtruth, use).mean()))
    first = picks(human, "breeds")
    hit = 0
    for d in dev:
        g = dna[(dna["dog_id"] == d) & dna["pct"].notna()]
        top = {b.lower() for b in g.loc[g["pct"].astype(float) == g["pct"].astype(float).max(), "breed_raw"]}
        hit += bool(first[d] and dna_names(first[d][0], all_dna_lower) & top)
    rows.append(dict(space="dna_name", mapping="none", picks="first_breed", truth="dominant", metric="top1",
                     n=len(dev), human=hit / len(dev), hits=hit))

    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(3).to_string(index=False))
    print(f"\n저장 → {out}  (집계값만)")


if __name__ == "__main__":
    main()
