"""순종 보조 평가의 임시 채점. 정식 채점이 아니다 (A 의 src/eval/eval.py 가 나오면 그것으로 바꾼다).

질문: 순종 사진을 순종으로 답하는가. 믹스견 평가(provisional_score_uf.py)와 따로 보고하고 한 숫자로 합치지 않는다.
정답은 Tsinghua 가 사진에 붙인 품종 라벨이다. DNA 가 아니다.

입력
  preds/dev/pred_RQ1-val-{추출기}-proto-r{r}[-l2].csv   prototype_fcls --eval-kind purebred 의 출력
                                                        (image_id, label, split, <클래스>). 사진 한 장 = 개 한 마리
  breed_map_UF_draft.csv                                 (b) 기준에서 클래스를 품종으로 묶을 때 쓴다

지표 (사진마다 계산해 평균)
  정답 비율   정답 클래스(품종)에 준 비율
  Top-1       비율이 가장 큰 클래스(품종)가 정답인 사진의 비율
  퍼짐        1 − 정답 비율 (정답이 아닌 곳으로 새어 나간 비율)
  유효 품종 수  exp(엔트로피). 한 품종에 다 주면 1, 고르게 n 개에 주면 n

두 가지 기준
  (a) Tsinghua 클래스 그대로
  (b) 매핑 초안으로 DNA 품종에 묶은 뒤. 같은 품종으로 묶이는 클래스(teddy · toy_poodle · miniature_poodle ·
      standard_poodle 등)끼리의 혼동은 틀린 것으로 세지 않는다.

사용 (프로젝트 루트에서. DOGMIX_PATHS 로 경로 설정을 고른다)
  python scripts/provisional_score_purebred.py --uf-dir data/evalset_uf
  옵션: --with-similar (status=유사 묶음도 적용), --include-check (CHECK- 점검 실행도), --detail all (모든 설정의 상세)
"""
import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_yaml  # noqa: E402

BANNER = "임시 채점, 순종 보조 평가(정답은 사진으로 붙인 품종 라벨이며 DNA가 아님)"
NAME_RE = re.compile(r"^pred_(?P<kind>RQ1|CHECK)-(?P<ev>val[^-]*)-(?P<bb>clip|dinov2|dinov3|sift|dummy)"
                     r"-proto-r(?P<r>\d+)(?P<l2>-l2)?(?P<pool>-mean)?(?P<tr>-tr_.+)?\.csv$")
METRICS = ["정답 비율", "Top-1", "퍼짐", "유효 품종 수"]


def class_to_breed(map_csv: Path, allowed: set[str]) -> dict[str, str]:
    """클래스 → 묶음 이름. 클래스가 둘 이상인 묶음만 남긴다."""
    m = pd.read_csv(map_csv, encoding="utf-8-sig")
    out = {}
    for s in m.loc[m["status"].isin(allowed), "tsinghua_classes"].dropna().unique():
        names = [c.strip() for c in s.split(";") if c.strip()]
        if len(names) > 1:
            for c in names:
                if out.get(c, "; ".join(names)) != "; ".join(names):
                    raise SystemExit(f"매핑 표에서 클래스 '{c}' 가 두 묶음에 걸침")
                out[c] = "; ".join(names)
    return out


def metrics(W: np.ndarray, true: np.ndarray) -> dict:
    """W: (N, K) 행 합 1, true: (N,) 정답 열 번호 → 사진별 값."""
    p_true = W[np.arange(len(W)), true]
    logw = np.log(np.clip(W, 1e-12, None))
    return {"정답 비율": p_true, "Top-1": (W.argmax(axis=1) == true).astype(float),
            "퍼짐": 1 - p_true, "유효 품종 수": np.exp(-(W * logw).sum(axis=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uf-dir", default="data/evalset_uf", help="breed_map_UF_draft.csv 가 있는 폴더")
    ap.add_argument("--preds-root", default=None, help="기본: 경로 설정의 outputs.preds_root")
    ap.add_argument("--with-similar", action="store_true", help="status=유사 인 묶음도 (b) 기준에 적용한다")
    ap.add_argument("--include-check", action="store_true", help="CHECK- (형식 점검) 실행도 채점한다")
    ap.add_argument("--detail", default="best", choices=["best", "all", "none"],
                    help="클래스별 · 혼동 쌍 상세를 보일 설정 (best: (a) Top-1 이 가장 높은 설정)")
    args = ap.parse_args()

    paths = load_yaml("paths")
    preds_dir = Path(args.preds_root or paths["outputs"]["preds_root"]) / "dev"
    allowed = {"정확", "유사"} if args.with_similar else {"정확"}
    c2b = class_to_breed(Path(args.uf_dir) / "breed_map_UF_draft.csv", allowed)

    print("=" * 78)
    print(BANNER)
    print("=" * 78)
    print(f"예측: {preds_dir}")
    groups = sorted(set(c2b.values()))
    print(f"(b) 기준에서 한 품종으로 묶는 클래스 ({' + '.join(sorted(allowed))} 매핑): {groups or '없음'}")

    runs = []
    for p in sorted(preds_dir.glob("pred_*.csv")):
        m = NAME_RE.match(p.name)
        if m and (m["kind"] == "RQ1" or args.include_check):
            runs.append((p, m))
    if not runs:
        print(f"\n채점할 예측 파일이 없음. {preds_dir} 에 pred_RQ1-val-*.csv 가 있는지 확인 (점검 실행은 --include-check)")
        return

    res = collections.defaultdict(dict)      # 설정 → 반복 → {기준: {지표: 평균}}
    detail = collections.defaultdict(lambda: {"cls_hit": collections.defaultdict(list),
                                              "pairs": collections.Counter(), "n": 0})
    print(f"\n예측 파일 {len(runs)}개")
    for p, m in runs:
        df = pd.read_csv(p)
        classes = [c for c in df.columns if c not in ("image_id", "label", "split")]
        known = df["label"].isin(classes)
        notes = []
        if len(classes) != 127:
            notes.append(f"클래스 {len(classes)}개(127 아님)")
        if (~known).any():
            notes.append(f"정답 라벨이 클래스에 없는 사진 {int((~known).sum())}장 제외")
        if set(df["split"]) != {"dev"}:
            notes.append(f"split 이 dev 가 아닌 행 있음: {sorted(set(df['split']))}")
        df = df[known & (df["split"] == "dev")].reset_index(drop=True)
        W = df[classes].to_numpy(dtype=float)
        W = W / W.sum(axis=1, keepdims=True)
        true_a = np.array([classes.index(x) for x in df["label"]])
        # (b) 품종으로 묶기
        breeds = sorted({c2b.get(c, c) for c in classes})
        col = np.array([breeds.index(c2b.get(c, c)) for c in classes])
        Wb = np.zeros((len(W), len(breeds)))
        np.add.at(Wb, (slice(None), col), W)
        true_b = col[true_a]
        ma, mb = metrics(W, true_a), metrics(Wb, true_b)
        cfg = f"{m['bb']}{'+L2' if m['l2'] else ''}{'+mean' if m['pool'] else ''}{' [점검]' if m['kind'] == 'CHECK' else ''}"
        res[cfg][int(m["r"])] = {"a": {k: float(v.mean()) for k, v in ma.items()},
                                 "b": {k: float(v.mean()) for k, v in mb.items()},
                                 "n": len(W), "k": len(classes), "kb": len(breeds)}
        d = detail[cfg]
        top = W.argmax(axis=1)
        for t, h in zip(true_a, ma["Top-1"]):
            d["cls_hit"][classes[t]].append(h)
        for t, q in zip(true_a, top):
            if t != q:
                d["pairs"][(classes[t], classes[q])] += 1
        d["n"] += 1
        print(f"  {p.name}: 사진 {len(W)}, 클래스 {len(classes)}" + (f"  ← {'; '.join(notes)}" if notes else ""))

    cfgs = sorted(res)
    for basis, label in (("a", "(a) Tsinghua 클래스 그대로"), ("b", "(b) 매핑 초안으로 DNA 품종에 묶은 뒤")):
        any_run = next(iter(res[cfgs[0]].values()))
        k = any_run["k"] if basis == "a" else any_run["kb"]
        print(f"\n[{label}]  사진 {any_run['n']}장, 품종 {k}개, 무작위 기준 Top-1 = {1 / k:.3f}")
        print(f"  {'설정':<16}{'반복':>4}" + "".join(f"{name + ' 평균(최소~최대)':>26}" for name in METRICS))
        for cfg in cfgs:
            reps = sorted(res[cfg])
            cells = []
            for name in METRICS:
                v = [res[cfg][r][basis][name] for r in reps]
                cells.append(f"{np.mean(v):.3f} ({min(v):.3f}~{max(v):.3f})")
            print(f"  {cfg:<16}{len(reps):>4}" + "".join(f"{c:>30}" for c in cells))
        for with_l2 in [c for c in cfgs if "+L2" in c and c.replace("+L2", "") in cfgs]:
            base = with_l2.replace("+L2", "")
            d1 = np.mean([res[with_l2][r][basis]["Top-1"] for r in res[with_l2]]) - \
                np.mean([res[base][r][basis]["Top-1"] for r in res[base]])
            d2 = np.mean([res[with_l2][r][basis]["정답 비율"] for r in res[with_l2]]) - \
                np.mean([res[base][r][basis]["정답 비율"] for r in res[base]])
            print(f"  L2 비교 {base}: L2 있음 − 없음 = Top-1 {d1:+.3f}, 정답 비율 {d2:+.3f}")

    if args.detail != "none":
        best = max(cfgs, key=lambda c: np.mean([res[c][r]["a"]["Top-1"] for r in res[c]]))
        for cfg in (cfgs if args.detail == "all" else [best]):
            d = detail[cfg]
            print(f"\n[상세: {cfg}" + (" = (a) Top-1 이 가장 높은 설정" if args.detail == "best" else "")
                  + f"]  반복 {d['n']}회를 합쳐서 셈")
            low = sorted(((float(np.mean(v)), c, len(v)) for c, v in d["cls_hit"].items()))[:10]
            print("  클래스별 Top-1 이 낮은 10개 ((a) 기준)")
            for acc, c, n in low:
                print(f"    {acc:.3f}  {c}  (사진×반복 {n})")
            print("  가장 자주 헷갈리는 쌍 10개 (정답 → 가장 큰 비율을 받은 클래스, 횟수)")
            for (t, q), n in d["pairs"].most_common(10):
                same = c2b.get(t, t) == c2b.get(q, q)
                print(f"    {n:>4}  {t} → {q}" + ("   ← (b) 기준에서는 같은 품종" if same else ""))

    print("\n읽는 법: 이 표는 '순종을 순종으로 답하는가'만 본다. 믹스견 표와 합쳐서 한 숫자로 내지 않는다.")
    print(BANNER)


if __name__ == "__main__":
    main()
