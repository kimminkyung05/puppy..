"""결합 모델 검토 (dev, 참고 분석). 결합 여부를 정하기 위한 근거를 모은다. 연구 질문 판정에는 쓰지 않는다.

질문
  1. 여지: 개마다 가장 좋은 설정을 고르면(오라클) 얼마나 오르나. 결합으로 얻을 수 있는 몫의 상한
  2. 결합 방식: 같은 가중치로 섞는 방법(산술 평균 · 기하 평균 · 순위 합산 · 상위 3 투표)마다 점수가 어떻게 다른가
  3. 정직한 추정: 조합을 dev 로 고르면 20마리에 맞춘 값이 된다. 한 마리씩 빼고 나머지 19마리로
     후보 조합 중 가장 좋은 것을 고른 뒤 뺀 개에서 채점(leave-one-dog-out)해, 고르는 과정까지 포함한 점수를 낸다
  4. 단일 최고 설정과 개체 단위 짝 차이 ± 2SE
  5. 개를 복원 추출(부트스트랩 2,000회)했을 때 결합이 단일 최고(같은 표본에서 다시 고른 최고)를 이기는 비율

입력: collect_rq1_preds.py 가 모은 {preds}/dev/pred_RQ1-{bb}-{clf|proto|mix}-{s|r}{n}.csv
채점: provisional_score_uf.py 함수 (METRICS_FINAL.md, 정확 매핑, 원래 비율 포착@3)
출력: results/dev_combo_analysis.csv (집계값만, 개별 개 값 없음)

사용 (B/ 안에서)
  python scripts/combo_analysis_dev.py --preds ../outputs/rq1_preds/merged --uf-dir ../data/evalset_uf
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provisional_score_uf as P  # noqa: E402
from compare_rq1_dev import METHODS, Scorer, load_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEEP = ["clip", "dinov2", "dinov3"]
N_BOOT = 2000


def combine(probs: list[pd.DataFrame], how: str) -> pd.DataFrame:
    """같은 가중치 결합. 결과는 행 합 1 (상위 3 고르기에는 순서만 중요하다)."""
    if how == "mean":
        out = sum(probs) / len(probs)
    elif how == "geo":       # 기하 평균: 모두가 어느 정도 주는 클래스를 올린다
        out = np.exp(sum(np.log(p + 1e-6) for p in probs) / len(probs))
    elif how == "rank":      # Borda: 각 설정 안에서의 순위만 쓴다 (확률 크기 차이를 없앰)
        out = sum(p.rank(axis=1, method="average") for p in probs)
    elif how == "vote":      # 각 설정의 상위 3 클래스에 1표씩
        out = sum((p.rank(axis=1, ascending=False, method="first") <= 3).astype(float) for p in probs)
        out = out + 1e-3 * sum(probs) / len(probs)       # 동표는 평균 확률로 가른다
    else:
        raise ValueError(how)
    return out.div(out.sum(axis=1), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--uf-dir", type=Path, required=True)
    args = ap.parse_args()

    runs = load_runs(args.preds / "dev")
    ens = {k: sum(fs) / 3 for k, fs in runs.items()}
    first = next(iter(ens.values()))
    sc = Scorer(args.uf_dir, list(first.columns), list(first.index))
    dogs = sc.dogs
    singles = {f"{k} {bb}": ens[(k, bb)] for k, *_ in METHODS for bb in DEEP if (k, bb) in ens}
    S = pd.DataFrame({n: sc(p)["co"] for n, p in singles.items()}).loc[dogs]   # 개 × 설정 (포착@3 원래)
    rows = []

    # 1. 여지
    best_single = S.mean().idxmax()
    rows.append(dict(analysis="1_headroom", config=f"단일 최고 ({best_single})", captured_orig=S[best_single].mean()))
    rows.append(dict(analysis="1_headroom", config="오라클: 개마다 9개 중 최고", captured_orig=S.max(axis=1).mean()))
    rows.append(dict(analysis="1_headroom", config="9개 설정 모두 0인 개 수", captured_orig=float((S.max(axis=1) == 0).sum())))
    rows.append(dict(analysis="1_headroom", config="상한(정답 상위 3)", captured_orig=P.ceiling(sc.orig, dogs).mean()))

    # 2. 결합 방식 × 후보 묶음
    groups = {"딥 3종 × 3방법 (9개)": list(singles)}
    for k, *_ in METHODS:
        groups[f"{k} 딥 3종"] = [n for n in singles if n.startswith(f"{k} ")]
    for bb in DEEP:
        groups[f"{bb} 3방법"] = [n for n in singles if n.endswith(f" {bb}")]
    cand = {}                                    # (방법, 묶음) → 개별 점수
    for how in ["mean", "geo", "rank", "vote"]:
        for g, members in groups.items():
            s = sc(combine([singles[m] for m in members], how))
            cand[(how, g)] = s["co"]
            d, se2, v = P.paired(s["co"], S[best_single])
            rows.append(dict(analysis="2_combine", config=f"{how} | {g}", captured_orig=s["co"].mean(),
                             top1=f"{len(s['hits'])}/{s['n']}", vs_best_single=f"{d:+.3f}±{se2:.3f} {v}"))

    # 3. leave-one-dog-out: 단일 9개 + 결합 후보 전부 중 19마리 평균이 가장 높은 것을 골라 뺀 개에 적용
    pool = {**{("single", n): S[n] for n in singles}, **cand}
    M = pd.DataFrame(pool).loc[dogs]
    loo, picked = [], []
    for d in dogs:
        rest = M.drop(index=d).mean()
        choice = rest.idxmax(); picked.append(choice)
        loo.append(M.loc[d, choice])
    loo = pd.Series(loo, index=dogs)
    d_, se2, v = P.paired(loo, S[best_single])
    rows.append(dict(analysis="3_loo_select", config="단일 + 결합 후보 전부에서 19마리로 고름",
                     captured_orig=loo.mean(), vs_best_single=f"{d_:+.3f}±{se2:.3f} {v}",
                     note=f"고른 후보 종류 {len(set(picked))}개, 가장 많이 고른 것: {pd.Series(picked).value_counts().index[0]}"))
    Ms = M[[c for c in M.columns if c[0] == "single"]]
    loo_s = pd.Series([Ms.loc[d, Ms.drop(index=d).mean().idxmax()] for d in dogs], index=dogs)
    rows.append(dict(analysis="3_loo_select", config="단일 9개에서만 19마리로 고름", captured_orig=loo_s.mean()))

    # 5. 부트스트랩: 같은 표본에서 단일 최고 vs 결합(전부 산술 평균)
    allmean = cand[("mean", "딥 3종 × 3방법 (9개)")]
    rng = np.random.default_rng(20261124)
    wins = 0
    for _ in range(N_BOOT):
        idx = rng.choice(len(dogs), len(dogs), replace=True)
        wins += allmean.iloc[idx].mean() > S.iloc[idx].mean().max()
    rows.append(dict(analysis="5_bootstrap", config="9개 산술 평균 > 같은 표본의 단일 최고",
                     captured_orig=wins / N_BOOT, note=f"{N_BOOT}회 중 비율"))

    out = pd.DataFrame(rows)
    out.round(4).to_csv(ROOT / "results" / "dev_combo_analysis.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(out.round(3).fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
