"""믹스견 dev 예측이 어느 단계에서 실패하는가 (RQ1 보조 분석, 결합 모델 판단의 근거).

개체의 DNA 혈통(원래 비율, 합 1 기준)을 다음 몫으로 나눈다. 설정(딥 3종 × 세 방법 = 9개)마다 계산해 개체 평균을 낸다.
  미보고         1 − DNA 비율 합 (검사 결과가 100% 를 채우지 않은 몫)
  학습 품종 밖   DNA 에는 있으나 정확 매핑된 Tsinghua 클래스가 없는 몫 (매핑 단계에서 이미 잃는 몫)
  먼 순위        정확 매핑된 품종이지만 예측(DNA 품종 공간) 순위가 31위 이하
  중간 순위      11~30위
  가까운 순위    4~10위 (상위 3개 잘림에서 놓침)
  포착           1~3위 (= 주 지표, 포착 혈통 비율)
9개 설정의 합의 (개체 × 정확 매핑 품종, 비율 가중):
  9개 모두 포착 / 일부만 포착(결합으로 얻을 수 있는 몫) / 아무도 포착 못함(그중 9개 모두 11위 이하 = 특징 단계 실패)
풀이 방식 없는 특징 단계 점검 (DINOv2만: 로컬에 학습셋 전체 특징이 있는 추출기):
  학습셋 품종 평균점과의 코사인 유사도로 품종 순위를 매긴다(품종 묶음은 클래스 중 최댓값). 얼굴 칸 · 전신 칸 · 두 칸 평균 특징 각각.

dev 만 쓴다(hold 행은 읽은 뒤 바로 버린다). 출력은 집계값만: results/dev_failure_stages.csv
사용 (B/ 안에서):
  python scripts/failure_stage_dev.py --preds ../outputs/rq1_preds/merged --uf-dir ../data/evalset_uf --features ../outputs/features_drive
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provisional_score_uf as P  # noqa: E402
from compare_rq1_dev import METHODS, Scorer, load_runs  # noqa: E402
from src.common.features_io import load_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEEP = ["clip", "dinov2", "dinov3"]
BUCKETS = [("포착 (1~3위)", 1, 3), ("가까운 순위 (4~10위)", 4, 10), ("중간 순위 (11~30위)", 11, 30), ("먼 순위 (31위~)", 31, 10**9)]
NAME = {"A": "분류기", "B": "평균점 섞기", "C": "섞기 학습"}


def ranks_in(scores: pd.Series) -> pd.Series:
    """점수가 큰 순서의 순위(1부터). 동점은 captured() 와 같은 stable 정렬 순서."""
    order = scores.sort_values(ascending=False, kind="stable").index
    return pd.Series(np.arange(1, len(order) + 1), index=order)


def bucket_shares(rank_of: dict, truth_d: dict) -> dict:
    out = {b: 0.0 for b, *_ in BUCKETS}
    for breed, share in truth_d.items():
        r = rank_of[breed]
        for b, lo, hi in BUCKETS:
            if lo <= r <= hi:
                out[b] += share
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--uf-dir", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=None, help="features_root (DINOv2 특징 단계 점검용, 없으면 건너뜀)")
    args = ap.parse_args()

    runs = load_runs(args.preds / "dev")
    ens = {k: sum(fs) / 3 for k, fs in runs.items() if k[1] in DEEP}
    first = next(iter(ens.values()))
    sc = Scorer(args.uf_dir, list(first.columns), list(first.index))
    dogs = sc.dogs
    t = pd.read_csv(args.uf_dir / "truth_UF.csv", encoding="utf-8-sig")
    t = t[t["dog_id"].isin(dogs)].groupby("dog_id").first()
    reported = (t["pct_sum"] / 100).clip(upper=1.0)
    coverage = t["coverage"]
    base = {"미보고": float((1 - reported).mean()), "학습 품종 밖": float((reported - coverage).mean())}

    rows = []
    rank_tables = {}                       # 설정 → 개 → {품종: 순위}
    for (k, bb), prob in ens.items():
        merged = P.merge_to_breeds(prob.loc[dogs], sc.c2b)
        rk = {d: ranks_in(merged.loc[d]) for d in dogs}
        rank_tables[(k, bb)] = rk
        per = pd.DataFrame({d: bucket_shares(rk[d], sc.orig[d]) for d in dogs}).T
        row = dict(analysis="1_stage_by_setting", config=f"{NAME[k]} {bb}", **base, **per.mean().to_dict())
        rows.append(row)

    # 9개 설정 합의 (개체 × 정확 매핑 품종, 비율 가중)
    cons = {"9개 모두 포착": 0.0, "일부 설정만 포착": 0.0, "아무 설정도 포착 못함 (최고 4~10위)": 0.0,
            "아무 설정도 포착 못함 (9개 모두 11위 이하)": 0.0}
    by_bb = {bb: {"그 추출기의 세 방법 중 하나라도 포착": 0.0, "세 방법 모두 놓침": 0.0} for bb in DEEP}
    n_pairs = 0
    for d in dogs:
        for breed, share in sc.orig[d].items():
            n_pairs += 1
            rs = [rank_tables[s][d][breed] for s in rank_tables]
            hit = sum(r <= 3 for r in rs)
            if hit == len(rs):
                cons["9개 모두 포착"] += share
            elif hit > 0:
                cons["일부 설정만 포착"] += share
            elif min(rs) <= 10:
                cons["아무 설정도 포착 못함 (최고 4~10위)"] += share
            else:
                cons["아무 설정도 포착 못함 (9개 모두 11위 이하)"] += share
            for bb in DEEP:
                r3 = [rank_tables[(k, bb)][d][breed] for k, *_ in METHODS if (k, bb) in rank_tables]
                key = "그 추출기의 세 방법 중 하나라도 포착" if min(r3) <= 3 else "세 방법 모두 놓침"
                by_bb[bb][key] += share
    n = len(dogs)
    rows.append(dict(analysis="2_consensus_9", config="딥 9개 설정", **{k: v / n for k, v in cons.items()},
                     note=f"개체 {n}마리, 정확 매핑 (개체, 품종) 쌍 {n_pairs}개. 값은 개체 평균 비율(원래 비율)"))
    for bb in DEEP:
        rows.append(dict(analysis="3_consensus_by_extractor", config=bb, **{k: v / n for k, v in by_bb[bb].items()}))

    # 풀이 방식 없는 특징 단계 (DINOv2)
    if args.features:
        ids, X, _ = load_features(str(args.features), "dinov2", "train")
        lab = pd.read_csv(args.features / "dinov2" / "train" / "index.csv").set_index("image_id")["label"]
        y = lab.reindex(ids).to_numpy()
        classes = list(first.columns)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        protos = np.stack([Xn[y == c].mean(0) for c in classes])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        eids, E, eidx = load_features(str(args.features), "dinov2", "eval_uf_panels")
        eidx = eidx.set_index("image_id").reindex(eids)
        keep = eidx["dog_id"].isin(dogs).to_numpy()          # dev 만 남긴다
        eids, E, eidx = eids[keep], E[keep], eidx[keep]
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        sims = pd.DataFrame(En @ protos.T, index=eids, columns=classes)
        panel = pd.Series([i.rsplit("_", 1)[-1] for i in eids], index=eids)
        views = {"얼굴 칸": sims[panel == "face"], "전신 칸": sims[panel == "body"]}
        mean_feat = pd.DataFrame(En, index=eids).groupby(eidx["dog_id"].to_numpy()).mean()
        mean_feat = mean_feat.div(np.linalg.norm(mean_feat, axis=1), axis=0)
        views["두 칸 평균"] = pd.DataFrame(mean_feat.to_numpy() @ protos.T, index=mean_feat.index, columns=classes)
        for vname, S in views.items():
            S = S.copy()
            S.index = [i if i in dogs else eidx.loc[i, "dog_id"] for i in S.index]
            # 품종 묶음 점수 = 묶음 안 클래스 유사도의 최댓값 (합하면 클래스 수가 많은 묶음이 유리해진다)
            B = S.T.groupby(lambda c: sc.c2b.get(c, c)).max().T
            per = pd.DataFrame({d: bucket_shares(ranks_in(B.loc[d]), sc.orig[d]) for d in dogs}).T
            rows.append(dict(analysis="4_feature_only_dinov2", config=vname, **base, **per.mean().to_dict(),
                             note="학습셋 평균점 코사인 순위, 풀이 방식 없음"))

    out = pd.DataFrame(rows)
    out.round(6).to_csv(ROOT / "results" / "dev_failure_stages.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.drop(columns=["note"], errors="ignore").round(3).fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
