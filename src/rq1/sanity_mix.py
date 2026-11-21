"""[B] 평균점 섞기 점검 실험: 특징 공간에서 만든 가짜 믹스의 비율을 FCLS가 되찾는가. 평가셋 없이 돌아간다.

질문: 순종 사진 두 장의 특징을 아는 비율로 섞었을 때, 품종 평균점 + FCLS가 그 두 품종과 비율을 되찾는가?
      (실제 믹스견은 특징의 선형 혼합이 아니므로 이 결과는 상한에 가까운 참고값이다. RQ1 결과가 아니다.)

절차
  1. 평균점: --proto-subset 특징의 품종별 평균 (기본 train).
  2. 섞을 재료: --mix-subset 특징 (기본 val).
     두 subset이 같으면 품종마다 사진을 무작위로 나눠 (1 - holdout)는 평균점에, holdout은 재료에 쓴다.
  3. 순종 점검: 재료 사진 한 장씩 FCLS → 가중치가 가장 큰 품종이 정답인 비율(top-1), 정답 품종 가중치.
  4. 가짜 믹스: 서로 다른 두 품종의 재료 사진 한 장씩을 r : (1 - r)로 섞어 FCLS →
     두 품종 가중치 합(포착 비율), 비율 오차 |w_a - r|, 가중치 상위 2개가 두 품종과 같은 비율.

예:
  # CPU 축소판: val 특징만으로 (품종당 절반은 평균점, 절반은 재료)
  python -m src.rq1.sanity_mix --backbone dinov2 --proto-subset val --mix-subset val
  # 학습셋 특징이 있으면
  python -m src.rq1.sanity_mix --backbone dinov3 --l2
"""
import argparse
import time

import numpy as np

from src.common.config import load_yaml, log_run
from src.common.features_io import load_features
from src.rq1.prototype_fcls import build_prototypes, fcls, l2norm


def split_holdout(labels: np.ndarray, classes: list[str], holdout: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """품종마다 holdout 비율을 재료로 떼어 낸다. (평균점용 인덱스, 재료용 인덱스)"""
    proto_idx, mix_idx = [], []
    for c in classes:
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_hold = min(len(idx) - 1, max(1, int(round(len(idx) * holdout))))
        mix_idx.extend(idx[:n_hold])
        proto_idx.extend(idx[n_hold:])
    return np.array(proto_idx), np.array(mix_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--proto-subset", default="train", help="평균점을 만들 특징 subset")
    ap.add_argument("--mix-subset", default="val", help="섞을 재료 특징 subset")
    ap.add_argument("--holdout", type=float, default=0.5, help="두 subset이 같을 때 재료로 떼는 비율")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--l2", action="store_true", help="사진 특징을 L2 정규화한 뒤 평균 · 혼합")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.625, 0.75, 0.875])
    ap.add_argument("--n-mix", type=int, default=500, help="비율마다 만드는 가짜 믹스 수")
    ap.add_argument("--n-pure", type=int, default=1000, help="순종 점검에 쓰는 사진 수 (0이면 전부)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    paths = load_yaml("paths")
    feat_root = paths["outputs"]["features_root"]
    rng = np.random.default_rng(args.seed)

    _, Xp, ip = load_features(feat_root, args.backbone, args.proto_subset, args.pool)
    lp = ip["label"].to_numpy()
    classes = sorted(set(lp))
    if args.proto_subset == args.mix_subset:
        proto_idx, mix_idx = split_holdout(lp, classes, args.holdout, rng)
        Xm, lm = Xp[mix_idx], lp[mix_idx]
        Xp, lp = Xp[proto_idx], lp[proto_idx]
    else:
        _, Xm, im = load_features(feat_root, args.backbone, args.mix_subset, args.pool)
        lm = im["label"].to_numpy()
        keep = np.isin(lm, classes)
        Xm, lm = Xm[keep], lm[keep]
    if args.l2:
        Xp, Xm = l2norm(Xp), l2norm(Xm)
    Xm = Xm.astype(np.float64)

    P = build_prototypes(Xp, lp, classes, frac=1.0, seed=args.seed)
    delta = 10.0 * float(np.abs(P).max())
    cls_index = {c: k for k, c in enumerate(classes)}
    ym = np.array([cls_index[c] for c in lm])
    print(f"[{args.backbone}/{args.pool}{'/l2' if args.l2 else ''}] 품종 {len(classes)}개, "
          f"평균점용 {len(Xp)}장({args.proto_subset}), 재료 {len(Xm)}장({args.mix_subset})")

    # 순종 점검
    pure = rng.permutation(len(Xm))[: args.n_pure or len(Xm)]
    W = np.array([fcls(P, Xm[i], delta) for i in pure])
    top1 = float((W.argmax(axis=1) == ym[pure]).mean())
    w_true = float(W[np.arange(len(pure)), ym[pure]].mean())
    print(f"  순종 {len(pure)}장: top-1 {top1:.3f}, 정답 품종 가중치 평균 {w_true:.3f}  (무작위 기준 {1 / len(classes):.3f})")

    # 가짜 믹스
    by_class = [np.flatnonzero(ym == k) for k in range(len(classes))]
    rows = []
    print(f"  {'비율 r':>8} {'포착 비율':>9} {'비율 오차':>9} {'top-2 일치':>10}")
    for r in args.ratios:
        captured, ratio_err, top2 = [], [], []
        for _ in range(args.n_mix):
            a, b = rng.choice(len(classes), size=2, replace=False)
            x = r * Xm[rng.choice(by_class[a])] + (1 - r) * Xm[rng.choice(by_class[b])]
            w = fcls(P, x, delta)
            captured.append(w[a] + w[b])
            ratio_err.append(abs(w[a] - r))
            top2.append(set(np.argsort(w)[-2:]) == {a, b})
        row = {"ratio": r, "captured": round(float(np.mean(captured)), 3),
               "ratio_abs_err": round(float(np.mean(ratio_err)), 3), "top2_match": round(float(np.mean(top2)), 3)}
        rows.append(row)
        print(f"  {r:>8.3f} {row['captured']:>9.3f} {row['ratio_abs_err']:>9.3f} {row['top2_match']:>10.3f}")

    log_run(paths["outputs"]["logs_root"], {
        "who": "B", "run": f"SANITY-{args.backbone}-mix-s{args.seed}" + ("-l2" if args.l2 else "")
                           + (f"-{args.pool}" if args.pool != "cls" else ""),
        "status": "ok", "proto_subset": args.proto_subset, "mix_subset": args.mix_subset,
        "n_classes": len(classes), "n_proto": len(Xp), "n_mix_pool": len(Xm), "l2": args.l2, "pool": args.pool,
        "pure_top1": round(top1, 3), "pure_w_true": round(w_true, 3), "mix": rows,
        "seconds": round(time.time() - t0, 1),
    })


if __name__ == "__main__":
    main()
