"""[B] 데모용 평균점 파일을 만든다 (모델 없이, 이미 뽑은 특징만 쓴다).

데모 설정 (dev 20마리로 고름, docs 참고): dinov2 · CLS · L2 없음 · 부분 표본 80% · 반복 r1~r3.
prototype_fcls.py 와 같은 함수(build_prototypes, fcls)와 같은 시드(seed=r)를 쓴다.

출력 (B/ 밖. git에 올리지 않는다)
  {out}  예: ../outputs/demo/demo_dinov2.npz
    classes [C]          학습셋 label 정렬 순서
    P       [R, C, D]    반복별 평균점
    delta   [R]          FCLS 합=1 가중치 (prototype_fcls.py 와 같은 식)
    repeats [R]
    dev_recon_q, dev_recon_levels   dev 사진의 재구성 오차 분위수 (판단 보류 기준값 후보)
    meta    json 문자열 (설정 · 사진 수 · 점검용 여부)

판단 보류 기준값은 평가셋 split == dev 행만으로 정한다. hold 행은 읽은 뒤 바로 버리고 숫자에 쓰지 않는다.

예 (B/ 안에서)
  python scripts/build_demo_assets.py --features-root ../outputs/features_drive --out ../outputs/demo/demo_dinov2.npz
형식 점검 (로컬에 학습셋 샤드가 없을 때. 결과는 데모 화면에 '점검용'으로 표시된다)
  python scripts/build_demo_assets.py --features-root ../outputs/features --train-subset train_parents_n100 \
      --eval-subset eval_uf_parents --out ../outputs/demo/CHECK_demo_dinov2.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.features_io import load_features, tmp_path  # noqa: E402
from src.rq1.prototype_fcls import build_prototypes, fcls  # noqa: E402

LEVELS = [0.5, 0.75, 0.9, 0.95, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--train-subset", default="train")
    ap.add_argument("--eval-subset", default="eval_uf_panels")
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, Xtr, itr = load_features(args.features_root, args.backbone, args.train_subset, args.pool)
    labels = itr["label"].to_numpy()
    classes = sorted(set(labels))
    Ps, deltas = [], []
    for r in args.repeats:
        P = build_prototypes(Xtr, labels, classes, args.frac, seed=r)
        Ps.append(P)
        deltas.append(10.0 * float(np.abs(P).max()))
    P = np.stack(Ps)

    _, Xev, iev = load_features(args.features_root, args.backbone, args.eval_subset, args.pool)
    keep = (iev["split"].astype(str) == "dev").to_numpy()
    Xev = Xev[keep].astype(np.float64)          # hold 행은 여기서 버린다
    if len(Xev) == 0:
        raise ValueError(f"{args.eval_subset} 에 split=dev 행이 없음")
    err = np.zeros(len(Xev))
    for Pr, d in zip(P, deltas):
        for i, x in enumerate(Xev):
            w = fcls(Pr, x, d)
            err[i] += np.linalg.norm(Pr.T @ w - x) / max(np.linalg.norm(x), 1e-12)
    err /= len(P)
    q = np.quantile(err, LEVELS)

    meta = {"backbone": args.backbone, "pool": args.pool, "l2": False, "frac": args.frac,
            "train_subset": args.train_subset, "n_train_images": int(len(Xtr)), "n_classes": len(classes),
            "eval_subset": args.eval_subset, "n_dev_images": int(len(Xev)),
            "n_dev_dogs": int(iev.loc[keep, "dog_id"].nunique()),
            "check_only": args.train_subset != "train"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(out)
    with open(tmp, "wb") as f:   # np.savez 가 이름 끝에 .npz 를 붙이지 않게 파일 객체로 쓴다
        np.savez(f, classes=np.array(classes), P=P.astype(np.float32), delta=np.array(deltas),
                 repeats=np.array(args.repeats), dev_recon_q=q, dev_recon_levels=np.array(LEVELS),
                 meta=json.dumps(meta, ensure_ascii=False))
    os.replace(tmp, out)

    print(json.dumps(meta, ensure_ascii=False))
    print("dev 재구성 오차 분위수:", {f"q{int(l * 100)}": round(float(v), 4) for l, v in zip(LEVELS, q)})
    print("저장:", out)


if __name__ == "__main__":
    main()
