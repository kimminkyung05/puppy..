"""성능 개선 실험의 입력을 공용 특징 샤드에서 만든다 (추출 없음). Colab 에서 Drive 특징으로 학습셋 전체를 쓸 때 쓴다.

로컬에서는 학습 특징이 없어 improve_extract.py 로 일부만 뽑았다. Colab 에는 Drive 에 학습셋 전체 특징이 있으므로
그것을 그대로 improve_eval.py 가 읽는 형식으로 옮긴다.

입력  {features_root}/{backbone}/train/  (shard_*.npz: image_id, cls, mean + index.csv: image_id, label)
      {features_root}/{backbone}/val/
      --val-split  val_split_seed1.csv (image_id, label, split). dev 행만 쓴다 (test 절반은 안 씀)
출력  {out}/{backbone}/train_sub.npz, val_sub.npz  (image_id, view, cls, mean, label; float32)
      --per-class 0 이면 학습셋 전체, N 이면 품종마다 N장(시드 0). --val-per-class 도 같다

사용 (B/ 안에서)
  python scripts/improve_from_shards.py --backbone dinov3 --features-root {DRIVE}/features \
      --val-split {DRIVE}/evalset/val_split_seed1.csv --out /content/improve_full
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.features_io import load_features  # noqa: E402


def pick(index: pd.DataFrame, per_class: int, seed=0) -> np.ndarray:
    if per_class <= 0:
        return np.arange(len(index))
    sel = index.reset_index().groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(per_class, len(g)), random_state=seed))
    return np.sort(sel["index"].to_numpy())


def write(path: Path, ids, cls, mean, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, image_id=np.asarray(ids, dtype=str), view=np.array(["orig"] * len(ids)),
                 cls=cls.astype(np.float32), mean=mean.astype(np.float32), label=np.asarray(label, dtype=str))
    os.replace(tmp, path)
    print(f"저장 {path}: {len(ids)}장, cls {cls.shape[1]}차원, mean {mean.shape[1]}차원", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=0, help="0 = 학습셋 전체")
    ap.add_argument("--val-per-class", type=int, default=0, help="0 = val dev 절반 전체")
    args = ap.parse_args()

    for part in ("train_sub", "val_sub"):
        if (args.out / args.backbone / f"{part}.npz").exists():
            print(f"[건너뜀] {args.out / args.backbone / part}.npz 이미 있음")
    out_tr, out_va = args.out / args.backbone / "train_sub.npz", args.out / args.backbone / "val_sub.npz"

    if not out_tr.exists():
        ids, C, idx = load_features(args.features_root, args.backbone, "train", "cls")
        _, M, _ = load_features(args.features_root, args.backbone, "train", "mean")
        k = pick(idx, args.per_class)
        write(out_tr, ids[k], C[k], M[k], idx["label"].to_numpy()[k])

    if not out_va.exists():
        ids, C, idx = load_features(args.features_root, args.backbone, "val", "cls")
        _, M, _ = load_features(args.features_root, args.backbone, "val", "mean")
        dev = set(pd.read_csv(args.val_split).query("split == 'dev'")["image_id"])
        keep = np.isin(ids, list(dev))
        if keep.sum() == 0:
            raise SystemExit("val 특징에서 val_split 의 dev 사진을 하나도 찾지 못함 (image_id 형식 확인)")
        ids, C, M, idx = ids[keep], C[keep], M[keep], idx[keep].reset_index(drop=True)
        k = pick(idx, args.val_per_class)
        write(out_va, ids[k], C[k], M[k], idx["label"].to_numpy()[k])


if __name__ == "__main__":
    main()
