"""C 의 SIFT 특징(.npy 한 덩어리)을 공용 특징 형식(features/sift/{subset}/shard_XXX.npz + index.csv)으로 옮긴다.

SIFT 추출은 C 담당이다. B 는 특징을 새로 뽑지 않고(코드북이 달라지면 A · C 와 다른 특징이 된다)
C 가 A 에게 준 것과 같은 파일을 그대로 형식만 바꾼다. C 의 파일은 고치지 않는다.

입력 (C 의 Drive C_work 에서 받은 것. 로컬에서는 C/ 폴더에 받아 두었다. git 에는 B/ 만 add 한다)
  sift_all_v2.npy      (N, 512) 학습셋 + validation BoVW. 행 순서 = index_local_v2.csv
  index_local_v2.csv   path, breed_name, split(train/validation), keep_127 ...  (image_id 는 path 에서 만든다)
  sift_eval_uf.npy     (200, 512) UF 패널(얼굴 · 전신) BoVW. 행 순서 = index_eval_uf.csv   (A 가 쓴 원본 패널)
  index_eval_uf.csv    image_id, dog_id, split ...

출력 (공용 형식. SIFT 는 cls 와 mean 에 같은 히스토그램을 넣는다)
  {features_root}/sift/train/          학습셋 (B 의 학습셋 index 와 같은 사진 · 같은 label 인지 확인)
  {features_root}/sift/val/            validation 전체 (prototype_fcls 가 val_split 으로 dev 절반을 고른다)
  {features_root}/sift/eval_uf_panels/ UF 패널 (B 의 eval_uf_panels index 와 같은 사진인지 확인)

사진 ID 맞추기: C 의 image_id 는 폴더가 빠져 있을 수 있어 마지막 경로 조각으로 B 의 ID 와 짝을 짓는다.
짝이 안 맞는 사진, 품종 이름이 다른 사진, 겹치는 ID 가 하나라도 있으면 멈춘다.

이미 만든 subset 이 있으면 건너뛴다(--overwrite 로 다시 만든다). 샤드는 임시 파일에 쓴 뒤 이름을 바꾼다.

사용 (B/ 안에서)
  python scripts/import_sift_from_c.py --c-dir ../C --features-root ../outputs/features \
      --b-train-index ../outputs/features_drive/clip/train/index.csv \
      --b-eval-index ../outputs/features_drive/clip/eval_uf_panels/index.csv \
      --val-split ../data/evalset/val_split_seed1.csv
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.features_io import save_shard  # noqa: E402

SHARD = 5000


def last(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]


def match(c_ids: pd.Series, b_ids: pd.Series, what: str) -> pd.Series:
    """C 의 ID → B 의 ID. 정확히 같으면 그대로, 아니면 마지막 경로 조각으로 짝짓는다."""
    if set(c_ids) <= set(b_ids):
        return c_ids
    b_short = last(b_ids)
    if not b_short.is_unique:
        raise SystemExit(f"{what}: B 의 사진 ID 마지막 조각이 겹쳐 짝을 지을 수 없음")
    m = dict(zip(b_short, b_ids))
    out = last(c_ids).map(m)
    if out.isna().any():
        raise SystemExit(f"{what}: B 에 없는 C 사진 {int(out.isna().sum())}장 (예: {c_ids[out.isna()].iloc[0]})")
    return out


def write(root: Path, subset: str, ids, X, index: pd.DataFrame, overwrite: bool):
    d = root / "sift" / subset
    if d.exists() and any(d.glob("shard_*.npz")):
        if not overwrite:
            print(f"[건너뜀] {d} 이미 있음 (--overwrite 로 다시 만든다)")
            return
        shutil.rmtree(d)
    for k, s in enumerate(range(0, len(ids), SHARD)):
        save_shard(d, k, ids[s:s + SHARD], X[s:s + SHARD], X[s:s + SHARD])
    tmp = d / "index.csv.tmp"
    index.to_csv(tmp, index=False)
    tmp.replace(d / "index.csv")
    print(f"저장 {d}: {len(ids)}장, {X.shape[1]}차원, 샤드 {k + 1}개")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-dir", type=Path, required=True)
    ap.add_argument("--features-root", type=Path, required=True)
    ap.add_argument("--b-train-index", type=Path, required=True)
    ap.add_argument("--b-eval-index", type=Path, required=True)
    ap.add_argument("--val-split", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # 학습셋 · validation
    X = np.load(args.c_dir / "sift_all_v2.npy", mmap_mode="r")
    idx = pd.read_csv(args.c_dir / "index_local_v2.csv")
    print(f"C 학습 특징 {X.shape}, index {len(idx)}행, 열 {list(idx.columns)}")
    if len(idx) != X.shape[0]:
        raise SystemExit("sift_all_v2.npy 행 수와 index_local_v2.csv 행 수가 다름")
    if "image_id" not in idx:
        # C 의 index_local_v2.csv 는 path 만 있다: .../202-n000028-curly_coated_retriever/n111302.jpg → n000028/n111302
        p = idx["path"].astype(str).str.replace("\\", "/", regex=False).str.split("/")
        idx["image_id"] = p.str[-2].str.split("-").str[1] + "/" + p.str[-1].str.rsplit(".", n=1).str[0]
    keep = idx["keep_127"].astype(bool) if "keep_127" in idx else pd.Series(True, index=idx.index)

    b_tr = pd.read_csv(args.b_train_index)
    sel = keep & (idx["split"] == "train")
    c = idx[sel]
    ids = match(c["image_id"], b_tr["image_id"], "train")
    lab = dict(zip(b_tr["image_id"], b_tr["label"]))
    bad = (ids.map(lab) != c["breed_name"].values).sum()
    if bad:
        raise SystemExit(f"train: 품종 이름이 B 와 다른 사진 {bad}장")
    if len(ids) != len(b_tr) or not ids.is_unique:
        raise SystemExit(f"train: C {len(ids)}장 / B {len(b_tr)}장 (같아야 함) 또는 ID 중복")
    write(args.features_root, "train", ids.to_numpy(), np.asarray(X[sel.to_numpy()], np.float32),
          pd.DataFrame({"image_id": ids.to_numpy(), "label": c["breed_name"].to_numpy()}), args.overwrite)

    vs = pd.read_csv(args.val_split)
    sel = keep & (idx["split"].isin(["validation", "val"]))
    c = idx[sel]
    ids = match(c["image_id"], vs["image_id"], "val")
    lab = dict(zip(vs["image_id"], vs["label"]))
    bad = (ids.map(lab) != c["breed_name"].values).sum()
    if bad:
        raise SystemExit(f"val: 품종 이름이 B 와 다른 사진 {bad}장")
    if len(ids) != len(vs) or not ids.is_unique:
        raise SystemExit(f"val: C {len(ids)}장 / B val_split {len(vs)}장 (같아야 함) 또는 ID 중복")
    write(args.features_root, "val", ids.to_numpy(), np.asarray(X[sel.to_numpy()], np.float32),
          pd.DataFrame({"image_id": ids.to_numpy(), "label": c["breed_name"].to_numpy()}), args.overwrite)

    # UF 패널
    Xe = np.load(args.c_dir / "sift_eval_uf.npy")
    ie = pd.read_csv(args.c_dir / "index_eval_uf.csv")
    print(f"C UF 특징 {Xe.shape}, index {len(ie)}행, 열 {list(ie.columns)}")
    if len(ie) != Xe.shape[0]:
        raise SystemExit("sift_eval_uf.npy 행 수와 index_eval_uf.csv 행 수가 다름")
    b_ev = pd.read_csv(args.b_eval_index)
    ids = match(ie["image_id"], b_ev["image_id"], "eval_uf_panels")
    if set(ids) != set(b_ev["image_id"]) or not ids.is_unique:
        raise SystemExit(f"eval_uf_panels: C 와 B 의 사진 목록이 다름 (C {len(ids)}장, B {len(b_ev)}장)")
    meta = b_ev.set_index("image_id").loc[ids].reset_index()      # dog_id, split 은 B 의 평가셋 index 를 따른다
    write(args.features_root, "eval_uf_panels", ids.to_numpy(), Xe.astype(np.float32), meta, args.overwrite)


if __name__ == "__main__":
    main()
