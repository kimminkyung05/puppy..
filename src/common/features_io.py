"""특징 파일 저장·읽기. B(딥)와 C(SIFT)가 같은 형식으로 저장한다.

구조:
  {features_root}/{backbone}/{subset}/shard_000.npz
      image_id: (N,) str
      cls:      (N, D) float16   # 딥: CLS 토큰(또는 이미지 임베딩), SIFT: BoVW 히스토그램
      mean:     (N, D) float16   # 딥: 패치 토큰 평균, SIFT: cls와 같은 값
  {features_root}/{backbone}/{subset}/index.csv    # image_id, label(학습셋) 또는 dog_id(평가셋)

subset 예: train, val, eval, eval_gray, eval_blur2 ...

체크포인트: 샤드는 임시 파일(shard_XXX.npz.tmp)에 다 쓴 뒤 이름을 바꿔 저장한다.
쓰다가 끊긴 파일은 .tmp로 남고 완료된 샤드로 세지 않으므로, 같은 명령을 다시 실행하면 그 샤드부터 이어서 뽑는다.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

TMP_SUFFIX = ".tmp"   # 쓰는 중인 파일의 접미사 (특징 샤드, 예측 CSV 공통)


def subset_dir(features_root: str, backbone: str, subset: str) -> Path:
    return Path(features_root) / backbone / subset


def tmp_path(final: Path) -> Path:
    """final을 쓰는 동안 쓸 임시 파일 경로. 예: shard_000.npz → shard_000.npz.tmp"""
    return final.with_name(final.name + TMP_SUFFIX)


def tmp_files(d: Path) -> list[Path]:
    """폴더에 남아 있는 임시 파일. 실행 중이 아닌데 있으면 쓰다가 끊긴 것이다."""
    return sorted(d.glob(f"*{TMP_SUFFIX}"))


def save_shard(out_dir: Path, shard_idx: int, image_ids, cls, mean) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"shard_{shard_idx:03d}.npz"
    tmp = tmp_path(final)
    with open(tmp, "wb") as f:
        np.savez(
            f,
            image_id=np.asarray(image_ids, dtype=str),
            cls=np.asarray(cls, dtype=np.float16),
            mean=np.asarray(mean, dtype=np.float16),
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)   # 다 쓴 뒤에만 완료된 샤드 이름을 얻는다


def done_shards(out_dir: Path) -> set[int]:
    """이미 저장된 샤드 번호. 세션이 끊긴 뒤 이어서 뽑을 때 쓴다. (.tmp는 세지 않는다)"""
    return {int(p.stem.split("_")[1]) for p in out_dir.glob("shard_*.npz")}


def load_features(features_root: str, backbone: str, subset: str, pool: str = "cls"):
    """(image_ids, X[N, D] float32, index DataFrame)를 돌려준다."""
    d = subset_dir(features_root, backbone, subset)
    shards = sorted(d.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"특징 파일 없음: {d}")
    ids, xs = [], []
    for p in shards:
        z = np.load(p)
        ids.append(z["image_id"])
        xs.append(z[pool].astype(np.float32))
    ids = np.concatenate(ids)
    X = np.concatenate(xs)
    index = pd.read_csv(d / "index.csv")
    index = index.set_index("image_id").loc[ids].reset_index()
    return ids, X, index
