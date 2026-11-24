"""성능 개선 실험용 특징 추출 (로컬 CPU, 백그라운드). 공용 특징 폴더(features/)와 섞지 않고 따로 저장한다.

딥 학습셋 전체 특징은 Drive 에만 있어서, 로컬에서는 학습셋 일부를 직접 뽑는다.
여기서 효과가 보인 설정만 나중에 Colab 에서 학습셋 전체로 다시 확인한다.

부분 (추출기마다)
  train_sub   학습셋에서 품종마다 --per-class 장 (시드 0, 공식 추출과 같은 몸통 박스 크롭)  → 평균점 · 분류기 재료
  val_sub     val 의 dev 절반(val_split_seed1.csv)에서 품종마다 --val-per-class 장      → UF 를 쓰지 않는 대리 평가
  UF 사진은 뽑지 않는다. 평가에는 공식 UF 특징(features_drive/{추출기}/eval_uf_panels)을 쓴다 (시험 시 증강은 하지 않기로 함).

저장: {out}/{backbone}/{part}.npz  (image_id, view, cls, mean; float32). 임시 파일에 다 쓴 뒤 이름을 바꾼다.
      작업 중에는 640장마다 {part}_chunks/chunk_XXX.npz 로 중간 저장하고, 다 합친 뒤 조각을 지운다.
      이미 있는 부분 · 조각은 건너뛴다 (끊기면 같은 명령을 다시 실행하면 마지막 조각부터 잇는다).

사용 (B/ 안에서)
  $env:DOGMIX_PATHS = "..\\data\\paths_local.yaml"
  python scripts/improve_extract.py --backbones dinov2 clip dinov3 --out ../outputs/improve
"""
import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import tsinghua  # noqa: E402
from src.common.config import load_yaml  # noqa: E402
from src.features.backbones import build_backbone  # noqa: E402

def pick_train(paths, per_class, seed=0):
    t = paths["tsinghua"]
    items = tsinghua.filter_classes(tsinghua.parse_split(t["train_list"]), paths["classes"]["exclude"],
                                    paths["classes"]["exclude_dingo"])
    df = pd.DataFrame({"i": range(len(items)), "label": [it.class_name for it in items]})
    sel = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(per_class, len(g)), random_state=seed))
    return [items[i] for i in sorted(sel["i"])]


def pick_val(paths, per_class, val_split, seed=0):
    t = paths["tsinghua"]
    items = tsinghua.filter_classes(tsinghua.parse_split(t["val_list"]), paths["classes"]["exclude"],
                                    paths["classes"]["exclude_dingo"])
    dev = set(pd.read_csv(val_split).query("split == 'dev'")["image_id"])
    items = [it for it in items if it.image_id in dev]
    df = pd.DataFrame({"i": range(len(items)), "label": [it.class_name for it in items]})
    sel = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(per_class, len(g)), random_state=seed))
    return [items[i] for i in sorted(sel["i"])]


CHUNK = 640   # 이 장수마다 중간 저장한다 (끊기면 다시 실행할 때 저장된 조각은 건너뛴다)


def run(bb, loaders, ids, chunk_dir: Path, batch=32):
    """조각(CHUNK 장)마다 {chunk_dir}/chunk_XXX.npz 로 저장하고, 다 끝나면 이어 붙여 돌려준다."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(loaders) + CHUNK - 1) // CHUNK
    t0, n_new = time.time(), 0
    for k in range(n_chunks):
        f = chunk_dir / f"chunk_{k:03d}.npz"
        lo, hi = k * CHUNK, min((k + 1) * CHUNK, len(loaders))
        if f.exists():
            with np.load(f) as z:
                if list(z["image_id"]) == list(ids[lo:hi]):
                    print(f"    [건너뜀] 조각 {k + 1}/{n_chunks}", flush=True)
                    continue
            raise SystemExit(f"{f} 의 사진 목록이 지금과 다름. 폴더를 확인한 뒤 지우고 다시 실행")
        cls_l, mean_l = [], []
        for s in range(lo, hi, batch):
            imgs = [loaders[i]() for i in range(s, min(s + batch, hi))]
            c, m = bb.encode(torch.stack([bb.preprocess(im) for im in imgs]))
            cls_l.append(c); mean_l.append(m)
        save(f, image_id=np.array(ids[lo:hi]), cls=np.concatenate(cls_l), mean=np.concatenate(mean_l))
        n_new += hi - lo
        print(f"    조각 {k + 1}/{n_chunks} 저장 ({hi}/{len(loaders)}장, {n_new / (time.time() - t0):.1f}장/초)",
              flush=True)
    parts = [np.load(chunk_dir / f"chunk_{k:03d}.npz") for k in range(n_chunks)]
    return np.concatenate([p["cls"] for p in parts]), np.concatenate([p["mean"] for p in parts])


def save(path: Path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["dinov2", "clip", "dinov3"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--val-per-class", type=int, default=8)
    ap.add_argument("--val-split", default="../data/evalset/val_split_seed1.csv")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    paths, models = load_yaml("paths"), load_yaml("backbones")
    t = paths["tsinghua"]
    tr, va = pick_train(paths, args.per_class), pick_val(paths, args.val_per_class, args.val_split)
    print(f"train_sub {len(tr)}장, val_sub {len(va)}장", flush=True)

    for name in args.backbones:
        bb = None
        jobs = [
            ("train_sub", [partial(tsinghua.load_crop, it, t["image_root"], t["annotation_root"]) for it in tr],
             [it.image_id for it in tr], ["orig"] * len(tr), {"label": [it.class_name for it in tr]}),
            ("val_sub", [partial(tsinghua.load_crop, it, t["image_root"], t["annotation_root"]) for it in va],
             [it.image_id for it in va], ["orig"] * len(va), {"label": [it.class_name for it in va]}),
        ]
        for part, loaders, ids, views, meta in jobs:
            out = args.out / name / f"{part}.npz"
            if out.exists():
                print(f"[건너뜀] {out}", flush=True)
                continue
            if bb is None:
                bb = build_backbone(name, models, "cpu")
            print(f"[{name}] {part}: {len(ids)}장", flush=True)
            chunk_dir = args.out / name / f"{part}_chunks"
            c, m = run(bb, loaders, ids, chunk_dir)
            save(out, image_id=np.array(ids), view=np.array(views), cls=c, mean=m,
                 **{k: np.array(v) for k, v in meta.items()})
            for f in chunk_dir.glob("chunk_*.npz"):      # 합친 파일을 저장한 뒤에만 조각을 지운다
                f.unlink()
            chunk_dir.rmdir()
            print(f"[{name}] {part} 저장 → {out}", flush=True)


if __name__ == "__main__":
    main()
