"""[B] 딥 특징 추출: CLIP / DINOv2 / DINOv3 (+ dummy 점검용).

예:
  # 1,000장으로 시간 재기
  python -m src.features.extract_deep --backbone dinov3 --subset train --limit 1000 --tag timing
  # 학습셋 전체 (끊기면 같은 명령으로 다시 실행하면 이어서 뽑음)
  python -m src.features.extract_deep --backbone dinov3 --subset train
  python -m src.features.extract_deep --backbone dinov3 --subset val
  # 평가셋 (A의 eval_index.csv, D의 SAM 3 크롭)
  python -m src.features.extract_deep --backbone dinov3 --subset eval
  # D의 변형 이미지: 변형별 인덱스 csv를 받아 subset 이름을 바꿔 저장
  python -m src.features.extract_deep --backbone dinov3 --subset eval \
      --eval-index /path/eval_gray_index.csv --save-as eval_gray
"""
import argparse
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common import evalset, tsinghua
from src.common.config import load_yaml, log_run
from src.common.features_io import done_shards, save_shard, subset_dir
from src.features.backbones import build_backbone


class ImageDataset(Dataset):
    def __init__(self, loaders, preprocess):
        self.loaders = loaders          # 이미지를 여는 함수들의 목록 (index 순서)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.loaders)

    def __getitem__(self, i):
        return self.preprocess(self.loaders[i]())


def build_items(args, paths):
    """(image_ids, 이미지 로더 목록, index DataFrame)"""
    if args.subset in ("train", "val"):
        t = paths["tsinghua"]
        items = tsinghua.parse_split(t["train_list"] if args.subset == "train" else t["val_list"])
        items = tsinghua.filter_classes(items, paths["classes"]["exclude"], paths["classes"]["exclude_dingo"])
        if args.limit:
            items = items[: args.limit]
        # lambda 대신 partial: Windows는 DataLoader 작업자를 spawn으로 띄워 로더를 pickle해야 한다
        loaders = [partial(tsinghua.load_crop, it, t["image_root"], t["annotation_root"]) for it in items]
        index = pd.DataFrame({"image_id": [it.image_id for it in items],
                              "label": [it.class_name for it in items]})
    else:
        df = evalset.load_index(args.eval_index or paths["evalset"]["index_csv"])
        if args.limit:
            df = df.head(args.limit)
        loaders = [partial(evalset.load_image, p) for p in df["path"]]
        index = df[["image_id", "dog_id", "split"]].copy()
    return index["image_id"].tolist(), loaders, index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["clip", "dinov2", "dinov3", "dummy"])
    ap.add_argument("--subset", required=True, choices=["train", "val", "eval"])
    ap.add_argument("--eval-index", default=None, help="평가셋 인덱스 csv (기본은 paths.yaml)")
    ap.add_argument("--save-as", default=None, help="저장할 subset 이름 (예: eval_gray)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (시간 재기용)")
    ap.add_argument("--tag", default="", help="로그에 남길 메모")
    args = ap.parse_args()

    paths, bcfg = load_yaml("paths"), load_yaml("backbones")
    ex = bcfg["extract"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.backbone, bcfg, device)

    image_ids, loaders, index = build_items(args, paths)
    save_name = args.save_as or args.subset
    if args.limit:
        save_name += f"_limit{args.limit}"
    out_dir = subset_dir(paths["outputs"]["features_root"], args.backbone, save_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_dir / "index.csv", index=False)

    S = ex["shard_size"]
    n_shards = (len(image_ids) + S - 1) // S
    skip = done_shards(out_dir)
    t0, n_done = time.time(), 0
    status = "ok"
    try:
        for s in range(n_shards):
            if s in skip:
                continue
            lo, hi = s * S, min((s + 1) * S, len(image_ids))
            ds = ImageDataset(loaders[lo:hi], backbone.preprocess)
            dl = DataLoader(ds, batch_size=ex["batch_size"], num_workers=ex["num_workers"])
            cls_all, mean_all = [], []
            for pv in tqdm(dl, desc=f"{args.backbone}/{save_name} shard {s + 1}/{n_shards}"):
                c, m = backbone.encode(pv)
                cls_all.append(c)
                mean_all.append(m)
            save_shard(out_dir, s, image_ids[lo:hi], np.concatenate(cls_all), np.concatenate(mean_all))
            n_done += hi - lo
    except Exception as e:  # 실패한 실행도 로그에 남긴다
        status = f"failed: {e!r}"
        raise
    finally:
        sec = time.time() - t0
        rate = n_done / sec if sec > 0 else 0.0
        print(f"[{status}] {n_done}장, {sec:.1f}초, {rate:.1f}장/초 → {out_dir}")
        log_run(paths["outputs"]["logs_root"], {
            "who": "B", "run": f"FEAT-{args.backbone}-{save_name}", "status": status,
            "images": n_done, "seconds": round(sec, 1), "img_per_sec": round(rate, 1),
            "device": device, "batch_size": ex["batch_size"], "tag": args.tag,
        })


if __name__ == "__main__":
    main()
