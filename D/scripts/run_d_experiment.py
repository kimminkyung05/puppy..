from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent / "d_input"
OUT = ROOT / "D_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3]
CONDITIONS = ["original", "area_head", "area_body", "area_background", "blur_01", "blur_03", "blur_06", "blur_12", "gray",
              "control_area_head", "control_area_body", "control_area_background", "control_blur_01", "control_blur_03", "control_blur_06", "control_blur_12", "control_gray"]
PAIR = {"area_head": "control_area_head", "area_body": "control_area_body", "area_background": "control_area_background",
        "blur_01": "control_blur_01", "blur_03": "control_blur_03", "blur_06": "control_blur_06", "blur_12": "control_blur_12", "gray": "control_gray"}
BLUR_RATIOS = {"blur_01": 0.01, "blur_03": 0.03, "blur_06": 0.06, "blur_12": 0.12}

meta = pd.read_csv(ROOT / "mixdog_metadata.csv")
dna = pd.read_csv(ROOT / "dna_long.csv")
dna["fraction"] = dna["fraction"].astype(float)
dna["fraction"] = dna["fraction"] / dna.groupby("dog_id")["fraction"].transform("sum")
breed_map = pd.read_csv(ROOT / "breed_map.csv")
map_dict = dict(zip(breed_map["tsinghua_class"].astype(str), breed_map["dna_breed"].astype(str)))


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def masks_for(arr):
    h, w = arr.shape[:2]
    body = np.zeros((h, w), np.uint8)
    rect = (max(1, int(0.05 * w)), max(1, int(0.04 * h)), max(2, int(0.90 * w)), max(2, int(0.92 * h)))
    try:
        bg = np.zeros((h, w), np.uint8)
        cv2.grabCut(arr, bg, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 2, cv2.GC_INIT_WITH_RECT)
        body = np.isin(bg, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    except Exception:
        body[int(0.08*h):int(0.95*h), int(0.08*w):int(0.92*w)] = 1
    if body.sum() < 0.03 * h * w:
        body[int(0.08*h):int(0.95*h), int(0.08*w):int(0.92*w)] = 1
    ys = np.where(body > 0)[0]
    y0, y1 = (int(ys.min()), int(ys.max())) if len(ys) else (0, h - 1)
    head = np.zeros_like(body)
    head[y0:min(y1 + 1, y0 + max(1, int(0.48 * (y1 - y0 + 1))))] = body[y0:min(y1 + 1, y0 + max(1, int(0.48 * (y1 - y0 + 1))))]
    return body.astype(bool), head.astype(bool)


def fill_outside(arr, keep):
    fill = np.median(arr.reshape(-1, 3), axis=0).round().astype(np.uint8)
    out = np.broadcast_to(fill, arr.shape).copy()
    out[keep] = arr[keep]
    return out


def same_area_mask(shape, area, seed):
    rng = np.random.default_rng(seed)
    n = min(max(int(area), 1), shape[0] * shape[1])
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    out[rng.choice(shape[0] * shape[1], size=n, replace=False)] = True
    return out.reshape(shape)


def matched_noise(original, target, seed):
    a = original.astype(np.float32)
    b = target.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, a.shape).astype(np.float32)
    noise *= math.sqrt(max(mse, 1e-8) / float(np.mean(noise ** 2)))
    return np.clip(a + noise, 0, 255).astype(np.uint8)


def condition_image(path, condition, seed):
    arr = np.asarray(Image.open(path).convert("RGB"))
    body, head = masks_for(arr)
    if condition == "original":
        return arr, int(body.sum()), int(head.sum())
    if condition == "area_head":
        out = fill_outside(arr, head)
    elif condition == "area_body":
        out = fill_outside(arr, body & ~head)
    elif condition == "area_background":
        out = fill_outside(arr, ~body)
    elif condition.startswith("blur_"):
        ratio = BLUR_RATIOS[condition]
        sigma = max(0.5, ratio * min(arr.shape[:2]))
        k = 2 * max(1, int(math.ceil(3 * sigma))) + 1
        out = cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma)
    elif condition == "gray":
        out = np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))
    elif condition.startswith("control_area_"):
        base = {"control_area_head": head, "control_area_body": body & ~head, "control_area_background": ~body}[condition]
        out = matched_noise(arr, fill_outside(arr, base), stable_seed(seed, condition))
    elif condition.startswith("control_blur_"):
        base_name = condition.replace("control_", "")
        ratio = BLUR_RATIOS[base_name]
        sigma = max(0.5, ratio * min(arr.shape[:2]))
        k = 2 * max(1, int(math.ceil(3 * sigma))) + 1
        target = cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma)
        out = matched_noise(arr, target, stable_seed(seed, condition))
    elif condition == "control_gray":
        out = matched_noise(arr, np.asarray(Image.fromarray(arr).convert("L").convert("RGB")), stable_seed(seed, condition))
    else:
        raise ValueError(condition)
    return out, int(body.sum()), int(head.sum())


class ClipExtractor:
    def __init__(self):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from transformers import CLIPImageProcessor, CLIPModel
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()
        self.vision = model.vision_model
        self.proj = model.visual_projection

    @torch.inference_mode()
    def encode(self, images):
        x = self.processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)
        return self.proj(self.vision(pixel_values=x).pooler_output).float().cpu().numpy()


class DinoV2Extractor:
    def __init__(self):
        from torchvision import transforms
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE).eval()
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack([self.tf(Image.fromarray(im)) for im in images]).to(DEVICE)
        return self.model(x).float().cpu().numpy()


def load_npz(name, seed):
    path = ROOT / "A_checkpoints" / f"RQ1-{name}-clf-s{seed}_linear_weights.npz"
    z = np.load(path, allow_pickle=False)
    return {"weight": z["weight"].astype(np.float32), "bias": z["bias"].astype(np.float32), "classes": [str(x) for x in z["classes"]], "path": str(path)}


def predict(features, clf):
    x = features.astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    logits = x @ clf["weight"].T + clf["bias"]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def aggregate(probs, classes):
    names = [map_dict.get(c, "__DROP__") for c in classes]
    keep = sorted(set(names) - {"__DROP__"})
    out = np.zeros((len(probs), len(keep)), dtype=np.float32)
    for j, name in enumerate(names):
        if name in keep:
            out[:, keep.index(name)] += probs[:, j]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)
    return pd.DataFrame(out, columns=keep)


def truth_for(dog_ids):
    t = dna[dna.dog_id.isin(dog_ids)].pivot_table(index="dog_id", columns="breed", values="fraction", aggfunc="sum", fill_value=0)
    return t.reindex(dog_ids, fill_value=0).div(t.sum(axis=1).replace(0, 1), axis=0)


def score(pred):
    cols = [c for c in pred.columns if c.startswith("pred_")]
    dog_pred = pred.groupby("dog_id")[cols].mean()
    dog_pred.columns = [c.replace("pred_", "") for c in dog_pred.columns]
    gt = truth_for(dog_pred.index.tolist())
    all_names = sorted(set(gt.columns) | set(dog_pred.columns))
    p = dog_pred.reindex(columns=all_names, fill_value=0).to_numpy()
    y = gt.reindex(columns=all_names, fill_value=0).to_numpy()
    p /= np.clip(p.sum(axis=1, keepdims=True), 1e-8, None)
    rows = []
    for i, dog_id in enumerate(dog_pred.index):
        top3 = np.argsort(-p[i])[:3]
        rows.append({"dog_id": dog_id, "capture_top3": float(y[i, top3].sum()), "overlap": float(np.minimum(p[i], y[i]).sum()), "kl": float(np.sum(y[i] * np.log((y[i] + 1e-8) / (p[i] + 1e-8))))})
    return pd.DataFrame(rows)


def bootstrap(x, seed=20261120, n=1000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    print(json.dumps({"device": DEVICE, "n_images": len(meta), "n_dogs": meta.dog_id.nunique(), "conditions": len(CONDITIONS)}, ensure_ascii=False))
    extractors = {}
    skipped = []
    for name, cls in [("clip", ClipExtractor), ("dinov2", DinoV2Extractor)]:
        try:
            extractors[name] = cls()
        except Exception as e:
            skipped.append({"extractor": name, "reason": repr(e)})
    for name in ["dinov3", "sift"]:
        skipped.append({"extractor": name, "reason": "A handoff has classifier weights but no reproducible extractor asset in the bundle"})
    pd.DataFrame(skipped).to_csv(OUT / "skipped_extractors.csv", index=False)

    mask_rows = []
    for r in meta.itertuples(index=False):
        arr = np.asarray(Image.open(ROOT / r.relative_path).convert("RGB"))
        body, head = masks_for(arr)
        mask_rows.append({"image_id": r.image_id, "body_area": int(body.sum()), "head_area": int(head.sum()), "body_fraction": float(body.mean()), "head_fraction": float(head.mean()), "mask_method": "grabcut_heuristic_fallback"})
    pd.DataFrame(mask_rows).to_csv(OUT / "mask_stats_dev.csv", index=False)

    all_detail = []
    for name, extractor in extractors.items():
        for condition in CONDITIONS:
            cache = OUT / "feature_cache" / name
            cache.mkdir(parents=True, exist_ok=True)
            feat_path = cache / f"{condition}.npy"
            if feat_path.exists():
                features = np.load(feat_path)
            else:
                imgs = [condition_image(ROOT / r.relative_path, condition, 20261120)[0] for r in meta.itertuples(index=False)]
                features = extractor.encode(imgs).astype(np.float32)
                np.save(feat_path, features)
            for seed in SEEDS:
                clf = load_npz(name, seed)
                probs = predict(features, clf)
                dna_pred = aggregate(probs, clf["classes"])
                pred = pd.concat([meta[["image_id", "dog_id"]].reset_index(drop=True), dna_pred.add_prefix("pred_")], axis=1)
                pred.to_csv(OUT / f"pred_dev_{name}_{condition}_seed{seed}.csv", index=False)
                scored = score(pred)
                scored["extractor"] = name
                scored["seed"] = seed
                scored["condition"] = condition
                all_detail.append(scored)
            print(f"done {name}/{condition}")
    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(OUT / "detail_scores_dev.csv", index=False)

    summary_rows = []
    for (name, seed, condition), g in detail.groupby(["extractor", "seed", "condition"]):
        for metric in ["capture_top3", "overlap", "kl"]:
            mean, lo, hi = bootstrap(g[metric].to_numpy())
            summary_rows.append({"extractor": name, "seed": seed, "condition": condition, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n_dogs": len(g)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "condition_summary_dev.csv", index=False)

    base = detail[detail.condition.eq("original")][["extractor", "seed", "dog_id", "capture_top3", "overlap", "kl"]].rename(columns={"capture_top3": "base_capture", "overlap": "base_overlap", "kl": "base_kl"})
    extra_rows = []
    for variant, control in PAIR.items():
        v = detail[detail.condition.eq(variant)][["extractor", "seed", "dog_id", "capture_top3", "overlap", "kl"]].rename(columns={"capture_top3": "variant_capture", "overlap": "variant_overlap", "kl": "variant_kl"})
        c = detail[detail.condition.eq(control)][["extractor", "seed", "dog_id", "capture_top3", "overlap", "kl"]].rename(columns={"capture_top3": "control_capture", "overlap": "control_overlap", "kl": "control_kl"})
        m = base.merge(v, on=["extractor", "seed", "dog_id"]).merge(c, on=["extractor", "seed", "dog_id"])
        m["condition"] = variant
        m["extra_drop_capture_top3"] = m.control_capture - m.variant_capture
        m["extra_drop_overlap"] = m.control_overlap - m.variant_overlap
        m["extra_drop_kl"] = m.variant_kl - m.control_kl
        extra_rows.append(m)
    extra = pd.concat(extra_rows, ignore_index=True)
    extra.to_csv(OUT / "extra_drop_detail_dev.csv", index=False)
    extra_summary = extra.groupby(["extractor", "seed", "condition"])[["extra_drop_capture_top3", "extra_drop_overlap", "extra_drop_kl"]].mean().reset_index()
    extra_summary.to_csv(OUT / "extra_drop_summary_dev.csv", index=False)

    run_info = {"split": "dev", "n_images": int(len(meta)), "n_dogs": int(meta.dog_id.nunique()), "extractors_run": sorted(extractors), "extractors_skipped": skipped, "conditions": CONDITIONS, "seeds": SEEDS, "device": DEVICE, "mask_method": "GrabCut heuristic fallback; SAM3 checkpoint unavailable", "dna_map_rows": int(len(breed_map)), "dna_rows": int(len(dna))}
    (OUT / "run_info_dev.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", len(detail), "dog-condition-extractor-seed rows")


if __name__ == "__main__":
    main()
