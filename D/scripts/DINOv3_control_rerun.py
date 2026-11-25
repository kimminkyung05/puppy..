"""
DINOv3 control-only rerun for the D RQ2 experiment.

Run this as the last code cell of D_RQ2_Colab.ipynb, after the notebook has
loaded `meta`, `dna`, `groups`, `MASK_ROOT`, `OUTPUT_ROOT`, and the helper
functions/classes from cells 3, 6, 12, 15, 16, and 17.

This deliberately runs only the agreed scope:
  original, gray, blur_01/03/06/12, panel_face, panel_body
and their matched controls. It does not create background-only or optional
selection conditions.

The script writes a new subdirectory and does not overwrite the earlier GPU
outputs. The final extra-drop table is valid only after this cell completes.
"""

from pathlib import Path
import hashlib
import json
import math
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Scope and output paths
# ---------------------------------------------------------------------------

RERUN_CONDITIONS = [
    "original",
    "gray",
    "blur_01",
    "blur_03",
    "blur_06",
    "blur_12",
    "panel_face",
    "panel_body",
    "control_gray",
    "control_blur_01",
    "control_blur_03",
    "control_blur_06",
    "control_blur_12",
    "control_panel_face",
    "control_panel_body",
]

RERUN_PAIR = {
    "gray": "control_gray",
    "blur_01": "control_blur_01",
    "blur_03": "control_blur_03",
    "blur_06": "control_blur_06",
    "blur_12": "control_blur_12",
    "panel_face": "control_panel_face",
    "panel_body": "control_panel_body",
}

BLUR_RATIOS_RERUN = {
    "blur_01": 0.01,
    "blur_03": 0.03,
    "blur_06": 0.06,
    "blur_12": 0.12,
}

RERUN_SEED = 20261125
RERUN_ROOT = OUTPUT_ROOT / "dinov3_control_rerun"  # noqa: F821 - notebook global
IMAGE_ROOT_RERUN = RERUN_ROOT / "images"
FEATURE_ROOT_RERUN = RERUN_ROOT / "features"
RERUN_ROOT.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT_RERUN.mkdir(parents=True, exist_ok=True)
FEATURE_ROOT_RERUN.mkdir(parents=True, exist_ok=True)


def stable_seed_rerun(*parts):
    raw = "|".join(map(str, parts)).encode("utf-8")
    return RERUN_SEED + int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1000000


def load_mask_rerun(image_id):
    z = np.load(MASK_ROOT / f"{image_id}.npz")  # noqa: F821 - notebook global
    return z["body"].astype(bool), z["head"].astype(bool)


def resize_mask_rerun(mask, shape):
    h, w = shape
    if mask.shape == (h, w):
        return mask.astype(bool)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (w, h), Image.Resampling.NEAREST
        )
    ) > 0


def crop_to_body_rerun(img, body):
    ys, xs = np.where(body)
    if len(xs) == 0:
        raise ValueError("empty body mask")
    h, w = body.shape
    pad_x = int(0.10 * (xs.max() - xs.min() + 1))
    pad_y = int(0.10 * (ys.max() - ys.min() + 1))
    box = (
        max(0, xs.min() - pad_x),
        max(0, ys.min() - pad_y),
        min(w, xs.max() + 1 + pad_x),
        min(h, ys.max() + 1 + pad_y),
    )
    return img.crop(box)


def fill_outside_rerun(arr, keep):
    fill = np.median(arr.reshape(-1, 3), axis=0).round().astype(np.uint8)
    out = np.broadcast_to(fill, arr.shape).copy()
    out[keep] = arr[keep]
    return out


def random_same_area_mask_rerun(shape, area, seed):
    total = shape[0] * shape[1]
    n = min(max(int(area), 1), total)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=n, replace=False)
    out = np.zeros(total, dtype=bool)
    out[chosen] = True
    return out.reshape(shape)


def gaussian_blur_rerun(arr, ratio):
    sigma = max(0.5, ratio * min(arr.shape[:2]))
    radius = max(1, int(math.ceil(3 * sigma)))
    kernel = 2 * radius + 1
    return cv2.GaussianBlur(arr, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def psnr_db_rerun(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def psnr_matched_noise_rerun(original, target, seed):
    a = np.asarray(original).astype(np.float32)
    b = np.asarray(target).astype(np.float32)
    target_mse = float(np.mean((a - b) ** 2))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=a.shape).astype(np.float32)
    noise *= math.sqrt(max(target_mse, 1e-8) / float(np.mean(noise**2)))
    return np.clip(a + noise, 0, 255).astype(np.uint8)


def make_rerun_image(image_id, image_path, condition):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    body, head = load_mask_rerun(image_id)
    body = resize_mask_rerun(body, arr.shape[:2])
    head = resize_mask_rerun(head, arr.shape[:2])
    body_only = body & ~head

    if condition == "original":
        out = arr
        mask_area = np.nan
    elif condition == "gray":
        out = np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))
        mask_area = np.nan
    elif condition in BLUR_RATIOS_RERUN:
        out = gaussian_blur_rerun(arr, BLUR_RATIOS_RERUN[condition])
        mask_area = np.nan
    elif condition == "panel_face":
        out = fill_outside_rerun(arr, head)
        mask_area = float(head.mean())
    elif condition == "panel_body":
        out = fill_outside_rerun(arr, body_only)
        mask_area = float(body_only.mean())
    elif condition == "control_gray":
        gray = np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))
        out = psnr_matched_noise_rerun(arr, gray, stable_seed_rerun(image_id, condition))
        mask_area = np.nan
    elif condition.startswith("control_blur_"):
        variant = condition.replace("control_", "")
        blurred = gaussian_blur_rerun(arr, BLUR_RATIOS_RERUN[variant])
        out = psnr_matched_noise_rerun(
            arr, blurred, stable_seed_rerun(image_id, condition)
        )
        mask_area = np.nan
    elif condition == "control_panel_face":
        random_mask = random_same_area_mask_rerun(
            head.shape, int(head.sum()), stable_seed_rerun(image_id, condition)
        )
        out = fill_outside_rerun(arr, random_mask)
        mask_area = float(random_mask.mean())
    elif condition == "control_panel_body":
        random_mask = random_same_area_mask_rerun(
            body_only.shape,
            int(body_only.sum()),
            stable_seed_rerun(image_id, condition),
        )
        out = fill_outside_rerun(arr, random_mask)
        mask_area = float(random_mask.mean())
    else:
        raise KeyError(condition)

    cropped = crop_to_body_rerun(Image.fromarray(out), body)
    return cropped, arr, out, mask_area


def build_control_manifest_rerun():
    rows = []
    for r in tqdm(meta.itertuples(index=False), total=len(meta), desc="build controls"):  # noqa: F821
        for condition in RERUN_CONDITIONS:
            cropped, original, transformed, mask_area = make_rerun_image(
                r.image_id, r.image_path, condition
            )
            target_dir = IMAGE_ROOT_RERUN / str(r.split)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{r.image_id}__{condition}.jpg"
            cropped.save(target, quality=95)
            rows.append(
                {
                    "image_id": str(r.image_id),
                    "dog_id": str(r.dog_id),
                    "split": str(r.split),
                    "condition": condition,
                    "variant": condition.replace("control_", ""),
                    "control_for": next(
                        (v for v, c in RERUN_PAIR.items() if c == condition), ""
                    ),
                    "source_original": str(r.image_path),
                    "image_path": str(target),
                    "mask_area_ratio": mask_area,
                    "psnr_db": psnr_db_rerun(original, transformed),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RERUN_ROOT / "dinov3_control_manifest.csv", index=False)
    return manifest


def load_dinov3_classifier_rerun(seed):
    """Use the notebook's .pt loader, or the bundled A .npz format."""
    try:
        return load_classifier("dinov3", seed)  # noqa: F821 - notebook global
    except (FileNotFoundError, ValueError):
        candidates = [
            CHECKPOINT_ROOT / f"RQ1-dinov3-clf-s{seed}_linear_weights.npz",  # noqa: F821
            CHECKPOINT_ROOT / f"dinov3_seed{seed}.npz",  # noqa: F821
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "DINOv3 A classifier not found as .pt or .npz: "
                + ", ".join(map(str, candidates))
            )
        q = np.load(path, allow_pickle=True)
        weight = q["weight"].astype("float32")
        bias = q["bias"].astype("float32")
        classes = q["classes"].astype(str).tolist()
        return {
            "npz_weight": weight,
            "npz_bias": bias,
            "class_names": classes,
            "temperature": 1.0,
            "feature_mean": None,
            "feature_std": None,
            "path": str(path),
        }


def run_classifier_rerun(features, clf):
    if "npz_weight" in clf:
        logits = features @ clf["npz_weight"].T + clf["npz_bias"]
        logits = logits / max(float(clf.get("temperature", 1.0)), 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    return run_classifier(features, clf)  # noqa: F821 - notebook global


def run_dinov3_control_scoring_rerun(manifest):
    # get_extractor/load_classifier/run_classifier/aggregate_to_dna/
    # score_predictions are the validated helpers from the notebook.
    extractor = get_extractor("dinov3")  # noqa: F821 - notebook global
    detail_rows = []
    prediction_dir = RERUN_ROOT / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    for condition in RERUN_CONDITIONS:
        m = manifest[manifest.condition.eq(condition)].reset_index(drop=True)
        cache = FEATURE_ROOT_RERUN / f"{condition}.npy"
        if cache.exists():
            features = np.load(cache)
        else:
            feats = []
            for start in tqdm(
                range(0, len(m), FEATURE_BATCH_SIZE),  # noqa: F821 - notebook global
                desc=f"dinov3/{condition}",
            ):
                batch = m.iloc[start : start + FEATURE_BATCH_SIZE]
                images = [Image.open(p).convert("RGB") for p in batch.image_path]
                feats.append(extractor.encode(images))
            features = np.concatenate(feats, axis=0).astype(np.float32)
            np.save(cache, features)

        for seed in SEEDS:  # noqa: F821 - notebook global
            clf = load_dinov3_classifier_rerun(seed)
            probs = run_classifier_rerun(features, clf)
            dna_pred = aggregate_to_dna(probs, clf["class_names"])  # noqa: F821
            pred = pd.concat(
                [m[["image_id", "dog_id"]].reset_index(drop=True), dna_pred.add_prefix("pred_")],
                axis=1,
            )
            pred.to_csv(
                prediction_dir / f"pred_{condition}_seed{seed}.csv", index=False
            )
            scored = score_predictions(pred, m)  # noqa: F821 - notebook global
            scored["extractor"] = "dinov3"
            scored["seed"] = seed
            scored["condition"] = condition
            detail_rows.append(scored)

    detail = pd.concat(detail_rows, ignore_index=True)
    detail.to_csv(RERUN_ROOT / f"dinov3_detail_{SPLIT}.csv", index=False)  # noqa: F821

    # Reuse the notebook's exact metric and formula implementation, but with
    # the reduced scope above.
    global PAIR  # noqa: PLW0603
    old_pair = PAIR if "PAIR" in globals() else None
    PAIR = RERUN_PAIR
    try:
        summary = condition_scores(detail)  # noqa: F821 - notebook global
        extra = extra_drop(detail)  # noqa: F821 - notebook global
    finally:
        if old_pair is not None:
            PAIR = old_pair

    summary.to_csv(RERUN_ROOT / f"dinov3_condition_summary_{SPLIT}.csv", index=False)  # noqa: F821
    extra.to_csv(RERUN_ROOT / f"dinov3_extra_drop_detail_{SPLIT}.csv", index=False)  # noqa: F821

    if len(extra):
        extra_summary_rows = []
        for (extractor_name, seed, condition, breed_group), g in extra.groupby(
            ["extractor", "seed", "condition", "breed_group"], dropna=False
        ):
            for metric in [
                "extra_drop_capture_top3",
                "extra_drop_overlap",
                "extra_drop_kl",
            ]:
                mean, lo, hi = bootstrap_ci(g[metric].to_numpy())  # noqa: F821
                extra_summary_rows.append(
                    {
                        "extractor": extractor_name,
                        "seed": seed,
                        "condition": condition,
                        "breed_group": breed_group,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_dogs": len(g),
                    }
                )
        extra_summary = pd.DataFrame(extra_summary_rows)
    else:
        extra_summary = pd.DataFrame()
    extra_summary.to_csv(
        RERUN_ROOT / f"dinov3_extra_drop_summary_{SPLIT}.csv", index=False  # noqa: F821
    )

    manifest_info = {
        "created_unix": time.time(),
        "conditions": RERUN_CONDITIONS,
        "pair_controls": RERUN_PAIR,
        "seeds": list(SEEDS),  # noqa: F821
        "n_manifest_rows": int(len(manifest)),
        "n_detail_rows": int(len(detail)),
        "n_extra_drop_rows": int(len(extra)),
        "background_only": False,
        "optional_selection_experiment": False,
        "formula": "(original - variant) - (original - matched_control)",
        "panel_body_definition": "body mask excluding head mask, then fixed body crop",
        "output_root": str(RERUN_ROOT),
    }
    (RERUN_ROOT / "run_status.json").write_text(
        json.dumps(manifest_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_info


if __name__ == "__main__":
    raise RuntimeError(
        "이 파일은 D_RQ2_Colab.ipynb의 마지막 셀에서 %run으로 실행하세요. "
        "노트북 전역(meta, MASK_ROOT, DINOv3 extractor, A classifier)이 필요합니다."
    )


# In Colab, run:
#   %run /content/drive/MyDrive/.../DINOv3_control_rerun.py
# Then call:
#   _manifest = build_control_manifest_rerun()
#   _status = run_dinov3_control_scoring_rerun(_manifest)
#   print(json.dumps(_status, ensure_ascii=False, indent=2))
