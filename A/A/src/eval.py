#!/usr/bin/env python3
"""Apply A-part classifiers to one RQ2 feature set and score UF Top-1.

The script follows the frozen A-part policy:
- L2-normalize each image feature.
- Apply the three linear-softmax classifiers for one extractor.
- Average face/body probabilities per dog.
- Average probabilities across seeds 1, 2, and 3.
- Select co-dominant breeds from the full DNA records before mapping.
- Accept exact mappings only; never promote a lower-percentage mapped breed.
- Exclude dogs when none of the co-dominant breeds has an exact mapping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


META_COLUMNS = {"image_id", "dog_id", "split", "panel"}


def load_features(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        bundle = np.load(path, allow_pickle=False)
        if key not in bundle.files:
            raise ValueError(f"{path}: key '{key}' not found; available={bundle.files}")
        values = bundle[key]
    else:
        raise ValueError("--features must be .npy or .npz")
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got {values.shape}")
    return values.astype(np.float32, copy=False)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_weight(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bundle = np.load(path, allow_pickle=False)
    required = {"weight", "bias", "classes"}
    missing = required.difference(bundle.files)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    weight = bundle["weight"].astype(np.float32, copy=False)
    bias = bundle["bias"].astype(np.float32, copy=False)
    classes = bundle["classes"].astype(str).tolist()
    if weight.shape != (len(classes), weight.shape[1]) or bias.shape != (len(classes),):
        raise ValueError(f"{path}: inconsistent weight/bias/classes shapes")
    return weight, bias, classes


def parse_mapped_classes(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def build_truth(
    dna_path: Path,
    map_path: Path,
    split_path: Path,
    model_classes: set[str],
) -> pd.DataFrame:
    dna = pd.read_csv(dna_path, encoding="utf-8-sig")
    mapping = pd.read_csv(map_path, encoding="utf-8-sig")
    split = pd.read_csv(split_path, encoding="utf-8-sig")

    required_dna = {"dog_id", "breed_raw", "pct"}
    required_map = {"breed_raw", "tsinghua_classes", "status"}
    if not required_dna.issubset(dna.columns):
        raise ValueError(f"DNA file needs columns {sorted(required_dna)}")
    if not required_map.issubset(mapping.columns):
        raise ValueError(f"Breed map needs columns {sorted(required_map)}")
    if not {"dog_id", "split"}.issubset(split.columns):
        raise ValueError("Split file needs dog_id,split columns")

    dna["dog_id"] = dna["dog_id"].astype(str).str.strip()
    dna["breed_raw"] = dna["breed_raw"].astype(str).str.strip()
    dna["pct"] = pd.to_numeric(dna["pct"], errors="raise")
    mapping["breed_raw"] = mapping["breed_raw"].astype(str).str.strip()
    mapping["status"] = mapping["status"].astype(str).str.strip()
    mapping["class_set"] = mapping["tsinghua_classes"].map(parse_mapped_classes)
    lookup = {
        row.breed_raw: row.class_set if row.status == "정확" else set()
        for row in mapping.itertuples(index=False)
    }

    max_pct = dna.groupby("dog_id")["pct"].transform("max")
    dominant = dna[max_pct.eq(dna["pct"])].copy()
    rows: list[dict[str, object]] = []
    for row in split[["dog_id", "split"]].drop_duplicates().itertuples(index=False):
        dog_id = str(row.dog_id).strip()
        dog_rows = dominant[dominant["dog_id"].eq(dog_id)]
        if dog_rows.empty:
            raise ValueError(f"No DNA rows for {dog_id}")
        raw_breeds = dog_rows["breed_raw"].tolist()
        allowed: set[str] = set()
        for breed in raw_breeds:
            allowed.update(lookup.get(breed, set()))
        unknown_classes = allowed.difference(model_classes)
        if unknown_classes:
            raise ValueError(
                f"breed_map contains classes outside the checkpoint for {dog_id}: "
                f"{sorted(unknown_classes)}"
            )
        rows.append(
            {
                "dog_id": dog_id,
                "split": str(row.split).strip(),
                "dominant_pct": float(dog_rows["pct"].max()),
                "dominant_breeds_raw": "; ".join(raw_breeds),
                "allowed_model_classes": "; ".join(sorted(allowed)),
                "scorable": bool(allowed),
            }
        )
    return pd.DataFrame(rows)


def dog_probabilities(
    image_index: pd.DataFrame,
    probabilities: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    if len(image_index) != len(probabilities):
        raise ValueError(
            f"Index/features row mismatch: {len(image_index)} vs {len(probabilities)}"
        )
    required = {"image_id", "dog_id", "panel"}
    if not required.issubset(image_index.columns):
        raise ValueError(f"Evaluation index needs columns {sorted(required)}")
    metadata = image_index[["image_id", "dog_id", "panel"]].reset_index(drop=True)
    probability_frame = pd.DataFrame(probabilities, columns=classes)
    frame = pd.concat([metadata, probability_frame], axis=1)
    panel_counts = frame.groupby("dog_id")["panel"].nunique()
    if not panel_counts.eq(2).all():
        bad = panel_counts[~panel_counts.eq(2)].to_dict()
        raise ValueError(f"Each dog must have face and body panels; bad counts={bad}")
    return frame.groupby("dog_id", as_index=False)[classes].mean()


def wilson_interval(hits: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def score(
    dog_probs: pd.DataFrame,
    truth: pd.DataFrame,
    classes: list[str],
    target_split: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = truth if target_split == "all" else truth[truth["split"].eq(target_split)]
    merged = selected.merge(dog_probs, on="dog_id", how="left", validate="one_to_one")
    if merged[classes].isna().any(axis=None):
        missing = merged.loc[merged[classes].isna().any(axis=1), "dog_id"].tolist()
        raise ValueError(f"Missing predictions for dogs: {missing}")
    numeric = merged[classes].astype(float)
    merged["pred_top1"] = numeric.idxmax(axis=1)
    merged["pred_top1_probability"] = numeric.max(axis=1)
    merged["hit"] = merged.apply(
        lambda row: (
            row["pred_top1"] in set(str(row["allowed_model_classes"]).split("; "))
            if row["scorable"]
            else pd.NA
        ),
        axis=1,
    )
    scored = merged[merged["scorable"]]
    hits = int(scored["hit"].astype(bool).sum())
    total = int(len(scored))
    low, high = wilson_interval(hits, total)
    summary: dict[str, object] = {
        "split": target_split,
        "total_dogs": int(len(merged)),
        "scorable_dogs": total,
        "unscorable_dogs": int((~merged["scorable"]).sum()),
        "coverage": float(total / len(merged)),
        "top1_hits": hits,
        "top1_accuracy": float(hits / total),
        "top1_ci95_wilson_low": float(low),
        "top1_ci95_wilson_high": float(high),
    }
    return merged, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True, help="Image features (.npy or .npz)")
    parser.add_argument("--feature-key", default="cls", help="Array key when --features is .npz")
    parser.add_argument("--index", type=Path, required=True, help="Image-order CSV")
    parser.add_argument("--weights", type=Path, nargs=3, required=True, metavar=("S1", "S2", "S3"))
    parser.add_argument("--dna", type=Path, required=True)
    parser.add_argument("--breed-map", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--target-split", choices=["dev", "hold", "all"], default="dev")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = l2_normalize(load_features(args.features, args.feature_key))
    image_index = pd.read_csv(args.index, encoding="utf-8-sig")

    seed_dog_probs: list[pd.DataFrame] = []
    seed_summaries: list[dict[str, object]] = []
    classes_ref: list[str] | None = None
    truth: pd.DataFrame | None = None
    for seed, weight_path in enumerate(args.weights, start=1):
        weight, bias, classes = load_weight(weight_path)
        if features.shape[1] != weight.shape[1]:
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match {weight_path.name} "
                f"dimension {weight.shape[1]}"
            )
        if classes_ref is None:
            classes_ref = classes
            truth = build_truth(args.dna, args.breed_map, args.split, set(classes))
        elif classes != classes_ref:
            raise ValueError("The three checkpoints have different class orders")
        probs = softmax(features @ weight.T + bias)
        dog_probs = dog_probabilities(image_index, probs, classes)
        dog_probs.to_csv(
            args.output_dir / f"seed{seed}_dog_probabilities.csv",
            index=False,
            encoding="utf-8-sig",
        )
        details, summary = score(dog_probs, truth, classes, args.target_split)
        summary.update({"seed": seed, "checkpoint": weight_path.name})
        details.to_csv(
            args.output_dir / f"seed{seed}_details.csv", index=False, encoding="utf-8-sig"
        )
        seed_dog_probs.append(dog_probs)
        seed_summaries.append(summary)

    assert classes_ref is not None and truth is not None
    ordered = [frame.sort_values("dog_id").reset_index(drop=True) for frame in seed_dog_probs]
    if not all(ordered[0]["dog_id"].equals(frame["dog_id"]) for frame in ordered[1:]):
        raise ValueError("Dog order differs across seeds")
    ensemble_values = sum(frame[classes_ref] for frame in ordered) / 3
    ensemble = pd.concat(
        [ordered[0][["dog_id"]].reset_index(drop=True), ensemble_values], axis=1
    )
    ensemble.to_csv(
        args.output_dir / "ensemble_dog_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    ensemble_details, ensemble_summary = score(
        ensemble, truth, classes_ref, args.target_split
    )
    ensemble_summary.update({"seed": "ensemble", "checkpoints_averaged": 3})
    ensemble_details.to_csv(
        args.output_dir / "ensemble_details.csv", index=False, encoding="utf-8-sig"
    )

    summary_frame = pd.DataFrame(seed_summaries + [ensemble_summary])
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"individual_seeds": seed_summaries, "ensemble": ensemble_summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_frame.to_string(index=False))
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
