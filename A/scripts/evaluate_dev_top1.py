#!/usr/bin/env python3
"""Evaluate mixed-dog Top-1 on DEV only. HOLD labels are never scored."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mapped_classes(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def build_dev_truth(dna_path: Path, map_path: Path, index_path: Path) -> pd.DataFrame:
    dna = pd.read_csv(dna_path, encoding="utf-8-sig")
    mapping = pd.read_csv(map_path, encoding="utf-8-sig")
    index = pd.read_csv(index_path, encoding="utf-8-sig")
    dna["dog_id"] = dna["dog_id"].astype(str).str.strip()
    dna["breed_raw"] = dna["breed_raw"].astype(str).str.strip()
    dna["pct"] = pd.to_numeric(dna["pct"], errors="raise")
    mapping["breed_raw"] = mapping["breed_raw"].astype(str).str.strip()
    mapping["class_set"] = mapping["tsinghua_classes"].map(mapped_classes)
    map_lookup = dict(zip(mapping["breed_raw"], mapping["class_set"]))

    dogs = index[["dog_id", "split"]].drop_duplicates().copy()
    dogs["dog_id"] = dogs["dog_id"].astype(str).str.strip()
    dev_ids = set(dogs.loc[dogs["split"].eq("dev"), "dog_id"])
    if len(dev_ids) != 20:
        raise ValueError(f"Expected 20 dev dogs, found {len(dev_ids)}")

    dev_dna = dna[dna["dog_id"].isin(dev_ids)].copy()
    max_pct = dev_dna.groupby("dog_id")["pct"].transform("max")
    dominant = dev_dna[max_pct.eq(dev_dna["pct"])].copy()

    rows = []
    for dog_id in sorted(dev_ids):
        dog_rows = dominant[dominant["dog_id"].eq(dog_id)]
        raw_breeds = dog_rows["breed_raw"].tolist()
        allowed: set[str] = set()
        for breed in raw_breeds:
            allowed.update(map_lookup.get(breed, set()))
        rows.append(
            {
                "dog_id": dog_id,
                "split": "dev",
                "dominant_pct": float(dog_rows["pct"].max()),
                "dominant_breeds_raw": "; ".join(raw_breeds),
                "allowed_model_classes": "; ".join(sorted(allowed)),
                "scorable": bool(allowed),
            }
        )
    return pd.DataFrame(rows)


def score_prediction(pred_path: Path, truth: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    required = {"dog_id", "split"}
    if not required.issubset(pred.columns):
        raise ValueError(f"{pred_path}: missing {required - set(pred.columns)}")
    pred = pred[pred["split"].eq("dev")].copy()
    if len(pred) != 20:
        raise ValueError(f"{pred_path}: expected 20 dev rows, found {len(pred)}")
    probability_columns = [column for column in pred.columns if column not in {"dog_id", "split"}]
    numeric = pred[probability_columns].apply(pd.to_numeric, errors="raise")
    pred["pred_top1"] = numeric.idxmax(axis=1)
    pred["pred_top1_probability"] = numeric.max(axis=1)

    merged = truth.merge(
        pred[["dog_id", "pred_top1", "pred_top1_probability"]], on="dog_id", how="left", validate="one_to_one"
    )
    merged["hit"] = merged.apply(
        lambda row: row["pred_top1"] in set(str(row["allowed_model_classes"]).split("; "))
        if row["scorable"]
        else pd.NA,
        axis=1,
    )
    scored = merged[merged["scorable"]].copy()
    run_name = pred_path.parent.name
    summary = {
        "run_name": run_name,
        "model_name": run_name.split("-")[1],
        "seed": int(run_name.rsplit("s", 1)[1]),
        "dev_total_dogs": int(len(merged)),
        "dev_scorable_dogs": int(len(scored)),
        "dev_unscorable_dogs": int((~merged["scorable"]).sum()),
        "coverage": float(len(scored) / len(merged)),
        "top1_hits": int(scored["hit"].astype(bool).sum()),
        "top1_accuracy": float(scored["hit"].astype(bool).mean()),
    }
    return summary, merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dna", type=Path, required=True)
    parser.add_argument("--breed-map", type=Path, required=True)
    parser.add_argument("--eval-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth = build_dev_truth(args.dna, args.breed_map, args.eval_index)
    truth.to_csv(args.output_dir / "dev_truth_top1.csv", index=False, encoding="utf-8-sig")
    summaries = []
    per_run_frames: dict[str, pd.DataFrame] = {}
    for path in sorted(args.results_root.glob("RQ1-*-clf-s*/pred_*_dogs.csv")):
        summary, details = score_prediction(path, truth)
        summaries.append(summary)
        per_run_frames[summary["run_name"]] = details
        details.to_csv(
            args.output_dir / f"{summary['run_name']}_dev_details.csv", index=False, encoding="utf-8-sig"
        )

    runs = pd.DataFrame(summaries).sort_values(["model_name", "seed"])
    if len(runs) != 12:
        raise ValueError(f"Expected 12 runs, found {len(runs)}")
    runs.to_csv(args.output_dir / "dev_top1_all_runs.csv", index=False, encoding="utf-8-sig")
    model_summary = (
        runs.groupby("model_name")
        .agg(
            seeds=("seed", "count"),
            scorable_dogs=("dev_scorable_dogs", "first"),
            coverage=("coverage", "first"),
            top1_accuracy_mean=("top1_accuracy", "mean"),
            top1_accuracy_std=("top1_accuracy", "std"),
            top1_accuracy_min=("top1_accuracy", "min"),
            top1_accuracy_max=("top1_accuracy", "max"),
        )
        .reset_index()
        .sort_values("top1_accuracy_mean", ascending=False)
    )
    model_summary.to_csv(args.output_dir / "dev_top1_model_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = model_summary["top1_accuracy_mean"].to_numpy() * 100
    errors = model_summary["top1_accuracy_std"].fillna(0).to_numpy() * 100
    bars = ax.bar(
        model_summary["model_name"], means, yerr=errors, capsize=5,
        color=["#4c78a8", "#59a14f", "#f28e2b", "#9d9da1"][: len(model_summary)],
    )
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in means], padding=4)
    ax.set_ylabel("Mixed-dog DEV Top-1 accuracy (%)")
    ax.set_ylim(0, max(20, means.max() + 12))
    ax.set_title("DEV only — 19 scorable dogs, 3 seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dev_top1_comparison.png", dpi=180)
    plt.close(fig)

    report = {
        "policy": {
            "evaluated_split": "dev only",
            "hold_scored": False,
            "partial_DNA_labels": True,
            "dominant_ties": "any mapped co-dominant breed counts as correct",
            "unmapped_dominant_breed": "excluded and reported as coverage",
        },
        "dev_total": int(len(truth)),
        "dev_scorable": int(truth["scorable"].sum()),
        "dev_unscorable": int((~truth["scorable"]).sum()),
        "unscorable_dogs": truth.loc[~truth["scorable"], "dog_id"].tolist(),
    }
    (args.output_dir / "dev_evaluation_policy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(model_summary.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
