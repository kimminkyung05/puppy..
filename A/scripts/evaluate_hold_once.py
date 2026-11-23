#!/usr/bin/env python3
"""One-time HOLD evaluation using the frozen A-part Top-1 policy."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mapped_classes(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def build_hold_truth(dna_path: Path, map_path: Path, index_path: Path) -> pd.DataFrame:
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
    hold_ids = set(dogs.loc[dogs["split"].eq("hold"), "dog_id"])
    if len(hold_ids) != 80:
        raise ValueError(f"Expected 80 hold dogs, found {len(hold_ids)}")

    hold_dna = dna[dna["dog_id"].isin(hold_ids)].copy()
    if set(hold_dna["dog_id"]) != hold_ids:
        raise ValueError("Some hold dogs have no DNA rows")
    max_pct = hold_dna.groupby("dog_id")["pct"].transform("max")
    dominant = hold_dna[max_pct.eq(hold_dna["pct"])].copy()

    rows = []
    for dog_id in sorted(hold_ids):
        dog_rows = dominant[dominant["dog_id"].eq(dog_id)]
        raw_breeds = dog_rows["breed_raw"].tolist()
        allowed: set[str] = set()
        for breed in raw_breeds:
            allowed.update(map_lookup.get(breed, set()))
        rows.append(
            {
                "dog_id": dog_id,
                "split": "hold",
                "dominant_pct": float(dog_rows["pct"].max()),
                "dominant_breeds_raw": "; ".join(raw_breeds),
                "allowed_model_classes": "; ".join(sorted(allowed)),
                "scorable": bool(allowed),
            }
        )
    return pd.DataFrame(rows)


def score_frame(probability_frame: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = probability_frame[probability_frame["split"].eq("hold")].copy()
    if len(frame) != 80:
        raise ValueError(f"Expected 80 hold prediction rows, found {len(frame)}")
    probability_columns = [column for column in frame.columns if column not in {"dog_id", "split"}]
    numeric = frame[probability_columns].apply(pd.to_numeric, errors="raise")
    frame["pred_top1"] = numeric.idxmax(axis=1)
    frame["pred_top1_probability"] = numeric.max(axis=1)
    merged = truth.merge(
        frame[["dog_id", "pred_top1", "pred_top1_probability"]],
        on="dog_id", how="left", validate="one_to_one",
    )
    merged["hit"] = merged.apply(
        lambda row: row["pred_top1"] in set(str(row["allowed_model_classes"]).split("; "))
        if row["scorable"] else pd.NA,
        axis=1,
    )
    scored = merged[merged["scorable"]].copy()
    hits = int(scored["hit"].astype(bool).sum())
    total = int(len(scored))
    low, high = wilson_interval(hits, total)
    summary = {
        "hold_total_dogs": int(len(merged)),
        "hold_scorable_dogs": total,
        "hold_unscorable_dogs": int((~merged["scorable"]).sum()),
        "coverage": float(total / len(merged)),
        "top1_hits": hits,
        "top1_accuracy": float(hits / total),
        "top1_ci95_wilson_low": float(low),
        "top1_ci95_wilson_high": float(high),
    }
    return merged, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dna", type=Path, required=True)
    parser.add_argument("--breed-map", type=Path, required=True)
    parser.add_argument("--eval-index", type=Path, required=True)
    parser.add_argument("--pure-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != "FINAL_HOLD_ONCE":
        raise SystemExit("Hold evaluation requires --confirm FINAL_HOLD_ONCE")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = args.output_dir / "HOLD_EVALUATED_ONCE.json"
    if sentinel.exists():
        raise SystemExit(f"REFUSING TO RE-RUN: {sentinel} already exists")

    truth = build_hold_truth(args.dna, args.breed_map, args.eval_index)
    truth.to_csv(args.output_dir / "hold_truth_top1.csv", index=False, encoding="utf-8-sig")

    individual_rows = []
    model_probabilities: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(args.results_root.glob("RQ1-*-clf-s*/pred_*_dogs.csv")):
        run_name = path.parent.name
        model_name = run_name.split("-")[1]
        seed = int(run_name.rsplit("s", 1)[1])
        prediction = pd.read_csv(path, encoding="utf-8-sig")
        details, metrics = score_frame(prediction, truth)
        metrics.update({"run_name": run_name, "model_name": model_name, "seed": seed})
        individual_rows.append(metrics)
        details.to_csv(
            args.output_dir / f"{run_name}_hold_details.csv", index=False, encoding="utf-8-sig"
        )
        model_probabilities.setdefault(model_name, []).append(prediction)

    individual = pd.DataFrame(individual_rows).sort_values(["model_name", "seed"])
    if len(individual) != 12:
        raise ValueError(f"Expected 12 runs, found {len(individual)}")
    individual.to_csv(args.output_dir / "hold_top1_individual_seeds.csv", index=False, encoding="utf-8-sig")

    ensemble_rows = []
    for model_name, frames in sorted(model_probabilities.items()):
        if len(frames) != 3:
            raise ValueError(f"{model_name}: expected 3 seed frames, found {len(frames)}")
        prepared = [
            frame.sort_values(["split", "dog_id"]).reset_index(drop=True) for frame in frames
        ]
        if not all(prepared[0]["dog_id"].equals(frame["dog_id"]) for frame in prepared[1:]):
            raise ValueError(f"{model_name}: seed dog order mismatch")
        probability_columns = [
            column for column in prepared[0].columns if column not in {"dog_id", "split"}
        ]
        mean_probability = sum(frame[probability_columns].astype(float) for frame in prepared) / 3
        ensemble = prepared[0][["dog_id", "split"]].copy()
        ensemble[probability_columns] = mean_probability
        details, metrics = score_frame(ensemble, truth)
        metrics.update({"model_name": model_name, "seeds_averaged": 3})
        ensemble_rows.append(metrics)
        details.to_csv(
            args.output_dir / f"RQ1-{model_name}-clf-seed-ensemble_hold_details.csv",
            index=False, encoding="utf-8-sig",
        )
        ensemble.to_csv(
            args.output_dir / f"RQ1-{model_name}-clf-seed-ensemble_hold_probabilities.csv",
            index=False, encoding="utf-8-sig",
        )

    ensemble_summary = pd.DataFrame(ensemble_rows).sort_values(
        "top1_accuracy", ascending=False
    )
    ensemble_summary.to_csv(
        args.output_dir / "hold_top1_seed_ensemble_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    pure = pd.read_csv(args.pure_summary, encoding="utf-8-sig")
    combined = pure[["model_name", "validation_accuracy_mean", "validation_accuracy_std"]].merge(
        ensemble_summary[[
            "model_name", "top1_hits", "hold_scorable_dogs", "hold_total_dogs",
            "coverage", "top1_accuracy", "top1_ci95_wilson_low", "top1_ci95_wilson_high",
        ]],
        on="model_name", how="inner", validate="one_to_one",
    )
    combined = combined.rename(
        columns={
            "validation_accuracy_mean": "purebred_top1_mean",
            "validation_accuracy_std": "purebred_top1_std",
            "top1_accuracy": "mixed_hold_top1",
            "top1_ci95_wilson_low": "mixed_hold_top1_ci95_low",
            "top1_ci95_wilson_high": "mixed_hold_top1_ci95_high",
        }
    ).sort_values("mixed_hold_top1", ascending=False)
    combined.to_csv(args.output_dir / "final_purebred_vs_mixed_top1.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pure_plot = combined.sort_values("purebred_top1_mean", ascending=False)
    bars0 = axes[0].bar(pure_plot["model_name"], pure_plot["purebred_top1_mean"] * 100, color="#4c78a8")
    axes[0].bar_label(bars0, labels=[f"{v*100:.1f}%" for v in pure_plot["purebred_top1_mean"]], padding=3)
    axes[0].set_title("Purebred validation")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)

    mixed_plot = combined.sort_values("mixed_hold_top1", ascending=False)
    means = mixed_plot["mixed_hold_top1"].to_numpy() * 100
    lower = means - mixed_plot["mixed_hold_top1_ci95_low"].to_numpy() * 100
    upper = mixed_plot["mixed_hold_top1_ci95_high"].to_numpy() * 100 - means
    bars1 = axes[1].bar(mixed_plot["model_name"], means, yerr=np.vstack([lower, upper]), capsize=5, color="#f28e2b")
    axes[1].bar_label(bars1, labels=[f"{v:.1f}%" for v in means], padding=3)
    axes[1].set_title("Mixed-dog holdout")
    axes[1].set_ylim(0, max(40, means.max() + upper.max() + 8))
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Top-1 accuracy reported separately")
    fig.tight_layout()
    fig.savefig(args.output_dir / "final_purebred_vs_mixed_top1.png", dpi=180)
    plt.close(fig)

    executed_at = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
    sentinel.write_text(
        json.dumps(
            {
                "executed_at_asia_seoul": executed_at,
                "confirmation": args.confirm,
                "policy": {
                    "hold_scored_once": True,
                    "partial_labels_not_renormalized": True,
                    "co_dominant_tie": "any mapped co-dominant class is correct",
                    "all_dominant_breeds_unmapped": "excluded and reported in coverage",
                    "face_body": "probability mean per dog",
                    "seeds": "probability mean across seeds 1, 2, 3",
                },
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print("\nIndividual seeds\n", individual[["run_name", "top1_hits", "hold_scorable_dogs", "top1_accuracy"]].to_string(index=False))
    print("\nSeed ensembles\n", ensemble_summary.to_string(index=False))
    print("\nFinal purebred vs mixed\n", combined.to_string(index=False))
    print(f"\nOne-time marker written: {sentinel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
