#!/usr/bin/env python3
"""Evaluate A-part DEV predictions under the shared ratio-estimation policy.

This script intentionally scores DEV only. It reproduces B's mapping and
captured-ancestry definitions, adds distribution overlap, and keeps the
existing Top-1 + coverage policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXACT = "정확"
SIMILAR = "유사"
NONE = "없음"
TOPK = 3
RUN_RE = re.compile(r"^RQ1-(clip|dinov2|dinov3|sift)-clf-s([123])$")


def split_classes(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def load_inputs(
    dna_path: Path, map_path: Path, split_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dna = pd.read_csv(dna_path, encoding="utf-8-sig")
    breed_map = pd.read_csv(map_path, encoding="utf-8-sig")
    split = pd.read_csv(split_path, encoding="utf-8-sig")
    if not {"dog_id", "breed_raw", "pct"}.issubset(dna.columns):
        raise ValueError("DNA file needs dog_id, breed_raw, pct")
    if not {"breed_raw", "tsinghua_classes", "status"}.issubset(breed_map.columns):
        raise ValueError("Breed map needs breed_raw, tsinghua_classes, status")
    if not {"dog_id", "split"}.issubset(split.columns):
        raise ValueError("Split file needs dog_id, split")
    dna["dog_id"] = dna["dog_id"].astype(str).str.strip()
    dna["breed_raw"] = dna["breed_raw"].astype(str).str.strip()
    dna["pct"] = pd.to_numeric(dna["pct"], errors="coerce")
    breed_map["breed_raw"] = breed_map["breed_raw"].astype(str).str.strip()
    breed_map["status"] = breed_map["status"].astype(str).str.strip()
    split["dog_id"] = split["dog_id"].astype(str).str.strip()
    split["split"] = split["split"].astype(str).str.strip()
    unknown = set(dna["breed_raw"]) - set(breed_map["breed_raw"])
    if unknown:
        raise ValueError(f"Breeds missing from map: {sorted(unknown)}")
    return dna, breed_map, split[["dog_id", "split"]].drop_duplicates()


def build_ratio_truth(
    dna: pd.DataFrame, breed_map: pd.DataFrame, split: pd.DataFrame
) -> pd.DataFrame:
    lookup = {
        row.breed_raw: ("; ".join(split_classes(row.tsinghua_classes)), row.status)
        for row in breed_map.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for dog in split.itertuples(index=False):
        dog_rows = dna[dna["dog_id"].eq(dog.dog_id)]
        if dog_rows.empty:
            raise ValueError(f"No DNA rows for {dog.dog_id}")
        pct_sum = float(dog_rows["pct"].sum(skipna=True))
        aggregate: OrderedDict[str, dict[str, object]] = OrderedDict()
        for dna_row in dog_rows.itertuples(index=False):
            if pd.isna(dna_row.pct):
                continue
            target, status = lookup[dna_row.breed_raw]
            if status == NONE or not target:
                continue
            item = aggregate.setdefault(
                target, {"pct": 0.0, "statuses": set(), "raw": []}
            )
            item["pct"] = float(item["pct"]) + float(dna_row.pct)
            item["statuses"].add(status)
            item["raw"].append(dna_row.breed_raw)
        for item in aggregate.values():
            item["status"] = SIMILAR if SIMILAR in item["statuses"] else EXACT
        exact_total = sum(
            float(item["pct"])
            for item in aggregate.values()
            if item["status"] == EXACT
        )
        both_total = sum(float(item["pct"]) for item in aggregate.values())
        common = {
            "dog_id": dog.dog_id,
            "split": dog.split,
            "coverage": exact_total / 100.0,
            "coverage_with_similar": both_total / 100.0,
            "pct_sum": pct_sum,
        }
        if not aggregate:
            rows.append(
                {
                    **common,
                    "target_classes": "",
                    "status": "",
                    "breed_raw": "",
                    "pct": np.nan,
                    "share_norm": np.nan,
                    "share_norm_with_similar": np.nan,
                }
            )
        for target, item in aggregate.items():
            pct = float(item["pct"])
            status = str(item["status"])
            rows.append(
                {
                    **common,
                    "target_classes": target,
                    "status": status,
                    "breed_raw": "; ".join(item["raw"]),
                    "pct": pct,
                    "share_norm": (
                        pct / exact_total
                        if status == EXACT and exact_total > 0
                        else np.nan
                    ),
                    "share_norm_with_similar": (
                        pct / both_total if both_total > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def class_to_breed_group(
    breed_map: pd.DataFrame, allowed: set[str]
) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in breed_map[breed_map["status"].isin(allowed)].itertuples(index=False):
        classes = split_classes(row.tsinghua_classes)
        key = "; ".join(classes)
        for class_name in classes:
            if class_name in output and output[class_name] != key:
                raise ValueError(
                    f"Class {class_name} belongs to two groups: "
                    f"{output[class_name]} and {key}"
                )
            output[class_name] = key
    return output


def truth_dicts(
    truth_long: pd.DataFrame, target_split: str, allowed: set[str]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    selected = truth_long[truth_long["split"].eq(target_split)]
    orig: dict[str, dict[str, float]] = {}
    norm: dict[str, dict[str, float]] = {}
    for dog_id, group in selected.groupby("dog_id"):
        mapped = group[group["status"].isin(allowed)]
        orig[dog_id] = {
            row.target_classes: float(row.pct) / 100.0
            for row in mapped.itertuples(index=False)
            if pd.notna(row.pct)
        }
        norm_column = "share_norm_with_similar" if SIMILAR in allowed else "share_norm"
        norm[dog_id] = {
            row.target_classes: float(getattr(row, norm_column))
            for row in mapped.itertuples(index=False)
            if pd.notna(getattr(row, norm_column))
        }
    return orig, norm


def merge_to_breeds(prob: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return prob.T.groupby(lambda column: mapping.get(column, column), sort=False).sum().T


def captured_share(merged: pd.DataFrame, truth: dict[str, dict[str, float]]) -> pd.Series:
    values: dict[str, float] = {}
    for dog_id, row in merged.iterrows():
        top = row.sort_values(ascending=False, kind="stable").index[:TOPK]
        values[dog_id] = sum(truth.get(dog_id, {}).get(name, 0.0) for name in top)
    return pd.Series(values, dtype=float)


def distribution_overlap(
    merged: pd.DataFrame, truth: dict[str, dict[str, float]]
) -> pd.Series:
    values: dict[str, float] = {}
    for dog_id, row in merged.iterrows():
        values[dog_id] = sum(
            min(float(row.get(name, 0.0)), float(value))
            for name, value in truth.get(dog_id, {}).items()
        )
    return pd.Series(values, dtype=float)


def captured_ceiling(truth: dict[str, dict[str, float]], dogs: list[str]) -> pd.Series:
    return pd.Series(
        {
            dog_id: sum(sorted(truth.get(dog_id, {}).values(), reverse=True)[:TOPK])
            for dog_id in dogs
        },
        dtype=float,
    )


def overlap_ceiling(truth: dict[str, dict[str, float]], dogs: list[str]) -> pd.Series:
    return pd.Series(
        {dog_id: sum(truth.get(dog_id, {}).values()) for dog_id in dogs}, dtype=float
    )


def top1_answers(
    dna: pd.DataFrame,
    breed_map: pd.DataFrame,
    classes: set[str],
    dogs: list[str],
    allowed_statuses: set[str],
) -> dict[str, set[str]]:
    lookup = {
        row.breed_raw: (row.status, split_classes(row.tsinghua_classes))
        for row in breed_map.itertuples(index=False)
    }
    answers: dict[str, set[str]] = {}
    for dog_id in dogs:
        group = dna[dna["dog_id"].eq(dog_id) & dna["pct"].notna()]
        max_pct = group["pct"].max()
        allowed: set[str] = set()
        for breed in group.loc[group["pct"].eq(max_pct), "breed_raw"]:
            status, names = lookup[breed]
            if status in allowed_statuses:
                allowed.update(name for name in names if name in classes)
        answers[dog_id] = allowed
    return answers


def top1_metrics(prob: pd.DataFrame, answers: dict[str, set[str]]) -> dict[str, float]:
    top = prob.idxmax(axis=1)
    usable = [dog_id for dog_id in prob.index if answers.get(dog_id)]
    hits = sum(top.loc[dog_id] in answers[dog_id] for dog_id in usable)
    total = len(usable)
    return {
        "top1_hits": int(hits),
        "top1_scorable": int(total),
        "top1_accuracy": float(hits / total),
        "top1_coverage": float(total / len(prob)),
    }


def paired_vs_baseline(values: pd.Series, baseline: pd.Series) -> dict[str, object]:
    difference = (values - baseline).dropna()
    mean = float(difference.mean())
    two_se = (
        float(2 * difference.std(ddof=1) / math.sqrt(len(difference)))
        if len(difference) >= 3
        else float("nan")
    )
    if not np.isfinite(two_se) or abs(mean) <= two_se:
        verdict = "판단 불가"
    else:
        verdict = "찍기보다 높음" if mean > 0 else "찍기보다 낮음"
    return {"difference_mean": mean, "two_se": two_se, "verdict": verdict}


def load_predictions(
    preds_root: Path, target_split: str
) -> tuple[dict[tuple[str, int], pd.DataFrame], list[str]]:
    runs: dict[tuple[str, int], pd.DataFrame] = {}
    classes_ref: list[str] | None = None
    for run_dir in sorted(preds_root.glob("RQ1-*-clf-s*")):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        files = sorted(run_dir.glob("pred_*_dogs.csv"))
        if len(files) != 1:
            raise ValueError(f"{run_dir}: expected one dog prediction CSV, got {len(files)}")
        frame = pd.read_csv(files[0], encoding="utf-8-sig")
        classes = [column for column in frame.columns if column not in {"dog_id", "split"}]
        if classes_ref is None:
            classes_ref = classes
        elif classes != classes_ref:
            raise ValueError(f"Class order mismatch in {run_dir.name}")
        frame = frame[frame["split"].eq(target_split)].copy()
        if frame["dog_id"].duplicated().any():
            raise ValueError(f"Duplicate dog rows in {run_dir.name}")
        values = frame[classes].astype(float)
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"Probability rows do not sum to one in {run_dir.name}")
        runs[(match.group(1), int(match.group(2)))] = values.set_axis(frame["dog_id"])
    if len(runs) != 12 or classes_ref is None:
        raise ValueError(f"Expected 12 prediction runs, found {len(runs)}")
    return runs, classes_ref


def training_frequency(train_index_path: Path, classes: list[str]) -> pd.Series:
    index = pd.read_csv(train_index_path, encoding="utf-8-sig")
    label_column = "breed_name" if "breed_name" in index.columns else "label"
    selected = index[index["split"].eq("train")].copy()
    if "keep_127" in selected.columns:
        keep = selected["keep_127"].astype(str).str.lower().isin({"true", "1"})
        selected = selected[keep]
    frequency = selected[label_column].value_counts(normalize=True)
    output = frequency.reindex(classes).fillna(0.0)
    if not np.isclose(output.sum(), 1.0):
        raise ValueError("Training-frequency baseline does not sum to one")
    return output


def run_metrics(
    prob: pd.DataFrame,
    mapping: dict[str, str],
    orig: dict[str, dict[str, float]],
    norm: dict[str, dict[str, float]],
    answers: dict[str, set[str]],
) -> tuple[dict[str, float], dict[str, pd.Series]]:
    merged = merge_to_breeds(prob, mapping)
    per_dog = {
        "captured_orig": captured_share(merged, orig),
        "captured_norm": captured_share(merged, norm),
        "overlap_orig": distribution_overlap(merged, orig),
        "overlap_norm": distribution_overlap(merged, norm),
    }
    summary = top1_metrics(prob, answers)
    summary.update({name: float(values.mean()) for name, values in per_dog.items()})
    return summary, per_dog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-root", type=Path, required=True)
    parser.add_argument("--dna", type=Path, required=True)
    parser.add_argument("--breed-map", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-split", choices=["dev"], default="dev")
    parser.add_argument(
        "--mapping-policy",
        choices=["exact", "exact_similar"],
        default="exact",
        help="Which breed-map statuses count as mappable",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dna, breed_map, split = load_inputs(args.dna, args.breed_map, args.split)
    truth_long = build_ratio_truth(dna, breed_map, split)
    truth_long.to_csv(
        args.output_dir / "truth_UF.csv", index=False, encoding="utf-8-sig"
    )
    allowed = {EXACT} if args.mapping_policy == "exact" else {EXACT, SIMILAR}
    orig, norm = truth_dicts(truth_long, args.target_split, allowed)
    dogs = sorted(split.loc[split["split"].eq(args.target_split), "dog_id"].tolist())
    if len(dogs) != 20:
        raise ValueError(f"Expected 20 DEV dogs, found {len(dogs)}")

    runs, classes = load_predictions(args.preds_root, args.target_split)
    mapping = class_to_breed_group(breed_map, allowed)
    answers = top1_answers(dna, breed_map, set(classes), dogs, allowed)

    frequency = training_frequency(args.train_index, classes)
    guessing_prob = pd.DataFrame(
        np.repeat(frequency.to_numpy()[None, :], len(dogs), axis=0),
        index=dogs,
        columns=classes,
    )
    guessing_summary, guessing_per_dog = run_metrics(
        guessing_prob, mapping, orig, norm, answers
    )
    ceilings = {
        "captured_orig": captured_ceiling(orig, dogs),
        "captured_norm": captured_ceiling(norm, dogs),
        "overlap_orig": overlap_ceiling(orig, dogs),
        "overlap_norm": overlap_ceiling(norm, dogs),
    }
    ceiling_means = {name: float(values.mean()) for name, values in ceilings.items()}

    run_rows: list[dict[str, object]] = []
    run_per_dog: dict[tuple[str, int], dict[str, pd.Series]] = {}
    ensemble_rows: list[dict[str, object]] = []
    ensemble_per_dog: dict[str, dict[str, pd.Series]] = {}
    for model in ["clip", "dinov2", "dinov3", "sift"]:
        for seed in [1, 2, 3]:
            summary, per_dog = run_metrics(runs[(model, seed)], mapping, orig, norm, answers)
            run_rows.append({"model_name": model, "seed": seed, **summary})
            run_per_dog[(model, seed)] = per_dog
        ensemble_prob = sum(runs[(model, seed)] for seed in [1, 2, 3]) / 3.0
        ensemble_summary, per_dog = run_metrics(
            ensemble_prob, mapping, orig, norm, answers
        )
        ensemble_rows.append({"model_name": model, "seed": "ensemble", **ensemble_summary})
        ensemble_per_dog[model] = per_dog

    run_frame = pd.DataFrame(run_rows)
    ensemble_frame = pd.DataFrame(ensemble_rows)
    run_frame.to_csv(
        args.output_dir / "dev_ratio_metrics_individual_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ensemble_frame.to_csv(
        args.output_dir / "dev_ratio_metrics_seed_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_names = ["captured_orig", "captured_norm", "overlap_orig", "overlap_norm"]
    combined_rows: list[dict[str, object]] = []
    for model in ["clip", "dinov2", "dinov3", "sift"]:
        ensemble_record = ensemble_frame[ensemble_frame["model_name"].eq(model)].iloc[0]
        row: dict[str, object] = {
            "model_name": model,
            "top1_accuracy": float(ensemble_record["top1_accuracy"]),
            "top1_hits": int(ensemble_record["top1_hits"]),
            "top1_scorable": int(ensemble_record["top1_scorable"]),
            "top1_coverage": float(ensemble_record["top1_coverage"]),
        }
        model_runs = run_frame[run_frame["model_name"].eq(model)]
        for metric in metric_names:
            comparison = paired_vs_baseline(
                ensemble_per_dog[model][metric], guessing_per_dog[metric]
            )
            row[f"{metric}_ensemble"] = float(ensemble_record[metric])
            row[f"{metric}_seed_mean"] = float(model_runs[metric].mean())
            row[f"{metric}_seed_min"] = float(model_runs[metric].min())
            row[f"{metric}_seed_max"] = float(model_runs[metric].max())
            row[f"{metric}_guessing"] = float(guessing_summary[metric])
            row[f"{metric}_ceiling"] = float(ceiling_means[metric])
            row[f"{metric}_vs_guessing_diff"] = comparison["difference_mean"]
            row[f"{metric}_vs_guessing_2se"] = comparison["two_se"]
            row[f"{metric}_vs_guessing_verdict"] = comparison["verdict"]
        combined_rows.append(row)
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(
        args.output_dir / "dev_all_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    b_reference = {
        "captured_orig": {"clip": 0.217, "dinov2": 0.205, "dinov3": 0.130},
        "captured_norm": {"clip": 0.378, "dinov2": 0.334, "dinov3": 0.222},
        "guessing": {"captured_orig": 0.088, "captured_norm": 0.168},
        "ceiling": {"captured_orig": 0.541, "captured_norm": 0.970},
    }
    comparison_rows = []
    for model in ["clip", "dinov2", "dinov3"]:
        a_row = combined[combined["model_name"].eq(model)].iloc[0]
        comparison_rows.append(
            {
                "model_name": model,
                "A_mapping_policy": args.mapping_policy,
                "B_reference_mapping_inferred": "exact",
                "A_captured_orig": float(a_row["captured_orig_ensemble"]),
                "B_captured_orig": b_reference["captured_orig"][model],
                "A_minus_B_orig": float(a_row["captured_orig_ensemble"])
                - b_reference["captured_orig"][model],
                "A_captured_norm": float(a_row["captured_norm_ensemble"]),
                "B_captured_norm": b_reference["captured_norm"][model],
                "A_minus_B_norm": float(a_row["captured_norm_ensemble"])
                - b_reference["captured_norm"][model],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "A_vs_B_captured_ancestry_dev.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    chart_specs = [
        ("captured_orig", "Captured ancestry@3 — original DNA share"),
        ("captured_norm", "Captured ancestry@3 — renormalized"),
        ("overlap_orig", "Distribution overlap — original DNA share"),
        ("overlap_norm", "Distribution overlap — renormalized"),
    ]
    models = combined["model_name"].tolist()
    for axis, (metric, title) in zip(axes.flat, chart_specs):
        values = combined[f"{metric}_ensemble"].to_numpy(float)
        bars = axis.bar(models, values, color="#5b4bd5")
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
        axis.axhline(guessing_summary[metric], color="#f0a000", linestyle="--", label="guessing")
        axis.axhline(ceiling_means[metric], color="#333333", linestyle=":", label="ceiling")
        axis.set_title(title)
        axis.set_ylim(0, max(1.0, ceiling_means[metric] + 0.08))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(
        f"A classifier — UF DEV ratio-estimation metrics (20 dogs, {args.mapping_policy})"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "dev_ratio_metrics.png", dpi=180)
    plt.close(fig)

    metadata = {
        "split": "dev",
        "n_dogs": len(dogs),
        "n_runs": len(runs),
        "face_body": "already averaged in pred_*_dogs.csv",
        "seed_representative": "mean probabilities across seeds 1,2,3 before scoring",
        "mapping_status": sorted(allowed),
        "mapping_policy": args.mapping_policy,
        "captured_topk": TOPK,
        "overlap_definition": "sum min(predicted merged breed probability, DNA share)",
        "hold_evaluated_for_new_metrics": False,
        "B_reference_source": "B/docs/METRICS.md and B/scripts/provisional_score_uf.py",
        "B_script_sha": "ca52aa9a448cc7ec94e76babebe45263b61b025a",
        "B_published_baselines_inferred_mapping": "exact",
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nA seed-ensemble DEV metrics")
    print(
        combined[
            [
                "model_name",
                "top1_accuracy",
                "top1_coverage",
                "captured_orig_ensemble",
                "captured_norm_ensemble",
                "overlap_orig_ensemble",
                "overlap_norm_ensemble",
            ]
        ].to_string(index=False)
    )
    print("\nBaselines")
    print({name: round(float(guessing_summary[name]), 6) for name in metric_names})
    print({name: round(float(ceiling_means[name]), 6) for name in metric_names})
    print(f"\nResults written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
