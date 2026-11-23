"""Create extra-drop detail and summary tables from private score details."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAIRS = {
    "area_head": "control_area_head",
    "area_body": "control_area_body",
    "area_background": "control_area_background",
    "blur_01": "control_blur_01",
    "blur_03": "control_blur_03",
    "blur_06": "control_blur_06",
    "blur_12": "control_blur_12",
    "gray": "control_gray",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detail", type=Path, required=True, help="detail_scores_<split>.csv")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", default="dev")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.detail)
    required = {"extractor", "seed", "condition", "dog_id", "capture_top3", "overlap", "kl"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    base = detail[detail.condition.eq("original")][["extractor", "seed", "dog_id", "capture_top3", "overlap", "kl"]].rename(
        columns={"capture_top3": "base_capture", "overlap": "base_overlap", "kl": "base_kl"}
    )
    rows = []
    for variant, control in PAIRS.items():
        variant_df = detail[detail.condition.eq(variant)][["extractor", "seed", "dog_id", "capture_top3", "overlap", "kl"]].rename(
            columns={"capture_top3": "variant_capture", "overlap": "variant_overlap", "kl": "variant_kl"}
        )
        control_df = detail[detail.condition.eq(control)][["extractor", "seed", "dog_id", "capture_top3", "overlap", "kl"]].rename(
            columns={"capture_top3": "control_capture", "overlap": "control_overlap", "kl": "control_kl"}
        )
        merged = base.merge(variant_df, on=["extractor", "seed", "dog_id"]).merge(control_df, on=["extractor", "seed", "dog_id"])
        merged["condition"] = variant
        merged["extra_drop_capture_top3"] = merged.control_capture - merged.variant_capture
        merged["extra_drop_overlap"] = merged.control_overlap - merged.variant_overlap
        merged["extra_drop_kl"] = merged.variant_kl - merged.control_kl
        rows.append(merged)

    if not rows:
        raise RuntimeError("No complete original/variant/control triplets found")
    extra = pd.concat(rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra.to_csv(args.output_dir / f"extra_drop_detail_{args.split}.csv", index=False)
    summary = extra.groupby(["extractor", "seed", "condition"])[
        ["extra_drop_capture_top3", "extra_drop_overlap", "extra_drop_kl"]
    ].mean().reset_index()
    summary.to_csv(args.output_dir / f"extra_drop_summary_{args.split}.csv", index=False)
    print(f"detail_rows={len(extra)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
