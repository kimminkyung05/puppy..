"""Validate the expected extractor/condition/seed prediction matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXTRACTORS = ("clip", "dinov2", "dinov3", "sift")
SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", type=Path, required=True, help="conditions.csv")
    p.add_argument("--output-root", type=Path, required=True, help="Private folder containing prediction CSV files")
    p.add_argument("--output", type=Path, required=True, help="Validation CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.conditions.open(encoding="utf-8-sig", newline="") as f:
        conditions = [row["condition"] for row in csv.DictReader(f)]

    rows = []
    for extractor in EXTRACTORS:
        for condition in conditions:
            for seed in SEEDS:
                expected = args.output_root / f"pred_dev_{extractor}_{condition}_seed{seed}.csv"
                rows.append({
                    "extractor": extractor,
                    "condition": condition,
                    "seed": seed,
                    "expected_path": str(expected),
                    "present": expected.exists(),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"checked={len(rows)} present={sum(row['present'] for row in rows)} missing={sum(not row['present'] for row in rows)}")


if __name__ == "__main__":
    main()
