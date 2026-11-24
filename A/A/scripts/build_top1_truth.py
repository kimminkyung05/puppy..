#!/usr/bin/env python3
"""Build the final exact-only Top-1 truth table from the full DNA records.

Dominant breeds are selected before mapping.  Only exact mappings among the
co-dominant breeds are accepted; a lower-percentage mapped breed is never
promoted when the dominant breed is unmapped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXACT = "정확"


def split_classes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dna", type=Path, required=True)
    parser.add_argument("--breed-map", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dna = pd.read_csv(args.dna, encoding="utf-8-sig")
    breed_map = pd.read_csv(args.breed_map, encoding="utf-8-sig")
    split = pd.read_csv(args.split, encoding="utf-8-sig")
    classes = set(
        pd.read_csv(args.classes, encoding="utf-8-sig")["class_name"]
        .astype(str)
        .str.strip()
    )

    for frame, columns, name in [
        (dna, {"dog_id", "breed_raw", "pct"}, "DNA"),
        (breed_map, {"breed_raw", "tsinghua_classes", "status"}, "breed map"),
        (split, {"dog_id", "split"}, "split"),
    ]:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} columns missing: {sorted(missing)}")

    dna["dog_id"] = dna["dog_id"].astype(str).str.strip()
    dna["breed_raw"] = dna["breed_raw"].astype(str).str.strip()
    dna["pct"] = pd.to_numeric(dna["pct"], errors="coerce")
    breed_map["breed_raw"] = breed_map["breed_raw"].astype(str).str.strip()
    breed_map["status"] = breed_map["status"].astype(str).str.strip()
    split["dog_id"] = split["dog_id"].astype(str).str.strip()
    split["split"] = split["split"].astype(str).str.strip()

    lookup = {
        row.breed_raw: (row.status, split_classes(row.tsinghua_classes))
        for row in breed_map.itertuples(index=False)
    }
    unknown = sorted(set(dna["breed_raw"]) - set(lookup))
    if unknown:
        raise ValueError(f"DNA breeds missing from breed map: {unknown}")

    rows: list[dict[str, object]] = []
    for dog in split.itertuples(index=False):
        records = dna[dna["dog_id"].eq(dog.dog_id) & dna["pct"].notna()]
        if records.empty:
            raise ValueError(f"No DNA records for {dog.dog_id}")
        dominant_pct = float(records["pct"].max())
        dominant = records[records["pct"].eq(dominant_pct)]
        allowed: set[str] = set()
        for breed in dominant["breed_raw"]:
            status, mapped_classes = lookup[breed]
            if status == EXACT:
                allowed.update(name for name in mapped_classes if name in classes)
        rows.append(
            {
                "dog_id": dog.dog_id,
                "split": dog.split,
                "dominant_pct": dominant_pct,
                "dominant_breeds_raw": "; ".join(dominant["breed_raw"].tolist()),
                "allowed_model_classes": "; ".join(sorted(allowed)),
                "scorable": bool(allowed),
                "mapping_policy": "exact_only",
                "truth_source": "full_dna_before_mapping",
            }
        )

    output = pd.DataFrame(rows)
    if len(output) != len(split) or output["dog_id"].duplicated().any():
        raise ValueError("Top-1 truth must contain exactly one row per split dog")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")

    dev = output[output["split"].eq("dev")]
    excluded = dev.loc[~dev["scorable"], "dog_id"].tolist()
    print(f"wrote {len(output)} dogs to {args.output}")
    print(f"DEV scorable: {int(dev['scorable'].sum())}/{len(dev)}")
    print(f"DEV excluded: {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
