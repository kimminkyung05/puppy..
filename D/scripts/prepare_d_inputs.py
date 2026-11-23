"""Prepare a private Drive export for the D development run.

This script does not download data and does not write source images to GitHub.
Pass local/private paths explicitly; no user-specific absolute path is embedded.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_DEV_IDS = [
    "UF001", "UF004", "UF009", "UF013", "UF016", "UF018", "UF030", "UF038", "UF041", "UF059",
    "UF061", "UF067", "UF069", "UF073", "UF076", "UF077", "UF092", "UF097", "UF103", "UF110",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Private D data directory")
    p.add_argument("--received-dir", type=Path, required=True, help="Private directory containing dna_UF.csv and breed_map_UF_draft.csv")
    p.add_argument("--output", type=Path, required=True, help="Private output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output
    image_out = out / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    with (args.data_root / "index_eval_uf.csv").open(encoding="utf-8-sig", newline="") as f:
        index_rows = list(csv.DictReader(f))
    available = {r["dog_id"] for r in index_rows if r.get("split") == "dev"}
    dev_ids = [dog_id for dog_id in DEFAULT_DEV_IDS if dog_id in available]
    if not dev_ids:
        raise RuntimeError("No development dog IDs were found in index_eval_uf.csv")

    originals = args.data_root / "uf_originals_dev"
    meta_rows = []
    for dog_id in dev_ids:
        source = originals / f"{dog_id}.jpg"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, image_out / source.name)
        meta_rows.append({"image_id": dog_id, "dog_id": dog_id, "relative_path": f"images/{source.name}", "split": "dev"})

    with (out / "mixdog_metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "dog_id", "relative_path", "split"])
        writer.writeheader()
        writer.writerows(meta_rows)

    with (args.received_dir / "dna_UF.csv").open(encoding="utf-8-sig", newline="") as f:
        dna_rows = [r for r in csv.DictReader(f) if r.get("dog_id") in set(dev_ids)]
    with (out / "dna_long.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dog_id", "breed", "fraction"])
        writer.writeheader()
        for row in dna_rows:
            writer.writerow({"dog_id": row["dog_id"], "breed": row["breed_raw"], "fraction": float(row["pct"]) / 100.0})

    with (args.received_dir / "breed_map_UF_draft.csv").open(encoding="utf-8-sig", newline="") as f:
        map_rows = list(csv.DictReader(f))
    mapped = []
    for row in map_rows:
        target = (row.get("tsinghua_classes") or "").strip()
        for ts_class in (part.strip() for part in target.split(";")):
            if ts_class:
                mapped.append({"tsinghua_class": ts_class, "dna_breed": row["breed_raw"].strip(), "map_status": row["status"].strip()})
    with (out / "breed_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tsinghua_class", "dna_breed", "map_status"])
        writer.writeheader()
        writer.writerows(mapped)

    print(f"prepared dogs={len(dev_ids)} images={len(meta_rows)} dna_rows={len(dna_rows)} map_rows={len(mapped)}")


if __name__ == "__main__":
    main()
