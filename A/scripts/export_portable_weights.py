#!/usr/bin/env python3
"""Export portable linear weights from completed joblib checkpoints."""

from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent / "results"


def main() -> int:
    count = 0
    for checkpoint_path in sorted(ROOT.glob("RQ1-*-clf-s*/*.joblib")):
        bundle = joblib.load(checkpoint_path)
        classifier = bundle["classifier"]
        classes = np.asarray(bundle["classes"], dtype=str)
        output = checkpoint_path.with_name(checkpoint_path.stem + "_linear_weights.npz")
        np.savez_compressed(
            output,
            weight=np.asarray(classifier.coefs_[0].T, dtype=np.float32),
            bias=np.asarray(classifier.intercepts_[0], dtype=np.float32),
            classes=classes,
        )
        print(output)
        count += 1
    print(f"exported={count}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
