#!/usr/bin/env python3
"""Train provisional SIFT classifiers from sift_all.npy and its index.csv."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, top_k_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    values /= norms
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--eval-features", type=Path)
    parser.add_argument("--eval-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    features = np.load(args.features, mmap_mode="r")
    index = pd.read_csv(args.index)
    if len(features) != len(index):
        raise SystemExit(f"특징 {len(features)}행과 index {len(index)}행이 다릅니다")
    required = {"path", "breed_name", "split", "keep_127"}
    missing = required - set(index.columns)
    if missing:
        raise SystemExit(f"index 필수 열 누락: {sorted(missing)}")

    keep = truthy(index["keep_127"])
    selected = index.loc[keep].reset_index(drop=True)
    selected_features = np.asarray(features[keep.to_numpy()], dtype=np.float32)
    train_mask = selected["split"].eq("train").to_numpy()
    val_mask = selected["split"].eq("validation").to_numpy()
    x_train = l2_normalize(selected_features[train_mask])
    x_val = l2_normalize(selected_features[val_mask])
    train_index = selected.loc[train_mask].reset_index(drop=True)
    val_index = selected.loc[val_mask].reset_index(drop=True)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_index["breed_name"].astype(str))
    y_val = encoder.transform(val_index["breed_name"].astype(str))
    sample_weight = compute_sample_weight("balanced", y_train).astype(np.float32)

    x_eval = None
    eval_index = None
    if bool(args.eval_features) != bool(args.eval_index):
        raise SystemExit("--eval-features와 --eval-index는 함께 지정해야 합니다")
    if args.eval_features and args.eval_index:
        eval_raw = np.load(args.eval_features, mmap_mode="r")
        eval_index = pd.read_csv(args.eval_index)
        if len(eval_raw) != len(eval_index):
            raise SystemExit(
                f"평가 특징 {len(eval_raw)}행과 평가 index {len(eval_index)}행이 다릅니다"
            )
        eval_required = {"image_id", "dog_id", "split"}
        eval_missing = eval_required - set(eval_index.columns)
        if eval_missing:
            raise SystemExit(f"평가 index 필수 열 누락: {sorted(eval_missing)}")
        x_eval = l2_normalize(eval_raw)

    for seed in args.seeds:
        run_name = f"RQ1-sift-clf-s{seed}"
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        classifier = MLPClassifier(
            hidden_layer_sizes=(),
            solver="adam",
            alpha=1e-4,
            batch_size=1024,
            learning_rate_init=args.learning_rate,
            max_iter=args.max_epochs,
            shuffle=True,
            random_state=seed,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            verbose=True,
        )
        print(f"[{run_name}] 학습: {x_train.shape}, validation: {x_val.shape}, classes={len(encoder.classes_)}")
        classifier.fit(x_train, y_train, sample_weight=sample_weight)
        probability = classifier.predict_proba(x_val)
        prediction = np.argmax(probability, axis=1)
        metrics = {
            "run_name": run_name,
            "model_name": "sift",
            "seed": seed,
            "feature_key": "BoVW-512",
            "l2_normalized": True,
            "class_weight": "balanced",
            "learning_rate_init": float(args.learning_rate),
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "feature_dim": int(x_train.shape[1]),
            "n_classes": int(len(encoder.classes_)),
            "epochs_run": int(classifier.n_iter_),
            "validation_accuracy": float(accuracy_score(y_val, prediction)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, prediction)),
            "validation_top5_accuracy": float(
                top_k_accuracy_score(y_val, probability, k=5, labels=np.arange(len(encoder.classes_)))
            ),
            "validation_log_loss": float(
                log_loss(y_val, probability, labels=np.arange(len(encoder.classes_)))
            ),
            "eval_image_rows": int(len(x_eval)) if x_eval is not None else 0,
            "eval_dog_rows": int(eval_index["dog_id"].nunique()) if eval_index is not None else 0,
            "status": "final_127_classes" if len(encoder.classes_) == 127 else "provisional_non127_classes",
            "elapsed_seconds": float(time.time() - started),
        }
        checkpoint = {
            "classifier": classifier,
            "classes": encoder.classes_.tolist(),
            "feature_key": "BoVW-512",
            "l2_normalized": True,
            "model_name": "sift",
            "seed": seed,
            "status": metrics["status"],
        }
        joblib.dump(checkpoint, run_dir / f"{run_name}.joblib", compress=3)
        np.savez_compressed(
            run_dir / f"{run_name}_linear_weights.npz",
            weight=np.asarray(classifier.coefs_[0].T, dtype=np.float32),
            bias=np.asarray(classifier.intercepts_[0], dtype=np.float32),
            classes=np.asarray(encoder.classes_, dtype=str),
        )
        pd.DataFrame({"class_index": np.arange(len(encoder.classes_)), "breed_label": encoder.classes_}).to_csv(
            run_dir / "class_order.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "image_id": val_index["path"].astype(str).map(lambda value: Path(value).stem),
                "true_label": val_index["breed_name"],
                "pred_label": encoder.inverse_transform(prediction),
                "correct": prediction == y_val,
                "confidence": probability.max(axis=1),
            }
        ).to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        if x_eval is not None and eval_index is not None:
            eval_probability = classifier.predict_proba(x_eval)
            probability_columns = encoder.classes_.tolist()
            image_probability = pd.concat(
                [
                    eval_index.reset_index(drop=True),
                    pd.DataFrame(eval_probability, columns=probability_columns),
                ],
                axis=1,
            )
            image_probability.to_csv(
                run_dir / f"pred_{run_name}_images.csv", index=False, encoding="utf-8-sig"
            )
            dog_probability = image_probability.groupby(
                ["dog_id", "split"], as_index=False
            )[probability_columns].mean()
            dog_probability.to_csv(
                run_dir / f"pred_{run_name}_dogs.csv", index=False, encoding="utf-8-sig"
            )
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(1, len(classifier.loss_curve_) + 1), classifier.loss_curve_, marker="o", markersize=3)
        plt.xlabel("epoch")
        plt.ylabel("training log loss")
        plt.title("SIFT BoVW single-layer softmax")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(run_dir / "training_curve.png", dpi=160)
        plt.close()
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
