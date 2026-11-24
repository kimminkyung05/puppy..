#!/usr/bin/env python3
"""Train A-part linear softmax classifiers directly from B's feature ZIP files.

The classifier is an sklearn MLPClassifier with no hidden layers, so the model is
exactly a single affine layer followed by softmax. Features are L2-normalized.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, top_k_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


def shard_number(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def load_split(zip_path: Path, model_name: str, split: str, feature_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    prefix = f"{model_name}/{split}"
    with zipfile.ZipFile(zip_path) as archive:
        index_name = f"{prefix}/index.csv"
        if index_name not in archive.namelist():
            raise FileNotFoundError(f"ZIP 내부에 {index_name}이 없습니다")
        index = pd.read_csv(archive.open(index_name), dtype=str)
        shard_names = sorted(
            [name for name in archive.namelist() if name.startswith(f"{prefix}/shard_") and name.endswith(".npz")],
            key=shard_number,
        )
        if not shard_names:
            raise FileNotFoundError(f"{prefix}에 shard_*.npz가 없습니다")

        feature_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        for name in shard_names:
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as shard:
                if feature_key not in shard.files:
                    raise KeyError(f"{name}에 {feature_key!r} 배열이 없습니다: {shard.files}")
                feature_parts.append(np.asarray(shard[feature_key], dtype=np.float32))
                id_parts.append(np.asarray(shard["image_id"]).astype(str))

    features = np.concatenate(feature_parts, axis=0)
    image_ids = np.concatenate(id_parts, axis=0)
    if len(features) != len(index):
        raise ValueError(f"{model_name}/{split}: 특징 {len(features)}행, index {len(index)}행으로 불일치")
    if "image_id" not in index.columns:
        raise ValueError(f"{model_name}/{split}/index.csv에 image_id가 없습니다")
    index_ids = index["image_id"].astype(str).to_numpy()
    if not np.array_equal(image_ids, index_ids):
        mismatches = np.flatnonzero(image_ids != index_ids)
        first = int(mismatches[0])
        raise ValueError(
            f"{model_name}/{split}: NPZ와 index 순서 불일치. 첫 위치 {first}: "
            f"{image_ids[first]!r} != {index_ids[first]!r}"
        )
    return features, index


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    features /= norms
    return features


def save_training_curve(model: MLPClassifier, output: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(model.loss_curve_) + 1), model.loss_curve_, marker="o", markersize=3)
    plt.xlabel("epoch")
    plt.ylabel("training log loss")
    plt.title("Single-layer softmax training curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def train_one(
    zip_path: Path,
    model_name: str,
    output_root: Path,
    seed: int,
    feature_key: str,
    max_epochs: int,
) -> dict[str, object]:
    run_name = f"RQ1-{model_name}-clf-s{seed}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"[{run_name}] train 특징 읽는 중")
    x_train, train_index = load_split(zip_path, model_name, "train", feature_key)
    print(f"[{run_name}] validation 특징 읽는 중")
    x_val, val_index = load_split(zip_path, model_name, "val", feature_key)
    if "label" not in train_index or "label" not in val_index:
        raise ValueError("train/val index.csv에 label 열이 필요합니다")

    x_train = l2_normalize(x_train)
    x_val = l2_normalize(x_val)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_index["label"].astype(str))
    unknown_val = sorted(set(val_index["label"].astype(str)) - set(encoder.classes_))
    if unknown_val:
        raise ValueError(f"train에 없는 validation 클래스: {unknown_val}")
    y_val = encoder.transform(val_index["label"].astype(str))
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32)

    classifier = MLPClassifier(
        hidden_layer_sizes=(),
        solver="adam",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=max_epochs,
        shuffle=True,
        random_state=seed,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=True,
    )
    print(f"[{run_name}] 단일 선형층 학습 시작: {x_train.shape}, 클래스 {len(encoder.classes_)}개")
    classifier.fit(x_train, y_train, sample_weight=sample_weight)

    val_prob = classifier.predict_proba(x_val)
    val_pred = np.argmax(val_prob, axis=1)
    metrics: dict[str, object] = {
        "run_name": run_name,
        "model_name": model_name,
        "seed": seed,
        "feature_key": feature_key,
        "l2_normalized": True,
        "class_weight": "balanced",
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": int(len(encoder.classes_)),
        "epochs_run": int(classifier.n_iter_),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
        "validation_top5_accuracy": float(
            top_k_accuracy_score(y_val, val_prob, k=5, labels=np.arange(len(encoder.classes_)))
        ),
        "validation_log_loss": float(log_loss(y_val, val_prob, labels=np.arange(len(encoder.classes_)))),
    }

    checkpoint = {
        "classifier": classifier,
        "classes": encoder.classes_.tolist(),
        "feature_key": feature_key,
        "l2_normalized": True,
        "model_name": model_name,
        "seed": seed,
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

    val_out = pd.DataFrame(
        {
            "image_id": val_index["image_id"],
            "true_label": val_index["label"],
            "pred_label": encoder.inverse_transform(val_pred),
            "correct": val_pred == y_val,
            "confidence": val_prob.max(axis=1),
        }
    )
    val_out.to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    save_training_curve(classifier, run_dir / "training_curve.png")

    print(f"[{run_name}] 평가 패널 특징 읽는 중")
    x_eval, eval_index = load_split(zip_path, model_name, "eval_uf_panels", feature_key)
    x_eval = l2_normalize(x_eval)
    eval_prob = classifier.predict_proba(x_eval)
    probability_columns = encoder.classes_.tolist()
    image_prob = pd.concat(
        [eval_index.reset_index(drop=True), pd.DataFrame(eval_prob, columns=probability_columns)], axis=1
    )
    image_prob.to_csv(run_dir / f"pred_{run_name}_images.csv", index=False, encoding="utf-8-sig")

    if "dog_id" not in eval_index.columns:
        raise ValueError("eval_uf_panels/index.csv에 dog_id가 필요합니다")
    dog_meta_columns = [column for column in ("dog_id", "split") if column in image_prob.columns]
    dog_prob = image_prob.groupby(dog_meta_columns, as_index=False)[probability_columns].mean()
    dog_prob.to_csv(run_dir / f"pred_{run_name}_dogs.csv", index=False, encoding="utf-8-sig")

    metrics["eval_image_rows"] = int(len(image_prob))
    metrics["eval_dog_rows"] = int(len(dog_prob))
    metrics["elapsed_seconds"] = float(time.time() - started)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--model", choices=["clip", "dinov2", "dinov3"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--feature-key", choices=["cls", "mean"], default="cls")
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"ZIP 파일을 찾을 수 없습니다: {args.zip}")
    all_metrics = []
    for seed in args.seeds:
        all_metrics.append(
            train_one(args.zip, args.model, args.output_dir, seed, args.feature_key, args.max_epochs)
        )
    pd.DataFrame(all_metrics).to_csv(
        args.output_dir / f"summary_{args.model}.csv", index=False, encoding="utf-8-sig"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
