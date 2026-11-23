#!/usr/bin/env python3
"""Combine all run metrics into meeting-ready tables and a comparison plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> int:
    rows = []
    for path in sorted(RESULTS.glob("RQ1-*-clf-s*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("metrics.json을 찾지 못했습니다")

    runs = pd.DataFrame(rows).sort_values(["model_name", "seed"])
    runs.to_csv(ROOT / "all_runs.csv", index=False, encoding="utf-8-sig")

    metric_columns = [
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_top5_accuracy",
        "validation_log_loss",
        "elapsed_seconds",
    ]
    summary = runs.groupby("model_name")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("validation_accuracy_mean", ascending=False)
    summary.to_csv(ROOT / "model_summary.csv", index=False, encoding="utf-8-sig")

    plot_data = summary.set_index("model_name")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = plot_data.index.tolist()
    means = plot_data["validation_accuracy_mean"].to_numpy() * 100
    errors = plot_data["validation_accuracy_std"].fillna(0).to_numpy() * 100
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#9d9da1"]
    bars = ax.bar(order, means, yerr=errors, capsize=5, color=colors[: len(order)])
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in means], padding=4)
    ax.set_ylabel("Tsinghua validation accuracy (%)")
    ax.set_ylim(max(0, means.min() - 8), min(100, means.max() + 5))
    ax.set_title("A baseline: single-layer classifier on frozen features")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "validation_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# A파트 B 특징 실험 결과",
        "",
        "## 설정",
        "",
        "- Tsinghua Dogs 127클래스: train 61,592장 / validation 5,080장",
        "- 특징: 각 추출기의 `cls` 벡터, 사진별 L2 정규화",
        "- 분류기: 은닉층 없는 단일 선형층 + softmax",
        "- 클래스 불균형: balanced sample weight",
        "- seed: 1, 2, 3",
        "- 최대 30 epoch",
        "",
        "## Validation 결과",
        "",
        "| 특징 | Accuracy 평균±표준편차 | Top-5 평균 | Log loss 평균 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.validation_accuracy_mean*100:.2f}% ± "
            f"{row.validation_accuracy_std*100:.2f}%p | "
            f"{row.validation_top5_accuracy_mean*100:.2f}% | "
            f"{row.validation_log_loss_mean:.3f} |"
        )
    lines += [
        "",
        "## 현재 해석",
        "",
        "- 순종 validation에서는 DINOv3가 가장 높고, DINOv2, CLIP 순입니다.",
        "- 세 seed의 편차가 작아 결과가 안정적입니다.",
        "- 평가 패널 200장은 100마리로 평균 집계해 예측 CSV를 만들었습니다.",
        "- DNA 정답과 `breed_map.csv`가 아직 연결되지 않아 혈통 비율 지표·순열 검정은 다음 단계입니다.",
        "- B에게 `cls` 사용이 맞는지 확인하기 전까지 이 결과는 `CLS baseline`으로 기록합니다.",
        "- 30 epoch에서 손실이 아직 감소 중이므로 최종 설정 확정 전 epoch 수는 dev에서 재검토합니다.",
    ]
    if "sift" in set(summary["model_name"]):
        lines += [
            "",
            "## SIFT 상태",
            "",
            "- 수정본은 127클래스(train 61,592 / validation 5,080)이며 `Fila Braziliero` 220장이 포함됐습니다.",
            "- 평가 특징도 200장(100마리 × face/body)으로 전달받아 개체별 예측을 생성했습니다.",
            "- SIFT는 고전 특징 하한선으로 딥 특징보다 성능이 크게 낮게 나타났습니다.",
            "- SIFT learning rate는 0.001 / 0.01 / 0.03을 비교해 0.01을 선택했습니다.",
        ]
    (ROOT / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(runs[["run_name", "validation_accuracy", "validation_top5_accuracy", "validation_log_loss"]].to_string(index=False))
    print("\n모델 요약\n", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
