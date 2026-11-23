# Dog Ancestry Estimation — Part A Classifier Baseline

This directory contains the code and aggregate results for Part A of the mixed-dog ancestry project.

## Task

Frozen image features are used to train a single linear softmax classifier over 127 Tsinghua Dogs classes. Purebred validation and mixed-dog Top-1 performance are reported separately.

## Feature extractors

| Extractor | Input feature | Dimension |
|---|---|---:|
| CLIP | CLS token | 512 |
| DINOv2 | CLS token | 768 |
| DINOv3 | CLS token | 768 |
| SIFT | BoVW histogram | 512 |

All input features are normalized row-wise with L2 normalization. Each classifier is trained with seeds 1, 2, and 3.

## Main results

### Purebred validation

| Feature | Top-1 accuracy |
|---|---:|
| DINOv3 | 87.94% |
| DINOv2 | 85.63% |
| CLIP | 72.45% |
| SIFT | 9.53% |

### Mixed-dog holdout

The holdout was evaluated once under the frozen policy. Seventy-one of 80 dogs were scorable (coverage: 88.75%).

| Feature | Hits / scorable | Top-1 accuracy |
|---|---:|---:|
| CLIP | 10/71 | 14.1% |
| DINOv3 | 9/71 | 12.7% |
| DINOv2 | 8/71 | 11.3% |
| SIFT | 1/71 | 1.4% |

The confidence intervals of the three deep features overlap substantially; the numerical ordering should not be interpreted as a statistically established ranking.

## Repository structure

```text
configs/    frozen class order, mapping, split, and preprocessing notes
notebooks/  Colab quick-start notebook
scripts/    training, export, summary, and locked evaluation scripts
src/        reusable evaluation entry point
results/    aggregate CSV results
figures/    result plots
docs/       evaluation policy and final holdout report
data/       data availability notes only
```

## Current evaluation scope

The committed evaluator implements the classification formulation agreed for
the current review: purebred and mixed-dog experiments are reported separately,
and mixed-dog performance uses Top-1 accuracy plus coverage. The team is still
deciding whether the final task will remain classification or become ancestry
proportion estimation. If the latter is selected, captured ancestry at Top-3
and any additional distribution metrics must be added before final reporting.

## Data availability

Raw Tsinghua Dogs images, Reddit/UF images, DNA screenshots, precomputed features, ZIP archives, and local checkpoints are intentionally excluded. See `data/README.md`.

## Reproduction notes

The scripts expect precomputed feature files and the evaluation metadata described in the project documentation. The one-time holdout marker and final aggregate outputs are included, but the holdout evaluation should not be rerun or used to tune the model.
