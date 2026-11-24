# Classifier input specification

| Extractor | Input array | Dimension | Classifier preprocessing |
|---|---|---:|---|
| CLIP | `cls` | 512 | Per-row L2 normalization |
| DINOv2 | `cls` | 768 | Per-row L2 normalization |
| DINOv3 | `cls` | 768 | Per-row L2 normalization |
| SIFT | BoVW histogram | 512 | Per-row L2 normalization |

Each checkpoint stores:

```text
weight:  (127, feature_dimension)
bias:    (127,)
classes: (127,)
```

Inference is:

```python
x = x / max(norm(x), 1e-12)
logits = x @ weight.T + bias
probability = softmax(logits)
```

The image-level resize, crop, normalization, model checkpoint and SIFT codebook
belong to the feature-extraction pipelines. Obtain the exact final versions
from B (CLIP/DINOv2/DINOv3) and C (SIFT).
