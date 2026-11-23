# Feature extraction specification (CLIP / DINOv2 / DINOv3)

B owns the image-level preprocessing for the three deep extractors. A's classifiers and B's prototype mixing
read the same shared feature files, so any new feature (e.g. D's transformed images) must follow this exactly.

## Image crop (before the model processor)

| Data | Crop |
|---|---|
| Tsinghua Dogs train / val | `bodybndbox` from the per-image annotation xml (falls back to `bndbox`, then the full image). The box is clipped to the actual image size. Image converted to RGB. |
| UF mixed-dog evaluation (`eval_uf_panels`) | Each face / body panel image is used as-is (no further crop). |

Excluded classes: `chinese_rural_dog`, `dhole`, `African_hunting_dog` → 127 classes (`dingo` kept).

## Models and processors

Weights are frozen. Inference runs in fp16 on a Colab T4 (batch 128). Versions at extraction time:
`transformers 5.22.1`, `torch 2.14.1+cu130`.

| Extractor | Hugging Face ID | Processor | Resize | Crop | Input |
|---|---|---|---|---|---|
| CLIP ViT-B/16 | `openai/clip-vit-base-patch16` | `CLIPImageProcessor` | shortest edge 224 | center 224×224 | 224×224 |
| DINOv2 ViT-B/14 | `facebook/dinov2-base` | `BitImageProcessor` | shortest edge 256 | center 224×224 | 224×224 |
| DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `DINOv3ViTImageProcessor` | 224×224 (no aspect keep) | none | 224×224 |

Normalization (mean / std) is each processor's default.

## Output vectors

| Extractor | `cls` | dim | `mean` | dim | Tokens (CLS + register + patch) |
|---|---|---:|---|---:|---|
| CLIP | projected image embedding (`visual_projection(pooler_output)`) | 512 | mean of patch tokens | 768 | 1 + 0 + 196 = 197 |
| DINOv2 | CLS token | 768 | mean of patch tokens | 768 | 1 + 0 + 256 = 257 |
| DINOv3 | CLS token | 768 | mean of patch tokens (4 register tokens skipped) | 768 | 1 + 4 + 196 = 201 |

Token counts were checked by `python -m src.features.check_models` on Colab.

## File format

```text
features/{backbone}/{subset}/shard_XXX.npz   image_id, cls, mean   (float16, 5,000 images per shard)
features/{backbone}/{subset}/index.csv       train/val: image_id, label   eval: image_id, dog_id, split
```

Row order of every shard equals the order in `index.csv`. Class order is the sorted training label order
(same as `A/configs/classes_127.csv`).

## Prototype mixing input (B's method)

- `cls` features, no L2 normalization (L2 changed results by at most 0.2 pp on dev; see `docs/DEV_RESULTS.md`).
- Prototypes: per-class mean of training features, rebuilt on a random 80% subsample per class for repeats r1–r3.
