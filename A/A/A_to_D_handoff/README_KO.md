# A → D 분류기 전달물

## 포함 내용

- 특징 4종 × seed 3개 = 선형 분류기 가중치 12개
- `model_summary.csv`: 순종 validation 성능 요약
- `all_runs.csv`: 실행별 상세 지표

각 `*_linear_weights.npz`에는 다음 배열이 있습니다.

```text
weight: (127, feature_dim)
bias: (127,)
classes: (127,)
```

## D가 적용하는 방법

입력 특징을 먼저 사진별 L2 정규화한 뒤 아래 식을 사용합니다.

```python
import numpy as np

bundle = np.load("RQ1-dinov3-clf-s1_linear_weights.npz")

features = features.astype(np.float32)
features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

logits = features @ bundle["weight"].T + bundle["bias"]
logits -= logits.max(axis=1, keepdims=True)
probabilities = np.exp(logits)
probabilities /= probabilities.sum(axis=1, keepdims=True)

breed_names = bundle["classes"]
```

## 특징 규칙

| 추출기 | 입력 | 차원 |
|---|---|---:|
| CLIP | NPZ의 `cls` | 512 |
| DINOv2 | NPZ의 `cls` | 768 |
| DINOv3 | NPZ의 `cls` | 768 |
| SIFT | BoVW | 512 |

D의 변형 이미지 특징도 원본과 동일한 추출기·동일한 crop/전처리를 사용해야 합니다.

각 조건은 seed 1·2·3 모두로 예측한 뒤 확률 또는 지표를 평균하고 범위를 함께 기록합니다.

## 아직 적용하지 않은 것

- 온도 보정
- DNA 품종으로 합치는 `breed_map.csv`
- 실제 DNA 비율 채점

따라서 현재 가중치는 순종 분류 baseline입니다. 온도 보정값이 확정되면 D에게 추가 전달해야 합니다.
