# A파트 학습 모델 정리

## 1. 전체 구성

A파트에서 학습한 모델은 총 **12개 분류기 체크포인트**이다.

- 특징 추출기 4종: CLIP, DINOv2, DINOv3, SIFT-BoVW
- 각 특징마다 seed 1·2·3으로 분류기 3개 학습
- 총 모델 수: `4종 × 3 seeds = 12개`
- 출력: Tsinghua Dogs 127개 품종 확률
- 분류기 구조: 은닉층 없는 단일 선형층 + softmax
- 학습 데이터: Tsinghua Dogs train 61,592장
- 검증 데이터: Tsinghua Dogs validation 5,080장
- 전처리: 이미지별 특징 벡터에 행 단위 L2 정규화 적용
- 학습 범위: 특징 추출기는 고정하고 마지막 분류기만 학습

## 2. 모델별 입력

| 모델 | 입력 특징 | 입력 차원 | 체크포인트 수 |
|---|---|---:|---:|
| CLIP | CLS 특징 | 512 | 3개 (seed 1·2·3) |
| DINOv2 | CLS 특징 | 768 | 3개 (seed 1·2·3) |
| DINOv3 | CLS 특징 | 768 | 3개 (seed 1·2·3) |
| SIFT | BoVW 히스토그램 | 512 | 3개 (seed 1·2·3) |

각 `.npz` 체크포인트에는 다음 값이 저장되어 있다.

```text
weight:  (127, feature_dimension)
bias:    (127,)
classes: (127,)
```

추론 과정은 다음과 같다.

```python
x = x / max(norm(x), 1e-12)
logits = x @ weight.T + bias
probability = softmax(logits)
```

## 3. 체크포인트 파일

### CLIP

```text
RQ1-clip-clf-s1_linear_weights.npz
RQ1-clip-clf-s2_linear_weights.npz
RQ1-clip-clf-s3_linear_weights.npz
```

### DINOv2

```text
RQ1-dinov2-clf-s1_linear_weights.npz
RQ1-dinov2-clf-s2_linear_weights.npz
RQ1-dinov2-clf-s3_linear_weights.npz
```

### DINOv3

```text
RQ1-dinov3-clf-s1_linear_weights.npz
RQ1-dinov3-clf-s2_linear_weights.npz
RQ1-dinov3-clf-s3_linear_weights.npz
```

### SIFT-BoVW

```text
RQ1-sift-clf-s1_linear_weights.npz
RQ1-sift-clf-s2_linear_weights.npz
RQ1-sift-clf-s3_linear_weights.npz
```

## 4. 순종견 validation 성능

Tsinghua Dogs 공식 validation 데이터에서 측정한 seed 1·2·3 평균 Top-1 정확도이다.

| 특징 | Top-1 평균 정확도 |
|---|---:|
| DINOv3 | 87.94% |
| DINOv2 | 85.63% |
| CLIP | 72.45% |
| SIFT | 9.53% |

## 5. 최종 HOLD 믹스견 성능

- 전체 HOLD 개체: 80마리
- Top-1 평가 가능: 71마리
- 제외: 9마리
- Top-1 scorable coverage: 71/80 = 88.75%
- 대표 결과: seed 1·2·3 확률 평균 앙상블

| 특징 | 정답/평가 가능 | Seed 앙상블 Top-1 |
|---|---:|---:|
| CLIP | 10/71 | 14.1% |
| DINOv3 | 9/71 | 12.7% |
| DINOv2 | 8/71 | 11.3% |
| SIFT | 1/71 | 1.4% |

## 6. 자산 범위

이 12개 체크포인트는 **이미 추출된 특징 벡터를 Tsinghua Dogs 127개 품종으로 분류하는 마지막 분류기 가중치**이다.

A파트 전달물에 이미지 특징 추출기 자체는 포함되지 않는다.

- CLIP·DINOv2·DINOv3 특징 추출 코드, 체크포인트 및 이미지 전처리: B파트 자산
- SIFT-BoVW 코드북, 특징 추출 코드 및 파라미터: C파트 자산
- A파트: 위 특징을 입력받는 127클래스 선형 softmax 분류기와 평가 코드

새로운 이미지에 A 분류기를 적용하려면 B·C파트가 사용한 것과 동일한 특징 추출 파이프라인으로 특징을 먼저 생성해야 한다.

## 7. 전달 파일 위치

12개 분류기 체크포인트와 클래스 순서·전처리 정보를 묶은 파일:

```text
outputs/A_classifier_drive_upload/models/A_to_D_Classifier_Weights.zip
```

개별 모델 위치:

```text
outputs/A_classifier_drive_upload/models/
```
