# A 담당: RQ1 순종 분류기 기준선 · 공용 평가 절차

혼합 시각 범주의 구성 비율 추정(믹스견 사진 → DNA 혈통 비율) 팀 프로젝트에서
A 담당이 만든 코드와 요약 결과를 정리한 폴더이다.

현재 코드는 동결된 특징 벡터 위에 127클래스 선형 softmax 분류기를 학습하고,
순종견과 믹스견을 서로 다른 데이터셋에서 별도로 평가한다.

## 1. 실험 구성

| 특징 추출기 | 분류기 입력 | 차원 |
|---|---|---:|
| CLIP ViT-B/16 | CLS 특징 | 512 |
| DINOv2 ViT-B/14 | CLS 특징 | 768 |
| DINOv3 ViT-B/16 | CLS 특징 | 768 |
| SIFT + BoVW | 시각 단어 히스토그램 | 512 |

- 모든 입력 특징은 사진별 L2 정규화를 적용한다.
- 특징마다 seed 1, 2, 3으로 선형 분류기를 학습한다.
- 전체 학습 횟수는 `특징 4종 × seed 3개 = 12회`이다.
- 모델 출력 클래스는 Tsinghua Dogs 127개이다.

## 2. 현재 평가 기준

### 순종견

- Tsinghua Dogs validation에서 Top-1 정확도를 계산한다.

### 믹스견

- 순종견 결과와 합치지 않고 UF 평가셋에서 별도로 계산한다.
- face와 body 사진의 확률을 개체 단위로 평균한다.
- seed 1·2·3의 확률을 평균해 대표 결과를 만든다.
- `없음` 혈통까지 포함한 전체 DNA에서 공동 1위를 먼저 결정한다.
- 공동 1위 중 `status=정확`인 클래스 하나를 맞히면 정답으로 처리한다.
- 정확 매핑 가능한 공동 1위가 없으면 제외하고 Top-1 scorable coverage를 보고한다.
- dev에서 기준을 고정했으며 최종 HOLD 평가는 아직 실행하지 않았다.

현재 리뷰에서는 `Top-1 + coverage`를 유지하면서, dev에서 비율 추정 지표인
포착 혈통 비율@3과 분포 겹침을 원래 비율/재정규화 기준으로 모두 계산했다.
최종 기준은 `정확 매핑만` 사용하며, 원래 DNA 비율 포착@3을 주 지표로,
재정규화 포착@3과 재정규화 분포 겹침을 보조지표로 사용한다.

## 3. 주요 결과

### 순종견 validation

| 특징 | 평균 Top-1 정확도 |
|---|---:|
| DINOv3 | 87.94% |
| DINOv2 | 85.63% |
| CLIP | 72.45% |
| SIFT | 9.53% |

### 믹스견 DEV

- 평가 개체: 20마리
- 주 지표와 함께 보고하는 DNA mapping coverage: 0.5665(56.65%)
- Top-1 평가 가능 개체: 18/20
- Top-1 scorable coverage: 90%
- Top-1 제외 개체: `UF001`, `UF076`

최종 HOLD 평가는 설정 고정 후 한 번만 수행할 예정이며 현재 결과에는 포함하지
않는다.

## 4. 폴더 구조

```text
A/
├─ configs/     클래스 순서, 최종 매핑, 분할, 전처리 설명
├─ data/        데이터 파일을 저장소에서 제외한 이유와 준비 방법
├─ docs/        평가 기준, DEV 결과, A 방법론
├─ figures/     validation·DEV 결과 그래프
├─ results/     집계된 DEV 결과 CSV
├─ scripts/     학습·내보내기·집계·DEV 평가 코드
├─ src/         특징 → 분류기 → 개체 단위 평가 실행 코드
├─ requirements.txt
└─ README.md
```

## 5. 실행 환경

```bash
pip install -r requirements.txt
```

학습 및 평가 스크립트는 원본 사진이 아니라 B·C가 공유한 사전 추출 특징과
평가 메타데이터를 입력으로 받는다. Google Colab 무료 CPU 환경에서 실행할
수 있으며 고성능 GPU는 필요하지 않다. 자세한 인자는 각 스크립트의
`--help`에서 확인할 수 있다.

예를 들어 Colab에서 저장소를 연 뒤 다음과 같이 설치한다.

```bash
!pip install -r requirements.txt
!python scripts/train_from_feature_zips.py --help
!python scripts/train_sift_npy.py --help
!python scripts/evaluate_dev_ratio_metrics.py --help
```

## 6. 데이터 및 체크포인트

다음 파일은 용량·저작권·개인정보 및 저장소 관리 문제로 GitHub에 포함하지 않는다.

- Tsinghua Dogs 원본 이미지와 annotation
- UF/Reddit 원본 이미지와 DNA 원본 자료
- 사전 추출 특징 파일(`.npy`, `.npz`, 특징 ZIP)
- 로컬 학습 체크포인트와 D 전달용 ZIP

분류기 체크포인트 12개와 D 전달용 실행 패키지는 팀 공유 드라이브에서 관리한다.
공개 저장소에는 재현에 필요한 코드, 설정, 요약 결과만 올린다.

## 7. 재현 시 주의 사항

- 새 특징은 학습 때와 동일한 추출기·crop·전처리를 사용해야 한다.
- 특징 배열의 행 순서는 대응하는 index CSV와 같아야 한다.
- Top-1 정답은 전체 `dna_UF.csv`에서 지배 혈통을 먼저 결정한 뒤 정확 매핑만
  적용한다. 비율 평가용 `truth_UF.csv`로 Top-1을 만들지 않는다.
- 주 지표에는 DNA mapping coverage, Top-1에는 Top-1 scorable coverage를
  각각 구분해 보고한다.
- HOLD는 아직 실행하지 않았으며 최종 설정 고정 후 한 번만 평가한다.
- 공식 결과는 정확 매핑과 원래 DNA 비율 포착@3을 기준으로 해석한다.
- 새 비율 지표는 DEV에서만 계산했다.
