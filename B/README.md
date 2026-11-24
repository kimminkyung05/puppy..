# B 담당: 딥 특징 추출 · RQ1 평균점 섞기(FCLS) · UF 평가셋 준비

혼합 시각 범주의 구성 비율 추정(믹스견 사진 → DNA 혈통 비율) 팀 프로젝트에서
B 담당이 만든 코드와 요약 결과를 정리한 폴더이다.

B는 팀이 함께 쓰는 CLIP · DINOv2 · DINOv3 특징을 추출하고, 그 특징 위에서
**학습 없이** 품종 평균 특징(평균점)을 섞어 혼합 비율을 추정한다(FCLS).
또 UF 믹스견 평가셋의 인덱스 · 정답 비율 · 품종 매핑 초안을 만들었다.

## 1. 실험 구성

| 특징 추출기 | 평균점 섞기 입력 | 차원 |
|---|---|---:|
| CLIP ViT-B/16 | CLS 특징 (투영 후 이미지 임베딩) | 512 |
| DINOv2 ViT-B/14 | CLS 특징 | 768 |
| DINOv3 ViT-B/16 | CLS 특징 | 768 |

- 추출 전처리(몸통 박스 크롭, 모델별 resize · crop, 레지스터 토큰)는 `configs/PREPROCESSING.md`에 있다.
- 평균점: 학습셋(Tsinghua train 61,592장, 127클래스) 특징의 품종별 평균.
- 풀이: 평가 사진 특징 x마다 `min ||Pᵀw − x||²,  w ≥ 0,  Σw = 1` (FCLS, NNLS에 합 1 제약 행을 덧붙여 푼다).
- 반복 r1 · r2 · r3마다 품종별로 학습 사진 80%를 무작위로 골라 평균점을 다시 만든다.
- 전체 실행: `추출기 3종 × 반복 3회 × L2 없음/있음 = 18회`를 믹스견과 순종에 각각 돌렸다.
- 출력 클래스는 Tsinghua Dogs 127개이고, 각 비율은 합 1이다.

## 2. 현재 평가 기준

### 순종견 (보조 평가)

- Tsinghua val을 클래스별로 반씩 나눈 dev 절반(2,540장)에서 Top-1 · Top-5를 계산한다(`scripts/make_val_split.py`, seed 1).
- 나머지 절반(test)은 쓰지 않았다.

### 믹스견

- 순종견 결과와 합치지 않고 UF 평가셋에서 따로 계산한다.
- face와 body 사진의 비율을 개체 단위로 평균한다.
- 반복 r1 · r2 · r3의 확률을 평균해 대표 결과를 만든다.
- 지표 정의는 A의 최종 정의를 복사한 `docs/METRICS_FINAL.md`를 따른다. B 적용 메모는 `docs/METRICS.md`.
- 매핑은 `A/configs/breed_map.csv`의 `status=정확`만 쓴다.
- **주 지표:** 원래 DNA 비율 포착 혈통 비율@3(예측 상위 3품종의 실제 DNA 비율 합) + coverage. 찍기 · 상한과 함께 적는다.
- **보조 지표:** 재정규화 포착@3, 재정규화 분포 겹침, Top-1 + coverage(DNA 공동 1위 중 정확 매핑 클래스 하나를 맞히면 정답).
- **dev(20마리)만 채점했다.** hold는 아직 채점하지 않았다(`docs/HOLD_STATUS.md`).

현재 값은 A의 `eval.py` 이전에 B의 채점 스크립트로 낸 **임시 채점**이다.

## 3. 주요 결과 (dev)

### 순종견 val dev 절반 (2,540장)

| 특징 | Top-1 | Top-5 |
|---|---:|---:|
| DINOv3 | 82.2% | 94.6% |
| DINOv2 | 80.8% | 93.6% |
| CLIP | 61.9% | 82.3% |

### 믹스견 dev

지표는 `docs/METRICS_FINAL.md`(정확 매핑) 기준이다. Top-1은 20마리 중 18마리가 평가 가능했으며 coverage는 90%이다. A의 dev와 같은 20마리다.

| 특징 | **주 지표** 포착@3 원래 (찍기 0.088, 상한 0.541) | 포착@3 재정규화 (찍기 0.168) | 분포 겹침 재정규화 (찍기 0.047) | Top-1 (Wilson 95% CI) |
|---|---:|---:|---:|---:|
| CLIP | 0.217 (찍기보다 높음) | 0.378 | 0.167 (찍기보다 높음) | 5/18, 27.8% (12.5–50.9%) |
| DINOv2 | 0.205 (찍기보다 높음) | 0.334 | 0.183 (찍기보다 높음) | 5/18, 27.8% (12.5–50.9%) |
| DINOv3 | 0.130 (판단 불가) | 0.222 | 0.136 (찍기보다 높음) | 3/18, 16.7% (5.8–39.2%) |

- 같은 dev 20마리에서 A 분류기는 Top-1 CLIP 5/18, DINOv2 5/18, DINOv3 4/18이었다. 사람(UF 설문)은 포착@3 0.180, Top-1 6/18이었다(`docs/DEV_RESULTS.md` 8절). DINOv3는 A가 수치상 높지만 신뢰구간이 크게 겹쳐 우위라고 단정하지 않는다.
- 순종에서 가장 좋은 DINOv3가 믹스견에서는 CLIP · DINOv2보다 낮다. A의 hold에서도 같은 방향이다.
- L2 정규화는 결과를 거의 바꾸지 않았다(순종 Top-1 ±0.2%p 이내, 믹스견 Top-1 동일).

개체 수가 적어 신뢰구간이 크게 겹치므로 수치상 순서를 성능 순위로 해석하지 않는다.
자세한 표와 해설은 `docs/DEV_RESULTS.md`, A 분류기 · C 섞기 학습과의 비교 해석은 `docs/A_VS_B_INTERPRETATION.md`에 있다.

## 4. 폴더 구조

```text
B/
├─ B_TO_D/     D(RQ2)에 넘기는 추출기 사용법과 확인 사항
├─ configs/     경로 설정, 추출기 설정, 특징 추출 전처리 명세(PREPROCESSING.md)
├─ data/        데이터 파일을 저장소에서 제외한 이유와 위치
├─ docs/        지표 정의, dev 결과 해설, A와의 비교 해석, hold 평가 상태
├─ figures/     dev 결과 그래프 (순종·믹스견 Top-1, 포착 혈통 비율, A와 비교, 헷갈린 품종)
├─ notebooks/   Colab 실행 노트북 (B_colab_run.ipynb)
├─ results/     집계된 결과 CSV (A와 같은 열 구성)
├─ scripts/     UF 평가셋 준비 · 임시 채점 · 결과 요약 · 그림
├─ src/         특징 추출, 평균점 섞기, 진행 상태, 데모, 공용 모듈
├─ tests/       모델 다운로드 없이 도는 toy 점검
├─ requirements.txt
└─ README.md
```

주요 파일:

| 파일 | 하는 일 |
|---|---|
| `src/features/extract_deep.py` | 딥 특징 추출 (5,000장 샤드, 이어서 뽑기) |
| `src/features/check_models.py` | 모델 ID · 출력 차원 · 토큰 수 점검 |
| `src/rq1/prototype_fcls.py` | 평균점 섞기 → 예측 CSV |
| `src/common/status.py` | 샤드 · 예측 · 로그 진행 상태 |
| `src/common/`(config, tsinghua, evalset, features_io) | 팀 공용 모듈. 고칠 때는 팀과 먼저 맞춘다 |
| `scripts/provisional_score_uf.py` · `provisional_score_purebred.py` | 믹스견 · 순종 임시 채점 |
| `scripts/summarize_results.py` · `plot_results.py` | `results/` 요약 CSV와 `figures/` 그림 |
| `scripts/` 나머지 | UF 평가셋 준비 (다운로드, DNA 표, 칸 나누기, 인덱스, 정답 비율, 매핑 초안, val 분할) |

## 5. 실행 환경

```bash
pip install -r requirements.txt
```

본 실행은 Colab T4 GPU에서 한다. `notebooks/B_colab_run.ipynb`를 GitHub에서 열고
셀 1의 Drive 경로를 채운 뒤 위에서부터 실행한다. 결과(특징 · 예측 · 로그)는 Drive에 바로 쓰고,
세션이 끊기면 같은 셀을 다시 실행하면 끝난 샤드 · 예측은 건너뛰고 이어서 진행한다.
진행 상태는 `python -m src.common.status`로 본다.

결과 요약과 그림은 CPU에서 돈다(B/ 안에서):

```bash
python scripts/summarize_results.py
python scripts/plot_results.py          # L2 설정은 --l2
```

모델 다운로드 없이 로컬에서 파이프라인만 점검하려면:

```powershell
python tests/make_toy_data.py "$env:TEMP\dogmix_toy"
python -m tests.test_fcls
$env:DOGMIX_PATHS = "$env:TEMP\dogmix_toy\paths.yaml"
python -m src.features.extract_deep --backbone dummy --subset train
python -m src.features.extract_deep --backbone dummy --subset eval
python -m src.rq1.prototype_fcls --backbone dummy --repeats 1 2 3 --split dev --skip-existing
python -m src.common.status --logs 5
Remove-Item Env:DOGMIX_PATHS
```

## 6. 데이터 및 체크포인트

다음 파일은 용량 · 저작권 · 개인정보 및 저장소 관리 문제로 GitHub에 포함하지 않는다(`data/README.md`).

- Tsinghua Dogs 원본 이미지와 annotation
- UF 원본 사진 · 칸 사진과 DNA 원본 자료
- 사전 추출 특징 샤드(`.npz`), 예측 · 재구성 오차 CSV, 실행 로그

특징 · 예측 · 로그는 팀 공유 드라이브의 `features/`, `preds/`, `logs/`에서 관리한다.
A · C · D는 같은 특징 파일을 쓴다. 공개 저장소에는 재현에 필요한 코드, 설정, 요약 결과만 올린다.

## 7. 재현 시 주의 사항

- 새 특징(예: D의 변형 이미지)은 `configs/PREPROCESSING.md`와 같은 추출기 · 크롭 · 전처리로 뽑아야 한다.
- 특징 배열의 행 순서는 대응하는 `index.csv`와 같아야 한다.
- `scripts/download_uf_dogs.py`는 `--help`를 받지 않는다. 인자 없이 실행하면 바로 `./uf_dogs`에 사진을 받는다.
- hold는 조건을 고정한 뒤 한 번만 채점한다. 결과를 본 뒤 설정을 바꾸지 않는다.
- 노트북은 출력을 지우고 올린다. Colab의 "GitHub에 사본 저장"은 쓰지 않는다(출력과 Drive 목록이 올라간다).
