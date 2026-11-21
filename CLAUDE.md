# dog-mix-cv

혼합 시각 범주의 구성 비율 추정: 믹스견 사진 → DNA 혈통 비율 (CV 팀 프로젝트, 4명). 답변은 한국어로 한다.

- **RQ1:** 혼합 비율 복원이 특징 종류(SIFT · CLIP · DINOv2 · DINOv3)와 풀이 방식(A 분류기 · B 평균점 섞기 · C 섞기 학습)에 따라 어떻게 다른가
- **RQ2 (D):** 그 추정은 어떤 시각 단서(영역 · 질감 · 색)에 기대는가

흐름: Tsinghua Dogs 학습셋(몸통 박스 크롭)과 믹스견 평가셋(SAM 3 크롭)에서 특징을 한 번만 뽑아 팀이 공유한다 → 같은 특징에 세 가지 풀이를 적용한다 → 예측 CSV를 A의 `eval.py`가 DNA 품종 공간으로 합쳐 채점한다.

실행 방식: 코드는 VS Code + Git, 실행은 Colab T4 GPU(`notebooks/B_colab_run.ipynb`). 결과는 팀 공유 Google Drive 폴더에 쓴다.

## 담당

이 저장소의 사용자는 **담당 B**다: CLIP · DINOv2 · DINOv3 특징 추출, 평균점 섞기(FCLS), D의 변형 이미지 특징 추출, 사람 기준 집계, 2D 투영 그림, 가계도형 Gradio 데모.

| 담당 | 파일 |
|---|---|
| **B (사용자)** | `src/features/backbones.py`, `src/features/extract_deep.py`, `src/features/check_models.py`, `src/rq1/prototype_fcls.py`, `src/rq1/sanity_mix.py`, `src/common/status.py`, `src/common/check_data.py`, `src/demo/app.py`, `notebooks/B_colab_run.ipynb` |
| A | `src/rq1/classifier.py`, `src/eval/eval.py`, `eval_index.csv` |
| C | `src/features/extract_sift.py`, `src/rq1/mix_train.py` |
| D | `src/rq2/transforms.py`, 평가셋 SAM 3 크롭, 변형별 인덱스 csv |
| 공용 | `src/common/`(config, tsinghua, evalset, features_io), `configs/`, `README.md` |

## 공용 인터페이스

- **특징:** `features/{backbone}/{subset}/shard_XXX.npz` (`image_id`, `cls`, `mean`; float16) + `index.csv`
  - `index.csv`: 학습셋은 `image_id, label`, 평가셋은 `image_id, dog_id, split`
  - 딥: `cls`는 CLS 토큰(CLIP은 투영 후 이미지 임베딩), `mean`은 패치 토큰 평균. SIFT: 둘 다 BoVW 히스토그램
  - backbone: `clip`, `dinov2`, `dinov3`, `sift` (`dummy`는 점검용). subset: `train`, `val`, `eval`, D의 변형은 `eval_gray` 등
- **예측:** `preds/{split}/pred_{실행이름}.csv` (`dog_id, split, <Tsinghua 클래스>`, 행 합 1, 개 단위)
  - `{split}`은 `dev` / `test` / `all`. Tsinghua 클래스 공간으로 내고, DNA 품종으로 합치는 일은 `eval.py`가 `breed_map.csv`로 한다
  - B는 같은 폴더에 `recon_{실행이름}.csv` (`image_id, dog_id, split, recon_err`)도 낸다
- **평가셋 인덱스:** `eval_index.csv` (`image_id, dog_id, path, split(dev/test)`). A가 만들고 `path`는 D의 SAM 3 크롭
- **클래스 순서:** 학습셋 label 정렬 순서. 모든 담당이 같은 순서를 쓴다
- **실행 이름:** `RQ-추출기-방법-시드` (예: `RQ1-dinov3-proto-r1`, `RQ1-dinov3-mix-s1`)
  - B의 평균점 섞기: `RQ1-{backbone}-proto-r{반복}` 뒤에 L2면 `-l2`, 패치 평균이면 `-mean`. 변형 평가셋이면 `RQ1-{eval_subset}-{backbone}-...`
  - 특징 추출 로그: `FEAT-{backbone}-{subset}`
- **제외 클래스:** `chinese_rural_dog`, `dhole`, `African_hunting_dog` → 127클래스 (`dingo`는 `exclude_dingo`로 결정, 제외하면 126)
- **로그:** `logs/runs.jsonl`. 한 줄에 실행 하나, 실패한 실행도 남긴다
- **경로:** `configs/paths.yaml`. 환경 변수 `DOGMIX_PATHS`로 다른 파일을 바꿔 끼운다. Colab에서는 노트북이 `/content/paths_colab.yaml`을 만들어 쓰고 `configs/paths.yaml`은 고치지 않는다

## 체크포인트 규칙

Colab 세션은 언제든 끊길 수 있으므로 아래 동작을 유지한다.

- 결과(특징 · 예측 · 로그)는 Drive 경로에 바로 쓴다. `/content`에서는 입력 데이터만 읽는다.
- 특징은 5,000장 단위 샤드로 저장한다. 임시 파일(`shard_XXX.npz.tmp`)에 다 쓴 뒤 이름을 바꿔 저장하고, 같은 명령을 다시 실행하면 끝난 샤드는 건너뛴다.
- 예측은 `preds/{split}/`에 저장하고(임시 파일에 쓴 뒤 이름 바꾸기), `--skip-existing`으로 이미 있는 실행을 건너뛴다.
- 진행 상태는 `python -m src.common.status`로 본다(샤드 완료 여부, `tmp_files`, 예측 목록, 최근 로그).
- **이 동작(이어서 뽑기, 임시 파일 저장, 건너뛰기, `status.py`)을 바꾸거나 지우는 수정은 먼저 사용자에게 물어본다.**

## 코드 구조 규칙

- 실험 로직은 `.py`(`src/`)에만 둔다.
- 노트북은 환경 준비 · 스크립트 실행 · 결과 훑어보기만 한다.
- 보고서에 들어갈 그림 코드도 `.py`로 만든다.
- 노트북은 커밋 전에 출력을 지운다(`nbstripout`).

## 작업 규칙

1. 모르는 API · 모델 ID · 숫자는 추측하지 않는다. "확인 필요"라고 말하고 확인할 코드를 제시한다.
2. 공용 인터페이스(`src/common`, 파일 형식, 클래스 순서)는 사용자 확인 없이 바꾸지 않는다.
3. A · C · D 담당 파일(`classifier.py`, `mix_train.py`, `extract_sift.py`, `transforms.py`, `eval.py`)은 수정하지 않는다. 필요한 것은 담당자에게 보낼 요청 문구로 정리한다.
4. 삭제 · 덮어쓰기 · `git push`는 실행 전에 사용자에게 묻는다.
5. test 분할은 D8 전까지 쓰지 않는다. 실제 평가셋에는 `--split dev`만 쓴다.
6. 수집 이미지와 DNA 결과는 재배포하지 않는다. 저장소, 공개 링크, 외부 서비스에 올리지 않는다(Drive 폴더 링크도 저장소에 적지 않는다).

## 로컬 점검 (Windows, 모델 다운로드 없이)

`.venv`(Python 3.12, CPU용 torch)를 쓴다. 프로젝트 루트에서 PowerShell로:

```powershell
.\.venv\Scripts\Activate.ps1
python tests/make_toy_data.py "$env:TEMP\dogmix_toy"
python -m tests.test_fcls
$env:DOGMIX_PATHS = "$env:TEMP\dogmix_toy\paths.yaml"
python -m src.features.extract_deep --backbone dummy --subset train
python -m src.features.extract_deep --backbone dummy --subset eval
python -m src.rq1.prototype_fcls --backbone dummy --repeats 1 2 3 --split dev --skip-existing
python -m src.common.status --logs 5
python -m src.common.check_data
python -m src.features.check_models --backbones dummy
Remove-Item Env:DOGMIX_PATHS
```

같은 extract 명령을 한 번 더 실행하면 0장을 뽑아야 하고, 같은 prototype_fcls 명령은 `[건너뜀]`을 출력해야 한다.
toy 데이터에서는 `check_data`가 클래스 수(4개)를 "확인 필요"로 표시하는 것이 정상이다.

### 로컬의 실제 데이터

- Tsinghua 데이터는 `data/`에 풀려 있다(`low-resolution/`, `Low-Annotations/`, `TrainValSplit/TrainAndValList/`). 경로 설정은 `data/paths_local.yaml`, 결과는 `outputs/`에 쓴다. 둘 다 git에 올라가지 않는다.
- 이 노트북 PC의 CPU 추출 속도는 CLIP · DINOv2 모두 약 1.4장/초다(2026-11-21 측정, 200장). val 5,080장에 약 1시간, 학습셋 61,592장에 약 12시간이 걸리므로 전체 추출은 Colab GPU에서 한다. 로컬에서는 `--limit` 수백 장까지만 돌린다.
- 평가셋 없이 돌릴 수 있는 점검 실험: `python -m src.rq1.sanity_mix --backbone dinov2 --proto-subset val_limit200 --mix-subset val_limit200` (특징 공간 가짜 믹스. RQ1 결과가 아니라 참고값이다)
