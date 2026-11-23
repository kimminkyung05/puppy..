# D 담당: RQ2 단서 분석 · 변형 조건 평가

혼합 시각 범주의 구성 비율 추정 프로젝트에서 D가 담당하는 단서 분석 실험이다. A의 분류기를 고정하고, 원본 평가 사진에 영역·질감·색 변형을 적용해 품종 비율 예측이 얼마나 변하는지 비교한다.

- 원본 1조건 + 변형 8조건 + 대조군 8조건 = 총 17조건
- 추출기 4종 계획: CLIP · DINOv2 · DINOv3 · SIFT
- 분류기 seed 3개를 기준으로 조건별 점수와 추가 하락을 집계
- 최종 테스트셋은 팀이 별도로 수집하므로 D 개발 결과에 포함하지 않음

## 질문

모델이 원본 사진의 어떤 시각 단서를 이용해 품종 구성 비율을 맞히는지, 그리고 그 단서가 품종군마다 다른지를 확인한다.

변형 점수만 비교하지 않고 대응 대조군을 함께 사용한다.

extra_drop = (original - transformed) - (original - control)

## 폴더 구조
```text
D/
├─ README.md
├─ notebooks/
│  └─ D_RQ2_Colab.ipynb
├─ scripts/
│  ├─ build_d_notebook.py
│  ├─ prepare_d_inputs.py
│  ├─ run_d_experiment.py
│  ├─ validate_condition_matrix.py
│  └─ summarize_extra_drop.py
├─ configs/
│  ├─ conditions.csv
│  ├─ conditions.yaml
│  ├─ breed_groups.csv
│  └─ paths.example.yaml
├─ docs/
│  ├─ EXPERIMENT_POLICY.md
│  ├─ ASSET_REQUIREMENTS.md
│  ├─ D_STATUS.md
│  └─ EXPERIMENT_STATUS.md
└─ results/
   ├─ condition_summary_dev.csv
   ├─ condition_matrix_validation_dev.csv
   ├─ breed_group_summary_dev.csv
   └─ plots/
       └─ README.md
```

`[공용]` 표시가 붙은 A·B·C 산출물은 담당자와 버전을 먼저 맞춘 뒤 사용한다.

## 공용 인터페이스

### 입력

- `truth_UF.csv`: UF 평가셋의 DNA 기준 정답 비율. A에게 받아 비공개 Drive에서 사용한다.
- `breed_map.csv`: DNA 품종을 Tsinghua 127개 클래스에 연결한 최종 매핑. A 공유 Drive의 최종본을 사용한다.
- A의 최종 분류기 checkpoint와 `eval.py`
- B의 CLIP·DINOv2·DINOv3 특징 산출물
- C의 SIFT 특징·BoVW 코드북 산출물
- 평가 이미지와 마스크는 비공개 Drive에서 관리한다.

### 조건 정의

`configs/conditions.csv`와 `configs/conditions.yaml`은 같은 17조건을 표현한다.

| 종류 | 조건 | 목적 |
| --- | --- | --- |
| 원본 | `original` | 기준 점수 |
| 영역 | `area_head`, `area_body`, `area_background` | 머리·몸·배경 단서 비교 |
| 질감 | `blur_01`, `blur_03`, `blur_06`, `blur_12` | 세부 질감 의존도 비교 |
| 색 | `gray` | 색 정보 의존도 비교 |
| 대조군 | `control_*` | 변형 자체의 영향 보정 |

### 결과

- `condition_summary_<split>.csv`: 조건·추출기·seed별 집계 점수
- `condition_matrix_validation_<split>.csv`: 기대 파일과 실제 파일의 존재 여부
- `breed_group_summary_<split>.csv`: provisional 품종군별 집계
- `extra_drop_detail_<split>.csv`: 개 단위 변형·대조군 비교
- `extra_drop_summary_<split>.csv`: 조건별 추가 하락 요약
- `results/plots/`: 전체 실행이 완료되면 heatmap 등 그림 저장

결과 CSV에는 원본 이미지와 DNA 원자료를 넣지 않는다.

## 실행

### Colab 본 실행

`notebooks/D_RQ2_Colab.ipynb`를 Colab에서 열고 팀 공유 Drive를 마운트한 뒤 경로를 설정한다. 실제 SAM3·특징 추출기·분류기 자산이 없는 상태에서는 해당 조건을 건너뛰거나 fallback 상태로 기록한다.

### 입력 준비

개인 경로를 인자로 넘겨 비공개 개발 입력을 만든다.

```bash
python scripts/prepare_d_inputs.py --data-root <private-data-root> --received-dir <private-received-dir> --output <private-output>
```

### 조건 매트릭스 점검

```bash
python scripts/validate_condition_matrix.py --conditions configs/conditions.csv --output-root <private-prediction-dir> --output <private-output>/condition_matrix_validation_dev.csv
```

### extra_drop 재집계

```bash
python scripts/summarize_extra_drop.py --detail <private-output>/detail_scores_dev.csv --output-dir <private-output> --split dev
```

같은 입력과 출력이 이미 있으면 결과를 덮어쓰기 전에 Drive의 이전 산출물을 확인한다.

## 현재 상태

- CLIP·DINOv2: 확보된 변형 결과를 개발 집계표에 반영
- DINOv3·SIFT: 원본 기준선은 확인했지만 변형 특징 결과는 미완료
- SAM3: 실제 SAM3 마스크가 아니라 GrabCut 휴리스틱 fallback 결과만 일부 존재
- PSNR 메타데이터: 미확보
- 기대 매트릭스: 17조건 × 4추출기 × 3 seed = 204개
- 마지막 점검: 108개 확인, 96개 누락
- 최종 테스트셋: 별도 수집 예정

따라서 현재 CSV는 개발 단계 산출물이며, 68번 전체 채점 완료 결과로 해석하지 않는다.

## 자산 상태

- 확보 완료: `truth_UF.csv`, A 공유 Drive의 최종 `breed_map.csv`
- 추가 확인 필요: A의 최종 분류기 checkpoint와 `eval.py` 버전
- B/C에서 추가 필요: DINOv3·SIFT 변형 특징 및 추출 설정
- SAM 담당에서 추가 필요: SAM3 checkpoint와 실제 마스크

## 올리지 않는 것

수집 이미지, Tsinghua 원자료, UF 원자료, DNA 정답 원자료, 개인 Drive 링크, per-dog 정답·예측, 특징 배열, SIFT 코드북, 모델 checkpoint와 ZIP 파일은 저장소에 올리지 않는다.
공개 저장소에는 실행 코드, 설정 예시, 문서와 식별정보를 제거한 집계 결과만 둔다.

## 재현성 주의

최종 수치를 보고하기 전에 truth/breed map 버전, 추출기 버전, seed 목록, 마스크 종류, PSNR, 누락 조건을 다시 확인한다. 변형 실험이 완전히 채워지기 전에는 품종군 결과를 provisional로 표시한다.
