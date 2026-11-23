# D 담당: RQ2 단서 분석 · 변형 조건 평가

## 질문

모델이 원본 사진의 어떤 시각 단서를 이용해 품종 구성 비율을 맞히는지, 그리고 그 단서가 품종군마다 다른지를 확인합니다.

## 실험 구성

A 파트의 분류기를 고정하고, 평가 이미지에 변형을 한 번에 하나씩 적용한 뒤 품종 비율 예측 변화를 비교합니다. D 파트에서는 새 분류기를 학습하지 않습니다.

- 원본 1조건 + 변형 8조건 + 대조군 8조건 = 총 17조건
- 추출기 4종 계획: CLIP, DINOv2, DINOv3, SIFT
- 분류기 seed 3개 기준으로 조건별 결과를 정리
- 핵심 비교량: 원본 점수, 변형 점수, 대조군 점수, extra_drop
- 품종군별 비교: breed_group_summary_dev.csv에 집계 결과 저장

### 조건

| 종류 | 조건 예시 | 목적 |
| --- | --- | --- |
| 원본 | original | 기준 점수 |
| 영역 | 머리만, 몸만, 배경만 | 어느 영역을 사용하는지 확인 |
| 질감 | 블러 단계 4개 | 세부 질감 의존도 확인 |
| 색 | 흑백 | 색 정보 의존도 확인 |
| 대조군 | 영역·질감 조건별 matched control | 변형 자체의 영향 보정 |

추가 하락은 다음과 같이 계산합니다.

extra_drop = (original - transformed) - (original - control)

## 현재 공개된 산출물

- results/condition_summary_dev.csv: 조건·추출기·seed별 집계 점수
- results/condition_matrix_validation_dev.csv: 기대 파일과 실제 파일의 존재 여부 점검
- results/breed_group_summary_dev.csv: 품종군별 집계표
- configs/conditions.csv: 17조건 정의
- configs/paths.example.yaml: 개인 환경용 경로 예시
- notebooks/D_RQ2_Colab.ipynb: 무료 Colab GPU 실행용 노트북
- scripts/: 조건 생성·실행·노트북 생성 코드
- docs/EXPERIMENT_STATUS.md: 입력 자산과 실행 상태 기록

## 현재 상태와 제한

현재 결과표는 확보된 평가 결과를 통합한 개발(dev) 산출물입니다. 17조건 × 3 seed 기준 기대 파일은 204개이며, 마지막 점검에서 108개가 확인되고 96개가 누락되었습니다. 따라서 이것을 68개 채점 전체 완료로 해석하면 안 됩니다.

- CLIP·DINOv2: 확보된 변형 결과를 집계표에 반영
- DINOv3·SIFT: 현재 원본 기준선만 있고 변형 특징 결과는 미완료
- SAM3: 실제 SAM3 마스크가 아니라 GrabCut 휴리스틱 fallback 결과만 일부 존재
- PSNR 메타데이터: 미확보
- 최종 테스트셋: 팀에서 별도 수집하기로 했으므로 사용하지 않음
- 품종군 비교: 개발용 집계이며 최종 품종군 정의가 확정되면 재집계 필요

## 디렉터리

```text
D/
├─ configs/
│  ├─ conditions.csv
│  └─ paths.example.yaml
├─ docs/
│  └─ EXPERIMENT_STATUS.md
├─ notebooks/
│  └─ D_RQ2_Colab.ipynb
├─ results/
│  ├─ breed_group_summary_dev.csv
│  ├─ condition_matrix_validation_dev.csv
│  └─ condition_summary_dev.csv
├─ scripts/
│  ├─ build_d_notebook.py
│  └─ run_d_experiment.py
└─ README.md
```

## 실행

Colab에서 팀 공유 Drive를 마운트한 뒤 개인 경로 설정을 확인하고 실행합니다.

```bash
pip install -r ../B/requirements.txt
python scripts/run_d_experiment.py
```

노트북과 스크립트는 데이터·체크포인트·특징 자산의 실제 위치를 자동으로 알 수 없습니다. 실행 전에 configs/paths.example.yaml을 복사해 환경에 맞는 경로를 설정해야 합니다.

## 추가로 필요한 자산

- B: CLIP·DINOv2·DINOv3 특징 추출 코드와 최종 원본/변형 특징 자산
- C: SIFT 코드북·추출 코드와 최종 SIFT 특징 자산
- A: 최종 분류기 체크포인트, eval.py, 최종 breed_map.csv와 truth_UF.csv 확인
- SAM3를 실제로 실행하려면 SAM3 설치·체크포인트·프롬프트/박스 처리 코드가 별도로 필요

## 저장소에 올리지 않는 파일

원본 이미지, Tsinghua 원자료, UF 원자료와 DNA 정답 원자료, 개인 Drive 링크, per-dog 예측 원본, 특징 배열, SIFT 코드북, 모델 체크포인트, ZIP 파일은 저장소에 포함하지 않습니다. 공개 저장소에는 코드·설정 예시·식별정보가 제거된 집계 결과만 올립니다.

## 재현성 주의

이 저장소의 결과 CSV는 개발 단계 집계표입니다. 최종 수치 보고 전에는 최종 truth/breed map, 추출기 버전, seed 목록, 마스크 종류, PSNR, 누락 조건을 다시 확인해야 합니다. 특히 현재 D 결과에는 실제 SAM3 마스크와 최종 테스트셋 평가가 포함되어 있지 않습니다.
