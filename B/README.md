# B 담당: 딥 특징 추출 · 평균점 섞기(FCLS) · UF 평가셋 준비

혼합 시각 범주의 구성 비율 추정(믹스견 사진 → DNA 혈통 비율) 팀 프로젝트에서 B 담당이 만든 코드다.
이 폴더 하나만으로 실행된다. 명령은 모두 `B/` 안에서 실행한다.

- CLIP · DINOv2 · DINOv3 특징 추출 (Tsinghua Dogs 학습셋 · val, 평가셋)
- 평균점 섞기(FCLS): 품종별 평균 특징을 비음수 · 합 1로 섞어 혼합 비율 추정
- UF 평가셋(믹스견 100마리, 얼굴 · 전신) 인덱스 · 정답 비율 · 품종 매핑 초안
- 임시 채점(믹스견 Top-1 · Coverage, 순종 Top-1 · Top-5). 정식 채점은 A의 `eval.py`가 한다

## 폴더 구조

```
B/
├── requirements.txt
├── configs/
│   ├── paths.yaml              # 데이터 · 출력 경로 (환경 변수 DOGMIX_PATHS로 다른 파일을 바꿔 끼움)
│   └── backbones.yaml          # 추출기 HF 모델 ID, 배치 크기, 샤드 크기
├── src/
│   ├── common/                 # 공용 모듈 (실행에 필요해서 함께 넣음)
│   │   ├── config.py           #   설정 읽기, 실행 로그(runs.jsonl)       [공용]
│   │   ├── tsinghua.py         #   분할 파일, 제외 클래스, 몸통 박스 크롭  [공용]
│   │   ├── evalset.py          #   평가셋 인덱스 읽기                     [공용]
│   │   ├── features_io.py      #   특징 샤드 저장 · 읽기                  [공용]
│   │   ├── status.py           #   진행 상태 (샤드 · 예측 · 로그)
│   │   └── check_data.py       #   데이터 점검 (경로 · 몸통 박스 · 클래스 수)
│   ├── features/
│   │   ├── backbones.py        #   CLIP · DINOv2 · DINOv3 (+ dummy 점검용)
│   │   ├── extract_deep.py     #   딥 특징 추출 (5,000장 샤드, 이어서 뽑기)
│   │   └── check_models.py     #   모델 ID · 출력 차원 점검
│   ├── rq1/
│   │   ├── prototype_fcls.py   #   평균점 섞기 → 예측 CSV
│   │   └── sanity_mix.py       #   특징 공간 가짜 믹스 점검 (참고값)
│   └── demo/app.py             #   가계도형 Gradio 데모
├── scripts/                    # UF 평가셋 준비 · 임시 채점
│   ├── download_uf_dogs.py     #   UF 공개 페이지 저장 (사진은 저장소에 올리지 않음)
│   ├── parse_uf_dna.py         #   페이지 → DNA 표 · 개 표
│   ├── split_uf_panels.py      #   사진을 얼굴 / 전신 칸으로 나눔
│   ├── make_uf_eval_index.py   #   평가셋 인덱스 (공용 형식)
│   ├── build_uf_truth.py       #   개별 정답 비율 (truth_UF.csv)
│   ├── draft_uf_breed_map.py   #   DNA 품종 → Tsinghua 클래스 매핑 초안
│   ├── make_val_split.py       #   순종 보조 평가용 val dev/test 분할
│   ├── provisional_score_uf.py       # 믹스견 임시 채점
│   └── provisional_score_purebred.py # 순종 임시 채점
├── tests/                      # 모델 다운로드 없이 도는 toy 점검
└── notebooks/B_colab_run.ipynb # Colab T4 실행 노트북 (환경 준비 · 실행 · 결과 확인만)
```

`[공용]` 표시 파일은 팀 공용 인터페이스다. 고칠 때는 팀과 먼저 맞춘다.

## 공용 인터페이스

- **특징:** `features/{backbone}/{subset}/shard_XXX.npz` (`image_id`, `cls`, `mean`; float16) + `index.csv`
- **예측:** `preds/{split}/pred_{실행이름}.csv` (`dog_id, split, <Tsinghua 클래스 127개>`, 행 합 1, 개 단위)
  - 재구성 오차: 같은 폴더의 `recon_{실행이름}.csv` (`image_id, dog_id, split, recon_err`)
- **실행 이름:** `RQ1-{backbone}-proto-r{반복}`, L2 정규화면 `-l2`, 패치 평균이면 `-mean`. 변형 평가셋이면 `RQ1-{eval_subset}-{backbone}-...`
- **제외 클래스:** `chinese_rural_dog`, `dhole`, `African_hunting_dog` → 127클래스
- **로그:** `logs/runs.jsonl` (실패한 실행도 남김)

## 실행

### Colab (본 실행)

`notebooks/B_colab_run.ipynb`를 Colab에서 열고 런타임을 T4 GPU로 바꾼 뒤 셀 1의 저장소 주소와 Drive 경로를 채운다.
결과(특징 · 예측 · 로그)는 Drive에 바로 쓰고, 세션이 끊기면 같은 셀을 다시 실행하면 이어서 진행한다(끝난 샤드 · 예측은 건너뜀).
진행 상태는 `python -m src.common.status`로 본다.

### 로컬 점검 (모델 다운로드 없이)

```powershell
pip install -r requirements.txt
python tests/make_toy_data.py "$env:TEMP\dogmix_toy"
python -m tests.test_fcls
$env:DOGMIX_PATHS = "$env:TEMP\dogmix_toy\paths.yaml"
python -m src.features.extract_deep --backbone dummy --subset train
python -m src.features.extract_deep --backbone dummy --subset eval
python -m src.rq1.prototype_fcls --backbone dummy --repeats 1 2 3 --split dev --skip-existing
python -m src.common.status --logs 5
Remove-Item Env:DOGMIX_PATHS
```

같은 extract 명령을 다시 실행하면 0장을 뽑고, 같은 prototype_fcls 명령은 `[건너뜀]`을 출력해야 한다.

## 올리지 않는 것

수집 이미지, DNA 결과, 특징 · 예측 파일, Drive 링크는 저장소에 넣지 않는다(`.gitignore`).
