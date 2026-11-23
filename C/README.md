# C파트: 믹스견 품종 비율 추정 (RQ1) 및 평가 입력

## 구조
- `code/index_build.py`: 파일 경로 → image_id, 품종, 분할, 127클래스 필터
- `code/sift_extract_train.py`: SIFT BoVW 특징 추출 (train/validation, 코드북). 원본 셀의 나머지 저장 부분은 확인 후 추가
- `code/deep_mix_train.py`: 딥 특징(CLIP, DINOv2, DINOv3) 혼합 학습 v2 (KMAX=4, LR=3e-3)
- `code/ablation_purebred.py`: 순종만 학습한 대조군
- `code/make_uf_features.py`: UF 패널 평가 입력 (SIFT 특징 v5, PNG 이미지 zip, manifest)
- `code/score_dev.py`: dev 채점 (평가 지표 정의 기준, 찍기 대비 ±2SE)
- `docs/evaluation_spec.md`: 평가 지표 정의
- `docs/experiment_log_RQ1.csv`: 실험 로그 24건 (혼합 12, 순종 대조군 12)

## 실행 순서
1. `index_build.py`로 인덱스 생성 (품종 이름은 폴더의 `번호-n번호-` 접두어를 제거)
2. `sift_extract_train.py`로 코드북과 train/validation SIFT 특징 생성
3. `deep_mix_train.py`를 EXT = clip, dinov2, dinov3 순서로 실행 (혼합 학습 12회)
4. `ablation_purebred.py`로 순종 대조군 실행
5. `make_uf_features.py`로 UF 평가 입력 생성 (SIFT σ=1,3,6,12, 이미지 zip)
6. `score_dev.py`로 dev 채점 (hold는 설정 고정 후 1회만)

## 알려진 주의점
- **SIFT 블러 σ:** 이전에 만든 `sift_eval_uf_v3_all.npy`는 σ=1,2,3,4였습니다. 현재 기준은 σ=1,3,6,12이며 `make_uf_features.py`(v5)를 쓰세요.
- **KMAX:** 혼합 학습 v2는 KMAX=4입니다. 이 값은 노트북 셀에서 확인했습니다.
- **KL 미사용:** 교차엔트로피를 사용합니다(정답 비율 벡터 기준 KL과 기울기가 같음).
- **SIFT 학습 셀:** `sift_extract_train.py`는 bovw 함수까지만 있습니다. 나머지 저장 부분을 원본에서 채워 넣어야 합니다.
- **Poodle 계열:** teddy, toy_poodle, miniature_poodle, standard_poodle 4개를 한 품종으로 합산합니다.

## 저장소에 올리지 않는 것
- 원본 이미지, 어노테이션, zip 압축 파일
- 특징 파일(.npy, .npz), 체크포인트(.pt)
- 예측 CSV(개 단위 결과는 팀 드라이브에 보관)
- 데이터 경로는 팀 드라이브 `C_work`를 기준으로 한다.
