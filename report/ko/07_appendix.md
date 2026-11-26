## APPENDIX

### 부록 A. Tsinghua Dogs 클래스 정리

- 클래스별 사진 수는 공개 분할 파일(train.lst, validation.lst)에서 세었다(표 A2, 130클래스, train 65,228장, val 5,200장).
- 제외하거나 따로 처리한 클래스는 표 A1과 같다.

**표 A1. 제외하거나 따로 처리한 Tsinghua Dogs 클래스** (사진 수는 train 분할 기준)

| 클래스 | train 사진 수 | 처리 | 이유 |
|---|---|---|---|
| chinese_rural_dog | 3,296 | 제외 | 여러 형태의 개를 묶은 범주 |
| dhole, African_hunting_dog | 169, 171 | 제외 | 집개가 아닌 야생 갯과 |
| dingo | 177 | 포함 | 학습 127클래스에 포함한다. DNA 표에 dingo가 없으므로 채점에는 대응하지 않는다 |
| teddy, toy_poodle, miniature_poodle, standard_poodle | 7,409, 2,885, 160, 276 | 채점 시 Poodle로 합침 | teddy는 푸들의 미용 형태 `[확인 필요]` |

- 구현 주의: `Fila Braziliero` 클래스는 폴더 이름에 공백이 있다. 분할 파일에는 BOM이 있다. 주석 파일의 몸 박스 태그는 `bodybndbox`다.

**표 A2. Tsinghua Dogs 클래스별 사진 수** (공개 분할 기준, *는 제외한 클래스)

| 클래스 | train | val | 클래스 | train | val |
|---|---|---|---|---|---|
| affenpinscher | 160 | 40 | Irish_water_spaniel | 179 | 40 |
| Afghan_hound | 182 | 40 | Irish_wolfhound | 184 | 40 |
| African_hunting_dog* | 171 | 40 | Italian_greyhound | 198 | 40 |
| Airedale | 160 | 40 | Japanese_spaniel | 209 | 40 |
| American_Staffordshire_terrier | 176 | 40 | Japanese_Spitzes | 213 | 40 |
| Appenzeller | 194 | 40 | keeshond | 183 | 40 |
| Australian_Shepherd | 205 | 40 | kelpie | 169 | 40 |
| Australian_terrier | 162 | 40 | Kerry_blue_terrier | 189 | 40 |
| basenji | 203 | 40 | komondor | 278 | 40 |
| basset | 201 | 40 | kuvasz | 183 | 40 |
| beagle | 236 | 40 | Labrador_retriever | 3,540 | 40 |
| Bedlington_terrier | 174 | 40 | Lakeland_terrier | 176 | 40 |
| Bernese_mountain_dog | 171 | 40 | Leonberg | 174 | 40 |
| Bichon_Frise | 3,043 | 40 | Lhasa | 175 | 40 |
| black_and_tan_coonhound | 185 | 40 | malamute | 1,284 | 40 |
| Black_sable | 218 | 40 | malinois | 193 | 40 |
| Blenheim_spaniel | 167 | 40 | Maltese_dog | 209 | 40 |
| bloodhound | 167 | 40 | Mexican_hairless | 180 | 40 |
| bluetick | 177 | 40 | miniature_pinscher | 521 | 40 |
| Border_collie | 2,554 | 40 | miniature_poodle | 160 | 40 |
| Border_terrier | 166 | 40 | miniature_schnauzer | 2,302 | 40 |
| borzoi | 186 | 40 | Newfoundland | 175 | 40 |
| Boston_bull | 190 | 40 | Norfolk_terrier | 163 | 40 |
| Bouvier_des_Flandres | 193 | 40 | Norwegian_elkhound | 177 | 40 |
| boxer | 185 | 40 | Norwich_terrier | 163 | 40 |
| Brabancon_griffo | 169 | 40 | Old_English_sheepdog | 217 | 40 |
| briard | 190 | 40 | otterhound | 177 | 40 |
| Brittany_spaniel | 192 | 40 | papillon | 766 | 40 |
| bull_mastiff | 191 | 40 | Pekinese | 440 | 40 |
| cairn | 171 | 40 | Pembroke | 165 | 40 |
| Cane_Carso | 213 | 40 | Pomeranian | 1,896 | 40 |
| Cardigan | 2,869 | 40 | pug | 758 | 40 |
| Chesapeake_Bay_retriever | 175 | 40 | redbone | 167 | 40 |
| Chihuahua | 380 | 40 | Rhodesian_ridgeback | 190 | 40 |
| Chinese_Crested_Dog | 194 | 40 | Rottweiler | 184 | 40 |
| chinese_rural_dog* | 3,296 | 40 | Saint_Bernard | 171 | 40 |
| chow | 289 | 40 | Saluki | 166 | 40 |
| clumber | 195 | 40 | Samoyed | 2,152 | 40 |
| cocker_spaniel | 246 | 40 | schipperke | 161 | 40 |
| collie | 187 | 40 | Scotch_terrier | 166 | 40 |
| curly_coated_retriever | 162 | 40 | Scottish_deerhound | 184 | 40 |
| Dandie_Dinmont | 171 | 40 | Sealyham_terrier | 171 | 40 |
| dhole* | 169 | 40 | Shetland_sheepdog | 234 | 40 |
| dingo | 177 | 40 | Shiba_Dog | 1,003 | 40 |
| Doberman | 169 | 40 | Shih_Tzu | 321 | 40 |
| English_foxhound | 183 | 40 | Siberian_husky | 1,120 | 40 |
| English_setter | 163 | 40 | silky_terrier | 191 | 40 |
| English_springer | 236 | 40 | soft_coated_wheaten_terrier | 195 | 40 |
| EntleBucher | 181 | 40 | Staffordshire_bullterrier | 163 | 40 |
| Eskimo_dog | 192 | 40 | standard_poodle | 276 | 40 |
| Fila Braziliero | 180 | 40 | standard_schnauzer | 180 | 40 |
| flat_coated_retriever | 166 | 40 | Sussex_spaniel | 177 | 40 |
| French_bulldog | 1,081 | 40 | teddy | 7,409 | 40 |
| German_shepherd | 171 | 40 | Tibetan_mastiff | 165 | 40 |
| German_short_haired_pointer | 171 | 40 | Tibetan_terrier | 163 | 40 |
| giant_schnauzer | 192 | 40 | toy_poodle | 2,885 | 40 |
| golden_retriever | 5,315 | 40 | toy_terrier | 197 | 40 |
| Gordon_setter | 188 | 40 | vizsla | 170 | 40 |
| Great_Dane | 166 | 40 | Walker_hound | 176 | 40 |
| Great_Pyrenees | 185 | 40 | Weimaraner | 195 | 40 |
| Greater_Swiss_Mountain_dog | 197 | 40 | Welsh_springer_spaniel | 162 | 40 |
| groenendael | 187 | 40 | West_Highland_white_terrier | 169 | 40 |
| Ibizan_hound | 170 | 40 | whippet | 194 | 40 |
| Irish_setter | 167 | 40 | wire_haired_fox_terrier | 186 | 40 |
| Irish_terrier | 167 | 40 | Yorkshire_terrier | 300 | 40 |

### 부록 B. 품종 매핑 규칙

<!--
작업 메모 (docx에는 나오지 않는다). 출처: A 최종 A/A/configs/breed_map.csv (138행: 정확 89, 유사 6, 없음 43), classes_127.csv, A/A/docs/METRICS_FINAL.md 2절 (2026-11-24 확인). 정확 매핑에 쓰인 Tsinghua 클래스는 85개
-->

DNA 검사 결과의 품종명과 Tsinghua Dogs의 클래스는 품종 매핑 표로 잇는다. 매핑 표는 DNA 표에 나온 품종명 138개(띄어쓰기, 오타 같은 표기 변형을 각각 한 줄로 셈)를 다음 세 상태로 나눈다(표 B1).

**규칙.**

1. DNA 표의 품종명 하나를 Tsinghua 클래스 0개 이상에 대응시키고, 대응마다 "정확", "유사", "없음" 가운데 하나를 붙인다.
2. 채점에는 "정확" 대응만 쓴다. 이름만 다르거나 표기가 다른 같은 품종(예: Belgian Sheepdog과 groenendael, Boston Terrier와 Boston_bull)은 "정확"으로 둔다.
3. 여러 클래스가 한 품종에 대응하면 예측 비율을 더한다. 푸들 계열 4개 클래스(teddy, toy_poodle, miniature_poodle, standard_poodle)는 Poodle과 Miniature Poodle 모두에 묶는다.
4. 가깝지만 다른 품종을 잇는 "유사" 대응(표 B2)은 공식 결과에 쓰지 않는다. 이 품종들은 "없음"과 마찬가지로 학습 품종 밖으로 둔다.
5. 대응하는 클래스가 없는 품종("없음")은 학습 품종 밖으로 두고 커버리지에서 뺀다. 예를 들어 학습 클래스에 닥스훈트가 없으므로 닥스훈트 4종은 모두 "없음"이다.
6. DNA 비율은 퍼센트를 100으로 나눈 값을 그대로 쓰며, 개체의 비율 합이 100%가 아니어도 늘리거나 줄이지 않는다.

**표 B1. 품종 매핑 표의 구성**

| 상태 | 품종명 수 | 채점에서의 처리 |
|---|---|---|
| 정확 | 89 | 대응 클래스의 예측 비율을 합해 그 품종의 예측으로 씀 (Tsinghua 클래스 85개 사용) |
| 유사 | 6 | 공식 결과에 쓰지 않음 (학습 품종 밖으로 둠) |
| 없음 | 43 | 학습 품종 밖 |
| 합계 | 138 | |

**표 B2. 공식 결과에서 뺀 유사 대응**

| DNA 품종명 | 후보 Tsinghua 클래스 | 빼는 이유 |
|---|---|---|
| American Eskimo | Japanese_Spitzes | 외형은 흰 스피츠로 비슷하지만 다른 품종 |
| American Foxhound | English_foxhound, Walker_hound | 같은 폭스하운드 계열이지만 별도 품종 |
| English Cocker Spaniel | cocker_spaniel | 학습 클래스가 American과 English 코커를 구분하지 않음 |
| English Toy Terrier | toy_terrier | toy_terrier 클래스에 여러 토이 테리어형이 섞여 있음 |
| Fox Terrier | wire_haired_fox_terrier | DNA 표는 Smooth와 Wire를 구분하지 않고, 학습 클래스는 Wire만 있음 |
| Toy Fox Terrier | toy_terrier | toy_terrier 클래스에 여러 토이 테리어형이 섞여 있음 |

**남은 확인 사항.** `[확인 필요: 담당 A]`

- teddy를 푸들 계열로 묶는 것은 테디컷 푸들로 보이는 학습 사진에 근거했다. 원 데이터셋의 정의는 확인하지 못했다.
- Black_sable(검은 셰퍼드형)과 German Shepherd, Eskimo_dog(허스키형 썰매개)과 Siberian Husky는 현재 묶지 않았다.
- DNA 표의 표기 문제: 비율 표기가 깨진 항목 2건과 비율 합이 100%를 넘는 개체 2마리의 처리.

### 부록 C. 실행 이름과 로그 형식

- 실행 이름: `RQ-추출기-방법-시드`. 예: `RQ1-dinov3-proto-r1`, `RQ1-dinov3-mix-s1`. 특징 추출은 `FEAT-추출기-데이터`.
- 특징: 추출기와 데이터별 폴더에 5,000장 단위 묶음과 인덱스 파일.
- 예측: 분할별 폴더에 실행마다 CSV 하나(개체 단위, Tsinghua 클래스 공간, 행 합 1).
- 로그: `logs/runs.jsonl`, 한 줄에 실행 하나.

### 부록 D. 재현 절차

실행 순서는 다음과 같다. 명령은 저장소의 `B/` 폴더 안에서 실행한다.

1. **환경 준비.** `pip install -r requirements.txt`. 특징 추출과 학습은 Colab T4 GPU에서 `notebooks/B_colab_run.ipynb`를 열어 위에서부터 실행한다. 경로는 `configs/paths.yaml`에 두고, 환경 변수 `DOGMIX_PATHS`로 다른 경로 파일을 쓸 수 있다.
2. **데이터 점검.** `python -m src.common.check_data`로 학습셋과 평가셋의 클래스 수, 분할, 경로를 점검한다. UF 평가셋은 `scripts/`의 내려받기, 칸 나누기, 정답 비율 스크립트로 준비한다.
3. **특징 추출.** `python -m src.features.extract_deep --backbone {clip,dinov2,dinov3} --subset {train,val,eval}`. 같은 명령을 다시 실행하면 끝난 묶음은 건너뛰며, 진행 상태는 `python -m src.common.status`로 본다. SIFT+BoVW 특징은 담당 C의 스크립트로 만든다.
4. **방법별 실행.** 분류기(담당 A)와 섞기 학습(담당 C)은 각 담당의 스크립트로 시드 1~3을 실행한다. 평균점 섞기는 `python -m src.rq1.prototype_fcls --backbone {…} --repeats 1 2 3 --split {dev,all} --skip-existing`.
5. **채점.** `python scripts/collect_rq1_preds.py --split {dev,hold}`로 세 방법의 예측을 같은 형식으로 모은 뒤, `python scripts/score_human_uf.py --split …`와 `python scripts/compare_rq1_dev.py --split …`로 표 IV, V와 IV-T, V-T의 값을 계산한다. hold(test)는 설정을 고정한 뒤 한 번만 채점한다.
6. **RQ2.** `python scripts/final_rq2.py --split {dev,hold}`로 변형 사진, 특징, 개체별 점수, 추가 하락을 만든다. 중간 결과는 임시 파일에 쓴 뒤 이름을 바꿔 저장하므로, 끊기면 같은 명령으로 이어서 실행한다.

딥 모델의 공개 가중치는 `openai/clip-vit-base-patch16`, `facebook/dinov2-base`, `facebook/dinov3-vitb16-pretrain-lvd1689m`(Hugging Face)이다. 특징은 추출기와 데이터별로 5,000장 단위 묶음으로 나눠 저장하며, 실행이 중단되면 끝난 묶음은 건너뛰고 이어서 추출한다. 실험 로직은 저장소의 `.py` 파일에만 두고 노트북은 실행에만 쓴다. 실행마다 실행 이름, 날짜, 담당, 커밋, 추출기, 설정, 시드, 데이터 버전, 분할, 지표를 `logs/runs.jsonl`에 한 줄로 남기고, 실패한 실행도 지우지 않는다.

UF 자료(사진, 칸으로 나눈 사진, 변형 사진, DNA 결과, 이로부터 만든 특징과 개체별 예측)는 재배포하지 않으며 GitHub 저장소에도 올리지 않는다. 재현하려면 UF 페이지에서 직접 내려받아야 하고, 저장소에는 내려받기와 전처리 스크립트만 둔다.
