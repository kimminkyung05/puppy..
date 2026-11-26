# 제출 정리 안내

제출물(GitHub, 팀 Drive)을 정리하는 사람이 이 파일만 보고 움직일 수 있도록 정리했습니다. 이 파일은 2026-11-26에 B가 썼습니다.

A, B, C, D가 각자 다른 방식으로 실험한 파일을 한 저장소에 합쳤기 때문에, 같은 실험의 옛 버전과 최종 버전이 섞여 있습니다. **A, C, D 폴더에서 무엇이 최종인지는 각 폴더 주인에게 물어서 정합니다**(아래 3절). 다른 사람의 폴더는 주인 확인 없이 지우거나 옮기지 않습니다.

## 1. 제출물

| 무엇 | 위치 | 상태 |
|---|---|---|
| 원고 (영문, 제출용) | `report/cvpr/paper_en.docx`, `paper_en.pdf` | 본문 8쪽 + 참고문헌. TBD 2곳과 팀 확인이 남음 |
| 원고 (한국어, 검토용) | `report/cvpr/paper_ko.docx`, `paper_ko.pdf` | 영문과 내용이 같음 |
| 원고 원본과 빌드 | `report/cvpr/paper_*.md`, `build_cvpr_docx.py`, `make_cvpr_figures.py` | md를 고친 뒤 `python build_cvpr_docx.py en`(또는 `ko`)으로 다시 빌드 |
| 긴 한국어 보고서 | `report/ko/*.md` | CVPR판과 숫자가 같음. 제출 형식이 CVPR판이면 쓰지 않음 |
| 코드 | GitHub `main`의 `A/`, `B/`, `C/`, `D/` | 각 담당이 올림 |
| 특징, 예측, 로그 | 팀 Drive의 `features/`, `preds/`, `logs/` | GitHub에는 올리지 않음 |

원고의 숫자를 고치려면 md만 고치고 다시 빌드합니다. docx를 직접 고치면 다음 빌드 때 덮어써집니다.

## 2. 원고의 표와 그림은 어디서 왔나

| 원고 | 근거 파일 | 만든 스크립트 |
|---|---|---|
| 표 2 (RQ1 test, 마지막 열 dev) | `B/results/hold_rq1_comparison.csv`, `dev_rq1_comparison.csv`, `hold_human_baseline.csv`, `dev_human_baseline.csv` | `B/scripts/collect_rq1_preds.py` → `score_human_uf.py` → `compare_rq1_dev.py` (`--split dev` 또는 `hold`) |
| 표 3 (test 방법 짝 비교) | `B/results/hold_rq1_paired.csv` | 위와 같음 |
| 표 4 (순종 Top-1) | `B/results/purebred_top1_same_set.csv` | 위와 같음 |
| 표 5 (RQ2 test) | `outputs/final_rq2/hold/extra_drop_hold.csv`, `scores_hold.csv` (로컬 전용, 개체별 값 포함) | `B/scripts/final_rq2.py --split hold` |
| 그림 1 (흐름도) | 데이터 없음 | `report/cvpr/make_cvpr_figures.py` |
| 그림 2 (순종 대 믹스견, dev→test) | 표 2, 표 4와 같은 파일 | `report/cvpr/make_cvpr_figures.py` |
| 그림 3 (실패 단계, dev) | `B/results/dev_failure_stages.csv` | `B/scripts/failure_stage_dev.py` |

이 표들의 입력으로 각 담당이 넘긴 것은 다음과 같습니다. 어느 파일이 최종인지는 3절에서 주인에게 확인합니다.

- **A:** 분류기 예측(시드 1~3)과 선형 분류기 가중치. RQ2의 딥 특징 3종에도 이 분류기를 썼습니다.
- **C:** 섞기 학습 예측(시드 1~3). RQ1의 SIFT는 v2 특징, RQ2의 SIFT는 v3 특징과 v3 모델을 썼습니다.
- **D:** SAM 3 마스크(몸 "dog", 머리 "dog head"). test 1마리는 머리 상자를 손으로 그렸습니다.
- **B:** 딥 특징 3종 추출, 평균점 섞기, 모든 채점, RQ2 최종 실행.

hold(test)는 2026-11-26에 한 번 채점했습니다. 다시 채점하지 않습니다.

## 3. 각 폴더 주인에게 물을 것

모든 담당에게 공통으로 묻습니다.

1. 내 폴더에서 **원고 숫자의 근거인 최종 파일**은 무엇인가?
2. 옛 버전(이전 매핑, 이전 경로, 예비 실험)은 무엇이고, 지울지 `old/`로 옮길지?
3. GitHub에 남길 것과 Drive에만 둘 것은 무엇인가?
4. 내 폴더에 UF 사진, 개체별 DNA 값, 개 ID가 붙은 결과가 있는가? 있으면 GitHub에서 빼야 합니다.

담당별로 더 확인할 것:

`develop` 브랜치에는 B가 zip이나 메신저로 받아 쓴 A·C·D 파일 복사본도 함께 올렸습니다. 주인이 가진 최종본과 다를 수 있으니, 아래 파일은 주인에게 최종본인지 확인한 뒤 남기거나 지웁니다.

- **A:** 최종 결과는 `A/A/`이고, 바깥 `A/`의 11/23 hold 채점은 옛 기준이 맞는지. B가 올린 복사본은 `A/A/A_to_D_handoff/`, `A/A/docs/A_TRAINED_MODELS_SUMMARY.md`입니다.
- **B:** 코드와 결과는 `main`에도 올라가 있습니다. 루트의 `src/`, `scripts/` 등은 git 밖의 옛 복사본이라 제출하지 않습니다.
- **C:** RQ1 공식은 SIFT v2, RQ2는 v3라는 구분이 맞는지. B가 올린 복사본은 `C/C_add/`, `C/README.txt`, `C/methodology_C.md`, `C/sift_analysis.md`, `C/index_eval_uf.csv`, `C/index_local_v2.csv`입니다.
- **D:** 최종은 `D/v2/`인지. 첫 dev 실행(GrabCut 대체 마스크) 결과는 원고에서 B의 최종 실행으로 대체했는데, 이 결과를 어떻게 남길지. `D/v2/`는 B가 올린 복사본입니다.

**개 ID가 붙은 개체별 파일:** 저장소에는 개 ID가 붙은 개체별 파일이 이미 있습니다. 예를 들어 `C/results/results/metrics/perdog2_*`, `C/results/results/pred/`, `A/A/configs/split.csv`, 그리고 이번에 올린 `D/v2/CLIP_DINOv2_condition_scores.csv`입니다. DNA 원본은 아니지만 개체별 점수가 들어 있으므로, 저장소를 공개(public)하기 전에 남길지 정해야 합니다.

## 4. 올리면 안 되는 것

- `data/`, `outputs/` 전체: UF 사진, 변형 사진, 특징, 개체별 예측과 점수
- UF 사진, 칸으로 나눈 사진, DNA 결과, 개 ID가 붙은 개체별 표
- 루트의 개 사진(`*.jpg`), `EVALUATION_POLICY.md`(개 ID와 DNA 예시), `.mcp.json`, `CLAUDE.md`, `.venv/`
- 모델 가중치(`*.pt`, `*.pth`), 특징 배열(`*.npz`, `*.npy`), `*.zip`: `.gitignore`가 막고 있으니 `-f`로 강제로 넣지 않습니다.
- Drive 폴더 링크: 저장소의 어떤 파일에도 적지 않습니다.

## 5. 남은 확인 사항 (원고)

- **D:** RQ2 결과를 B의 최종 test 실행 값으로 넣은 것, 품종군 분석을 뺀 것
- **A, D:** "공식 채점 전에 팀원 두 명이 test를 먼저 채점했으나 설정은 바꾸지 않았다"는 문장
- **C:** RQ2의 SIFT에 v3 특징과 C 모델을 쓴 것
- **팀:** 대표 설정(데모는 DINOv2), 저장소 공개 여부와 링크
- **TBD 2곳:** UF 사용 허락 회신, 공개 저장소 링크

영문 원고는 본문이 정확히 8쪽이라, 문장을 더하면 다른 곳을 줄여야 할 수 있습니다.
