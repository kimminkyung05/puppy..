# 믹스견 사진으로 품종 구성 비율 추정하기 (dog-mix-cv)

품종 라벨이 붙은 순종 사진만으로 학습해서, 믹스견 사진의 품종 구성 비율을 추정하고 DNA 검사 결과로 채점한 컴퓨터 비전 팀 프로젝트입니다(4명).

- **RQ1:** 이미지 특징으로 믹스견의 품종 구성을 얼마나 복원할 수 있는가? 그 정도가 특징 추출기(SIFT+BoVW, CLIP, DINOv2, DINOv3)와 풀이 방식(분류기, 평균점 섞기, 섞기 학습)에 따라 어떻게 다른가?
- **RQ2:** 그 추정은 어떤 시각 단서(영역, 질감, 색)에 기대는가?

> 지금 `develop` 브랜치에서 제출 전 정리를 하고 있습니다. 무엇이 최종 파일인지와 정리 방법은 [`report/SUBMISSION_GUIDE.md`](report/SUBMISSION_GUIDE.md)를 보세요.

## 결과 요약 (test 77마리, 1회 채점)

- 딥 특징 9개 설정의 포착 혈통 비율은 0.052~0.083이었습니다. 찍기 기준은 0.058, 상한은 0.548이고, 찍기와 구별되는 설정은 없었습니다.
- dev 20마리에서 0.13~0.26이던 점수는 test에서 0.05~0.08로 떨어졌습니다.
- RQ2에서 딥 특징은 강한 블러에 반응했고 색에는 반응하지 않았습니다. 다만 원본 점수가 찍기 수준이라 경향으로만 읽습니다.

자세한 내용은 원고 [`report/cvpr/paper_en.pdf`](report/cvpr/paper_en.pdf)(한국어판 [`paper_ko.pdf`](report/cvpr/paper_ko.pdf))에 있습니다.

## 저장소 구조

| 폴더 | 담당 | 내용 |
|---|---|---|
| [`A/`](A/) | A | 분류기 학습, 공식 지표와 품종 매핑. 최종본은 `A/A/` |
| [`B/`](B/) | B | CLIP, DINOv2, DINOv3 특징 추출, 평균점 섞기(FCLS), 세 방법 통합 채점, RQ2 최종 실행, 가계도형 데모 |
| [`C/`](C/) | C | SIFT+BoVW 특징, 섞기 학습 |
| [`D/`](D/) | D | RQ2 단서 변형 설계, SAM 3 마스크 |
| [`report/`](report/) | B | 원고(CVPR 형식, 영문과 한국어), 긴 한국어 보고서, 제출 정리 안내 |

실행 방법은 각 폴더의 README를 보세요. 네 담당은 같은 특징 파일과 같은 예측 형식을 공유합니다.

- 특징: `features/{backbone}/{subset}/shard_XXX.npz` + `index.csv`
- 예측: `preds/{split}/pred_{실행이름}.csv`. 개 단위이고, Tsinghua 클래스 127개에 대한 비율의 행 합이 1입니다.

## 흐름

```
Tsinghua Dogs (127품종, 학습) ─┐
                               ├─ 동결 특징 추출기 (SIFT+BoVW / CLIP / DINOv2 / DINOv3)
UF 믹스견 사진 (평가) ──────────┘          │
                                           ├─ RQ1: 분류기(A) · 평균점 섞기(B) · 섞기 학습(C)
                                           │       → 개 단위 비율 → 품종 매핑 → DNA와 채점
UF 사진 + SAM 3 마스크 ── 단서 변형(D) ─────┴─ RQ2: 변형 대 대조군의 추가 하락
```

## 데이터

이 저장소에는 데이터를 넣지 않습니다. 아래에서 직접 받아야 합니다.

| 데이터 | 받는 곳 | 비고 |
|---|---|---|
| Tsinghua Dogs (학습) | <https://cg.cs.tsinghua.edu.cn/ThuDogs/> | low-resolution 이미지, Low-Annotations(몸 박스), TrainValSplit. `chinese_rural_dog`, `dhole`, `African_hunting_dog`를 빼고 127클래스를 씁니다 |
| UF 믹스견 사진과 DNA 결과 (평가) | [UF Shelter Medicine, Dog Breed Identification Photo Chart](https://sheltermedicine.vetmed.ufl.edu/research/current-studies/dog-breed-identification/dog-breed-dna-and-survey-results/) | 100마리 중 믹스견 97마리를 씁니다(dev 20, test 77). 내려받기와 전처리 스크립트는 `B/scripts/`에 있습니다 |

- **UF 자료는 재배포하지 않습니다.** 원본 사진, 칸으로 나눈 사진, 변형 사진, DNA 결과, 이로부터 만든 특징은 이 저장소와 공개 링크에 올리지 않습니다.
- 팀이 만든 특징, 예측, 로그는 팀 공유 Google Drive의 `features/`, `preds/`, `logs/`에 있습니다. Drive 링크는 위 이유로 저장소에 적지 않으니, 팀 내부 채널에서 받으세요.

## 모델과 데모

| 무엇 | 링크 |
|---|---|
| CLIP ViT-B/16 | <https://huggingface.co/openai/clip-vit-base-patch16> |
| DINOv2 ViT-B/14 | <https://huggingface.co/facebook/dinov2-base> |
| DINOv3 ViT-B/16 | <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m> (접근 승인이 필요합니다) |
| SAM 3 (RQ2 마스크) | <https://arxiv.org/abs/2511.16719> (ICLR 2026) |
| 가계도형 데모 (Gradio) | <https://huggingface.co/spaces/alsrudee/cvpuppy> |

데모는 사진 한 장에 대해 세 풀이 방식의 품종 비율을 가계도 모양으로 나란히 보여 줍니다. test에서 어느 설정도 찍기를 넘지 못했으므로, 화면의 비율은 방법이 어떻게 동작하는지를 보여 주는 예시이고 DNA 검사를 대체하지 않습니다.

## 실행 환경

- 특징 추출과 학습은 Colab T4 GPU에서 했습니다(`B/notebooks/B_colab_run.ipynb` 등).
- 채점과 그림은 CPU에서 돌아갑니다.
- 패키지는 각 폴더의 `requirements.txt`를 쓰세요.
