# B → D 전달 사항 (RQ2 특징 추출)

D의 17조건 사진에서 CLIP · DINOv2 · DINOv3 특징을 **A 분류기가 학습한 특징과 같은 방식**으로 뽑기 위한 자료다.
B는 마스크(SAM)와 변형 사진을 만들지 않으므로 특징 파일이 아니라 **추출기 코드**를 넘긴다. 실행은 D가 한다.

```text
B_TO_D/
├─ README.md          이 문서
├─ d_cells.py         D 노트북 · run_d_experiment.py 에 붙여 넣을 코드 조각
├─ code/              추출기 코드 복사본 (원본과 같은 폴더 구조, 이 폴더만으로 실행됨)
│  ├─ configs/
│  │  ├─ PREPROCESSING.md   추출 설정 명세 (모델 ID, 전처리, fp16, cls 정의)
│  │  └─ backbones.yaml     모델 ID · fp16 설정 (추출기가 읽음)
│  └─ src/
│     ├─ common/config.py   configs/*.yaml 읽기
│     └─ features/
│        ├─ backbones.py    모델 로드 · 전처리 · cls/mean 계산 (B 공유 특징과 같은 코드)
│        └─ d_bridge.py     D용 추출기 BExtractor
├─ reference/
│  └─ extract_deep.py  B가 공유 특징 파일을 만든 스크립트 (읽기용. 다른 모듈이 없어 단독 실행 안 됨)
└─ check_sync.py      복사본이 B 원본과 같은지 확인 (B 담당용)
```

## 1. 넘기는 것

| 항목 | 복사본 | 원본 | 설명 |
|---|---|---|---|
| 추출 설정 | `code/configs/PREPROCESSING.md` | `B/configs/PREPROCESSING.md` | A 분류기가 학습한 특징이 이 설정으로 뽑혔다 |
| 추출기 | `code/src/features/d_bridge.py` | `B/src/features/d_bridge.py` | `BExtractor("clip" \| "dinov2" \| "dinov3")`. `encode(images)` → `[N, D]` cls 특징 (CLIP 512, DINOv2 · DINOv3 768). 이미지는 PIL 또는 uint8 RGB 배열 |
| 모델 · 전처리 | `code/src/features/backbones.py` | `B/src/features/backbones.py` | B가 Tsinghua · UF 특징을 뽑은 코드 그대로 |

- **복사본과 원본은 같은 내용이다:** 복사본은 코드를 확인하고, 저장소 없이도 돌릴 수 있게 넣었다. 원본을 고치면 B가 `python B_TO_D/check_sync.py --update`로 복사본을 맞춘다.
- **불러오는 경로는 둘 중 하나다:**
  - 저장소를 clone했으면 `/content/dog-mix-cv/B`
  - 이 폴더만 받았으면 `.../B_TO_D/code`
  - 어느 쪽이든 `from src.features.d_bridge import BExtractor`로 같다.
- **필요한 패키지:** `torch`, `transformers`, `pyyaml`, `pillow`, `numpy` (B `requirements.txt` 기준)

## 2. 사용법

`d_cells.py`의 조각을 붙여 넣는다. 요약하면 다음과 같다.

```python
import sys; sys.path.insert(0, "/content/dog-mix-cv/B")
from src.features.d_bridge import BExtractor

# 노트북(D_RQ2_Colab.ipynb): 셀 7 다음, 셀 10 전에
_extractor_cache["dinov3"] = BExtractor("dinov3")

# run_d_experiment.py: main()의 extractors 사전에
extractors["dinov3"] = BExtractor("dinov3")
```

- **Hugging Face 로그인:** DINOv3는 접근 승인이 필요한 모델이다. 승인된 계정으로 `from huggingface_hub import login; login()`을 먼저 실행한다.
- **L2 정규화:** 출력은 L2 정규화 전 값이다. A 분류기에 넣을 때 하던 대로 L2 정규화를 한다.
- **CLIP · DINOv2도 바꾸기를 권한다:** 지금 D 코드는 DINOv2를 `torch.hub` · fp32로, DINOv3를 Meta 저장소 `.pth`로 불러온다. B 명세와 달라 A 분류기 입력과 어긋날 수 있다.
- **캐시 비우기:** 추출기를 바꾸면 `feature_cache`의 해당 추출기 `.npy`를 지우거나 다른 폴더로 옮긴다. 남아 있으면 그 파일을 그대로 읽는다.

## 3. 기존 B 특징 파일은 D 실험에 쓰이지 않는다

팀 Drive의 `features/{clip,dinov2,dinov3}/{train,val,eval_uf_panels}`는 다음 사진의 특징이다.

- `train` · `val`: Tsinghua 학습셋 · 검증셋 (A 분류기 학습, B 평균점, 순종 평가용)
- `eval_uf_panels`: UF 얼굴 · 전신 칸 사진 200장 (A · B 믹스견 채점용)

D의 원본(개당 원본 1장, `uf_originals_dev`)과 17조건 사진은 여기에 없으므로 위 추출기로 새로 뽑는다.

## 4. 확인 부탁

1. **어느 경로로 돌렸나:** 지금 CLIP · DINOv2 결과는 `run_d_experiment.py`(GrabCut, 크롭 없음)로 냈는가, 노트북(SAM3, 몸통 크롭)으로 냈는가? DINOv3도 같은 경로로 돌려야 추출기끼리 비교된다.
2. **Hugging Face 승인:** B는 코드만 넘기고 실행은 D가 한다. `facebook/dinov3-vitb16-pretrain-lvd1689m` 페이지에서 접근 승인을 받은 계정으로 로그인해야 DINOv3가 내려받아진다.

## 5. 참고

- **SAM3로 바뀌면 다시 돌린다:** 영역 조건 사진이 달라진다. 노트북 경로라면 몸통 크롭 때문에 17조건 전부가 달라진다. 추출기는 그대로이므로 D가 다시 돌리기만 하면 된다.
- **지표 정의가 B · A와 다르다:** D의 포착 비율 · 분포 겹침 계산은 B · A가 쓰는 정의(`B/docs/METRICS.md`)와 다르다.
  - D는 매핑 밖 클래스의 예측을 버리고 다시 나눈다.
  - D는 DNA 비율을 페이지 합으로 정규화한다.
  - 변형 대 대조군 비교(extra_drop)에는 큰 영향이 없지만, RQ1 숫자와 나란히 놓으려면 정의를 맞춰야 한다.
- **비공개 자료:** `d_input/`의 DNA 표 · 사진과 추출한 특징은 저장소에 올리지 않는다.
