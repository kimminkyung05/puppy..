import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D_RQ2_Colab.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = []

cells.append(md(r'''# D파트 — RQ2 단서 분석 Colab 실행 노트북

이 노트북은 **A의 분류기 체크포인트를 고정**하고, 믹스견 사진의 입력만 바꿔서 RQ2를 채점한다.

실행 조건은 다음과 같다.

- `original` 1개
- 영역 변형 3개: `area_head`, `area_body`, `area_background`
- 블러 변형 4개: `blur_01`, `blur_03`, `blur_06`, `blur_12`
- 색 변형 1개: `gray`
- 위 8개의 같은 면적/같은 PSNR 대조군 8개
- 특징 추출기 4개 × A의 분류기 seed 3개

최종 산출물은 보고서가 아니라 실행 결과 파일이다.

1. SAM 마스크 캐시 (`masks/*.npz`)
2. 조건·추출기·seed별 이미지 단위 예측 CSV
3. 개체 단위 점수 CSV
4. `original - variant`에서 대조군 하락을 뺀 `extra_drop` 요약표
5. 추가 하락 heatmap

먼저 `SPLIT = "dev"`로 실행한다. 테스트는 설정이 고정된 뒤 `RUN_TEST_ONCE = True`로 한 번만 실행한다.'''))

cells.append(md(r'''## 입력 계약

Google Drive의 프로젝트 폴더를 아래처럼 둔다.

```text
PROJECT_ROOT/
  mixdog_images/                 # 사진 폴더
  mixdog_metadata.csv            # image_id,dog_id,relative_path,split
  dna_long.csv                   # dog_id,breed,fraction
  dog_groups.csv                 # dog_id,breed_group (선택: primary_breed)
  breed_map.csv                  # tsinghua_class,dna_breed
  A_checkpoints/
    sift_seed1.pt
    sift_seed2.pt
    sift_seed3.pt
    clip_seed1.pt
    ...
  extractor_assets/
    dinov3_vitb16.pth             # DINOv3만 필요
    sift_bovw.joblib              # SIFT를 쓸 때 필요
    dinov3_repo/                  # 선택; 없으면 자동 clone
  head_boxes.csv                 # 선택; SAM head 실패 시 수동 box
```

`A_checkpoints/*.pt`는 최소한 다음을 포함해야 한다.

```python
{
    "model_state_dict": linear_layer.state_dict(),
    "feature_dim": 768,
    "class_names": ["breed_a", "breed_b", ...],
    "temperature": 1.0,
    "feature_mean": None,        # 선택
    "feature_std": None,         # 선택
}
```

분류기의 입력 특징은 A/B/C가 사용한 것과 **완전히 같은 전처리**여야 한다. 특히 body crop, 이미지 크기, 정규화, DINOv3 체크포인트를 임의로 바꾸지 않는다.'''))

cells.append(code(r'''#@title 1. Colab 패키지 설치
!pip -q install -U pandas numpy pillow opencv-python-headless scikit-learn scikit-image joblib tqdm matplotlib seaborn transformers accelerate huggingface_hub

# SAM 3 공식 저장소. 이미 설치했으면 clone을 건너뛴다.
import os, subprocess, sys
if not os.path.exists("/content/sam3"):
    !git clone -q https://github.com/facebookresearch/sam3.git /content/sam3
!pip -q install -e /content/sam3
print("패키지 설치 완료. 런타임을 재시작하라는 메시지가 나오면 재시작한 뒤 다음 셀부터 다시 실행하세요.")'''))

cells.append(code(r'''#@title 2. Drive 연결과 설정
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import os, json, math, random, hashlib, warnings
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/content/drive/MyDrive/mixdog_project")  #@param {type:"string"}
IMAGE_ROOT = PROJECT_ROOT / "mixdog_images"
MIX_META_PATH = PROJECT_ROOT / "mixdog_metadata.csv"
DNA_PATH = PROJECT_ROOT / "dna_long.csv"
GROUP_PATH = PROJECT_ROOT / "dog_groups.csv"
BREED_MAP_PATH = PROJECT_ROOT / "breed_map.csv"
CHECKPOINT_ROOT = PROJECT_ROOT / "A_checkpoints"
ASSET_ROOT = PROJECT_ROOT / "extractor_assets"
OUTPUT_ROOT = PROJECT_ROOT / "D_outputs"
MASK_ROOT = OUTPUT_ROOT / "masks"
FEATURE_ROOT = OUTPUT_ROOT / "feature_cache"
for p in [OUTPUT_ROOT, MASK_ROOT, FEATURE_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

SPLIT = "dev"  #@param ["dev", "test"]
RUN_TEST_ONCE = False  #@param {type:"boolean"}
SMOKE_TEST = False  #@param {type:"boolean"}
SMOKE_MAX_DOGS = 8  #@param {type:"integer"}
SEEDS = [1, 2, 3]
EXTRACTORS = ["sift", "clip", "dinov2", "dinov3"]
IMAGE_BATCH_SIZE = 16
FEATURE_BATCH_SIZE = 16
HEAD_PROMPT = "dog head"
BODY_PROMPT = "dog"
BLUR_RATIOS = {"blur_01": 0.01, "blur_03": 0.03, "blur_06": 0.06, "blur_12": 0.12}
SAVE_TRANSFORMED_IMAGES = False
RANDOM_SEED = 20261120

if SPLIT == "test" and not RUN_TEST_ONCE:
    raise ValueError("테스트는 RUN_TEST_ONCE=True를 명시해야 합니다.")
if SPLIT == "test" and (OUTPUT_ROOT / "TEST_LOCK.txt").exists():
    raise RuntimeError("TEST_LOCK.txt가 이미 있습니다. 테스트는 한 번만 실행하도록 잠겨 있습니다.")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("SPLIT:", SPLIT)
print("GPU:", os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null").read().strip())'''))

cells.append(md(r'''## (선택) Tsinghua Dogs 공식 파일 받기

D의 최종 채점에는 Tsinghua 원본 전체가 직접 필요하지 않지만, A/B/C가 특징·분류기를 만들 때는 팀 Drive에 다음 공식 파일을 받아 두어야 한다. 공식 페이지의 저해상도 이미지, 저해상도 annotation, `TrainValSplit` 링크를 그대로 사용한다. 2.5GB 이미지는 Colab 임시 디스크보다 Drive에 보관하는 편이 안전하다.'''))

cells.append(code(r'''#@title 선택: Tsinghua Dogs 저해상도/annotation/split 다운로드
DOWNLOAD_Tsinghua = False  #@param {type:"boolean"}
THU_ROOT = PROJECT_ROOT / "TsinghuaDogs"
THU_ROOT.mkdir(parents=True, exist_ok=True)

if DOWNLOAD_Tsinghua:
    import subprocess
    files = {
        "low-resolution.zip": "https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1",
        "low-annotations.zip": "https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip",
        "TrainValSplit.zip": "https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip",
    }
    for name, url in files.items():
        target = THU_ROOT / name
        if not target.exists():
            print("downloading", name)
            subprocess.run(["wget", "-c", url, "-O", str(target)], check=True)
        else:
            print("exists", name)
    print("압축파일 저장 위치:", THU_ROOT)
else:
    print("건너뜀. 필요할 때 DOWNLOAD_Tsinghua=True로 바꾸세요.")'''))

cells.append(code(r'''#@title 3. 메타데이터 로드와 입력 검증
from pathlib import Path

def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}에 필요한 열이 없습니다: {missing}")

meta = pd.read_csv(MIX_META_PATH)
require_columns(meta, ["image_id", "dog_id", "relative_path", "split"], "mixdog_metadata.csv")
meta = meta[meta["split"].astype(str).eq(SPLIT)].copy()
meta["image_path"] = meta["relative_path"].map(lambda x: str(IMAGE_ROOT / str(x)))
missing_images = meta.loc[~meta["image_path"].map(os.path.exists)]
if len(missing_images):
    raise FileNotFoundError(f"사진이 {len(missing_images)}장 없습니다. 예: {missing_images.iloc[0]['image_path']}")

dna = pd.read_csv(DNA_PATH)
require_columns(dna, ["dog_id", "breed", "fraction"], "dna_long.csv")
dna["fraction"] = dna["fraction"].astype(float)
if dna["fraction"].max() > 1.0:
    dna["fraction"] = dna["fraction"] / 100.0
dna = dna[dna["dog_id"].isin(meta["dog_id"].unique())].copy()
dna = dna[dna["fraction"] > 0].copy()
dna["fraction"] = dna["fraction"] / dna.groupby("dog_id")["fraction"].transform("sum")

if GROUP_PATH.exists():
    groups = pd.read_csv(GROUP_PATH)
    require_columns(groups, ["dog_id", "breed_group"], "dog_groups.csv")
else:
    groups = pd.DataFrame({"dog_id": meta["dog_id"].unique(), "breed_group": "unassigned"})

breed_map = pd.read_csv(BREED_MAP_PATH)
require_columns(breed_map, ["tsinghua_class", "dna_breed"], "breed_map.csv")
breed_map = breed_map.dropna(subset=["tsinghua_class"]).copy()
breed_map["tsinghua_class"] = breed_map["tsinghua_class"].astype(str)
breed_map["dna_breed"] = breed_map["dna_breed"].fillna("__DROP__").astype(str)

if SMOKE_TEST:
    keep = meta["dog_id"].drop_duplicates().sample(min(SMOKE_MAX_DOGS, meta["dog_id"].nunique()), random_state=RANDOM_SEED)
    meta = meta[meta["dog_id"].isin(keep)].copy()
    dna = dna[dna["dog_id"].isin(keep)].copy()
    groups = groups[groups["dog_id"].isin(keep)].copy()

print("images:", len(meta), "dogs:", meta["dog_id"].nunique(), "DNA rows:", len(dna))
display(meta.head(2))
display(dna.head(2))'''))

cells.append(code(r'''#@title 4. SAM 3 로드
# SAM 3 체크포인트는 공식 Hugging Face 저장소 접근 승인이 필요할 수 있습니다.
# 승인 후 `notebook_login()`을 한 번 실행하세요.
import torch
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "Colab 메뉴에서 GPU 런타임을 선택하세요."
print("torch:", torch.__version__, "cuda:", torch.version.cuda)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_model = build_sam3_image_model()
sam3_model.eval()
sam3_processor = Sam3Processor(sam3_model)
print("SAM 3 준비 완료")'''))

cells.append(code(r'''#@title 5. 마스크 추출과 캐시
def to_numpy_mask(mask, h, w):
    if torch.is_tensor(mask):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    if mask.shape != (h, w):
        mask = np.array(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
    return mask.astype(bool)

def select_union_mask(output, h, w, score_margin=0.15):
    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None or len(masks) == 0:
        return np.zeros((h, w), dtype=bool), 0.0
    if torch.is_tensor(scores):
        scores = scores.detach().float().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    order = np.argsort(-scores)
    best = float(scores[order[0]])
    chosen = order[scores[order] >= best - score_margin]
    union = np.zeros((h, w), dtype=bool)
    for i in chosen[:5]:
        union |= to_numpy_mask(masks_np[i], h, w)
    return union, best

def manual_head_mask(image_id, h, w):
    box_path = PROJECT_ROOT / "head_boxes.csv"
    if not box_path.exists():
        return None
    boxes = pd.read_csv(box_path)
    needed = {"image_id", "x1", "y1", "x2", "y2"}
    if not needed.issubset(boxes.columns):
        raise ValueError("head_boxes.csv는 image_id,x1,y1,x2,y2 열이 필요합니다.")
    row = boxes[boxes["image_id"].astype(str).eq(str(image_id))]
    if len(row) == 0:
        return None
    x1, y1, x2, y2 = row.iloc[0][["x1", "y1", "x2", "y2"]].astype(float).round().astype(int)
    m = np.zeros((h, w), dtype=bool)
    m[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True
    return m

def infer_masks(image_path, image_id):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    state = sam3_processor.set_image(image)
    body_out = sam3_processor.set_text_prompt(state=state, prompt=BODY_PROMPT)
    body, body_score = select_union_mask(body_out, h, w)
    head_out = sam3_processor.set_text_prompt(state=state, prompt=HEAD_PROMPT)
    head, head_score = select_union_mask(head_out, h, w)
    if head.sum() == 0:
        manual = manual_head_mask(image_id, h, w)
        if manual is not None:
            head = manual
            head_score = -1.0
    if body.sum() == 0:
        raise RuntimeError(f"body mask 실패: {image_id}")
    head &= body
    return body, head, body_score, head_score, (w, h)

mask_rows = []
for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="SAM masks"):
    out_path = MASK_ROOT / f"{row.image_id}.npz"
    if out_path.exists():
        z = np.load(out_path)
        mask_rows.append({"image_id": row.image_id, "body_area": int(z["body"].sum()), "head_area": int(z["head"].sum()), "body_score": float(z["body_score"]), "head_score": float(z["head_score"])})
        continue
    body, head, body_score, head_score, size = infer_masks(row.image_path, row.image_id)
    np.savez_compressed(out_path, body=body, head=head, body_score=body_score, head_score=head_score, size=np.asarray(size))
    mask_rows.append({"image_id": row.image_id, "body_area": int(body.sum()), "head_area": int(head.sum()), "body_score": float(body_score), "head_score": float(head_score)})
mask_stats = pd.DataFrame(mask_rows)
mask_stats.to_csv(OUTPUT_ROOT / f"mask_stats_{SPLIT}.csv", index=False)
display(mask_stats.describe())'''))

cells.append(code(r'''#@title 6. 변형과 대조군 생성
CONDITIONS = ["original", "area_head", "area_body", "area_background", *BLUR_RATIOS.keys(), "gray",
              "control_area_head", "control_area_body", "control_area_background", *[f"control_{k}" for k in BLUR_RATIOS], "control_gray"]
PAIR = {
    "area_head": "control_area_head", "area_body": "control_area_body", "area_background": "control_area_background",
    "blur_01": "control_blur_01", "blur_03": "control_blur_03", "blur_06": "control_blur_06", "blur_12": "control_blur_12",
    "gray": "control_gray",
}

def load_mask(image_id):
    z = np.load(MASK_ROOT / f"{image_id}.npz")
    return z["body"].astype(bool), z["head"].astype(bool)

def crop_to_body(img, body):
    ys, xs = np.where(body)
    if len(xs) == 0:
        raise ValueError("empty body mask")
    h, w = body.shape
    pad_x = int(0.10 * (xs.max() - xs.min() + 1))
    pad_y = int(0.10 * (ys.max() - ys.min() + 1))
    box = (max(0, xs.min()-pad_x), max(0, ys.min()-pad_y), min(w, xs.max()+1+pad_x), min(h, ys.max()+1+pad_y))
    return img.crop(box)

def fill_outside(arr, keep):
    fill = np.median(arr.reshape(-1, 3), axis=0).round().astype(np.uint8)
    out = np.broadcast_to(fill, arr.shape).copy()
    out[keep] = arr[keep]
    return out

def random_same_area_mask(shape, area, seed):
    rng = np.random.default_rng(seed)
    n = int(area)
    n = min(max(n, 1), shape[0] * shape[1])
    idx = rng.choice(shape[0] * shape[1], size=n, replace=False)
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    out[idx] = True
    return out.reshape(shape)

def stable_seed(*parts):
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)

def psnr_matched_noise(original, target, seed):
    a = np.asarray(original).astype(np.float32)
    b = np.asarray(target).astype(np.float32)
    target_mse = float(np.mean((a - b) ** 2))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=a.shape).astype(np.float32)
    noise *= math.sqrt(max(target_mse, 1e-8) / float(np.mean(noise ** 2)))
    return np.clip(a + noise, 0, 255).astype(np.uint8)

def gaussian_blur(arr, ratio):
    sigma = max(0.5, ratio * min(arr.shape[:2]))
    radius = max(1, int(math.ceil(3 * sigma)))
    kernel = 2 * radius + 1
    # cv2 is faster and deterministic for this use.
    import cv2
    return cv2.GaussianBlur(arr, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

def make_condition(image_id, image_path, condition):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    body, head = load_mask(image_id)
    if body.shape != arr.shape[:2]:
        body = np.array(Image.fromarray(body.astype(np.uint8)*255).resize((arr.shape[1], arr.shape[0]), Image.Resampling.NEAREST)) > 0
        head = np.array(Image.fromarray(head.astype(np.uint8)*255).resize((arr.shape[1], arr.shape[0]), Image.Resampling.NEAREST)) > 0
    if condition == "original":
        out = arr
    elif condition == "area_head":
        out = fill_outside(arr, head)
    elif condition == "area_body":
        out = fill_outside(arr, body & ~head)
    elif condition == "area_background":
        out = fill_outside(arr, ~body)
    elif condition in BLUR_RATIOS:
        out = gaussian_blur(arr, BLUR_RATIOS[condition])
    elif condition == "gray":
        out = np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))
    elif condition.startswith("control_area_"):
        base = {"control_area_head": head, "control_area_body": body & ~head, "control_area_background": ~body}[condition]
        random_mask = random_same_area_mask(base.shape, int(base.sum()), RANDOM_SEED + stable_seed(image_id, condition) % 100000)
        out = fill_outside(arr, random_mask)
    elif condition.startswith("control_blur_"):
        variant = condition.replace("control_", "")
        blurred = gaussian_blur(arr, BLUR_RATIOS[variant])
        out = psnr_matched_noise(arr, blurred, RANDOM_SEED + stable_seed(image_id, condition) % 100000)
    elif condition == "control_gray":
        gray = np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))
        out = psnr_matched_noise(arr, gray, RANDOM_SEED + stable_seed(image_id, condition) % 100000)
    else:
        raise KeyError(condition)
    # A의 순종 학습과 맞추기 위해 모든 조건에 같은 원본 body bbox crop을 적용한다.
    return crop_to_body(Image.fromarray(out), body)

# 한 장의 변형을 눈으로 확인하고, 최종 전체 실행 전에 반드시 검토한다.
sample = meta.iloc[0]
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for ax, cond in zip(axes.flat, CONDITIONS):
    ax.imshow(make_condition(sample.image_id, sample.image_path, cond))
    ax.set_title(cond, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / f"condition_preview_{sample.image_id}.png", dpi=160)
plt.show()
print("조건 수:", len(CONDITIONS), CONDITIONS)'''))

cells.append(code(r'''#@title 7. 특징 추출기 로드
import cv2, joblib
from torch import nn
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Extractor:
    def encode(self, images):
        raise NotImplementedError

class ClipExtractor(Extractor):
    def __init__(self):
        from transformers import CLIPModel, CLIPImageProcessor
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()
        self.model = clip.vision_model
        self.proj = clip.visual_projection
    @torch.inference_mode()
    def encode(self, images):
        x = self.processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)
        h = self.model(pixel_values=x).pooler_output
        return self.proj(h).float().cpu().numpy()

class DinoV2Extractor(Extractor):
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE).eval()
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack([self.tf(im) for im in images]).to(DEVICE)
        return self.model(x).float().cpu().numpy()

class DinoV3Extractor(Extractor):
    def __init__(self):
        repo = ASSET_ROOT / "dinov3_repo"
        if not repo.exists():
            !git clone -q https://github.com/facebookresearch/dinov3.git {repo}
        weights = ASSET_ROOT / "dinov3_vitb16.pth"
        if not weights.exists():
            raise FileNotFoundError("DINOv3 체크포인트를 extractor_assets/dinov3_vitb16.pth에 넣으세요. Meta 공식 접근 승인이 필요할 수 있습니다.")
        self.model = torch.hub.load(str(repo), "dinov3_vitb16", source="local", weights=str(weights)).to(DEVICE).eval()
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack([self.tf(im) for im in images]).to(DEVICE)
        out = self.model(x)
        # 공식 모델의 global output을 사용한다.
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", out.get("x_norm_clstoken", out.get("x_prenorm")))
        return out.float().cpu().numpy()

class SiftBovwExtractor(Extractor):
    def __init__(self):
        asset = ASSET_ROOT / "sift_bovw.joblib"
        if not asset.exists():
            raise FileNotFoundError("SIFT BoVW asset가 없습니다: extractor_assets/sift_bovw.joblib")
        obj = joblib.load(asset)
        self.kmeans = obj["kmeans"] if isinstance(obj, dict) else obj
        self.vocab_size = int(getattr(self.kmeans, "n_clusters", 512))
        self.sift = cv2.SIFT_create(nfeatures=800)
    def encode(self, images):
        rows = []
        for im in images:
            gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
            _, desc = self.sift.detectAndCompute(gray, None)
            hist = np.zeros(self.vocab_size, dtype=np.float32)
            if desc is not None and len(desc):
                words = self.kmeans.predict(desc)
                hist += np.bincount(words, minlength=self.vocab_size).astype(np.float32)
            hist /= max(hist.sum(), 1.0)
            rows.append(hist)
        return np.stack(rows)

_extractor_cache = {}
def get_extractor(name):
    if name not in _extractor_cache:
        print("loading", name)
        _extractor_cache[name] = {"clip": ClipExtractor, "dinov2": DinoV2Extractor, "dinov3": DinoV3Extractor, "sift": SiftBovwExtractor}[name]()
    return _extractor_cache[name]

print("특징 추출기는 실행 루프에서 필요할 때 순차적으로 로드됩니다.")'''))

cells.append(code(r'''#@title 8. A 분류기 체크포인트와 품종 매핑
def checkpoint_path(extractor_name, seed):
    candidates = [CHECKPOINT_ROOT / f"{extractor_name}_seed{seed}.pt", CHECKPOINT_ROOT / f"clf_{extractor_name}_seed{seed}.pt", CHECKPOINT_ROOT / f"{extractor_name}_clf_seed{seed}.pt"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"체크포인트 없음: {candidates}")

def load_classifier(extractor_name, seed):
    path = checkpoint_path(extractor_name, seed)
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt or "class_names" not in ckpt:
        raise ValueError(f"{path}에 model_state_dict/class_names가 필요합니다.")
    class_names = [str(x) for x in ckpt["class_names"]]
    state = ckpt["model_state_dict"]
    weight_key = "weight" if "weight" in state else "linear.weight"
    dim = int(ckpt.get("feature_dim", state[weight_key].shape[1]))
    model = nn.Linear(dim, len(class_names))
    state = {k.replace("linear.", "", 1) if k.startswith("linear.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return {"model": model, "class_names": class_names, "temperature": float(ckpt.get("temperature", 1.0)),
            "feature_mean": ckpt.get("feature_mean"), "feature_std": ckpt.get("feature_std"), "path": str(path)}

map_dict = dict(zip(breed_map["tsinghua_class"], breed_map["dna_breed"]))
def aggregate_to_dna(probs, class_names):
    names = [map_dict.get(c, "__DROP__") for c in class_names]
    dna_names = sorted(set(names) - {"__DROP__"})
    out = np.zeros((len(probs), len(dna_names)), dtype=np.float32)
    for j, n in enumerate(names):
        if n in dna_names:
            out[:, dna_names.index(n)] += probs[:, j]
    denom = out.sum(axis=1, keepdims=True)
    out = out / np.clip(denom, 1e-8, None)
    return pd.DataFrame(out, columns=dna_names)

print("breed map rows:", len(breed_map))'''))

cells.append(code(r'''#@title 9. 평가 함수
def bootstrap_ci(values, seed=RANDOM_SEED, n_boot=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

def per_dog_ground_truth(dog_ids):
    wide = dna[dna["dog_id"].isin(dog_ids)].pivot_table(index="dog_id", columns="breed", values="fraction", aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(index=dog_ids, fill_value=0.0)
    return wide.div(wide.sum(axis=1).replace(0, 1), axis=0)

def score_predictions(image_pred, row_meta):
    # 이미지별 예측을 개체 평균으로 묶는다.
    pred_cols = [c for c in image_pred.columns if c.startswith("pred_")]
    img = image_pred[["image_id", "dog_id", *pred_cols]].copy()
    dog_pred = img.groupby("dog_id")[pred_cols].mean()
    dog_pred.columns = [c.replace("pred_", "") for c in dog_pred.columns]
    gt = per_dog_ground_truth(dog_pred.index.tolist())
    all_breeds = sorted(set(gt.columns) | set(dog_pred.columns))
    pred = dog_pred.reindex(columns=all_breeds, fill_value=0.0).to_numpy()
    true = gt.reindex(columns=all_breeds, fill_value=0.0).to_numpy()
    pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-8, None)
    rows = []
    for i, dog_id in enumerate(dog_pred.index):
        top3 = np.argsort(-pred[i])[:3]
        capture = float(true[i, top3].sum())
        overlap = float(np.minimum(pred[i], true[i]).sum())
        kl = float(np.sum(true[i] * np.log((true[i] + 1e-8) / (pred[i] + 1e-8))))
        rows.append({"dog_id": dog_id, "capture_top3": capture, "overlap": overlap, "kl": kl})
    out = pd.DataFrame(rows).merge(groups[["dog_id", "breed_group"]], on="dog_id", how="left")
    return out

def run_classifier(features, clf):
    x = torch.from_numpy(features).float()
    if clf["feature_mean"] is not None and clf["feature_std"] is not None:
        x = (x - torch.as_tensor(clf["feature_mean"]).float()) / torch.as_tensor(clf["feature_std"]).float().clamp_min(1e-8)
    with torch.inference_mode():
        logits = clf["model"](x) / max(clf["temperature"], 1e-6)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def condition_scores(detail_rows):
    rows = []
    for (extractor, seed, condition, breed_group), g in detail_rows.groupby(["extractor", "seed", "condition", "breed_group"], dropna=False):
        for metric in ["capture_top3", "overlap", "kl"]:
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy())
            rows.append({"extractor": extractor, "seed": seed, "condition": condition, "breed_group": breed_group,
                         "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n_dogs": len(g)})
    return pd.DataFrame(rows)

def extra_drop(detail_rows):
    keys = ["extractor", "seed", "dog_id", "breed_group"]
    base = detail_rows[detail_rows["condition"].eq("original")][keys + ["capture_top3", "overlap", "kl"]].rename(columns={"capture_top3":"base_capture", "overlap":"base_overlap", "kl":"base_kl"})
    out = []
    for variant, control in PAIR.items():
        v = detail_rows[detail_rows["condition"].eq(variant)][keys + ["capture_top3", "overlap", "kl"]].rename(columns={"capture_top3":"variant_capture", "overlap":"variant_overlap", "kl":"variant_kl"})
        c = detail_rows[detail_rows["condition"].eq(control)][keys + ["capture_top3", "overlap", "kl"]].rename(columns={"capture_top3":"control_capture", "overlap":"control_overlap", "kl":"control_kl"})
        m = base.merge(v, on=keys, how="inner").merge(c, on=keys, how="inner")
        m["condition"] = variant
        m["extra_drop_capture_top3"] = m["control_capture"] - m["variant_capture"]
        m["extra_drop_overlap"] = m["control_overlap"] - m["variant_overlap"]
        m["extra_drop_kl"] = m["variant_kl"] - m["control_kl"]
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
'''))

cells.append(code(r'''#@title 10. 전체 조건 채점
def extract_features_for_condition(extractor_name, condition):
    cache = FEATURE_ROOT / SPLIT / extractor_name
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"{condition}.npy"
    if out_path.exists():
        return np.load(out_path)
    extractor = get_extractor(extractor_name)
    feats = []
    rows = list(meta.itertuples(index=False))
    for start in tqdm(range(0, len(rows), FEATURE_BATCH_SIZE), desc=f"{extractor_name}/{condition}"):
        batch = rows[start:start+FEATURE_BATCH_SIZE]
        images = [make_condition(r.image_id, r.image_path, condition) for r in batch]
        feats.append(extractor.encode(images))
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    np.save(out_path, feats)
    return feats

all_detail = []
for extractor_name in EXTRACTORS:
    for condition in CONDITIONS:
        features = extract_features_for_condition(extractor_name, condition)
        for seed in SEEDS:
            clf = load_classifier(extractor_name, seed)
            probs = run_classifier(features, clf)
            dna_pred = aggregate_to_dna(probs, clf["class_names"])
            pred = pd.concat([meta[["image_id", "dog_id"]].reset_index(drop=True), dna_pred.add_prefix("pred_")], axis=1)
            pred_path = OUTPUT_ROOT / f"pred_{SPLIT}_{extractor_name}_{condition}_seed{seed}.csv"
            pred.to_csv(pred_path, index=False)
            scored = score_predictions(pred, meta)
            scored["extractor"] = extractor_name
            scored["seed"] = seed
            scored["condition"] = condition
            all_detail.append(scored)

detail = pd.concat(all_detail, ignore_index=True)
detail.to_csv(OUTPUT_ROOT / f"detail_scores_{SPLIT}.csv", index=False)
summary = condition_scores(detail)
summary.to_csv(OUTPUT_ROOT / f"condition_summary_{SPLIT}.csv", index=False)
extra = extra_drop(detail)
extra.to_csv(OUTPUT_ROOT / f"extra_drop_detail_{SPLIT}.csv", index=False)
print("채점 완료:", len(detail), "개체-조건-추출기-seed 행")
display(summary[summary.metric.eq("capture_top3")].head(20))'''))

cells.append(code(r'''#@title 11. 추가 하락 요약과 그림 저장
extra_summary_rows = []
for (extractor, seed, condition, breed_group), g in extra.groupby(["extractor", "seed", "condition", "breed_group"], dropna=False):
    for metric in ["extra_drop_capture_top3", "extra_drop_overlap", "extra_drop_kl"]:
        mean, lo, hi = bootstrap_ci(g[metric].to_numpy())
        extra_summary_rows.append({"extractor": extractor, "seed": seed, "condition": condition, "breed_group": breed_group,
                                   "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n_dogs": len(g)})
extra_summary = pd.DataFrame(extra_summary_rows)
extra_summary.to_csv(OUTPUT_ROOT / f"extra_drop_summary_{SPLIT}.csv", index=False)

import seaborn as sns
plot_df = extra_summary[extra_summary["metric"].eq("extra_drop_capture_top3")].copy()
plot_df["label"] = plot_df["condition"] + " / " + plot_df["breed_group"].fillna("unassigned")
pivot = plot_df.groupby(["condition", "extractor"])["mean"].mean().unstack("extractor").reindex(PAIR.keys())
plt.figure(figsize=(11, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", center=0, cmap="coolwarm")
plt.title(f"RQ2 extra drop: capture_top3 ({SPLIT})")
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / f"extra_drop_heatmap_{SPLIT}.png", dpi=180)
plt.show()

display(extra_summary[extra_summary["metric"].eq("extra_drop_capture_top3")].sort_values("mean", ascending=False).head(30))'''))

cells.append(code(r'''#@title 12. 실행 기록과 테스트 잠금
run_info = {
    "split": SPLIT,
    "conditions": CONDITIONS,
    "pair_controls": PAIR,
    "extractors": EXTRACTORS,
    "seeds": SEEDS,
    "n_images": int(len(meta)),
    "n_dogs": int(meta["dog_id"].nunique()),
    "project_root": str(PROJECT_ROOT),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": DEVICE,
}
with open(OUTPUT_ROOT / f"run_info_{SPLIT}.json", "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)

if SPLIT == "test":
    (OUTPUT_ROOT / "TEST_LOCK.txt").write_text("test run completed once\n", encoding="utf-8")
    print("TEST_LOCK.txt 생성: 테스트 결과를 고정했습니다.")
else:
    print("dev 실행 완료. DINOv3/품종군/변형 설정을 고정한 뒤 test를 한 번만 실행하세요.")'''))

cells.append(md(r'''## 실행할 때 생기는 파일

`D_outputs/`만 A와 공유하면 된다.

- `pred_<split>_<extractor>_<condition>_seed<seed>.csv`
- `detail_scores_<split>.csv`
- `condition_summary_<split>.csv`
- `extra_drop_detail_<split>.csv`
- `extra_drop_summary_<split>.csv`
- `extra_drop_heatmap_<split>.png`
- `mask_stats_<split>.csv`
- `run_info_<split>.json`

`test` 실행 전에 A에게서 받은 체크포인트, `breed_map.csv`, DNA 메타데이터 버전을 고정하고 Drive에 별도 보관한다. 이 노트북은 보고서 문장을 생성하지 않는다.'''))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "D_RQ2_Colab.ipynb", "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
