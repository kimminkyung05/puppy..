"""
UF 패널 평가 입력 생성 (C파트).

산출물
  1) SIFT 특징: sift_eval_uf_v5_all.npy, index_eval_uf_v5_all.csv
     조건: orig, gray, blur1, blur3, blur6, blur12  (D 기준 σ=1,3,6,12)
  2) PNG 이미지 zip: uf_orig_png_m, uf_gray_png_m, uf_blur{σ}_png, uf_head_png, uf_body_png
     각 zip 안에 manifest.csv(path = zip 안 상대경로) 포함
  3) manifest_all_uf.csv (전체 목록)

주의
  - SIFT 추출은 sift_codebook.npy(train으로 학습)를 그대로 사용한다.
  - 이미지 zip은 모두 PNG로 저장한다(JPEG 재저장 금지).
  - 입력: evalset/eval_index_UF_panels.csv, evalset/raw/UF/panels/*.jpg
"""
import os, shutil, zipfile
import numpy as np, pandas as pd, cv2

EV = "/content/evalset"
DRV = "/content/drive/MyDrive/C_work"
SIGMAS = [1, 3, 6, 12]           # D 기준 (확정)
PANEL_ZIPS = {"face": "uf_head_png", "body": "uf_body_png"}

idx = pd.read_csv(f"{EV}/eval_index_UF_panels.csv")
paths = [f"{EV}/raw/UF/panels/{i}.jpg" for i in idx["image_id"]]

# ---------- 1) SIFT BoVW (가장 가까운 코드워드, sqrt 정규화) ----------
centers = np.load(f"{DRV}/sift_codebook.npy").astype(np.float32)
cs = (centers ** 2).sum(1)
sift = cv2.SIFT_create()

def bovw(gray):
    h = np.zeros(len(centers), np.float32)
    _, d = sift.detectAndCompute(cv2.resize(gray, (336, 336)), None)
    if d is not None and len(d):
        d = d.astype(np.float32)
        a = ((d ** 2).sum(1, keepdims=True) - 2 * d @ centers.T + cs).argmin(1)
        h = np.bincount(a, minlength=len(centers)).astype(np.float32)
        h /= max(h.sum(), 1)
        h = np.sqrt(h)
    return h

gray_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
assert all(im is not None for im in gray_imgs), "이미지 읽기 실패"

feats, rows = [], []
def add(cond, F):
    feats.append(F)
    rows.extend({"condition": cond, **r} for r in idx.to_dict("records"))

add("orig", np.stack([bovw(im) for im in gray_imgs]))
add("gray", np.stack([bovw(im) for im in gray_imgs]))   # SIFT는 흑백 입력이므로 orig와 동일
for sg in SIGMAS:
    add(f"blur{sg}", np.stack([bovw(cv2.GaussianBlur(im, (0, 0), sg)) for im in gray_imgs]))

np.save(f"{DRV}/sift_eval_uf_v5_all.npy", np.concatenate(feats))
pd.DataFrame(rows).to_csv(f"{DRV}/index_eval_uf_v5_all.csv", index=False)
print("SIFT:", np.concatenate(feats).shape)

# ---------- 2) PNG zip + manifest ----------
def make_zip(name, rows_list):
    out = f"/content/_fix/{name}"
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    for r in rows_list:
        img = r.pop("_img")
        cv2.imwrite(f"{out}/{r['path']}", img)
    man = pd.DataFrame([{k: v for k, v in r.items()} for r in rows_list])
    man.to_csv(f"{out}/manifest.csv", index=False)
    shutil.make_archive(f"/content/_fix/{name}", "zip", out)
    shutil.copy(f"/content/_fix/{name}.zip", f"{DRV}/{name}.zip")
    return man

def base_row(r, cond, sigma, path_src):
    return {"zip_file": None, "path": None, "image_id": r.image_id, "dog_id": r.dog_id,
            "panel": r.panel, "split": r.split, "condition": cond, "sigma": sigma,
            "source_original": path_src}

all_man = []
color = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]

# 원본 대조군 (컬러 PNG)
rows = []
for r, img, p in zip(idx.itertuples(), color, paths):
    rows.append({**base_row(r, "orig", 0, p), "_img": img, "path": f"{r.image_id}.png"})
man = make_zip("uf_orig_png_m", rows); man["zip_file"] = "uf_orig_png_m.zip"; all_man.append(man)

# 흑백 (그레이 PNG)
rows = []
for r, img, p in zip(idx.itertuples(), gray_imgs, paths):
    rows.append({**base_row(r, "gray", None, p), "_img": img, "path": f"{r.image_id}.png"})
man = make_zip("uf_gray_png_m", rows); man["zip_file"] = "uf_gray_png_m.zip"; all_man.append(man)

# 블러 (σ별 zip)
for sg in SIGMAS:
    rows = []
    for r, img, p in zip(idx.itertuples(), color, paths):
        b = cv2.GaussianBlur(img, (0, 0), sg)
        rows.append({**base_row(r, f"blur{sg}", sg, p), "_img": b, "path": f"{r.image_id}.png"})
    man = make_zip(f"uf_blur{sg}_png", rows); man["zip_file"] = f"uf_blur{sg}_png.zip"; all_man.append(man)

# 패널 사진 (얼굴 / 몸, 머리·몸 마스킹 아님)
for panel, name in PANEL_ZIPS.items():
    rows = []
    for r, img, p in zip(idx.itertuples(), color, paths):
        if r.panel != panel:
            continue
        rows.append({**base_row(r, f"panel_{panel}", None, p), "_img": img, "path": f"{r.image_id}.png"})
    man = make_zip(name, rows); man["zip_file"] = f"{name}.zip"; all_man.append(man)

allm = pd.concat(all_man, ignore_index=True)
allm = allm[["zip_file", "path", "image_id", "dog_id", "panel", "split", "condition", "sigma", "source_original"]]
allm.to_csv(f"{DRV}/manifest_all_uf.csv", index=False)
print("manifest:", len(allm), allm["condition"].value_counts().to_dict())
