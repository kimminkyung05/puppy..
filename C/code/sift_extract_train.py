"""
SIFT BoVW 특징 추출 (train/validation) 및 코드북 생성.
사용자가 실행한 Colab 셀 원본 (앞부분).
주의: 이 파일은 bovw(r)까지만 있다. 저장 단계(sift_all_v2.npy 생성)와
      sqrt 정규화 적용 위치는 원본 셀의 나머지 부분을 확인해서 추가해야 한다.
"""
import os, shutil, numpy as np, pandas as pd, cv2
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

K, CH = 512, 5000
DRV = "/content/drive/MyDrive/C_work"
df = pd.read_csv("/content/index.csv")
df["path"] = df["path"].str.replace(
    "/content/drive/MyDrive/Tsinghua Dogs Dataset/low-resolution/low-resolution",
    "/content/low-resolution", regex=False)
os.makedirs(f"{DRV}/sift_chunks", exist_ok=True)
os.makedirs("/content/feats/sift_chunks", exist_ok=True)
sift = cv2.SIFT_create()

def descriptors(r):
    img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    crop = img[r.y1:r.y2, r.x1:r.x2]
    if crop.size == 0: return None
    _, des = sift.detectAndCompute(cv2.resize(crop, (336, 336)), None)
    return des

# 1) 코드북: Drive에 있으면 재사용, 없으면 학습 사진 2만 장으로 만들고 백업
cb_local, cb_drv = "/content/feats/sift_codebook.npy", f"{DRV}/sift_codebook.npy"
if os.path.exists(cb_drv):
    shutil.copy(cb_drv, cb_local)
else:
    pool = [d for r in df[df.split == "train"].sample(20000, random_state=0).itertuples()
            if (d := descriptors(r)) is not None]
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=0, n_init=3).fit(np.vstack(pool))
    np.save(cb_local, km.cluster_centers_)
    shutil.copy(cb_local, cb_drv)
centers = np.load(cb_local)
km = MiniBatchKMeans(n_clusters=K, init=centers, n_init=1, max_iter=1).fit(centers)

def bovw(r):
    des = descriptors(r)
    h = np.zeros(K, np.float32)
    if des is not None and len(des):
        h = np.bincount(km.predict(des), minlength=K).astype(np.float32)
        h /= max(h.sum(), 1)
    return h
# (이하 원본 셀의 나머지: 전체 행 추출, sqrt 정규화, sift_all_v2.npy 저장)
