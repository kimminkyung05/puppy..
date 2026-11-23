"""
순종만 학습한 대조군 (혼합 없음). 같은 선형 헤드, 같은 LR/에폭/배치.
dev 2,540장에서 top-1, top-3을 계산해 CSV로 저장한다.
"""
import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DRV = "/content/drive/MyDrive/C_work"
LR, EPOCHS, BS = 3e-3, 20, 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(f"{DRV}/index_local_v2.csv")
classes = sorted(df.loc[df["keep_127"], "breed_name"].unique())
cid = {c: i for i, c in enumerate(classes)}
N_CLS = len(classes)

tr = (df["keep_127"] & (df["split"] == "train")).values
va = (df["keep_127"] & (df["split"] == "validation")).values
dev_ids = set(pd.read_csv(f"{DRV}/val_split_seed1.csv").query("split == 'dev'")["image_id"])
dsel = va & df["image_id"].isin(dev_ids).values

y_tr = df.loc[tr, "breed_name"].map(cid).values
y_dev = df.loc[dsel, "breed_name"].map(cid).values


def get_X(ext):
    if ext == "sift":
        X = np.load(f"{DRV}/sift_all_v2.npy").astype(np.float32)
        return X[tr], X[dsel]
    FE = f"/content/{ext}_feat/{ext}"
    if not os.path.exists(FE):
        os.system(f'unzip -q "{DRV}/{ext}.zip" -d /content/{ext}_feat')
    emb = {}
    for sp in ["train", "val"]:
        for s in sorted(glob.glob(f"{FE}/{sp}/shard_*.npz")):
            z = np.load(s)
            emb.update(zip(z["image_id"], z["cls"].astype(np.float32)))
    g = lambda m: np.stack([emb[i] for i in df.loc[m, "image_id"]])
    return g(tr), g(dsel)


rows = []
for ext in ["sift", "clip", "dinov2", "dinov3"]:
    Xtr, Xdev = get_X(ext)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xdev = (Xdev - mu) / sd
    Xt = torch.tensor(Xtr, device=dev)
    Yt = torch.tensor(y_tr, device=dev)

    for SEED in (1, 2, 3):
        torch.manual_seed(SEED)
        model = nn.Linear(Xtr.shape[1], N_CLS).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), BS):
                b = perm[i:i + BS]
                loss = F.cross_entropy(model(Xt[b]), Yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            P = F.softmax(model(torch.tensor(Xdev, device=dev)), -1).cpu().numpy()
        top1 = float((P.argmax(1) == y_dev).mean())
        top3 = float(np.mean([y in np.argsort(-p)[:3] for p, y in zip(P, y_dev)]))
        rows.append({"추출기": ext, "시드": SEED, "dev_n": len(P),
                     "dev_top1": round(top1, 4), "dev_top3": round(top3, 4)})

res = pd.DataFrame(rows)
res.to_csv(f"{DRV}/RQ1_ablation_purebred_v2.csv", index=False)
print(res.groupby("추출기")[["dev_top1", "dev_top3"]].mean().round(4).to_string())
