"""
혼합 학습 (v2, 딥 특징 추출기). 사용자가 실행한 Colab 셀 원본 그대로.
설정: ALPHA=1.0, KMAX=4, LR=3e-3, EPOCHS=20, N_MIX=20000, BS=512
에폭마다 순종 샘플 전체 + 혼합 샘플 N_MIX개를 함께 학습한다.
SIFT v2도 같은 구조이며, X 대신 sift_all_v2.npy를 쓴다(확인 필요).
EXT를 "clip" → "dinov2" → "dinov3" 순서로 바꿔 실행한다.
"""
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

EXT = "dinov3"          # 처음 실행: "clip" → 다음 "dinov2" → 다음 "dinov3"
DRV = "/content/drive/MyDrive/C_work"
FE = f"/content/{EXT}_feat/{EXT}"
if not os.path.exists(FE):
    os.system(f'unzip -q "{DRV}/{EXT}.zip" -d /content/{EXT}_feat')

emb = {}
for sp in ["train", "val", "eval_uf_panels"]:
    for s in sorted(glob.glob(f"{FE}/{sp}/shard_*.npz")):
        z = np.load(s)
        emb.update(zip(z["image_id"], z["cls"].astype(np.float32)))

df = pd.read_csv(f"{DRV}/index_local_v2.csv")
df["image_id"] = df["path"].str.extract(r"(n\d{6})")[0] + "/" + df["path"].str.extract(r"/(n\d+)\.[a-z]+$")[0]
classes = sorted(df.loc[df["keep_127"], "breed_name"].unique())
cid = {c: i for i, c in enumerate(classes)}
N_CLS = len(classes)

tr = df["keep_127"] & (df["split"] == "train")
va = df["keep_127"] & (df["split"] == "validation")
X_tr = np.stack([emb[i] for i in df.loc[tr, "image_id"]])
X_va = np.stack([emb[i] for i in df.loc[va, "image_id"]])
y_tr = df.loc[tr, "breed_name"].map(cid).values
y_va = df.loc[va, "breed_name"].map(cid).values

mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
X_tr = (X_tr - mu) / sd
X_va = (X_va - mu) / sd
by_cls = [np.where(y_tr == c)[0] for c in range(N_CLS)]
print(EXT, "| 클래스:", N_CLS, "| train:", X_tr.shape, "| val:", X_va.shape)

ALPHA, KMAX, LR, EPOCHS, N_MIX, BS = 1.0, 4, 3e-3, 20, 20000, 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

dev_ids = set(pd.read_csv(f"{DRV}/val_split_seed1.csv").query("split == 'dev'")["image_id"])
dsel = va.values & df["image_id"].isin(dev_ids).values
X_dev = (np.stack([emb[i] for i in df.loc[dsel, "image_id"]]) - mu) / sd
meta_dev = df.loc[dsel, ["image_id", "breed_name"]].reset_index(drop=True)

ev = pd.read_csv(f"{DRV}/index_eval_uf.csv")
X_ev = (np.stack([emb[i] for i in ev["image_id"]]) - mu) / sd

for SEED in (1, 2, 3):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)

    def make_mixed(n):
        F_, T_ = [], []
        for _ in range(n):
            k = rng.integers(2, KMAX + 1)
            cl = rng.choice(N_CLS, k, replace=False)
            r = rng.dirichlet([ALPHA] * k)
            f = sum(w * X_tr[rng.choice(by_cls[c])] for c, w in zip(cl, r))
            t = np.zeros(N_CLS, np.float32); t[cl] = r
            F_.append(f); T_.append(t)
        return np.stack(F_).astype(np.float32), np.stack(T_)

    model = nn.Linear(X_tr.shape[1], N_CLS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(EPOCHS):
        Xm, Tm = make_mixed(N_MIX)
        Xb = torch.tensor(np.vstack([X_tr, Xm]), device=dev)
        Tb = torch.tensor(np.vstack([np.eye(N_CLS, dtype=np.float32)[y_tr], Tm]), device=dev)
        perm = torch.randperm(len(Xb), device=dev)
        for i in range(0, len(Xb), BS):
            b = perm[i:i + BS]
            loss = -(Tb[b] * F.log_softmax(model(Xb[b]), -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        P_va = F.softmax(model(torch.tensor(X_va, device=dev)), -1).cpu().numpy()
        P_dev = F.softmax(model(torch.tensor(X_dev, dtype=torch.float32, device=dev)), -1).cpu().numpy()
        P_ev = F.softmax(model(torch.tensor(X_ev, dtype=torch.float32, device=dev)), -1).cpu().numpy()
    print(f"{EXT} SEED {SEED} | val top-1: {(P_va.argmax(1) == y_va).mean():.4f}")

    out = pd.DataFrame(P_dev, columns=classes)
    out.insert(0, "split", "dev"); out.insert(0, "label", meta_dev["breed_name"].values)
    out.insert(0, "image_id", meta_dev["image_id"].values)
    out.to_csv(f"{DRV}/pred_RQ1-val-{EXT}-mix-v2-s{SEED}.csv", index=False)

    e = pd.DataFrame(P_ev, columns=classes)
    e.insert(0, "split", ev["split"].values); e.insert(0, "dog_id", ev["dog_id"].values)
    dog = e.groupby(["dog_id", "split"], as_index=False).mean(numeric_only=True)
    dog[classes] = dog[classes].div(dog[classes].sum(1), axis=0)
    dog.to_csv(f"{DRV}/pred_eval_uf-{EXT}-mix-v2-s{SEED}.csv", index=False)
    print(f"   dev 행 {len(out)} | UF 개 {len(dog)} {dog['split'].value_counts().to_dict()}")
print(EXT, "완료")
