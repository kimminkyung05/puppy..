"""
혼합 학습 (C파트 RQ1): 고정 특징 위에 선형 헤드를 혼합 데이터로 학습.

입력: 특징 행렬(train/dev), 인덱스(breed_name, image_id, split, keep_127)
출력: dev 예측 CSV (image_id, label, split, classes)

설정: AdamW lr 3e-3, wd 1e-4, 20 에폭, 배치 512, 에폭당 혼합 샘플 20,000
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

LR, EPOCHS, BS, N_MIX = 3e-3, 20, 512, 20000
KMAX = 3  # TODO: Colab 셀의 값으로 맞출 것 (확인 필요)
ALPHA = 1.0


def standardize(Xtr, Xo):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xo - mu) / sd, mu, sd


def make_mixed(X, y, n, n_cls, rng):
    """k in [2, KMAX] 품종을 골라 Dirichlet(ALPHA) 가중치로 특징을 섞고, 목표는 가중치 벡터."""
    by_cls = [np.where(y == c)[0] for c in range(n_cls)]
    Xs, Ts = [], []
    for _ in range(n):
        k = rng.integers(2, KMAX + 1)
        cls = rng.choice(n_cls, size=k, replace=False)
        w = rng.dirichlet(np.full(k, ALPHA))
        feat = sum(wi * X[rng.choice(by_cls[c])] for wi, c in zip(w, cls))
        t = np.zeros(n_cls, dtype=np.float32)
        t[cls] = w
        Xs.append(feat)
        Ts.append(t)
    return np.stack(Xs).astype(np.float32), np.stack(Ts)


def train_one(X_tr, y_tr, n_cls, seed, device="cuda"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = nn.Linear(X_tr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    for _ in range(EPOCHS):
        Xm, Tm = make_mixed(X_tr, y_tr, N_MIX, n_cls, rng)
        Xt = torch.tensor(Xm, device=device)
        Tt = torch.tensor(Tm, device=device)
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), BS):
            b = perm[i:i + BS]
            logp = F.log_softmax(model(Xt[b]), -1)
            loss = -(Tt[b] * logp).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def predict(model, X, device="cuda"):
    with torch.no_grad():
        return F.softmax(model(torch.tensor(X, device=device)), -1).cpu().numpy()


def save_dev_predictions(P, index_dev, classes, out_csv):
    out = pd.DataFrame(P, columns=classes)
    out.insert(0, "label", index_dev["breed_name"].values)
    out.insert(0, "split", "dev")
    out.insert(0, "image_id", index_dev["image_id"].values)
    out.to_csv(out_csv, index=False)
