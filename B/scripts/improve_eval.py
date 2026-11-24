"""성능 개선 실험 (dev, 참고). improve_extract.py 가 뽑은 특징으로 미리 정한 후보만 비교한다.

후보 (실행 전에 고정. dev 점수를 보고 늘리지 않는다)
  풀이   A_lr   분류기. --a-impl sklearn(기본, 로컬): L2 정규화 + 다항 로지스틱 회귀, 규제 C 는 val_sub(UF 아님) 순종 Top-1 로
                {1, 10, 100} 중 고름. --a-impl torch(Colab, 학습셋 전체용): A 의 실제 학습법을 옮긴 것(A/A/docs/A_METHODOLOGY_FINAL.md
                2절) — L2 정규화 + 선형층, Adam 1e-3, 배치 1,024, 최대 30에폭, L2 규제 1e-4, 클래스 균형 가중치,
                학습셋 10%로 조기 종료(5에폭 동안 1e-4 이상 안 줄면 멈춤)
         B_fcls 품종 평균점 + FCLS (평균점 섞기, L2 없음, 부분 표본 80% × 3회 평균)
         C_mix  z-score + 가짜 믹스로 학습한 선형층 (섞기 학습: k 2~4, Dirichlet α=1, 20,000개/에폭, 20에폭, AdamW 3e-3)
  특징   cls | mean(패치 평균) | cat(cls 와 mean 을 각각 L2 정규화해 이어 붙임)
  UF 특징은 공식 추출본(features_drive/{추출기}/eval_uf_panels, dev 행만)을 그대로 쓴다. 시험 시 증강은 하지 않는다

평가
  대리(UF 안 씀)  val_sub 순종 Top-1, val_sub 특징으로 만든 가짜 믹스(k 2~3, Dirichlet α=1, 1,000개, 시드 0)의 포착@3
  dev            UF dev 20마리 포착@3 원래 비율 (provisional_score_uf.py 함수, METRICS_FINAL.md)
  비교           풀이마다 기준(cls) 대비 개체 단위 짝 차이 ± 2SE
  정직한 추정     (1) 대리 지표로 고른 설정의 dev 점수 (선택에 UF 를 안 씀)
                 (2) 한 마리씩 빼고 19마리로 고른 설정을 뺀 개에 적용 (leave-one-dog-out)

입력: {feat}/{추출기}/train_sub.npz, val_sub.npz
      로컬: improve_extract.py 가 학습셋 일부(품종당 30장)를 직접 뽑은 것
      Colab: improve_from_shards.py 가 Drive 공용 특징에서 학습셋 전체를 옮긴 것
클래스 순서: 학습 라벨을 정렬한 순서(공용 규칙). hold 는 쓰지 않는다.

출력: --out-csv (기본 results/improve_dev.csv, 집계값만)
사용 (B/ 안에서)
  로컬:  python scripts/improve_eval.py --feat ../outputs/improve --backbones dinov2 clip dinov3
  Colab: python scripts/improve_eval.py --feat /content/improve_full --uf-feat {DRIVE}/features              --uf-dir {DRIVE}/evalset/uf_draft --dna /content/uf_private/dna_UF.csv --a-impl torch              --out-csv {DRIVE}/results/improve_dev_full.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provisional_score_uf as Psc  # noqa: E402
from compare_rq1_dev import Scorer  # noqa: E402
from src.common.features_io import load_features  # noqa: E402
from src.rq1.prototype_fcls import build_prototypes, fcls  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOLS, SOLVERS = ["cls", "mean", "cat"], ["A_lr", "B_fcls", "C_mix"]


def l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def feats(z, pool):
    if pool == "cat":
        return np.hstack([l2(z["cls"]), l2(z["mean"])])
    return z[pool].astype(np.float64)


def synth_mix(X, y, n_cls, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = [np.where(y == c)[0] for c in range(n_cls)]
    by = [b for b in by]
    Xs, Ts = [], []
    for _ in range(n):
        k = rng.integers(2, 4)
        cl = rng.choice([c for c in range(n_cls) if len(by[c])], k, replace=False)
        r = rng.dirichlet(np.ones(k))
        Xs.append(sum(w * X[rng.choice(by[c])] for c, w in zip(cl, r)))
        t = np.zeros(n_cls); t[cl] = r; Ts.append(t)
    return np.array(Xs), np.array(Ts)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_a_torch(X, y, n_cls, seed=1):
    """A 의 분류기 학습법 (L2 정규화된 입력). 학습셋 10%로 조기 종료."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X)); n_val = max(1, len(X) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    w = len(y) / (n_cls * np.bincount(y, minlength=n_cls).clip(1))           # 클래스 균형 가중치
    Xt = torch.tensor(X[ti], dtype=torch.float32, device=DEVICE); yt = torch.tensor(y[ti], device=DEVICE)
    Xv = torch.tensor(X[vi], dtype=torch.float32, device=DEVICE); yv = torch.tensor(y[vi], device=DEVICE)
    W = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    m = nn.Linear(X.shape[1], n_cls).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bad = float("inf"), 0
    for _ in range(30):
        m.train(); order = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), 1024):
            b = order[s:s + 1024]
            loss = F.cross_entropy(m(Xt[b]), yt[b], weight=W)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(m(Xv), yv, weight=W))
        if vl < best - 1e-4:
            best, bad = vl, 0
        else:
            bad += 1
            if bad >= 5:
                break
    return m.cpu().eval()


class Solver:
    def __init__(self, kind, Xtr, ytr, n_cls, Xval, yval, seed=1, a_impl="sklearn"):
        self.kind, self.n, self.a_impl = kind, n_cls, a_impl
        if kind == "A_lr" and a_impl == "torch":
            self.C, self.m = "torch(A 방식)", train_a_torch(l2(Xtr), ytr, n_cls, seed)
        elif kind == "A_lr":
            best = None
            for C in (1, 10, 100):
                m = LogisticRegression(C=C, max_iter=3000).fit(l2(Xtr), ytr)
                acc = (m.predict(l2(Xval)) == yval).mean()
                if best is None or acc > best[0]:
                    best = (acc, C, m)
            self.C, self.m = best[1], best[2]
        elif kind == "B_fcls":
            classes = list(range(n_cls))
            self.P = [build_prototypes(Xtr, ytr, classes, 0.8, seed=r) for r in (1, 2, 3)]
        elif kind == "C_mix":
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Z = ((Xtr - self.mu) / self.sd).astype(np.float32)
            by = [np.where(ytr == c)[0] for c in range(n_cls)]
            self.m = nn.Linear(Z.shape[1], n_cls).to(DEVICE)
            opt = torch.optim.AdamW(self.m.parameters(), lr=3e-3, weight_decay=1e-4)
            eye = np.eye(n_cls, dtype=np.float32)
            n_mix, kmax = 20000, 4
            for _ in range(20):
                # 벡터로 만든 가짜 믹스: 품종 kmax 개를 겹치지 않게 뽑고 앞 k 개만 쓴다 (k 는 2~4 고르게)
                k = rng.integers(2, kmax + 1, n_mix)
                cl = np.argsort(rng.random((n_mix, n_cls)), 1)[:, :kmax]
                g = rng.gamma(1.0, size=(n_mix, kmax)) * (np.arange(kmax)[None] < k[:, None])
                r = g / g.sum(1, keepdims=True)                                   # Dirichlet(1) on k slots
                pick = np.array([[by[c][rng.integers(len(by[c]))] for c in row] for row in cl])
                Xm = (r[:, :, None] * Z[pick]).sum(1)
                Tm = np.zeros((n_mix, n_cls), np.float32)
                np.add.at(Tm, (np.repeat(np.arange(n_mix), kmax), cl.ravel()), r.ravel())
                Xb = torch.tensor(np.vstack([Z, Xm.astype(np.float32)]), device=DEVICE)
                Tb = torch.tensor(np.vstack([eye[ytr], Tm]), device=DEVICE)
                perm = torch.randperm(len(Xb), device=DEVICE)
                for s in range(0, len(Xb), 512):
                    b = perm[s:s + 512]
                    loss = -(Tb[b] * F.log_softmax(self.m(Xb[b]), -1)).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            self.m = self.m.cpu().eval()

    def predict(self, X):
        if self.kind == "A_lr" and self.a_impl == "torch":
            with torch.no_grad():
                return F.softmax(self.m(torch.tensor(l2(X), dtype=torch.float32)), -1).numpy()
        if self.kind == "A_lr":
            return self.m.predict_proba(l2(X))
        if self.kind == "B_fcls":
            out = np.zeros((len(X), self.n))
            for P in self.P:
                delta = 10.0 * float(np.abs(P).max())
                out += np.array([fcls(P, x, delta) for x in X.astype(np.float64)])
            return out / len(self.P)
        with torch.no_grad():
            Z = torch.tensor(((X - self.mu) / self.sd).astype(np.float32))
            return F.softmax(self.m(Z), -1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=Path, required=True)
    ap.add_argument("--backbones", nargs="+", default=["dinov2", "clip", "dinov3"])
    ap.add_argument("--uf-dir", type=Path, default=ROOT.parent / "data" / "evalset_uf")
    ap.add_argument("--uf-feat", type=Path, default=ROOT.parent / "outputs" / "features_drive",
                    help="공식 UF 특징 루트 ({추출기}/eval_uf_panels)")
    ap.add_argument("--dna", type=Path, default=None, help="Top-1 정답용 dna_UF.csv (기본: uf-dir 안)")
    ap.add_argument("--a-impl", choices=["sklearn", "torch"], default="sklearn")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "results" / "improve_dev.csv")
    args = ap.parse_args()
    print(f"장치: {DEVICE}, 분류기 구현: {args.a_impl}", flush=True)

    classes = None
    rows, per_dog = [], {}
    for bb in args.backbones:
        d = args.feat / bb
        if not all((d / f"{p}.npz").exists() for p in ("train_sub", "val_sub")):
            print(f"[건너뜀] {bb}: 특징이 아직 없음"); continue
        tr, va = (np.load(d / f"{p}.npz") for p in ("train_sub", "val_sub"))
        if classes is None:                                       # 공용 클래스 순서 = 학습 라벨 정렬 순서
            classes = sorted(set(tr["label"].tolist()))
            if len(classes) != 127:
                raise SystemExit(f"학습 라벨의 클래스 수가 {len(classes)}개 (127이어야 함)")
            cid = {c: i for i, c in enumerate(classes)}
        print(f"[{bb}] 학습 {len(tr['label'])}장, val {len(va['label'])}장", flush=True)
        ids_uf, cls_uf, idx_uf = load_features(str(args.uf_feat), bb, "eval_uf_panels", "cls")
        _, mean_uf, _ = load_features(str(args.uf_feat), bb, "eval_uf_panels", "mean")
        dev = (idx_uf["split"] == "dev").to_numpy()                       # hold 는 쓰지 않는다
        uf = {"cls": cls_uf[dev], "mean": mean_uf[dev]}
        uf_dog = idx_uf.loc[dev, "dog_id"].to_numpy()
        ytr = np.array([cid[c] for c in tr["label"]]); yva = np.array([cid[c] for c in va["label"]])
        sc = None
        for pool in POOLS:
            Xtr, Xva, Xuf = feats(tr, pool), feats(va, pool), feats(uf, pool)
            Xmix, Tmix = synth_mix(Xva, yva, len(classes))
            for kind in SOLVERS:
                s = Solver(kind, Xtr, ytr, len(classes), Xva, yva, a_impl=args.a_impl)
                pv = s.predict(Xva)
                proxy_top1 = float((pv.argmax(1) == yva).mean())
                pm = s.predict(Xmix)
                top3 = np.argsort(-pm, 1)[:, :3]
                proxy_mix = float(np.mean([Tmix[i, top3[i]].sum() for i in range(len(pm))]))
                img = pd.DataFrame(s.predict(Xuf), columns=classes)
                img["dog_id"] = uf_dog
                dog = img.groupby("dog_id").mean()                                   # 얼굴 · 전신 평균
                dog = dog.div(dog.sum(axis=1), axis=0)
                if sc is None:
                    sc = Scorer(args.uf_dir, classes, list(dog.index), args.dna)
                r = sc(dog)
                per_dog[(bb, kind, pool)] = r["co"]
                rows.append(dict(backbone=bb, solver=kind, pool=pool, proxy_purebred_top1=proxy_top1,
                                 proxy_mix_captured3=proxy_mix, dev_captured3=float(r["co"].mean()),
                                 dev_top1=f"{len(r['hits'])}/{r['n']}",
                                 note=f"C={s.C}" if kind == "A_lr" else ""))
                print(f"{bb} {pool:4s} {kind:6s} 대리 Top-1 {proxy_top1:.3f} 대리 믹스 {proxy_mix:.3f} "
                      f"| dev 포착@3 {r['co'].mean():.3f}", flush=True)
    if not rows:
        return
    res = pd.DataFrame(rows)
    # 기준(cls, orig) 대비 짝 차이
    diffs = []
    for _, r in res.iterrows():
        base = per_dog[(r.backbone, r.solver, "cls")]
        d_, se2, v = Psc.paired(per_dog[(r.backbone, r.solver, r.pool)], base)
        diffs.append(f"{d_:+.3f}±{se2:.3f} {v}")
    res["vs_cls"] = diffs
    # 정직한 추정
    M = pd.DataFrame({k: v for k, v in per_dog.items()})
    extra = []
    for bb in res.backbone.unique():
        sub = res[res.backbone == bb]
        pick = sub.sort_values(["proxy_mix_captured3", "proxy_purebred_top1"], ascending=False).iloc[0]
        extra.append(dict(backbone=bb, solver="(대리 지표로 고름)", pool=pick.pool,
                          dev_captured3=pick.dev_captured3, note=f"고른 풀이 {pick.solver}"))
        cols = [k for k in per_dog if k[0] == bb]
        Mb = M[cols]
        loo = [Mb.loc[d, Mb.drop(index=d).mean().idxmax()] for d in Mb.index]
        extra.append(dict(backbone=bb, solver="(19마리로 고름, LOO)", dev_captured3=float(np.mean(loo)),
                          note=f"후보 {len(cols)}개"))
    res = pd.concat([res, pd.DataFrame(extra)], ignore_index=True)
    out = args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
