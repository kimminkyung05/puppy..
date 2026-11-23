"""
dev 채점 (팀 평가 지표 정의 기준). hold는 사용하지 않는다.

지표
  주 지표: captured@3 (DNA 원래 비율, 재정규화 없음), coverage 함께 보고
  보조: captured_norm@3, overlap, top-1 (Wilson 95%)
  기준: 찍기(train 빈도), 상한(정확 매핑 DNA 큰 3개)
  비교: 찍기 대비 개체 단위 차이 ± 2SE

입력
  pred_eval_uf_v2-sift-mix-s{1,2,3}.csv, pred_eval_uf-{clip,dinov2,dinov3}-mix-v2-s{1,2,3}.csv
  breed_map_UF_draft.csv (status=정확만 사용)
  truth_UF.csv
  index_local_v2.csv (찍기 빈도)
"""
import numpy as np, pandas as pd

DRV = "/content/drive/MyDrive/C_work"
POODLE = {"teddy", "toy_poodle", "miniature_poodle", "standard_poodle"}
EXT_FILES = {
    "sift": "pred_eval_uf_v2-sift-mix-s{}.csv",
    "clip": "pred_eval_uf-clip-mix-v2-s{}.csv",
    "dinov2": "pred_eval_uf-dinov2-mix-v2-s{}.csv",
    "dinov3": "pred_eval_uf-dinov3-mix-v2-s{}.csv",
}

# 매핑 (정확만)
bm = pd.read_csv("/content/breed_map_UF_draft.csv")
bm = bm[bm["status"] == "정확"]
bmap = {r.breed_raw: [t.strip() for t in str(r.tsinghua_classes).split(";")]
        for r in bm.itertuples() if isinstance(r.tsinghua_classes, str)}

def dkey(breed):
    if breed not in bmap:
        return None
    cs = bmap[breed]
    return "Poodle" if any(c in POODLE for c in cs) else cs[0]

# 정답
truth = pd.read_csv("/content/evalset/uf_draft/truth_UF.csv")
truth["key"] = truth["breed_raw"].map(dkey)
truth["p"] = truth["pct"] / 100
T = truth.dropna(subset=["key"]).groupby(["dog_id", "key"])["p"].sum().unstack(fill_value=0.0)

def tie_set(g):
    m = g["pct"].max()
    cls = set()
    for b in g.loc[g["pct"] == m, "breed_raw"]:
        if b in bmap:
            cls |= set(bmap[b])
    return cls if cls else None

TT = {d: tie_set(g) for d, g in truth.groupby("dog_id")}

def merge(p):
    out = {}
    for c, v in p.items():
        k = "Poodle" if c in POODLE else c
        out[k] = out.get(k, 0.0) + v
    return pd.Series(out)

def score_table(P):
    recs = []
    for d, p in P.iterrows():
        pm = merge(p)
        Tr = T.loc[d] if d in T.index else pd.Series(dtype=float)
        Tsum = float(Tr.sum()) if len(Tr) else 0.0
        top3 = pm.nlargest(3).index
        cap = float(Tr.reindex(top3).fillna(0).sum()) if len(Tr) else 0.0
        if Tsum > 0:
            Tn = Tr / Tsum
            capn = float(Tn.reindex(top3).fillna(0).sum())
            keys = pm.index.union(Tn.index)
            ov = float(np.minimum(pm.reindex(keys).fillna(0), Tn.reindex(keys).fillna(0)).sum())
        else:
            capn, ov = np.nan, np.nan
        tset = TT.get(d)
        ok = np.nan if tset is None else float(p.idxmax() in tset)
        recs.append({"dog_id": d, "cap3": cap, "tsum": Tsum, "capn3": capn, "overlap": ov, "top1": ok})
    return pd.DataFrame(recs)

def summarize(df):
    n = len(df)
    ev = df["top1"].dropna()
    m, k = len(ev), int(ev.sum())
    p = k / m if m else np.nan
    if m:
        z = 1.96; den = 1 + z * z / m
        c = (p + z * z / (2 * m)) / den
        h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / den
        ci = (round(c - h, 3), round(c + h, 3))
    else:
        ci = (np.nan, np.nan)
    return {"captured@3": round(df["cap3"].mean(), 4), "coverage": round(df["tsum"].mean(), 4),
            "captured_norm@3": round(df["capn3"].mean(), 4), "overlap": round(df["overlap"].mean(), 4),
            "top1": round(p, 4), "top1_CI": ci, "top1_n": m, "top1_cov": round(m / n, 3)}

def load(e, s):
    x = pd.read_csv(f"{DRV}/" + EXT_FILES[e].format(s))
    return x[x["split"] == "dev"].set_index("dog_id").drop(columns=["split"])

rows = []
for e in EXT_FILES:
    Ps = [load(e, s) for s in (1, 2, 3)]
    cols = Ps[0].columns
    Ps = [P[cols] for P in Ps]
    Pm = (Ps[0] + Ps[1] + Ps[2]) / 3
    seed_sc = [summarize(score_table(P)) for P in Ps]
    rep = summarize(score_table(Pm))
    rows.append({"방법": e, **rep,
                 "seed_captured_range": (min(s["captured@3"] for s in seed_sc), max(s["captured@3"] for s in seed_sc)),
                 "seed_top1_range": (min(s["top1"] for s in seed_sc), max(s["top1"] for s in seed_sc))})
    if e == "sift":
        dev_ids, cls_cols = Pm.index, cols

# 찍기 기준선
tr = pd.read_csv(f"{DRV}/index_local_v2.csv")
freq = tr[(tr["split"] == "train") & tr["keep_127"]]["breed_name"].value_counts(normalize=True)
freq = freq.reindex(cls_cols).fillna(0)
Pch = pd.DataFrame([freq.values] * len(dev_ids), index=dev_ids, columns=cls_cols)
chance = summarize(score_table(Pch))
rows.append({"방법": "찍기(train 빈도)", **chance})

# 상한
ub = np.mean([T.loc[d].nlargest(3).sum() for d in dev_ids if d in T.index])
rows.append({"방법": "상한(captured@3)", "captured@3": round(ub, 4)})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(f"{DRV}/RQ1_dev_metrics_spec.csv", index=False)

# 찍기 대비 개체 단위 차이 ± 2SE
cmp_rows = []
ch = score_table(Pch).set_index("dog_id")
for e in EXT_FILES:
    a, b, c = [load(e, s)[cls_cols] for s in (1, 2, 3)]
    m = score_table((a + b + c) / 3).set_index("dog_id")
    d = (m["cap3"] - ch.loc[m.index, "cap3"]).dropna()
    mean = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = mean - 2 * se, mean + 2 * se
    verdict = "판단 불가" if lo <= 0 <= hi else ("찍기보다 높음" if lo > 0 else "찍기보다 낮음")
    cmp_rows.append({"방법": e, "차이": round(mean, 4), "±2SE": round(2 * se, 4),
                     "하한": round(lo, 4), "상한": round(hi, 4), "n": len(d), "판단": verdict})
print(pd.DataFrame(cmp_rows).to_string(index=False))
