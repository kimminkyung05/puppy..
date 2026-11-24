"""RQ1 dev 결과 해석용 진단: SIFT 가 왜 낮은지, 예측이 어디로 쏠리는지, 어떤 개가 어려운지 (집계값만 저장).

세 가지를 본다.
  1. 특징 진단 (SIFT 가 낮은 이유)
     - 품종 분리도: 같은 학습 사진에서 품종 평균점끼리의 cos, 품종이 설명하는 분산 비율(η²),
       가장 가까운 평균점 정답률(학습 사진 그대로라 낙관값), 비음수 최소제곱(NNLS)에서 정답 품종이 받는 몫
     - 평가셋 이동: UF dev 패널 특징이 학습셋 분포에서 얼마나 벗어나는지 (0 인 칸 비율, 학습셋 평균과의 cos,
       가장 가까운 품종 평균점과의 cos)
  2. 예측 쏠림: 방법 × 추출기마다 최대 확률 평균, 실질 품종 수(exp 엔트로피), 서로 다른 Top-1 수, 가장 흔한 Top-1
     + 불리(bully) 계열 점검: Top-1 이 AmStaff · Staffordshire_bullterrier 인 개 가운데 실제 DNA 에 그 품종이 있는 비율,
       예측 확률 몫과 실제 비율 몫의 평균
  3. 개 난이도: 딥 9개 설정(방법 3 × 딥 추출기 3)에서 개마다 Top-1 을 맞힌 설정 수와 포착@3 평균을
     coverage · 주 품종 비율 · DNA 품종 수 · 불리 혈통 여부로 묶어 집계 (개별 개 값은 저장하지 않는다)

평가셋 특징은 dev 행만 쓴다 (hold 는 최종 한 번만). 딥 학습셋 전체 특징은 로컬에 없어서 분리도 비교는
세 추출기 모두 같은 사진 800장(--common-subset, 8품종 × 100장)에서 하고, SIFT 는 127클래스 학습셋 전체도 따로 낸다.
DINOv3 는 로컬에 학습셋 특징이 없어 분리도 · 이동 비교에서 빠진다 (평가셋 행만 있음).

입력
  {feat_root}/{sift,clip,dinov2}/...          SIFT train · eval_uf_panels, 딥 --common-subset
  {deep_eval_root}/{clip,dinov2,dinov3}/eval_uf_panels   딥 평가셋 특징 (Drive 에서 받은 것)
  {preds}/dev/pred_RQ1-*.csv                  scripts/collect_rq1_preds.py 가 모은 예측
  {uf_dir}/truth_UF.csv, dna_UF.csv, breed_map_UF_draft.csv   정답 · 매핑 (B/ 밖, 저장소에 없음)

출력
  results/feature_separability.csv     추출기 × 사진 집합별 분리도
  results/feature_eval_shift.csv       추출기별 학습셋 → UF dev 이동
  results/pred_concentration.csv       방법 × 추출기 예측 쏠림
  results/bully_bias.csv               방법 × 추출기 불리 계열 Top-1 점검
  results/dog_difficulty.csv           개 묶음별 난이도 (딥 9개 설정)
  figures/sift_diagnostics.png

사용 (B/ 안에서)
  python scripts/feature_diagnostics.py --feat-root ../outputs/features --deep-eval-root ../outputs/features_drive \
      --preds ../outputs/rq1_preds/merged --uf-dir ../data/evalset_uf
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provisional_score_uf as P  # noqa: E402
from plot_results import INK, INK2, SURFACE, style  # noqa: E402
from src.common.features_io import load_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"chinese_rural_dog", "dhole", "African_hunting_dog"}
DEEP = ["clip", "dinov2", "dinov3"]
BACKBONES = ["sift", "clip", "dinov2", "dinov3"]
METHODS = [("A_classifier", "clf", "s"), ("B_fcls", "proto", "r"), ("C_mix", "mix", "s")]
BULLY = {"American_Staffordshire_terrier", "Staffordshire_bullterrier"}
# 매핑 밖까지 넓힌 불리 계열 DNA 표기 (참고값). 정확 매핑 기준은 BULLY 클래스로 판단한다
BULLY_RAW = r"Staffordshire|Pit Bull|Bull Terrier|American Bulldog"
NNLS_N = 2000
SEED = 0
TRAIN_COLOR, EVAL_COLOR = "#2a78d6", "#eb6834"   # dataviz 검증 통과 (light, CVD ΔE 24.7)


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- 1. 특징 진단
def separability(X: np.ndarray, y: np.ndarray, rng) -> dict:
    Xn = unit(X)
    cls = np.unique(y)
    lab = np.searchsorted(cls, y)
    Pm = np.stack([Xn[lab == i].mean(0) for i in range(len(cls))])
    Pn = unit(Pm)
    mu = Xn.mean(0)
    off = (Pn @ Pn.T)[~np.eye(len(cls), dtype=bool)]
    total = ((Xn - mu) ** 2).sum(1).mean()
    between = ((Pm[lab] - mu) ** 2).sum(1).mean()
    within = ((Xn - Pm[lab]) ** 2).sum(1).mean()
    sims = Xn @ Pn.T
    own = sims[np.arange(len(Xn)), lab].copy()
    sims[np.arange(len(Xn)), lab] = -np.inf
    sub = rng.choice(len(Xn), min(NNLS_N, len(Xn)), replace=False)
    w = np.array([nnls(Pn.T, x)[0] for x in Xn[sub]])
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return dict(images=len(X), classes=len(cls), proto_cos_mean=off.mean(), proto_cos_min=off.min(),
                eta2_between_over_total=between / total, within_over_between=within / between,
                nearest_proto_acc_insample=float(((Xn @ Pn.T).argmax(1) == lab).mean()),
                own_beats_other_protos=float((own > sims.max(1)).mean()),
                nnls_true_share=float(w[np.arange(len(sub)), lab[sub]].mean()),
                nnls_breeds_over_1pct=float((w > 0.01).sum(1).mean()), chance_top1=1 / len(cls))


def train_set(root: Path, bb: str, subset: str, keep_ids=None):
    ids, X, idx = load_features(str(root), bb, subset)
    m = ~idx["label"].isin(EXCLUDED).to_numpy()
    if keep_ids is not None:
        m &= idx["image_id"].astype(str).isin(keep_ids).to_numpy()
    return X[m], idx["label"].to_numpy()[m]


def dev_eval(root: Path, bb: str) -> np.ndarray:
    _, X, idx = load_features(str(root), bb, "eval_uf_panels")
    return X[(idx["split"] == "dev").to_numpy()]


def shift(Xtr, ytr, Xev) -> dict:
    Tn, En = unit(Xtr), unit(Xev)
    mu = unit(Tn.mean(0))
    cls = np.unique(ytr)
    Pn = unit(np.stack([Tn[ytr == c].mean(0) for c in cls]))
    return dict(train_images=len(Xtr), eval_dev_images=len(Xev),
                zero_frac_train=float((Xtr == 0).mean()), zero_frac_eval=float((Xev == 0).mean()),
                cos_to_train_mean_train=float((Tn @ mu).mean()), cos_to_train_mean_eval=float((En @ mu).mean()),
                max_proto_cos_train=float((Tn @ Pn.T).max(1).mean()), max_proto_cos_eval=float((En @ Pn.T).max(1).mean()))


def feature_checks(a) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    common_ids = set(pd.read_csv(a.feat_root / "clip" / a.common_subset / "index.csv")["image_id"].astype(str))
    sep, sh = [], []
    common = {"sift": train_set(a.feat_root, "sift", "train", common_ids)}
    for bb in ("clip", "dinov2"):
        if (a.feat_root / bb / a.common_subset).exists():
            common[bb] = train_set(a.feat_root, bb, a.common_subset, common_ids)
    sizes = {len(v[0]) for v in common.values()}
    if len(sizes) != 1:
        raise SystemExit(f"--common-subset 사진 수가 추출기마다 다름: { {k: len(v[0]) for k, v in common.items()} }")
    for bb, (X, y) in common.items():
        sep.append(dict(backbone=bb, image_set=f"{a.common_subset} (같은 사진)", **separability(X, y, rng)))
        eval_root = a.feat_root if bb == "sift" else a.deep_eval_root
        sh.append(dict(backbone=bb, reference=f"{a.common_subset} (같은 사진)", **shift(X, y, dev_eval(eval_root, bb))))
    Xs, ys = train_set(a.feat_root, "sift", "train")
    sep.append(dict(backbone="sift", image_set="train 전체 (127클래스)", **separability(Xs, ys, rng)))
    sh.append(dict(backbone="sift", reference="train 전체 (127클래스)", **shift(Xs, ys, dev_eval(a.feat_root, "sift"))))
    return pd.DataFrame(sep), pd.DataFrame(sh)


# ---------------------------------------------------------------- 2 · 3. 예측
def load_probs(pred_dir: Path) -> dict:
    out = {}
    for name, tag, rep in METHODS:
        for bb in BACKBONES:
            fs = [pred_dir / f"pred_RQ1-{bb}-{tag}-{rep}{n}.csv" for n in (1, 2, 3)]
            if not all(f.exists() for f in fs):
                continue
            frames = []
            for f in fs:
                d = pd.read_csv(f)
                frames.append(d[d["split"] == "dev"].set_index("dog_id").drop(columns="split"))
            out[(name, bb)] = P.mean_prob(frames, f"{name} {bb}")
    if not out:
        raise SystemExit(f"예측 파일 없음: {pred_dir}")
    return out


def concentration(probs: dict) -> pd.DataFrame:
    rows = []
    for (meth, bb), p in probs.items():
        v = p.to_numpy()
        ent = -(np.where(v > 0, v * np.log(np.where(v > 0, v, 1)), 0)).sum(1)
        top = p.idxmax(axis=1).value_counts()
        rows.append(dict(method=meth, backbone=bb, dogs=len(p), max_prob_mean=v.max(1).mean(),
                         effective_breeds=np.exp(ent).mean(), distinct_top1=len(top),
                         most_common_top1=top.index[0], most_common_top1_dogs=int(top.iloc[0])))
    return pd.DataFrame(rows)


def dog_table(a, dogs) -> pd.DataFrame:
    """개별 개 속성 (메모리 안에서만 쓰고 저장하지 않는다)."""
    t = pd.read_csv(a.uf_dir / "truth_UF.csv", encoding="utf-8-sig")
    dna = pd.read_csv(a.uf_dir / "dna_UF.csv", encoding="utf-8-sig")
    rows = []
    for d in dogs:
        g = dna[(dna["dog_id"] == d) & dna["pct"].notna()]
        tg = t[(t["dog_id"] == d) & (t["status"] == P.EXACT)]
        bully = tg[tg["target_classes"].isin(BULLY)]["pct"].astype(float).sum() / 100
        rows.append(dict(dog_id=d, coverage=float(t.loc[t["dog_id"] == d, "coverage"].iloc[0]),
                         dominant_pct=float(g["pct"].max()), n_breeds=len(g), bully_share=bully,
                         bully_raw=bool(g["breed_raw"].str.contains(BULLY_RAW, case=False).any())))
    return pd.DataFrame(rows).set_index("dog_id")


def bully_bias(probs: dict, info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = (info["bully_share"] > 0)
    for (meth, bb), p in probs.items():
        p = p.loc[info.index]
        top = p.idxmax(axis=1)
        pick = top.isin(BULLY)
        cols = [c for c in p.columns if c in BULLY]
        rows.append(dict(method=meth, backbone=bb, dogs=len(p), dogs_with_bully_dna_exact=int(base.sum()),
                         dogs_with_bully_dna_broad=int(info["bully_raw"].sum()),
                         top1_bully_dogs=int(pick.sum()),
                         top1_bully_and_dna_exact=int((pick & base).sum()),
                         top1_bully_and_dna_broad=int((pick & info["bully_raw"]).sum()),
                         pred_bully_prob_mean=float(p[cols].sum(axis=1).mean()),
                         true_bully_share_mean=float(info["bully_share"].mean())))
    return pd.DataFrame(rows)


def difficulty(a, probs: dict, info: pd.DataFrame) -> pd.DataFrame:
    c2b = P.breed_of_class(a.uf_dir / "breed_map_UF_draft.csv", {P.EXACT})
    orig, _, _ = P.load_truth(a.uf_dir / "truth_UF.csv", False)
    dna = pd.read_csv(a.uf_dir / "dna_UF.csv", encoding="utf-8-sig")
    bmap = pd.read_csv(a.uf_dir / "breed_map_UF_draft.csv", encoding="utf-8-sig")
    dogs = list(info.index)
    deep = {k: v.loc[dogs] for k, v in probs.items() if k[1] in DEEP}
    if len(deep) != 9:
        raise SystemExit(f"딥 설정이 {len(deep)}개뿐 (9개 필요)")
    classes = set(next(iter(deep.values())).columns)
    answers = P.top1_answers(dna, bmap, {P.EXACT}, classes, dogs)
    hits = pd.Series(0, index=dogs)
    cap = pd.Series(0.0, index=dogs)
    for p in deep.values():
        top = p.idxmax(axis=1)
        hits += pd.Series({d: int(top[d] in answers[d]) for d in dogs})
        cap += P.captured(P.merge_to_breeds(p, c2b), orig).reindex(dogs)
    cap /= len(deep)
    ceil = P.ceiling(orig, dogs)
    df = info.assign(scorable=[bool(answers[d]) for d in dogs], hit_configs=hits, captured=cap, ceiling=ceil)
    groups = [("전체", df.index == df.index),
              ("coverage < 0.5", df["coverage"] < 0.5), ("coverage ≥ 0.5", df["coverage"] >= 0.5),
              ("주 품종 비율 < 40%", df["dominant_pct"] < 40), ("주 품종 비율 ≥ 40%", df["dominant_pct"] >= 40),
              ("DNA 품종 ≤ 3", df["n_breeds"] <= 3), ("DNA 품종 4", df["n_breeds"] >= 4),
              ("불리 혈통 있음 (정확 매핑)", df["bully_share"] > 0), ("불리 혈통 없음 (정확 매핑)", df["bully_share"] == 0)]
    rows = []
    for name, m in groups:
        g = df[np.asarray(m)]
        s = g[g["scorable"]]
        rows.append(dict(group=name, dogs=len(g), top1_scorable=len(s),
                         hit_by_any_config=int((s["hit_configs"] > 0).sum()),
                         hit_by_none=int((s["hit_configs"] == 0).sum()),
                         hit_configs_mean=s["hit_configs"].mean() if len(s) else np.nan,
                         captured3_mean=g["captured"].mean(), ceiling_mean=g["ceiling"].mean(),
                         captured_over_ceiling=g["captured"].sum() / g["ceiling"].sum() if g["ceiling"].sum() else np.nan,
                         configs=len(deep)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림
def figure(sep: pd.DataFrame, sh: pd.DataFrame, out: Path) -> None:
    s = sep[sep["image_set"].str.contains("같은 사진")]
    h = sh[sh["reference"].str.contains("같은 사진")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    x = np.arange(len(s))
    a1.bar(x, s["eta2_between_over_total"], width=0.55, color=INK2, edgecolor=SURFACE, linewidth=2)
    for xi, v in zip(x, s["eta2_between_over_total"]):
        a1.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=10)
    style(a1, max(0.5, s["eta2_between_over_total"].max() * 1.2), "between-breed / total variance (η²)")
    a1.set_xticks(x, [b.upper() if b == "sift" else b for b in s["backbone"]])
    a1.set_title(f"Breed separability ({int(s['images'].iloc[0])} same train images, "
                 f"{int(s['classes'].iloc[0])} breeds)", color=INK, fontsize=11)
    x = np.arange(len(h))
    w = 0.36
    for off, col, color, lab in [(-w / 2, "cos_to_train_mean_train", TRAIN_COLOR, "train images"),
                                 (w / 2, "cos_to_train_mean_eval", EVAL_COLOR, "UF dev images")]:
        a2.bar(x + off, h[col], width=w, color=color, edgecolor=SURFACE, linewidth=2, label=lab)
        for xi, v in zip(x + off, h[col]):
            a2.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    style(a2, 1.05, "cos(image, train mean)")
    a2.set_xticks(x, [b.upper() if b == "sift" else b for b in h["backbone"]])
    a2.set_title("Shift from train to UF dev", color=INK, fontsize=11)
    a2.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper right")
    fig.suptitle("Why SIFT fails: weak breed signal and a large train → UF shift", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feat-root", type=Path, required=True)
    ap.add_argument("--deep-eval-root", type=Path, required=True)
    ap.add_argument("--common-subset", default="train_parents_n100")
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--uf-dir", type=Path, required=True)
    a = ap.parse_args()
    res, figs = ROOT / "results", ROOT / "figures"

    sep, sh = feature_checks(a)
    probs = load_probs(a.preds / "dev")
    dogs = sorted(next(iter(probs.values())).index)
    info = dog_table(a, dogs)
    conc = concentration(probs)
    bully = bully_bias(probs, info)
    diff = difficulty(a, probs, info)

    for df, name in [(sep, "feature_separability"), (sh, "feature_eval_shift"), (conc, "pred_concentration"),
                     (bully, "bully_bias"), (diff, "dog_difficulty")]:
        df.round(4).to_csv(res / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"\n== {name}\n{df.round(3).to_string(index=False)}")
    figure(sep, sh, figs / "sift_diagnostics.png")
    print(f"\n저장: results/*.csv 5개, figures/sift_diagnostics.png")


if __name__ == "__main__":
    main()
