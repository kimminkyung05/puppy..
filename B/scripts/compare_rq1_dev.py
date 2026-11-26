"""RQ1 비교 (기본 dev, 최종 때 --split hold 한 번): A 분류기 · B 평균점 섞기 · C 섞기 학습을 같은 예측 폴더 · 같은 채점 코드로 비교한다 (집계값만 저장).

입력
  {preds}/dev/pred_RQ1-{bb}-{clf|proto|mix}-{s|r}{n}.csv   scripts/collect_rq1_preds.py 가 모은 공용 형식 예측 (dev 행만)
  {uf_dir}/truth_UF.csv, breed_map_UF_draft.csv, dna_UF.csv  정답 · 매핑 (B/ 밖, 저장소에 없음)
  --train-index                                              학습셋 index.csv (찍기 기준: 클래스 빈도)
  results/dev_human_baseline.csv                             사람 기준 (score_human_uf.py)
  순종 Top-1 (val dev 절반 2,540장):
    A  {a_dir}/RQ1-{bb}-clf-s{n}/validation_predictions.csv 를 --val-split 의 dev 행으로 다시 센다
    B  results/model_summary.csv (L2 없음). 거기 없는 추출기(SIFT)는 --b-val-dir 의 pred_RQ1-val-*-proto-r*.csv 에서 센다
    C  ../C/docs/experiment_log_RQ1.csv 의 mix · purebred 행

채점은 provisional_score_uf.py 의 함수를 그대로 쓴다. 정의는 docs/METRICS_FINAL.md (정확 매핑만).
  대표값 = 시드(반복) 1~3 확률 평균 후 채점. 찍기 대비 · 방법 간 비교 = 개 단위 짝 차이 평균 ± 2SE.

출력 (개별 개 값은 저장하지 않는다)
  results/dev_rq1_comparison.csv     방법 × 추출기 지표 + 사람 · 찍기 · 상한
  results/dev_rq1_paired.csv         방법 짝(A-B, A-C, B-C) 차이 · 개별 점수 상관 · Top-1 맞힌 개 겹침
  results/dev_rq1_equal_weight.csv   같은 가중치 결합 참고값 (가중치를 고르지 않음. 설정 선택에 쓰지 않는다)
  results/purebred_top1_same_set.csv 순종 Top-1 (세 방법 모두 val dev 절반)
  figures/dev_rq1_with_human.png

사용 (B/ 안에서)
  python scripts/compare_rq1_dev.py --preds ../outputs/rq1_preds/merged --uf-dir ../data/evalset_uf \
      --train-index ../outputs/features_drive/clip/train/index.csv --a-dir ../outputs/rq1_preds/A \
      --val-split ../data/evalset/val_split_seed1.csv --b-val-dir ../outputs/rq1_preds/B/dev
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provisional_score_uf as P  # noqa: E402
from plot_results import INK, INK2, SURFACE, style, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["clip", "dinov2", "dinov3", "sift"]
DEEP = ["clip", "dinov2", "dinov3"]
# (키, 파일 이름의 방법, 반복 접두어, 표에 쓰는 이름, 색)
METHODS = [("A", "clf", "s", "A_classifier", "#2a78d6"),
           ("B", "proto", "r", "B_fcls", "#eb6834"),
           ("C", "mix", "s", "C_mix", "#1b9e77")]
HUMAN_COLOR = "#7a3fb0"


def load_runs(pred_dir: Path, split: str = "dev"):
    """(방법 키, 추출기) → [반복별 확률 표 (개 × 클래스)]."""
    runs = {}
    for key, tag, rep, _, _ in METHODS:
        for bb in BACKBONES:
            frames = []
            for n in (1, 2, 3):
                f = pred_dir / f"pred_RQ1-{bb}-{tag}-{rep}{n}.csv"
                if f.exists():
                    d = pd.read_csv(f)
                    frames.append(d[d["split"] == split].set_index("dog_id").drop(columns="split"))
            if frames:
                if len(frames) != 3:
                    raise SystemExit(f"{key} {bb}: 반복 파일이 {len(frames)}개뿐 (3개 필요)")
                runs[(key, bb)] = frames
    return runs


class Scorer:
    def __init__(self, uf_dir: Path, classes, dogs, dna_csv: Path | None = None):
        """dna_csv: Top-1 정답용 dna_UF.csv. 기본은 uf_dir 안 (Colab 은 /content/uf_private 에 따로 둔다)"""
        self.c2b = P.breed_of_class(uf_dir / "breed_map_UF_draft.csv", {P.EXACT})
        self.orig, self.norm, _ = P.load_truth(uf_dir / "truth_UF.csv", False)
        dna = pd.read_csv(dna_csv or uf_dir / "dna_UF.csv", encoding="utf-8-sig")
        bmap = pd.read_csv(uf_dir / "breed_map_UF_draft.csv", encoding="utf-8-sig")
        self.dogs = [d for d in dogs if d in self.orig]
        self.use_norm = [d for d in self.dogs if self.norm[d]]
        self.answers = P.top1_answers(dna, bmap, {P.EXACT}, set(classes), self.dogs)

    def __call__(self, prob: pd.DataFrame) -> dict:
        prob = prob.loc[self.dogs]
        m = P.merge_to_breeds(prob, self.c2b)
        top = prob.idxmax(axis=1)
        scorable = [d for d in self.dogs if self.answers[d]]
        return dict(co=P.captured(m, self.orig).reindex(self.dogs),
                    cn=P.captured(m, self.norm).reindex(self.use_norm),
                    on=P.overlap(m, self.norm).reindex(self.use_norm),
                    hits={d for d in scorable if top[d] in self.answers[d]}, n=len(scorable), top=top)


def purebred(a_dir: Path | None, val_split: Path | None, b_val_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    if a_dir and val_split and val_split.exists():
        vs = pd.read_csv(val_split)
        full = set(vs.loc[vs["split"] == "dev", "image_id"])
        short = {i.split("/")[-1] for i in full}          # A 의 SIFT 파일은 폴더 없이 사진 id 만 적혀 있다
        for bb in BACKBONES:
            accs, n = [], 0
            for s in (1, 2, 3):
                f = a_dir / f"RQ1-{bb}-clf-s{s}" / "validation_predictions.csv"
                if not f.exists():
                    break
                v = pd.read_csv(f, encoding="utf-8-sig")
                v = v[v["image_id"].isin(full) | v["image_id"].isin(short)]
                accs.append(float((v["true_label"] == v["pred_label"]).mean())); n = len(v)
            if len(accs) == 3:
                rows.append(dict(method="A_classifier", backbone=bb, images=n, top1=np.mean(accs),
                                 rep_min=min(accs), rep_max=max(accs)))
    b = pd.read_csv(ROOT / "results" / "model_summary.csv")
    for _, r in b[b["model_name"].isin(DEEP)].iterrows():
        rows.append(dict(method="B_fcls", backbone=r["model_name"], images=int(r["images"]), top1=r["top1_accuracy"],
                         rep_min=r["top1_rep_min"], rep_max=r["top1_rep_max"]))
    # model_summary.csv 에 없는 추출기(SIFT, 로컬 실행)는 예측 파일에서 직접 센다.
    # provisional_score_purebred.py 와 같은 방식: 대표값 = 반복 확률 평균 후 Top-1, 범위 = 반복별 Top-1
    for bb in BACKBONES:
        if bb in set(b["model_name"]) or not b_val_dir:
            continue
        fs = [b_val_dir / f"pred_RQ1-val-{bb}-proto-r{n}.csv" for n in (1, 2, 3)]
        if not all(f.exists() for f in fs):
            continue
        frames = [pd.read_csv(f).set_index("image_id") for f in fs]
        cls = [c for c in frames[0].columns if c not in ("label", "split")]
        lab = frames[0]["label"]
        each = [float((f[cls].idxmax(axis=1) == lab).mean()) for f in frames]
        avg = sum(f[cls] for f in frames) / 3
        rows.append(dict(method="B_fcls", backbone=bb, images=len(lab), top1=float((avg.idxmax(axis=1) == lab).mean()),
                         rep_min=min(each), rep_max=max(each)))
    c_log = ROOT.parent / "C" / "docs" / "experiment_log_RQ1.csv"
    if c_log.exists():
        c = pd.read_csv(c_log)
        for (meth, bb), g in c.groupby(["method", "extractor"]):
            name = {"mix": "C_mix", "purebred": "C_single_label_control"}[meth]
            rows.append(dict(method=name, backbone=bb, images=2540, top1=g["dev_top1"].mean(),
                             rep_min=g["dev_top1"].min(), rep_max=g["dev_top1"].max()))
    out = pd.DataFrame(rows)
    if len(out) and (out["images"] != 2540).any():
        raise SystemExit(f"순종 사진 수가 2,540 이 아닌 행이 있음:\n{out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True, help="collect_rq1_preds.py 의 --out")
    ap.add_argument("--uf-dir", type=Path, required=True)
    ap.add_argument("--train-index", type=Path, required=True)
    ap.add_argument("--a-dir", type=Path, default=None, help="A 의 preds 폴더 (순종 Top-1 을 val dev 절반으로 다시 셀 때)")
    ap.add_argument("--val-split", type=Path, default=None)
    ap.add_argument("--b-val-dir", type=Path, default=None,
                    help="B 순종 예측(pred_RQ1-val-*-proto-r*.csv) 폴더. model_summary.csv 에 없는 추출기만 여기서 센다")
    ap.add_argument("--split", default="dev", choices=["dev", "hold"],
                    help="hold 는 최종 채점 때 한 번만 (collect_rq1_preds.py --split hold 를 먼저)")
    args = ap.parse_args()
    sp = args.split

    runs = load_runs(args.preds / sp, sp)
    if not runs:
        raise SystemExit(f"{args.preds / sp} 에 예측이 없음")
    first = next(iter(runs.values()))[0]
    classes = list(first.columns)
    for k, fs in runs.items():
        for f in fs:
            if list(f.columns) != classes or set(f.index) != set(first.index):
                raise SystemExit(f"{k}: 클래스 순서나 개 목록이 다름")
    sc = Scorer(args.uf_dir, classes, list(first.index))
    freq = pd.read_csv(args.train_index)["label"].value_counts(normalize=True)
    chance = sc(pd.DataFrame([freq.reindex(classes).fillna(0).to_numpy()] * len(sc.dogs), columns=classes,
                             index=sc.dogs))

    ens = {k: sum(fs) / 3 for k, fs in runs.items()}
    S = {k: sc(p) for k, p in ens.items()}
    rows = []
    for key, _, _, name, _ in METHODS:
        for bb in BACKBONES:
            if (key, bb) not in S:
                rows.append(dict(method=name, backbone=bb, note="예측 없음"))
                continue
            s = S[(key, bb)]
            reps = [float(sc(f)["co"].mean()) for f in runs[(key, bb)]]
            d, se2, v = P.paired(s["co"], chance["co"])
            rows.append(dict(method=name, backbone=bb, captured_orig=s["co"].mean(), captured_norm=s["cn"].mean(),
                             overlap_norm=s["on"].mean(), top1=len(s["hits"]) / s["n"], top1_hits=len(s["hits"]),
                             top1_n=s["n"], coverage=s["n"] / len(sc.dogs), captured_orig_rep_min=min(reps),
                             captured_orig_rep_max=max(reps), vs_chance_diff=d, vs_chance_2se=se2,
                             vs_chance={"앞쪽이 높음": "찍기보다 높음", "뒤쪽이 높음": "찍기보다 낮음"}.get(v, v)))
    hum = pd.read_csv(ROOT / "results" / f"{sp}_human_baseline.csv", encoding="utf-8-sig")
    hm = hum[(hum["space"] == "model") & (hum["mapping"] == "exact")]
    h = lambda metric, truth: float(hm[(hm["metric"] == metric) & (hm["truth"] == truth)
                                       & hm["picks"].isin(["breeds", "first_breed"])]["human"].iloc[0])
    ht = hm[hm["metric"] == "top1"].iloc[0]
    rows.append(dict(method="human_UF_survey", backbone="-", captured_orig=h("captured@3", "orig"),
                     captured_norm=h("captured@3", "norm"), top1=ht["human"], top1_hits=ht["hits"], top1_n=ht["n"],
                     coverage=ht["coverage"], note="UF Top Responses 앞 3품종 (비율 없어 겹침 없음)"))
    rows.append(dict(method="chance", backbone="-", captured_orig=chance["co"].mean(),
                     captured_norm=chance["cn"].mean(), overlap_norm=chance["on"].mean(), note="학습셋 클래스 빈도"))
    rows.append(dict(method="ceiling", backbone="-", captured_orig=P.ceiling(sc.orig, sc.dogs).mean(),
                     captured_norm=P.ceiling(sc.norm, sc.use_norm).mean(), overlap_norm=1.0, note="정확 매핑"))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(ROOT / "results" / f"{sp}_rq1_comparison.csv", index=False, encoding="utf-8-sig")
    print(res[["method", "backbone", "captured_orig", "captured_norm", "overlap_norm", "top1_hits", "top1_n",
               "vs_chance"]].round(3).to_string(index=False))

    # 방법 짝 비교
    prow = []
    for bb in BACKBONES:
        for x, y in combinations([m[0] for m in METHODS], 2):
            if (x, bb) not in S or (y, bb) not in S:
                continue
            a, b = S[(x, bb)], S[(y, bb)]
            d, se2, v = P.paired(a["co"], b["co"])
            prow.append(dict(backbone=bb, pair=f"{x}-{y}", captured_orig_diff=d, two_se=se2,
                             judgement={"앞쪽이 높음": f"{x}가 높음", "뒤쪽이 높음": f"{y}가 높음"}.get(v, v),
                             per_dog_corr=float(np.corrcoef(a["co"], b["co"])[0, 1]),
                             top1_both=len(a["hits"] & b["hits"]), top1_only_first=len(a["hits"] - b["hits"]),
                             top1_only_second=len(b["hits"] - a["hits"]),
                             same_top1_class=float((a["top"] == b["top"]).mean())))
    paired = pd.DataFrame(prow)
    paired.round(4).to_csv(ROOT / "results" / f"{sp}_rq1_paired.csv", index=False, encoding="utf-8-sig")
    print("\n[방법 짝 비교: 포착@3 원래, 개 단위 짝 차이 ± 2SE]")
    print(paired.round(3).to_string(index=False))
    have = [k for k in S if k[1] in DEEP]
    union = set().union(*[S[k]["hits"] for k in have])
    print(f"\n딥 추출기 × 방법 {len(have)}개 가운데 하나라도 Top-1 을 맞힌 개: {len(union)} / {sc.answers and sum(map(bool, sc.answers.values()))}")

    # 같은 가중치 결합 (참고값)
    erow = []

    def add(name, probs):
        s = sc(sum(probs) / len(probs))
        d, se2, v = P.paired(s["co"], chance["co"])
        erow.append(dict(config=name, members=len(probs), captured_orig=s["co"].mean(), captured_norm=s["cn"].mean(),
                         overlap_norm=s["on"].mean(), top1_hits=len(s["hits"]), top1_n=s["n"]))
    for bb in DEEP:
        ms = [ens[(k, bb)] for k, *_ in METHODS if (k, bb) in ens]
        add(f"{'+'.join(k for k, *_ in METHODS if (k, bb) in ens)} {bb}", ms)
    for k, *_ in METHODS:
        ms = [ens[(k, bb)] for bb in DEEP if (k, bb) in ens]
        if ms:
            add(f"{k} 딥 3종 평균", ms)
    add("전부 (방법 × 딥 3종)", [ens[k] for k in have])
    eq = pd.DataFrame(erow)
    eq.round(4).to_csv(ROOT / "results" / f"{sp}_rq1_equal_weight.csv", index=False, encoding="utf-8-sig")
    print("\n[같은 가중치 결합 참고값 (가중치를 고르지 않음, 설정 선택에 쓰지 않음)]")
    print(eq.round(3).to_string(index=False))

    if sp != "dev":   # 순종 보조 평가는 val dev 절반이라 hold 와 무관하다
        fig_dev(res, sp)
        return
    pure = purebred(args.a_dir, args.val_split, args.b_val_dir)
    pure.round(4).to_csv(ROOT / "results" / "purebred_top1_same_set.csv", index=False, encoding="utf-8-sig")
    print("\n[순종 Top-1, val dev 절반 2,540장]")
    print(pure.pivot(index="method", columns="backbone", values="top1").reindex(columns=BACKBONES).round(3).to_string())

    fig_dev(res, sp)


def fig_dev(res: pd.DataFrame, sp: str = "dev"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), facecolor=SURFACE)
    x = np.arange(len(BACKBONES))
    w = 0.27
    ref = res.set_index("method")
    panels = [("captured_orig", "Main: captured ancestry@3 (raw DNA)", 0.6),
              ("captured_norm", "Captured ancestry@3 (renormalized)", 1.05),
              ("overlap_norm", "Distribution overlap (renormalized)", 1.05),
              ("top1", "Top-1 (dominant breed, exact map)", 0.6)]
    labels = {"A_classifier": "A classifier", "B_fcls": "B prototype mix", "C_mix": "C mix training"}
    for ax, (col, title, ymax) in zip(axes, panels):
        style(ax, ymax, "")
        for i, (_, _, _, name, color) in enumerate(METHODS):
            df = res[res["method"] == name].set_index("backbone").reindex(BACKBONES)
            off = (i - 1) * w
            v = df[col].to_numpy(dtype=float)
            ax.bar(x + off, np.nan_to_num(v), w, color=color, label=labels[name], edgecolor=SURFACE)
            if col == "top1":
                for xi, hh, n in zip(x + off, df["top1_hits"], df["top1_n"]):
                    if pd.notna(hh):
                        lo, hi = wilson(hh, int(n))
                        ax.errorbar(xi, hh / n, yerr=[[hh / n - lo], [hi - hh / n]], color=INK2, capsize=2, lw=1)
            for xi, val in zip(x + off, v):
                ax.text(xi, (0 if np.isnan(val) else val) + ymax * 0.015, "n/a" if np.isnan(val) else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7, color=INK)
        lines = [] if col == "overlap_norm" else [("human", ref.loc["human_UF_survey", col], HUMAN_COLOR, "--")]
        if col != "top1":
            lines += [("chance", ref.loc["chance", col], INK2, ":"), ("ceiling", ref.loc["ceiling", col], INK, "-")]
        for name, val, color, ls in lines:
            ax.axhline(val, color=color, ls=ls, lw=1.3)
            ax.text(len(BACKBONES) - 0.45, val + ymax * 0.01, f"{name} {val:.2f}", ha="right", va="bottom",
                    fontsize=8.5, color=color)
        ax.set_xticks(x, [b.upper() if b == "sift" else b for b in BACKBONES])
        ax.set_title(title, color=INK, fontsize=11)
    hd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hd, lab, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=9.5)
    fig.suptitle(f"RQ1 on UF {sp} ({20 if sp == 'dev' else 77} dogs): classifier vs prototype mixing vs mix training vs human survey",
                 color=INK, fontsize=13)
    fig.text(0.5, 0.01, f"{sp} scoring, exact mapping only (METRICS_FINAL). Seeds/repeats 1-3 probabilities "
             "averaged before scoring. Top-1 bars show Wilson 95% CI.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    out_png = ROOT / "figures" / f"{sp}_rq1_with_human.png"
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    print(f"\n저장 → results/{sp}_rq1_*.csv, {out_png}")


if __name__ == "__main__":
    main()
