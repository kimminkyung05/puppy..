"""B 결과 그림: 평균점 섞기(FCLS)의 dev 임시 채점 결과를 보고서용 그림으로 만든다.

입력 (집계값만. 개별 개의 DNA 값은 없다)
  results/raw_dev_scores.csv        provisional_score_uf.py · provisional_score_purebred.py 출력의 집계 표
  results/purebred_confusions.csv   순종 평가에서 가장 자주 헷갈린 클래스 쌍 (dinov3+L2, 반복 3회 합)

출력 (figures/)
  dev_purebred_vs_mixed_top1.png  순종 val dev Top-1 · Top-5 와 믹스견 UF dev Top-1. 오차 막대는 Wilson 95% 구간
  dev_captured_share.png          믹스견 포착 혈통 비율 (예측 상위 3품종의 실제 DNA 비율 합), 찍기 · 상한 기준선
  purebred_confusions.png         순종
  dev_top1_a_vs_b.png             A 분류기와 B 평균점 섞기 비교 (A/results 가 있을 때)에서 자주 헷갈린 클래스 쌍

모두 임시 채점(A 의 eval.py 이전) 값이고 dev 만 쓴다. 믹스견은 19~20마리라 구간이 넓다.

사용 (B/ 안에서)
  python scripts/plot_b_results.py
  python scripts/plot_b_results.py --l2      # L2 정규화한 설정으로 그림
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["clip", "dinov2", "dinov3"]                     # 고정 순서. 색은 추출기를 따른다
COLOR = {"clip": "#2a78d6", "dinov2": "#eb6834", "dinov3": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(hit: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 구간."""
    p = hit / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax, ymax: float, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def bars(ax, names, values, lo=None, hi=None, labels=None, width=0.62) -> None:
    x = np.arange(len(names))
    ax.bar(x, values, width=width, color=[COLOR[b] for b in names], edgecolor=SURFACE, linewidth=2)
    if lo is not None:
        ax.errorbar(x, values, yerr=[np.array(values) - lo, np.array(hi) - values],
                    fmt="none", ecolor=INK, elinewidth=1.2, capsize=5)
    top = hi if hi is not None else values
    for xi, t, lab in zip(x, top, labels or [f"{v:.1f}" for v in values]):
        ax.text(xi, t + ax.get_ylim()[1] * 0.02, lab, ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(x, names)


def pick(df: pd.DataFrame, section: str, l2: bool) -> pd.DataFrame:
    want = [b + ("+L2" if l2 else "") for b in BACKBONES]
    sub = df[(df["section"] == section) & df["config"].isin(want)].set_index("config").loc[want]
    sub.index = BACKBONES
    return sub


def fig_top1(df: pd.DataFrame, l2: bool, out: Path) -> None:
    pure1, pure5 = pick(df, "purebred_top1", l2), pick(df, "purebred_top5", l2)
    mixed = pick(df, "mixed_top1", l2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)

    # 순종: Top-1 막대 + Top-5 표식
    n = int(pure1["n"].iloc[0])
    v1 = 100 * pure1["value"].to_numpy()
    ci = np.array([wilson(round(p * n), n) for p in pure1["value"]]) * 100
    style(a1, 105, "accuracy (%)")
    bars(a1, BACKBONES, v1, ci[:, 0], ci[:, 1], labels=["" for _ in v1])
    for xi, v in enumerate(v1):     # Top-1 값은 막대 안, Top-5 값은 표식 위
        a1.text(xi, v - 4, f"Top-1\n{v:.1f}%", ha="center", va="top", color=SURFACE, fontsize=10, fontweight="bold")
    a1.scatter(np.arange(3), 100 * pure5["value"], marker="_", s=900, color=INK, linewidths=2, zorder=3)
    for xi, v in enumerate(100 * pure5["value"]):
        a1.text(xi, v + 1.5, f"Top-5 {v:.1f}%", va="bottom", ha="center", color=INK2, fontsize=9)
    a1.axhline(100 / 127, color=INK2, linestyle=":", linewidth=1)
    a1.set_title(f"Purebred (Tsinghua val dev, {n} images)", color=INK, fontsize=11)

    # 믹스견: Top-1 + Wilson 구간
    hit, nn = mixed["hit"].astype(int).to_numpy(), mixed["n"].astype(int).to_numpy()
    v = 100 * hit / nn
    ci = np.array([wilson(h, k) for h, k in zip(hit, nn)]) * 100
    style(a2, 70, "Top-1 accuracy (%)")
    bars(a2, BACKBONES, v, ci[:, 0], ci[:, 1], [f"{x:.1f}% ({h}/{k})" for x, h, k in zip(v, hit, nn)])
    a2.set_title(f"Mixed-breed (UF dev, {nn[0]} scorable dogs)", color=INK, fontsize=11)

    fig.suptitle(f"Prototype mixing (FCLS{' + L2' if l2 else ''}): Top-1 reported separately",
                 color=INK, fontsize=13)
    fig.text(0.5, 0.005, "Error bars: Wilson 95% interval. Dotted line: chance Top-1 (1/127). "
             "Mixed-breed Top-1 uses exact mapping (METRICS_FINAL). Provisional dev scoring.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_captured(df: pd.DataFrame, l2: bool, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (section, title) in zip(axes, [("captured_all20_orig", "All dev dogs (20)"),
                                           ("captured_cov05_orig", "Dogs with DNA coverage >= 0.5 (12)")]):
        sub = pick(df, section, l2)
        style(ax, 0.8, "share of true DNA captured by top-3 breeds")
        v = sub["value"].to_numpy()
        bars(ax, BACKBONES, v, labels=[f"{x:.3f}" for x in v])
        chance, ceil = float(sub["chance"].iloc[0]), float(sub["ceiling"].iloc[0])
        ax.set_xlim(-0.5, 3.15)     # 오른쪽 빈 자리에 기준선 이름을 적는다
        ax.axhline(chance, color=INK2, linestyle="--", linewidth=1.2)
        ax.text(3.12, chance + 0.008, f"chance\n{chance:.3f}", va="bottom", ha="right", color=INK2, fontsize=9)
        ax.axhline(ceil, color=INK, linestyle="-", linewidth=1)
        ax.text(3.12, ceil + 0.008, f"ceiling\n{ceil:.3f}", va="bottom", ha="right", color=INK, fontsize=9)
        ticks = []
        for b, d, se, j in zip(BACKBONES, sub["diff_vs_chance"], sub["two_se"], sub["judgement"]):
            ticks.append(f"{b}\n+{d:.3f} ± {se:.3f}\n{'above chance' if j == 'above_chance' else 'unclear'}")
        ax.set_xticks(np.arange(3), ticks)
        ax.set_title(title, color=INK, fontsize=11)
    fig.suptitle(f"Mixed-breed: captured ancestry share (FCLS{' + L2' if l2 else ''}, UF dev)",
                 color=INK, fontsize=13)
    fig.text(0.5, 0.005, "Under each bar: difference from chance ± 2 SE (paired by dog). "
             "Ceiling = picking the true top-3 breeds. Provisional dev scoring.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_confusions(conf: pd.DataFrame, out: Path) -> None:
    conf = conf.sort_values("count")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(conf))
    same = conf["same_breed_in_b"].astype(bool).to_numpy()
    ax.barh(y, conf["count"], color=COLOR["dinov3"], edgecolor=SURFACE, linewidth=2,
            hatch=["//" if s else "" for s in same], height=0.7)
    ax.set_yticks(y, [f"{t} → {p}" for t, p in zip(conf["true_class"], conf["predicted_class"])])
    for yi, c, s in zip(y, conf["count"], same):
        ax.text(c + 0.6, yi, f"{c}" + ("  (same breed after poodle merge)" if s else ""),
                va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, conf["count"].max() * 1.55)
    ax.set_xlabel("images × repeats (60 per class)", color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.set_title("Most frequent purebred confusions (dinov3 + L2, val dev)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_vs_a(df: pd.DataFrame, a_dir: Path, a_exact: Path, l2: bool, out: Path) -> bool:
    """A 분류기와 B 평균점 섞기를 같은 평가 대상에서 비교. A 결과 파일이 없으면 건너뛴다.
    믹스견 Top-1 은 METRICS_FINAL.md 기준(정확 매핑)인 A 의 dev_ratio_metrics_exact.csv 를 쓴다."""
    a_val = a_dir / "model_summary.csv"
    if not (a_exact.exists() and a_val.exists()):
        print(f"[건너뜀] A 결과 없음: {a_exact} / {a_val}")
        return False
    ad = pd.read_csv(a_exact, encoding="utf-8-sig").set_index("model_name")
    av = pd.read_csv(a_val, encoding="utf-8-sig").set_index("model_name")
    names = BACKBONES + ["sift"]
    mixed, pure = pick(df, "mixed_top1", l2), pick(df, "purebred_top1", l2)
    method_color = {"A": "#6250d6", "B": "#eda100"}     # 방법 두 개: 추출기 색과 겹치지 않는 슬롯
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(SURFACE)
    x, w = np.arange(len(names)), 0.38

    def draw(ax, vals, cis, labels, ymax, ylabel):
        style(ax, ymax, ylabel)
        for k, (m, off) in enumerate((("A", -w / 2), ("B", w / 2))):
            v, ci, lab = vals[m], cis[m], labels[m]
            ok = ~np.isnan(v)
            ax.bar(x[ok] + off, v[ok], width=w, color=method_color[m], edgecolor=SURFACE, linewidth=2,
                   label={"A": "A: classifier", "B": "B: prototype mixing (FCLS)"}[m])
            ax.errorbar(x[ok] + off, v[ok], yerr=[v[ok] - ci[ok, 0], ci[ok, 1] - v[ok]], fmt="none",
                        ecolor=INK, elinewidth=1.1, capsize=4)
            for xi, t, s, good in zip(x + off, ci[:, 1], lab, ok):
                ax.text(xi, (t if good else 0) + ymax * 0.015, s if good else "n/a", ha="center",
                        va="bottom", color=INK if good else INK2, fontsize=8.5)
        ax.set_xticks(x, names)
        ax.legend(frameon=False, loc="upper right", fontsize=9)

    # 믹스견 dev: 같은 20마리, 정확 매핑으로 채점 가능한 같은 개
    n_a = ad.loc[names, "top1_scorable"].to_numpy(dtype=float)
    hit_a = ad.loc[names, "top1_hits"].to_numpy(dtype=float)
    hit_b = np.array([mixed.loc[b, "hit"] if b in mixed.index else np.nan for b in names], dtype=float)
    n_b = np.array([mixed.loc[b, "n"] if b in mixed.index else np.nan for b in names], dtype=float)
    vals = {"A": 100 * hit_a / n_a, "B": 100 * hit_b / n_b}
    cis = {"A": np.array([wilson(h, n) for h, n in zip(hit_a, n_a)]) * 100,
           "B": np.array([wilson(h, n) if n == n else (np.nan, np.nan) for h, n in zip(hit_b, n_b)]) * 100}
    labels = {"A": [f"{v:.1f}%\n{int(h)}/{int(n)}" for v, h, n in zip(vals["A"], hit_a, n_a)],
              "B": [f"{v:.1f}%\n{int(h)}/{int(n)}" if n == n else "" for v, h, n in zip(vals["B"], hit_b, n_b)]}
    draw(a1, vals, cis, labels, 75, "Top-1 accuracy (%)")
    a1.set_title(f"Mixed-breed UF dev (same {int(n_a[0])} scorable dogs, exact map)", color=INK, fontsize=11)

    # 순종: A 는 val 전체 5,080장, B 는 val dev 절반 2,540장 (같은 사진 집합이 아니다)
    va = 100 * av.loc[names, "validation_accuracy_mean"].to_numpy()
    vb = np.array([100 * pure.loc[b, "value"] if b in pure.index else np.nan for b in names])
    cis = {"A": np.array([wilson(round(v / 100 * 5080), 5080) for v in va]) * 100,
           "B": np.array([wilson(round(v / 100 * 2540), 2540) if v == v else (np.nan, np.nan) for v in vb]) * 100}
    labels = {"A": [f"{v:.1f}%" for v in va], "B": [f"{v:.1f}%" if v == v else "" for v in vb]}
    draw(a2, {"A": va, "B": vb}, cis, labels, 110, "Top-1 accuracy (%)")
    a2.set_title("Purebred Tsinghua val (A: all 5,080 / B: dev half 2,540)", color=INK, fontsize=11)

    fig.suptitle(f"RQ1 by extractor: classifier (A) vs prototype mixing (B{', L2' if l2 else ''})",
                 color=INK, fontsize=13)
    fig.text(0.5, 0.005, "Error bars: Wilson 95% interval. B has no SIFT run yet (n/a). Purebred panels use "
             "different image sets, compare within method only. Provisional dev scoring.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", action="store_true", help="L2 정규화한 설정으로 그린다 (기본: L2 없음)")
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--a-results", default=str(ROOT.parent / "A" / "results"),
                    help="A 의 results 폴더 (순종 val Top-1 용. 없으면 건너뜀)")
    ap.add_argument("--a-exact", default=str(ROOT.parent / "A" / "A" / "results" / "dev_ratio_metrics_exact.csv"),
                    help="A 의 정확 매핑 dev 지표 (믹스견 Top-1 용)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "results" / "raw_dev_scores.csv")
    conf = pd.read_csv(ROOT / "results" / "purebred_confusions.csv")
    suffix = "_l2" if args.l2 else ""
    fig_top1(df, args.l2, out / f"dev_purebred_vs_mixed_top1{suffix}.png")
    fig_captured(df, args.l2, out / f"dev_captured_share{suffix}.png")
    fig_confusions(conf, out / "purebred_confusions.png")
    fig_vs_a(df, Path(args.a_results), Path(args.a_exact), args.l2, out / f"dev_top1_a_vs_b{suffix}.png")
    for p in sorted(out.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
