"""보고서 4장 그림 (집계값만 쓴다. 개체별 DNA 값이나 UF 사진은 쓰지 않는다).

그림 1  실험 흐름도 (데이터 없음, 상자와 화살표만)
그림 3  dev 포착 혈통 비율 (원래 비율): 분류기(A) · 평균점 섞기(B) · 섞기 학습(C), 찍기 · 상한 · UF 설문 응답 기준선
        입력: results/dev_rq1_comparison.csv  (compare_rq1_dev.py 출력)
그림 2  분류기(A)의 Top-1: (a) 순종 val, (b) 믹스견 test(hold, Wilson 95% 구간)
        입력: ../A/results/final_purebred_vs_mixed_top1.csv

색: 분류기 #2a78d6, 평균점 섞기 #eb6834 (dataviz 기본 팔레트 1·2번, validate_palette.js 통과), 섞기 학습 #1b9e77
사용 (B/ 안에서): python scripts/report_figures.py [--out ../report/figures]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXTRACTORS = [("sift", "SIFT+BoVW"), ("clip", "CLIP"), ("dinov2", "DINOv2"), ("dinov3", "DINOv3")]
C_CLF, C_PROTO, C_MIX = "#2a78d6", "#eb6834", "#1b9e77"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def setup():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
    })


def bar_labels(ax, bars, values):
    for b, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)


def fig_dev_captured(out: Path):
    df = pd.read_csv(ROOT / "results" / "dev_rq1_comparison.csv")
    get = lambda m, bb: df.loc[(df.method == m) & (df.backbone == bb), "captured_orig"].astype(float)
    val = lambda m: [float(get(m, k).iloc[0]) if len(get(m, k).dropna()) else np.nan for k, _ in EXTRACTORS]
    methods = [("A_classifier", C_CLF, "분류기"), ("B_fcls", C_PROTO, "평균점 섞기"), ("C_mix", C_MIX, "섞기 학습")]
    ref = lambda m: float(df.loc[df.method == m, "captured_orig"].iloc[0])
    chance, ceiling, human = ref("chance"), ref("ceiling"), ref("human_UF_survey")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(EXTRACTORS)); w = 0.27
    for i, (m, color, label) in enumerate(methods):
        v = val(m)
        bars = ax.bar(x + (i - 1) * (w + 0.01), np.nan_to_num(v, nan=0), w, color=color, label=label)
        for b, vv in zip(bars, v):
            if np.isnan(vv):
                b.set_visible(False)
                ax.text(b.get_x() + b.get_width() / 2, 0.012, "미실행", ha="center", va="bottom", fontsize=6.5,
                        color=MUTED, rotation=90)
        bar_labels(ax, bars, v)
    for y, name, ls in [(chance, "찍기", "--"), (human, "UF 설문 응답", "-."), (ceiling, "상한", ":")]:
        ax.axhline(y, color=MUTED, lw=1, ls=ls)
        ax.text(-0.48, y + 0.006, f"{name} {y:.3f}", ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x, [n for _, n in EXTRACTORS])
    ax.set_ylabel("포착 혈통 비율 (원래 비율)")
    ax.set_ylim(0, 0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab) for _, c, lab in methods],
              frameon=False, loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_dev_captured_share.png")
    plt.close(fig)


def fig_top1(out: Path):
    df = pd.read_csv(ROOT.parent / "A" / "results" / "final_purebred_vs_mixed_top1.csv").set_index("model_name")
    names = [n for _, n in EXTRACTORS]; keys = [k for k, _ in EXTRACTORS]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=False)

    ax = axes[0]
    v = df.loc[keys, "purebred_top1_mean"].to_numpy()
    bars = ax.bar(names, v, 0.55, color=C_CLF)
    bar_labels(ax, bars, v)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Top-1")
    ax.set_title("(a) 순종 val (5,080장)", fontsize=9, color=INK, loc="left")

    ax = axes[1]
    v = df.loc[keys, "mixed_hold_top1"].to_numpy()
    lo = v - df.loc[keys, "mixed_hold_top1_ci95_low"].to_numpy()
    hi = df.loc[keys, "mixed_hold_top1_ci95_high"].to_numpy() - v
    bars = ax.bar(names, v, 0.55, color=C_CLF)
    ax.errorbar(np.arange(len(keys)), v, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=3)
    for b, val, h in zip(bars, v, hi):
        ax.text(b.get_x() + b.get_width() / 2 + 0.2, val + 0.004, f"{val:.3f}", ha="left", va="bottom", fontsize=7, color=INK)
    n = int(df["hold_scorable_dogs"].iloc[0])
    ax.set_ylim(0, 0.4)
    ax.set_title(f"(b) 믹스견 test ({n}마리, 95% CI)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_classifier_top1.png")
    plt.close(fig)


def fig_pipeline(out: Path):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc="#ffffff", ec=MUTED, ls="-", weight="normal", size=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, linespacing=1.4)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, ls=ls, shrinkA=1, shrinkB=1))

    box(1, 31, 16, 9, "품종 라벨 사진\nTsinghua Dogs\n(127클래스, 학습)")
    box(1, 10, 16, 9, "믹스견 사진\nUF 보호소 개\n(얼굴·전신 칸, 평가)")
    box(1, 0.5, 16, 5.5, "단서 변형 · 대조군 (RQ2)", ls="--", size=6.5)
    box(23, 14, 17, 22, "동결 특징 추출기\n\nSIFT+BoVW\nCLIP\nDINOv2\nDINOv3\n\n(사진마다 1회 추출)", fc="#f3f2ee", weight="normal")
    arrow(17.5, 35.5, 22.5, 30); arrow(17.5, 14.5, 22.5, 20); arrow(9, 6.5, 9, 9.5, ls="--")
    for i, name in enumerate(["분류기", "평균점 섞기", "섞기 학습"]):
        y = 31 - i * 9.5
        box(46, y, 14, 6.5, name, fc="#ffffff", ec=[C_CLF, C_PROTO, MUTED][i], weight="bold", size=7.5)
        arrow(40.5, 25, 45.5, y + 3.25)
    ax.text(53, 40.5, "RQ1: 세 가지 풀이 방식", ha="center", fontsize=7, color=MUTED)
    box(65, 19, 13, 12, "사진별 비율\n(Tsinghua\n클래스 공간)\n→ 개체 평균")
    for i in range(3):
        arrow(60.5, 34.25 - i * 9.5, 64.5, 25)
    box(83, 19, 15.5, 12, "품종 매핑 표\n→ DNA 품종 공간\n\n채점: DNA 결과와\n포착 혈통 비율")
    arrow(78.5, 25, 82.5, 25)
    fig.tight_layout()
    fig.savefig(out / "fig1_pipeline.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "report" / "figures"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    setup()
    fig_pipeline(out)
    fig_dev_captured(out)
    fig_top1(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
