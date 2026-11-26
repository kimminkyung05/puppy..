"""CVPR 2단 원고용 그림 (칼럼 폭 3.25 in, 한국어/영어 두 벌).

B/scripts/report_figures.py 와 같은 집계 CSV(B/results/)를 읽기만 한다. 개체별 DNA 값이나 UF 사진은 쓰지 않는다.
  fig1_pipeline            실험 흐름 (데이터 없음)
  fig2_purebred_vs_mixed   순종 Top-1 대 믹스견 포착 혈통 비율, dev(속 빈 점)→test(찬 점) (딥 9개 설정)
  fig3_failure_stages      혈통이 사라지는 단계 (a) 설정별 (b) 9개 설정 합의
사용 (report/cvpr 안에서): python make_cvpr_figures.py
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "B" / "results"
C_CLF, C_PROTO, C_MIX = "#2a78d6", "#eb6834", "#1b9e77"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
W = 3.25                      # CVPR 칼럼 폭 (in). docx 에서 줄이지 않는다
FS_LABEL, FS_TICK, FS_NOTE = 8, 7, 7      # 인쇄 크기 그대로의 pt. 최소 7 pt

T = {
    "en": dict(
        font="Arial",
        clf="Classifier", proto="Prototype unmixing", mix="Mixture training",
        train="Breed-labeled photos\nTsinghua Dogs\n127 classes (training)",
        evalset="Mixed-breed photos\nUF shelter dogs\nface + body crops\n(evaluation)",
        cue="Cue\nperturbation\n(RQ2)",
        ext="Frozen feature extractor\nSIFT+BoVW / CLIP / DINOv2 / DINOv3",
        rq1="RQ1: three proportion estimators",
        agg="Per-image proportions → per-dog mean",
        score="Breed map → DNA breed space → captured ancestry share",
        prior_dev="Prior (dev)", prior_test="Prior (test)", dev="Dev (20)", test="Test (77)",
        xpure="Purebred Top-1 accuracy (val dev, 2,540 images)",
        ymix="Captured ancestry share\n(mixed-breed dogs)",
        segs=["Not reported", "Outside training breeds", "Rank ≥31", "Rank 11–30", "Rank 4–10", "Captured (1–3)"],
        xstage="Mean ancestry share per dog (original, sums to 1)",
        a="(a) By setting", b="(b) Exact-mapped ancestry by agreement of the 9 settings",
        parts=["Captured by all 9", "Captured by some", "Missed by all (best rank 4–10)", "Missed by all (all ranks ≥11)"],
    ),
    "ko": dict(
        font="Malgun Gothic",
        clf="분류기", proto="평균점 섞기", mix="섞기 학습",
        train="품종 라벨 사진\nTsinghua Dogs\n127클래스 (학습)",
        evalset="믹스견 사진\nUF 보호소 개\n얼굴·전신 칸\n(평가)",
        cue="단서 변형\n(RQ2)",
        ext="동결 특징 추출기\nSIFT+BoVW / CLIP / DINOv2 / DINOv3",
        rq1="RQ1: 세 가지 풀이 방식",
        agg="사진별 비율 → 개체 평균",
        score="품종 매핑 표 → DNA 품종 공간 → 포착 혈통 비율",
        prior_dev="찍기 (dev)", prior_test="찍기 (test)", dev="dev (20)", test="test (77)",
        xpure="순종 보조 평가 Top-1 정확도 (val dev 2,540장)",
        ymix="포착 혈통 비율\n(믹스견)",
        segs=["미보고", "학습 품종 밖", "31위 이하", "11~30위", "4~10위", "포착 (1~3위)"],
        xstage="개체 평균 혈통 비율 (원래 비율, 합 1)",
        a="(a) 설정별", b="(b) 정확 매핑 혈통을 딥 9개 설정이 포착했는가",
        parts=["9개 모두 포착", "일부 설정만 포착", "모두 놓침 (최고 4~10위)", "모두 놓침 (모두 11위 이하)"],
    ),
}


def save(fig, out, name):
    """PNG(300 dpi, docx 용)와 벡터 PDF(LaTeX · 인쇄용)를 함께 저장한다."""
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")


def f3(v: float) -> str:
    return str(Decimal(repr(float(v))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def setup(lang):
    plt.rcParams.update({
        "font.family": T[lang]["font"], "axes.unicode_minus": False, "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.linewidth": 0.6, "lines.linewidth": 0.8, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def fig_pipeline(t, out):
    fig, ax = plt.subplots(figsize=(W, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 80); ax.axis("off"); ax.grid(False)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)

    def box(x, y, w, h, text, ec=MUTED, fc="#ffffff", ls="-", weight="normal", lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS_NOTE, color=INK,
                weight=weight, linespacing=1.3)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, ls=ls, shrinkA=0, shrinkB=0,
                                    mutation_scale=6))

    box(1, 60, 40, 18, t["train"])
    box(46, 60, 30, 18, t["evalset"])
    box(81, 64, 18, 10, t["cue"], ls="--")
    arrow(80.4, 69, 76.6, 69, ls="--")
    box(1, 41, 98, 11, t["ext"], fc="#f3f2ee")
    arrow(21, 59.4, 21, 52.6); arrow(61, 59.4, 61, 52.6)
    for i, (k, c) in enumerate([("clf", C_CLF), ("proto", C_PROTO), ("mix", C_MIX)]):
        x = 1 + i * 33.3
        box(x, 22, 31.4, 9, t[k], ec=c, weight="bold", lw=1.1)
        arrow(x + 15.7, 40.4, x + 15.7, 31.6)
        arrow(x + 15.7, 21.4, x + 15.7, 18.6)
    box(1, 11, 98, 7, t["agg"])
    arrow(50, 10.4, 50, 8.6)
    box(1, 1, 98, 7, t["score"])
    save(fig, out, "fig1_pipeline")
    plt.close(fig)


def fig_purebred_vs_mixed(t, out):
    """순종 Top-1(x) 대 믹스견 포착 혈통 비율(y). 속 빈 점 = dev 20마리, 찬 점 = test 77마리, 세로선으로 잇는다."""
    pure = pd.read_csv(RES / "purebred_top1_same_set.csv")
    dev = pd.read_csv(RES / "dev_rq1_comparison.csv")
    test = pd.read_csv(RES / "hold_rq1_comparison.csv")
    methods = [("A_classifier", C_CLF, t["clf"]), ("B_fcls", C_PROTO, t["proto"]), ("C_mix", C_MIX, t["mix"])]
    markers = {"clip": ("o", "CLIP"), "dinov2": ("s", "DINOv2"), "dinov3": ("^", "DINOv3")}

    def val(df, m, bb):
        return float(df.loc[(df.method == m) & (df.backbone == bb), "captured_orig"].iloc[0])

    fig, ax = plt.subplots(figsize=(W, 2.45))
    for m, color, _ in methods:
        for bb, (mk, _) in markers.items():
            x = float(pure.loc[(pure.method == m) & (pure.backbone == bb), "top1"].iloc[0])
            yd, yt = val(dev, m, bb), val(test, m, bb)
            ax.plot([x, x], [yt, yd], color=color, lw=0.7, alpha=0.6, zorder=2)
            ax.scatter(x, yd, s=20, marker=mk, facecolor="white", edgecolor=color, linewidth=0.9, zorder=3)
            ax.scatter(x, yt, s=20, marker=mk, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    for df, ls, lab in [(dev, ":", t["prior_dev"]), (test, "--", t["prior_test"])]:
        ch = float(df.loc[df.method == "chance", "captured_orig"].iloc[0])
        ax.axhline(ch, color=MUTED, lw=0.8, ls=ls, zorder=1)
        ax.text(0.603, ch + 0.003, f"{lab} {f3(ch)}", fontsize=FS_NOTE, color=MUTED, va="bottom")
    ax.set_xlim(0.60, 0.90); ax.set_ylim(0, 0.335); ax.set_yticks(np.arange(0, 0.31, 0.05))
    ax.set_xlabel(t["xpure"]); ax.set_ylabel(t["ymix"])
    ax.grid(True, axis="both")
    h1 = [Line2D([], [], ls="", marker="o", color=c, label=lab, markersize=4.5) for _, c, lab in methods]
    h2 = [Line2D([], [], ls="", marker=mk, color=MUTED, label=n, markersize=4.5) for mk, n in markers.values()]
    h3 = [Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=MUTED, color=MUTED,
                 label=t["dev"], markersize=4.5),
          Line2D([], [], ls="", marker="o", color=MUTED, label=t["test"], markersize=4.5)]
    ax.legend(handles=h1 + h2 + h3, frameon=False, loc="upper left", ncol=3, fontsize=FS_NOTE - 0.5,
              handletextpad=0.2, columnspacing=0.6, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout(pad=0.3)
    save(fig, out, "fig2_purebred_vs_mixed")
    plt.close(fig)


def fig_failure_stages(t, out):
    df = pd.read_csv(RES / "dev_failure_stages.csv")
    st = df[df.analysis == "1_stage_by_setting"].set_index("config")
    mk = ["분류기", "평균점 섞기", "섞기 학습"]
    bbs = ["clip", "dinov2", "dinov3"]
    pretty = {"clip": "CLIP", "dinov2": "DINOv2", "dinov3": "DINOv3"}
    order = [f"{m} {bb}" for m in mk for bb in bbs]
    st = st.loc[order]
    cols = ["미보고", "학습 품종 밖", "먼 순위 (31위~)", "중간 순위 (11~30위)", "가까운 순위 (4~10위)", "포착 (1~3위)"]
    colors = ["#dcdbd5", "#b5b4ad", "#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W, 3.75), gridspec_kw=dict(height_ratios=[9, 1.3], hspace=0.8))
    # 방법마다 한 칸씩 띄운다
    y = np.array([11 - (i // 3) * 4 - (i % 3) for i in range(9)], dtype=float)
    left = np.zeros(9)
    for col, lab, color in zip(cols, t["segs"], colors):
        v = st[col].to_numpy(float)
        ax.barh(y, v, left=left, height=0.78, color=color, edgecolor="white", linewidth=0.6, label=lab)
        if col == "포착 (1~3위)":
            for yy, l, vv in zip(y, left, v):
                ax.text(l + vv / 2, yy, f3(vv), ha="center", va="center", fontsize=FS_NOTE, color="white")
        left += v
    ax.set_yticks(y, [pretty[o.rsplit(" ", 1)[1]] for o in order])
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for i, k in enumerate(["clf", "proto", "mix"]):   # 방법 이름은 비워 둔 줄에 왼쪽 정렬로
        ax.text(0.02, 11 - i * 4 + 0.95, t[k], transform=tr, ha="left", va="center",
                fontsize=FS_NOTE, weight="bold", color=INK)
    ax.set_ylim(0.4, 12.6)
    ax.set_xlim(0, 1); ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(t["xstage"], fontsize=FS_TICK)
    fig.text(0.02, 0.99, t["a"], ha="left", va="top", fontsize=FS_TICK)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.01, 0.955), fontsize=FS_NOTE, handlelength=1.0, columnspacing=0.7, handletextpad=0.3)

    c = df[df.analysis == "2_consensus_9"].iloc[0]
    pcols = ["9개 모두 포착", "일부 설정만 포착", "아무 설정도 포착 못함 (최고 4~10위)", "아무 설정도 포착 못함 (9개 모두 11위 이하)"]
    pcolors = ["#184f95", "#5598e7", "#86b6ef", "#cde2fb"]
    l = 0.0
    for col, lab, color in zip(pcols, t["parts"], pcolors):
        v = float(c[col])
        bx.barh(0, v, left=l, height=0.9, color=color, edgecolor="white", linewidth=0.6, label=lab)
        ink = "white" if color in ("#184f95", "#5598e7") else INK
        bx.text(l + v / 2, 0, f3(v), ha="center", va="center", fontsize=FS_NOTE, color=ink)
        l += v
    bx.set_xlim(0, l); bx.set_ylim(-0.5, 0.5); bx.axis("off")
    bx.text(0.02, 0.62, t["b"], transform=blended_transform_factory(fig.transFigure, bx.transData),
            ha="left", va="bottom", fontsize=FS_TICK)
    bx.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.24, -0.05), fontsize=FS_NOTE,
              handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.subplots_adjust(left=0.2, right=0.965, top=0.875, bottom=0.11)
    save(fig, out, "fig3_failure_stages")
    plt.close(fig)


def main():
    for lang, t in T.items():
        out = HERE / "figures" / lang
        out.mkdir(parents=True, exist_ok=True)
        setup(lang)
        fig_pipeline(t, out)
        fig_purebred_vs_mixed(t, out)
        fig_failure_stages(t, out)
        print("저장:", out)


if __name__ == "__main__":
    main()
