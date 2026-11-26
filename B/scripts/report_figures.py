"""보고서 4장 그림 (집계값만 쓴다. 개체별 DNA 값이나 UF 사진은 쓰지 않는다).

그림 1  실험 흐름도 (데이터 없음, 상자와 화살표만)
그림 2  dev 포착 혈통 비율 (원래 비율): 분류기(A) · 평균점 섞기(B) · 섞기 학습(C), 찍기 · 상한 · UF 설문 응답 기준선
        입력: results/dev_rq1_comparison.csv  (compare_rq1_dev.py 출력)
그림 3  순종 Top-1 대 믹스견 dev 포착 혈통 비율 (딥 특징 3종 × 세 방법 = 9개 설정)
        입력: results/purebred_top1_same_set.csv, results/dev_rq1_comparison.csv
그림 4  혈통이 사라지는 단계 (설정별 분해 + 9개 설정 합의)
        입력: results/dev_failure_stages.csv  (failure_stage_dev.py 출력)
그림 5  개체 묶음별 포착 혈통 비율 평균과 상한 평균 (딥 9개 설정 평균, 집계값만)
        입력: results/dog_difficulty.csv  (feature_diagnostics.py 출력)
hold 결과는 그리지 않는다 (A/A/ 최종 정리본 기준 hold 는 아직 결과에 넣지 않음. 설정 고정 후 한 번만 채점)

색: 분류기 #2a78d6, 평균점 섞기 #eb6834 (dataviz 기본 팔레트 1·2번, validate_palette.js 통과), 섞기 학습 #1b9e77
크기 · 글꼴 (논문식): 모든 그림을 docx 본문 폭 그대로 FIG_W(5.8 in)로 그린다(docx 에서 줄이지 않으므로 pt 가 그대로 인쇄된다).
      글꼴은 본문과 같은 맑은 고딕. 축 이름 9 pt, 눈금 · 범례 8 pt, 값 표시 · 직접 라벨 7.5 pt(최소). 그림 안 제목은 두지 않는다(설명은 그림 아래 캡션).
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
FIG_W = 5.8          # docx 본문에 들어가는 폭 (in). pandoc 최대 폭 5.83 in 이하
FS_LABEL, FS_TICK, FS_NOTE = 9, 8, 7.5


def setup():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
    })


def f3(v: float) -> str:
    """보고서 표와 같은 반올림(소수 셋째 자리, 0.5 올림). 0.0375 → 0.038"""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(float(v))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def bar_labels(ax, bars, values):
    for b, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f3(v), ha="center", va="bottom", fontsize=FS_NOTE, color=INK,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)


def fig_dev_captured(out: Path):
    df = pd.read_csv(ROOT / "results" / "dev_rq1_comparison.csv")
    get = lambda m, bb: df.loc[(df.method == m) & (df.backbone == bb), "captured_orig"].astype(float)
    val = lambda m: [float(get(m, k).iloc[0]) if len(get(m, k).dropna()) else np.nan for k, _ in EXTRACTORS]
    methods = [("A_classifier", C_CLF, "분류기"), ("B_fcls", C_PROTO, "평균점 섞기"), ("C_mix", C_MIX, "섞기 학습")]
    ref = lambda m: float(df.loc[df.method == m, "captured_orig"].iloc[0])
    chance, ceiling, human = ref("chance"), ref("ceiling"), ref("human_UF_survey")

    fig, ax = plt.subplots(figsize=(FIG_W, 3.1))
    x = np.arange(len(EXTRACTORS)); w = 0.27
    for i, (m, color, label) in enumerate(methods):
        v = val(m)
        bars = ax.bar(x + (i - 1) * (w + 0.01), np.nan_to_num(v, nan=0), w, color=color, label=label)
        for b, vv in zip(bars, v):
            if np.isnan(vv):
                b.set_visible(False)
                ax.text(b.get_x() + b.get_width() / 2, 0.012, "미실행", ha="center", va="bottom", fontsize=FS_NOTE,
                        color=MUTED, rotation=90)
        bar_labels(ax, bars, v)
    for y, name, ls in [(chance, "찍기", "--"), (human, "UF 설문 응답", "-."), (ceiling, "상한", ":")]:
        ax.axhline(y, color=MUTED, lw=1, ls=ls)
        ax.text(-0.48, y + 0.006, f"{name} {f3(y)}", ha="left", va="bottom", fontsize=FS_NOTE, color=MUTED)
    ax.set_xticks(x, [n for _, n in EXTRACTORS])
    ax.set_ylabel("포착 혈통 비율 (원래 비율)")
    ax.set_ylim(0, 0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab) for _, c, lab in methods],
              frameon=False, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig2_dev_captured_share.png")
    plt.close(fig)


def fig_pipeline(out: Path):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc="#ffffff", ec=MUTED, ls="-", weight="normal", size=FS_NOTE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, linespacing=1.4)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, ls=ls, shrinkA=1, shrinkB=1))

    box(1, 30, 19, 10, "품종 라벨 사진\nTsinghua Dogs\n127클래스 (학습)")
    box(1, 11, 19, 10, "믹스견 사진\nUF 보호소 개\n얼굴·전신 칸 (평가)")
    box(1, 0.5, 19, 5.5, "단서 변형 (RQ2)", ls="--")
    box(25, 13, 15, 24, "동결 특징\n추출기\n\nSIFT+BoVW\nCLIP\nDINOv2\nDINOv3", fc="#f3f2ee")
    arrow(20.5, 35, 24.5, 30); arrow(20.5, 16, 24.5, 20); arrow(10.5, 6.5, 10.5, 10.5, ls="--")
    for i, name in enumerate(["분류기", "평균점 섞기", "섞기 학습"]):
        y = 31 - i * 9.5
        box(45, y, 15, 6.5, name, ec=[C_CLF, C_PROTO, C_MIX][i], weight="bold")
        arrow(40.5, 25, 44.5, y + 3.25)
    ax.text(52.5, 40.5, "RQ1: 세 가지 풀이 방식", ha="center", fontsize=FS_NOTE, color=MUTED)
    box(65, 19, 14, 12, "사진별 비율\n↓\n개체 평균")
    for i in range(3):
        arrow(60.5, 34.25 - i * 9.5, 64.5, 25)
    box(84, 17, 15, 16, "품종 매핑 표\n↓\nDNA 품종 공간\n\n포착 혈통 비율\n채점")
    arrow(79.5, 25, 83.5, 25)
    fig.tight_layout()
    fig.savefig(out / "fig1_pipeline.png")
    plt.close(fig)


def fig_purebred_vs_mixed(out: Path):
    pure = pd.read_csv(ROOT / "results" / "purebred_top1_same_set.csv")
    mix = pd.read_csv(ROOT / "results" / "dev_rq1_comparison.csv")
    methods = [("A_classifier", C_CLF, "분류기"), ("B_fcls", C_PROTO, "평균점 섞기"), ("C_mix", C_MIX, "섞기 학습")]
    markers = {"clip": ("o", "CLIP"), "dinov2": ("s", "DINOv2"), "dinov3": ("^", "DINOv3")}
    chance = float(mix.loc[mix.method == "chance", "captured_orig"].iloc[0])
    # 겹치는 라벨만 자리 옮김 (x 이동, y 이동, 정렬)
    LABEL_POS = {("C_mix", "dinov3"): (-0.005, 0.004, "right"), ("A_classifier", "dinov3"): (0.005, -0.017, "left")}
    fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
    for m, color, _ in methods:
        for bb, (mk, name) in markers.items():
            x = float(pure.loc[(pure.method == m) & (pure.backbone == bb), "top1"].iloc[0])
            y = float(mix.loc[(mix.method == m) & (mix.backbone == bb), "captured_orig"].iloc[0])
            ax.scatter(x, y, s=55, marker=mk, color=color, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
            dx, dy, ha = LABEL_POS.get((m, bb), (0.005, 0.004, "left"))
            ax.text(x + dx, y + dy, name, fontsize=FS_NOTE, color=MUTED, va="bottom", ha=ha)
    ax.axhline(chance, color=MUTED, lw=1, ls="--")
    ax.text(0.602, chance + 0.004, f"찍기 {f3(chance)}", fontsize=FS_NOTE, color=MUTED, va="bottom")
    ax.set_xlim(0.60, 0.93); ax.set_ylim(0, 0.30)
    ax.set_xlabel("순종 보조 평가 Top-1 정확도 (val dev 2,540장)")
    ax.set_ylabel("믹스견 포착 혈통 비율 (dev 20마리)")
    ax.grid(True, axis="both")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], ls="", marker="o", color=c, label=lab, markersize=7) for _, c, lab in methods]
    h2 = [Line2D([], [], ls="", marker=mk, color=MUTED, label=n, markersize=7) for mk, n in markers.values()]
    ax.legend(handles=h1 + h2, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig3_purebred_vs_mixed.png")
    plt.close(fig)


def fig_dog_groups(out: Path):
    df = pd.read_csv(ROOT / "results" / "dog_difficulty.csv")
    groups = [("전체", "전체"), ("coverage < 0.5", "커버리지 < 0.5"), ("coverage ≥ 0.5", "커버리지 ≥ 0.5"),
              ("불리 혈통 있음 (정확 매핑)", "불리 계열\n혈통 있음"), ("불리 혈통 없음 (정확 매핑)", "불리 계열\n혈통 없음")]
    rows = [df.loc[df.group == g].iloc[0] for g, _ in groups]
    cap = [float(r.captured3_mean) for r in rows]
    ceil = [float(r.ceiling_mean) for r in rows]
    labels = [f"{lab}\n({int(r.dogs)}마리)" for (_, lab), r in zip(groups, rows)]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.1))
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, cap, w, color=C_CLF, label="포착 혈통 비율 (딥 9개 설정 평균)")
    b2 = ax.bar(x + w / 2 + 0.01, ceil, w, color="#c3c2b7", label="상한")
    bar_labels(ax, b1, cap); bar_labels(ax, b2, ceil)
    ax.set_xticks(x, labels)
    ax.set_ylabel("dev 개체 평균 (원래 비율)")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig5_dev_dog_groups.png")
    plt.close(fig)


def fig_failure_stages(out: Path):
    """그림 4: 혈통이 어느 단계에서 사라지나. (a) 설정별 누적 막대 (b) 9개 설정 합의."""
    df = pd.read_csv(ROOT / "results" / "dev_failure_stages.csv")
    st = df[df.analysis == "1_stage_by_setting"].set_index("config")
    order = [f"{m} {bb}" for m in ["분류기", "평균점 섞기", "섞기 학습"] for bb in ["clip", "dinov2", "dinov3"]]
    st = st.loc[order]
    pretty = {"clip": "CLIP", "dinov2": "DINOv2", "dinov3": "DINOv3"}
    ylabels = [f"{c.rsplit(' ', 1)[0]} · {pretty[c.rsplit(' ', 1)[1]]}" for c in order]
    # 회색 = 사진을 보기 전에 잃는 몫, 파랑(순서형, 밝음→짙음) = 정답 품종의 예측 순위
    segs = [("미보고", "미보고", "#dcdbd5"), ("학습 품종 밖", "학습 품종 밖", "#b5b4ad"),
            ("먼 순위 (31위~)", "31위 이하", "#86b6ef"), ("중간 순위 (11~30위)", "11~30위", "#5598e7"),
            ("가까운 순위 (4~10위)", "4~10위", "#2a78d6"), ("포착 (1~3위)", "포착 (1~3위)", "#184f95")]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(FIG_W, 4.3), gridspec_kw=dict(height_ratios=[9, 1.7], hspace=0.6))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for col, lab, color in segs:
        v = st[col].to_numpy(float)
        ax.barh(y, v, left=left, height=0.7, color=color, edgecolor="white", linewidth=1, label=lab)
        if col == "포착 (1~3위)":
            for yy, l, vv in zip(y, left, v):
                ax.text(l + vv / 2, yy, f3(vv), ha="center", va="center", fontsize=FS_NOTE, color="white")
        left += v
    ax.set_yticks(y, ylabels)
    ax.set_xlim(0, 1); ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.set_xlabel("개체 평균 혈통 비율 (원래 비율, 합 1)")
    ax.legend(frameon=False, ncol=6, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=7.5,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
    ax.set_title("(a) 설정별", loc="left", fontsize=FS_LABEL, pad=18)

    c = df[df.analysis == "2_consensus_9"].iloc[0]
    parts = [("9개 모두 포착", "모두 포착", "#184f95"), ("일부 설정만 포착", "일부만 포착", "#5598e7"),
             ("아무 설정도 포착 못함 (최고 4~10위)", "모두 놓침\n(최고 4~10위)", "#86b6ef"),
             ("아무 설정도 포착 못함 (9개 모두 11위 이하)", "모두 놓침 (9개 모두 11위 이하)", "#cde2fb")]
    parts[2] = (parts[2][0], "모두 놓침\n(4~10위)", parts[2][2])
    l = 0.0
    for col, lab, color in parts:
        v = float(c[col])
        bx.barh(0, v, left=l, height=0.8, color=color, edgecolor="white", linewidth=1)
        ink = "white" if color in ("#184f95", "#5598e7") else INK
        bx.text(l + v / 2, 0, f"{lab}\n{f3(v)}", ha="center", va="center", fontsize=FS_NOTE, color=ink, linespacing=1.15)
        l += v
    # (b) 는 정확 매핑 혈통만 펼친다: 막대 전체 = 평균 커버리지
    bx.set_xlim(0, l); bx.set_ylim(-0.45, 0.45); bx.axis("off")
    bx.set_title("(b) 정확 매핑 혈통(막대 전체)을 딥 9개 설정이 포착했는가",
                 loc="left", fontsize=FS_LABEL)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.04)
    fig.savefig(out / "fig4_dev_failure_stages.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "report" / "figures"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    setup()
    fig_pipeline(out)
    fig_dev_captured(out)
    fig_purebred_vs_mixed(out)
    fig_dog_groups(out)
    fig_failure_stages(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
