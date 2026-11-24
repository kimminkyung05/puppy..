"""RQ1 dev 비교표: A 분류기 · B 평균점 섞기 · 사람 · 찍기 · 상한을 같은 지표로 한 표에 모은다 (집계값만).

입력 (모두 저장소 안의 집계 CSV)
  ../A/A/results/dev_ratio_metrics_exact.csv          A 포착@3 · 분포 겹침 · Top-1 (정확 매핑, seed 1~3 확률 평균)
  results/dev_scores_final.csv                        B 전 지표 (정확 매핑, provisional_score_uf.py --summary-csv)
  results/dev_human_baseline.csv                      사람 기준 (score_human_uf.py)

지표 정의는 docs/METRICS_FINAL.md (A 최종 정의 복사본) 를 따른다. 모든 지표가 정확 매핑이다.
  주 지표: 원래 비율 포착@3 / 보조: 재정규화 포착@3, 재정규화 분포 겹침, Top-1 + coverage
B 값은 results/dev_scores_final.csv (provisional_score_uf.py --summary-csv, Drive 예측 2026-11-24 재채점, L2 없음).

주의: B 값은 provisional_score_uf.py 의 '같은 클래스로 가는 DNA 품종 비율을 더하기' 수정 전에 낸 값이다.
      Drive 의 B 예측으로 다시 채점하면 바뀔 수 있다.

출력
  results/dev_rq1_comparison.csv
  figures/dev_rq1_with_human.png

사용 (B/ 안에서): python scripts/compare_rq1_dev.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_results import INK, INK2, SURFACE, style, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A_RES = ROOT.parent / "A" / "A" / "results"
BACKBONES = ["clip", "dinov2", "dinov3", "sift"]
A_COLOR, B_COLOR, HUMAN_COLOR = "#2a78d6", "#eb6834", "#7a3fb0"


def main():
    a_ex = pd.read_csv(A_RES / "dev_ratio_metrics_exact.csv", encoding="utf-8-sig").set_index("model_name")
    b_fin = pd.read_csv(ROOT / "results" / "dev_scores_final.csv", encoding="utf-8-sig")
    b_fin = b_fin[(b_fin["mapping"] == "정확") & (b_fin["subset"].str.startswith("dev"))]
    hum = pd.read_csv(ROOT / "results" / "dev_human_baseline.csv", encoding="utf-8-sig")

    def b_val(metric, truth, bb, col="value"):
        r = b_fin[(b_fin["metric"] == metric) & (b_fin["truth"] == truth) & (b_fin["config"] == bb)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def h_val(metric, truth, mapping="exact", col="human"):
        r = hum[(hum["space"] == "model") & (hum["mapping"] == mapping) & (hum["metric"] == metric)
                & (hum["truth"] == truth) & (hum["picks"].isin(["breeds", "first_breed"]))]
        return float(r[col].iloc[0])

    rows = []
    for bb in BACKBONES:
        rows.append(dict(method="A_classifier", backbone=bb,
                         captured_orig=a_ex.loc[bb, "captured_orig_ensemble"],
                         captured_norm=a_ex.loc[bb, "captured_norm_ensemble"],
                         overlap_norm=a_ex.loc[bb, "overlap_norm_ensemble"],
                         top1=a_ex.loc[bb, "top1_accuracy"], top1_hits=a_ex.loc[bb, "top1_hits"],
                         top1_n=a_ex.loc[bb, "top1_scorable"], coverage=a_ex.loc[bb, "top1_coverage"]))
        if bb == "sift":
            rows.append(dict(method="B_fcls", backbone=bb, note="SIFT 특징 미수령 (C)"))
            continue
        rows.append(dict(method="B_fcls", backbone=bb,
                         captured_orig=b_val("captured", "orig", bb), captured_norm=b_val("captured", "norm", bb),
                         overlap_norm=b_val("overlap", "norm", bb), top1=b_val("top1", "dominant", bb),
                         top1_hits=b_val("top1", "dominant", bb, "hits"), top1_n=b_val("top1", "dominant", bb, "n"),
                         coverage=b_val("top1", "dominant", bb, "coverage")))
    rows.append(dict(method="C_mix", backbone="all", note="UF 개 단위 예측 없음 (C)"))
    h_top1 = hum[(hum["space"] == "model") & (hum["metric"] == "top1") & (hum["mapping"] == "exact")].iloc[0]
    rows.append(dict(method="human_UF_survey", backbone="-", captured_orig=h_val("captured@3", "orig"),
                     captured_norm=h_val("captured@3", "norm"), top1=h_top1["human"], top1_hits=h_top1["hits"],
                     top1_n=h_top1["n"], coverage=h_top1["coverage"], note="UF Top Responses 앞 3품종 (비율 없어 겹침 없음)"))
    rows.append(dict(method="chance", backbone="-", captured_orig=h_val("captured@3", "orig", col="chance"),
                     captured_norm=h_val("captured@3", "norm", col="chance"),
                     overlap_norm=a_ex["overlap_norm_guessing"].iloc[0], note="학습셋 클래스 빈도"))
    rows.append(dict(method="ceiling", backbone="-", captured_orig=h_val("captured@3", "orig", col="ceiling"),
                     captured_norm=h_val("captured@3", "norm", col="ceiling"),
                     overlap_norm=a_ex["overlap_norm_ceiling"].iloc[0], note="정확 매핑"))
    res = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "dev_rq1_comparison.csv"
    res.round(4).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(res.round(3).to_string(index=False))

    # 그림: 포착 혈통 비율(원래 · 재정규화) + Top-1, A/B 막대와 사람 · 찍기 · 상한 선
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), facecolor=SURFACE)
    x = np.arange(len(BACKBONES))
    w = 0.36
    A = res[res["method"] == "A_classifier"].set_index("backbone").reindex(BACKBONES)
    B = res[res["method"] == "B_fcls"].set_index("backbone").reindex(BACKBONES)
    ref = res.set_index("method")
    panels = [("captured_orig", "Main: captured ancestry@3 (raw DNA)", 0.6),
              ("captured_norm", "Captured ancestry@3 (renormalized)", 1.05),
              ("overlap_norm", "Distribution overlap (renormalized)", 1.05),
              ("top1", "Top-1 (dominant breed, exact map)", 0.6)]
    for ax, (col, title, ymax) in zip(axes, panels):
        style(ax, ymax, "")
        for off, df, color, lab in [(-w / 2, A, A_COLOR, "A classifier"), (w / 2, B, B_COLOR, "B prototype mix")]:
            v = df[col].to_numpy(dtype=float)
            ax.bar(x + off, np.nan_to_num(v), w, color=color, label=lab, edgecolor=SURFACE)
            if col == "top1":
                for xi, h, n in zip(x + off, df["top1_hits"], df["top1_n"]):
                    if pd.notna(h):
                        lo, hi = wilson(h, int(n))
                        ax.errorbar(xi, h / n, yerr=[[h / n - lo], [hi - h / n]], color=INK2, capsize=3, lw=1)
            for xi, val in zip(x + off, v):
                ax.text(xi, (0 if np.isnan(val) else val) + ymax * 0.015, "n/a" if np.isnan(val) else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
        lines = [] if col == "overlap_norm" else [("human", ref.loc["human_UF_survey", col], HUMAN_COLOR, "--")]
        if col != "top1":
            lines += [("chance", ref.loc["chance", col], INK2, ":"), ("ceiling", ref.loc["ceiling", col], INK, "-")]
        for name, val, color, ls in lines:
            ax.axhline(val, color=color, ls=ls, lw=1.3)
            ax.text(len(BACKBONES) - 0.45, val + ymax * 0.01, f"{name} {val:.2f}", ha="right", va="bottom",
                    fontsize=8.5, color=color)
        ax.set_xticks(x, [b.upper() if b == "sift" else b for b in BACKBONES])
        ax.set_title(title, color=INK, fontsize=11)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=9.5)
    fig.suptitle("RQ1 on UF dev (20 dogs): classifier vs prototype mixing vs human survey", color=INK, fontsize=13)
    fig.text(0.5, 0.01, "Provisional dev scoring, exact mapping only (METRICS_FINAL). Top-1 bars show Wilson 95% CI. "
             "B SIFT and C (mix training) have no UF predictions yet.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    out_png = ROOT / "figures" / "dev_rq1_with_human.png"
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    print(f"\n저장 → {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
