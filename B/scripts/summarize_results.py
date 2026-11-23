"""B 결과 요약: 임시 채점 집계값(results/raw_dev_scores.csv)을 A 와 같은 형식의 요약 CSV 로 만든다.

raw_dev_scores.csv 는 Colab 노트북의 채점 셀 출력(provisional_score_uf.py, provisional_score_purebred.py)을
옮겨 적은 집계값이다. 개별 개의 DNA 값은 없다.

출력 (results/)
  model_summary.csv                 순종 Tsinghua val dev 절반 Top-1 · Top-5 (A 의 model_summary.csv 대응)
  dev_top1_model_summary.csv        믹스견 UF dev Top-1 + coverage + Wilson 95% 구간 (A 의 hold 요약과 같은 열)
  dev_purebred_vs_mixed_top1.csv    순종 · 믹스견 Top-1 한 표 (A 의 final_purebred_vs_mixed_top1.csv 대응)
  dev_captured_share_summary.csv    믹스견 포착 혈통 비율 (B 의 보고서 지표)

대표 설정은 L2 없음(model_name = clip / dinov2 / dinov3). L2 설정은 model_name 에 _l2 를 붙여 함께 적는다.

사용 (B/ 안에서)
  python scripts/summarize_results.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TOTAL_DEV_DOGS = 20
REPEATS = 3


def wilson(hit: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = hit / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def model_name(cfg: str) -> str:
    return cfg.replace("+L2", "_l2")


def main():
    raw = pd.read_csv(RES / "raw_dev_scores.csv")
    sec = {s: g.set_index("config") for s, g in raw.groupby("section")}
    cfgs = list(sec["mixed_top1"].index)

    pure = []
    for c in cfgs:
        t1, t5 = sec["purebred_top1"].loc[c], sec["purebred_top5"].loc[c]
        n = int(t1["n"])
        lo, hi = wilson(round(t1["value"] * n), n)
        pure.append({"model_name": model_name(c), "split": "val_dev_half", "images": n,
                     "top1_accuracy": t1["value"], "top1_rep_min": t1["rep_min"], "top1_rep_max": t1["rep_max"],
                     "top1_ci95_wilson_low": lo, "top1_ci95_wilson_high": hi,
                     "top5_accuracy": t5["value"], "repeats_averaged": REPEATS})
    pure = pd.DataFrame(pure)

    mixed = []
    for c in cfgs:
        r = sec["mixed_top1"].loc[c]
        hit, n = int(r["hit"]), int(r["n"])
        lo, hi = wilson(hit, n)
        mixed.append({"dev_total_dogs": TOTAL_DEV_DOGS, "dev_scorable_dogs": n,
                      "dev_unscorable_dogs": TOTAL_DEV_DOGS - n, "coverage": n / TOTAL_DEV_DOGS,
                      "top1_hits": hit, "top1_accuracy": hit / n,
                      "top1_ci95_wilson_low": lo, "top1_ci95_wilson_high": hi,
                      "top1_rep_min": r["rep_min"], "top1_rep_max": r["rep_max"],
                      "model_name": model_name(c), "repeats_averaged": REPEATS})
    mixed = pd.DataFrame(mixed)

    both = pd.DataFrame({
        "model_name": mixed["model_name"],
        "purebred_top1": pure["top1_accuracy"],
        "purebred_split": "val_dev_half",
        "top1_hits": mixed["top1_hits"],
        "dev_scorable_dogs": mixed["dev_scorable_dogs"],
        "dev_total_dogs": mixed["dev_total_dogs"],
        "coverage": mixed["coverage"],
        "mixed_dev_top1": mixed["top1_accuracy"],
        "mixed_dev_top1_ci95_low": mixed["top1_ci95_wilson_low"],
        "mixed_dev_top1_ci95_high": mixed["top1_ci95_wilson_high"],
    })

    cap = []
    for s, subset in (("captured_all20_orig", "all_dev_dogs"), ("captured_cov05_orig", "coverage_ge_0.5")):
        for c in cfgs:
            r = sec[s].loc[c]
            cap.append({"model_name": model_name(c), "subset": subset, "dogs": int(r["n"]),
                        "captured_share_top3": r["value"], "rep_min": r["rep_min"], "rep_max": r["rep_max"],
                        "chance": r["chance"], "ceiling": r["ceiling"],
                        "diff_vs_chance": r["diff_vs_chance"], "two_se": r["two_se"], "judgement": r["judgement"]})
    cap = pd.DataFrame(cap)

    for name, df in (("model_summary.csv", pure), ("dev_top1_model_summary.csv", mixed),
                     ("dev_purebred_vs_mixed_top1.csv", both), ("dev_captured_share_summary.csv", cap)):
        df.to_csv(RES / name, index=False, encoding="utf-8-sig", float_format="%.6g")
        print(f"{RES / name}  ({len(df)}행)")


if __name__ == "__main__":
    main()
