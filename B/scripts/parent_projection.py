"""[B] 두 부모 평균점 사이 2D 그림: 믹스견 특징이 두 부모 품종 평균점을 잇는 선분 어디에 놓이는가.

평균점 섞기(FCLS)는 "믹스견 특징 ≈ 부모 품종 평균점의 가중 평균"을 가정한다. 이 그림은 그 가정을 눈으로 본다.

대상 개 (dev만):
  DNA 상위 두 품종이 모두 정확 매핑되고, 두 품종 비율 합이 --min-top2 이상인 개.
  주 부모 = 비율이 큰 품종(왼쪽 끝), 부 부모 = 다음 품종(오른쪽 끝).

좌표 (선형 투영이 아니라, 선분 기준 좌표다):
  u = P_부 - P_주
  가로 t = (x - P_주)·u / |u|²       0 = 주 부모 평균점, 1 = 부 부모 평균점
  세로 d = |x - P_주 - t·u| / |u|      선분에서 벗어난 거리 (두 평균점 사이 거리 단위)
  두 부모만 쓰는 비율 기준값 t* = 부 비율 / (주 비율 + 부 비율)

하위 명령:
  extract  부모 품종 학습 사진(클래스당 최대 N장)과 대상 개 패널 사진의 특징을 뽑는다 (로컬 CPU용).
           Colab에 전체 학습셋 특징이 있으면 이 단계 없이 plot 에 --train-subset train 을 쓴다.
  plot     그림(png), 좌표 요약 csv, 그림용 데이터(*_plotdata.json)를 만든다.
  redraw   *_plotdata.json 만 읽어 그림을 다시 그린다. 특징 파일 없이 로컬에서 디자인만 고칠 때 쓴다.
           python scripts/parent_projection.py redraw --plot-data ../outputs/figures_colab/..._plotdata.json

개별 개의 DNA 비율이 그림에 들어가므로 출력은 저장소(B/) 밖에 둔다. 기본 출력: features_root 옆 figures_local/.

예 (B/ 안에서, 로컬):
  $env:DOGMIX_PATHS = "..\\data\\paths_local.yaml"
  python scripts/parent_projection.py extract --backbones clip dinov2 dinov3 --per-class 100
  python scripts/parent_projection.py plot --train-subset train_parents_n100 --eval-subset eval_uf_parents
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import evalset, tsinghua  # noqa: E402
from src.common.config import load_yaml  # noqa: E402
from src.common.features_io import load_features, save_shard, subset_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UF = ROOT / "data" / "evalset_uf"
BACKBONES = ["clip", "dinov2", "dinov3"]
LABEL = {"clip": "CLIP", "dinov2": "DINOv2", "dinov3": "DINOv3"}


# ---------------------------------------------------------------- 대상 개 고르기
def select_dogs(dna_csv: Path, map_csv: Path, index_csv: Path, min_top2: float) -> pd.DataFrame:
    """dev 개 중 상위 두 품종이 모두 정확 매핑(클래스 하나)인 개. 열: dog_id, major, minor, p_major, p_minor, p_rest_top"""
    dna = pd.read_csv(dna_csv)
    bm = pd.read_csv(map_csv)
    exact = {r.breed_raw: [c.strip() for c in str(r.tsinghua_classes).split(";")]
             for r in bm.itertuples() if r.status == "정확"}
    dev = sorted(set(pd.read_csv(index_csv).query("split == 'dev'")["dog_id"]))
    rows = []
    for d in dev:
        g = dna[(dna["dog_id"] == d) & dna["pct"].notna()].sort_values("pct", ascending=False, kind="stable")
        if len(g) < 2:
            continue
        top = g.head(2)
        names = top["breed_raw"].tolist()
        if not all(b in exact and len(exact[b]) == 1 for b in names):
            continue          # 여러 클래스로 가는 품종(푸들 계열)은 평균점이 하나가 아니라 뺀다
        p = top["pct"].astype(float).tolist()
        if p[0] + p[1] < min_top2:
            continue
        rows.append({"dog_id": d, "major": exact[names[0]][0], "minor": exact[names[1]][0],
                     "p_major": p[0], "p_minor": p[1],
                     "p_third": float(g["pct"].iloc[2]) if len(g) > 2 else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- extract
def cmd_extract(args, paths, dogs):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.features.backbones import build_backbone
    from src.features.extract_deep import ImageDataset

    t = paths["tsinghua"]
    items = tsinghua.filter_classes(tsinghua.parse_split(t["train_list"]),
                                    paths["classes"]["exclude"], paths["classes"]["exclude_dingo"])
    need = sorted(set(dogs["major"]) | set(dogs["minor"]))
    rng = np.random.default_rng(0)
    picked = []
    for c in need:
        cl = [it for it in items if it.class_name == c]
        if not cl:
            raise SystemExit(f"학습셋에 클래스 없음: {c}")
        idx = np.sort(rng.choice(len(cl), size=min(args.per_class, len(cl)), replace=False))
        picked += [cl[i] for i in idx]
    ev = evalset.load_index(args.eval_index)
    ev = ev[ev["dog_id"].isin(dogs["dog_id"])].reset_index(drop=True)
    print(f"부모 클래스 {len(need)}개, 학습 사진 {len(picked)}장, 평가 사진 {len(ev)}장")

    bcfg = load_yaml("backbones")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [
        (f"train_parents_n{args.per_class}", [it.image_id for it in picked],
         [partial(tsinghua.load_crop, it, t["image_root"], t["annotation_root"]) for it in picked],
         pd.DataFrame({"image_id": [it.image_id for it in picked], "label": [it.class_name for it in picked]})),
        ("eval_uf_parents", ev["image_id"].tolist(), [partial(evalset.load_image, p) for p in ev["path"]],
         ev[["image_id", "dog_id", "split"]]),
    ]
    for b in args.backbones:
        backbone = None
        for name, ids, loaders, index in jobs:
            out = subset_dir(paths["outputs"]["features_root"], b, name)
            if (out / "shard_000.npz").exists():
                print(f"[건너뜀] {out}")
                continue
            backbone = backbone or build_backbone(b, bcfg, device)
            dl = DataLoader(ImageDataset(loaders, backbone.preprocess), batch_size=16, num_workers=0)
            cs, ms = [], []
            for pv in tqdm(dl, desc=f"{b}/{name}"):
                c, m = backbone.encode(pv)
                cs.append(c)
                ms.append(m)
            out.mkdir(parents=True, exist_ok=True)
            index.to_csv(out / "index.csv", index=False)
            save_shard(out, 0, ids, np.concatenate(cs), np.concatenate(ms))
            print(f"[저장] {out} ({len(ids)}장)")


# ---------------------------------------------------------------- plot
def line_coords(X: np.ndarray, pa: np.ndarray, pb: np.ndarray):
    u = pb - pa
    uu = float(u @ u)
    t = (X - pa) @ u / uu
    r = X - pa - np.outer(t, u)
    return t, np.linalg.norm(r, axis=1) / np.sqrt(uu)


def fcls_two_parent_share(preds_dir: Path | None, backbone: str, eval_subset_tag: str):
    """B 예측(반복 r1~r3 평균)에서 dog → {클래스: 확률}. 파일이 없으면 None."""
    if preds_dir is None:
        return None
    fs = sorted(preds_dir.glob(f"pred_RQ1-{eval_subset_tag}-{backbone}-proto-r[0-9].csv"))
    if not fs:
        return None
    P = sum(pd.read_csv(f).set_index("dog_id").drop(columns="split") for f in fs) / len(fs)
    return P


INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C_MAJOR, C_MINOR, C_PRED = "#2a78d6", "#eb6834", "#6250d6"


def nice(name: str) -> str:
    return name.replace("_", " ")


def wrap(name: str, pct: float, width: int = 24) -> str:
    """왼쪽 설명 칸이 넓어지지 않게 긴 품종 이름은 두 줄로 나눈다."""
    import textwrap
    return "\n   ".join(textwrap.wrap(f"{name} {pct:g}%", width))


def setup_fonts(plt):
    from matplotlib import font_manager
    nanum = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")     # Colab: apt-get install fonts-nanum
    if nanum.exists():
        font_manager.fontManager.addfont(str(nanum))
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8})


def compute(args, paths, dogs):
    """패널마다 그릴 좌표와 요약 행을 계산한다 (그리기와 분리)."""
    froot = paths["outputs"]["features_root"]
    preds_dir = Path(args.preds_dir) if args.preds_dir else None
    rng = np.random.default_rng(0)
    cells, recs = {}, []
    for b in args.backbones:
        _, Xtr, itr = load_features(froot, b, args.train_subset, args.pool)
        _, Xev, iev = load_features(froot, b, args.eval_subset, args.pool)
        P = fcls_two_parent_share(preds_dir, b, args.pred_tag)
        lab = itr["label"].to_numpy()
        for r in dogs.itertuples():
            A, B = Xtr[lab == r.major], Xtr[lab == r.minor]
            pa, pb = A.mean(0), B.mean(0)                     # 평균점은 해당 클래스 전체로
            sel = (iev["dog_id"] == r.dog_id).to_numpy()
            panels = iev.loc[sel, "image_id"].str.rsplit("_", n=1).str[-1].tolist()
            Xd = Xev[sel]
            tA, dA = line_coords(A, pa, pb)
            tB, dB = line_coords(B, pa, pb)
            tp, dp = line_coords(Xd, pa, pb)
            tm, dm = line_coords(Xd.mean(0, keepdims=True), pa, pb)
            iA = rng.choice(len(A), size=min(len(A), args.max_points), replace=False)
            iB = rng.choice(len(B), size=min(len(B), args.max_points), replace=False)
            t_star = r.p_minor / (r.p_major + r.p_minor)
            share, t_pred = None, np.nan
            if P is not None and r.dog_id in P.index:
                wa, wb = float(P.loc[r.dog_id, r.major]), float(P.loc[r.dog_id, r.minor])
                share = wa + wb
                if share > 0:
                    t_pred = wb / share
            cells[(b, r.dog_id)] = dict(tA=tA[iA], dA=dA[iA], tB=tB[iB], dB=dB[iB], tp=tp, dp=dp, panels=panels,
                                        tm=float(tm[0]), dm=float(dm[0]), t_star=t_star, t_pred=t_pred, share=share)
            recs.append({"backbone": b, "dog_id": r.dog_id, "major": r.major, "minor": r.minor,
                         "t_star": round(t_star, 3), "t_mean": round(float(tm[0]), 3),
                         **{f"t_{p}": round(float(v), 3) for p, v in zip(panels, tp)},
                         "d_mean": round(float(dm[0]), 3),
                         "d_purebred_median": round(float(np.median(np.r_[dA, dB])), 3),
                         "t_pred_B": round(float(t_pred), 3) if not np.isnan(t_pred) else None,
                         "B_share_two_parents": round(share, 3) if share is not None else None,
                         "n_major": len(A), "n_minor": len(B)})
    return cells, pd.DataFrame(recs)


def draw(args, dogs, cells, png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    setup_fonts(plt)

    n, m = len(dogs), len(args.backbones)
    ymax = 1.08 * max(max(np.percentile(np.r_[c["dA"], c["dB"]], 99), c["dp"].max()) for c in cells.values())
    LEFT = 2.2                                             # 왼쪽 개 설명 칸 + 눈금 (인치)
    W, H = LEFT + 3.1 * m + 0.2, 1.85 + 2.15 * n
    fig, axes = plt.subplots(n, m, figsize=(W, H), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=LEFT / W, right=1 - 0.2 / W, top=1 - 1.6 / H, bottom=0.62 / H,
                        wspace=0.08, hspace=0.18)
    mk = {"face": "^", "body": "v"}
    for j, b in enumerate(args.backbones):
        axes[0, j].set_title(LABEL.get(b, b), fontsize=11, fontweight="bold", color=INK, pad=6)
        for i, r in enumerate(dogs.itertuples()):
            ax, c = axes[i, j], cells[(b, r.dog_id)]
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.scatter(c["tA"], c["dA"], s=5, c=C_MAJOR, alpha=0.28, lw=0, rasterized=True)
            ax.scatter(c["tB"], c["dB"], s=5, c=C_MINOR, alpha=0.28, lw=0, rasterized=True)
            ax.plot([0, 1], [0, 0], c=INK2, lw=1.4, solid_capstyle="round", clip_on=False, zorder=3)
            ax.scatter([0, 1], [0, 0], s=46, c=[C_MAJOR, C_MINOR], edgecolor="white", lw=1.5,
                       clip_on=False, zorder=4)
            ax.axvline(c["t_star"], color=INK2, ls=(0, (4, 3)), lw=1.0, zorder=2)
            if not np.isnan(c["t_pred"]):
                ax.axvline(c["t_pred"], color=C_PRED, ls=(0, (1, 2)), lw=1.6, zorder=2)
            for k, pnl in enumerate(c["panels"]):
                ax.plot([c["tm"], c["tp"][k]], [c["dm"], c["dp"][k]], c=INK2, lw=0.7, zorder=5)
                ax.scatter(c["tp"][k], c["dp"][k], marker=mk.get(pnl, "o"), s=26, facecolor="white",
                           edgecolor=INK, lw=1.0, zorder=6)
            ax.scatter(c["tm"], c["dm"], marker="D", s=40, c=INK, edgecolor="white", lw=1.2, zorder=7)
            ax.plot([c["tm"], c["tm"]], [0, c["dm"]], color=INK, lw=0.6, alpha=0.35, zorder=2)
            ax.scatter(c["tm"], 0, marker=10, s=38, c=INK, clip_on=False, zorder=5)   # 선 위로 내린 위치 (밑변이 선에 닿음)
            if c["share"] is not None:
                ax.text(0.985, 0.04, f"B가 두 부모에 준 몫 {c['share']:.2f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7, color=INK2,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=8)
            if j == 0:
                pos = ax.get_position()
                yc = pos.y0 + pos.height * 0.62
                fig.text(0.2 / W, yc, r.dog_id, ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
                fig.text(0.2 / W, yc - 0.06 / H,
                         f"{wrap(nice(r.major), r.p_major)}\n+ {wrap(nice(r.minor), r.p_minor)}\n기타 품종 최대 {r.p_third:g}%",
                         ha="left", va="top", fontsize=8, color=INK2, linespacing=1.4)
                ax.set_ylabel("선에서 거리", fontsize=8, color=INK2, labelpad=4)
    ax0 = axes[0, 0]
    ax0.set_xlim(-0.45, 1.45)
    ax0.set_ylim(0, ymax)
    ax0.set_xticks([0, 0.5, 1])
    fig.supxlabel("두 부모 평균점을 잇는 선 위의 위치   (0 = 주 부모 평균점, 1 = 부 부모 평균점)",
                  fontsize=9, color=INK2, y=0.12 / H)
    fig.text(0.2 / W, 1 - 0.22 / H, "믹스견 특징은 두 부모 품종 평균점 사이 어디에 놓이나",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.2 / W, 1 - 0.52 / H,
             f"dev {n}마리 (DNA 상위 두 품종이 정확 매핑, 합 ≥ {args.min_top2:g}%) · {args.pool} 특징 · "
             f"세로축은 두 평균점을 잇는 선에서 떨어진 거리 (두 평균점 사이 거리 = 1)\n"
             f"평균점은 학습셋({args.train_subset})의 해당 품종 전체로 계산 · 순종 점은 품종당 최대 {args.max_points}장만 표시",
             fontsize=8.5, color=INK2, va="top", linespacing=1.5)
    # 범례는 열 우선으로 채워진다: 1행 = 순종 두 종류 + 기준선 두 개, 2행 = 믹스견 표시 세 개
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, mfc=C_MAJOR, mec="none", alpha=0.6, label="주 부모 순종 사진"),
        Line2D([], [], ls="", marker="D", ms=6, mfc=INK, mec="white", label="믹스견 (두 사진 평균)"),
        Line2D([], [], ls="", marker="o", ms=5, mfc=C_MINOR, mec="none", alpha=0.6, label="부 부모 순종 사진"),
        Line2D([], [], ls="", marker=10, ms=6, mfc=INK, mec=INK, label="믹스견을 선 위로 내린 위치"),
        Line2D([], [], color=INK2, ls=(0, (4, 3)), lw=1.0, label="DNA 비율 위치 (두 부모만)"),
        Line2D([], [], ls="", marker="^", ms=6, mfc="white", mec=INK, label="얼굴 사진"),
        Line2D([], [], color=C_PRED, ls=(0, (1, 2)), lw=1.6, label="B 예측 위치 (두 부모 몫만)"),
        Line2D([], [], ls="", marker="v", ms=6, mfc="white", mec=INK, label="몸 사진"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.2 / W - 0.006, 1 - 0.95 / H), ncol=4,
               fontsize=8, frameon=False, handletextpad=0.4, columnspacing=1.3)
    fig.savefig(png, dpi=200, facecolor="white")
    plt.close(fig)


def cmd_plot(args, paths, dogs):
    froot = paths["outputs"]["features_root"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(froot).parent / "figures_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, table = compute(args, paths, dogs)
    tag = f"{args.train_subset}_{args.eval_subset}_{args.pool}"
    png = out_dir / f"dev_parent_projection_{tag}.png"
    csv = out_dir / f"dev_parent_projection_{tag}.csv"
    data = out_dir / f"dev_parent_projection_{tag}_plotdata.json"
    draw(args, dogs, cells, png)
    table.to_csv(csv, index=False)
    save_plot_data(data, args, dogs, cells)
    print(table.drop(columns=["major", "minor"]).to_string(index=False))
    print(f"[저장] {png}\n[저장] {csv}\n[저장] {data}  (redraw 로 특징 없이 다시 그릴 때 쓴다)")


# ---------------------------------------------------------------- 그림용 데이터 (특징 없이 다시 그리기)
META_KEYS = ["backbones", "train_subset", "eval_subset", "pool", "min_top2", "max_points"]


def save_plot_data(path: Path, args, dogs, cells):
    """그림에 쓴 좌표만 json 으로 남긴다. 개별 DNA 비율이 들어 있으므로 그림과 같은 곳(저장소 밖)에 둔다."""
    import json

    def conv(v):
        if isinstance(v, np.ndarray):
            return [round(float(x), 5) for x in v]
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    out = {"meta": {k: getattr(args, k) for k in META_KEYS},
           "dogs": dogs.to_dict("records"),
           "cells": [{"backbone": b, "dog_id": d, **{k: conv(v) for k, v in c.items()}}
                     for (b, d), c in cells.items()]}
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def cmd_redraw(args):
    """plot 이 남긴 *_plotdata.json 만 읽어 다시 그린다 (특징 · 예측 · DNA 파일 불필요)."""
    import json
    src = Path(args.plot_data)
    d = json.loads(src.read_text(encoding="utf-8"))
    meta = argparse.Namespace(**d["meta"])
    dogs = pd.DataFrame(d["dogs"])
    cells = {}
    for c in d["cells"]:
        c = dict(c)
        key = (c.pop("backbone"), c.pop("dog_id"))
        for k in ("tA", "dA", "tB", "dB", "tp", "dp"):
            c[k] = np.asarray(c[k], dtype=float)
        c["t_pred"] = np.nan if c["t_pred"] is None else c["t_pred"]
        cells[key] = c
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / src.name.replace("_plotdata.json", ".png")
    draw(meta, dogs, cells, png)
    print(f"[저장] {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "extract", "plot", "redraw"])
    ap.add_argument("--plot-data", default=None, help="redraw: plot 이 남긴 *_plotdata.json")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--dna", default=str(UF / "dna_UF.csv"))
    ap.add_argument("--breed-map", default=str(UF / "breed_map_UF_draft.csv"))
    ap.add_argument("--eval-index", default=str(UF / "eval_index_UF_panels.csv"))
    ap.add_argument("--min-top2", type=float, default=75.0, help="상위 두 품종 비율 합 하한(%%)")
    ap.add_argument("--per-class", type=int, default=100, help="extract: 부모 클래스당 학습 사진 수")
    ap.add_argument("--train-subset", default="train")
    ap.add_argument("--eval-subset", default="eval_uf_panels")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--preds-dir", default=None, help="B 예측 csv 폴더 (있으면 B 예측 위치도 표시)")
    ap.add_argument("--pred-tag", default="eval_uf_panels", help="예측 파일 이름의 평가 subset 부분")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-points", type=int, default=150,
                    help="plot: 품종당 표시할 순종 사진 수 (평균점 계산에는 전체를 쓴다)")
    args = ap.parse_args()

    if args.cmd == "redraw":
        if not args.plot_data:
            raise SystemExit("redraw 에는 --plot-data 가 필요하다")
        cmd_redraw(args)
        return
    paths = load_yaml("paths")
    dogs = select_dogs(Path(args.dna), Path(args.breed_map), Path(args.eval_index), args.min_top2)
    print(f"대상 개 {len(dogs)}마리 (dev, 상위 두 품종 정확 매핑, 합 >= {args.min_top2:g}%)")
    if dogs.empty:
        raise SystemExit("대상 개 없음")
    if args.cmd == "select":
        print(dogs.to_string(index=False))
    elif args.cmd == "extract":
        cmd_extract(args, paths, dogs)
    else:
        cmd_plot(args, paths, dogs)


if __name__ == "__main__":
    main()
