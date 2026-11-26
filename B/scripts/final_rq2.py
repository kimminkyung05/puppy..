"""RQ2 단서 변형 채점 (dev 확인 → 최종 hold 한 번). CLIP · DINOv2 · DINOv3 + A 분류기, SIFT v3 + C 섞기 학습.

outputs/d_rq2_check/rerun_dev.py · sift_rerun.py (dev 재현, D 노트북과 PSNR 280/280 일치)를 split 을 고를 수 있게 옮긴 것이다.
변형 함수는 D/scripts/DINOv3_control_rerun.py 의 것을 그대로 옮겼다. 특징은 B 방식(src/features/d_bridge.py)으로 뽑는다.

  --split dev   dev 20마리. 이전 dev 재현(outputs/d_rq2_check/scores_team.csv)과 같은 값이 나오는지 확인용
  --split hold  hold 80마리 (채점은 truth_UF.csv 에 있는 믹스견 77마리). 팀이 '최종'으로 정한 뒤 한 번만 돌린다

이어서 돌리기: 변형 사진 · 특징(.npy)은 조건마다 저장하고, 같은 명령을 다시 실행하면 있는 파일은 건너뛴다.
특징은 임시 파일에 쓴 뒤 이름을 바꿔 저장한다.

입력 (모두 저장소 밖)
  --photos   UF 원본 사진 {dog}.jpg (face + body 칸이 붙은 한 장)
  --masks    SAM 3 마스크 {dog}.npz (body, head). hold 는 D 의 sam3_test_masks + UF023 수동 머리 박스판
  --uf-dir   eval_index_UF.csv, truth_UF.csv, dna_UF.csv, breed_map_UF_draft.csv
  --a-weights  A 의 RQ1-{bb}-clf-s{n}_linear_weights.npz 폴더
  --sift-dir   C 의 RQ1-sift-mix-v3-s{n}.pt, sift_codebook.npy
  --b-features B 공유 특징 루트 (DINOv3 대체 추출기 검증용, dev 칸 사진만 읽는다)
출력 --out (기본 ../outputs/final_rq2/{split})
  images/{dog}__{cond}.jpg, features/{bb}__{cond}.npy        변형 사진 · 특징 (개 단위 자료, git 밖)
  per_dog_{split}.csv      개 × 추출기 × 조건 × 시드(평균 포함) 포착@3 (원래 비율). hold 를 다시 돌리지 않으려고 남긴다
  scores_{split}.csv       조건별 점수 (시드 평균 후 채점) + 시드 범위
  extra_drop_{split}.csv   추가 하락 = 대조군 점수 - 변형 점수, 개 단위 평균 ± 2SE
  로그: {logs}/runs.jsonl 에 RQ2-final-{split} 한 줄

사용 (B/ 안에서, 로컬 CPU. hold 는 1시간 이상 걸린다)
  python scripts/final_rq2.py --split dev
  python scripts/final_rq2.py --split hold --masks ../data/d_rq2/sam3_masks_all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

B = Path(__file__).resolve().parents[1]
ROOT = B.parent
sys.path.insert(0, str(B))
sys.path.insert(0, str(B / "scripts"))

DEEP = ["clip", "dinov2", "dinov3"]
SEEDS = [1, 2, 3]

# ---- D/scripts/DINOv3_control_rerun.py 에서 그대로 옮김 ----
CONDS = ["original", "gray", "blur_01", "blur_03", "blur_06", "blur_12", "panel_face", "panel_body",
         "control_gray", "control_blur_01", "control_blur_03", "control_blur_06", "control_blur_12",
         "control_panel_face", "control_panel_body"]
PAIR = {"gray": "control_gray", "blur_01": "control_blur_01", "blur_03": "control_blur_03",
        "blur_06": "control_blur_06", "blur_12": "control_blur_12",
        "panel_face": "control_panel_face", "panel_body": "control_panel_body"}
BLUR = {"blur_01": 0.01, "blur_03": 0.03, "blur_06": 0.06, "blur_12": 0.12}
RERUN_SEED = 20261125

# ---- B 추가 (설계 조건, D 범위 밖): 배경만 + 대조군, σ 를 '개 영역(SAM 3 몸 박스) 짧은 변' 비율로 한 블러 + 대조군 ----
EXTRA_PAIR = {"area_background": "control_area_background",
              **{f"blurdog_{k[-2:]}": f"control_blurdog_{k[-2:]}" for k in BLUR}}
CONDS = CONDS + list(EXTRA_PAIR) + list(EXTRA_PAIR.values())
PAIR = {**PAIR, **EXTRA_PAIR}


def stable_seed(*parts):
    raw = "|".join(map(str, parts)).encode("utf-8")
    return RERUN_SEED + int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1000000


def crop_to_body(img, body):
    ys, xs = np.where(body)
    h, w = body.shape
    px = int(0.10 * (xs.max() - xs.min() + 1))
    py = int(0.10 * (ys.max() - ys.min() + 1))
    return img.crop((max(0, xs.min() - px), max(0, ys.min() - py),
                     min(w, xs.max() + 1 + px), min(h, ys.max() + 1 + py)))


def fill_outside(arr, keep):
    fill = np.median(arr.reshape(-1, 3), axis=0).round().astype(np.uint8)
    out = np.broadcast_to(fill, arr.shape).copy()
    out[keep] = arr[keep]
    return out


def random_same_area_mask(shape, area, seed):
    total = shape[0] * shape[1]
    n = min(max(int(area), 1), total)
    out = np.zeros(total, dtype=bool)
    out[np.random.default_rng(seed).choice(total, size=n, replace=False)] = True
    return out.reshape(shape)


def blur(arr, ratio, short_side=None):
    sigma = max(0.5, ratio * (short_side if short_side is not None else min(arr.shape[:2])))
    k = 2 * max(1, int(math.ceil(3 * sigma))) + 1
    return cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma)


def matched_noise(original, target, seed):
    a = np.asarray(original).astype(np.float32)
    b = np.asarray(target).astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    noise = np.random.default_rng(seed).normal(0, 1, size=a.shape).astype(np.float32)
    noise *= math.sqrt(max(mse, 1e-8) / float(np.mean(noise ** 2)))
    return np.clip(a + noise, 0, 255).astype(np.uint8)


def make_image(photos: Path, masks: Path, dog: str, cond: str):
    arr = np.asarray(Image.open(photos / f"{dog}.jpg").convert("RGB"))
    z = np.load(masks / f"{dog}.npz", allow_pickle=False)
    body, head = z["body"].astype(bool), z["head"].astype(bool)
    assert body.shape == arr.shape[:2], (dog, body.shape, arr.shape)
    body_only = body & ~head
    gray = lambda a: np.asarray(Image.fromarray(a).convert("L").convert("RGB"))  # noqa: E731
    if cond == "original":
        out = arr
    elif cond == "gray":
        out = gray(arr)
    elif cond in BLUR:
        out = blur(arr, BLUR[cond])
    elif cond == "panel_face":
        out = fill_outside(arr, head)
    elif cond == "panel_body":
        out = fill_outside(arr, body_only)
    elif cond == "control_gray":
        out = matched_noise(arr, gray(arr), stable_seed(dog, cond))
    elif cond.startswith("control_blur_"):
        out = matched_noise(arr, blur(arr, BLUR[cond.replace("control_", "")]), stable_seed(dog, cond))
    elif cond == "control_panel_face":
        out = fill_outside(arr, random_same_area_mask(head.shape, int(head.sum()), stable_seed(dog, cond)))
    elif cond == "control_panel_body":
        out = fill_outside(arr, random_same_area_mask(body_only.shape, int(body_only.sum()), stable_seed(dog, cond)))
    elif cond == "area_background":
        out = fill_outside(arr, ~body)
    elif cond == "control_area_background":
        out = fill_outside(arr, random_same_area_mask(body.shape, int((~body).sum()), stable_seed(dog, cond)))
    elif cond.startswith("blurdog_") or cond.startswith("control_blurdog_"):
        ys, xs = np.where(body)
        dog_short = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        b = blur(arr, BLUR["blur_" + cond[-2:]], dog_short)
        out = b if cond.startswith("blurdog_") else matched_noise(arr, b, stable_seed(dog, cond))
    else:
        raise KeyError(cond)
    return crop_to_body(Image.fromarray(out), body)
# ---- 옮김 끝 ----


class Dinov3NoTorchvision:
    """HF DINOv3 전처리기는 torchvision 이 있어야 한다. 없는 환경(로컬 .venv)에서는 같은 계산
    (224x224 bilinear antialias, /255, ImageNet 정규화)을 torch 로 직접 한다. 쓰기 전에 validate() 로 확인한다."""

    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def encode(self, images):
        import torch.nn.functional as F
        t = self.torch
        out = []
        for i in range(0, len(images), 20):
            xs = [t.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32)).permute(2, 0, 1)[None]
                  for im in images[i:i + 20]]
            x = t.cat([F.interpolate(v, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
                       for v in xs])
            x = (x / 255.0 - self.mean) / self.std
            with t.no_grad():
                out.append(self.model(pixel_values=x).last_hidden_state[:, 0].numpy())
        return np.concatenate(out)

    def validate(self, b_features: Path, panels: Path, dev_dogs: list[str]) -> float:
        """dev 칸 사진 40장에서 B 공유 특징(Colab, HF 전처리기)과 cos 를 잰다. hold 는 읽지 않는다."""
        ref = {}
        for f in sorted((b_features / "dinov3" / "eval_uf_panels").glob("shard_*.npz")):
            z = np.load(f, allow_pickle=False)
            ref.update(zip(z["image_id"].astype(str), z["cls"].astype(np.float32)))
        ids = [f"{d}_{p}" for d in dev_dogs for p in ("face", "body")]
        mine = self.encode([Image.open(panels / f"{i}.jpg") for i in ids])
        r = np.stack([ref[i] for i in ids])
        cos = (mine * r).sum(1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(r, axis=1)
        return float(cos.min())


def deep_extractor(bb: str, args, dev_dogs):
    from src.features.d_bridge import BExtractor
    if bb == "dinov3":
        try:
            import torchvision  # noqa: F401
        except ImportError:
            ext = Dinov3NoTorchvision()
            c = ext.validate(args.b_features, args.panels, dev_dogs)
            print(f"  DINOv3 대체 추출기 검증: B 공유 특징과 cos 최소 {c:.4f}", flush=True)
            if c < 0.99:
                raise SystemExit("DINOv3 대체 추출기가 B 공유 특징과 다르다 (cos < 0.99). 중단")
            return ext
    return BExtractor(bb, device=args.device, batch_size=20)


class Sift:
    """C/code/make_uf_features.py 의 bovw (336x336 흑백, 저장 코드북 최근접 배정, L1 → sqrt) + C 섞기 학습 v3."""

    def __init__(self, sift_dir: Path):
        import torch
        self.torch = torch
        self.centers = np.load(sift_dir / "sift_codebook.npy", allow_pickle=False).astype(np.float32)
        self.cs = (self.centers ** 2).sum(1)
        self.sift = cv2.SIFT_create()
        allow = [np.dtype, np.ndarray, np._core.multiarray._reconstruct,
                 type(np.dtype("float32")), type(np.dtype("float64"))]
        self.cks = []
        for s in SEEDS:
            with torch.serialization.safe_globals(allow):
                ck = torch.load(sift_dir / f"RQ1-sift-mix-v3-s{s}.pt", map_location="cpu", weights_only=True)
            lin = torch.nn.Linear(len(ck["mu"]), len(ck["classes"]))
            lin.load_state_dict(ck["state"])
            lin.eval()
            self.cks.append((lin, np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32),
                             [str(c) for c in ck["classes"]]))
        self.classes = self.cks[0][3]

    def encode(self, paths):
        """사진 경로를 받는다 (C 와 같이 cv2 로 흑백으로 바로 읽는다)."""
        out = []
        for p in paths:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            assert g is not None, p
            h = np.zeros(len(self.centers), np.float32)
            _, d = self.sift.detectAndCompute(cv2.resize(g, (336, 336)), None)
            if d is not None and len(d):
                d = d.astype(np.float32)
                a = ((d ** 2).sum(1, keepdims=True) - 2 * d @ self.centers.T + self.cs).argmin(1)
                h = np.bincount(a, minlength=len(self.centers)).astype(np.float32)
                h /= max(h.sum(), 1)
                h = np.sqrt(h)
            out.append(h)
        return np.stack(out)

    def predict(self, X):
        Ps = []
        with self.torch.no_grad():
            for lin, mu, sd, _ in self.cks:
                Ps.append(self.torch.softmax(lin(self.torch.from_numpy((X - mu) / sd)), dim=1).numpy())
        return Ps


def a_predict(X, a_dir: Path, bb: str):
    """A README 방식: 사진별 L2 정규화 → W x + b → softmax. 시드 3개."""
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Ps, classes = [], None
    for s in SEEDS:
        z = np.load(a_dir / f"RQ1-{bb}-clf-s{s}_linear_weights.npz", allow_pickle=False)
        L = X @ z["weight"].astype(np.float32).T + z["bias"].astype(np.float32)
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L)
        Ps.append(P / P.sum(axis=1, keepdims=True))
        classes = [str(c) for c in z["classes"]]
    return Ps, classes


def save_npy(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def save_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def mean_2se(x):
    x = np.asarray(x, float)
    return x.mean(), 2 * x.std(ddof=1) / math.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "hold"], help="hold 는 최종 채점 때 한 번만")
    ap.add_argument("--extractors", nargs="+", default=DEEP + ["sift"], choices=DEEP + ["sift"])
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "uf_dogs" / "photos")
    ap.add_argument("--panels", type=Path, default=ROOT / "data" / "uf_dogs" / "panels")
    ap.add_argument("--masks", type=Path, default=ROOT / "data" / "d_rq2" / "sam3_masks")
    ap.add_argument("--uf-dir", type=Path, default=ROOT / "data" / "evalset_uf")
    ap.add_argument("--a-weights", type=Path, default=ROOT / "A" / "A" / "A_to_D_handoff")
    ap.add_argument("--sift-dir", type=Path, default=ROOT / "data" / "d_rq2" / "sift_v3")
    ap.add_argument("--b-features", type=Path, default=ROOT / "outputs" / "features_drive")
    ap.add_argument("--out", type=Path, default=None, help="기본 ../outputs/final_rq2/{split}")
    ap.add_argument("--logs", type=Path, default=ROOT / "outputs" / "logs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.out or ROOT / "outputs" / "final_rq2" / args.split
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    record = {"run": f"RQ2-final-{args.split}", "split": args.split, "extractors": args.extractors,
              "n_conditions": len(CONDS), "masks": str(args.masks), "status": "failed"}
    try:
        idx = pd.read_csv(args.uf_dir / "eval_index_UF.csv")
        dogs = sorted(set(idx.loc[idx["split"] == args.split, "dog_id"]))
        dev_dogs = sorted(set(idx.loc[idx["split"] == "dev", "dog_id"]))
        miss = [d for d in dogs if not (args.photos / f"{d}.jpg").exists() or not (args.masks / f"{d}.npz").exists()]
        if miss:
            raise SystemExit(f"사진이나 마스크가 없는 개 {len(miss)}마리: {miss[:10]}")
        print(f"{args.split} {len(dogs)}마리 × 조건 {len(CONDS)}개, 추출기 {args.extractors}", flush=True)

        print("1/3 변형 사진", flush=True)
        for d in dogs:
            for c in CONDS:
                p = out / "images" / f"{d}__{c}.jpg"
                if not p.exists():
                    tmp = p.with_name(p.stem + ".tmp.jpg")
                    make_image(args.photos, args.masks, d, c).save(tmp, quality=95)
                    tmp.replace(p)

        print("2/3 특징 (조건마다 저장, 있으면 건너뜀)", flush=True)
        feats, sift = {}, None
        for bb in args.extractors:
            ext = None
            for c in CONDS:
                f = out / "features" / f"{bb}__{c}.npy"
                if not f.exists():
                    if ext is None:
                        if bb == "sift":
                            ext = sift = sift or Sift(args.sift_dir)
                        else:
                            ext = deep_extractor(bb, args, dev_dogs)
                    paths = [out / "images" / f"{d}__{c}.jpg" for d in dogs]
                    imgs = paths if bb == "sift" else [Image.open(p).convert("RGB") for p in paths]
                    save_npy(f, np.asarray(ext.encode(imgs), dtype=np.float32))
                    print(f"  {bb}/{c} ({time.time() - t0:.0f}s)", flush=True)
                feats[(bb, c)] = np.load(f)
                assert len(feats[(bb, c)]) == len(dogs), (bb, c)

        print("3/3 채점 (팀 정의: 정확 매핑, 원래 비율, 시드 평균 후)", flush=True)
        from compare_rq1_dev import Scorer
        if "sift" in args.extractors and sift is None:
            sift = Sift(args.sift_dir)
        per_dog, score_rows, extra_rows = [], [], []
        for bb in args.extractors:
            res, res_s = {}, {s: {} for s in SEEDS}
            for c in CONDS:
                if bb == "sift":
                    Ps, classes = sift.predict(feats[(bb, c)]), sift.classes
                else:
                    Ps, classes = a_predict(feats[(bb, c)], args.a_weights, bb)
                sc = Scorer(args.uf_dir, classes, dogs)
                res[c] = sc(pd.DataFrame(np.mean(Ps, axis=0), index=dogs, columns=classes))
                for s, P in zip(SEEDS, Ps):
                    res_s[s][c] = sc(pd.DataFrame(P, index=dogs, columns=classes))
                for tag, r in [("mean", res[c])] + [(f"s{s}", res_s[s][c]) for s in SEEDS]:
                    for d, v in r["co"].items():
                        per_dog.append({"extractor": bb, "condition": c, "seed": tag, "dog_id": d, "captured_orig": v})
                seed_co = [res_s[s][c]["co"].mean() for s in SEEDS]
                score_rows.append({"extractor": bb, "condition": c, "n_dogs": len(res[c]["co"]),
                                   "captured_orig": res[c]["co"].mean(),
                                   "captured_orig_seed_min": min(seed_co), "captured_orig_seed_max": max(seed_co),
                                   "captured_norm": res[c]["cn"].mean(), "overlap_norm": res[c]["on"].mean(),
                                   "top1_hits": len(res[c]["hits"]), "top1_n": res[c]["n"]})
            for v, c in PAIR.items():
                mu, se2 = mean_2se((res[c]["co"] - res[v]["co"]).values)
                seed_ed = [(res_s[s][c]["co"] - res_s[s][v]["co"]).mean() for s in SEEDS]
                extra_rows.append({"extractor": bb, "condition": v, "scope": "B 추가" if v in EXTRA_PAIR else "D 범위",
                                   "drop_variant": mean_2se((res["original"]["co"] - res[v]["co"]).values)[0],
                                   "drop_control": mean_2se((res["original"]["co"] - res[c]["co"]).values)[0],
                                   "extra_drop": mu, "two_se": se2,
                                   "extra_drop_seed_min": min(seed_ed), "extra_drop_seed_max": max(seed_ed),
                                   "judgement": "판단 불가" if abs(mu) <= se2
                                   else ("변형이 더 떨어뜨림" if mu > 0 else "대조군이 더 떨어뜨림")})
        save_csv(pd.DataFrame(per_dog).round(6), out / f"per_dog_{args.split}.csv")
        save_csv(pd.DataFrame(score_rows).round(4), out / f"scores_{args.split}.csv")
        save_csv(pd.DataFrame(extra_rows).round(4), out / f"extra_drop_{args.split}.csv")
        record.update(status="done", n_dogs=len(dogs), n_scored=int(score_rows[0]["n_dogs"]) if score_rows else 0)
        print(f"끝: {out} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        record["seconds"] = round(time.time() - t0, 1)
        args.logs.mkdir(parents=True, exist_ok=True)
        with open(args.logs / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
