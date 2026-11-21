"""[B] RQ1 평균점 섞기: 품종 평균 특징(평균점) + FCLS. 학습 없음.

질문: 학습 없이, 품종 평균을 섞는 고정된 선형 풀이로 혼합 비율이 복원되나?

절차
  1. 학습셋 특징에서 품종별 평균(평균점) P[C, D]를 구한다.
     부분 표본 r(=1,2,3)마다 품종별로 무작위 80%만 써서 평균점을 다시 구한다.
  2. 평가 사진 특징 x마다 min ||P^T w - x||^2,  w >= 0,  sum(w) = 1 을 푼다 (FCLS).
     구현: NNLS에 delta * [1 ... 1] = delta 행을 덧붙여 합=1을 강하게 건다 (Heinz & Chang 2001).
  3. 사진별 w를 개 단위로 평균 내어 비율로 쓴다.
  4. 재구성 오차 ||P^T w - x|| / ||x|| 를 사진별로 기록한다.

출력 (Tsinghua 클래스 공간. DNA 품종으로 합치는 일은 A의 eval.py가 한다)
  {preds_root}/{split}/pred_RQ1-{backbone}-proto-r{r}.csv   : dog_id, split, <클래스 127개>
  {preds_root}/{split}/recon_RQ1-{backbone}-proto-r{r}.csv  : image_id, dog_id, split, recon_err
  - split 폴더는 --split 값(dev / test / all). dev와 test 결과가 서로 덮어쓰지 않는다.
  - 임시 파일에 쓴 뒤 이름을 바꿔 저장한다. --skip-existing이면 pred 파일이 이미 있는 실행은 건너뛴다.

예:
  python -m src.rq1.prototype_fcls --backbone dinov3 --repeats 1 2 3 --split dev --skip-existing
  python -m src.rq1.prototype_fcls --backbone dinov3 --repeats 1 2 3 --split dev --l2 --skip-existing

형식 점검 (학습셋 특징이 없는 로컬에서 파이프라인이 끝까지 도는지만 볼 때)
  python -m src.rq1.prototype_fcls --backbone clip --repeats 1 --split dev \
      --eval-subset eval_uf_raw_limit5 --train-subset val_limit200
  --train-subset 이 train 이 아니면 실행 이름이 RQ1- 대신 CHECK- 로 시작하고 끝에 -tr_{subset} 이 붙는다.
  평균점이 일부 품종 · 적은 사진으로 만들어지므로 RQ1 결과가 아니다. 보고서에 쓰지 않는다.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.common.config import load_yaml, log_run
from src.common.features_io import load_features, tmp_path


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def build_prototypes(X: np.ndarray, labels: np.ndarray, classes: list[str],
                     frac: float, seed: int) -> np.ndarray:
    """품종별로 frac 비율만 무작위로 골라 평균. frac=1.0이면 전체 평균."""
    rng = np.random.default_rng(seed)
    P = np.zeros((len(classes), X.shape[1]), dtype=np.float64)
    for k, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        if frac < 1.0:
            n = max(1, int(round(len(idx) * frac)))
            idx = rng.choice(idx, size=n, replace=False)
        P[k] = X[idx].mean(axis=0)
    return P


def fcls(P: np.ndarray, x: np.ndarray, delta: float) -> np.ndarray:
    """가중치 w >= 0, sum(w) = 1 인 최소제곱 해. P: [C, D], x: [D]"""
    A = np.vstack([P.T, delta * np.ones((1, P.shape[0]))])
    b = np.concatenate([x, [delta]])
    w, _ = nnls(A, b, maxiter=50 * P.shape[0])
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def make_run_name(backbone: str, repeat: int, pool: str, use_l2: bool, eval_subset: str,
                  train_subset: str = "train") -> str:
    name = f"RQ1-{backbone}-proto-r{repeat}" + ("-l2" if use_l2 else "") + (f"-{pool}" if pool != "cls" else "")
    if eval_subset != "eval":
        name = name.replace("RQ1-", f"RQ1-{eval_subset}-")
    if train_subset != "train":   # 형식 점검용 실행. RQ1 결과와 섞이지 않게 이름을 바꾼다
        name = "CHECK-" + name[len("RQ1-"):] + f"-tr_{train_subset}"
    return name


def pred_path(preds_root: str, split: str, run_name: str) -> Path:
    return Path(preds_root) / split / f"pred_{run_name}.csv"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """임시 파일에 다 쓴 뒤 이름을 바꾼다 (쓰다 끊긴 파일이 완료된 결과로 보이지 않게)."""
    tmp = tmp_path(path)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(backbone: str, repeat: int, split: str, pool: str, use_l2: bool,
        frac: float, eval_subset: str, paths: dict, train_subset: str = "train") -> dict:
    feat_root = paths["outputs"]["features_root"]
    _, Xtr, itr = load_features(feat_root, backbone, train_subset, pool)
    ids_ev, Xev, iev = load_features(feat_root, backbone, eval_subset, pool)
    if split != "all":
        keep = (iev["split"] == split).to_numpy()
        ids_ev, Xev, iev = ids_ev[keep], Xev[keep], iev[keep].reset_index(drop=True)

    if use_l2:
        Xtr, Xev = l2norm(Xtr), l2norm(Xev)
    labels = itr["label"].to_numpy()
    classes = sorted(set(labels))
    P = build_prototypes(Xtr, labels, classes, frac, seed=repeat)
    delta = 10.0 * float(np.abs(P).max())   # 합=1 조건의 가중치. 너무 크면 수치가 불안정해짐

    W = np.zeros((len(Xev), len(classes)))
    err = np.zeros(len(Xev))
    for i, x in enumerate(Xev.astype(np.float64)):
        W[i] = fcls(P, x, delta)
        err[i] = np.linalg.norm(P.T @ W[i] - x) / max(np.linalg.norm(x), 1e-12)

    run_name = make_run_name(backbone, repeat, pool, use_l2, eval_subset, train_subset)
    pred_file = pred_path(paths["outputs"]["preds_root"], split, run_name)
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    per_img = pd.DataFrame(W, columns=classes)
    per_img.insert(0, "split", iev["split"].to_numpy())
    per_img.insert(0, "dog_id", iev["dog_id"].to_numpy())
    pred = per_img.groupby(["dog_id", "split"], sort=True)[classes].mean()
    pred = pred.div(pred.sum(axis=1), axis=0).reset_index()          # 개 단위 평균 후 합 1

    # recon을 먼저, pred를 마지막에 쓴다: pred 파일이 있으면 그 실행은 끝난 것이다 (--skip-existing의 기준)
    save_csv(pd.DataFrame({"image_id": ids_ev, "dog_id": iev["dog_id"], "split": iev["split"],
                           "recon_err": err}), pred_file.with_name(f"recon_{run_name}.csv"))
    save_csv(pred, pred_file)
    return {"run": run_name, "n_images": len(Xev), "n_dogs": len(pred),
            "n_classes": len(classes), "mean_recon_err": round(float(err.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="clip | dinov2 | dinov3 | sift | dummy")
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--l2", action="store_true", help="특징을 L2 정규화 (dev로 결정)")
    ap.add_argument("--frac", type=float, default=0.8, help="부분 표본 비율")
    ap.add_argument("--eval-subset", default="eval", help="예: eval, eval_gray (D의 변형)")
    ap.add_argument("--train-subset", default="train",
                    help="평균점을 만들 특징 subset. train 이 아니면 형식 점검용 실행(CHECK-)이 된다")
    ap.add_argument("--skip-existing", action="store_true", help="pred 파일이 이미 있는 실행은 건너뛴다")
    args = ap.parse_args()

    paths = load_yaml("paths")
    logs_root = paths["outputs"]["logs_root"]
    check = args.train_subset != "train"
    if check:
        print(f"[점검] 평균점을 학습셋(train)이 아니라 '{args.train_subset}' 특징으로 만든다. "
              "RQ1 결과가 아니라 파일 형식과 흐름만 보는 실행이다 (실행 이름 CHECK-...).")
    for r in args.repeats:
        name = make_run_name(args.backbone, r, args.pool, args.l2, args.eval_subset, args.train_subset)
        target = pred_path(paths["outputs"]["preds_root"], args.split, name)
        if args.skip_existing and target.exists():
            print(f"[건너뜀] {name}: 이미 있음 → {target}")
            continue
        t0 = time.time()
        common = {"who": "B", "split": args.split, "frac": args.frac, "l2": args.l2, "pool": args.pool}
        if check:
            common.update({"train_subset": args.train_subset, "note": "형식 점검용. RQ1 결과 아님"})
        try:
            info = run(args.backbone, r, args.split, args.pool, args.l2, args.frac, args.eval_subset, paths,
                       args.train_subset)
        except Exception as e:  # 실패한 실행도 로그에 남긴다
            log_run(logs_root, {"run": name, "status": f"failed: {e!r}",
                                "seconds": round(time.time() - t0, 1), **common})
            raise
        info.update({"status": "ok", "seconds": round(time.time() - t0, 1), **common})
        print(info)
        log_run(logs_root, info)


if __name__ == "__main__":
    main()
