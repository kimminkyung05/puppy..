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

순종 보조 평가 (--eval-kind purebred): 순종 사진을 순종으로 답하는지 본다. 믹스견 평가와 따로 보고한다.
  python -m src.rq1.prototype_fcls --backbone dinov3 --repeats 1 2 3 --split dev --eval-kind purebred --skip-existing
  - 평가 대상: {backbone}/val 특징 가운데 val_split_seed1.csv(scripts/make_val_split.py)의 dev 행. test 행은 쓰지 않는다.
  - 사진 한 장을 개 한 마리로 본다 (개 단위 평균 없음).
  - 평균점은 믹스견 평가와 똑같이 train 에서만 만든다. val 사진이 평균점 재료에 하나라도 있으면 멈춘다.
  - 출력: {preds_root}/dev/pred_RQ1-val-{backbone}-proto-r{r}.csv   : image_id, label, split, <클래스>
          {preds_root}/dev/recon_RQ1-val-{backbone}-proto-r{r}.csv  : image_id, label, split, recon_err
  - 기본값(--eval-kind mixed)의 동작과 출력 이름은 그대로다.
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


def select_purebred(ids_tr, Xtr, itr, ids_ev, Xev, iev, val_split: str, split: str, strict: bool):
    """순종 보조 평가의 평가 사진과 평균점 재료를 고른다.

    - 평가 사진: val 특징 가운데 val_split 파일에서 split 이 같은 행.
    - 평균점 재료에는 val 사진을 넣지 않는다.
      strict(평균점 = train): val 사진이 하나라도 섞여 있거나 평가할 사진의 특징이 빠져 있으면 멈춘다.
      strict 가 아니면(CHECK- 형식 점검): 평가 사진과 겹치는 것만 재료에서 빼고 계속한다.
    """
    vs = pd.read_csv(val_split)
    want = vs.loc[vs["split"] == split, "image_id"].to_numpy()
    keep = np.isin(ids_ev, want)
    n_missing = len(want) - int(keep.sum())
    if n_missing and strict:
        raise ValueError(f"{val_split} 의 {split} {len(want)}장 중 {n_missing}장의 특징이 없음")
    ids_ev, Xev, iev = ids_ev[keep], Xev[keep], iev[keep].reset_index(drop=True)
    iev["split"] = split

    banned = vs["image_id"].to_numpy() if strict else ids_ev
    overlap = np.isin(ids_tr, banned)
    if overlap.any():
        if strict:
            raise ValueError(f"평균점 재료에 val 사진이 {int(overlap.sum())}장 들어 있음. train 특징을 확인")
        ids_tr, Xtr, itr = ids_tr[~overlap], Xtr[~overlap], itr[~overlap].reset_index(drop=True)
        print(f"[점검] 평가 사진과 겹치는 {int(overlap.sum())}장을 평균점 재료에서 뺐다 (남은 재료 {len(ids_tr)}장)")
    if np.isin(ids_tr, ids_ev).any():
        raise AssertionError("평가 사진이 평균점 재료에 남아 있음")
    if n_missing:
        print(f"[점검] {split} {len(want)}장 중 특징이 있는 {len(ids_ev)}장만 평가한다")
    return ids_tr, Xtr, itr, ids_ev, Xev, iev


def run(backbone: str, repeat: int, split: str, pool: str, use_l2: bool,
        frac: float, eval_subset: str, paths: dict, train_subset: str = "train",
        eval_kind: str = "mixed", val_split: str | None = None) -> dict:
    feat_root = paths["outputs"]["features_root"]
    ids_tr, Xtr, itr = load_features(feat_root, backbone, train_subset, pool)
    ids_ev, Xev, iev = load_features(feat_root, backbone, eval_subset, pool)
    have = None
    if eval_kind == "purebred":
        ids_tr, Xtr, itr, ids_ev, Xev, iev = select_purebred(
            ids_tr, Xtr, itr, ids_ev, Xev, iev, val_split, split, strict=(train_subset == "train"))
    elif split != "all":
        have = sorted(iev["split"].astype(str).unique())
        keep = (iev["split"] == split).to_numpy()
        ids_ev, Xev, iev = ids_ev[keep], Xev[keep], iev[keep].reset_index(drop=True)
    if len(Xev) == 0:   # 빈 예측 파일이 저장되면 --skip-existing 이 그 실행을 끝난 것으로 본다
        raise ValueError(f"평가할 사진이 0장: {backbone}/{eval_subset} 에 split='{split}' 인 행이 없음"
                         + (f" (인덱스에 있는 split 값: {have})" if have is not None else ""))

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

    if eval_kind == "purebred":   # 사진 한 장 = 개 한 마리. 평균 내지 않는다
        pred = pd.DataFrame(W, columns=classes)
        pred.insert(0, "split", iev["split"].to_numpy())
        pred.insert(0, "label", iev["label"].to_numpy())
        pred.insert(0, "image_id", ids_ev)
        save_csv(pd.DataFrame({"image_id": ids_ev, "label": iev["label"], "split": iev["split"],
                               "recon_err": err}), pred_file.with_name(f"recon_{run_name}.csv"))
        save_csv(pred, pred_file)
        return {"run": run_name, "n_images": len(Xev), "n_dogs": len(pred), "n_classes": len(classes),
                "n_proto_images": len(Xtr), "mean_recon_err": round(float(err.mean()), 4)}

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
    ap.add_argument("--eval-subset", default=None,
                    help="예: eval, eval_gray (D의 변형). 기본: eval (--eval-kind purebred 면 val)")
    ap.add_argument("--eval-kind", default="mixed", choices=["mixed", "purebred"],
                    help="mixed: 믹스견 평가셋(기본). purebred: Tsinghua val 순종 사진 보조 평가")
    ap.add_argument("--val-split", default=None,
                    help="순종 보조 평가의 임시 분할 csv (기본: 평가셋 인덱스와 같은 폴더의 val_split_seed1.csv)")
    ap.add_argument("--train-subset", default="train",
                    help="평균점을 만들 특징 subset. train 이 아니면 형식 점검용 실행(CHECK-)이 된다")
    ap.add_argument("--skip-existing", action="store_true", help="pred 파일이 이미 있는 실행은 건너뛴다")
    args = ap.parse_args()

    paths = load_yaml("paths")
    logs_root = paths["outputs"]["logs_root"]
    purebred = args.eval_kind == "purebred"
    if args.eval_subset is None:
        args.eval_subset = "val" if purebred else "eval"
    if purebred:
        if args.split != "dev":
            ap.error("순종 보조 평가는 --split dev 만 쓴다 (test 쪽은 쓰지 않는다)")
        if args.val_split is None:
            args.val_split = str(Path(paths["evalset"]["index_csv"]).parent / "val_split_seed1.csv")
        if not Path(args.val_split).exists():
            ap.error(f"임시 분할 파일이 없음: {args.val_split} → scripts/make_val_split.py 로 먼저 만든다")
        print(f"[순종 보조 평가] {args.backbone}/{args.eval_subset} 중 {args.val_split} 의 dev 행. "
              "정답은 사진으로 붙인 품종 라벨이며 DNA 가 아니다.")
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
        if purebred:
            common.update({"eval_kind": "purebred"})
        try:
            info = run(args.backbone, r, args.split, args.pool, args.l2, args.frac, args.eval_subset, paths,
                       args.train_subset, args.eval_kind, args.val_split)
        except BaseException as e:  # 실패한 실행도 로그에 남긴다 (중단 = KeyboardInterrupt 포함)
            log_run(logs_root, {"run": name, "status": f"failed: {e!r}",
                                "seconds": round(time.time() - t0, 1), **common})
            raise
        info.update({"status": "ok", "seconds": round(time.time() - t0, 1), **common})
        print(info)
        log_run(logs_root, info)


if __name__ == "__main__":
    main()
