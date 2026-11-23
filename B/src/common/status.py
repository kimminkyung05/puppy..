"""[B] 진행 상태 확인: 특징 샤드 · 예측 파일 · 실행 로그. 읽기만 하고 아무것도 바꾸지 않는다.

세션을 다시 열면 이것부터 보고, 끝나지 않은 명령을 그대로 다시 실행한다.

예:
  python -m src.common.status              # 특징 · 예측 요약
  python -m src.common.status --logs 5     # + 최근 로그 5줄
  python -m src.common.status --eta        # + 마지막 timing 로그의 장/초로 전체 추출 시간 추정

특징 상태
  완료     : index.csv의 장수에 필요한 샤드가 모두 저장됨
  진행 중  : 일부만 저장됨 → 같은 extract 명령을 다시 실행하면 이어서 뽑는다
  시작 전  : 저장된 샤드 없음
tmp_files  : 쓰다가 끊긴 임시 파일 수. 실행 중이 아닌데 0이 아니면 그 샤드는 다시 뽑힌다.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import evalset, tsinghua
from src.common.config import load_yaml
from src.common.features_io import done_shards, tmp_files


def feature_rows(features_root: Path, shard_size: int) -> list[dict]:
    rows = []
    for d in sorted(p for p in features_root.glob("*/*") if p.is_dir()):
        done = done_shards(d)
        row = {"backbone": d.parent.name, "subset": d.name, "images": None, "done": len(done),
               "need": None, "tmp_files": len(tmp_files(d))}
        index_csv = d / "index.csv"
        if not index_csv.exists():
            row["state"] = "인덱스 없음"
        else:
            n = len(pd.read_csv(index_csv, usecols=["image_id"]))
            need = (n + shard_size - 1) // shard_size
            row.update(images=n, need=need)
            if need > 0 and done >= set(range(need)):
                row["state"] = "완료"
            elif done:
                row["state"] = "진행 중"
            else:
                row["state"] = "시작 전"
        rows.append(row)
    return rows


def print_features(features_root: Path, shard_size: int) -> None:
    print(f"== 특징: {features_root} (샤드 {shard_size}장) ==")
    if not features_root.exists():
        print("  (폴더 없음: 아직 추출한 적 없음)")
        return
    rows = feature_rows(features_root, shard_size)
    if not rows:
        print("  (비어 있음)")
        return
    print(f"  {'backbone':<10} {'subset':<22} {'images':>7} {'shards':>8} {'tmp_files':>9}  상태")
    for r in rows:
        images = "-" if r["images"] is None else str(r["images"])
        shards = f"{r['done']}/{'?' if r['need'] is None else r['need']}"
        print(f"  {r['backbone']:<10} {r['subset']:<22} {images:>7} {shards:>8} {r['tmp_files']:>9}  {r['state']}")


def print_preds(preds_root: Path) -> None:
    print(f"\n== 예측: {preds_root} ==")
    if not preds_root.exists():
        print("  (폴더 없음: 아직 실행한 적 없음)")
        return
    split_dirs = sorted(p for p in preds_root.iterdir() if p.is_dir())
    if not split_dirs:
        print("  (split 폴더 없음)")
    for d in split_dirs:
        preds = sorted(p.stem[len("pred_"):] for p in d.glob("pred_*.csv"))
        n_recon = len(list(d.glob("recon_*.csv")))
        print(f"  {d.name}: pred {len(preds)}개, recon {n_recon}개, tmp_files {len(tmp_files(d))}")
        for name in preds:
            print(f"    {name}")
    old = sorted(preds_root.glob("pred_*.csv"))
    if old:
        print(f"  주의: split 폴더 밖(예전 위치)에 pred 파일 {len(old)}개가 있음 → 확인 필요")


def read_runs(logs_root: Path) -> list[dict]:
    path = logs_root / "runs.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_logs(logs_root: Path, runs: list[dict], n_last: int) -> None:
    print(f"\n== 로그: {logs_root / 'runs.jsonl'} ==")
    if not runs:
        print("  (로그 없음)")
        return
    n_ok = sum(r.get("status") == "ok" for r in runs)
    print(f"  실행 {len(runs)}건: ok {n_ok}, 실패 {len(runs) - n_ok}")
    extras = ("images", "img_per_sec", "n_dogs", "n_classes", "mean_recon_err", "split", "seconds", "tag")
    for r in runs[-n_last:] if n_last > 0 else []:
        extra = ", ".join(f"{k}={r[k]}" for k in extras if r.get(k) not in (None, ""))
        print(f"  {r.get('time', '?')} | {r.get('who', '?')} | {r.get('run', '?')} | "
              f"{str(r.get('status', '?'))[:100]} | {extra}")


def print_eta(paths: dict, runs: list[dict], n_backbones: int) -> None:
    print("\n== 전체 추출 시간 추정 ==")
    timing = [r for r in runs if r.get("tag") == "timing" and r.get("status") == "ok"
              and r.get("images", 0) > 0 and r.get("seconds", 0) > 0]
    if not timing:
        print("  쓸 수 있는 timing 로그가 없음 (실패했거나 0장). "
              "'--limit 1000 --tag timing'으로 뽑은 뒤 다시 실행. 다시 재려면 train_limit1000 폴더를 지워야 한다.")
        return
    last = timing[-1]
    rate = last["images"] / last["seconds"]
    t, c = paths["tsinghua"], paths["classes"]
    n_train = len(tsinghua.filter_classes(tsinghua.parse_split(t["train_list"]), c["exclude"], c["exclude_dingo"]))
    n_val = len(tsinghua.filter_classes(tsinghua.parse_split(t["val_list"]), c["exclude"], c["exclude_dingo"]))
    index_csv = Path(paths["evalset"]["index_csv"])
    n_eval = len(evalset.load_index(str(index_csv))) if index_csv.exists() else 0
    total = n_train + n_val + n_eval
    print(f"  기준: {last['run']} ({last['time']}), {last['images']}장 / {last['seconds']}초 = {rate:.1f}장/초, "
          f"{last.get('device')}, batch_size={last.get('batch_size')}")
    print(f"  전체 {total}장 = 학습 {n_train} + val {n_val} + 평가 {n_eval}"
          + ("" if index_csv.exists() else " (eval_index.csv 없음)"))
    print(f"  추출기 하나에 약 {total / rate / 60:.0f}분, {n_backbones}개면 약 {n_backbones * total / rate / 60:.0f}분")
    print("  (측정한 추출기 기준. 다른 추출기는 속도가 다를 수 있고, 1,000장에는 작업자 시작 시간이 포함된다)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=int, default=0, help="최근 로그 N줄을 함께 보여 준다")
    ap.add_argument("--eta", action="store_true", help="마지막 timing 로그로 전체 추출 시간을 추정한다")
    args = ap.parse_args()

    paths, bcfg = load_yaml("paths"), load_yaml("backbones")
    out = paths["outputs"]
    print_features(Path(out["features_root"]), bcfg["extract"]["shard_size"])
    print_preds(Path(out["preds_root"]))
    runs = read_runs(Path(out["logs_root"]))
    print_logs(Path(out["logs_root"]), runs, args.logs)
    if args.eta:
        n_backbones = sum(isinstance(v, dict) and "hf_id" in v for v in bcfg.values())
        print_eta(paths, runs, n_backbones)


if __name__ == "__main__":
    main()
