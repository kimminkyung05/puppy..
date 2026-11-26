"""A · B · C 의 믹스견 예측을 공용 형식 한 폴더로 모은다 (--split 의 행만, 기본 dev). 원본 파일은 바꾸지 않는다.

세 담당의 파일 내용(dog_id, split, 127클래스 · 공용 클래스 순서 · 행 합 1)은 같고 이름과 폴더만 다르다.
이 스크립트는 이름을 공용 인터페이스(CLAUDE.md)의 실행 이름 규칙으로 맞춰 복사한다.

  A  {a_dir}/RQ1-{bb}-clf-s{n}/pred_RQ1-{bb}-clf-s{n}_dogs.csv        → pred_RQ1-{bb}-clf-s{n}.csv
  B  {b_dir}/pred_RQ1-eval_uf_panels-{bb}-proto-r{n}.csv             → pred_RQ1-{bb}-proto-r{n}.csv  (L2 없음)
  C  {c_dir}/pred_eval_uf-{bb}-mix-v2-s{n}.csv                       → pred_RQ1-{bb}-mix-s{n}.csv
     {c_dir}/pred_eval_uf_v2-sift-mix-s{n}.csv  (C 의 SIFT 이름)

출력: {out}/{split}/pred_RQ1-{bb}-{clf|proto|mix}-{s|r}{n}.csv  (그 split 행만. 기본 dev 에서는 hold 행을 복사하지 않는다)
      --split hold 는 최종 채점 때 한 번만 쓴다 (B 의 hold 예측은 prototype_fcls --split all 의 preds/all/ 에 있다)
      out 은 git 밖(루트 outputs/)에 둔다. UF 개체별 예측이라 저장소에 올리지 않는다.

점검: 열 이름 · 순서가 모든 파일에서 같은지, 행 합이 1인지, 개 목록이 같은지. 다르면 멈춘다.

로컬 폴더 배치 (모두 루트 outputs/rq1_preds/, git 밖)
  A/          A 가 Drive A_Work/preds 로 준 것 그대로 (RQ1-{bb}-clf-s{n}/ 12개)
  B/dev/      B 예측 원본: Drive preds/dev 복사본(CLIP · DINOv2 · DINOv3, Colab) + 로컬 SIFT 실행 결과
              (SIFT: scripts/import_sift_from_c.py 로 C 특징을 옮긴 뒤 로컬 CPU 실행, outputs/preds/dev 에서 복사)
  C/          C 가 준 pred_eval_uf-*.csv 그대로 (12개)
  merged/     이 스크립트의 출력 (공용 형식, dev 만). 채점은 여기서만 읽는다

사용 (B/ 안에서)
  python scripts/collect_rq1_preds.py --a-dir ../outputs/rq1_preds/A --b-dir ../outputs/rq1_preds/B/dev \
      --c-dir ../outputs/rq1_preds/C --out ../outputs/rq1_preds/merged
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BACKBONES = ["clip", "dinov2", "dinov3", "sift"]
SEEDS = [1, 2, 3]


def sources(a_dir: Path, b_dirs: list[Path], c_dir: Path):
    """(공용 실행 이름, 원본 경로) 목록. 없는 파일은 원본 경로만 남겨 나중에 '없음'으로 보인다.
    B 는 여러 폴더(Drive 복사본, 로컬 실행 결과)를 차례로 보고 처음 찾은 파일을 쓴다."""
    for bb in BACKBONES:
        for n in SEEDS:
            yield f"RQ1-{bb}-clf-s{n}", a_dir / f"RQ1-{bb}-clf-s{n}" / f"pred_RQ1-{bb}-clf-s{n}_dogs.csv"
            name = f"pred_RQ1-eval_uf_panels-{bb}-proto-r{n}.csv"
            yield f"RQ1-{bb}-proto-r{n}", next((d / name for d in b_dirs if (d / name).exists()), b_dirs[0] / name)
            c_name = f"pred_eval_uf_v2-sift-mix-s{n}.csv" if bb == "sift" else f"pred_eval_uf-{bb}-mix-v2-s{n}.csv"
            yield f"RQ1-{bb}-mix-s{n}", c_dir / c_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", type=Path, required=True)
    ap.add_argument("--b-dir", type=Path, nargs="+", required=True, help="여러 개면 앞 폴더부터 찾는다")
    ap.add_argument("--c-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "hold"], help="hold 는 최종 채점 때 한 번만")
    args = ap.parse_args()

    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_cols, ref_dogs, done, missing = None, None, [], []
    for run, src in sources(args.a_dir, args.b_dir, args.c_dir):
        if not src.exists():
            missing.append(f"{run} ← {src}")
            continue
        df = pd.read_csv(src, encoding="utf-8-sig")
        cols = list(df.columns)
        if cols[:2] != ["dog_id", "split"]:
            raise SystemExit(f"{src}: 앞 두 열이 dog_id, split 이 아님")
        if ref_cols is None:
            ref_cols = cols
        elif cols != ref_cols:
            raise SystemExit(f"{src}: 열 이름이나 클래스 순서가 다른 파일과 다름")
        dev = df[df["split"] == args.split].sort_values("dog_id").reset_index(drop=True)
        dogs = dev["dog_id"].tolist()
        if ref_dogs is None:
            ref_dogs = dogs
        elif dogs != ref_dogs:
            raise SystemExit(f"{src}: {args.split} 개 목록이 다른 파일과 다름")
        if not dogs:
            raise SystemExit(f"{src}: {args.split} 행이 없음")
        if not np.allclose(dev[cols[2:]].sum(axis=1), 1, atol=1e-4):
            raise SystemExit(f"{src}: 행 합이 1 이 아님")
        tmp = out_dir / f"pred_{run}.csv.tmp"
        dev.to_csv(tmp, index=False)
        tmp.replace(out_dir / f"pred_{run}.csv")
        done.append((run, len(df) - len(dev)))

    print(f"모음 → {out_dir}  ({args.split} {len(ref_dogs or [])}마리, 클래스 {len(ref_cols or []) - 2}개)")
    for run, dropped in done:
        print(f"  pred_{run}.csv" + (f"  ({args.split} 아닌 {dropped}행 뺌)" if dropped else ""))
    if missing:
        print("없음:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
