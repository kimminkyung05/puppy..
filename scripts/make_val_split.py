"""순종 보조 평가용 임시 분할: Tsinghua val 을 클래스별로 반씩 dev / test 로 나눈다.

정식 dev/test 분할은 A 가 정한다. 이 파일은 그 전까지 B 의 순종 보조 평가(prototype_fcls --eval-kind purebred)
가 dev 쪽만 쓰도록 하는 임시 분할이다. test 쪽 행은 만들어만 두고 쓰지 않는다.

  - 대상: validation.lst 에서 제외 클래스를 뺀 사진 (127 클래스면 5,080장, 클래스당 40장)
  - 클래스마다 image_id 를 정렬한 뒤 seed 로 섞어 앞 절반을 dev, 나머지를 test 로 둔다 (홀수면 dev 가 하나 적다).
  - 출력: val_split_seed{seed}.csv  (image_id, label, split)
    기본 위치는 경로 설정의 평가셋 인덱스와 같은 폴더 (Colab 에서는 Drive 의 evalset/).
  - 이미 있으면 덮어쓰지 않는다 (같은 seed 면 내용이 같다). 다시 만들려면 --force.

사용 (프로젝트 루트에서. DOGMIX_PATHS 로 경로 설정을 고른다):
  python scripts/make_val_split.py
  python scripts/make_val_split.py --out data/evalset/val_split_seed1.csv
"""
import argparse
import collections
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import tsinghua  # noqa: E402
from src.common.config import load_yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="기본: {평가셋 인덱스 폴더}/val_split_seed{seed}.csv")
    ap.add_argument("--force", action="store_true", help="이미 있는 파일을 덮어쓴다")
    args = ap.parse_args()

    paths = load_yaml("paths")
    out = Path(args.out) if args.out else Path(paths["evalset"]["index_csv"]).parent / f"val_split_seed{args.seed}.csv"
    items = tsinghua.parse_split(paths["tsinghua"]["val_list"])
    items = tsinghua.filter_classes(items, paths["classes"]["exclude"], paths["classes"]["exclude_dingo"])
    by_class = collections.defaultdict(list)
    for it in items:
        by_class[it.class_name].append(it.image_id)

    rng = random.Random(args.seed)
    rows = []
    for name in sorted(by_class):
        ids = sorted(by_class[name])
        rng.shuffle(ids)
        n_dev = len(ids) // 2
        rows += [[i, name, "dev"] for i in sorted(ids[:n_dev])] + [[i, name, "test"] for i in sorted(ids[n_dev:])]

    n_dev = sum(r[2] == "dev" for r in rows)
    per_class = collections.Counter((r[1], r[2]) for r in rows)
    dev_sizes = collections.Counter(per_class[(c, "dev")] for c in by_class)
    test_sizes = collections.Counter(per_class[(c, "test")] for c in by_class)
    print(f"val {len(rows)}장, 클래스 {len(by_class)}개 | dev {n_dev}장, test {len(rows) - n_dev}장 | seed {args.seed}")
    print(f"클래스당 dev 장수 분포: {dict(dev_sizes)} | test 장수 분포: {dict(test_sizes)}")
    if len({r[0] for r in rows}) != len(rows):
        raise SystemExit("image_id 가 겹침")

    if out.exists() and not args.force:
        with open(out, encoding="utf-8") as f:
            old = [r for r in csv.reader(f)][1:]
        same = old == rows
        print(f"[건너뜀] 이미 있음 → {out}  (내용이 지금 만든 것과 {'같음' if same else '다름: seed 나 클래스 설정 확인'})")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "split"])
        w.writerows(rows)
    os.replace(tmp, out)
    print(f"저장 → {out}")


if __name__ == "__main__":
    main()
