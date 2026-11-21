"""UF 초안: 평가셋 인덱스 (eval_index_UF.csv). 공용 형식 그대로: image_id, dog_id, path, split

정식 eval_index.csv 와 dev/test 분할은 A 담당이다. 이 파일은 파이프라인 점검용 임시 분할이다.
  - split: 믹스견 가운데 무작위 --n-dev 마리(seed 고정)만 dev, 나머지(순종 포함)는 hold. test 는 만들지 않는다.
  - path: 사진 원본 경로 (D 의 SAM 3 크롭이 아직 없다). 그래서 특징은 eval_uf_raw 라는 이름으로 저장한다.
  - 행 순서: dev 먼저, 그다음 hold (각각 dog_id 순). --limit 로 앞에서 몇 장만 뽑아도 dev 가 들어가게 하려는 것이다.
  - 한 사진에 얼굴 칸과 전신 칸이 나란히 들어 있다. 칸을 나누는 일은 여기서 하지 않는다.

사용 (프로젝트 루트에서):
  # 로컬: 사진 폴더의 절대 경로를 path 에 적는다
  python scripts/make_uf_eval_index.py --dir data/evalset_uf --photos data/uf_dogs/photos
  # Colab: path 앞부분만 Drive 경로로 바꿔 다른 파일로 저장 (분할은 같은 seed 라 같다)
  python scripts/make_uf_eval_index.py --dir data/evalset_uf --photos data/uf_dogs/photos \
      --path-root /content/drive/MyDrive/cv_dog/evalset/raw/UF/photos --name eval_index_UF_colab.csv
"""
import argparse
import csv
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/evalset_uf", help="dogs_UF.csv 가 있고 결과를 쓸 폴더")
    ap.add_argument("--photos", default="data/uf_dogs/photos", help="사진이 실제로 있는 폴더 (있는지 확인용)")
    ap.add_argument("--path-root", default=None, help="path 에 적을 사진 폴더 (기본: --photos 의 절대 경로)")
    ap.add_argument("--name", default="eval_index_UF.csv")
    ap.add_argument("--n-dev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    d, photos = Path(args.dir), Path(args.photos)
    root = args.path_root.rstrip("/") if args.path_root else photos.resolve().as_posix()

    with open(d / "dogs_UF.csv", encoding="utf-8-sig") as f:
        dogs = list(csv.DictReader(f))
    mixed = sorted(r["dog_id"] for r in dogs if r["is_purebred"] != "True")
    dev = set(random.Random(args.seed).sample(mixed, args.n_dev))

    rows, missing = [], []
    for r in dogs:
        split = "dev" if r["dog_id"] in dev else "hold"
        for fname in [x for x in r["photos"].split(";") if x]:
            if not (photos / fname).exists():
                missing.append(fname)
            rows.append([Path(fname).stem, r["dog_id"], f"{root}/{fname}", split])
    if missing:
        raise SystemExit(f"사진 파일이 없음: {missing}")
    rows.sort(key=lambda x: (x[3] != "dev", x[1], x[0]))

    target = d / args.name
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "dog_id", "path", "split"])
        w.writerows(rows)
    os.replace(tmp, target)

    n_dev_img = sum(r[3] == "dev" for r in rows)
    print(f"사진 {len(rows)}장, 개 {len({r[1] for r in rows})}마리 | dev {len(dev)}마리({n_dev_img}장), "
          f"hold {len(dogs) - len(dev)}마리({len(rows) - n_dev_img}장) | test 없음 | seed {args.seed}")
    print("dev:", " ".join(sorted(dev)))
    print("path 예:", rows[0][2])
    print(f"저장 → {target}")


if __name__ == "__main__":
    main()
