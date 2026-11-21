"""UF 초안: 평가셋 인덱스. 공용 형식 그대로: image_id, dog_id, path, split (+ 칸 사진이면 panel)

정식 eval_index.csv 와 dev/test 분할은 A 담당이다. 이 파일은 파이프라인 점검용 임시 분할이다.
  - split: 믹스견 가운데 무작위 --n-dev 마리(seed 고정)만 dev, 나머지(순종 포함)는 hold. test 는 만들지 않는다.
  - 행 순서: dev 먼저, 그다음 hold (각각 dog_id 순). --limit 로 앞에서 몇 장만 뽑아도 dev 가 들어가게 하려는 것이다.

두 가지 인덱스
  1) 원본 사진 (기본)      eval_index_UF.csv         image_id=UF001,       특징 이름 eval_uf_raw
     한 사진에 얼굴 칸과 전신 칸이 나란히 들어 있어 가운데 자르기가 두 칸의 경계에 걸린다.
  2) 칸 사진 (--panels)    eval_index_UF_panels.csv  image_id=UF001_face / UF001_body, 특징 이름 eval_uf_panels
     split_uf_panels.py 가 만든 panels.csv 를 읽는다. 공용 4열 뒤에 panel 열(face/body)이 붙는다.
     개마다 사진이 2장이고, 평균점 섞기는 두 장의 결과를 개 단위로 평균 낸다.
  어느 쪽도 D 의 SAM 3 크롭이 아니므로 특징 이름에 eval 을 그대로 쓰지 않는다.

같은 폴더에 다른 UF 인덱스가 이미 있으면 dev 개 목록이 같은지 확인하고, 다르면 멈춘다.

사용 (프로젝트 루트에서):
  python scripts/make_uf_eval_index.py --dir data/evalset_uf --photos data/uf_dogs/photos
  python scripts/make_uf_eval_index.py --dir data/evalset_uf --panels data/uf_dogs/panels.csv --photos data/uf_dogs/panels
  # Colab: path 앞부분만 Drive 경로로 바꿔 다른 파일로 저장 (분할은 같은 seed 라 같다)
  python scripts/make_uf_eval_index.py --dir data/evalset_uf --panels data/uf_dogs/panels.csv --photos data/uf_dogs/panels \
      --path-root "/content/drive/MyDrive/<공유폴더 바로가기 이름>/evalset/raw/UF/panels" --name eval_index_UF_panels_colab.csv
"""
import argparse
import csv
import os
import random
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/evalset_uf", help="dogs_UF.csv 가 있고 결과를 쓸 폴더")
    ap.add_argument("--photos", default=None,
                    help="사진이 실제로 있는 폴더 (있는지 확인용). 기본: data/uf_dogs/photos, --panels 면 panels.csv 옆의 panels/")
    ap.add_argument("--panels", default=None, help="split_uf_panels.py 가 만든 panels.csv. 주면 칸 사진 인덱스를 만든다")
    ap.add_argument("--path-root", default=None, help="path 에 적을 사진 폴더 (기본: --photos 의 절대 경로)")
    ap.add_argument("--name", default=None, help="기본: eval_index_UF.csv, --panels 면 eval_index_UF_panels.csv")
    ap.add_argument("--n-dev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    d = Path(args.dir)
    if args.panels:
        photos = Path(args.photos) if args.photos else Path(args.panels).parent / "panels"
        name = args.name or "eval_index_UF_panels.csv"
    else:
        photos = Path(args.photos or "data/uf_dogs/photos")
        name = args.name or "eval_index_UF.csv"
    root = args.path_root.rstrip("/") if args.path_root else photos.resolve().as_posix()

    dogs = read_csv(d / "dogs_UF.csv")
    mixed = sorted(r["dog_id"] for r in dogs if r["is_purebred"] != "True")
    dev = set(random.Random(args.seed).sample(mixed, args.n_dev))

    # 이미 있는 UF 인덱스와 dev 목록이 같은지 확인
    for other in sorted(d.glob("eval_index_UF*.csv")):
        if other.name == name:
            continue
        other_dev = {r["dog_id"] for r in read_csv(other) if r["split"] == "dev"}
        if other_dev != dev:
            raise SystemExit(f"dev 개 목록이 {other.name} 과 다름 (seed · n-dev 확인). "
                             f"여기만 있음: {sorted(dev - other_dev)} | 저쪽만 있음: {sorted(other_dev - dev)}")
        print(f"dev 목록 확인: {other.name} 과 같음 ({len(dev)}마리)")

    split_of = {r["dog_id"]: ("dev" if r["dog_id"] in dev else "hold") for r in dogs}
    rows, missing = [], []
    if args.panels:
        order = {"face": 0, "body": 1}
        for r in read_csv(Path(args.panels)):
            if r["dog_id"] not in split_of:
                raise SystemExit(f"panels.csv 의 {r['dog_id']} 가 dogs_UF.csv 에 없음")
            if not (photos / r["file"]).exists():
                missing.append(r["file"])
            rows.append([Path(r["file"]).stem, r["dog_id"], f"{root}/{r['file']}", split_of[r["dog_id"]], r["panel"]])
        no_panel = sorted(set(split_of) - {r[1] for r in rows})
        per_dog = {}
        for r in rows:
            per_dog.setdefault(r[1], set()).add(r[4])
        odd = sorted(k for k, v in per_dog.items() if v != {"face", "body"})
        if no_panel or odd:
            raise SystemExit(f"칸 사진이 없는 개: {no_panel} | face·body 가 한 장씩이 아닌 개: {odd}")
        rows.sort(key=lambda x: (x[3] != "dev", x[1], order.get(x[4], 9)))
        header = ["image_id", "dog_id", "path", "split", "panel"]
    else:
        for r in dogs:
            for fname in [x for x in r["photos"].split(";") if x]:
                if not (photos / fname).exists():
                    missing.append(fname)
                rows.append([Path(fname).stem, r["dog_id"], f"{root}/{fname}", split_of[r["dog_id"]]])
        rows.sort(key=lambda x: (x[3] != "dev", x[1], x[0]))
        header = ["image_id", "dog_id", "path", "split"]
    if missing:
        raise SystemExit(f"사진 파일이 없음: {missing[:10]} … 모두 {len(missing)}개")
    if len({r[0] for r in rows}) != len(rows):
        raise SystemExit("image_id 가 겹침")

    target = d / name
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
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
