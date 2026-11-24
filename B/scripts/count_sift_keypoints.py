"""SIFT 사진당 키포인트 수: Tsinghua 학습셋 크롭과 UF dev 패널을 비교한다 (집계값만 저장).

배경: UF 평가 사진의 SIFT BoVW 히스토그램에서 0 인 칸 비율이 학습셋보다 높다 (0.60 → 0.80,
docs/A_VS_B_INTERPRETATION.md ⑥(a)). 키포인트가 덜 잡힌 탓인지 센다.

C 의 추출 코드(C/code/sift_extract_train.py, make_uf_features.py)와 같은 전처리를 따른다.
  흑백으로 읽기 → (학습셋만) 몸통 박스 크롭 → 336×336 리사이즈 → cv2.SIFT_create() 기본값으로 검출
C 의 파일은 읽기만 하고 고치지 않는다. 특징(BoVW)은 다시 만들지 않고 키포인트 수와 크롭 크기만 센다.

입력
  --c-index    C 의 index_local_v2.csv (path, breed_name, x1, y1, x2, y2, split, keep_127). 몸통 박스를 여기서 읽는다
  --tsinghua   low-resolution/ 폴더 (C 의 path 에 있는 /content/low-resolution 을 이것으로 바꾼다)
  --uf-index   eval_index_UF_panels.csv (image_id, dog_id, path, split, panel). dev 행만 쓴다 (hold 는 쓰지 않는다)
  --n-train    학습셋에서 뽑을 사진 수 (시드 고정, 기본 2000)

출력
  results/sift_keypoints.csv   사진 집합별 키포인트 수 · 크롭 크기 요약

사용 (B/ 안에서)
  python scripts/count_sift_keypoints.py --c-index ../C/index_local_v2.csv --tsinghua ../data/low-resolution \
      --uf-index ../data/evalset_uf/eval_index_UF_panels.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIZE = 336
SEED = 0


def count(gray: np.ndarray, sift) -> int:
    return len(sift.detect(cv2.resize(gray, (SIZE, SIZE)), None))


def summarize(name: str, kp: list[int], wh: list[tuple[int, int]]) -> dict:
    k = np.array(kp, dtype=float)
    w = np.array([a for a, _ in wh], dtype=float)
    h = np.array([b for _, b in wh], dtype=float)
    return dict(image_set=name, images=len(k), kp_mean=k.mean(), kp_median=np.median(k),
                kp_p10=np.percentile(k, 10), kp_p90=np.percentile(k, 90), kp_under_100=float((k < 100).mean()),
                src_width_median=np.median(w), src_height_median=np.median(h),
                upscaled_frac=float(((w < SIZE) | (h < SIZE)).mean()), opencv=cv2.__version__)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c-index", type=Path, required=True)
    ap.add_argument("--tsinghua", type=Path, required=True)
    ap.add_argument("--uf-index", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    a = ap.parse_args()
    sift = cv2.SIFT_create()
    rows = []

    c = pd.read_csv(a.c_index)
    c = c[(c["split"] == "train") & c["keep_127"].astype(bool)]
    c = c.sample(n=min(a.n_train, len(c)), random_state=SEED)
    kp, wh, missing = [], [], 0
    for r in c.itertuples():
        p = a.tsinghua / r.path.split("/low-resolution/", 1)[1]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1
            continue
        crop = img[r.y1:r.y2, r.x1:r.x2]
        if crop.size == 0:
            missing += 1
            continue
        kp.append(count(crop, sift))
        wh.append((crop.shape[1], crop.shape[0]))
    if missing > len(c) * 0.01:
        raise SystemExit(f"학습셋 사진 {missing}/{len(c)} 장을 읽지 못함: --tsinghua 경로 확인")
    rows.append(summarize("Tsinghua train 몸통 크롭 (표본)", kp, wh))

    u = pd.read_csv(a.uf_index)
    u = u[u["split"] == "dev"]
    per = {"face": ([], []), "body": ([], [])}
    for r in u.itertuples():
        img = cv2.imread(str(r.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"UF 패널을 읽지 못함: {r.path}")
        per[r.panel][0].append(count(img, sift))
        per[r.panel][1].append((img.shape[1], img.shape[0]))
    rows.append(summarize("UF dev 얼굴 칸", *per["face"]))
    rows.append(summarize("UF dev 전신 칸", *per["body"]))
    rows.append(summarize("UF dev 전체", per["face"][0] + per["body"][0], per["face"][1] + per["body"][1]))

    df = pd.DataFrame(rows)
    df.round(3).to_csv(ROOT / "results" / "sift_keypoints.csv", index=False, encoding="utf-8-sig")
    print(df.round(2).to_string(index=False))
    if missing:
        print(f"(학습셋 읽기 실패 {missing}장 제외)")


if __name__ == "__main__":
    main()
