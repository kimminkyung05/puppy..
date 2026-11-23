"""스모크 테스트용 가짜 데이터: Tsinghua 형식 학습셋 + 믹스견 평가셋.

- 클래스 5개(그중 chinese_rural_dog는 제외 대상, 'Fila Braziliero'는 공백 폴더 이름 점검용)
- 클래스마다 고유 색 이미지. 평가셋은 두 클래스 색을 비율대로 섞은 이미지(정답 비율을 아는 믹스)
사용: python tests/make_toy_data.py /tmp/toy
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/toy")
CLASSES = {
    "1-n000001-Shiba_Dog": (220, 120, 40),
    "2-n000002-Poodle_a": (240, 240, 240),
    "3-n000003-Labrador_retriever": (30, 30, 30),
    "4-n000032-Fila Braziliero": (60, 160, 60),
    "5-n000121-chinese_rural_dog": (120, 120, 200),
}
rng = np.random.default_rng(0)


def img(color, noise=10):
    a = np.clip(np.array(color) + rng.normal(0, noise, (40, 40, 3)), 0, 255).astype(np.uint8)
    canvas = np.full((60, 60, 3), 128, np.uint8)
    canvas[10:50, 10:50] = a          # 몸통 박스 = (10,10)-(50,50)
    return Image.fromarray(canvas)


train, val = [], []
for folder, color in CLASSES.items():
    for k in range(12):
        name = f"n1{abs(hash(folder)) % 1000:03d}{k:02d}.jpg"
        rel = f"{folder}/{name}"
        p = root / "low-resolution" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img(color).save(p)
        x = root / "Low-Annotations" / Path(rel).with_suffix(".xml")
        x.parent.mkdir(parents=True, exist_ok=True)
        x.write_text("<annotation><object><bodybndbox><xmin>10</xmin><ymin>10</ymin>"
                     "<xmax>50</xmax><ymax>50</ymax></bodybndbox></object></annotation>")
        (val if k >= 10 else train).append(f".//{rel}")
(root / "split").mkdir(parents=True, exist_ok=True)
(root / "split" / "train.lst").write_text("﻿" + "\n".join(train), encoding="utf-8")
(root / "split" / "validation.lst").write_text("﻿" + "\n".join(val), encoding="utf-8")

# 평가셋: 두 클래스 색을 섞은 '믹스견'. 개마다 사진 2장
cols = list(CLASSES.values())[:4]
names = ["Shiba_Dog", "Poodle_a", "Labrador_retriever", "Fila Braziliero"]
rows, truth = [], []
for d in range(10):
    i, j = rng.choice(4, 2, replace=False)
    w = rng.uniform(0.2, 0.8)
    color = w * np.array(cols[i]) + (1 - w) * np.array(cols[j])
    for s in range(2):
        p = root / "eval" / f"dog{d}_{s}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        img(color, noise=3).crop((10, 10, 50, 50)).save(p)
        rows.append({"image_id": f"dog{d}_{s}", "dog_id": f"dog{d}", "path": str(p),
                     "split": "dev" if d < 5 else "test"})
    truth.append({"dog_id": f"dog{d}", names[i]: w, names[j]: 1 - w})
pd.DataFrame(rows).to_csv(root / "eval_index.csv", index=False)
pd.DataFrame(truth).fillna(0).to_csv(root / "eval_truth.csv", index=False)

(root / "paths.yaml").write_text(f"""tsinghua:
  image_root: {root}/low-resolution
  annotation_root: {root}/Low-Annotations
  train_list: {root}/split/train.lst
  val_list: {root}/split/validation.lst
evalset:
  index_csv: {root}/eval_index.csv
outputs:
  features_root: {root}/out/features
  preds_root: {root}/out/preds
  logs_root: {root}/out/logs
classes:
  exclude: [chinese_rural_dog, dhole, African_hunting_dog]
  exclude_dingo: false
""", encoding="utf-8")
print("toy data →", root)
