"""믹스견 평가셋 인덱스 읽기 (A가 만든 eval_index.csv).

열: image_id, dog_id, path, split
- path: D의 SAM 3 크롭 이미지 경로
- split: dev 또는 test
"""
import pandas as pd
from PIL import Image

REQUIRED = ["image_id", "dog_id", "path", "split"]


def load_index(index_csv: str, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"eval_index.csv에 열이 없음: {missing}")
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
