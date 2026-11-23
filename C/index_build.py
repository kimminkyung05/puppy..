"""
인덱스 생성: 이미지 경로 -> image_id, 품종, 분할, 127클래스 필터.

image_id 형식: "폴더번호/키" (예: n000032/n112628)
  - 키: 파일명에서 확장자(.xml, .jpg, .jpeg, .png)를 반복 제거한 값
"""
import os, re
import pandas as pd

EXCLUDE = {"chinese_rural_dog", "dhole", "African_hunting_dog"}


def key_of(path):
    name = os.path.basename(path)
    while True:
        new = re.sub(r"\.(xml|jpg|jpeg|png)$", "", name, flags=re.IGNORECASE)
        if new == name:
            return name
        name = new


def load_split(lst_path):
    """train.lst / validation.lst 읽기 (utf-8-sig). 중복 키는 첫 번째만 유지."""
    keys = []
    with open(lst_path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if s:
                keys.append(key_of(s))
    return keys


def build_index(photo_root, train_lst, val_lst):
    """
    photo_root: /content/low-resolution/<품종>/ 아래 jpg 경로를 찾는다.
    반환: DataFrame(image_id, breed_name, split, keep_127, path)
    """
    train_keys = set(load_split(train_lst))
    val_keys = set(load_split(val_lst))

    rows = []
    for root, _, files in os.walk(photo_root):
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(root, fn)
            k = key_of(fn)
            breed = os.path.basename(os.path.dirname(path))
            m = re.search(r"(n\d{6})", path)
            folder = m.group(1) if m else ""
            split = "validation" if k in val_keys else ("train" if k in train_keys else None)
            rows.append({
                "image_id": f"{folder}/{k}",
                "breed_name": breed,
                "split": split,
                "keep_127": breed not in EXCLUDE,
                "path": path,
            })
    df = pd.DataFrame(rows)
    df = df[df["split"].notna()].drop_duplicates("image_id", keep="first")
    return df.reset_index(drop=True)
