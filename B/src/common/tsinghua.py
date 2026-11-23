"""Tsinghua Dogs: 분할 파일 읽기, 클래스 정리, 몸통 박스 크롭.

분할 파일 한 줄 예: './/1043-n000001-Shiba_Dog/n100001.jpeg'
- 첫 줄에 BOM이 있을 수 있다.
- 'Fila Braziliero'처럼 폴더 이름에 공백이 있으므로 반드시 줄 단위로 읽는다.
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_CLASS_RE = re.compile(r"^(\d+)-(n\d+)-(.+)$")


@dataclass
class Item:
    image_id: str      # 예: n000001/n100001
    rel_path: str      # 예: 1043-n000001-Shiba_Dog/n100001.jpeg
    class_id: str      # 예: n000001
    class_name: str    # 예: Shiba_Dog


def parse_split(list_path: str) -> list[Item]:
    items = []
    with open(list_path, encoding="utf-8-sig") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            rel = line.lstrip("./")
            folder, fname = rel.rsplit("/", 1)
            m = _CLASS_RE.match(folder)
            if m is None:
                raise ValueError(f"클래스 폴더 이름을 해석할 수 없음: {folder}")
            _, cid, cname = m.groups()
            stem = fname.rsplit(".", 1)[0]
            items.append(Item(f"{cid}/{stem}", rel, cid, cname))
    return items


def filter_classes(items: list[Item], exclude: list[str], exclude_dingo: bool) -> list[Item]:
    drop = set(exclude) | ({"dingo"} if exclude_dingo else set())
    return [it for it in items if it.class_name not in drop]


def class_list(items: list[Item]) -> list[str]:
    """학습에 쓰는 클래스 이름 목록 (정렬 고정). 모든 담당이 같은 순서를 쓴다."""
    return sorted({it.class_name for it in items})


def read_body_box(xml_path: Path):
    """xml에서 몸통 박스(xmin, ymin, xmax, ymax)를 읽는다.

    Tsinghua 주석의 태그 이름은 실제 파일로 확인 필요.
    'bodybndbox'를 먼저 찾고, 없으면 'bndbox', 그것도 없으면 None.
    """
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    for tag in ("bodybndbox", "bndbox"):
        node = root.find(f".//{tag}")
        if node is not None:
            vals = [node.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
            if all(v is not None for v in vals):
                return tuple(int(float(v)) for v in vals)
    return None


def load_crop(item: Item, image_root: str, annotation_root: str) -> Image.Image:
    """이미지를 열고 몸통 박스로 자른다. 박스가 없으면 전체 이미지를 쓴다."""
    img = Image.open(Path(image_root) / item.rel_path).convert("RGB")
    xml_path = (Path(annotation_root) / item.rel_path).with_suffix(".xml")
    if not xml_path.exists():  # 확장자가 .jpeg.xml 형태일 가능성 대비
        xml_path = Path(annotation_root) / (item.rel_path + ".xml")
    box = read_body_box(xml_path)
    if box is not None:
        x0, y0, x1, y1 = box
        w, h = img.size
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            img = img.crop((x0, y0, x1, y1))
    return img
