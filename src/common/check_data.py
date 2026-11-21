"""[B] 데이터 점검: 실제 데이터로 처음 돌리기 전에 확인한다. 읽기만 하고 아무것도 바꾸지 않는다.

  1) paths 설정(DOGMIX_PATHS 반영)의 경로가 실제로 있는지
  2) 분할 파일 줄 수, 제외 후 학습 클래스 수(127, dingo 제외 시 126), 제외 클래스 이름이 실제 목록과 맞는지
  3) Tsinghua 주석 xml에서 bodybndbox 태그를 찾는지, 몸통 박스를 못 찾은 비율,
     박스가 여러 개인 사진, xml의 size와 실제 이미지 크기가 같은지(앞 300장)
  4) 평가셋 인덱스(A의 eval_index.csv)

예:
  python -m src.common.check_data               # 학습셋에서 2,000장을 무작위로 골라 xml 점검
  python -m src.common.check_data --sample 0    # 학습셋 전체
"""
import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from src.common import evalset, tsinghua
from src.common.config import load_yaml


def find_xml(annotation_root: str, rel_path: str) -> Path | None:
    """tsinghua.load_crop과 같은 순서로 주석 파일을 찾는다."""
    p = (Path(annotation_root) / rel_path).with_suffix(".xml")
    if p.exists():
        return p
    p = Path(annotation_root) / (rel_path + ".xml")
    return p if p.exists() else None


def check_paths(paths: dict, todo: list[str]) -> None:
    print("== 1) 경로가 실제로 있는지 ==")
    src = os.environ.get("DOGMIX_PATHS")
    print(f"  설정 파일: {src if src else 'configs/paths.yaml'}")
    t = paths["tsinghua"]
    required = {
        "tsinghua.image_root": t["image_root"],
        "tsinghua.annotation_root": t["annotation_root"],
        "tsinghua.train_list": t["train_list"],
        "tsinghua.val_list": t["val_list"],
        "evalset.index_csv": paths["evalset"]["index_csv"],
    }
    for k, p in required.items():
        ok = os.path.exists(p)
        print(f"  [{'있음' if ok else '없음'}] {k}: {p}")
        if not ok:
            todo.append(f"경로 없음: {k}")
    for k, p in paths["outputs"].items():
        print(f"  [{'있음' if os.path.exists(p) else '없음(실행할 때 만들어짐)'}] outputs.{k}: {p}")


def check_classes(paths: dict, todo: list[str]) -> list:
    print("\n== 2) 분할 파일과 학습 클래스 수 ==")
    t, c = paths["tsinghua"], paths["classes"]
    if not (os.path.exists(t["train_list"]) and os.path.exists(t["val_list"])):
        print("  분할 파일이 없어 건너뜀")
        return []
    raw_train, raw_val = tsinghua.parse_split(t["train_list"]), tsinghua.parse_split(t["val_list"])
    train = tsinghua.filter_classes(raw_train, c["exclude"], c["exclude_dingo"])
    val = tsinghua.filter_classes(raw_val, c["exclude"], c["exclude_dingo"])
    print(f"  train 분할 파일: {len(raw_train)}줄 → 제외 후 {len(train)}장")
    print(f"  val 분할 파일:   {len(raw_val)}줄 → 제외 후 {len(val)}장")

    raw_names = {it.class_name for it in raw_train}
    classes = tsinghua.class_list(train)
    expected = 126 if c["exclude_dingo"] else 127
    ok = len(classes) == expected
    print(f"  학습 클래스: 제외 전 {len(raw_names)}개 → 제외 후 {len(classes)}개 "
          f"(기대값 {expected}): {'OK' if ok else '불일치 → 확인 필요'}")
    if not ok:
        todo.append(f"학습 클래스 수 {len(classes)} (기대값 {expected})")
    for name in c["exclude"]:
        if name in raw_names:
            print(f"  제외 클래스 '{name}': 목록에 있음 → 제외됨")
        else:
            near = sorted(n for n in raw_names if n.lower() == name.lower())
            print(f"  제외 클래스 '{name}': 목록에 없음 → 철자 확인 필요 (대소문자만 다른 이름: {near})")
            todo.append(f"제외 클래스 '{name}'이 목록에 없음")
    print(f"  dingo: {'목록에 있음' if 'dingo' in raw_names else '목록에 없음'}, exclude_dingo={c['exclude_dingo']}")
    only_val = sorted({it.class_name for it in val} - set(classes))
    if only_val:
        print(f"  val에만 있는 클래스: {only_val} → 확인 필요")
        todo.append("val에만 있는 클래스가 있음")
    return train


def check_boxes(paths: dict, items: list, n_sample: int, todo: list[str]) -> None:
    print("\n== 3) 주석 xml의 bodybndbox ==")
    t = paths["tsinghua"]
    if not items:
        print("  학습 목록이 없어 건너뜀")
        return
    sample = items if (n_sample <= 0 or n_sample >= len(items)) else random.Random(0).sample(items, n_sample)
    n_img_missing = n_xml_missing = n_body = n_fallback = n_none = n_multi = 0
    n_size_checked = n_size_diff = n_box_outside = 0
    first_xml = None
    for i, it in enumerate(sample):
        img_path = Path(t["image_root"]) / it.rel_path
        if not img_path.exists():
            n_img_missing += 1
        xml_path = find_xml(t["annotation_root"], it.rel_path)
        if xml_path is None:
            n_xml_missing += 1
            n_none += 1
            continue
        first_xml = first_xml or xml_path
        root = ET.parse(xml_path).getroot()
        bodies = root.findall(".//bodybndbox")
        n_multi += len(bodies) > 1
        box = tsinghua.read_body_box(xml_path)
        if box is None:
            n_none += 1
        elif bodies:
            n_body += 1
        else:
            n_fallback += 1
        # 앞 300장은 이미지를 열어 xml의 <size>와 실제 크기, 박스가 이미지 안에 있는지 본다
        if i < 300 and img_path.exists():
            with Image.open(img_path) as im:
                w, h = im.size
            n_size_checked += 1
            xw, xh = root.findtext(".//size/width"), root.findtext(".//size/height")
            if xw is not None and xh is not None and (int(float(xw)), int(float(xh))) != (w, h):
                n_size_diff += 1
            if box is not None and (box[2] > w or box[3] > h or box[0] < 0 or box[1] < 0):
                n_box_outside += 1
    n = len(sample)
    print(f"  점검 {n}장 (학습셋 {len(items)}장 중)")
    print(f"  이미지 파일 없음: {n_img_missing}장")
    print(f"  xml 파일 없음: {n_xml_missing}장")
    print(f"  bodybndbox 태그로 박스를 읽음: {n_body}장 ({n_body / n:.1%})")
    print(f"  bodybndbox 없이 bndbox로 읽음: {n_fallback}장 ({n_fallback / n:.1%})")
    print(f"  몸통 박스를 못 찾음(전체 이미지를 씀): {n_none}장 ({n_none / n:.1%})")
    print(f"  bodybndbox가 2개 이상인 xml(첫 번째 박스만 씀): {n_multi}장 ({n_multi / n:.1%})")
    if n_size_checked:
        print(f"  이미지 {n_size_checked}장을 열어 봄: xml의 size와 실제 크기가 다름 {n_size_diff}장, "
              f"박스가 이미지 밖으로 나감 {n_box_outside}장")
        if n_size_diff:
            todo.append(f"xml의 size와 실제 이미지 크기가 다름 {n_size_diff}/{n_size_checked}장 (박스 좌표 기준 확인)")
        if n_box_outside:
            todo.append(f"몸통 박스가 이미지 밖으로 나감 {n_box_outside}/{n_size_checked}장 (크롭 때 잘라 냄)")
    if n_img_missing:
        todo.append(f"이미지 파일 없음 {n_img_missing}장")
    if n_body == 0:
        todo.append("bodybndbox 태그를 하나도 찾지 못함")
    elif n_none or n_fallback:
        todo.append(f"몸통 박스를 못 찾은 비율 {n_none / n:.1%}, bndbox로 대신 읽은 비율 {n_fallback / n:.1%}")
    if first_xml is None:
        print("  xml을 하나도 찾지 못함 → annotation_root 경로와 폴더 구조 확인 필요")
        return
    print(f"  예시 xml: {first_xml}")
    print(f"  이 xml의 태그: {sorted({e.tag for e in ET.parse(first_xml).getroot().iter()})}")
    print("  --- 앞 500자 ---")
    print(first_xml.read_text(encoding="utf-8", errors="replace")[:500])


def check_evalset(paths: dict, todo: list[str]) -> None:
    print("\n== 4) 평가셋 인덱스 ==")
    index_csv = paths["evalset"]["index_csv"]
    if not os.path.exists(index_csv):
        print("  eval_index.csv가 아직 없음 (A가 만듦) → 평가셋 추출과 평균점 섞기는 아직 못 돌림")
        return
    ev = evalset.load_index(index_csv)
    print(f"  사진 {len(ev)}장, 개 {ev['dog_id'].nunique()}마리, split: {ev['split'].value_counts().to_dict()}")
    print(f"  image_id 중복: {int(ev['image_id'].duplicated().sum())}개")
    probe = ev["path"].sample(min(200, len(ev)), random_state=0)
    n_missing = sum(not os.path.exists(p) for p in probe)
    print(f"  path {len(probe)}개 중 파일 없음: {n_missing}개")
    if n_missing:
        todo.append(f"평가셋 이미지 경로 없음 {n_missing}개")
    if ev["image_id"].duplicated().any():
        todo.append("평가셋 image_id 중복")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000, help="xml을 점검할 학습 이미지 수 (0이면 전체)")
    args = ap.parse_args()

    paths = load_yaml("paths")
    todo: list[str] = []
    check_paths(paths, todo)
    items = check_classes(paths, todo)
    check_boxes(paths, items, args.sample, todo)
    check_evalset(paths, todo)

    print("\n== 요약 ==")
    if todo:
        print(f"  확인 필요 {len(todo)}건:")
        for line in todo:
            print(f"    - {line}")
    else:
        print("  확인 필요 없음")


if __name__ == "__main__":
    main()
