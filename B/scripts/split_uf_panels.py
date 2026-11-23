"""UF 사진을 칸(얼굴 / 전신)으로 나눈다. 원본은 그대로 두고 칸 사진을 따로 저장한다.

UF 사진 한 장에는 흰 바탕 위에 칸 두 개가 나란히 놓여 있다 (왼쪽 얼굴, 오른쪽 전신).
칸 경계는 추측하지 않고 픽셀로 찾는다:
  - 세로 줄(열)마다 '거의 흰색'인 픽셀의 비율을 잰다. 전부 흰색인 열은 칸 사이의 틈이다.
    칸은 옅은 회색 테두리(약 6px)를 두르고 있어서, 칸 안의 흰 벽은 위아래 테두리 때문에 틈으로 잡히지 않는다.
  - 틈이 아닌 열이 이어진 구간이 칸이다. 칸마다 같은 방법으로 위아래 여백도 잘라 낸다.
  - 테두리: 칸 가장자리에서 안쪽으로 "밝기가 고르고 옅은 회색인 줄"이 몇 줄 이어지는지 잰다.
    흰 벽이 테두리처럼 보이는 사진이 있어, 사진마다 잰 값이 아니라 전체 중앙값만큼 네 변에서 잘라 낸다.
  - 칸이 2개가 아니거나 크기가 다른 사진들과 많이 다르면 check 열에 적는다. 그런 사진은 사람이 본다.

입력:  {src}/photos/*.jpg, {src}/photos.csv
출력:  {src}/panels/UF001_face.jpg, UF001_body.jpg
       {src}/panels.csv   dog_id, panel, file, width, height, x0, y0, x1, y1, src_file, check
                          (x0~y1 은 원본 사진에서 잘라 낸 위치. 테두리를 뺀 뒤의 값)
       (--contact-sheet) {src}/panels_check/sheet_face_*.jpg, sheet_body_*.jpg   눈으로 훑어볼 모아 보기

  - 확대하지 않는다. 잘라 낸 크기 그대로 JPEG(품질 95)로 저장한다. 전처리는 추출기가 한다.
  - 이미 있는 칸 사진은 다시 쓰지 않는다. 다시 만들려면 --force.
  - 칸이 2개로 나오지 않은 사진은 저장하지 않고 목록만 보여 준다.

사용 (프로젝트 루트에서):
  python scripts/split_uf_panels.py --src data/uf_dogs --dry-run          # 경계만 찾아서 보고
  python scripts/split_uf_panels.py --src data/uf_dogs --contact-sheet    # 저장 + 모아 보기

사진과 칸 사진, 모아 보기는 data/ 아래에만 두고 git·보고서·발표에 넣지 않는다 (UF 이용 조건).
"""
import argparse
import collections
import csv
import os
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = ["face", "body"]          # 왼쪽 → 오른쪽 순서


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 가 이어진 구간들의 (시작, 끝). 끝은 포함하지 않는다."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def find_panels(img: Image.Image, white: int, frac: float, min_size: int, min_gap: int) -> list[tuple]:
    """칸들의 (x0, y0, x1, y1). 왼쪽부터."""
    a = np.asarray(img.convert("RGB"))
    is_white = a.min(axis=2) >= white                       # (H, W)
    content = is_white.mean(axis=0) < frac                  # 여백이 아닌 열
    cols = runs(content)
    merged = []                                             # 좁은 흰 틈(칸 안의 흰 벽 등)은 같은 칸으로 본다
    for x0, x1 in cols:
        if merged and x0 - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], x1)
        else:
            merged.append((x0, x1))
    boxes = []
    for x0, x1 in merged:
        if x1 - x0 < min_size:
            continue
        rows = runs(is_white[:, x0:x1].mean(axis=1) < frac)
        if not rows:
            continue
        y0, y1 = rows[0][0], rows[-1][1]
        if y1 - y0 >= min_size:
            boxes.append((x0, y0, x1, y1))
    return boxes


def frame_width(m: np.ndarray, box: tuple, side: str, max_w: int = 20) -> int:
    """칸의 한 변에서 안쪽으로 테두리(밝기가 고른 옅은 회색 줄)가 몇 줄 이어지는지.

    m: 채널 최솟값 (H, W). 둥근 모서리를 피하려고 줄의 가운데 60% 만 본다.
    """
    x0, y0, x1, y1 = box
    n = 0
    for k in range(max_w):
        if side in ("left", "right"):
            x = x0 + k if side == "left" else x1 - 1 - k
            lo, hi = y0 + int(0.2 * (y1 - y0)), y1 - int(0.2 * (y1 - y0))
            line = m[lo:hi, x]
        else:
            y = y0 + k if side == "top" else y1 - 1 - k
            lo, hi = x0 + int(0.2 * (x1 - x0)), x1 - int(0.2 * (x1 - x0))
            line = m[y, lo:hi]
        if line.std() < 4 and 190 <= np.median(line) <= 232:
            n += 1
        else:
            break
    return n


def contact_sheet(files: list[Path], out: Path, per_row: int = 10, cell: int = 110) -> None:
    rows = (len(files) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * cell, rows * (cell + 12)), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((cell - 4, cell - 4))
        x, y = (i % per_row) * cell, (i // per_row) * (cell + 12)
        d.text((x + 2, y), f.stem.split("_")[0], fill="black")
        sheet.paste(im, (x + 2, y + 12))
    sheet.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/uf_dogs")
    ap.add_argument("--white", type=int, default=235, help="세 채널 모두 이 값 이상이면 흰색으로 본다")
    ap.add_argument("--frac", type=float, default=0.99, help="열(행)의 흰 픽셀 비율이 이 값 이상이면 여백")
    ap.add_argument("--min-size", type=int, default=60, help="칸의 최소 가로·세로(px)")
    ap.add_argument("--min-gap", type=int, default=2, help="이보다 좁은 흰 틈은 칸 경계로 보지 않는다(px)")
    ap.add_argument("--keep-frame", action="store_true", help="회색 테두리를 잘라 내지 않는다")
    ap.add_argument("--dry-run", action="store_true", help="경계만 찾아 보고하고 저장하지 않는다")
    ap.add_argument("--force", action="store_true", help="이미 있는 칸 사진을 다시 쓴다")
    ap.add_argument("--contact-sheet", action="store_true", help="칸 사진 모아 보기를 panels_check/ 에 만든다")
    args = ap.parse_args()
    src = Path(args.src)
    out = src / "panels"

    with open(src / "photos.csv", encoding="utf-8-sig") as f:
        photos = [(r["dog_id"], r["file"]) for r in csv.DictReader(f)]

    found = []                                   # (dog_id, file, (W, H), boxes)
    for dog_id, fname in photos:
        img = Image.open(src / "photos" / fname)
        found.append((dog_id, fname, img.size, find_panels(img, args.white, args.frac, args.min_size, args.min_gap)))

    # 테두리 두께: 칸마다 네 변에서 재고, 전체 중앙값을 쓴다
    measured = collections.defaultdict(list)
    for dog_id, fname, _, boxes in found:
        m = np.asarray(Image.open(src / "photos" / fname).convert("RGB")).min(axis=2).astype(float)
        for box in boxes:
            for side in ("left", "right", "top", "bottom"):
                measured[side].append(frame_width(m, box, side))
    all_w = [w for ws in measured.values() for w in ws]
    frame = 0 if (args.keep_frame or not all_w) else int(statistics.median(all_w))
    raw = found
    found = [(d, f, size, [(x0 + frame, y0 + frame, x1 - frame, y1 - frame) for x0, y0, x1, y1 in boxes])
             for d, f, size, boxes in raw]

    two = [f for f in found if len(f[3]) == 2]
    med = {}
    if two:
        for k, name in enumerate(NAMES):
            med[name] = (statistics.median(b[3][k][2] - b[3][k][0] for b in two),
                         statistics.median(b[3][k][3] - b[3][k][1] for b in two))

    rows, problems, n_saved, n_kept = [], [], 0, 0
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    for dog_id, fname, (W, H), boxes in found:
        if len(boxes) != 2:
            problems.append(f"{dog_id} ({fname}, {W}x{H}): 칸 {len(boxes)}개 {boxes}")
            continue
        for name, (x0, y0, x1, y1) in zip(NAMES, boxes):
            w, h = x1 - x0, y1 - y0
            check = []
            if abs(w - med[name][0]) > 0.15 * med[name][0]:
                check.append(f"가로가 중앙값({med[name][0]:.0f})과 15% 넘게 다름")
            if abs(h - med[name][1]) > 0.15 * med[name][1]:
                check.append(f"세로가 중앙값({med[name][1]:.0f})과 15% 넘게 다름")
            pname = f"{Path(fname).stem}_{name}.jpg"
            if not args.dry_run:
                target = out / pname
                if target.exists() and not args.force:
                    n_kept += 1
                else:
                    tmp = target.with_name(target.name + ".tmp")
                    Image.open(src / "photos" / fname).convert("RGB").crop((x0, y0, x1, y1)).save(
                        tmp, format="JPEG", quality=95)
                    os.replace(tmp, target)
                    n_saved += 1
            rows.append([dog_id, name, pname, w, h, x0, y0, x1, y1, fname, " | ".join(check)])

    # ---- 보고 ----
    print(f"사진 {len(found)}장 | 원본 크기: {dict(collections.Counter(f[2] for f in found).most_common(4))} 등 "
          f"{len({f[2] for f in found})}종")
    print("칸 수 분포:", dict(sorted(collections.Counter(len(f[3]) for f in found).items())))

    def dist(vals):
        v = sorted(vals)
        return f"최소 {v[0]:>3} / 중앙 {v[len(v) // 2]:>3} / 최대 {v[-1]:>3}"
    print("\n테두리 두께 측정 (px, 칸의 네 변)")
    for side, label in (("left", "왼쪽"), ("right", "오른쪽"), ("top", "위"), ("bottom", "아래")):
        c = collections.Counter(measured[side])
        print(f"  {label:<4}: 중앙 {int(statistics.median(measured[side]))} | 분포 {dict(sorted(c.items()))}")
    print(f"  → 네 변에서 잘라 내는 두께: {frame}px" + (" (--keep-frame)" if args.keep_frame else " (전체 중앙값)"))
    raw_two = [f for f in raw if len(f[3]) == 2]
    if raw_two:
        gaps = sorted(f[3][1][0] - f[3][0][2] for f in raw_two)
        print(f"칸 사이 순백 틈(테두리 바깥 기준): 최소 {gaps[0]} / 중앙 {gaps[len(gaps) // 2]} / 최대 {gaps[-1]} px")
    if two:
        print("\n칸 2개인 사진의 위치 (px, 테두리를 뺀 뒤)")
        print("  왼쪽 칸 x 시작 :", dist([f[3][0][0] for f in two]))
        print("  왼쪽 칸 x 끝   :", dist([f[3][0][2] for f in two]))
        print("  오른쪽 칸 x 시작:", dist([f[3][1][0] for f in two]))
        print("  오른쪽 칸 x 끝 :", dist([f[3][1][2] for f in two]))
        print("  왼쪽 칸  가로×세로:", dist([f[3][0][2] - f[3][0][0] for f in two]), "×",
              dist([f[3][0][3] - f[3][0][1] for f in two]))
        print("  오른쪽 칸 가로×세로:", dist([f[3][1][2] - f[3][1][0] for f in two]), "×",
              dist([f[3][1][3] - f[3][1][1] for f in two]))
        wider_left = [f[0] for f in two if (f[3][0][2] - f[3][0][0]) >= (f[3][1][2] - f[3][1][0])]
        print(f"  왼쪽 칸이 오른쪽 칸보다 넓거나 같은 사진: {len(wider_left)}장 {wider_left}")
        short = sum(1 for r in rows if min(r[3], r[4]) < 224)
        print(f"  짧은 변이 224px 미만인 칸: {short} / {len(rows)}")
    print(f"\n[칸이 2개가 아닌 사진] {len(problems)}장")
    for p in problems:
        print("  ", p)
    flagged = [r for r in rows if r[10]]
    print(f"[크기가 다른 사진들과 다른 칸] {len(flagged)}개")
    for r in flagged:
        print(f"   {r[2]}  {r[3]}x{r[4]}  x={r[5]}~{r[7]}  {r[10]}")
    print("\n사진별 위치 예 (전체는 panels.csv):")
    print("   dog_id  원본      왼쪽 칸 x0~x1 (가로×세로)   오른쪽 칸 x0~x1 (가로×세로)")
    for dog_id, fname, (W, H), boxes in found[:8]:
        if len(boxes) == 2:
            (a0, b0, a1, b1), (c0, d0, c1, d1) = boxes
            print(f"   {dog_id}  {W}x{H}   {a0:>3}~{a1:<3} ({a1 - a0}×{b1 - b0})          {c0:>3}~{c1:<3} ({c1 - c0}×{d1 - d0})")

    if args.dry_run:
        print("\n(--dry-run: 저장하지 않음)")
        return
    target = src / "panels.csv"
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dog_id", "panel", "file", "width", "height", "x0", "y0", "x1", "y1", "src_file", "check"])
        w.writerows(rows)
    os.replace(tmp, target)
    print(f"\n칸 사진 {len(rows)}장 (새로 저장 {n_saved}, 이미 있음 {n_kept}) → {out}")
    print(f"저장 → {target}")
    if args.contact_sheet:
        cdir = src / "panels_check"
        cdir.mkdir(exist_ok=True)
        for name in NAMES:
            files = [out / r[2] for r in rows if r[1] == name]
            for k in range(0, len(files), 50):
                contact_sheet(files[k:k + 50], cdir / f"sheet_{name}_{k // 50 + 1}.jpg")
        print(f"모아 보기 → {cdir}  (눈으로 확인용. 밖으로 내보내지 않는다)")


if __name__ == "__main__":
    main()
