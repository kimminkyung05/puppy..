"""UF 'Dog Breed Identification Photo Chart' 페이지 → DNA 표 · 개 표 · 설문 표 (UF 초안).

입력 (download_uf_dogs.py 가 만든 폴더):
  {src}/page.html, {src}/photos.csv
출력:
  {out}/dna_UF.csv    긴 형식: dog_id, breed_raw, pct, flag, line_raw
  {out}/dogs_UF.csv   개 단위: dog_id, n_breeds, pct_sum, is_purebred, source, license,
                      can_publish_photo, test_company, photos, notes
  {out}/human_UF.csv  설문 응답: dog_id, order, response_raw, response_pct, kind

원칙
  - 페이지 표기를 고치지 않는다. breed_raw 는 철자·대소문자 그대로, line_raw 는 줄 전체 그대로.
  - 'N% 품종' 형식에서 벗어난 줄은 읽힌 숫자를 pct 에 적되 flag 를 붙인다. 사람이 원문을 보고 확정한다.
  - 숫자를 읽지 못하면 pct 를 비운다 (추측으로 채우지 않는다).
  - 개 번호는 행 제목("Dog 92")에서 읽는다. 번호는 1~119 사이에 빈 곳이 있다 (개는 100마리).

사용 (프로젝트 루트에서):
  python scripts/parse_uf_dna.py --src data/uf_dogs --out data/evalset_uf

출력 파일은 data/ 아래에 두고 git 에 올리지 않는다.
"""
import argparse
import collections
import csv
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup

DOG_RE = re.compile(r"dog[\s_-]*0*(\d{1,3})(?!\d)", re.I)
STRICT = re.compile(r"^(\d{1,3}(?:\.\d+)?)% (\S.*)$")                 # 표준: '25% Boxer'
LOOSE = re.compile(r"^(\d{1,3}(?:\.\d+)?)\s*([^\w\s]*)\s*(\S.*)$")    # 숫자, 사이 기호, 품종
NOT_BREED = {"No Predominant Breed"}                                   # 설문에서 품종이 아닌 응답

SOURCE, LICENSE, CAN_PUBLISH, TEST_COMPANY = "UF", "교육용", "불가", "미상"


def clean(s: str) -> str:
    """앞뒤·중복 공백만 정리한다. 철자는 건드리지 않는다."""
    return " ".join(s.replace("\xa0", " ").split())


def parse_line(line: str):
    """(breed_raw, pct 또는 None, flag)"""
    m = STRICT.match(line)
    if m:
        return m.group(2), float(m.group(1)), ""
    m = LOOSE.match(line)
    if not m:
        return line, None, "숫자 못 읽음"
    num, mid, breed = m.groups()
    if mid == "":
        flag = "퍼센트 기호 없음"
    elif mid == "%":
        flag = "띄어쓰기 다름"
    else:
        flag = f"숫자 뒤 기호 '{mid}'"
    return breed, float(num), flag


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """임시 파일에 쓴 뒤 이름을 바꾼다. 한글 값이 있어 Excel 에서 깨지지 않게 utf-8-sig 로 쓴다."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/uf_dogs")
    ap.add_argument("--out", default="data/evalset_uf")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    photos = collections.defaultdict(list)
    alts = {}
    with open(src / "photos.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            photos[r["dog_id"]].append(r["file"])
            alts.setdefault(r["dog_id"], r["alt"])
    missing_files = [f for fs in photos.values() for f in fs if not (src / "photos" / f).exists()]

    soup = BeautifulSoup((src / "page.html").read_text(encoding="utf-8"), "html.parser")
    table = soup.find("table")
    if table is None:
        raise SystemExit("page.html 에서 표를 찾지 못함. 페이지 구조가 바뀌었는지 확인 필요")

    dna_rows, dog_rows, human_rows = [], [], []
    seen, problems = set(), []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 3:
            problems.append(f"칸이 3개가 아닌 행: {clean(tr.get_text(' '))[:60]}")
            continue
        head = tds[1].find(["h3", "h4"])
        m = DOG_RE.search(clean(head.get_text(" "))) if head else None
        if m is None:
            continue                                   # 머리 행 (The Dogs / DNA Results / Survey Results)
        dog_id = f"UF{int(m.group(1)):03d}"
        if dog_id in seen:
            problems.append(f"{dog_id}: 같은 번호의 행이 두 번 나옴")
        seen.add(dog_id)
        notes = [f"페이지 제목 '{clean(head.get_text(' '))}'"]

        # DNA 칸
        lis = [clean(li.get_text(" ")) for li in tds[1].find_all("li")]
        loose_text = [t for t in tds[1].get_text("\n", strip=True).split("\n")][1:]
        if len(loose_text) != len(lis):
            notes.append("DNA 칸에 목록 밖 글자 있음")
        pcts, breeds = [], []
        for line in lis:
            breed, pct, flag = parse_line(line)
            dna_rows.append([dog_id, breed, "" if pct is None else fmt(pct), flag, line])
            breeds.append(breed)
            if pct is not None:
                pcts.append(pct)
        total = round(sum(pcts), 2)
        is_pure = len(lis) == 1 and len(pcts) == 1 and abs(pcts[0] - 100) < 1e-9
        if not lis:
            notes.append("DNA 줄 없음")
        if total > 100.005:
            notes.append(f"합 100 초과({fmt(total)})")
        if len(pcts) == len(lis) and any(a < b for a, b in zip(pcts, pcts[1:])):
            notes.append("비율이 큰 순서가 아님")
        if len(set(b.lower() for b in breeds)) != len(breeds):
            notes.append("같은 품종이 두 번 나옴")
        if any(r[0] == dog_id and r[3] for r in dna_rows):
            notes.append("flag 붙은 줄 있음")
        alt_m = DOG_RE.search(alts.get(dog_id, ""))
        if dog_id in alts and (alt_m is None or f"UF{int(alt_m.group(1)):03d}" != dog_id):
            notes.append(f"사진 alt 가 번호와 다름('{alts[dog_id]}')")
        if dog_id not in photos:
            notes.append("사진 없음")
        dog_rows.append([dog_id, len(lis), fmt(total), is_pure, SOURCE, LICENSE, CAN_PUBLISH,
                         TEST_COMPANY, ";".join(photos.get(dog_id, [])), " | ".join(notes)])

        # 설문 칸 (응답 비율은 페이지에 없다. 적힌 순서만 남긴다)
        s_head = tds[2].find(["h3", "h4"])
        if s_head is None or clean(s_head.get_text(" ")) != "Top Responses":
            problems.append(f"{dog_id}: 설문 칸 제목이 'Top Responses' 가 아님")
        for k, li in enumerate(tds[2].find_all("li"), 1):
            resp = clean(li.get_text(" "))
            human_rows.append([dog_id, k, resp, "", "기타" if resp in NOT_BREED else "품종"])

    write_csv(out / "dna_UF.csv", ["dog_id", "breed_raw", "pct", "flag", "line_raw"], dna_rows)
    write_csv(out / "dogs_UF.csv", ["dog_id", "n_breeds", "pct_sum", "is_purebred", "source", "license",
                                    "can_publish_photo", "test_company", "photos", "notes"], dog_rows)
    write_csv(out / "human_UF.csv", ["dog_id", "order", "response_raw", "response_pct", "kind"], human_rows)

    # ---- 검증 출력 ----
    n_pure = sum(1 for d in dog_rows if d[3])
    print(f"개 {len(dog_rows)}마리 (순종 {n_pure}, 믹스 {len(dog_rows) - n_pure}) | DNA 줄 {len(dna_rows)} | "
          f"품종 표기 {len({r[1] for r in dna_rows})}종 | 설문 줄 {len(human_rows)}")
    print("개당 DNA 줄 수:", dict(sorted(collections.Counter(d[1] for d in dog_rows).items())))

    sums = [(d[0], float(d[2])) for d in dog_rows]
    eq = [s for s in sums if abs(s[1] - 100) < 0.005]
    over = [s for s in sums if s[1] > 100.005]
    under = sorted((s for s in sums if s[1] < 99.995), key=lambda s: s[1])
    print(f"\n[비율 합] 100: {len(eq)}마리 | 100 초과: {len(over)}마리 | 100 미만: {len(under)}마리")
    print("  100 초과:", ", ".join(f"{d}={fmt(s)}" for d, s in over) or "없음")
    for d, _ in over:
        for r in dna_rows:
            if r[0] == d:
                print(f"      {d}  {r[4]!r}")
    if under:
        bins = collections.Counter("90 이상" if s >= 90 else "75~90" if s >= 75 else "60~75" if s >= 60 else "60 미만"
                                   for _, s in under)
        print("  100 미만 분포:", {k: bins.get(k, 0) for k in ["90 이상", "75~90", "60~75", "60 미만"]})
        print("  100 미만 (합 작은 순):")
        for i in range(0, len(under), 6):
            print("     ", "  ".join(f"{d}={fmt(s):>6}" for d, s in under[i:i + 6]))

    print("\n[숫자를 못 읽은 줄]", [r for r in dna_rows if r[2] == ""] or "없음")
    print("[사진이 없는 개]", [d[0] for d in dog_rows if not d[8]] or "없음")
    print("[DNA가 없는 개]", [d[0] for d in dog_rows if d[1] == 0] or "없음")
    print("[표에 없는데 사진만 있는 번호]", sorted(set(photos) - seen) or "없음")
    print("[photos.csv 에 있는데 파일이 없음]", missing_files or "없음")
    print("[그 밖의 구조 문제]", problems or "없음")

    print("\n[flag 붙은 줄: 원문 | 읽은 값]")
    for r in dna_rows:
        if r[3]:
            print(f"  {r[0]}  원문 {r[4]!r}  →  pct={r[2]}, breed_raw={r[1]!r}  ({r[3]})")
    print("\n[개 단위 표시]")
    for d in dog_rows:
        marks = [n for n in d[9].split(" | ") if not n.startswith("페이지 제목")]
        if marks:
            print(f"  {d[0]}  {' | '.join(marks)}")
    print(f"\n저장 → {out}  (dna_UF.csv, dogs_UF.csv, human_UF.csv)")


if __name__ == "__main__":
    main()
