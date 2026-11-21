"""UF Shelter Medicine 'Dog Breed Identification Photo Chart' 저장 (교육 목적).

사용:
  pip install requests beautifulsoup4
  python download_uf_dogs.py                     # 기본: ./uf_dogs 에 저장
  python download_uf_dogs.py data/uf_dogs
  python download_uf_dogs.py data/uf_dogs --refresh   # page.html 을 다시 받음

만드는 것:
  page.html      페이지 원본 (DNA 비율을 옮겨 적을 때 브라우저로 열어 봄)
  page_text.txt  페이지 글자만 (검색·복사용)
  photos/        UF001.jpg 처럼 개 번호로 저장한 사진
  photos.csv     dog_id, 파일명, 원본 주소, alt 글자

개 번호는 표 각 행의 제목("Dog 92")에서 읽는다. 사진의 alt 글자와 파일명은 틀린 곳이 있어
(Dog 92 의 alt 가 "dog 89", Dog 102 의 파일명이 dog-1021.jpg) 제목이 없을 때만 쓴다.

이어서 받기: 이미 있는 page.html 과 사진은 다시 요청하지 않는다. 사진은 임시 파일에 쓴 뒤 이름을 바꾼다.

주의: 사진은 수업 프로젝트 실험에만 쓰고 보고서·발표·GitHub 등에 싣지 않는다
     (UF 이용 조건: 교육 목적 저장은 허용, 게재는 서면 허락 필요).
"""
import csv
import os
import re
import sys
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

URL = ("https://sheltermedicine.vetmed.ufl.edu/research/current-studies/"
       "dog-breed-identification/dog-breed-dna-and-survey-results/")
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = ARGS[0] if ARGS else "uf_dogs"
REFRESH = "--refresh" in sys.argv
HEADERS = {"User-Agent": "Mozilla/5.0 (university class project; educational use)"}
DOG_RE = re.compile(r"dog[\s_-]*0*(\d{1,3})(?!\d)", re.I)
WAIT = 1.0                                                      # 요청 사이 간격(초). 서버에 부담 주지 않기


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def dog_number(img):
    """(개 번호, 번호를 읽은 곳). 행 제목 → 파일명 → alt 순서."""
    tr = img.find_parent("tr")
    if tr is not None:
        for h in tr.find_all(["h2", "h3", "h4"]):
            m = DOG_RE.search(h.get_text(" ", strip=True))
            if m:
                return int(m.group(1)), "제목"
    src = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
    m = DOG_RE.search(os.path.basename(src))
    if m:
        return int(m.group(1)), "파일명"
    m = DOG_RE.search(img.get("alt") or "")
    if m:
        return int(m.group(1)), "alt"
    return None, None


def main():
    os.makedirs(f"{OUT}/photos", exist_ok=True)
    page = f"{OUT}/page.html"
    if os.path.exists(page) and not REFRESH:
        print("page.html 이 이미 있어 다시 받지 않음 (--refresh 로 다시 받기)")
        with open(page, encoding="utf-8") as f:
            html = f.read()
    else:
        html = get(URL).text
        with open(page, "w", encoding="utf-8") as f:
            f.write(html)
        time.sleep(WAIT)
    soup = BeautifulSoup(html, "html.parser")
    with open(f"{OUT}/page_text.txt", "w", encoding="utf-8") as f:
        f.write(soup.get_text("\n", strip=True))

    imgs = soup.find_all("img")
    print(f"페이지의 이미지 태그: {len(imgs)}개")
    rows, count, skipped, failed, n_kept, n_new = [], {}, [], [], 0, 0
    for img in imgs:
        src = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
        alt = img.get("alt") or ""
        num, where = dog_number(img)
        if not src or num is None:
            if src:
                skipped.append(src)
            continue
        dog_id = f"UF{num:03d}"
        if where != "제목":
            print(f"주의: {dog_id} 번호를 {where}에서 읽음 (행 제목 없음)")
        src = urljoin(URL, src)
        full = re.sub(r"-\d+x\d+(\.\w+)$", r"\1", src)      # 썸네일 주소면 원본 크기 주소 먼저 시도
        n = count.get(dog_id, 0)                                # 같은 개의 두 번째 사진이면 _2
        count[dog_id] = n + 1
        stem = f"{dog_id}{'' if n == 0 else f'_{n + 1}'}"
        ext = os.path.splitext(full.split("?")[0])[1].lower() or ".jpg"
        name = stem + ext
        if os.path.exists(f"{OUT}/photos/{name}"):              # 이미 받은 사진은 건너뜀
            rows.append([dog_id, name, full, alt])
            n_kept += 1
            continue
        saved = None
        for u in dict.fromkeys([full, src]):
            try:
                r = get(u)
            except requests.RequestException:
                time.sleep(WAIT)
                continue
            if r.headers.get("content-type", "").startswith("image"):
                tmp = f"{OUT}/photos/{name}.tmp"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, f"{OUT}/photos/{name}")
                saved = (name, u)
                break
            time.sleep(WAIT)
        if saved:
            rows.append([dog_id, saved[0], saved[1], alt])
            n_new += 1
            print("저장:", saved[0])
        else:
            failed.append((dog_id, src))
            print("실패:", dog_id, src)
        time.sleep(WAIT)

    tmp = f"{OUT}/photos.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dog_id", "file", "url", "alt"])
        w.writerows(rows)
    os.replace(tmp, f"{OUT}/photos.csv")

    dogs = sorted({r[0] for r in rows})
    print(f"\n사진 {len(rows)}장 (새로 받음 {n_new}, 이미 있음 {n_kept}), 개 {len(dogs)}마리 → {OUT}")
    print(f"실패 {len(failed)}건:", failed)
    extra = sorted(set(os.listdir(f"{OUT}/photos")) - {r[1] for r in rows})
    if extra:
        print("photos.csv 에 없는 파일(예전 실행이 남긴 것일 수 있음. 지우지 않고 그대로 둠):", extra)
    if not rows:
        print("개 사진을 찾지 못함. 이미지 주소 예시(구조 확인용):")
        for s in skipped[:15]:
            print("  ", s)


if __name__ == "__main__":
    main()
