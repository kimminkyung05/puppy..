"""UF DNA 품종 표기 → Tsinghua 학습 클래스 매핑 초안 (breed_map_UF_draft.csv).

정식 breed_map.csv 는 A 담당이다. 이 파일은 A 에게 넘길 "UF 초안"이고, 사람이 고쳐 쓰는 표다.

출력 열: breed_raw, tsinghua_classes(; 구분), status(정확/유사/없음), note
  - 정확: 같은 품종. 이름만 다르거나(Boston Terrier = Boston_bull) 표기 변형인 경우 note 에 적는다.
  - 유사: 확신이 없는 대응. tsinghua_classes 는 후보이고 사용자가 정한다.
  - 없음: 학습 클래스에 없다. 외형이 가까운 후보가 있으면 note 에 "후보:" 로만 적는다(매핑에는 쓰지 않는다).
  - 클래스 이름은 실제 학습 클래스 목록(분할 파일 + 제외 규칙)에 있는 것만 쓴다. 없으면 실행이 멈춘다.
  - Poodle 계열은 teddy; toy_poodle; miniature_poodle; standard_poodle 로 묶는다 (teddy 는 확인 필요).

사용 (프로젝트 루트에서, PowerShell):
  $env:DOGMIX_PATHS = "data\\paths_local.yaml"
  python scripts/draft_uf_breed_map.py --out data/evalset_uf
  # 이미 있는 초안은 덮어쓰지 않는다(손으로 고친 내용 보호). 다시 만들려면 --force
"""
import argparse
import collections
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import tsinghua  # noqa: E402
from src.common.config import load_yaml  # noqa: E402

POODLE = ["teddy", "toy_poodle", "miniature_poodle", "standard_poodle"]
POODLE_NOTE = "규칙: Poodle 계열 묶음. teddy(테디컷 푸들로 보임)는 확인 필요"
E, S, N = "정확", "유사", "없음"

# breed_raw(페이지 표기 그대로): (클래스 목록, status, note)
MAP = {
    "Afghan Hound": (["Afghan_hound"], E, ""),
    "Airedale Terrier": (["Airedale"], E, ""),
    "Akita": ([], N, "Shiba_Dog 은 다른 품종"),
    "Alaskan Malamute": (["malamute"], E, ""),
    "American Bulldog": ([], N, ""),
    "American Eskimo": (["Japanese_Spitzes"], S,
                        "Eskimo_dog 은 이름만 비슷하고 허스키형 썰매개(학습 이미지로 확인). "
                        "외형은 흰 스피츠인 Japanese_Spitzes 에 가까우나 다른 품종"),
    "American Foxhound": (["English_foxhound", "Walker_hound"], S, "같은 폭스하운드 계열이나 별도 품종"),
    "American Staffordshire": (["American_Staffordshire_terrier"], E, "표기 변형: 'Terrier' 생략으로 봄"),
    "American Staffordshire Terrier": (["American_Staffordshire_terrier"], E, ""),
    "Anatolian Shepherd": ([], N, ""),
    "Australian Cattle Dog": ([], N, "kelpie 는 다른 품종"),
    "Australian Shepherd": (["Australian_Shepherd"], E, ""),
    "Australian Terrier": (["Australian_terrier"], E, ""),
    "Basset Hound": (["basset"], E, ""),
    "Beagle": (["beagle"], E, ""),
    "Beauceron": ([], N, ""),
    "Belgian Malinois": (["malinois"], E, ""),
    "Belgian Sheepdog": (["groenendael"], E, "이름만 다름: Belgian Sheepdog = Groenendael"),
    "Belgian Tervuren": ([], N, "후보: groenendael; malinois (같은 벨지안 셰퍼드 계열)"),
    "Bernese Mountain Dog": (["Bernese_mountain_dog"], E, ""),
    "Bichon Frise": (["Bichon_Frise"], E, ""),
    "Black and Tan Coonhound": (["black_and_tan_coonhound"], E, ""),
    "Black Russian Terrier": ([], N, ""),
    "Bloodhound": (["bloodhound"], E, ""),
    "Blue Tick Coonhound": (["bluetick"], E, ""),
    "Border Collie": (["Border_collie"], E, ""),
    "Borzoi": (["borzoi"], E, ""),
    "Boston Terrier": (["Boston_bull"], E, "이름만 다름"),
    "Boxer": (["boxer"], E, ""),
    "Briard": (["briard"], E, ""),
    "Brittany Spaniel": (["Brittany_spaniel"], E, ""),
    "Bull Mastiff": (["bull_mastiff"], E, ""),
    "Bull Terrier": ([], N, "Staffordshire_bullterrier 는 다른 품종"),
    "Bulldog": ([], N, "English Bulldog 로 보임. French_bulldog 는 다른 품종"),
    "Bullmastiff": (["bull_mastiff"], E, "표기 변형: 'Bull Mastiff' 와 같음"),
    "Cairn Terrier": (["cairn"], E, ""),
    "Cane Corso": (["Cane_Carso"], E, "학습 클래스 철자가 Cane_Carso"),
    "Catahoula Leopard Dog": ([], N, ""),
    "Chihuahua": (["Chihuahua"], E, ""),
    "Chinese Crested": (["Chinese_Crested_Dog"], E, ""),
    "Chinese Sharpei": ([], N, ""),
    "Chow Chow": (["chow"], E, ""),
    "Cocker Spaniel": (["cocker_spaniel"], E, "학습 클래스는 American/English 를 구분하지 않음"),
    "Collie": (["collie"], E, ""),
    "Curly Coated Retriever": (["curly_coated_retriever"], E, ""),
    "Curly-Coated Retriever": (["curly_coated_retriever"], E, "표기 변형: 하이픈"),
    "Dalmatian": ([], N, ""),
    "Dandie Dinmont Terrier": (["Dandie_Dinmont"], E, ""),
    "Doberman Pinscher": (["Doberman"], E, ""),
    "Dogo Argentino": ([], N, ""),
    "Dogue de Bordeaux": ([], N, ""),
    "English Cocker Spaniel": (["cocker_spaniel"], S, "학습 클래스가 American/English Cocker 를 구분하지 않음"),
    "English Springer Spaniel": (["English_springer"], E, ""),
    "English Toy Terrier": (["toy_terrier"], S, "toy_terrier 에 블랙탄 토이 테리어형이 섞여 있음(학습 이미지로 확인)"),
    "Entlebucher Mountain Dog": (["EntleBucher"], E, ""),
    "Field Spaniel": ([], N, "Sussex_spaniel 은 다른 품종"),
    "Flat Coated Retriever": (["flat_coated_retriever"], E, ""),
    "Fox Terrier": (["wire_haired_fox_terrier"], S, "Smooth/Wire 구분 없이 적힘. 학습 클래스는 Wire 만 있음"),
    "French Bulldog": (["French_bulldog"], E, ""),
    "German Shepherd": (["German_shepherd"], E, "Black_sable(검은 셰퍼드형, 학습 이미지로 확인)도 묶을지 확인 필요"),
    "German Short Haired Pointer": (["German_short_haired_pointer"], E, "표기 변형: 띄어쓰기"),
    "German Shorthaired Pointer": (["German_short_haired_pointer"], E, ""),
    "German Spitz": ([], N, "후보: Pomeranian; keeshond; Japanese_Spitzes (스피츠 계열이나 별도 품종으로 적힘)"),
    "German Wirehaired Pointer": ([], N, "후보: German_short_haired_pointer (털 종류가 다른 별도 품종)"),
    "Glen of Imaal Terrier": ([], N, ""),
    "Golden Retriever": (["golden_retriever"], E, ""),
    "Great Dane": (["Great_Dane"], E, ""),
    "Great Pyrenees": (["Great_Pyrenees"], E, ""),
    "Harrier": ([], N, "후보: beagle; English_foxhound (외형 유사, 별도 품종)"),
    "Havanese": ([], N, ""),
    "Ibizian Hound": (["Ibizan_hound"], E, "표기 변형: 'Ibizan' 의 오타로 봄"),
    "Irish Setter": (["Irish_setter"], E, ""),
    "Irish terrier": (["Irish_terrier"], E, "표기 변형: 소문자"),
    "Irish Terrier": (["Irish_terrier"], E, ""),
    "Irish Water Spaniel": (["Irish_water_spaniel"], E, ""),
    "Irish Wolfhound": (["Irish_wolfhound"], E, ""),
    "Italian Greyhound": (["Italian_greyhound"], E, ""),
    "Jindo": ([], N, ""),
    "Keeshond": (["keeshond"], E, ""),
    "Komondor": (["komondor"], E, ""),
    "Labrador Retriever": (["Labrador_retriever"], E, ""),
    "Lancashire Heeler": ([], N, ""),
    "Lhasa Apso": (["Lhasa"], E, ""),
    "Long Haired Dachshund": ([], N, "Dachshund 클래스 없음"),
    "Lowchen": ([], N, ""),
    "Manchester Terrier": ([], N, "후보: toy_terrier (English Toy Terrier 와 같은 계통, 크기 다름)"),
    "Mastiff": ([], N, "bull_mastiff, Tibetan_mastiff 는 다른 품종"),
    "Miniature Bull Terrier": ([], N, ""),
    "Miniature Pinscher": (["miniature_pinscher"], E, ""),
    "Miniature Poodle": (POODLE, E, POODLE_NOTE),
    "Miniature Schnauzer": (["miniature_schnauzer"], E, ""),
    "Miniature Short Haired Dachshund": ([], N, "Dachshund 클래스 없음"),
    "Neopolitan Mastiff": ([], N, "표기 변형: 'Neapolitan' 의 오타로 봄"),
    "Newfoundland": (["Newfoundland"], E, ""),
    "Norfolk Terrier": (["Norfolk_terrier"], E, ""),
    "Norwegian Buhund": ([], N, ""),
    "Norwegian Elkhound": (["Norwegian_elkhound"], E, ""),
    "Nova Scotia Duck Tolling Retriever": ([], N, ""),
    "Novia Scotia Duck Tolling Retriever": ([], N, "표기 변형: 'Nova' 의 오타로 봄"),
    "Old English Sheepdog": (["Old_English_sheepdog"], E, ""),
    "Pembroke Welsh Corgi": (["Pembroke"], E, ""),
    "Plott Hound": ([], N, ""),
    "Pointer": ([], N, "English Pointer. German_short_haired_pointer 는 다른 품종"),
    "Pomeranian": (["Pomeranian"], E, ""),
    "Poodle": (POODLE, E, POODLE_NOTE + ". 크기 구분 없이 적힘"),
    "Portuguese Water Dog": ([], N, ""),
    "Puli": ([], N, "komondor 는 다른 품종"),
    "RedBone Coonhound": (["redbone"], E, ""),
    "Rhodesian Ridgeback": (["Rhodesian_ridgeback"], E, ""),
    "Rottweiler": (["Rottweiler"], E, ""),
    "Russell Terrier": ([], N, "Jack Russell 계열 클래스 없음"),
    "Saint Bernard": (["Saint_Bernard"], E, ""),
    "Saluki": (["Saluki"], E, ""),
    "Samoyed": (["Samoyed"], E, ""),
    "Schipperke": (["schipperke"], E, ""),
    "Scottish Deerhound": (["Scottish_deerhound"], E, ""),
    "Scottish Terrier": (["Scotch_terrier"], E, "이름만 다름"),
    "Sealyham Terrier": (["Sealyham_terrier"], E, ""),
    "Shetland Sheepdog": (["Shetland_sheepdog"], E, ""),
    "Shiba Inu": (["Shiba_Dog"], E, ""),
    "Short Haired Dachshund": ([], N, "Dachshund 클래스 없음"),
    "Siberian Husky": (["Siberian_husky"], E, "Eskimo_dog(허스키형 썰매개)과 묶을지 확인 필요"),
    "Silky Terrier": (["silky_terrier"], E, ""),
    "Small Munsterlander": ([], N, ""),
    "Soft Coated Wheaton Terrier": (["soft_coated_wheaten_terrier"], E, "표기 변형: 'Wheaten' 의 오타로 봄"),
    "Staffordshire Bull Terrier": (["Staffordshire_bullterrier"], E, ""),
    "Standard Schnauzer": (["standard_schnauzer"], E, ""),
    "Swedish Vallhund": ([], N, ""),
    "Tibetan Mastiff": (["Tibetan_mastiff"], E, ""),
    "Tibetan Spaniel": ([], N, "Tibetan_terrier, Japanese_spaniel 은 다른 품종"),
    "Toy Fox Terrier": (["toy_terrier"], S, "toy_terrier 에 토이 폭스 테리어형이 섞여 있음(학습 이미지로 확인)"),
    "Vizsla": (["vizsla"], E, ""),
    "Weimeraner": (["Weimaraner"], E, "표기 변형: 'Weimaraner' 의 오타로 봄"),
    "Welsh Terrier": ([], N, "후보: Lakeland_terrier; Airedale (외형 유사, 별도 품종)"),
    "Wheaten Terrier": (["soft_coated_wheaten_terrier"], E, "표기 변형: Soft Coated Wheaten Terrier 로 봄"),
    "Wire Fox Terrier": (["wire_haired_fox_terrier"], E, ""),
    "Wire Haired Dachshund": ([], N, "Dachshund 클래스 없음"),
    "Yorkshire Terrier": (["Yorkshire_terrier"], E, ""),
}


def train_classes() -> list[str]:
    paths = load_yaml("paths")
    items = tsinghua.parse_split(paths["tsinghua"]["train_list"])
    items = tsinghua.filter_classes(items, paths["classes"]["exclude"], paths["classes"]["exclude_dingo"])
    return tsinghua.class_list(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/evalset_uf")
    ap.add_argument("--force", action="store_true", help="이미 있는 초안을 덮어쓴다")
    args = ap.parse_args()
    out = Path(args.out)
    target = out / "breed_map_UF_draft.csv"

    classes = train_classes()
    with open(out / "dna_UF.csv", encoding="utf-8-sig") as f:
        dna = list(csv.DictReader(f))
    raw = collections.Counter(r["breed_raw"] for r in dna)
    pct = collections.Counter()
    for r in dna:
        pct[r["breed_raw"]] += float(r["pct"] or 0)

    unknown = sorted({c for cs, _, _ in MAP.values() for c in cs} - set(classes))
    missing = sorted(set(raw) - set(MAP))
    bad = [b for b, (cs, st, _) in MAP.items() if (st == N) != (len(cs) == 0)]
    if unknown or missing or bad:
        raise SystemExit(f"매핑 표 오류. 학습 클래스에 없는 이름: {unknown} | 매핑이 없는 품종 표기: {missing} | "
                         f"status 와 클래스 칸이 맞지 않음: {bad}")
    unused = sorted(set(MAP) - set(raw))

    rows = [[b, "; ".join(MAP[b][0]), MAP[b][1], MAP[b][2]] for b in sorted(raw, key=str.lower)]
    if target.exists() and not args.force:
        print(f"[건너뜀] 이미 있음 → {target}  (손으로 고친 내용을 지키려고 덮어쓰지 않음. 다시 만들려면 --force)")
    else:
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["breed_raw", "tsinghua_classes", "status", "note"])
            w.writerows(rows)
        os.replace(tmp, target)
        print(f"저장 → {target}")

    print(f"\n학습 클래스 {len(classes)}개 기준. DNA 품종 표기 {len(raw)}종, DNA 줄 {len(dna)}개")
    print(f"  {'status':<6}{'표기 수':>8}{'DNA 줄':>8}{'비율 합(%p)':>12}")
    for st in (E, S, N):
        bs = [b for b in raw if MAP[b][1] == st]
        print(f"  {st:<6}{len(bs):>9}{sum(raw[b] for b in bs):>9}{sum(pct[b] for b in bs):>13.1f}")
    used = sorted({c for b in raw for c in MAP[b][0]})
    print(f"  매핑에 쓰인 학습 클래스 {len(used)}개 / {len(classes)}개")
    print("\n가장 자주 나오는 '없음' 품종 10개 (DNA 줄 수, 비율 합):")
    none = sorted((b for b in raw if MAP[b][1] == N), key=lambda b: (-raw[b], -pct[b]))
    for b in none[:10]:
        print(f"  {raw[b]:>3}줄 {pct[b]:>7.1f}%p  {b}")
    print("\nstatus=유사 (사용자가 정할 것):")
    for b in sorted(raw, key=str.lower):
        if MAP[b][1] == S:
            print(f"  {raw[b]:>3}줄  {b}  →  {'; '.join(MAP[b][0])}  | {MAP[b][2]}")
    if unused:
        print("\n(참고) 표에는 있는데 이번 DNA 표에 없는 표기:", unused)


if __name__ == "__main__":
    main()
