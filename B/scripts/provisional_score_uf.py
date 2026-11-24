"""UF 초안 임시 채점. 정식 채점이 아니다 (A 의 src/eval/eval.py 가 나오면 그것으로 바꾼다).

지표 정의는 docs/METRICS_FINAL.md (A 의 최종 정의 복사본) 를 따른다.
  - 매핑: status=정확 만. 유사 매핑은 --with-similar 를 줄 때만 참고값으로 본다 (Top-1 · 포착 · 겹침 모두)
  - 주 지표: 원래 DNA 비율 포착 혈통 비율@3 (+ coverage)
  - 보조: 재정규화 포착@3, 재정규화 분포 겹침, Top-1 + coverage. 원래 비율 분포 겹침은 참고값

목적: eval.py 가 나오기 전에 dev 결과를 숫자로 보고, L2 정규화 여부 같은 선택을 임시로 정한다.

입력
  preds/dev/pred_*.csv          Tsinghua 클래스 공간 예측 (dog_id, split, <클래스들>)
  truth_UF.csv                  build_uf_truth.py 가 만든 정답 비율 (믹스견만)
  breed_map_UF_draft.csv        품종 표기 → 학습 클래스 매핑 초안
  dna_UF.csv                    Top-1 정답용 (uf-dir, 없으면 그 윗폴더. 둘 다 없으면 Top-1 표만 건너뛴다)
  features/{backbone}/train/index.csv   학습셋 label (찍기 기준용. 없으면 찍기 줄을 비운다)

계산
  1. 예측을 DNA 품종 공간으로 합친다: 매핑 표에서 한 품종이 여러 클래스이면(예: Poodle 묶음) 그 클래스들의 예측을 더한다.
     매핑 표에 나오지 않는 클래스는 각각 그대로 한 품종으로 둔다 (상위 3품종 후보에서 빼지 않는다).
  2. 주 지표 = 예측 상위 3품종이 차지하는 실제 DNA 비율의 합 (개마다 계산해 평균).
       (a) 원래 비율: pct / 100. 학습 클래스에 없는 품종과 페이지에 안 적힌 나머지는 맞힐 수 없으므로 상한이 1 보다 작다.
       (b) 재정규화 비율: 매핑된 품종 안에서 합 1 로 다시 나눈 비율. 매핑된 품종이 없는 개는 뺀다.
  3. 찍기: 학습셋 클래스 빈도를 모든 개에 똑같이 낸 예측을 같은 방식으로 채점한다.
  4. 상한: 실제 비율이 큰 3품종을 그대로 맞혔을 때의 값.
  4-1. 보조 지표 분포 겹침 = 품종 공간에서 Σ min(예측 비율, 실제 비율). (a) · (b) 둘 다 낸다.
       예측은 다시 나누지 않는다(매핑 밖 클래스에 준 몫은 놓친 몫). 상한은 실제 비율을 그대로 낸 값 = (a) DNA coverage, (b) 1.
       찍기는 위와 같은 학습셋 클래스 빈도 예측이다.
  5. 대표값은 반복(r1~r3)의 확률을 평균한 뒤 채점한 값이다. 반복별 점수의 평균 · 최소 · 최대도 낸다. 비교(찍기 대비, L2 유무)는 개 단위 짝 차이의 평균 ± 2 표준오차로 보고,
     구간이 0 을 포함하면 "판단 불가"라고 적는다. 개가 20마리뿐이라 대부분 방향만 볼 수 있다.
     비교에는 대표값을 쓴다.
  6. 믹스견 Top-1 (METRICS_FINAL.md 3.4절, 분류 보조 지표). 맨 위에 보인다
       - 정답: DNA 에 적힌 비율이 가장 큰 품종(동률이면 모두)에 매핑된 클래스의 합집합. 하나를 맞히면 정답.
       - 예측: 클래스를 품종으로 합치지 않고 가장 큰 클래스. 동률이면 공용 클래스 순서에서 앞선 것.
       - 1위 품종이 전부 범위 밖이면 분모에서 빼고 Coverage(평가 가능 / 전체)를 낸다.
         아래 포착 혈통 비율 표의 "coverage 0.5 이상"(개마다의 DNA 비율 합)과 다른 값이다.
       - 매핑은 정확만 (METRICS_FINAL.md 2절). 포착 혈통 비율 · 분포 겹침도 같다.
       - --reference: 기준 문서에 없는 참고값(정확 + 유사 매핑 Top-1, 한 품종 고정 기준선)도 보인다.
     포착 혈통 비율 · 분포 겹침(1~5)은 Top-1 뒤에 보인다. 표 제목에 주 지표 / 보조 지표 / 참고값을 적는다.
       - 개별 개의 DNA 가 노트북 출력에 남지 않게 마릿수만 보인다.
  split 이 dev 가 아닌 행은 파일에 있어도 채점하지 않는다.

사용 (프로젝트 루트에서. DOGMIX_PATHS 로 경로 설정을 고른다)
  python scripts/provisional_score_uf.py --uf-dir data/evalset_uf
  python scripts/provisional_score_uf.py --uf-dir {DRIVE_DIR}/evalset/uf_draft --eval-subset eval_uf_panels
  옵션: --summary-csv (집계 CSV 저장), --with-similar (유사 매핑도 정답 품종으로 보는 참고 채점), --dna, --reference, --include-check (CHECK- 점검 실행도 채점)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_yaml  # noqa: E402

BANNER = "임시 채점: 매핑 초안 기준, 정식 채점 아님(A의 eval.py로 대체 예정)"
EXACT, SIMILAR = "정확", "유사"
NAME_RE = re.compile(r"^pred_(?P<kind>RQ1|CHECK)-(?:(?P<ev>.+)-)?(?P<bb>clip|dinov2|dinov3|sift|dummy)"
                     r"-proto-r(?P<r>\d+)(?P<l2>-l2)?(?P<pool>-mean)?(?P<tr>-tr_.+)?\.csv$")
TOPK = 3
REPEATS = [1, 2, 3]
SUMMARY: list[dict] = []   # --summary-csv 로 저장할 집계 행 (개별 개 값은 넣지 않는다)


def split_classes(s) -> list[str]:
    return [c.strip() for c in str(s).split(";") if c.strip()] if isinstance(s, str) else []


def breed_of_class(map_csv: Path, allowed: set[str]) -> dict[str, str]:
    """클래스 → 합쳐질 품종 이름(매핑 표의 tsinghua_classes 문자열). 한 클래스가 두 품종에 걸치면 멈춘다."""
    m = pd.read_csv(map_csv, encoding="utf-8-sig")
    out = {}
    for _, r in m[m["status"].isin(allowed)].iterrows():
        names = split_classes(r["tsinghua_classes"])
        key = "; ".join(names)
        for c in names:
            if out.get(c, key) != key:
                raise SystemExit(f"매핑 표에서 클래스 '{c}' 가 두 품종 묶음에 걸침: '{out[c]}' / '{key}'")
            out[c] = key
    return out


def load_truth(truth_csv: Path, with_similar: bool):
    """(개 → {품종: 원래 비율}, 개 → {품종: 재정규화 비율}, 개 → coverage)"""
    t = pd.read_csv(truth_csv, encoding="utf-8-sig")
    cov_col = "coverage_with_similar" if with_similar else "coverage"
    norm_col = "share_norm_with_similar" if with_similar else "share_norm"
    allowed = {EXACT, SIMILAR} if with_similar else {EXACT}
    orig, norm, cov = {}, {}, {}
    for dog, g in t.groupby("dog_id"):
        cov[dog] = float(g[cov_col].iloc[0])
        g = g[g["status"].isin(allowed)]
        # DNA 품종 둘이 같은 클래스 묶음으로 가면 비율을 더한다 (덮어쓰지 않는다)
        orig[dog] = (g.groupby("target_classes")["pct"].sum().astype(float) / 100).to_dict()
        norm[dog] = g.groupby("target_classes")[norm_col].sum().astype(float).to_dict()
    return orig, norm, cov


def merge_to_breeds(pred: pd.DataFrame, c2b: dict[str, str]) -> pd.DataFrame:
    """클래스 열을 품종 열로 합친다 (같은 품종으로 가는 클래스는 더한다)."""
    return pred.T.groupby(lambda c: c2b.get(c, c)).sum().T


def captured(merged: pd.DataFrame, truth: dict) -> pd.Series:
    """개마다: 예측 상위 3품종의 실제 비율 합."""
    vals = {}
    for dog, row in merged.iterrows():
        if dog not in truth:
            continue
        top = row.sort_values(ascending=False, kind="stable").index[:TOPK]
        vals[dog] = sum(truth[dog].get(b, 0.0) for b in top)
    return pd.Series(vals, dtype=float)


def overlap(merged: pd.DataFrame, truth: dict) -> pd.Series:
    """개마다: 품종 공간에서 Σ min(예측 비율, 실제 비율). 예측은 다시 나누지 않는다 (매핑 밖 클래스에 준 몫은 놓친 몫)."""
    vals = {}
    for dog, row in merged.iterrows():
        if dog not in truth:
            continue
        vals[dog] = sum(min(float(row.get(b, 0.0)), y) for b, y in truth[dog].items())
    return pd.Series(vals, dtype=float)


def ceiling(truth: dict, dogs) -> pd.Series:
    return pd.Series({d: sum(sorted(truth[d].values(), reverse=True)[:TOPK]) for d in dogs}, dtype=float)


def overlap_ceiling(truth: dict, dogs) -> pd.Series:
    """실제 비율을 그대로 냈을 때의 겹침 = 매핑된 실제 비율의 합. (a)는 DNA coverage, (b)는 1."""
    return pd.Series({d: sum(truth[d].values()) for d in dogs}, dtype=float)


# (키, 표 제목, 지표 설명, 개별 점수 함수, 상한 함수)
METRICS = [
    ("captured", f"포착 혈통 비율@{TOPK}", f"예측 상위 {TOPK}품종의 실제 비율 합", captured, ceiling),
    ("overlap", "분포 겹침", "Σ min(예측 비율, 실제 비율) (품종 공간)", overlap, overlap_ceiling),
]


def mean_prob(frames: list[pd.DataFrame], cfg: str) -> pd.DataFrame:
    """반복들의 확률 평균. 개나 클래스가 반복마다 다르면 멈춘다."""
    first = frames[0]
    for f in frames[1:]:
        if set(f.index) != set(first.index) or list(f.columns) != list(first.columns):
            raise SystemExit(f"{cfg}: 반복마다 개 목록이나 클래스 순서가 달라 확률을 평균할 수 없음")
    return sum(f.loc[first.index] for f in frames) / len(frames)


def top1_answers(dna: pd.DataFrame, bmap: pd.DataFrame, allowed: set[str], classes, dogs) -> dict[str, set[str]]:
    """개 → Top-1 정답 클래스 집합. 빈 집합이면 채점 대상이 아니다."""
    m = {r["breed_raw"]: (r["status"], split_classes(r["tsinghua_classes"])) for _, r in bmap.iterrows()}
    out = {}
    for d in dogs:
        g = dna[(dna["dog_id"] == d) & dna["pct"].notna()]
        if g.empty:
            raise SystemExit(f"DNA 표에 {d} 의 비율이 없음")
        pct = g["pct"].astype(float)
        ans = set()
        for breed in g.loc[pct == pct.max(), "breed_raw"]:
            if breed not in m:
                raise SystemExit(f"매핑 표에 없는 품종 표기: {breed}")
            status, names = m[breed]
            if status in allowed:
                ans.update(c for c in names if c in classes)
        out[d] = ans
    return out


def top1_score(prob: pd.DataFrame, answers: dict[str, set[str]]) -> tuple[int, int]:
    """(정답 수, 채점한 개 수). 열 순서가 공용 클래스 순서라 동률이면 앞선 클래스가 뽑힌다."""
    top = prob.idxmax(axis=1)
    use = [d for d in prob.index if answers[d]]
    return sum(top[d] in answers[d] for d in use), len(use)


def print_top1(probs: dict, ens: dict, dna: pd.DataFrame, bmap: pd.DataFrame, freq, reference: bool,
               allowed: set[str]) -> None:
    def pct(hit, n):
        return f"{100 * hit / n:.1f}%" if n else "-"

    tables = [("믹스견 Top-1 (분류 보조 지표)", allowed)]
    if reference:
        other = {EXACT} if SIMILAR in allowed else {EXACT, SIMILAR}
        tables.append((f"참고(기준 문서 밖): {' + '.join(s for s in (EXACT, SIMILAR) if s in other)} 매핑", other))
    for title, allowed in tables:
        print(f"\n[{title}]  정답: DNA 1위 품종(동률이면 그중 하나) | 매핑 {' + '.join(s for s in (EXACT, SIMILAR) if s in allowed)}")
        print(f"  {'설정':<16}{'Top-1(확률 평균)':>18}{'반복별 최소~최대':>20}{'Coverage':>16}{'정답/평가 가능/전체':>20}")
        baselines = {}
        for cfg, prob in ens.items():
            classes = list(prob.columns)
            answers = top1_answers(dna, bmap, allowed, set(classes), prob.index)
            hit, n = top1_score(prob, answers)
            reps = sorted(r for c, r in probs if c == cfg)
            each = [top1_score(probs[(cfg, r)], answers)[0] for r in reps]
            note = "" if reps == REPEATS else f"  ← 반복 {reps}뿐, 대표값으로 쓰지 않음"
            SUMMARY.append(dict(metric="top1", mapping="+".join(s for s in (EXACT, SIMILAR) if s in allowed),
                                role="분류 보조 지표" if title.startswith("믹스견") else "참고값", subset="dev",
                                truth="dominant", config=cfg, n=n, value=hit / n if n else np.nan, hits=hit,
                                coverage=n / len(prob), rep_min=min(each) / n if n else np.nan,
                                rep_max=max(each) / n if n else np.nan))
            print(f"  {cfg:<16}{f'{pct(hit, n)} ({hit}/{n})':>22}{pct(min(each), n) + '~' + pct(max(each), n):>26}"
                  f"{f'{pct(n, len(prob))} ({n}/{len(prob)})':>20}{f'{hit}/{n}/{len(prob)}':>16}{note}")
            if not reference or not n:
                continue
            best_hit, best = max(((sum(c in a for a in answers.values()), c) for c in classes), key=lambda x: x[0])
            baselines[f"dev 최빈 고정({best}): {pct(best_hit, n)} ({best_hit}/{n})"] = None
            if freq is not None:
                common = freq.reindex(classes).fillna(0).idxmax()
                k = sum(common in a for a in answers.values())
                baselines[f"학습셋 최빈 고정({common}): {pct(k, n)} ({k}/{n})"] = None
        for line in baselines:
            print(f"  참고 기준선 {line}")


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float, str]:
    """개 단위 짝 차이 a - b 의 (평균, 2 표준오차, 판단)."""
    d = (a - b).dropna()
    if len(d) < 3:
        return float("nan"), float("nan"), "판단 불가(개 수 부족)"
    mean, se2 = float(d.mean()), 2 * float(d.std(ddof=1)) / np.sqrt(len(d))
    if abs(mean) <= se2:
        return mean, se2, "판단 불가"
    return mean, se2, "앞쪽이 높음" if mean > 0 else "뒤쪽이 높음"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uf-dir", default="data/evalset_uf", help="truth_UF.csv 와 breed_map_UF_draft.csv 가 있는 폴더")
    ap.add_argument("--preds-root", default=None, help="기본: 경로 설정의 outputs.preds_root")
    ap.add_argument("--features-root", default=None, help="기본: 경로 설정의 outputs.features_root (찍기 기준용)")
    ap.add_argument("--split", default="dev", choices=["dev"], help="임시 채점은 dev 만 본다")
    ap.add_argument("--eval-subset", default="eval_uf_panels", help="이 평가 subset 의 실행만 채점한다")
    ap.add_argument("--with-similar", action="store_true",
                    help="status=유사 매핑도 정답 품종으로 본다 (참고 채점. 기본은 METRICS_FINAL.md 대로 정확만)")
    ap.add_argument("--exact-only", action="store_true", help=argparse.SUPPRESS)     # 예전 명령 호환. 지금은 기본값
    ap.add_argument("--include-check", action="store_true", help="CHECK- (형식 점검) 실행도 채점한다")
    ap.add_argument("--dna", default=None, help="Top-1 정답용 dna_UF.csv. 기본: uf-dir, 없으면 그 윗폴더")
    ap.add_argument("--reference", action="store_true", help="기준 문서 밖 참고값(정확 매핑만, 고정 기준선)도 보인다")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--summary-csv", default=None, help="집계 행(개별 개 값 없음)을 이 CSV 로 저장한다")
    args = ap.parse_args()

    paths = load_yaml("paths")
    preds_dir = Path(args.preds_root or paths["outputs"]["preds_root"]) / args.split
    feat_root = Path(args.features_root or paths["outputs"]["features_root"])
    uf = Path(args.uf_dir)
    if args.exact_only and args.with_similar:
        ap.error("--exact-only 와 --with-similar 는 같이 쓸 수 없음")
    with_similar = args.with_similar
    allowed = {EXACT, SIMILAR} if with_similar else {EXACT}

    print("=" * 78)
    print(BANNER)
    print("=" * 78)
    print(f"예측: {preds_dir} | 평가 subset: {args.eval_subset} | 정답: {uf / 'truth_UF.csv'}")
    print(f"정답 품종으로 보는 매핑 (Top-1 · 포착 · 겹침): {' + '.join(s for s in (EXACT, SIMILAR) if s in allowed)}"
          + ("  ← 참고 채점 (METRICS_FINAL.md 는 정확만)" if with_similar else "  (METRICS_FINAL.md 2절)"))

    c2b = breed_of_class(uf / "breed_map_UF_draft.csv", allowed)
    orig, norm, cov = load_truth(uf / "truth_UF.csv", with_similar)

    runs = []           # (보이는 이름, 설정 이름, 반복/시드, 예측 표)
    for p in sorted(preds_dir.glob("pred_*.csv")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        ev = m["ev"] or "eval"
        if ev != args.eval_subset and not ev.startswith(f"{args.eval_subset}_limit"):   # _limitN 은 로컬 점검용
            continue
        if m["kind"] == "CHECK" and not args.include_check:
            continue
        cfg = f"{m['bb']}{'+L2' if m['l2'] else ''}{'+mean' if m['pool'] else ''}{' [점검]' if m['kind'] == 'CHECK' else ''}"
        runs.append((p.name, cfg, int(m["r"]), pd.read_csv(p)))
    if not runs:
        print(f"\n채점할 예측 파일이 없음. {preds_dir} 에 pred_RQ1-{args.eval_subset}-*.csv 가 있는지 확인"
              " (점검 실행은 --include-check)")
        return

    # 찍기 기준: 학습셋 클래스 빈도
    freq = None
    for idx in sorted(feat_root.glob("*/train/index.csv")):
        freq = pd.read_csv(idx)["label"].value_counts(normalize=True)
        print(f"찍기 기준: {idx} 의 클래스 빈도 ({len(freq)}개 클래스)")
        break
    if freq is None:
        print(f"찍기 기준: 계산 불가 ({feat_root}/*/train/index.csv 없음)")

    per_dog = {}        # (설정 이름, 반복) → 품종 공간 예측 (믹스견 × 품종)
    probs = {}          # (설정 이름, 반복) → 믹스견 × 클래스 확률
    info, chance = [], {}
    for label, cfg, rep, pred in runs:
        classes = [c for c in pred.columns if c not in ("dog_id", "split")]
        row_sum = pred[classes].sum(axis=1)
        notes = []
        other = pred["split"] != args.split
        if other.any():     # hold 나 test 가 섞인 파일이어도 점수를 내지 않는다
            notes.append(f"split 이 {args.split} 가 아닌 {int(other.sum())}행 제외")
            pred, row_sum = pred[~other], row_sum[~other]
        if pred.empty:
            info.append((label, 0, len(classes), "; ".join(notes)))
            continue
        if len(classes) != 127:
            notes.append(f"클래스 {len(classes)}개(127 아님)")
        if not np.allclose(row_sum, 1, atol=1e-6):
            notes.append("행 합이 1 이 아님")
        absent = sorted(set(c2b) - set(classes))
        if absent:
            notes.append(f"매핑에 쓰인 클래스 {len(absent)}개가 예측에 없음(0 으로 봄)")
        dogs_all = pred["dog_id"].tolist()
        not_mixed = [d for d in dogs_all if d not in orig]
        if not_mixed:
            notes.append(f"정답에 없는 개 {len(not_mixed)}마리 제외")
        merged = merge_to_breeds(pred.set_index("dog_id")[classes], c2b)
        n_few = int(((merged[merged.index.isin(list(orig))] > 0).sum(axis=1) < TOPK).sum())
        if n_few:   # FCLS 는 대부분의 품종에 0 을 준다. 채점 규칙은 그대로 두고 몇 마리인지만 보인다
            notes.append(f"예측 비율이 0보다 큰 품종이 {TOPK}개 미만인 개 {n_few}마리(상위 {TOPK}의 남는 자리는 이름순 품종이 채움)")
        if (cfg, rep) in probs:
            raise SystemExit(f"{cfg} 반복 {rep} 가 두 번 들어옴 ({label})")
        by_dog = pred.set_index("dog_id")[classes]
        probs[(cfg, rep)] = by_dog[by_dog.index.isin(list(orig))]
        per_dog[(cfg, rep)] = merged[merged.index.isin(list(orig))]
        info.append((label, len(dogs_all), len(classes), "; ".join(notes)))
        if freq is not None and cfg not in chance:
            f = pd.DataFrame([freq.reindex(classes).fillna(0).to_numpy()] * len(pred), columns=classes,
                             index=pred["dog_id"])
            chance[cfg] = merge_to_breeds(f, c2b)

    print(f"\n예측 파일 {len(info)}개")
    for name, n_dogs, n_cls, note in info:
        print(f"  {name}: 개 {n_dogs}, 클래스 {n_cls}" + (f"  ← {note}" if note else ""))

    if not per_dog:
        print(f"\n{args.split} 행이 있는 예측 파일이 없음")
        return
    cfgs = sorted({k[0] for k in per_dog})
    ens = {c: mean_prob([probs[k] for k in sorted(probs) if k[0] == c], c) for c in cfgs}

    dna_files = [Path(args.dna)] if args.dna else [uf / "dna_UF.csv", uf.parent / "dna_UF.csv"]
    dna_csv = next((f for f in dna_files if f.exists()), None)
    if args.dna and dna_csv is None:
        raise SystemExit(f"DNA 표가 없음: {args.dna}")
    if dna_csv:
        print_top1(probs, ens, pd.read_csv(dna_csv, encoding="utf-8-sig"),
                   pd.read_csv(uf / "breed_map_UF_draft.csv", encoding="utf-8-sig"), freq, args.reference, allowed)
    else:
        print(f"\n[믹스견 Top-1] 건너뜀: dna_UF.csv 가 없음 ({', '.join(map(str, dna_files))})")
    dogs = sorted(set().union(*[set(v.index) for v in per_dog.values()]))
    subsets = [(f"dev 전체 {len(dogs)}마리", dogs),
               (f"coverage {args.min_coverage} 이상", [d for d in dogs if cov.get(d, 0) >= args.min_coverage])]
    kinds = [("orig", "(a) 원래 비율"), ("norm", "(b) 학습 품종 안에서 재정규화한 비율")]
    # METRICS_FINAL.md 3절의 역할
    role = {("captured", "orig"): "주 지표", ("captured", "norm"): "보조 지표",
            ("overlap", "norm"): "보조 지표", ("overlap", "orig"): "참고값"}
    ens_merged = {c: merge_to_breeds(ens[c], c2b) for c in cfgs}

    for metric, metric_name, metric_desc, score_fn, ceil_fn in METRICS:
        print(f"\n---- 아래는 {metric_name} (METRICS_FINAL.md 3절) ----")
        for sub_name, sub in subsets:
            for kind, kind_name in kinds:
                truth = orig if kind == "orig" else norm
                use = [d for d in sub if truth[d]] if kind == "norm" else list(sub)
                n_out = len(sub) - len(use)
                print(f"\n[{role[(metric, kind)]} · {metric_name} | {sub_name} → {len(use)}마리"
                      f"{f' (매핑된 혈통 없음 {n_out}마리 제외)' if n_out else ''} | 실제 비율 {kind_name}]  지표: {metric_desc}")
                if len(use) == 0:
                    print("  해당하는 개 없음")
                    continue
                ceil = float(ceil_fn(truth, use).mean())
                ceil_note = f"실제 상위 {TOPK}품종을 다 맞힘" if metric == "captured" else "실제 비율을 그대로 냄"
                print(f"  상한({ceil_note}): {ceil:.3f}")
                print(f"  {'설정':<16}{'반복':>5}{'대표':>8}{'평균':>8}{'최소':>8}{'최대':>8}{'찍기':>8}{'찍기와 차이 ± 2SE':>22}  판단")
                avg = {}
                for cfg in cfgs:
                    reps = sorted(r for c, r in per_dog if c == cfg)
                    run_means = [float(score_fn(per_dog[(cfg, r)], truth).reindex(use).mean()) for r in reps]
                    avg[cfg] = score_fn(ens_merged[cfg], truth).reindex(use)
                    ch_mean = d = se2 = np.nan
                    if cfg in chance:
                        ch = score_fn(chance[cfg], truth).reindex(use)
                        ch_mean = float(ch.mean())
                        d, se2, verdict = paired(avg[cfg], ch)
                        verdict = {"앞쪽이 높음": "찍기보다 높음", "뒤쪽이 높음": "찍기보다 낮음"}.get(verdict, verdict)
                        tail = f"{ch.mean():>8.3f}{d:>+13.3f} ± {se2:.3f}  {verdict}"
                    else:
                        verdict = "찍기 기준 없음"
                        tail = f"{'-':>8}{'-':>22}  찍기 기준 없음"
                    SUMMARY.append(dict(metric=metric, mapping="+".join(s for s in (EXACT, SIMILAR) if s in allowed),
                                        role=role[(metric, kind)], subset=sub_name, truth=kind, config=cfg, n=len(use),
                                        value=float(avg[cfg].mean()), rep_mean=float(np.mean(run_means)),
                                        rep_min=min(run_means), rep_max=max(run_means), chance=ch_mean, ceiling=ceil,
                                        diff_vs_chance=d, two_se=se2, judgement=verdict))
                    if reps != REPEATS:
                        tail += f"  ← 반복 {reps}뿐, 대표값으로 쓰지 않음"
                    print(f"  {cfg:<16}{len(reps):>5}{avg[cfg].mean():>8.3f}{np.mean(run_means):>8.3f}{min(run_means):>8.3f}{max(run_means):>8.3f}{tail}")
                pairs = [(c, c.replace("+L2", "")) for c in cfgs if "+L2" in c and c.replace("+L2", "") in cfgs]
                for with_l2, without in pairs:
                    d, se2, verdict = paired(avg[with_l2], avg[without])
                    verdict = {"앞쪽이 높음": "L2 가 높음", "뒤쪽이 높음": "L2 없는 쪽이 높음"}.get(verdict, verdict)
                    print(f"  L2 비교 {without}: L2 있음 − 없음 = {d:+.3f} ± {se2:.3f}  → {verdict}")

    if args.summary_csv:
        out = Path(args.summary_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(SUMMARY).round(4).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n집계 저장 → {out}")
    print("\n읽는 법: 개가 20마리 안팎이라 '판단 불가'가 많은 것이 정상이다. 방향만 보고, 결정은 정식 dev 로 다시 한다.")
    print(BANNER)


if __name__ == "__main__":
    main()
