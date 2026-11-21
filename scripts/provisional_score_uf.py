"""UF 초안 임시 채점. 정식 채점이 아니다 (A 의 src/eval/eval.py 가 나오면 그것으로 바꾼다).

목적: eval.py 가 나오기 전에 dev 결과를 숫자로 보고, L2 정규화 여부 같은 선택을 임시로 정한다.

입력
  preds/dev/pred_*.csv          Tsinghua 클래스 공간 예측 (dog_id, split, <클래스들>)
  truth_UF.csv                  build_uf_truth.py 가 만든 정답 비율 (믹스견만)
  breed_map_UF_draft.csv        품종 표기 → 학습 클래스 매핑 초안
  features/{backbone}/train/index.csv   학습셋 label (찍기 기준용. 없으면 찍기 줄을 비운다)

계산
  1. 예측을 DNA 품종 공간으로 합친다: 매핑 표에서 한 품종이 여러 클래스이면(예: Poodle 묶음) 그 클래스들의 예측을 더한다.
     매핑 표에 나오지 않는 클래스는 각각 그대로 한 품종으로 둔다 (상위 3품종 후보에서 빼지 않는다).
  2. 주 지표 = 예측 상위 3품종이 차지하는 실제 DNA 비율의 합 (개마다 계산해 평균).
       (a) 원래 비율: pct / 100. 학습 클래스에 없는 품종과 페이지에 안 적힌 나머지는 맞힐 수 없으므로 상한이 1 보다 작다.
       (b) 재정규화 비율: 매핑된 품종 안에서 합 1 로 다시 나눈 비율. 매핑된 품종이 없는 개는 뺀다.
  3. 찍기: 학습셋 클래스 빈도를 모든 개에 똑같이 낸 예측을 같은 방식으로 채점한다.
  4. 상한: 실제 비율이 큰 3품종을 그대로 맞혔을 때의 값.
  5. 반복(r1~r3)은 평균 · 최소 · 최대를 낸다. 비교(찍기 대비, L2 유무)는 개 단위 짝 차이의 평균 ± 2 표준오차로 보고,
     구간이 0 을 포함하면 "판단 불가"라고 적는다. 개가 20마리뿐이라 대부분 방향만 볼 수 있다.

사용 (프로젝트 루트에서. DOGMIX_PATHS 로 경로 설정을 고른다)
  python scripts/provisional_score_uf.py --uf-dir data/evalset_uf
  python scripts/provisional_score_uf.py --uf-dir {DRIVE_DIR}/evalset/uf_draft --eval-subset eval_uf_panels
  옵션: --with-similar (status=유사 도 정답 품종으로 본다), --include-check (CHECK- 점검 실행도 채점)
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
        orig[dog] = {r["target_classes"]: float(r["pct"]) / 100 for _, r in g.iterrows()}
        norm[dog] = {r["target_classes"]: float(r[norm_col]) for _, r in g.iterrows()}
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


def ceiling(truth: dict, dogs) -> pd.Series:
    return pd.Series({d: sum(sorted(truth[d].values(), reverse=True)[:TOPK]) for d in dogs}, dtype=float)


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
    ap.add_argument("--with-similar", action="store_true", help="status=유사 인 매핑도 정답 품종으로 본다")
    ap.add_argument("--include-check", action="store_true", help="CHECK- (형식 점검) 실행도 채점한다")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    args = ap.parse_args()

    paths = load_yaml("paths")
    preds_dir = Path(args.preds_root or paths["outputs"]["preds_root"]) / args.split
    feat_root = Path(args.features_root or paths["outputs"]["features_root"])
    uf = Path(args.uf_dir)
    allowed = {EXACT, SIMILAR} if args.with_similar else {EXACT}

    print("=" * 78)
    print(BANNER)
    print("=" * 78)
    print(f"예측: {preds_dir} | 평가 subset: {args.eval_subset} | 정답: {uf / 'truth_UF.csv'}")
    print(f"정답 품종으로 보는 매핑: {' + '.join(sorted(allowed))}")

    c2b = breed_of_class(uf / "breed_map_UF_draft.csv", allowed)
    orig, norm, cov = load_truth(uf / "truth_UF.csv", args.with_similar)

    runs = []
    for p in sorted(preds_dir.glob("pred_*.csv")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        ev = m["ev"] or "eval"
        if ev != args.eval_subset and not ev.startswith(f"{args.eval_subset}_limit"):   # _limitN 은 로컬 점검용
            continue
        if m["kind"] == "CHECK" and not args.include_check:
            continue
        runs.append((p, m))
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

    per_dog = {}        # (설정 이름, 반복) → {"orig": Series, "norm": Series}
    info, chance = [], {}
    for p, m in runs:
        pred = pd.read_csv(p)
        classes = [c for c in pred.columns if c not in ("dog_id", "split")]
        row_sum = pred[classes].sum(axis=1)
        notes = []
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
        cfg = f"{m['bb']}{'+L2' if m['l2'] else ''}{'+mean' if m['pool'] else ''}{' [점검]' if m['kind'] == 'CHECK' else ''}"
        per_dog[(cfg, int(m["r"]))] = {"orig": captured(merged, orig), "norm": captured(merged, norm)}
        info.append((p.name, len(dogs_all), len(classes), "; ".join(notes)))
        if freq is not None and cfg not in chance:
            f = pd.DataFrame([freq.reindex(classes).fillna(0).to_numpy()] * len(pred), columns=classes,
                             index=pred["dog_id"])
            fm = merge_to_breeds(f, c2b)
            chance[cfg] = {"orig": captured(fm, orig), "norm": captured(fm, norm)}

    print(f"\n예측 파일 {len(info)}개")
    for name, n_dogs, n_cls, note in info:
        print(f"  {name}: 개 {n_dogs}, 클래스 {n_cls}" + (f"  ← {note}" if note else ""))

    cfgs = sorted({k[0] for k in per_dog})
    dogs = sorted(set().union(*[set(v["orig"].index) for v in per_dog.values()]))
    subsets = [(f"dev 전체 {len(dogs)}마리", dogs),
               (f"coverage {args.min_coverage} 이상", [d for d in dogs if cov.get(d, 0) >= args.min_coverage])]
    kinds = [("orig", "(a) 원래 비율"), ("norm", "(b) 학습 품종 안에서 재정규화한 비율")]

    for sub_name, sub in subsets:
        for kind, kind_name in kinds:
            truth = orig if kind == "orig" else norm
            use = [d for d in sub if truth[d]] if kind == "norm" else list(sub)
            print(f"\n[{sub_name} → {len(use)}마리 | 실제 비율 {kind_name}]  지표: 예측 상위 {TOPK}품종의 실제 비율 합")
            if len(use) == 0:
                print("  해당하는 개 없음")
                continue
            ceil = float(ceiling(truth, use).mean())
            print(f"  상한(실제 상위 {TOPK}품종을 다 맞힘): {ceil:.3f}")
            print(f"  {'설정':<16}{'반복':>5}{'평균':>8}{'최소':>8}{'최대':>8}{'찍기':>8}{'찍기와 차이 ± 2SE':>22}  판단")
            avg = {}
            for cfg in cfgs:
                reps = sorted(r for c, r in per_dog if c == cfg)
                run_means = [float(per_dog[(cfg, r)][kind].reindex(use).mean()) for r in reps]
                avg[cfg] = pd.concat([per_dog[(cfg, r)][kind].reindex(use) for r in reps], axis=1).mean(axis=1)
                if cfg in chance:
                    ch = chance[cfg][kind].reindex(use)
                    d, se2, verdict = paired(avg[cfg], ch)
                    verdict = {"앞쪽이 높음": "찍기보다 높음", "뒤쪽이 높음": "찍기보다 낮음"}.get(verdict, verdict)
                    tail = f"{ch.mean():>8.3f}{d:>+13.3f} ± {se2:.3f}  {verdict}"
                else:
                    tail = f"{'-':>8}{'-':>22}  찍기 기준 없음"
                print(f"  {cfg:<16}{len(reps):>5}{np.mean(run_means):>8.3f}{min(run_means):>8.3f}{max(run_means):>8.3f}{tail}")
            pairs = [(c, c.replace("+L2", "")) for c in cfgs if "+L2" in c and c.replace("+L2", "") in cfgs]
            for with_l2, without in pairs:
                d, se2, verdict = paired(avg[with_l2], avg[without])
                verdict = {"앞쪽이 높음": "L2 가 높음", "뒤쪽이 높음": "L2 없는 쪽이 높음"}.get(verdict, verdict)
                print(f"  L2 비교 {without}: L2 있음 − 없음 = {d:+.3f} ± {se2:.3f}  → {verdict}")

    print("\n읽는 법: 개가 20마리 안팎이라 '판단 불가'가 많은 것이 정상이다. 방향만 보고, 결정은 정식 dev 로 다시 한다.")
    print(BANNER)


if __name__ == "__main__":
    main()
