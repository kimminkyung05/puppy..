"""B_TO_D/ 의 복사본이 B 원본과 같은지 확인한다 (B 담당이 원본을 고친 뒤 실행).

사용 (B/ 안에서): python B_TO_D/check_sync.py          → 다른 파일 목록
                 python B_TO_D/check_sync.py --update → 원본으로 복사본을 다시 만든다
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
PAIRS = [  # (원본, 복사본)
    ("src/__init__.py", "code/src/__init__.py"),
    ("src/common/__init__.py", "code/src/common/__init__.py"),
    ("src/common/config.py", "code/src/common/config.py"),
    ("src/features/__init__.py", "code/src/features/__init__.py"),
    ("src/features/backbones.py", "code/src/features/backbones.py"),
    ("src/features/d_bridge.py", "code/src/features/d_bridge.py"),
    ("configs/backbones.yaml", "code/configs/backbones.yaml"),
    ("configs/PREPROCESSING.md", "code/configs/PREPROCESSING.md"),
    ("src/features/extract_deep.py", "reference/extract_deep.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="다른 복사본을 원본으로 덮어쓴다")
    args = ap.parse_args()
    diff = [(s, d) for s, d in PAIRS if not (HERE / d).exists() or not filecmp.cmp(B / s, HERE / d, shallow=False)]
    if not diff:
        print(f"복사본 {len(PAIRS)}개 모두 원본과 같음")
        return
    for s, d in diff:
        print(f"[다름] {d}  ←  B/{s}")
        if args.update:
            (HERE / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(B / s, HERE / d)
    print("→ 갱신함" if args.update else "→ --update 로 다시 복사")
    sys.exit(0 if args.update else 1)


if __name__ == "__main__":
    main()
