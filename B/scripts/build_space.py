"""[B] Hugging Face Spaces 에 올릴 데모 폴더를 만든다 (업로드는 하지 않는다).

만드는 것 (--out, 기본 ../outputs/space. B/ 밖이라 git에 올라가지 않는다)
  README.md         Spaces 설정 머리말 (sdk: gradio)
  app.py            진입점. src/demo/app.py 를 고정 인자로 실행 (--host 0.0.0.0)
  requirements.txt  CPU용 torch 등
  src/...           데모에 필요한 모듈만 복사 (demo, common/config · features_io, rq1/prototype_fcls, features/backbones)
  configs/backbones.yaml
  assets/demo_dinov2.npz                     B 평균점 + 판단 보류 기준값
  assets/a/RQ1-dinov2-clf-s{1,2,3}_linear_weights.npz   A 가중치 (A 동의 후에만 올린다)
  assets/c/RQ1-dinov2-mix-v2-s{1,2,3}-lr3e-3.pt         C 체크포인트 (C 동의 후에만 올린다)
  assets/deco/*     --deco 를 줄 때만. 저작권 문제가 없는 사진만 넣는다

UF 사진 · DNA 결과 · 특징 샤드는 넣지 않는다. 다 만든 뒤 들어간 파일 목록과 크기를 출력한다.

예 (B/ 안에서)
  python scripts/build_space.py --assets ../outputs/demo/demo_dinov2.npz --a-weights ../A/A/A_to_D_handoff --c-weights ../C/C_add
로컬 점검: cd ../outputs/space; python app.py --local   (127.0.0.1:7861 에서 연다. ZeroGPU 설정은 쓰지 않는다)
무료 계정: Gradio Space 는 CPU 에서는 유료(PRO)이고, ZeroGPU 에서는 개인 계정 2개까지 무료다 (2026-11-25 확인).
  python scripts/build_space.py ... --zerogpu --out ../outputs/space_zerogpu
"""
import argparse
import shutil
from pathlib import Path

B = Path(__file__).resolve().parents[1]

MODULES = ["src/__init__.py", "src/demo/__init__.py", "src/demo/app.py",
           "src/common/__init__.py", "src/common/config.py", "src/common/features_io.py",
           "src/rq1/__init__.py", "src/rq1/prototype_fcls.py",
           "src/features/__init__.py", "src/features/backbones.py",
           "configs/backbones.yaml"]

README = """---
title: Dog Mix Breed Demo
emoji: 🐕
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: {gradio}
python_version: "3.12"
app_file: app.py
pinned: false
---

# 믹스견 사진 → 혈통 비율 (CV 팀 프로젝트 데모)

같은 사진 · 같은 특징(DINOv2 CLS)에 세 가지 풀이(A 분류기 · B 평균점 섞기 · C 섞기 학습)를 적용해
증조부모 8칸 가계도로 보여 줍니다. 가계도는 족보가 아니라 비율의 표현입니다.

- 학습: Tsinghua Dogs 학습셋 (127품종)
- 올린 사진은 결과 계산에만 쓰고 따로 저장하지 않습니다.
"""

REQUIREMENTS = """{torch}
transformers=={transformers}
numpy
scipy
pandas
pyyaml
pillow
"""

LAUNCHER = '''"""Spaces 진입점: src/demo/app.py 를 고정 인자로 실행한다. 로컬 점검은 python app.py --local"""
import sys

from src.demo import app

local = "--local" in sys.argv
sys.argv = ["app", "--assets", "assets/demo_dinov2.npz", "--a-weights", "assets/a"{c}{deco},
            "--host", "127.0.0.1" if local else "0.0.0.0", "--port", "7861" if local else "7860"]
if not local:
    sys.argv += {server_extra}
app.main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--a-weights", required=True)
    ap.add_argument("--c-weights", default=None)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--deco", nargs="*", default=[], help="저작권 문제가 없는 사진만")
    ap.add_argument("--out", default=str(B.parent / "outputs" / "space"))
    ap.add_argument("--zerogpu", action="store_true",
                    help="무료 계정용 ZeroGPU Space (GPU torch>=2.8, spaces 패키지, @spaces.GPU). 기본은 CPU Space")
    args = ap.parse_args()

    import gradio
    import transformers

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} 가 비어 있지 않음. 지우거나 다른 --out 을 주세요 (덮어쓰지 않는다)")
    out.mkdir(parents=True, exist_ok=True)

    for m in MODULES:
        (out / m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(B / m, out / m)
    (out / "assets").mkdir()
    shutil.copy2(args.assets, out / "assets" / "demo_dinov2.npz")
    (out / "assets" / "a").mkdir()
    for s in (1, 2, 3):
        name = f"RQ1-{args.backbone}-clf-s{s}_linear_weights.npz"
        shutil.copy2(Path(args.a_weights) / name, out / "assets" / "a" / name)
    c_arg = ""
    if args.c_weights:
        (out / "assets" / "c").mkdir()
        for s in (1, 2, 3):
            name = f"RQ1-{args.backbone}-mix-v2-s{s}-lr3e-3.pt"
            shutil.copy2(Path(args.c_weights) / name, out / "assets" / "c" / name)
        c_arg = ', "--c-weights", "assets/c"'
    deco_arg = ""
    if args.deco:
        (out / "assets" / "deco").mkdir()
        names = []
        for p in args.deco:
            shutil.copy2(p, out / "assets" / "deco" / Path(p).name)
            names.append(f'"assets/deco/{Path(p).name}"')
        deco_arg = ', "--deco", ' + ", ".join(names)

    (out / "README.md").write_bytes(README.format(gradio=gradio.__version__).encode("utf-8"))
    torch_req = ("torch>=2.8\nspaces" if args.zerogpu
                 else "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch")
    (out / "requirements.txt").write_bytes(
        REQUIREMENTS.format(torch=torch_req, transformers=transformers.__version__).encode("utf-8"))
    extra = '["--zerogpu"]' if args.zerogpu else "[]"
    (out / "app.py").write_bytes(LAUNCHER.format(c=c_arg, deco=deco_arg, server_extra=extra).encode("utf-8"))

    total = 0
    for f in sorted(out.rglob("*")):
        if f.is_file():
            total += f.stat().st_size
            print(f"{f.stat().st_size:>10,}  {f.relative_to(out).as_posix()}")
    print(f"합계 {total / 1e6:.1f} MB → {out}")


if __name__ == "__main__":
    main()
