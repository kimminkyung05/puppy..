"""[B] 모델 점검: configs/backbones.yaml의 모델이 실제로 불러와지는지 확인한다.

  - 설치된 transformers / torch / torchvision 버전
  - 모델 ID가 불러와지는지 (프로젝트의 build_backbone으로 불러온다)
  - 출력 차원(cls, mean)
  - DINOv3의 num_register_tokens 값, 토큰 수가 1(CLS) + 레지스터 + 패치와 맞는지
  불러오기 실패는 에러 메시지와 함께 '확인 필요'로 표시한다.

예:
  python -m src.features.check_models                     # backbones.yaml의 모든 모델 (가중치를 받는다)
  python -m src.features.check_models --config-only       # 가중치 없이 config만 받아 모델 ID를 확인
  python -m src.features.check_models --backbones dummy   # 모델 없이 흐름만 확인 (로컬용)
"""
import argparse
import gc
import importlib.util

import torch
import transformers
from PIL import Image

from src.common.config import load_yaml
from src.features.backbones import build_backbone


def check_config(name: str, hf_id: str, kind: str) -> None:
    """가중치를 받지 않고 config만 읽는다."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(hf_id)
    vcfg = cfg.vision_config if kind == "clip" else cfg
    print(f"[config 읽음] {name}: {hf_id}")
    print(f"    model_type={cfg.model_type}, hidden_size={getattr(vcfg, 'hidden_size', '?')}, "
          f"patch_size={getattr(vcfg, 'patch_size', '?')}"
          + (f", projection_dim={getattr(cfg, 'projection_dim', '?')}" if kind == "clip" else ""))
    print(f"    config.num_register_tokens = {getattr(cfg, 'num_register_tokens', '(config에 없음)')}")


def check_model(name: str, bcfg: dict, device: str) -> None:
    """모델을 불러와 가짜 이미지 한 장을 통과시킨다."""
    img = Image.new("RGB", (300, 260), (128, 128, 128))
    bb = build_backbone(name, bcfg, device)
    try:
        pv = bb.preprocess(img).unsqueeze(0)
        cls, mean = bb.encode(pv)
        if name == "dummy":
            print("[불러옴] dummy (모델 없음)")
            print(f"    입력 {tuple(pv.shape)} → 출력 차원: cls {cls.shape[1]}, mean {mean.shape[1]}")
            return
        hf_id, kind = bcfg[name]["hf_id"], bcfg[name]["kind"]
        cfg = bb.model.config
        print(f"[불러옴] {name}: {hf_id}")
        print(f"    모델 {type(bb.model).__name__}, 이미지 처리기 {type(bb.processor).__name__}, dtype {bb.dtype}")
        print(f"    입력 {tuple(pv.shape)} → 출력 차원: cls {cls.shape[1]}, mean {mean.shape[1]}")
        p = bb.processor   # 가운데 자르기를 하면 가로로 긴 사진은 양 끝이 잘린다. 추출기마다 다르므로 값을 그대로 보인다
        print("    전처리: " + ", ".join(f"{k}={getattr(p, k, '(없음)')}"
                                        for k in ("do_resize", "size", "do_center_crop", "crop_size")))
        print(f"    config.num_register_tokens = {getattr(cfg, 'num_register_tokens', '(config에 없음)')}"
              f" → 코드가 건너뛰는 레지스터 토큰 수 n_reg = {bb.n_reg}")
        try:
            vcfg = cfg.vision_config if kind == "clip" else cfg
            vision = bb.model.vision_model if kind == "clip" else bb.model
            with torch.no_grad():
                n_tokens = vision(pixel_values=pv.to(device, dtype=bb.dtype)).last_hidden_state.shape[1]
            n_patch = (pv.shape[-2] // vcfg.patch_size) * (pv.shape[-1] // vcfg.patch_size)
            want = 1 + bb.n_reg + n_patch
            verdict = "OK" if n_tokens == want else "불일치 → 패치 평균 범위 확인 필요"
            print(f"    토큰 수 {n_tokens}, 기대 1 + {bb.n_reg} + {n_patch} = {want}: {verdict}")
        except Exception as e:
            print(f"    토큰 수 점검을 못 함 (확인 필요): {type(e).__name__}: {e}")
        if name == "dinov3" and bb.n_reg == 0:
            print("    확인 필요: DINOv3인데 레지스터 토큰이 0으로 읽힘")
    finally:
        del bb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    bcfg = load_yaml("backbones")
    known = [k for k, v in bcfg.items() if isinstance(v, dict) and "hf_id" in v]
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=known, choices=known + ["dummy"])
    ap.add_argument("--config-only", action="store_true", help="가중치 없이 config만 받아 모델 ID를 확인")
    args = ap.parse_args()

    print(f"transformers {transformers.__version__} | torch {torch.__version__} | "
          f"torchvision {'설치됨' if importlib.util.find_spec('torchvision') else '없음'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, fp16 설정 {bcfg['extract']['fp16']}\n")

    failed = []
    for name in args.backbones:
        try:
            if args.config_only and name != "dummy":
                check_config(name, bcfg[name]["hf_id"], bcfg[name]["kind"])
            else:
                check_model(name, bcfg, device)
        except Exception as e:
            failed.append(name)
            hf_id = bcfg.get(name, {}).get("hf_id", "-")
            print(f"[확인 필요] {name}: {hf_id} 불러오기 실패")
            print(f"    {type(e).__name__}: {str(e)[:500]}")

    print("\n== 요약 ==")
    print(f"  불러옴: {[n for n in args.backbones if n not in failed]}")
    print(f"  확인 필요: {failed}")
    if failed:
        print("  DINOv3에서 401/403이면 모델 페이지의 접근 승인과 로그인 계정을, "
              "그 밖에는 backbones.yaml의 모델 ID와 transformers 버전을 확인한다.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
