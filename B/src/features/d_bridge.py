"""D(RQ2) 노트북에서 B와 같은 방식으로 특징을 뽑는 추출기.

A 분류기는 B가 뽑은 특징(configs/PREPROCESSING.md)으로 학습했다. D의 변형 · 대조군 사진도
같은 모델 · 전처리 · fp16 · cls 정의로 뽑아야 분류기 입력이 맞는다.

D 노트북의 추출기 자리(`extractor.encode(images)`, 결과 [N, D] 배열)에 그대로 끼운다.
D 파일은 고치지 않고, D 노트북 세션에서 아래 셀만 더 실행한다 (셀 6 · 7 다음, 셀 10 전):

    import sys; sys.path.insert(0, "/content/dog-mix-cv/B")      # 저장소 clone 위치
    from src.features.d_bridge import BExtractor
    for name in ["dinov3"]:                    # clip · dinov2 도 B 방식으로 바꾸려면 넣는다
        _extractor_cache[name] = BExtractor(name)

이미 다른 방식으로 뽑은 캐시(D_outputs/feature_cache/{split}/{name}/*.npy)가 있으면 셀 10이 그 파일을 읽으므로,
바꾸는 추출기는 캐시 폴더를 비우거나 다른 이름으로 옮긴 뒤 실행한다.

특징만 만들어 넘기려면 셀 10 대신 다음을 실행한다 (분류기 체크포인트 없이 캐시 .npy만 생긴다):

    for cond in CONDITIONS:
        extract_features_for_condition("dinov3", cond)

D/scripts/run_d_experiment.py 로 돌리는 경우에는 main() 의 extractors 사전에 넣는다 (D 담당이 한 줄 추가):

    extractors["dinov3"] = BExtractor("dinov3")

DINOv3 는 Hugging Face 접근 승인이 필요한 모델이라, 실행 세션에서 승인된 계정으로
`from huggingface_hub import login; login()` 을 먼저 한다.

출력: cls 특징 float32 [N, D] (CLIP 512, DINOv2 · DINOv3 768). L2 정규화는 하지 않는다(A 분류기 쪽에서 한다).
B의 공유 특징 파일은 float16 으로 저장되지만 여기서는 float32 그대로 돌려준다 (차이는 반올림 수준).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.common.config import load_yaml
from src.features.backbones import build_backbone


class BExtractor:
    """B 명세(configs/backbones.yaml, PREPROCESSING.md)와 같은 추출기. encode(images) → cls [N, D]."""

    def __init__(self, name: str, device: str | None = None, batch_size: int = 64):
        if name not in ("clip", "dinov2", "dinov3", "dummy"):
            raise ValueError(f"지원하지 않는 추출기: {name}")
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.backbone = build_backbone(name, load_yaml("backbones"), self.device)

    @staticmethod
    def _to_pil(im) -> Image.Image:
        if isinstance(im, Image.Image):
            return im.convert("RGB")
        arr = np.asarray(im)
        if arr.dtype != np.uint8:
            raise ValueError(f"numpy 이미지는 uint8 RGB 여야 함 (받은 dtype: {arr.dtype})")
        return Image.fromarray(arr).convert("RGB")

    def encode(self, images) -> np.ndarray:
        out = []
        images = list(images)
        for s in range(0, len(images), self.batch_size):
            batch = torch.stack([self.backbone.preprocess(self._to_pil(im)) for im in images[s:s + self.batch_size]])
            cls, _ = self.backbone.encode(batch)
            out.append(np.asarray(cls, dtype=np.float32))
        if not out:
            raise ValueError("이미지가 없음")
        return np.concatenate(out, axis=0)
