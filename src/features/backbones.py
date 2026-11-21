"""딥 특징 추출기 3종 (동결). 사진 한 장 → cls 벡터, mean 벡터.

- CLIP ViT-B/16: cls = 이미지 임베딩(투영 후 512차원), mean = 패치 토큰 평균(768)
- DINOv2 / DINOv3 ViT-B: cls = CLS 토큰(768), mean = 패치 토큰 평균(768)
  DINOv3는 레지스터 토큰이 있으므로 config.num_register_tokens 만큼 건너뛴다.
- dummy: 모델 없이 파이프라인을 점검하는 용도 (스모크 테스트)
"""
import numpy as np
import torch


class Backbone:
    name: str

    def preprocess(self, img):
        """PIL 이미지 → (3, H, W) 텐서. DataLoader 작업자에서 실행된다."""
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor):
        """(B, 3, H, W) → (cls[B, D1], mean[B, D2])"""
        raise NotImplementedError


class HFBackbone(Backbone):
    def __init__(self, name: str, hf_id: str, kind: str, device: str, fp16: bool):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel

        self.name, self.kind, self.device = name, kind, device
        self.dtype = torch.float16 if (fp16 and device.startswith("cuda")) else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        if kind == "clip":
            self.model = CLIPModel.from_pretrained(hf_id, torch_dtype=self.dtype)
        else:
            self.model = AutoModel.from_pretrained(hf_id, torch_dtype=self.dtype)
        self.model.eval().to(device)
        self.n_reg = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

    def preprocess(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def encode(self, pixel_values):
        pv = pixel_values.to(self.device, dtype=self.dtype)
        if self.kind == "clip":
            out = self.model.vision_model(pixel_values=pv)
            cls = self.model.visual_projection(out.pooler_output)
            mean = out.last_hidden_state[:, 1:].mean(dim=1)
        else:
            out = self.model(pixel_values=pv)
            h = out.last_hidden_state
            cls = h[:, 0]
            mean = h[:, 1 + self.n_reg:].mean(dim=1)
        return cls.float().cpu().numpy(), mean.float().cpu().numpy()


class DummyBackbone(Backbone):
    """모델 다운로드 없이 전체 흐름을 점검하는 가짜 추출기."""

    def __init__(self, dim: int = 16):
        self.name = "dummy"
        rng = np.random.default_rng(0)
        self.proj = torch.tensor(rng.normal(size=(3 * 8 * 8, dim)), dtype=torch.float32)

    def preprocess(self, img):
        img = img.resize((8, 8))
        return torch.tensor(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    @torch.no_grad()
    def encode(self, pixel_values):
        x = pixel_values.flatten(1) @ self.proj
        return x.numpy(), x.numpy()


def build_backbone(name: str, cfg: dict, device: str) -> Backbone:
    if name == "dummy":
        return DummyBackbone()
    b = cfg[name]
    return HFBackbone(name, b["hf_id"], b["kind"], device, cfg["extract"]["fp16"])
