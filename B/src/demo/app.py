"""[B] 가계도형 Gradio 데모: 같은 사진에 A 분류기 · B 평균점 섞기 · C 섞기 학습을 나란히 보여 준다.

화면: 증조부모 8칸(12.5% 단위) × 3 (A · B · C) · 판단 보류 · '이렇게 판단했어요'(RQ2, 보고서 안내) · '족보가 아닌 비율의 표현' 안내
설정은 하나만 쓴다: dinov2 · CLS (dev 20마리로 고른 데모용 선택).

  A: A 담당이 넘긴 선형 분류기 가중치(seed 1~3). 특징을 L2 정규화 → softmax → seed 평균. 온도 보정 전.
  B: scripts/build_demo_assets.py 가 만든 평균점(r1~r3) + FCLS → 반복 평균.
  C: C 담당이 넘긴 섞기 학습 체크포인트(seed 1~3). 원본 특징(L2 없음)을 (x - mu) / sd → Linear → softmax → seed 평균.
     C 의 dev 예측 CSV 와 개 단위 최대 차이 7e-7, 1위 일치 20/20 으로 확인함 (2026-11-25).
  판단 보류: B의 재구성 오차가 dev 분위수(기본 q95)보다 크면 세 가계도를 모두 비우고 참고 표만 보인다.

A · C 파일은 B/ 로 복사하지 않고 경로로만 읽는다 (--a-weights, --c-weights). --c-weights 를 빼면 A · B 둘만 보인다.

실행 (B/ 안에서. 공개 링크를 만들지 않고 이 PC에서만 연다)
  python -m src.demo.app --assets ../outputs/demo/demo_dinov2.npz --a-weights ../A/A/A_to_D_handoff \
      --c-weights ../C/C_add --deco ../outputs/demo/deco/banner1.jpg ../outputs/demo/deco/banner2.jpg
"""
import argparse
import base64
import colorsys
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np

from src.common.config import load_yaml
from src.rq1.prototype_fcls import fcls

TOP_K = 7          # 이름으로 보일 품종 수. 나머지는 '기타'로 묶어 8칸을 나눈다
SLOTS = 8          # 증조부모 8칸 = 12.5% 단위
OTHER = "기타"

NOTICE = """
**읽는 법**
- 가계도는 **족보가 아니라 비율의 표현**입니다. 증조부모 한 칸이 12.5%이고, 칸의 위치(어느 쪽 부모 쪽인지)에는 뜻이 없습니다.
  부모 · 조부모 칸은 증조부모 8칸을 같은 품종끼리 붙여 놓았을 때의 한 가지 배치이며, 두 칸이 다르면 '섞임'으로 적습니다.
- 사진 둘레의 고리는 상위 7품종과 '기타'의 실제 비율입니다 (마우스를 올리면 숫자가 보입니다).
- 가계도들은 같은 사진 · 같은 특징(DINOv2 CLS)에 **풀이 방식만 다르게** 적용한 결과입니다.
  - **A 분류기**: 순종 사진으로 학습한 선형 분류기의 확률. 온도 보정 전이라 한 품종에 몰리기 쉽습니다.
  - **B 평균점 섞기**: 품종 평균 특징들을 섞어 사진 특징을 다시 만드는 비율(FCLS). 학습 없음.
  - **C 섞기 학습**: 순종 특징을 섞어 만든 가짜 믹스로, 섞인 비율을 맞히도록 학습한 선형 모델의 출력.
- 설정(DINOv2 · CLS)은 믹스견 dev 20마리 결과로 고른 데모용 선택입니다. 성능 보증이 아닙니다.
- 개가 사진에 꽉 차게 나온 사진을 쓰세요. 학습 사진은 개 몸통 박스로 자른 사진입니다.
- 올린 사진은 결과 계산에만 쓰고 따로 저장하지 않습니다. 서버의 임시 파일은 1시간 안에 지워집니다.
"""


# ---------- 계산 (Gradio 없이 테스트할 수 있게 분리) ----------

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-12)


def predict_a(x: np.ndarray, weights: list[dict]) -> np.ndarray:
    """A 분류기: L2 정규화 → softmax → seed 평균."""
    z = l2norm(x.astype(np.float32))
    ps = []
    for w in weights:
        logits = w["weight"] @ z + w["bias"]
        logits = logits - logits.max()
        p = np.exp(logits)
        ps.append(p / p.sum())
    return np.mean(ps, axis=0)


def predict_c(x: np.ndarray, models: list[dict]) -> np.ndarray:
    """C 섞기 학습: 원본 특징을 (x - mu) / sd → Linear → softmax → seed 평균."""
    x = x.astype(np.float32)
    ps = []
    for m in models:
        logits = m["weight"] @ ((x - m["mu"]) / m["sd"]) + m["bias"]
        logits = logits - logits.max()
        p = np.exp(logits)
        ps.append(p / p.sum())
    return np.mean(ps, axis=0)


def load_c_checkpoint(path: Path) -> dict:
    """C 의 .pt 를 코드 실행 없이 연다 (weights_only=True, numpy 배열만 허용)."""
    import torch

    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype("float32")), type(np.dtype("float64")), type(np.dtype("int64")), type(np.dtype("<U32"))]
    with torch.serialization.safe_globals(safe):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    return {"weight": ck["state"]["weight"].numpy().astype(np.float32),
            "bias": ck["state"]["bias"].numpy().astype(np.float32),
            "mu": np.asarray(ck["mu"], dtype=np.float32), "sd": np.asarray(ck["sd"], dtype=np.float32),
            "classes": [str(c) for c in ck["classes"]]}


def predict_b(x: np.ndarray, P: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, float]:
    """B 평균점 섞기: 반복별 FCLS 비율과 재구성 오차의 평균."""
    x = x.astype(np.float64)
    ws, errs = [], []
    for Pr, d in zip(P.astype(np.float64), delta):
        w = fcls(Pr, x, float(d))
        ws.append(w)
        errs.append(np.linalg.norm(Pr.T @ w - x) / max(np.linalg.norm(x), 1e-12))
    w = np.mean(ws, axis=0)
    return w / w.sum(), float(np.mean(errs))


def top_parts(p: np.ndarray, classes: list[str], top_k: int = TOP_K) -> list[tuple[str, float]]:
    """상위 top_k 품종 + '기타' 묶음의 (이름, 비율). 비율 합 = 1."""
    order = np.argsort(-p)
    parts = [(classes[i], float(p[i])) for i in order[:top_k]]
    rest = float(p[order[top_k:]].sum())
    if rest > 0:
        parts.append((OTHER, rest))
    tot = sum(m for _, m in parts)
    return [(n, m / tot) for n, m in parts]


def to_slots(p: np.ndarray, classes: list[str], top_k: int = TOP_K, slots: int = SLOTS) -> list[str]:
    """비율 → 8칸. 상위 top_k 품종 + '기타'에 최대 나머지 방식으로 칸을 나눈다 (칸 수 합 = slots)."""
    parts = top_parts(p, classes, top_k)
    names = [n for n, _ in parts]
    mass = np.array([m for _, m in parts])
    exact = mass * slots
    n = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - n), kind="stable")[: slots - n.sum()]:
        n[i] += 1
    out = []
    for name, k in zip(names, n):
        out += [name] * int(k)
    return out


# ---------- 화면 ----------

def pretty(c: str) -> str:
    return c if c == OTHER else c.replace("_", " ").title()


def color(c: str) -> str:
    if c == OTHER:
        return "#bdbdbd"
    h = int(hashlib.md5(c.encode()).hexdigest()[:6], 16) / 0xFFFFFF   # 품종마다 같은 색 (A · B 공통)
    r, g, b = colorsys.hls_to_rgb(h, 0.72, 0.55)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


MIXED = "섞임"


def merge_up(level: list[str]) -> list[str]:
    """아래 두 칸이 같은 품종이면 그 품종, 다르면 '섞임'. 같은 품종끼리 붙여 놓은 한 가지 배치일 뿐 족보가 아니다."""
    return [a if a == b else MIXED for a, b in zip(level[::2], level[1::2])]


def abbr(c: str) -> str:
    if c in (OTHER, MIXED):
        return c
    words = pretty(c).split()
    return (words[0][:2] if len(words) == 1 else "".join(w[0] for w in words[:2])).upper()


def wrap2(text: str, width: int = 13) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return lines if len(lines) <= 2 else [lines[0], lines[1][: width - 1] + "…"]


def node_color(c: str) -> str:
    return "#b9a77a" if c == MIXED else color(c)


def photo_data_uri(img, size: int = 200) -> str:
    im = img.convert("RGB")
    w, h = im.size
    k = min(w, h)
    im = im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k)).resize((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tree_html(title: str, slots: list[str] | None, hold_reason: str = "",
              parts: list[tuple[str, float]] | None = None, photo: str | None = None, uid: str = "t") -> str:
    """위: 사진 + 비율 고리. 아래로 부모(2) · 조부모(4) · 증조부모(8) 원형 노드를 선으로 잇는다 (SVG)."""
    W, cx0 = 640, 320
    xs8 = [40 + 80 * i for i in range(8)]
    xs4 = [(xs8[2 * i] + xs8[2 * i + 1]) / 2 for i in range(4)]
    xs2 = [(xs4[2 * i] + xs4[2 * i + 1]) / 2 for i in range(2)]
    R0, r = 58, 25
    y0, y2, y4, y8 = 78, 250, 380, 510
    line = "stroke='currentColor' stroke-opacity='0.25' stroke-width='3' fill='none'"
    el = []

    # 사진과 비율 고리
    ring_r = R0 + 7
    circ = 2 * np.pi * ring_r
    if parts and slots is not None:
        off = 0.0
        for name, m in parts:
            el.append(f"<circle cx='{cx0}' cy='{y0}' r='{ring_r}' fill='none' stroke='{color(name)}' "
                      f"stroke-width='10' stroke-dasharray='{m * circ:.2f} {circ:.2f}' "
                      f"stroke-dashoffset='{-off * circ:.2f}' transform='rotate(-90 {cx0} {y0})'>"
                      f"<title>{html.escape(pretty(name))} {100 * m:.1f}%</title></circle>")
            off += m
    else:
        el.append(f"<circle cx='{cx0}' cy='{y0}' r='{ring_r}' fill='none' stroke='currentColor' "
                  f"stroke-opacity='0.3' stroke-width='10'/>")
    if photo:
        el.append(f"<clipPath id='clip-{uid}'><circle cx='{cx0}' cy='{y0}' r='{R0}'/></clipPath>"
                  f"<image href='{photo}' x='{cx0 - R0}' y='{y0 - R0}' width='{2 * R0}' height='{2 * R0}' "
                  f"clip-path='url(#clip-{uid})' preserveAspectRatio='xMidYMid slice'/>")
    else:
        el.append(f"<circle cx='{cx0}' cy='{y0}' r='{R0}' fill='currentColor' fill-opacity='0.08'/>")

    def heading(y, text):
        el.append(f"<text x='{cx0}' y='{y}' text-anchor='middle' font-size='14' font-weight='700' "
                  f"fill='currentColor' letter-spacing='1'>{text}</text>")

    if slots is None:
        el.append(f"<text x='{cx0}' y='{y0 + 120}' text-anchor='middle' font-size='22' font-weight='700' "
                  f"fill='currentColor'>판단 보류</text>")
        for k, ln in enumerate(wrap2(hold_reason, 46)):
            el.append(f"<text x='{cx0}' y='{y0 + 150 + 18 * k}' text-anchor='middle' font-size='13' "
                      f"fill='currentColor' fill-opacity='0.75'>{html.escape(ln)}</text>")
        H = y0 + 200
    else:
        g4, g2 = merge_up(slots), merge_up(merge_up(slots))

        def connect(px, py_bottom, cxa, cxb, cy):
            mid = (py_bottom + cy - r) / 2
            el.append(f"<path d='M{px} {py_bottom} V{mid} M{cxa} {cy - r} V{mid} H{cxb} V{cy - r}' {line}/>")

        top = y0 + ring_r + 5                    # 사진 → 부모: 세로줄을 '부모' 글자 앞뒤로 끊는다
        mid = (top + y2 - r) / 2
        el.append(f"<path d='M{cx0} {top} V{mid - 32} M{cx0} {mid - 8} V{mid} "
                  f"M{xs2[0]} {y2 - r} V{mid} H{xs2[1]} V{y2 - r}' {line}/>")
        for i in range(2):
            connect(xs2[i], y2 + r + 30, xs4[2 * i], xs4[2 * i + 1], y4)
        for i in range(4):
            connect(xs4[i], y4 + r + 30, xs8[2 * i], xs8[2 * i + 1], y8)
        heading(mid - 15, "부모")
        heading(y4 - r - 34, "조부모")
        heading(y8 - r - 34, "증조부모")

        def node(x, y, c, label=True):
            col = node_color(c)
            a = abbr(c)
            el.append(f"<g><title>{html.escape(pretty(c))}</title>"
                      f"<circle cx='{x}' cy='{y}' r='{r}' fill='{col}' fill-opacity='0.18' stroke='{col}' "
                      f"stroke-width='5'/><text x='{x}' y='{y + 4}' text-anchor='middle' "
                      f"font-size='{11 if len(a) > 2 else 13}' font-weight='700' fill='currentColor'>"
                      f"{html.escape(a)}</text>")
            if label and c not in (OTHER, MIXED):   # 원 안 글자와 같으면 이름을 다시 쓰지 않는다
                for k, ln in enumerate(wrap2(pretty(c))):
                    el.append(f"<text x='{x}' y='{y + r + 14 + 12 * k}' text-anchor='middle' font-size='10' "
                              f"fill='currentColor' fill-opacity='0.8'>{html.escape(ln)}</text>")
            el.append("</g>")

        for x, c in zip(xs2, g2):
            node(x, y2, c)
        for x, c in zip(xs4, g4):
            node(x, y4, c)
        for x, c in zip(xs8, slots):
            node(x, y8, c)
        H = y8 + r + 40
    svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' xmlns='http://www.w3.org/2000/svg' "
           f"style='font-family:inherit;color:inherit'>{''.join(el)}</svg>")
    note = ("" if slots is None else
            "<div style='font-size:11px;opacity:0.65;text-align:center'>증조부모 한 칸 = 12.5%. "
            "부모 · 조부모 칸은 같은 품종끼리 붙여 놓은 한 가지 배치이고 실제 족보가 아닙니다.</div>")
    return f"<div><h3 style='margin:4px 0;text-align:center'>{html.escape(title)}</h3>{svg}{note}</div>"


def top_table(p: np.ndarray, classes: list[str], k: int = 5) -> list[list]:
    order = np.argsort(-p)[:k]
    return [[pretty(classes[i]), f"{100 * p[i]:.1f}%"] for i in order]


class Demo:
    def __init__(self, assets: str, a_weights: str, device: str, hold_level: float, c_weights: str | None = None):
        z = np.load(assets, allow_pickle=False)
        self.classes = [str(c) for c in z["classes"]]
        self.P, self.delta = z["P"], z["delta"]
        self.meta = json.loads(str(z["meta"]))
        levels = list(z["dev_recon_levels"])
        if hold_level not in levels:
            raise ValueError(f"--hold-level 는 {levels} 중 하나")
        self.hold_thr = float(z["dev_recon_q"][levels.index(hold_level)])
        self.hold_level = hold_level

        bb = self.meta["backbone"]
        self.a = []
        for s in (1, 2, 3):
            f = Path(a_weights) / f"RQ1-{bb}-clf-s{s}_linear_weights.npz"
            w = np.load(f, allow_pickle=False)
            if [str(c) for c in w["classes"]] != self.classes:
                raise ValueError(f"클래스 순서가 평균점과 다름: {f}")
            self.a.append({"weight": w["weight"].astype(np.float32), "bias": w["bias"].astype(np.float32)})

        self.c = []
        if c_weights:
            for s in (1, 2, 3):
                f = Path(c_weights) / f"RQ1-{bb}-mix-v2-s{s}-lr3e-3.pt"
                ck = load_c_checkpoint(f)
                if ck["classes"] != self.classes:
                    raise ValueError(f"클래스 순서가 평균점과 다름: {f}")
                self.c.append(ck)
        self.methods = ["A 분류기", "B 평균점 섞기"] + (["C 섞기 학습"] if self.c else [])

        from src.features.backbones import build_backbone
        self.backbone = build_backbone(bb, load_yaml("backbones"), device)

    def features(self, img) -> np.ndarray:
        pv = self.backbone.preprocess(img.convert("RGB")).unsqueeze(0)
        cls, mean = self.backbone.encode(pv)
        return (cls if self.meta["pool"] == "cls" else mean)[0]

    def run(self, img):
        """반환: [가계도 HTML × 방법 수] + [상위 5 표 × 방법 수] + [상태 문구]"""
        n = len(self.methods)
        if img is None:
            return [""] * n + [[]] * n + ["사진을 올려 주세요."]
        x = self.features(img)
        pb, err = predict_b(x, self.P, self.delta)
        probs = [predict_a(x, self.a), pb] + ([predict_c(x, self.c)] if self.c else [])
        hold = err > self.hold_thr
        reason = (f"재구성 오차 {err:.3f} > 기준 {self.hold_thr:.3f} (dev q{int(self.hold_level * 100)}). "
                  "학습한 개 사진과 많이 달라 비율을 믿기 어렵습니다.")
        photo = photo_data_uri(img)
        trees = [tree_html(name, None if hold else to_slots(p, self.classes), reason,
                           parts=top_parts(p, self.classes), photo=photo, uid=f"m{k}")
                 for k, (name, p) in enumerate(zip(self.methods, probs))]
        tables = [top_table(p, self.classes) for p in probs]
        status = (f"B 재구성 오차 {err:.3f} (판단 보류 기준 {self.hold_thr:.3f}) → "
                  + ("**판단 보류** (아래 표는 참고용)" if hold else "판단함"))
        return trees + tables + [status]


def banner_html(paths: list[str]) -> str:
    """꾸밈 사진을 맨 위에 나란히 넣는다. 파일은 B/ 밖에 두고 경로로만 받는다 (저장소에 올리지 않음)."""
    if not paths:
        return ""
    imgs = []
    for p in paths:
        p = Path(p)
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode()
        imgs.append(f"<img src='data:{mime};base64,{data}' style='height:170px;max-width:48%;object-fit:contain'>")
    return ("<div style='display:flex;justify-content:center;align-items:center;gap:16px;"
            f"background:#fff;border-radius:10px;padding:8px'>{''.join(imgs)}</div>")


def build_ui(demo: Demo, deco: list[str] | None = None):
    import gradio as gr

    m = demo.meta
    setting = (f"설정: {m['backbone']} · {m['pool'].upper()} · 평균점 재료 {m['n_train_images']:,}장 "
               f"({m['n_classes']}품종) · 판단 보류 기준은 dev {m['n_dev_dogs']}마리 {m['n_dev_images']}장으로 정함")
    if m.get("check_only"):
        setting = "⚠️ **점검용 평균점**(학습셋 일부)이라 결과에 뜻이 없습니다. " + setting

    # delete_cache=(1시간마다, 1시간 지난 것): 올라온 사진의 임시 파일을 지운다
    with gr.Blocks(title="믹스견 혈통 비율 추정 데모", analytics_enabled=False, delete_cache=(3600, 3600)) as ui:
        if deco:
            gr.HTML(banner_html(deco))
        gr.Markdown("# 믹스견 사진 → 혈통 비율\n" + setting)
        with gr.Row():
            img = gr.Image(type="pil", label="개 사진 (개가 꽉 차게)", height=320)
            with gr.Column():
                btn = gr.Button("추정하기", variant="primary")
                status = gr.Markdown()
        trees, tables = [], []
        with gr.Row():
            for name in demo.methods:
                with gr.Column(min_width=240):
                    trees.append(gr.HTML())
                    tables.append(gr.Dataframe(headers=["품종", "비율"], label=f"{name[0]} 상위 5",
                                               interactive=False))
        with gr.Accordion("이렇게 판단했어요 (RQ2)", open=False):
            gr.Markdown("어떤 시각 단서(영역 · 질감 · 색)에 기대어 판단하는지는 사진 한 장이 아니라 "
                        "평가셋 전체에 흑백 · 흐림 · 얼굴/몸통만 남기기 변형을 적용해 분석했습니다. "
                        "결과는 보고서의 RQ2 절을 참고해 주세요.")
        gr.Markdown(NOTICE)
        outs = trees + tables + [status]
        btn.click(demo.run, inputs=img, outputs=outs)
        img.upload(demo.run, inputs=img, outputs=outs)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="scripts/build_demo_assets.py 출력 npz")
    ap.add_argument("--a-weights", required=True, help="A가 넘긴 *_linear_weights.npz 가 있는 폴더")
    ap.add_argument("--c-weights", default=None, help="C가 넘긴 RQ1-*-mix-v2-s*-lr3e-3.pt 가 있는 폴더 (없으면 A · B만)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hold-level", type=float, default=0.95, help="판단 보류 기준 dev 분위수")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="기본은 이 PC에서만. Hugging Face Spaces 처럼 서버에 올릴 때만 0.0.0.0")
    ap.add_argument("--deco", nargs="*", default=[], help="맨 위 꾸밈 사진 경로 (B/ 밖에 둔다)")
    ap.add_argument("--share", action="store_true",
                    help="gradio.live 임시 공개 주소를 만든다 (72시간, 실행 중일 때만). 기본은 끔")
    ap.add_argument("--zerogpu", action="store_true",
                    help="Hugging Face ZeroGPU Space 용: 모델을 cuda 에 올리고 run 을 @spaces.GPU 로 감싼다")
    args = ap.parse_args()
    if args.zerogpu:
        import spaces   # ZeroGPU Space 에만 있다. 모델을 만들기 전에 불러온다
        args.device = "cuda"
    demo = Demo(args.assets, args.a_weights, args.device, args.hold_level, args.c_weights)
    if args.zerogpu:
        demo.run = spaces.GPU(duration=20)(demo.run)   # 사진 1장 1~2초. 짧게 잡을수록 대기열 우선순위가 높다
    # 기본은 공개 링크를 만들지 않는다. 데모에는 UF 사진 · DNA 결과가 없다 (평균점 · A · C 가중치는 Tsinghua 학습셋만)
    build_ui(demo, args.deco).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
