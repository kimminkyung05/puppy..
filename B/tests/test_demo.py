"""데모 계산 함수 점검 (모델 · Gradio 없이). 실행: python -m tests.test_demo"""
import numpy as np

from src.demo.app import OTHER, predict_a, predict_b, predict_c, to_slots


def test_predict_c_standardizes():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(4, 6)).astype(np.float32)
    mu, sd = rng.normal(size=6).astype(np.float32), rng.uniform(0.5, 2, 6).astype(np.float32)
    x = rng.normal(size=6).astype(np.float32)
    m = {"weight": W, "bias": np.zeros(4, np.float32), "mu": mu, "sd": sd}
    p = predict_c(x, [m, m])
    logits = W @ ((x - mu) / sd)
    want = np.exp(logits - logits.max())
    assert np.allclose(p, want / want.sum(), atol=1e-6)


def test_slots_sum_to_8():
    classes = [f"c{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.dirichlet(np.ones(20) * rng.uniform(0.05, 2))
        s = to_slots(p, classes)
        assert len(s) == 8


def test_slots_known_mix():
    classes = ["a", "b", "c", "d"]
    assert to_slots(np.array([0.5, 0.25, 0.25, 0.0]), classes) == ["a"] * 4 + ["b"] * 2 + ["c"] * 2
    assert to_slots(np.array([1.0, 0, 0, 0]), classes) == ["a"] * 8


def test_slots_other():
    classes = [f"c{i}" for i in range(16)]
    s = to_slots(np.full(16, 1 / 16), classes, top_k=7)
    assert len(s) == 8 and s.count(OTHER) >= 4   # 9개 품종(56%)이 '기타'로 묶인다


def test_predict_a_is_distribution():
    rng = np.random.default_rng(1)
    ws = [{"weight": rng.normal(size=(5, 8)).astype(np.float32), "bias": np.zeros(5, np.float32)} for _ in range(3)]
    p = predict_a(rng.normal(size=8), ws)
    assert abs(p.sum() - 1) < 1e-5 and np.all(p >= 0)


def test_predict_b_recovers_mix():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(3, 6, 32))
    P[1] = P[0]
    P[2] = P[0]
    w_true = np.array([0.5, 0.375, 0.125, 0, 0, 0])
    w, err = predict_b(w_true @ P[0], P, np.full(3, 10 * np.abs(P).max()))
    assert np.abs(w - w_true).max() < 1e-3 and err < 1e-3
    assert to_slots(w, list("abcdef")) == ["a"] * 4 + ["b"] * 3 + ["c"]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("통과", name)
