"""FCLS가 알려진 혼합 비율을 되찾는지 확인 (모델 없이)."""
import numpy as np

from src.rq1.prototype_fcls import build_prototypes, fcls


def test_fcls_recovers_known_mixture():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(6, 32))                  # 품종 평균점 6개
    w_true = np.array([0.5, 0.3, 0.2, 0, 0, 0])
    x = w_true @ P
    w = fcls(P, x, delta=10 * np.abs(P).max())
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-9
    assert np.abs(w - w_true).max() < 1e-3


def test_prototypes_subsample_changes_with_seed():
    X = np.arange(40, dtype=float).reshape(20, 2)
    labels = np.array(["a"] * 10 + ["b"] * 10)
    p1 = build_prototypes(X, labels, ["a", "b"], 0.8, seed=1)
    p2 = build_prototypes(X, labels, ["a", "b"], 0.8, seed=2)
    pf = build_prototypes(X, labels, ["a", "b"], 1.0, seed=1)
    assert p1.shape == (2, 2) and not np.allclose(p1, p2)
    assert np.allclose(pf[0], X[:10].mean(0))


if __name__ == "__main__":
    test_fcls_recovers_known_mixture()
    test_prototypes_subsample_changes_with_seed()
    print("ok")
