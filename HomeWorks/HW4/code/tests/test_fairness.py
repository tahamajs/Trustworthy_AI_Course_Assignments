import numpy as np
from fairness import accuracy, disparate_impact, zemel_proxy_fairness, apply_promotion_demotion


def test_accuracy():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert abs(accuracy(y_true, y_pred) - 0.75) < 1e-8


def test_disparate_impact():
    y_pred = np.array([1, 1, 0, 0])
    sensitive = np.array([0, 1, 0, 1])
    # protected (0) positive rate = 0.5, privileged (1) positive rate = 0.5 -> ratio 1.0
    assert abs(disparate_impact(None, y_pred, sensitive) - 1.0) < 1e-8


def test_zemel_proxy_simple():
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    sensitive = np.array([0] * 5 + [1] * 5)
    val = zemel_proxy_fairness(X, y_pred, sensitive, n_clusters=2)
    assert val >= 0.0


def test_apply_promotion_demotion():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    y_proba = rng.rand(20)
    y_true = np.array([0]*10 + [1]*10)
    sensitive = np.array([1]*10 + [0]*10)
    mask = apply_promotion_demotion(X, y_proba, y_true, sensitive, k=3)
    assert mask.dtype == bool
    # at most 6 swapped (3 promo + 3 demo)
    assert mask.sum() <= 6
