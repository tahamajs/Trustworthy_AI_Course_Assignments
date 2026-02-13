import numpy as np
from fairness import (
    accuracy,
    apply_group_thresholds,
    apply_promotion_demotion,
    disparate_impact,
    optimize_group_thresholds,
    train_reweighed_model,
    zemel_proxy_fairness,
)


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
    y_proba = np.array([0.1, 0.4, 0.9, 0.8, 0.6, 0.2])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    sensitive = np.array([1, 1, 1, 0, 0, 0])
    mask = apply_promotion_demotion(y_proba, y_pred, sensitive, k=1)
    assert mask.dtype == bool
    # promotion candidate index 0, demotion candidate index 3
    assert mask.sum() == 2
    assert mask[0]
    assert mask[3]


def test_train_reweighed_model():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 4)
    y = (rng.rand(40) > 0.6).astype(int)
    sensitive = rng.randint(0, 2, size=40)
    model = train_reweighed_model(X, y, sensitive)
    assert hasattr(model, "_scaler")
    preds = model.predict(model._scaler.transform(X))
    assert preds.shape == (40,)


def test_group_thresholds_bounds():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.2, 0.7, 0.4, 0.8, 0.6, 0.9])
    sensitive = np.array([0, 0, 1, 1, 0, 1])
    thresholds = optimize_group_thresholds(y_true, y_proba, sensitive)
    assert 0.0 <= thresholds[0] <= 1.0
    assert 0.0 <= thresholds[1] <= 1.0
    pred = apply_group_thresholds(y_proba, sensitive, thresholds)
    assert pred.shape == y_true.shape
