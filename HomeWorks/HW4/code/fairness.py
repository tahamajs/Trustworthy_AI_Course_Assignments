"""Fairness utilities for HW4 (Q3 + bonus methods)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # assume `income` is the label (0/1) and `gender` is sensitive (0/1 already encoded)
    if "income" not in df.columns:
        raise ValueError("data.csv must contain an 'income' column")
    return df.drop(columns=["income"]), df["income"]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def disparate_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    protected_value: int = 0,
    privileged_value: int = 1,
) -> float:
    # Disparate Impact = P(pred=1 | protected) / P(pred=1 | privileged)
    protected_mask = sensitive == protected_value
    privileged_mask = sensitive == privileged_value
    p_prot = y_pred[protected_mask].mean() if protected_mask.any() else 0.0
    p_priv = y_pred[privileged_mask].mean() if privileged_mask.any() else 0.0
    if p_priv == 0:
        return float('inf') if p_prot > 0 else 1.0
    return float(p_prot / p_priv)


def zemel_proxy_fairness(
    X: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    n_clusters: int = 5,
    max_samples: int = 4000,
) -> float:
    """A clustering-based proxy for the Zemel fairness objective.

    Steps:
    - cluster X into `n_clusters` clusters (KMeans)
    - for each cluster compute positive-rate difference across sensitive groups
    - return the mean absolute difference (lower == fairer)
    """
    if X.shape[0] == 0:
        return 0.0
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(0)
        keep = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[keep]
        y_pred = y_pred[keep]
        sensitive = sensitive[keep]
    k = min(n_clusters, X.shape[0])
    km = KMeans(n_clusters=k, random_state=0, n_init=5)
    labels = km.fit_predict(X)
    diffs = []
    for c in np.unique(labels):
        idx = labels == c
        if idx.sum() == 0:
            continue
        groups = np.unique(sensitive[idx])
        if groups.size < 2:
            # cluster contains single sensitive group -> no cross-group diff
            continue
        rates = []
        for g in groups:
            mask = idx & (sensitive == g)
            rates.append(y_pred[mask].mean() if mask.sum() > 0 else 0.0)
        # absolute pairwise differences averaged
        arr = np.array(rates)
        diffs.append(float(np.abs(arr[:, None] - arr).mean()))
    return float(np.mean(diffs)) if len(diffs) > 0 else 0.0


def train_baseline_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(Xs, y)
    # attach scaler for convenience
    clf._scaler = scaler
    return clf


def apply_promotion_demotion(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    k: int,
) -> np.ndarray:
    """Prediction-based Promotion/Demotion index selection.

    Locked rule:
    - Promotion (CP): men with predicted label 0, ranked by ascending P(y=1).
    - Demotion  (CD): women with predicted label 1, ranked by descending P(y=1).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    n = y_proba.shape[0]
    if y_pred.shape[0] != n or sensitive.shape[0] != n:
        raise ValueError("y_proba, y_pred, and sensitive must have the same length")

    men_mask = sensitive == 1
    women_mask = sensitive == 0
    prom_cand = np.where(men_mask & (y_pred == 0))[0]
    dem_cand = np.where(women_mask & (y_pred == 1))[0]

    prom_scores = y_proba[prom_cand]
    dem_scores = y_proba[dem_cand]

    prom_order = prom_cand[np.argsort(prom_scores)]
    dem_order = dem_cand[np.argsort(-dem_scores)]

    k_prom = min(k, prom_order.size)
    k_dem = min(k, dem_order.size)

    swap_idx = np.zeros(n, dtype=bool)
    if k_prom > 0:
        swap_idx[prom_order[:k_prom]] = True
    if k_dem > 0:
        swap_idx[dem_order[:k_dem]] = True
    return swap_idx


def retrain_with_swapped_labels(X: np.ndarray, y: np.ndarray, swap_mask: np.ndarray) -> LogisticRegression:
    y_new = y.copy()
    y_new[swap_mask] = 1 - y_new[swap_mask]
    clf = train_baseline_model(X, y_new)
    return clf


def _group_label_reweighing(sensitive: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kamiran-Calders style reweighing weights."""
    n = y.shape[0]
    weights = np.ones(n, dtype=float)
    unique_s = np.unique(sensitive)
    unique_y = np.unique(y)

    p_s = {s: float((sensitive == s).mean()) for s in unique_s}
    p_y = {label: float((y == label).mean()) for label in unique_y}

    for s in unique_s:
        for label in unique_y:
            mask = (sensitive == s) & (y == label)
            joint = float(mask.mean())
            if joint <= 0:
                continue
            weights[mask] = (p_s[s] * p_y[label]) / joint
    return weights


def train_reweighed_model(X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> LogisticRegression:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sample_weight = _group_label_reweighing(sensitive=sensitive, y=y)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(Xs, y, sample_weight=sample_weight)
    clf._scaler = scaler
    clf._sample_weight = sample_weight
    return clf


def apply_group_thresholds(y_proba: np.ndarray, sensitive: np.ndarray, thresholds: Dict[int, float]) -> np.ndarray:
    y_pred = np.zeros_like(y_proba, dtype=int)
    for group in np.unique(sensitive):
        thr = float(thresholds.get(int(group), 0.5))
        mask = sensitive == group
        y_pred[mask] = (y_proba[mask] >= thr).astype(int)
    return y_pred


def optimize_group_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive: np.ndarray,
    grid: np.ndarray | None = None,
) -> Dict[int, float]:
    """Brute-force threshold search minimizing fairness gap with mild accuracy regularization."""
    if grid is None:
        grid = np.linspace(0.1, 0.9, 33)

    best_thresholds = {0: 0.5, 1: 0.5}
    best_score = float("inf")

    for t0 in grid:
        for t1 in grid:
            thresholds = {0: float(t0), 1: float(t1)}
            pred = apply_group_thresholds(y_proba, sensitive, thresholds)
            acc = accuracy(y_true, pred)
            di = disparate_impact(y_true, pred, sensitive)
            if not np.isfinite(di):
                di_gap = 1.0
            else:
                di_gap = abs(1.0 - di)
            score = di_gap + 0.2 * (1.0 - acc)
            if score < best_score:
                best_score = score
                best_thresholds = thresholds
    return best_thresholds


def compute_fairness_metrics(
    X_repr: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy(y_true, y_pred)),
        "disparate_impact": float(disparate_impact(y_true, y_pred, sensitive)),
        "zemel_proxy": float(zemel_proxy_fairness(X_repr, y_pred, sensitive)),
    }


if __name__ == "__main__":
    # quick demo when run directly
    import os
    path = os.path.join(os.path.dirname(__file__), "data.csv")
    Xdf, y = load_data(path)
    sensitive = Xdf["gender"].to_numpy()
    Xnum = Xdf.select_dtypes(include=[np.number]).drop(columns=["gender"]).to_numpy()
    Xtrain, Xtest, ytrain, ytest, sens_train, sens_test = train_test_split(
        Xnum, y.to_numpy(), sensitive, test_size=0.3, random_state=0
    )
    clf = train_baseline_model(Xtrain, ytrain)
    Xs_test = clf._scaler.transform(Xtest)
    yproba = clf.predict_proba(Xs_test)[:, 1]
    ypred = (yproba >= 0.5).astype(int)
    print(compute_fairness_metrics(Xs_test, ytest, ypred, sens_test))
