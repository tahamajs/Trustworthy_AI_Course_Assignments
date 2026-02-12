"""Fairness utilities for HW4 (Q3)

Provides:
- load_data(path)
- accuracy(y_true, y_pred)
- disparate_impact(y_true, y_pred, sensitive)  # ratio P(Y=1|protected)/P(Y=1|privileged)
- zemel_proxy_fairness(X, y_pred, sensitive, n_clusters=5)  # a simple Zemel-style proxy
- train_baseline_model(X_train, y_train)  # sklearn logistic regression
- apply_promotion_demotion(X, y_proba, y_pred, sensitive, k)
- retrain_with_swapped_labels(X, y, swap_idx)

Notes:
- The ``zemel_proxy_fairness`` implemented here is a practical proxy (clustering-based) useful for the assignment
  rather than a full reproduction of Zemel et al. 2013 training algorithm.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # assume `income` is the label (0/1) and `gender` is sensitive (0/1 already encoded)
    if "income" not in df.columns:
        raise ValueError("data.csv must contain an 'income' column")
    return df.drop(columns=["income"]), df["income"]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def disparate_impact(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, protected_value=0, privileged_value=1) -> float:
    # Disparate Impact = P(pred=1 | protected) / P(pred=1 | privileged)
    protected_mask = sensitive == protected_value
    privileged_mask = sensitive == privileged_value
    p_prot = y_pred[protected_mask].mean() if protected_mask.any() else 0.0
    p_priv = y_pred[privileged_mask].mean() if privileged_mask.any() else 0.0
    if p_priv == 0:
        return float('inf') if p_prot > 0 else 1.0
    return float(p_prot / p_priv)


def zemel_proxy_fairness(X: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, n_clusters: int = 5) -> float:
    """A clustering-based proxy for the Zemel fairness objective.

    Steps:
    - cluster X into `n_clusters` clusters (KMeans)
    - for each cluster compute positive-rate difference across sensitive groups
    - return the mean absolute difference (lower == fairer)
    """
    if X.shape[0] == 0:
        return 0.0
    k = min(n_clusters, X.shape[0])
    km = KMeans(n_clusters=k, random_state=0)
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
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(Xs, y)
    # attach scaler for convenience
    clf._scaler = scaler
    return clf


def apply_promotion_demotion(X: np.ndarray, y_proba: np.ndarray, y_true: np.ndarray, sensitive: np.ndarray, k: int) -> np.ndarray:
    """Return indices to swap according to Promotion/Demotion scheme described in the assignment.

    - Promotion (CP): men with y=0, rank by ascending probability P(y=1) — pick top k to flip to 1
    - Demotion (CD): women with y=1, rank by descending probability P(y=1) — pick top k to flip to 0

    Returns a boolean index array `swap_mask` indicating rows whose labels should be flipped.
    """
    # assume sensitive: 1 == male (privileged), 0 == female (protected)
    men_mask = (sensitive == 1)
    women_mask = (sensitive == 0)
    # promotion candidates: men with true label 0
    prom_cand = np.where(men_mask & (y_true == 0))[0]
    dem_cand = np.where(women_mask & (y_true == 1))[0]

    # rank
    prom_scores = y_proba[prom_cand]
    dem_scores = y_proba[dem_cand]

    prom_order = prom_cand[np.argsort(prom_scores)]  # ascending -> take top (largest uplift probability)
    dem_order = dem_cand[np.argsort(-dem_scores)]  # descending

    k_prom = min(k, prom_order.size)
    k_dem = min(k, dem_order.size)

    swap_idx = np.zeros(X.shape[0], dtype=bool)
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


if __name__ == "__main__":
    # quick demo when run directly
    import os
    path = os.path.join(os.path.dirname(__file__), "data.csv")
    Xdf, y = load_data(path)
    sensitive = Xdf["gender"].to_numpy()
    # drop categorical columns for demo
    Xnum = Xdf.select_dtypes(include=[np.number]).to_numpy()
    Xtrain, Xtest, ytrain, ytest, sens_train, sens_test = train_test_split(
        Xnum, y.to_numpy(), sensitive, test_size=0.3, random_state=0
    )
    clf = train_baseline_model(Xtrain, ytrain)
    Xs_test = clf._scaler.transform(Xtest)
    yproba = clf.predict_proba(Xs_test)[:, 1]
    ypred = (yproba >= 0.5).astype(int)
    print("Accuracy:", accuracy(ytest, ypred))
    print("Disparate impact:", disparate_impact(ytest, ypred, sens_test))
    print("Zemel-proxy fairness:", zemel_proxy_fairness(Xs_test, ypred, sens_test))
