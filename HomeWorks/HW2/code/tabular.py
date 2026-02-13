"""Tabular pipeline for HW2 (data download, preprocessing, training + eval).

- Loads local Pima Indians Diabetes CSV when available
- Attempts remote download when local data is missing
- Falls back to deterministic synthetic tabular data when offline
- Trains MLPClassifier (from models.py) and NAMClassifier
- Evaluates accuracy, recall, f1, confusion matrix
"""
import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from models import MLPClassifier, NAMClassifier


DATA_URL = (
    "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv"
)

DEFAULT_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def _make_synthetic_diabetes(n_samples: int = 768, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic diabetes-like tabular dataset for offline runs."""
    rng = np.random.default_rng(seed)
    preg = rng.poisson(3.5, size=n_samples).clip(0, 17)
    glucose = rng.normal(120, 30, size=n_samples).clip(50, 220)
    bp = rng.normal(72, 12, size=n_samples).clip(40, 130)
    skin = rng.normal(29, 10, size=n_samples).clip(7, 99)
    insulin = rng.lognormal(mean=4.8, sigma=0.5, size=n_samples).clip(10, 600)
    bmi = rng.normal(32, 7, size=n_samples).clip(15, 60)
    dpf = rng.gamma(shape=2.0, scale=0.22, size=n_samples).clip(0.05, 2.5)
    age = rng.normal(34, 11, size=n_samples).clip(21, 81)

    # Construct a plausible decision boundary plus noise.
    linear = (
        -6.0
        + 0.035 * glucose
        + 0.020 * bmi
        + 0.016 * age
        + 0.40 * dpf
        + 0.0018 * insulin
        + 0.050 * preg
        - 0.010 * bp
        - 0.004 * skin
        + rng.normal(0, 0.6, size=n_samples)
    )
    p = 1.0 / (1.0 + np.exp(-linear))
    outcome = rng.binomial(1, p).astype(int)

    return pd.DataFrame(
        {
            "Pregnancies": preg.astype(int),
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
            "Outcome": outcome,
        }
    )


def load_diabetes(local_path: str = "diabetes.csv", seed: int = 42) -> pd.DataFrame:
    def _normalize_target(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        lower_to_orig = {c.lower(): c for c in df_out.columns}
        if "outcome" in lower_to_orig:
            src = lower_to_orig["outcome"]
            if src != "Outcome":
                df_out = df_out.rename(columns={src: "Outcome"})
        elif "diabetes" in lower_to_orig:
            df_out = df_out.rename(columns={lower_to_orig["diabetes"]: "Outcome"})
        elif "class" in lower_to_orig:
            df_out = df_out.rename(columns={lower_to_orig["class"]: "Outcome"})
        return df_out

    if os.path.exists(local_path):
        df = _normalize_target(pd.read_csv(local_path))
    else:
        try:
            df = _normalize_target(pd.read_csv(DATA_URL))
            df.to_csv(local_path, index=False)
        except Exception as exc:  # pragma: no cover - offline branch
            print(
                f"[tabular] Could not download dataset ({exc.__class__.__name__}). "
                "Using deterministic synthetic fallback dataset."
            )
            df = _make_synthetic_diabetes(seed=seed)

    # Enforce assignment schema (8 tabular features + Outcome).
    # If local/remote file is incompatible, fallback to deterministic synthetic data.
    required = DEFAULT_COLUMNS
    if not set(required).issubset(set(df.columns)):
        print(
            "[tabular] Found incompatible local dataset schema. "
            "Falling back to deterministic synthetic diabetes-like data."
        )
        df = _make_synthetic_diabetes(seed=seed)
    df = df[required]
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    X = df.drop(columns=["Outcome"]).values.astype(float)
    y = df["Outcome"].values.astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def make_splits(X, y, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # from temp: 10% val, 20% test => ratio within temp: val 1/3, test 2/3
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def to_loader(X, y, batch_size=64, shuffle=True):
    tX = torch.from_numpy(X).float()
    ty = torch.from_numpy(y).float()
    ds = TensorDataset(tX, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=30,
    lr=1e-3,
    device="cpu",
    return_history: bool = False,
):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = 1e9
    best_state = None
    train_losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(epoch_losses)))
        if val_loader is not None:
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    epoch_val_losses.append(
                        F.binary_cross_entropy_with_logits(model(xb), yb).item()
                    )
            mean_val = float(np.mean(epoch_val_losses))
            val_losses.append(mean_val)
            if mean_val < best_val:
                best_val = mean_val
                best_state = copy.deepcopy(model.state_dict())
        else:
            val_losses.append(float("nan"))
    if best_state is not None:
        model.load_state_dict(best_state)
    if return_history:
        history: Dict[str, List[float]] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
        return model, history
    return model


def predict_binary(model, X, device="cpu"):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    return preds, probs


def evaluate_preds(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


if __name__ == "__main__":
    df = load_diabetes()
    X, y, scaler = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    tr = to_loader(X_train, y_train, batch_size=64)
    va = to_loader(X_val, y_val, batch_size=256, shuffle=False)

    mlp = MLPClassifier()
    mlp = train_model(mlp, tr, va, epochs=30, lr=1e-3)

    preds, probs = predict_binary(mlp, X_test)
    metrics = evaluate_preds(y_test, preds)
    print("MLP test metrics:")
    for k, v in metrics.items():
        print(k, v)

    # NAM (quick train)
    nam = NAMClassifier(n_features=X.shape[1], hidden=24)
    nam = train_model(nam, tr, va, epochs=30, lr=5e-3)
    preds_n, _ = predict_binary(nam, X_test)
    print("NAM test metrics:", evaluate_preds(y_test, preds_n))
