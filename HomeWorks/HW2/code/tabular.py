"""Tabular pipeline for HW2 (data download, preprocessing, training + eval).

- Auto-downloads Pima Indians Diabetes CSV if not present
- Trains MLPClassifier (from models.py) and NAMClassifier
- Evaluates accuracy, recall, f1, confusion matrix
"""
import os
from typing import Tuple

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


def load_diabetes(local_path: str = "diabetes.csv") -> pd.DataFrame:
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(DATA_URL)
        df.to_csv(local_path, index=False)
    # standardize column names to match assignment (8 features + Outcome)
    if "Outcome" not in df.columns:
        # this remote file uses 'diabetes' as column name for target
        if "diabetes" in df.columns:
            df = df.rename(columns={"diabetes": "Outcome"})
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


def train_model(model, train_loader, val_loader=None, epochs=30, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = 1e9
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(
                        F.binary_cross_entropy_with_logits(model(xb), yb).item()
                    )
            mean_val = float(np.mean(val_losses))
            if mean_val < best_val:
                best_val = mean_val
                best_state = model.state_dict()
    if val_loader is not None and "best_state" in locals():
        model.load_state_dict(best_state)
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
