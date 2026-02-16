"""Generate all core HW2 report plots (tabular + vision) in one command.

Usage:
    python code/generate_report_plots.py
    python code/generate_report_plots.py --images img1.jpg img2.png ...
    HW2_VISION_IMAGES="path1,path2,..." python code/generate_report_plots.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import models as tv_models

from interpretability import lime_explain, shap_explain
from models import MLPClassifier, NAMClassifier
from tabular import (
    evaluate_preds,
    load_diabetes,
    make_splits,
    predict_binary,
    preprocess,
    to_loader,
    train_model,
)
from vision import (
    GradCAM,
    GuidedBackprop,
    activation_maximization,
    get_vgg16,
    preprocess_image,
    smoothgrad,
    smoothgrad_guided_backprop,
    smoothgrad_guided_gradcam,
)


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
DATA_PATH = ROOT / "code" / "diabetes.csv"
METRICS_JSON = FIG_DIR / "metrics_summary.json"


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _predict_fn_factory(model: torch.nn.Module):
    def predict_fn(x_in: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_in).float())
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1.0 - probs, probs]).T

    return predict_fn


def _normalize_shap_output(shap_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Some SHAP versions return (classes, samples, features).
        arr = arr[0]
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _rule_to_feature_weight(rule_name: str, feature_names: list[str]) -> str | None:
    for feat in feature_names:
        if feat in rule_name:
            return feat
    return None


def _lime_vector(lime_items: list[tuple[str, float]], feature_names: list[str]) -> np.ndarray:
    vec = np.zeros(len(feature_names), dtype=float)
    for rule_name, weight in lime_items:
        feat = _rule_to_feature_weight(rule_name, feature_names)
        if feat is None:
            continue
        vec[feature_names.index(feat)] += float(weight)
    return vec


def _plot_training_curves(
    mlp_hist: dict[str, list[float]],
    nam_hist: dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    configs = [("MLP", mlp_hist, axes[0]), ("NAM", nam_hist, axes[1])]
    for name, hist, ax in configs:
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train loss", color="#1d3557")
        ax.plot(epochs, hist["val_loss"], label="Val loss", color="#e63946")
        ax.set_title(f"{name} training dynamics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "training_loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")


def _plot_confusion_matrices(
    cm_mlp: np.ndarray,
    cm_nam: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    configs = [("MLP", cm_mlp, axes[0]), ("NAM", cm_nam, axes[1])]
    for name, cm, ax in configs:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cbar=False,
            cmap="Blues",
            ax=ax,
            square=True,
        )
        ax.set_title(f"{name} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)
    fig.tight_layout()
    out = FIG_DIR / "confusion_matrix_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")


def _plot_roc_pr_curves(
    y_test: np.ndarray,
    probs_mlp: np.ndarray,
    probs_nam: np.ndarray,
) -> tuple[float, float, float, float]:
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, probs_mlp)
    fpr_nam, tpr_nam, _ = roc_curve(y_test, probs_nam)
    roc_auc_mlp = roc_auc_score(y_test, probs_mlp)
    roc_auc_nam = roc_auc_score(y_test, probs_nam)
    ap_mlp = average_precision_score(y_test, probs_mlp)
    ap_nam = average_precision_score(y_test, probs_nam)

    from sklearn.metrics import precision_recall_curve  # lazy import for clarity

    prec_mlp, rec_mlp, _ = precision_recall_curve(y_test, probs_mlp)
    prec_nam, rec_nam, _ = precision_recall_curve(y_test, probs_nam)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr_mlp, tpr_mlp, color="#1d3557", label=f"MLP AUC={roc_auc_mlp:.3f}")
    axes[0].plot(fpr_nam, tpr_nam, color="#e63946", label=f"NAM AUC={roc_auc_nam:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC comparison")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(rec_mlp, prec_mlp, color="#1d3557", label=f"MLP AP={ap_mlp:.3f}")
    axes[1].plot(rec_nam, prec_nam, color="#e63946", label=f"NAM AP={ap_nam:.3f}")
    axes[1].set_title("Precision-recall comparison")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "roc_pr_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")
    return roc_auc_mlp, roc_auc_nam, ap_mlp, ap_nam


def _plot_calibration(
    y_test: np.ndarray,
    probs_mlp: np.ndarray,
    probs_nam: np.ndarray,
) -> tuple[float, float]:
    frac_pos_mlp, mean_pred_mlp = calibration_curve(y_test, probs_mlp, n_bins=10)
    frac_pos_nam, mean_pred_nam = calibration_curve(y_test, probs_nam, n_bins=10)
    brier_mlp = brier_score_loss(y_test, probs_mlp)
    brier_nam = brier_score_loss(y_test, probs_nam)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], "k--", label="Ideal calibration")
    ax.plot(
        mean_pred_mlp,
        frac_pos_mlp,
        marker="o",
        color="#1d3557",
        label=f"MLP (Brier={brier_mlp:.3f})",
    )
    ax.plot(
        mean_pred_nam,
        frac_pos_nam,
        marker="o",
        color="#e63946",
        label=f"NAM (Brier={brier_nam:.3f})",
    )
    ax.set_title("Probability calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive frequency")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "calibration_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")
    return brier_mlp, brier_nam


def _plot_agreement(agreement_rows: list[dict[str, float]]) -> None:
    labels = [f"Sample {int(row['sample'])}" for row in agreement_rows]
    spearman_vals = [row["spearman"] for row in agreement_rows]
    overlap_vals = [row["top3_overlap"] for row in agreement_rows]

    x = np.arange(len(labels))
    width = 0.38
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, spearman_vals, width=width, label="Spearman")
    ax.bar(x + width / 2, overlap_vals, width=width, label="Top-3 overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Agreement score")
    ax.set_title("LIME-SHAP local agreement")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "lime_shap_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")


def _plot_threshold_sensitivity(
    y_test: np.ndarray,
    probs_mlp: np.ndarray,
    probs_nam: np.ndarray,
) -> dict[str, float]:
    thresholds = np.linspace(0.05, 0.95, 19)

    def _metrics_vs_t(probs: np.ndarray):
        precs, recs, f1s = [], [], []
        for t in thresholds:
            pred = (probs >= t).astype(int)
            precs.append(precision_score(y_test, pred, zero_division=0))
            recs.append(recall_score(y_test, pred, zero_division=0))
            f1s.append(f1_score(y_test, pred, zero_division=0))
        return np.asarray(precs), np.asarray(recs), np.asarray(f1s)

    p_mlp, r_mlp, f_mlp = _metrics_vs_t(probs_mlp)
    p_nam, r_nam, f_nam = _metrics_vs_t(probs_nam)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    configs = [
        ("MLP", p_mlp, r_mlp, f_mlp, axes[0], "#1d3557"),
        ("NAM", p_nam, r_nam, f_nam, axes[1], "#e63946"),
    ]
    for model_name, p, r, f, ax, base_color in configs:
        ax.plot(thresholds, p, color=base_color, label="Precision")
        ax.plot(thresholds, r, color="#2a9d8f", label="Recall")
        ax.plot(thresholds, f, color="#f4a261", label="F1")
        best_i = int(np.argmax(f))
        ax.scatter([thresholds[best_i]], [f[best_i]], color="black", s=25, zorder=5)
        ax.set_title(f"{model_name} threshold sensitivity")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = FIG_DIR / "threshold_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")

    best_mlp_i = int(np.argmax(f_mlp))
    best_nam_i = int(np.argmax(f_nam))
    return {
        "mlp_best_threshold": float(thresholds[best_mlp_i]),
        "mlp_best_f1": float(f_mlp[best_mlp_i]),
        "nam_best_threshold": float(thresholds[best_nam_i]),
        "nam_best_f1": float(f_nam[best_nam_i]),
    }


def _permutation_importance(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 8,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_preds, _ = predict_binary(model, X_test)
    base_acc = float(np.mean(base_preds == y_test))
    importances = np.zeros(X_test.shape[1], dtype=float)
    for feat_idx in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            col = X_perm[:, feat_idx].copy()
            rng.shuffle(col)
            X_perm[:, feat_idx] = col
            perm_preds, _ = predict_binary(model, X_perm)
            perm_acc = float(np.mean(perm_preds == y_test))
            drops.append(base_acc - perm_acc)
        importances[feat_idx] = float(np.mean(drops))
    return importances


def _plot_permutation_importance(
    feat_names: list[str],
    imp_mlp: np.ndarray,
    imp_nam: np.ndarray,
) -> None:
    order = np.argsort(np.abs(imp_mlp) + np.abs(imp_nam))
    sorted_feats = [feat_names[i] for i in order]
    sorted_mlp = imp_mlp[order]
    sorted_nam = imp_nam[order]

    y_pos = np.arange(len(sorted_feats))
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(111)
    ax.barh(y_pos - 0.18, sorted_mlp, height=0.34, label="MLP", color="#1d3557")
    ax.barh(y_pos + 0.18, sorted_nam, height=0.34, label="NAM", color="#e63946")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats)
    ax.set_xlabel("Accuracy drop after feature permutation")
    ax.set_title("Permutation feature importance (model comparison)")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = FIG_DIR / "permutation_importance_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")


def _top_correlation_pairs(df, feat_cols: list[str], top_n: int = 2) -> list[dict[str, str | float]]:
    """Top off-diagonal feature pairs by absolute correlation (excluding Outcome)."""
    corr_feat = df[feat_cols].corr()
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            a, b = feat_cols[i], feat_cols[j]
            r = float(corr_feat.loc[a, b])
            pairs.append((a, b, abs(r)))
    pairs.sort(key=lambda x: -x[2])
    out = []
    for a, b, abs_rho in pairs[:top_n]:
        out.append({"pair": f"{a} -- {b}", "abs_rho": round(abs_rho, 4)})
    return out


def _plot_eda_figures(df) -> list[dict[str, str | float]]:
    feat_cols = [c for c in df.columns if c != "Outcome"]

    corr = df[feat_cols + ["Outcome"]].corr()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    out = FIG_DIR / "eda_correlation_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")

    pair_cols = [c for c in ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"] if c in df.columns]
    pair_df = df[pair_cols + ["Outcome"]].copy()
    if len(pair_df) > 250:
        pair_df = pair_df.sample(n=250, random_state=SEED)
    g = sns.pairplot(pair_df, hue="Outcome", diag_kind="hist", corner=True, plot_kws={"s": 12, "alpha": 0.6})
    g.fig.suptitle("Pairplot (subset) for EDA", y=1.01)
    out = FIG_DIR / "eda_pairplot.png"
    g.fig.savefig(out, dpi=150)
    plt.close(g.fig)
    print(f"[tabular] saved {out}")

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111)
    sns.boxplot(data=df[feat_cols], orient="h", ax=ax, fliersize=2)
    ax.set_title("Feature dispersion and outlier profile")
    ax.set_xlabel("Raw feature value")
    fig.tight_layout()
    out = FIG_DIR / "eda_outlier_boxplots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")

    top_pairs = _top_correlation_pairs(df, feat_cols, top_n=2)
    print("[tabular] Top two correlation pairs (excluding Outcome):", top_pairs)
    return top_pairs


def _plot_corr_vs_shap_alignment(
    df,
    feature_names: list[str],
    shap_rows: list[np.ndarray],
) -> None:
    corr_abs = df[feature_names + ["Outcome"]].corr()["Outcome"].drop("Outcome").abs()
    mean_abs_shap = np.mean(np.abs(np.vstack(shap_rows)), axis=0)

    order = np.argsort(mean_abs_shap)
    feats = [feature_names[i] for i in order]
    corr_vals = corr_abs.values[order]
    shap_vals = mean_abs_shap[order]

    y = np.arange(len(feats))
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(111)
    ax.barh(y - 0.18, corr_vals, height=0.34, label="|Corr(feature, Outcome)|", color="#457b9d")
    ax.barh(y + 0.18, shap_vals, height=0.34, label="Mean |SHAP| (3 samples)", color="#e76f51")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Importance proxy value")
    ax.set_title("Correlation-Attribution alignment")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = FIG_DIR / "correlation_vs_shap_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")


def _plot_shap_force_plots(
    expected_value: float,
    shap_rows: list[np.ndarray],
    samples: list[np.ndarray],
    sample_idxs: np.ndarray,
    feature_names: list[str],
) -> None:
    for plot_i, (row, x, test_idx) in enumerate(zip(shap_rows, samples, sample_idxs)):
        fig = plt.figure(figsize=(10, 1.8))
        shap.force_plot(
            expected_value,
            row,
            x,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.title(f"SHAP force plot - sample {plot_i} (test idx={int(test_idx)})", fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / f"shap_force_sample_{plot_i}.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)
        print(f"[tabular] saved {out}")


def _plot_grace_counterfactual(
    predict_fn,
    feature_names: list[str],
    x: np.ndarray,
    shap_row: np.ndarray,
    shap_explainer,
) -> dict[str, float | str]:
    top_idx = int(np.argmax(np.abs(shap_row)))
    top_feat = feature_names[top_idx]

    x_cf = x.copy()
    direction = -np.sign(shap_row[top_idx]) if shap_row[top_idx] != 0 else -1.0
    x_cf[top_idx] = x_cf[top_idx] + direction * 1.0

    p_before = float(predict_fn(x.reshape(1, -1))[0, 1])
    p_after = float(predict_fn(x_cf.reshape(1, -1))[0, 1])

    shap_before = _normalize_shap_output(
        np.asarray(shap_explainer.shap_values(x.reshape(1, -1), nsamples=160))
    )
    shap_after = _normalize_shap_output(
        np.asarray(shap_explainer.shap_values(x_cf.reshape(1, -1), nsamples=160))
    )
    delta = shap_after - shap_before

    order = np.argsort(np.abs(delta))
    feats = [feature_names[i] for i in order]
    delta_sorted = delta[order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["Original", "Counterfactual"], [p_before, p_after], color=["#1d3557", "#e63946"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Predicted probability shift")
    axes[0].set_ylabel("p(Outcome=1)")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].barh(feats, delta_sorted, color="#2a9d8f")
    axes[1].set_title("SHAP change (counterfactual - original)")
    axes[1].set_xlabel("Attribution delta")
    axes[1].grid(alpha=0.2, axis="x")
    fig.suptitle(f"GRACE-style counterfactual on feature: {top_feat}")
    fig.tight_layout()
    out = FIG_DIR / "grace_counterfactual_shap_shift.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")
    return {
        "top_feature": top_feat,
        "p_before": p_before,
        "p_after": p_after,
        "delta_p": float(p_after - p_before),
    }


def _overlay_heatmap(
    pil_image: Image.Image,
    heat: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    base = np.asarray(pil_image).astype(float) / 255.0
    heat_rgb = plt.get_cmap("jet")(_normalize_map(heat))[..., :3]
    overlay = (1.0 - alpha) * base + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def _saliency_entropy(map2d: np.ndarray) -> float:
    x = _normalize_map(map2d)
    p = x.ravel() + 1e-12
    p = p / p.sum()
    ent = -np.sum(p * np.log(p))
    ent = ent / np.log(len(p))
    return float(ent)


def _saliency_total_variation(map2d: np.ndarray) -> float:
    x = _normalize_map(map2d)
    dx = np.abs(x[:, 1:] - x[:, :-1]).mean()
    dy = np.abs(x[1:, :] - x[:-1, :]).mean()
    return float(dx + dy)


def _plot_smoothgrad_convergence(
    ks: list[int],
    entropies: list[float],
    tvs: list[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(ks, entropies, marker="o", color="#1d3557")
    axes[0].set_title("SmoothGrad entropy vs K")
    axes[0].set_xlabel("Sample count K")
    axes[0].set_ylabel("Normalized entropy")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ks, tvs, marker="o", color="#e63946")
    axes[1].set_title("SmoothGrad total variation vs K")
    axes[1].set_xlabel("Sample count K")
    axes[1].set_ylabel("Total variation")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "smoothgrad_convergence_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")


def generate_tabular_figures() -> dict[str, float]:
    print("[tabular] loading data...")
    df = load_diabetes(local_path=str(DATA_PATH), seed=SEED)
    X, y, _ = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, seed=SEED)
    feature_names = df.columns[:-1].tolist()
    top_correlation_pairs = _plot_eda_figures(df)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Outcome", data=df)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"[tabular] saved {(FIG_DIR / 'class_distribution.png')}")

    tr = to_loader(X_train, y_train, batch_size=64)
    va = to_loader(X_val, y_val, batch_size=256, shuffle=False)

    mlp = MLPClassifier()
    mlp, mlp_hist = train_model(
        mlp, tr, va, epochs=30, lr=1e-3, return_history=True
    )
    y_pred_mlp, probs_mlp = predict_binary(mlp, X_test)
    metrics_mlp = evaluate_preds(y_test, y_pred_mlp)
    print(
        "[tabular] MLP metrics:",
        {k: v for k, v in metrics_mlp.items() if k != "confusion_matrix"},
    )

    predict_fn = _predict_fn_factory(mlp)
    shap_explainer = shap.KernelExplainer(lambda z: predict_fn(z)[:, 1], X_train[:100])
    sample_idxs = np.array([0, 1, 2])
    sample_idxs = sample_idxs[sample_idxs < len(X_test)]
    agreement_rows: list[dict[str, float]] = []
    shap_rows: list[np.ndarray] = []
    sample_vectors: list[np.ndarray] = []

    for plot_i, test_idx in enumerate(sample_idxs):
        x = X_test[test_idx]
        lime_exp = lime_explain(predict_fn, X_train, x, feature_names)
        shap_vals = shap_explainer.shap_values(x.reshape(1, -1), nsamples=200)
        shap_row = _normalize_shap_output(shap_vals)
        shap_rows.append(shap_row.copy())
        sample_vectors.append(x.copy())

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].barh(feature_names, shap_row, color="#2a9d8f")
        axes[0].set_title("SHAP (per-feature)")
        lime_items = lime_exp.as_list()
        lime_vec = _lime_vector(lime_items, feature_names)
        axes[1].barh(
            [name for name, _ in lime_items],
            [weight for _, weight in lime_items],
            color="#e76f51",
        )
        axes[1].set_title("LIME (local)")
        p_pos = float(predict_fn(x.reshape(1, -1))[0, 1])
        fig.suptitle(f"Sample {plot_i} (test idx={test_idx}) | p(pos)={p_pos:.3f}")
        fig.tight_layout()
        out = FIG_DIR / f"lime_shap_compare_sample_{plot_i}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[tabular] saved {out}")

        corr = spearmanr(shap_row, lime_vec).correlation
        if corr is None or np.isnan(corr):
            corr = 0.0
        shap_top = np.argsort(np.abs(shap_row))[-3:]
        lime_top = np.argsort(np.abs(lime_vec))[-3:]
        overlap = len(set(shap_top).intersection(set(lime_top))) / 3.0
        agreement_rows.append(
            {"sample": float(plot_i), "spearman": float(corr), "top3_overlap": float(overlap)}
        )

    _plot_shap_force_plots(
        expected_value=float(np.asarray(shap_explainer.expected_value).reshape(-1)[0]),
        shap_rows=shap_rows,
        samples=sample_vectors,
        sample_idxs=sample_idxs,
        feature_names=feature_names,
    )
    _plot_corr_vs_shap_alignment(df, feature_names, shap_rows)
    grace_summary = _plot_grace_counterfactual(
        predict_fn=predict_fn,
        feature_names=feature_names,
        x=sample_vectors[0],
        shap_row=shap_rows[0],
        shap_explainer=shap_explainer,
    )

    nam = NAMClassifier(n_features=X.shape[1], hidden=32)
    nam, nam_hist = train_model(
        nam, tr, va, epochs=35, lr=5e-3, return_history=True
    )
    pred_nam, probs_nam = predict_binary(nam, X_test)
    metrics_nam = evaluate_preds(y_test, pred_nam)
    print(
        "[tabular] NAM metrics:",
        {k: v for k, v in metrics_nam.items() if k != "confusion_matrix"},
    )

    med = np.median(X_train, axis=0)
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for i, ax in enumerate(axs.ravel()):
        xs = np.linspace(X_train[:, i].min(), X_train[:, i].max(), 200)
        inputs = np.tile(med, (200, 1))
        inputs[:, i] = xs
        with torch.no_grad():
            out = nam(torch.from_numpy(inputs).float()).cpu().numpy()
        ax.plot(xs, out, color="#264653")
        ax.set_title(feature_names[i])
    fig.tight_layout()
    out = FIG_DIR / "nam_feature_functions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[tabular] saved {out}")

    _plot_training_curves(mlp_hist, nam_hist)
    _plot_confusion_matrices(
        metrics_mlp["confusion_matrix"], metrics_nam["confusion_matrix"]
    )
    roc_auc_mlp, roc_auc_nam, ap_mlp, ap_nam = _plot_roc_pr_curves(
        y_test, probs_mlp, probs_nam
    )
    brier_mlp, brier_nam = _plot_calibration(y_test, probs_mlp, probs_nam)
    threshold_summary = _plot_threshold_sensitivity(y_test, probs_mlp, probs_nam)
    _plot_agreement(agreement_rows)
    perm_imp_mlp = _permutation_importance(mlp, X_test, y_test, n_repeats=8, seed=SEED)
    perm_imp_nam = _permutation_importance(nam, X_test, y_test, n_repeats=8, seed=SEED + 1)
    _plot_permutation_importance(feature_names, perm_imp_mlp, perm_imp_nam)

    summary = {
        "mlp": {k: float(v) for k, v in metrics_mlp.items() if k != "confusion_matrix"},
        "nam": {k: float(v) for k, v in metrics_nam.items() if k != "confusion_matrix"},
        "mlp_confusion_matrix": metrics_mlp["confusion_matrix"].tolist(),
        "nam_confusion_matrix": metrics_nam["confusion_matrix"].tolist(),
        "roc_auc": {"mlp": float(roc_auc_mlp), "nam": float(roc_auc_nam)},
        "avg_precision": {"mlp": float(ap_mlp), "nam": float(ap_nam)},
        "brier": {"mlp": float(brier_mlp), "nam": float(brier_nam)},
        "lime_shap_agreement": agreement_rows,
        "threshold_sensitivity": threshold_summary,
        "permutation_importance": {
            "feature_names": feature_names,
            "mlp_accuracy_drop": perm_imp_mlp.tolist(),
            "nam_accuracy_drop": perm_imp_nam.tolist(),
        },
        "grace_counterfactual": grace_summary,
        "top_correlation_pairs": top_correlation_pairs,
    }
    return summary


def _imagenet_categories() -> list[str]:
    weights_enum = getattr(tv_models, "VGG16_Weights", None)
    if weights_enum is not None:
        return list(weights_enum.IMAGENET1K_V1.meta.get("categories", []))
    return [f"class_{i}" for i in range(1000)]


def _denorm_to_rgb01(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    x = t * std + mean
    x = x.clamp(0.0, 1.0)
    return x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()


def _predict_top(model: torch.nn.Module, input_tensor: torch.Tensor, categories: list[str]) -> tuple[int, str, float]:
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
    name = categories[idx] if idx < len(categories) else f"class_{idx}"
    return idx, name, conf


def _load_images_from_paths(paths: list[str]) -> list[Image.Image]:
    """Load PIL Images from file paths or URLs (e.g. for HW2 six-image set)."""
    imgs: list[Image.Image] = []
    for p in paths:
        p = p.strip()
        if not p:
            continue
        try:
            if p.startswith("http://") or p.startswith("https://"):
                with urlopen(p, timeout=15) as resp:
                    img = Image.open(resp).convert("RGB")
            else:
                img = Image.open(p).convert("RGB")
            imgs.append(img)
        except Exception as exc:
            print(f"[vision] Skipping {p[:60]}...: {exc}")
    return imgs


def _build_demo_images() -> list[Image.Image]:
    imgs: list[Image.Image] = []

    img0 = Image.new("RGB", (224, 224), color=(160, 120, 90))
    d0 = ImageDraw.Draw(img0)
    d0.ellipse((40, 40, 185, 185), fill=(220, 180, 130), outline=(90, 60, 30), width=4)
    imgs.append(img0)

    img1 = Image.new("RGB", (224, 224), color=(90, 130, 200))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle((20, 120, 204, 210), fill=(40, 80, 60))
    d1.polygon([(40, 120), (112, 60), (184, 120)], fill=(180, 180, 180))
    imgs.append(img1)

    img2 = Image.new("RGB", (224, 224), color=(220, 210, 120))
    d2 = ImageDraw.Draw(img2)
    d2.ellipse((70, 50, 150, 200), fill=(250, 220, 80), outline=(200, 160, 40), width=3)
    imgs.append(img2)

    img3 = Image.new("RGB", (224, 224), color=(180, 200, 230))
    d3 = ImageDraw.Draw(img3)
    for i in range(0, 224, 16):
        d3.line((0, i, 223, i), fill=(120, 140, 170), width=2)
    d3.rectangle((60, 70, 170, 170), outline=(20, 40, 60), width=5)
    imgs.append(img3)

    img4 = Image.new("RGB", (224, 224), color=(120, 80, 140))
    d4 = ImageDraw.Draw(img4)
    d4.polygon([(30, 180), (110, 40), (194, 180)], fill=(230, 120, 80), outline=(60, 20, 20), width=3)
    imgs.append(img4)

    img5 = Image.new("RGB", (224, 224), color=(200, 200, 200))
    d5 = ImageDraw.Draw(img5)
    d5.ellipse((20, 20, 200, 200), outline=(30, 30, 30), width=5)
    d5.line((20, 112, 200, 112), fill=(30, 30, 30), width=4)
    d5.line((112, 20, 112, 200), fill=(30, 30, 30), width=4)
    imgs.append(img5)

    return imgs


def _make_candidate_image(rng: np.random.Generator, idx: int) -> Image.Image:
    base = tuple(int(c) for c in rng.integers(20, 235, size=3))
    img = Image.new("RGB", (224, 224), color=base)
    draw = ImageDraw.Draw(img)

    # Blend geometric primitives and texture so VGG receives varied structures.
    for _ in range(5 + (idx % 4)):
        shape_type = int(rng.integers(0, 3))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        x1, y1 = [int(v) for v in rng.integers(0, 160, size=2)]
        x2, y2 = [int(v) for v in rng.integers(64, 224, size=2)]
        if shape_type == 0:
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=color, width=3)
        elif shape_type == 1:
            draw.ellipse((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=color, width=3)
        else:
            xm, ym = int(rng.integers(0, 224)), int(rng.integers(0, 224))
            draw.polygon([(x1, y1), (x2, y2), (xm, ym)], outline=color)

    for k in range(0, 224, int(rng.integers(10, 28))):
        line_color = tuple(int(c) for c in rng.integers(30, 220, size=3))
        if idx % 2 == 0:
            draw.line((0, k, 223, (k + int(rng.integers(-20, 20))) % 224), fill=line_color, width=1)
        else:
            draw.line((k, 0, (k + int(rng.integers(-20, 20))) % 224, 223), fill=line_color, width=1)
    return img


def _plot_vgg16_image_set(
    model: torch.nn.Module,
    categories: list[str],
    custom_images: list[Image.Image] | None = None,
) -> tuple[list[dict[str, float | str]], Image.Image]:
    if custom_images and len(custom_images) >= 6:
        candidates = custom_images
    else:
        seed_imgs = _build_demo_images()
        rng = np.random.default_rng(SEED)
        candidates = (
            (custom_images or []) + seed_imgs + [_make_candidate_image(rng, i) for i in range(30)]
        )

    selected_imgs: list[Image.Image] = []
    rows: list[dict[str, float | str]] = []
    used_classes: set[int] = set()
    fallback_rows: list[tuple[Image.Image, dict[str, float | str]]] = []

    for cand_idx, img in enumerate(candidates):
        inp = preprocess_image(img)
        idx, name, conf = _predict_top(model, inp, categories)
        row = {"image_id": cand_idx, "pred_idx": idx, "pred_name": name, "confidence": conf}
        if idx not in used_classes and len(selected_imgs) < 6:
            used_classes.add(idx)
            selected_imgs.append(img)
            rows.append(row)
        else:
            fallback_rows.append((img, row))
        if len(selected_imgs) == 6:
            break

    # If unique-class selection is insufficient, fill deterministically by confidence.
    if len(selected_imgs) < 6:
        fallback_rows.sort(key=lambda item: float(item[1]["confidence"]), reverse=True)
        for img, row in fallback_rows:
            selected_imgs.append(img)
            rows.append(row)
            if len(selected_imgs) == 6:
                break

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, (img, row, ax) in enumerate(zip(selected_imgs, rows, axes.ravel())):
        ax.imshow(img)
        ax.set_title(f"Img {i}: {row['pred_name']}\nconf={float(row['confidence']):.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Six analyzed images with VGG16 predictions")
    fig.tight_layout()
    out = FIG_DIR / "vgg16_six_image_predictions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")
    return rows, selected_imgs[0]


def _plot_adversarial_comparison(
    model: torch.nn.Module,
    image: Image.Image,
    categories: list[str],
    epsilon: float = 0.12,
) -> dict[str, float | int | str]:
    x = preprocess_image(image).detach().clone().requires_grad_(True)
    logits = model(x)
    c0 = int(torch.argmax(logits, dim=1).item())
    loss = F.cross_entropy(logits, torch.tensor([c0]))
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x.grad.detach().sign()
    x_adv = (x + epsilon * grad_sign).detach()

    c_adv, name_adv, conf_adv = _predict_top(model, x_adv, categories)
    _, name_orig, conf_orig = _predict_top(model, x.detach(), categories)

    cam = GradCAM(model, model.features[28])
    cam_orig = cam(x.detach(), class_idx=c0)
    cam_adv = cam(x_adv, class_idx=c0)
    cam.close()

    rgb_orig = _denorm_to_rgb01(x.detach())
    rgb_adv = _denorm_to_rgb01(x_adv)
    pert = np.abs(rgb_adv - rgb_orig).mean(axis=2)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(rgb_orig)
    axes[0].set_title(f"Original\n{name_orig} ({conf_orig:.3f})", fontsize=9)
    axes[1].imshow(pert, cmap="magma")
    axes[1].set_title(f"|Perturbation| (eps={epsilon})", fontsize=9)
    axes[2].imshow(cam_orig, cmap="jet")
    axes[2].set_title(f"Grad-CAM @orig class\n{name_orig}", fontsize=9)
    axes[3].imshow(cam_adv, cmap="jet")
    axes[3].set_title(f"Adversarial prediction\n{name_adv} ({conf_adv:.3f})", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "adversarial_fgsm_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")
    return {
        "epsilon": float(epsilon),
        "orig_class_idx": int(c0),
        "orig_class_name": name_orig,
        "adv_class_idx": int(c_adv),
        "adv_class_name": name_adv,
        "changed": bool(c_adv != c0),
    }


def _plot_feature_visualization_hen(model: torch.nn.Module, categories: list[str]) -> dict[str, float | int]:
    hen_idx = categories.index("hen") if "hen" in categories else 8

    torch.manual_seed(SEED)
    img_raw = activation_maximization(
        model,
        hen_idx,
        steps=24,
        lr=0.7,
        tv_weight=0.0,
        device="cpu",
        use_random_shifts=False,
        image_size=128,
    )
    regs = []
    for i in range(5):
        torch.manual_seed(SEED + i + 1)
        regs.append(
            activation_maximization(
                model,
                hen_idx,
                steps=32,
                lr=0.7,
                tv_weight=2e-5,
                device="cpu",
                use_random_shifts=True,
                shift_max=12,
                image_size=128,
            )
        )

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    axes = axes.ravel()
    axes[0].imshow(np.transpose(img_raw, (1, 2, 0)))
    axes[0].set_title("Initial (no TV, no shifts)", fontsize=9)
    axes[0].axis("off")
    for i, img in enumerate(regs, start=1):
        axes[i].imshow(np.transpose(img, (1, 2, 0)))
        axes[i].set_title(f"Regularized #{i}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle('Activation maximization for class "hen"')
    fig.tight_layout()
    out = FIG_DIR / "feature_visualization_hen.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")
    return {"hen_class_idx": int(hen_idx), "n_regularized_images": 5}


def generate_vision_figures(
    custom_image_paths: list[str] | None = None,
) -> dict[str, float]:
    """Generate all vision figures. Optionally use custom images for the six-image set.

    Custom images can be provided via:
      - custom_image_paths: list of file paths or URLs (at least 6 for full replacement)
      - Environment HW2_VISION_IMAGES: comma-separated paths or URLs
    If fewer than 6 custom images are given, fallback synthetic/demo images are used to fill.
    """
    print("[vision] building model...")
    model = get_vgg16(device="cpu", prefer_pretrained=True)
    categories = _imagenet_categories()

    custom_images: list[Image.Image] | None = None
    if custom_image_paths:
        custom_images = _load_images_from_paths(custom_image_paths)
    else:
        env_paths = os.environ.get("HW2_VISION_IMAGES")
        if env_paths:
            custom_images = _load_images_from_paths([p.strip() for p in env_paths.split(",")])
    six_rows, attack_image = _plot_vgg16_image_set(model, categories, custom_images=custom_images)

    img_gradcam = Image.new("RGB", (224, 224), color=(100, 140, 200))
    inp_gradcam = preprocess_image(img_gradcam)
    cam = GradCAM(model, model.features[28])
    heat = cam(inp_gradcam)
    cam.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(heat, cmap="jet")
    plt.title("Grad-CAM heatmap (demo)")
    plt.axis("off")
    out = FIG_DIR / "gradcam_demo.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[vision] saved {out}")

    overlay = _overlay_heatmap(img_gradcam, heat)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_gradcam)
    axes[0].set_title("Input image")
    axes[1].imshow(heat, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "gradcam_overlay_demo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    img_guided = Image.new("RGB", (224, 224), color=(200, 160, 120))
    inp_guided = preprocess_image(img_guided)
    gb = GuidedBackprop(model)
    gb_grad = gb.generate(inp_guided)
    cam2 = GradCAM(model, model.features[28])
    heat2 = cam2(inp_guided)
    cam2.close()
    guided_gradcam = np.abs(gb_grad).max(axis=0) * heat2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_guided)
    axes[0].set_title("Image")
    axes[1].imshow(heat2, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(guided_gradcam, cmap="inferno")
    axes[2].set_title("Guided Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "guided_gradcam_example.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    img_sg = Image.new("RGB", (224, 224), color=(120, 130, 140))
    inp_sg = preprocess_image(img_sg)
    sg = smoothgrad(model, inp_sg, n_samples=20, stdev=0.12)
    sg_vis = np.abs(sg)
    sg_vis = (sg_vis - sg_vis.min()) / (sg_vis.max() - sg_vis.min() + 1e-8)
    gb2 = GuidedBackprop(model)
    gb_grad2 = gb2.generate(inp_sg)
    cam3 = GradCAM(model, model.features[28])
    heat3 = cam3(inp_sg)
    cam3.close()
    guided_gradcam2 = np.abs(gb_grad2).max(axis=0) * heat3

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.transpose(sg_vis, (1, 2, 0)))
    axes[0].set_title("SmoothGrad (avg grad)")
    axes[1].imshow(np.abs(gb_grad2).max(axis=0), cmap="gray")
    axes[1].set_title("Guided BP (abs)")
    axes[2].imshow(guided_gradcam2, cmap="inferno")
    axes[2].set_title("Guided Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "smoothgrad_guided_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    sg_guided_bp = smoothgrad_guided_backprop(
        model, inp_sg, n_samples=25, stdev=0.12
    )
    sg_guided_bp_vis = _normalize_map(np.abs(sg_guided_bp).max(axis=0))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(sg_guided_bp_vis, cmap="inferno")
    ax.set_title("SmoothGrad + Guided Backpropagation")
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "smoothgrad_guided_backprop.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    sg_guided_gcam = smoothgrad_guided_gradcam(
        model, model.features[28], inp_sg, n_samples=25, stdev=0.12
    )
    sg_guided_gcam_vis = _normalize_map(sg_guided_gcam)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(sg_guided_gcam_vis, cmap="inferno")
    ax.set_title("SmoothGrad + Guided Grad-CAM")
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "smoothgrad_guided_gradcam.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    smoothgrad_counts = [5, 10, 20, 50]
    sweep_maps = []
    for n_samples in smoothgrad_counts:
        sg_map = smoothgrad(model, inp_sg, n_samples=n_samples, stdev=0.12)
        sg_map = _normalize_map(np.abs(sg_map).max(axis=0))
        sweep_maps.append(sg_map)

    fig, axes = plt.subplots(1, len(sweep_maps), figsize=(14, 3.6))
    for ax, n_samples, sg_map in zip(axes, smoothgrad_counts, sweep_maps):
        ax.imshow(sg_map, cmap="inferno")
        ax.set_title(f"SmoothGrad K={n_samples}")
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "smoothgrad_sample_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    # Similarity and smoothness trends as K grows.
    flat_maps = [m.ravel() for m in sweep_maps]
    cos_5_20 = float(
        np.dot(flat_maps[0], flat_maps[2])
        / (np.linalg.norm(flat_maps[0]) * np.linalg.norm(flat_maps[2]) + 1e-8)
    )
    cos_20_50 = float(
        np.dot(flat_maps[2], flat_maps[3])
        / (np.linalg.norm(flat_maps[2]) * np.linalg.norm(flat_maps[3]) + 1e-8)
    )
    cos_5_50 = float(
        np.dot(flat_maps[0], flat_maps[3])
        / (np.linalg.norm(flat_maps[0]) * np.linalg.norm(flat_maps[3]) + 1e-8)
    )
    entropies = [_saliency_entropy(m) for m in sweep_maps]
    tvs = [_saliency_total_variation(m) for m in sweep_maps]
    _plot_smoothgrad_convergence(smoothgrad_counts, entropies, tvs)
    adv_summary = _plot_adversarial_comparison(model, attack_image, categories, epsilon=0.12)
    featvis_summary = _plot_feature_visualization_hen(model, categories)
    return {
        "smoothgrad_cosine_5_20": cos_5_20,
        "smoothgrad_cosine_20_50": cos_20_50,
        "smoothgrad_cosine_5_50": cos_5_50,
        "smoothgrad_ks": smoothgrad_counts,
        "smoothgrad_entropy": entropies,
        "smoothgrad_total_variation": tvs,
        "six_image_predictions": six_rows,
        "adversarial_demo": adv_summary,
        "feature_visualization": featvis_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HW2 report figures (tabular + vision).")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Paths or URLs for the six-image vision set (at least 6 for full set). Overrides HW2_VISION_IMAGES.",
    )
    args = parser.parse_args()
    _set_seed()
    _ensure_dirs()
    tabular_summary = generate_tabular_figures()
    vision_summary = generate_vision_figures(custom_image_paths=args.images)
    combined_summary = {"tabular": tabular_summary, "vision": vision_summary}
    METRICS_JSON.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print(f"[done] saved summary metrics at {METRICS_JSON}")
    print("[done] report figures generated in", FIG_DIR)


if __name__ == "__main__":
    main()
