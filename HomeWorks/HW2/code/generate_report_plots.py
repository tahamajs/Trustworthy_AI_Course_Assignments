"""Generate all core HW2 report plots (tabular + vision) in one command.

Usage:
    python code/generate_report_plots.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

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
from vision import GradCAM, GuidedBackprop, get_vgg16, preprocess_image, smoothgrad


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


def _overlay_heatmap(
    pil_image: Image.Image,
    heat: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    base = np.asarray(pil_image).astype(float) / 255.0
    heat_rgb = plt.get_cmap("jet")(_normalize_map(heat))[..., :3]
    overlay = (1.0 - alpha) * base + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def generate_tabular_figures() -> dict[str, float]:
    print("[tabular] loading data...")
    df = load_diabetes(local_path=str(DATA_PATH), seed=SEED)
    X, y, _ = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, seed=SEED)
    feature_names = df.columns[:-1].tolist()

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
    sample_idxs = np.array([0, 1, 2])
    sample_idxs = sample_idxs[sample_idxs < len(X_test)]
    agreement_rows: list[dict[str, float]] = []

    for plot_i, test_idx in enumerate(sample_idxs):
        x = X_test[test_idx]
        lime_exp = lime_explain(predict_fn, X_train, x, feature_names)
        shap_vals = shap_explain(
            lambda z: predict_fn(z)[:, 1], X_train, x.reshape(1, -1)
        )
        shap_row = _normalize_shap_output(shap_vals)

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
    _plot_agreement(agreement_rows)

    summary = {
        "mlp": {k: float(v) for k, v in metrics_mlp.items() if k != "confusion_matrix"},
        "nam": {k: float(v) for k, v in metrics_nam.items() if k != "confusion_matrix"},
        "mlp_confusion_matrix": metrics_mlp["confusion_matrix"].tolist(),
        "nam_confusion_matrix": metrics_nam["confusion_matrix"].tolist(),
        "roc_auc": {"mlp": float(roc_auc_mlp), "nam": float(roc_auc_nam)},
        "avg_precision": {"mlp": float(ap_mlp), "nam": float(ap_nam)},
        "brier": {"mlp": float(brier_mlp), "nam": float(brier_nam)},
        "lime_shap_agreement": agreement_rows,
    }
    return summary


def generate_vision_figures() -> dict[str, float]:
    print("[vision] building model...")
    model = get_vgg16(device="cpu", prefer_pretrained=False)

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

    smoothgrad_counts = [5, 20, 50]
    sweep_maps = []
    for n_samples in smoothgrad_counts:
        sg_map = smoothgrad(model, inp_sg, n_samples=n_samples, stdev=0.12)
        sg_map = _normalize_map(np.abs(sg_map).max(axis=0))
        sweep_maps.append(sg_map)

    fig, axes = plt.subplots(1, len(sweep_maps), figsize=(12, 3.6))
    for ax, n_samples, sg_map in zip(axes, smoothgrad_counts, sweep_maps):
        ax.imshow(sg_map, cmap="inferno")
        ax.set_title(f"SmoothGrad K={n_samples}")
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "smoothgrad_sample_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[vision] saved {out}")

    # Similarity increases as K grows if estimator stabilizes.
    flat5 = sweep_maps[0].ravel()
    flat20 = sweep_maps[1].ravel()
    flat50 = sweep_maps[2].ravel()
    cos_5_20 = float(np.dot(flat5, flat20) / (np.linalg.norm(flat5) * np.linalg.norm(flat20) + 1e-8))
    cos_20_50 = float(np.dot(flat20, flat50) / (np.linalg.norm(flat20) * np.linalg.norm(flat50) + 1e-8))
    cos_5_50 = float(np.dot(flat5, flat50) / (np.linalg.norm(flat5) * np.linalg.norm(flat50) + 1e-8))
    return {
        "smoothgrad_cosine_5_20": cos_5_20,
        "smoothgrad_cosine_20_50": cos_20_50,
        "smoothgrad_cosine_5_50": cos_5_50,
    }


def main() -> None:
    _set_seed()
    _ensure_dirs()
    tabular_summary = generate_tabular_figures()
    vision_summary = generate_vision_figures()
    combined_summary = {"tabular": tabular_summary, "vision": vision_summary}
    METRICS_JSON.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print(f"[done] saved summary metrics at {METRICS_JSON}")
    print("[done] report figures generated in", FIG_DIR)


if __name__ == "__main__":
    main()
