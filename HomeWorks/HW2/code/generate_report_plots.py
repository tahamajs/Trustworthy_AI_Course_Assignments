"""Generate all core HW2 report plots (tabular + vision) in one command.

Usage:
    python code/generate_report_plots.py
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

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


def generate_tabular_figures() -> None:
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
    mlp = train_model(mlp, tr, va, epochs=30, lr=1e-3)
    y_pred, _ = predict_binary(mlp, X_test)
    metrics = evaluate_preds(y_test, y_pred)
    print(
        "[tabular] MLP metrics:",
        {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    )

    predict_fn = _predict_fn_factory(mlp)
    sample_idxs = np.array([0, 1, 2])
    sample_idxs = sample_idxs[sample_idxs < len(X_test)]

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

    nam = NAMClassifier(n_features=X.shape[1], hidden=32)
    nam = train_model(nam, tr, va, epochs=35, lr=5e-3)
    pred_nam, _ = predict_binary(nam, X_test)
    print(
        "[tabular] NAM metrics:",
        {k: v for k, v in evaluate_preds(y_test, pred_nam).items() if k != "confusion_matrix"},
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


def generate_vision_figures() -> None:
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


def main() -> None:
    _set_seed()
    _ensure_dirs()
    generate_tabular_figures()
    generate_vision_figures()
    print("[done] report figures generated in", FIG_DIR)


if __name__ == "__main__":
    main()
