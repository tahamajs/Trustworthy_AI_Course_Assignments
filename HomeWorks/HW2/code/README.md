# HW2 Code Guide

This directory contains all implementations for HW2.

## Implemented methods

### 1) Tabular models (`models.py`)

- `MLPClassifier`
  - Architecture: `8 -> 100 -> 50 -> 50 -> 20 -> 1`
  - Uses BatchNorm, ReLU, Dropout.
  - Trained with `BCEWithLogitsLoss`.
- `NAMClassifier`
  - One small subnetwork per feature (`Linear -> ReLU -> Linear`).
  - Per-feature outputs are summed for interpretable additive effects.

### 2) Data + training pipeline (`tabular.py`)

- `load_diabetes(...)`: loads local CSV or downloads Pima dataset.
- `preprocess(...)`: standard scaling.
- `make_splits(...)`: stratified 70/10/20 train/val/test split.
- `train_model(...)`: Adam optimizer + validation selection.
- `evaluate_preds(...)`: accuracy, recall, F1, confusion matrix.

### 3) Tabular interpretability (`interpretability.py`)

- `lime_explain(...)`: local explanation per sample via LIME.
- `shap_explain(...)`: SHAP values with KernelExplainer for selected samples.

### 4) Vision interpretability (`vision.py`)

- `get_vgg16(...)`: pretrained VGG16 (or fallback without pretrained weights).
- `GradCAM`: class activation maps from feature gradients.
- `GuidedBackprop`: ReLU backward hooks for positive-gradient saliency.
- `smoothgrad(...)`: gradient averaging over noisy inputs.
- `activation_maximization(...)`: gradient ascent on input image with TV regularization and jitter.

## Quick run

```bash
cd HomeWorks/HW2/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tabular.py
python generate_report_plots.py
```

Notebook workflow:

- Open `HomeWorks/HW2/notebooks/HW2_solution.ipynb` for full experiments and report plots.

## Notes

- If internet is unavailable, `tabular.py` uses a synthetic fallback dataset.
- If internet is unavailable, `vision.py` falls back to non-pretrained VGG16 weights.
- Vision utilities are modular and can be imported independently into the notebook.
