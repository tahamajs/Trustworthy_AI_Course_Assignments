# HW2/code — Detailed developer guide

This document explains the code-level entry points, how to run experiments reproducibly, and where outputs are saved.

---

## Primary scripts & purpose

- `tabular.py` — training pipeline for Pima diabetes dataset (downloads fallback / synthetic if offline).
- `models.py` — `MLPClassifier` and `NAMClassifier` used in experiments.
- `interpretability.py` — LIME and SHAP wrappers for per-sample explanations.
- `vision.py` — Grad-CAM, Guided Backprop, SmoothGrad, and activation maximization utilities.
- `generate_report_plots.py` — convenience script to create PNGs used in the LaTeX report.

---

## How to reproduce the tabular baseline

1. Create environment and install (see `requirements.txt`).
2. Run:

   ```bash
   cd HomeWorks/HW2/code
   python tabular.py
   ```

   - `tabular.py` uses local `diabetes.csv` if present, otherwise attempts to download; it falls back to a deterministic synthetic dataset when offline.

3. Generated metrics are printed to stdout; use `generate_report_plots.py` to export figures.

---

## Vision experiments & explanations

- The notebook `../notebooks/HW2_solution.ipynb` demonstrates how to load models, compute Grad-CAM maps, combine Guided Backprop + Grad-CAM, and run activation maximization.
- `vision.py` contains reusable functions so you can call them programmatically from other scripts/notebooks.

---

## Notes on interpretability tools

- LIME/SHAP can be slow for larger datasets; use `n_samples`/`nsamples` parameters to limit runtime.
- SHAP `KernelExplainer` is model-agnostic but computationally expensive — use a small background subset for demos.

---

## Expected outputs

- `report/figures/` — standard plots used in the assignment report (saved by `generate_report_plots.py`).
- Notebook cells include figure exports and the code used to generate them.

---

## Troubleshooting & tips

- No internet: tabular dataset fallback ensures the notebook is runnable offline.
- Pretrained weights unavailable: `vision.py` tries a non-pretrained fallback; visualizations still work but results differ.

---

If you want, I can add explicit CLI flags to `tabular.py` and `vision.py` for dataset path, device selection, and logging to make experiments even more reproducible.
