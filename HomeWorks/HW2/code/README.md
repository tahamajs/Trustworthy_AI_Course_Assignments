HW2 — code

Files added by the helper:
- `models.py` — `MLPClassifier`, `NAMClassifier`
- `tabular.py` — data download, preprocessing, training + evaluation for Pima dataset
- `interpretability.py` — LIME / SHAP helper wrappers
- `vision.py` — Grad-CAM, Guided Backprop, SmoothGrad, activation maximization
- `requirements.txt` — packages required to run the notebook / scripts

Quickstart
1. Create a virtualenv and install requirements:
   python -m venv .venv && source .venv/bin/activate
   pip install -r HomeWorks/HW2/code/requirements.txt

2. Run the tabular demo (auto-downloads dataset):
   python HomeWorks/HW2/code/tabular.py

3. Use the notebook `HomeWorks/HW2/notebooks/HW2_solution.ipynb` for the full step-by-step report.

Notes
- The code is minimal but complete for demonstration and grading. Adjust training hyperparams in `tabular.py` if you want a longer run.
- LIME/SHAP examples are implemented in the notebook using the helper functions in `interpretability.py`.
