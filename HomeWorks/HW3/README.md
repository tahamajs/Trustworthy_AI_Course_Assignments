# HW3 — Causal Recourse (full-code + notebook)

This homework implements structural causal modeling, actionability constraints, and multiple recourse algorithms with evaluation on real datasets (Adult, COMPAS, Health, Loan).

---

## High-level summary
- Objective: produce actionable recourse (minimal-cost interventions) under *actionability constraints* and evaluate validity under causal models.
- Implemented recourse: linear constrained recourse, differentiable (gradient-based) recourse, and causal counterfactual evaluation via SCMs.

---

## Important locations
- `code/q5_codes/` — core implementation (training, SCMs, recourse algorithms, evaluation).
- `code/HW3_complete_assignment.ipynb` — full notebook covering Q1–Q6 and report artifact generation.
- `models/` (top-level) — pre-trained model weights included for quick evaluation.
- `results/` — CSV / .npy outputs from previous runs (validity/cost profiles).

---

## Quick start (run the Q5 pipeline)
```bash
cd HomeWorks/HW3/code/q5_codes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --seed 0
```
- Use `--dataset health|adult|loan|compas` where supported by the script flags.
- Save/load models are written under `models/` and evaluation artifacts under `results/`.

---

## Reproducibility & analysis
- Seed support: pass `--seed` to training/evaluation scripts.
- CVXPy: `LinearRecourse` prefers `cvxpy` for exact convex solves; a greedy fallback is implemented when `cvxpy` is unavailable.
- SCMs: `scm.py` contains both hand-coded and learned SCM variants; `SCM_Trainer` can fit structural equations from data.

---

## Evaluation metrics
- Recourse validity: proportion of previously-negative instances for which an action changes the classifier's decision under the SCM.
- Cost: L1-style or user-defined cost aggregated per instance and summarized by quantiles/means.
- Additional model metrics: MCC thresholding, AUROC, calibration where appropriate.

---

## Notebook → PDF
To export a report-quality PDF (XeLaTeX compatible):
```bash
cd HomeWorks/HW3
./scripts/export_notebook_pdf.sh
# Result: output/pdf/hw3_complete_assignment.pdf
```

---

## Next steps / extensions
- Add domain-specific constraints to the `data_utils` actionability sets.
- Replace learned SCMs with richer structural estimators and compare recourse validity.

If you want, I can expand the `code/q5_codes/README.md` with a CLI reference for every script and example reproduce commands for the provided `models/` and `results/` artifacts.
