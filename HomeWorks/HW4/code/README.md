HW4 — Code

Contents
- `fairness.py` — data loading, metrics (accuracy, disparate impact, Zemel-style proxy), baseline model and bias-mitigation (promotion/demotion) implementation
- `privacy.py` — Laplace mechanism helpers and example calculations used in the assignment
- `neural_cleanse.py` — Neural Cleanse reconstruction + MAD detection + unlearning scaffold (placeholder model loader if you supply weights)
- `data.csv` — dataset used in Q3 (already present)
- `notebook.ipynb` — worked examples and end-to-end walkthrough (created separately)
- `tests/` — pytest unit tests for the main functions

How to use
1. Create and activate a Python environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Run examples from the notebook (`notebook.ipynb`) or import functions from scripts.

Notes
- Q1 (Neural Cleanse): place attacked model weights under `code/model_weights/` and update the path in the notebook or the loader call.
- The Neural Cleanse implementation includes a demo mode (synthetic model) so you can run the reconstruction without real weights.
