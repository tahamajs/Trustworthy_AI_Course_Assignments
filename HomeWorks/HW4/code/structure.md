# HW4 — Code structure

This file documents the folder/file layout and quick usage for Homework 4 (Trustworthy AI).

```
HomeWorks/HW4/
├─ code/
│  ├─ data.csv                 # input dataset used in Q3 (already present)
│  ├─ requirements.txt         # Python dependencies for running HW4 code
│  ├─ README.md                # this folder's quick usage guide
│  ├─ structure.md             # (this file)
│  ├─ fairness.py              # Q3: metrics, baseline model, mitigation (promotion/demotion)
│  ├─ privacy.py               # Q2: Laplace mechanism helpers + examples
│  ├─ neural_cleanse.py        # Q1: trigger reconstruction + MAD detector + unlearning scaffold
│  ├─ notebook.ipynb           # end-to-end walkthrough (Q1, Q2, Q3 examples)
│  └─ tests/                   # unit tests (pytest)
│     ├─ test_fairness.py
│     ├─ test_privacy.py
│     └─ test_neural_cleanse.py
├─ description/
│  └─ README.md                # assignment statement and questions
└─ report/
   └─ assignment_template.tex  # report template
```

Quick run examples
- Install dependencies: `pip install -r code/requirements.txt`
- Open notebook walkthrough: `jupyter notebook code/notebook.ipynb`
- Run unit tests: `pytest code/tests`

Notes
- For Q1 (Neural Cleanse) put real attacked model weights in `code/model_weights/` (loader accepts a path) — the implementation contains a demo mode so you can run reconstruction without external checkpoints.
- `fairness.py` uses `data.csv` (numeric features) and provides utilities to reproduce the assignment tasks and compare Base vs Fair models.

If you want, I can also run the tests and add example output/figures to the `report/` template. Tell me which action you prefer next. ✅