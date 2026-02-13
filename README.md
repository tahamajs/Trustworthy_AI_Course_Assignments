# Trustworthy AI — Course Assignments 📚🔬

Comprehensive, runnable implementations for four Trustworthy AI homeworks (models, experiments, and report templates). Each homework is self-contained so you can reproduce experiments, generate figures, and export report-ready PDFs.

---

## Contents (quick)

- `HomeWorks/HW1` — Image classification: robustness & adversarial training (ResNet18, FGSM/PGD, UMAP).
- `HomeWorks/HW2` — Interpretability: tabular & vision explanations (LIME, SHAP, Grad-CAM, GuidedBackprop).
- `HomeWorks/HW3` — Causal modeling & algorithmic recourse (SCMs, differentiable & linear recourse).
- `HomeWorks/HW4` — Security, privacy & fairness (Neural Cleanse, Laplace utilities, fairness metrics).
- `template` — LaTeX templates used for assignment reports.

---

## Table of contents

1. Quick start
2. Repository layout
3. Per-homework quick commands
4. Data & offline fallbacks
5. Building reports (PDF)
6. Tests & CI
7. Contributing
8. License & citation
9. Contact / maintainers

---

## 1) Quick start — local setup ✅

Prerequisites: Python 3.8+ (recommended), Git, LaTeX (for PDF reports). GPU optional for vision training.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install requirements (example — install per-homework when working inside it):

```bash
pip install -r HomeWorks/HW1/code/requirements.txt
pip install -r HomeWorks/HW2/code/requirements.txt
pip install -r HomeWorks/HW3/code/q5_codes/requirements.txt
pip install -r HomeWorks/HW4/code/requirements.txt
```

Tip: use `pip` inside the activated venv or use a Conda environment if preferred.

---

## 2) Repository structure (high level)

- `HomeWorks/` — four homework folders (each with `code/`, `description/`, `report/`, `notebooks/` where applicable).
  - `HW1/code/` — training, attacks, datasets, evaluation utilities.
  - `HW2/code/` — tabular/vision models and interpretability tools.
  - `HW3/code/` — causal SCMs, recourse algorithms and evaluation.
  - `HW4/code/` — security/privacy/fairness scripts and small tests.
- `results/` — example outputs and saved numpy results used in reports.
- `template/` — LaTeX templates for report and assignment.

Each `code/README.md` contains method-level docs and exact CLI flags — see those files for detailed options.

---

## 3) Per-homework quick commands (examples) 💡

HW1 (image classification / robustness)

```bash
cd HomeWorks/HW1/code
python train.py --dataset svhn --epochs 80 --batch-size 128 --optimizer sgd --save-dir checkpoints/svhn_baseline
python eval.py  --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap
```

HW2 (interpretability — tabular & vision)

```bash
cd HomeWorks/HW2/code
python tabular.py                # run tabular experiments on diabetes dataset (local fallback)
python generate_report_plots.py  # generate figures used in the report
# notebook workflow
jupyter lab HomeWorks/HW2/notebooks/HW2_solution.ipynb
```

HW3 (causal recourse)

```bash
cd HomeWorks/HW3/code/q5_codes
python main.py --seed 0
# full notebook: HomeWorks/HW3/code/HW3_complete_assignment.ipynb
```

HW4 (security / privacy / fairness)

```bash
cd HomeWorks/HW4/code
pip install -r requirements.txt
pytest tests            # run unit tests included for HW4
python generate_report_figs.py
```

General: many experiments provide `--save-dir` for checkpoints and `run_report_pipeline.py` helpers for quick artifact generation.

---

## 4) Data & offline/demo fallbacks 🔁

- Public datasets used: `CIFAR10`, `SVHN`, `MNIST`, Pima Diabetes CSV (local copies included where required).
- Several scripts include an **offline fallback** (synthetic / FakeData) so reports and notebook cells can run on machines without internet or large dataset downloads.
- Dataset files (when present) live under `HomeWorks/*/code/data` or `HomeWorks/*/dataset`.

If you need a dataset mirror added to the repo, tell me which one and I can add download helpers.

---

## 5) Building reports (PDF) — LaTeX

Each homework has a `report/Makefile`. From the homework root run:

```bash
cd HomeWorks/HW1/report
make           # builds assignment_template.pdf
```

If `latexmk` is installed, the Makefile will use it for a clean build; otherwise `pdflatex` + `bibtex` fallback is used.

---

## 6) Tests & CI 🔍

- HW4 includes `pytest`-based unit tests: `cd HomeWorks/HW4/code && pytest tests`.
- There is no top-level CI configured — I can add GitHub Actions workflows (unit tests, lint, notebook checks) if you'd like.

---

## 7) Contributing 🤝

- Follow PEP8 for Python code and write concise docstrings for public functions.
- Suggested workflow: feature branch → PR with description and small, focused commits.
- Add tests for new behavior and update `README.md` / `code/README.md` when adding CLI flags.

If you want, I can add a `CONTRIBUTING.md` and a GitHub Actions CI pipeline.

---

## 8) License & citation 📄

- Current repository does not include an explicit `LICENSE` file. If this is intended for redistribution, I recommend adding an `MIT` or `Apache-2.0` license — tell me which and I will add it.
- To cite this work in reports, reference the repository and the course name.

---

## 9) Contact / maintainers

- Maintainer: repository owner (see git history).
- Need changes, a license added, CI, or an improved README section? Tell me what to add and I will update it.

---

## Acknowledgements & credits

Course assignments and code structure were developed for the Trustworthy AI coursework; several implementations reuse ideas from public libraries (PyTorch, torchvision, LIME, SHAP).

---

Enjoy exploring the experiments — tell me if you want me to add badges, CI, a `CONTRIBUTING.md`, or an explicit `LICENSE` file. ✅
