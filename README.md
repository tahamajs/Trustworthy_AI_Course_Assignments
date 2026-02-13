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

## 3) Per-homework detailed reference (complete) 🔎

### HW1 — Image classification & robustness (HomeWorks/HW1/code) 🔧

- Purpose: train image classifiers, evaluate robustness (FGSM/PGD), and generate visualization artifacts (UMAP, sample grids).
- Key files:
  - `train.py`, `eval.py`, `attacks.py`, `datasets.py`, `losses.py`, `utils.py`, `runner.py`, `run_report_pipeline.py`.
- Main scripts & common flags:
  - Training baseline: `python train.py --dataset {svhn,cifar10,mnist} --epochs <E> --batch-size <B>`
  - Adversarial training: add `--adv-train --attack {fgsm,pgd} --epsilon <eps> --alpha <alpha> --iters <k>`
  - Evaluation / UMAP: `python eval.py --dataset svhn --checkpoint <path> --umap`
  - Quick artifacts: `python run_report_pipeline.py --epochs 3` (demo-safe) or `--full-run` for full experiments.
- Recommended hyperparameters (starting point):
  - Baseline: `--epochs 80`, `--batch-size 128`, `--optimizer sgd --lr 0.01 --momentum 0.9`.
  - PGD adv-train: `--epsilon 8/255 --alpha 2/255 --iters 7`.
- Outputs / where to look:
  - Checkpoints: `HomeWorks/HW1/code/checkpoints/<exp>/best.pth` and `last.pth`.
  - Figures: `<checkpoint>.umap.png`, `<checkpoint>.grid.png`, `training_curves.png`.
  - Logs/history: `training_history.csv` or `training_history.json` under `--save-dir`.
- Reproduce report figures: run `python run_report_pipeline.py --full-run` (long) or demo `--epochs 3` (fast).
- Notes & troubleshooting:
  - Datasets stored in `HomeWorks/HW1/code/data/` or fallback to FakeData when offline.
  - Reduce `--batch-size` to avoid CUDA OOM; use CPU mode for small demo runs.
  - For deterministic runs, set seeds via `utils.set_seed(...)` (used internally by runner scripts).

---

### HW2 — Interpretability (HomeWorks/HW2/code) 🧭

- Purpose: train tabular and simple vision models and demonstrate interpretability techniques (LIME, SHAP, Grad-CAM, Guided Backprop, SmoothGrad, activation maximization).
- Key files:
  - `tabular.py`, `models.py`, `interpretability.py`, `vision.py`, `generate_report_plots.py`, `notebooks/HW2_solution.ipynb`.
- Tabular workflow:
  - `python tabular.py` — loads `diabetes.csv` (local or remote), preprocesses, trains `MLPClassifier` / `NAMClassifier`, prints metrics.
  - Internals: `load_diabetes()`, `preprocess()`, `make_splits()`, `train_model()`.
- Vision interpretability:
  - Utilities: `get_vgg16()`, `GradCAM`, `GuidedBackprop`, `smoothgrad()`, `activation_maximization()` in `vision.py`.
  - Example: generate Grad-CAM heatmap for an image using `vision.preprocess_image()` + `GradCAM(model, target_layer)(tensor)`.
- Notebook: `HomeWorks/HW2/notebooks/HW2_solution.ipynb` contains step‑by‑step experiments and plots used in the report.
- Outputs: SHAP/LIME plots, Grad-CAM heatmaps, activation-maximization images; saved by `generate_report_plots.py`.
- Runtime: tabular experiments are quick on CPU; vision utilities (activation maximization) are faster on GPU but runnable on CPU for small steps.
- Offline behavior: `tabular.py` falls back to deterministic synthetic diabetes data; `vision.get_vgg16()` falls back to non‑pretrained weights if internet is unavailable.

---

### HW3 — Causal modeling & algorithmic recourse (HomeWorks/HW3/code/q5_codes) ⚖️

- Purpose: implement SCMs, train classifiers (ERM/AF/ALLR/ROSS), and evaluate nearest-counterfactual vs causal recourse (linear & differentiable methods).
- Key files & modules:
  - `main.py`, `runner.py`, `trainers.py`, `recourse.py`, `scm.py`, `evaluate_recourse.py`, `utils.py`, `generate_report_artifacts.py`, `HW3_complete_assignment.ipynb`.
- Entry points & typical runs:
  - Full pipeline: `cd HomeWorks/HW3/code/q5_codes && python main.py --seed 0` (checks for existing checkpoints and reuses artifacts).
  - Notebook: open `HomeWorks/HW3/code/HW3_complete_assignment.ipynb` for interactive exploration.
- What the pipeline does:
  - trains classifiers (logistic / MLP) with several trainers (ERM, AF, ROSS), calibrates thresholds by MCC, computes recourse (linear LP via CVXPY or greedy fallback and differentiable recourse), and aggregates results into plot-ready artifacts.
- Important outputs (naming conventions):
  - Results saved under `results/` with deterministic names: `<model>_<trainer>_e{eps}_s{seed}_{metric}.npy` (`_ids.npy`, `_valid.npy`, `_cost.npy`, `_accs.npy`).
  - Trained model files under `HomeWorks/HW3/models/` (e.g., `health_AF_lin_s0.pth`).
- Metrics reported: classifier accuracy, MCC-thresholded performance, recourse validity rate, valid-only mean cost.
- Reproduction tips:
  - Use `--seed` for deterministic splits; the pipeline is restart-safe—existing checkpoints are reused.
  - CVXPY is used for LP-based linear recourse; a greedy solver fallback exists if CVXPY is not installed.
- Runtime & resources: training linear models is fast on CPU; training MLPs and running many recourse solves (differentiable recourse) benefits from GPU for speed.

---

### HW4 — Security (Neural Cleanse), Privacy & Fairness (HomeWorks/HW4/code) 🔐

- Purpose: demonstrate backdoor detection (Neural Cleanse), differential-privacy calculations (Laplace mechanism), and fairness measurement/mitigation.
- Key files:
  - `neural_cleanse.py`, `privacy.py`, `fairness.py`, `generate_report_figs.py`, `tests/`.
- Neural Cleanse features:
  - `reconstruct_trigger()` to optimize mask+pattern per target label, `detect_outlier_scales()` (MAD) for detection, `evaluate_asr()` for attack success rate.
  - Helpers to extract provided poisoned checkpoints: `extract_poisoned_models_if_needed()` + `resolve_checkpoint_path()`.
- Privacy utilities:
  - Laplace scale & noise helpers: `laplace_scale()`, `add_laplace_noise()`, `laplace_cdf_threshold()`, `compose_epsilons()`.
  - Scenario calculators for assignment Q2 are in `privacy.py`.
- Fairness utilities:
  - `train_baseline_model()`, `disparate_impact()`, `zemel_proxy_fairness()`, promotion/demotion label-swap mitigation, threshold optimization.
- How to run & tests:
  - `cd HomeWorks/HW4/code && pytest tests` runs the unit tests.
  - Quick demos: run `python neural_cleanse.py` or `python fairness.py` (each has a `__main__` quick demo).
- Outputs: detection plots, reconstructed trigger masks/patterns, fairness metric summaries, and report-ready figures produced by `generate_report_figs.py`.

---

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
