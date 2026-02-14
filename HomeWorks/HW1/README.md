# HW1 — Generalization & Robustness

Comprehensive image-classification experiments and analysis focused on generalization and adversarial robustness (datasets: SVHN, CIFAR‑10, MNIST).

---

## Overview

- Purpose: compare training choices (optimizer, BN, label smoothing, augmentation) and robustness strategies (FGSM/PGD attacks, adversarial training).
- Deliverables: training scripts, evaluation (UMAP, calibration, per-class metrics), pre-saved checkpoints and report figures.

---

## Folder layout (important)

- `code/` — all runnable scripts, models, and experiment helpers (primary working directory).
- `data/` — local dataset copies (SVHN, CIFAR10, MNIST).
- `checkpoints/` — saved models and training histories (best/last checkpoints, TensorBoard logs).
- `report/` — LaTeX report and figures (export targets).
- `description/` — assignment text and problem statement.

---

## Quick setup (recommended)

1. Create environment and install:

   ```bash
   cd HomeWorks/HW1/code
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare data: the code uses torchvision datasets and will download automatically to `data/` if missing.

3. Run a short demo train (fast):

   ```bash
   python train.py --dataset svhn --epochs 3 --batch-size 128 --demo --save-dir checkpoints/svhn_demo
   ```

---

## Reproduce common experiments (examples)

- Baseline training (SGD):

  ```bash
  cd HomeWorks/HW1/code
  python train.py --dataset svhn --epochs 80 --batch-size 128 --optimizer sgd --save-dir checkpoints/svhn_baseline
  ```

- Adversarial training (PGD):

  ```bash
  python train.py --dataset cifar10 --epochs 100 --adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7 --save-dir checkpoints/cifar_adv
  ```

- Evaluation + feature UMAP + example grid:

  ```bash
  python eval.py --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap --save-grid --grid-samples 8
  ```

- Full pipeline that trains+evaluates+saves report figures (demo mode by default):

  ```bash
  python run_report_pipeline.py --dataset svhn --epochs 3 --save-dir checkpoints/svhn_demo
  ```

---

## Reproducibility checklist ✅

- Use `--seed` to reproduce runs (default: `42`).
- Keep `--demo` for quick checkpoints; remove it for full experiments.
- Exact environment: see `code/requirements.txt` (recommended to use the pinned versions).
- Checkpoint naming: `checkpoints/<experiment>/{best.pth,last.pth,training_history.json}`.

---

## Outputs & artifacts

- Checkpoints: `best.pth` and `last.pth` in a save-dir.
- Training history: `training_history.{csv,json}`, `training_curves.png`.
- Evaluation figures: UMAP, confusion matrix, per-class plots, reliability diagrams in the save-dir (or copied to `report/figures/` by `run_report_pipeline.py`).

---

## Where to look next

- `code/README.md` — detailed developer guide and CLI reference for `train.py`, `eval.py`, and helper modules.
- `code/notebooks/` — quick analysis notebooks for UMAP / representation inspection.

---

## Troubleshooting (common issues)

- CUDA OOM: reduce `--batch-size` or use `--demo` for smaller batches.
- Missing dataset: `torchvision` will auto-download; ensure network access or place datasets under `code/data/`.
- UMAP fails (numba/umap missing): `eval.py` falls back to PCA automatically.

---

## Citation / authors

Course assignment; see repository history for author information. Use the `report/references.bib` for citations used in the assignment.

---

If you want, I can (a) expand the `code/` README next, or (b) generate step-by-step reproduction commands for a specific checkpoint.
