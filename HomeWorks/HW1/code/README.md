# HW1/code — Developer & Reproducibility Guide

This file documents the implementation, CLI, and reproducible experiment recipes for HW1 (image generalization & adversarial robustness).

---

## Contents & responsibilities

- `train.py` — training loop, scheduler, checkpointing, optional adversarial training.
- `eval.py` — evaluation utilities (UMAP, confusion, per-class metrics, calibration, robustness sweeps).
- `run_report_pipeline.py` — convenience script that trains, evaluates, and copies figures into `../report/figures/`.
- `attacks.py` — FGSM and PGD attack implementations used for evaluation and adversarial training.
- `datasets.py` — transforms and dataloaders for `svhn`, `cifar10`, `mnist` (MNIST→RGB fallback).
- `models/` — `resnet18_custom.py` (manual ResNet18 with optional BatchNorm and feature extraction).
- `utils.py`, `trainers.py`, `runner.py` — helpers for seeding, checkpoint I/O, and small reusable training utilities.

---

## Environment & dependencies

- Python 3.8+ recommended.
- Install with:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- For GPU acceleration install a PyTorch build that matches your CUDA version.
- Optional: `umap-learn` for UMAP visualizations (fallback to PCA is automatic).

---

## Key CLI examples (most-used scripts)

train.py (core training)

- Purpose: train a ResNet18 baseline or adversarially-trained model.
- Important flags:
  - `--dataset` (svhn|cifar10|mnist)
  - `--epochs`, `--batch-size`, `--lr`
  - `--optimizer` (sgd|adam)
  - `--use-bn` (true/false)
  - `--label-smoothing` (float)
  - `--adv-train` (enable adversarial training)
  - `--attack` (fgsm|pgd)
  - `--epsilon`, `--alpha`, `--iters` (attack params)
  - `--save-dir` (checkpoint folder)
  - `--demo` (reduced dataset/time for CI/demo)

Example — baseline:

```bash
python train.py --dataset svhn --epochs 80 --batch-size 128 --optimizer sgd --save-dir checkpoints/svhn_baseline
```

Example — adversarial training (PGD):

```bash
python train.py --dataset cifar10 --epochs 100 --adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7 --save-dir checkpoints/cifar_adv
```

eval.py (evaluation & diagnostic figures)

- Purpose: extract features, compute metrics (ECE, AURC), generate UMAP/UMAP fallback, confusion matrices, top-k, and robustness sweeps.
- Notable flags: `--checkpoint`, `--umap`, `--save-grid`, `--save-confusion`, `--save-prf1`, `--save-calibration`, `--save-attack-sweep`, `--demo`.

Example — generate UMAP & evaluation artifacts:

```bash
python eval.py --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap --save-grid --save-confusion --save-prf1
```

run_report_pipeline.py

- Purpose: single-command experiment → evaluation → copy report-ready figures into `../report/figures/`.
- Use `--full-run` to disable demo mode and run on the complete dataset.

Example (demo):

```bash
python run_report_pipeline.py --dataset svhn --epochs 3 --save-dir checkpoints/svhn_demo
```

---

## Reproducibility & tips

- Seed: pass `--seed` to `train.py` for deterministic splits and RNG seeding.
- Deterministic PyTorch: if strict determinism is required, enable `torch.use_deterministic_algorithms(True)` in `utils.set_seed` (beware of slower kernels).
- Checkpoint recovery: scripts save `{best.pth,last.pth}` and history files — re-run `eval.py` with `--checkpoint` to reproduce figures.

---

## Common troubleshooting

- UMAP/numba import errors → `eval.py` falls back to PCA automatically.
- CV memory / OOM → reduce `--batch-size`.
- Missing torchvision datasets → run training with `--demo` or place dataset under `code/data/`.

---

## Development notes

- Add new dataset: update `datasets.get_dataloaders()` and add normalization stats in `eval.py` (`_stats_for_dataset`).
- Add new model: place model under `models/` and add CLI option in `train.py` / `runner.py` as needed.

---

## Expected artifacts

- `checkpoints/<exp>/best.pth` — model weights
- `checkpoints/<exp>/training_history.json|csv` — scalar histories
- `<exp>.umap.png`, `<exp>.grid.png`, `<exp>.confusion.png` — evaluation figures

If you want, I can add detailed `argparse` tables for each script or create runnable examples for each saved checkpoint.
