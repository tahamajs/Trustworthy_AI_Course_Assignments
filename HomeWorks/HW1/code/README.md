# HW1 Code Guide

This folder contains the full implementation for HW1.

## Implemented methods

### 1) Model architecture

- `models/resnet18_custom.py`
  - `BasicBlock`: residual block with optional BatchNorm.
  - `ResNet` and `resnet18(...)`: full ResNet18 defined manually.
  - Supports `return_features=True` for embedding extraction.

### 2) Losses

- `losses.py`
  - `LabelSmoothingCrossEntropy`: soft target distribution for improved generalization.
  - `CircleLoss`: metric-learning style pair-similarity objective over normalized embeddings.

### 3) Adversarial methods

- `attacks.py`
  - `fgsm_attack(...)`: single-step sign-gradient perturbation with `epsilon`.
  - `pgd_attack(...)`: iterative projected attack with `epsilon`, `alpha`, and `iters`.

### 4) Data handling

- `datasets.py`
  - `get_transforms(...)`: resize, optional augmentation, normalization.
  - `get_dataloaders(...)`: SVHN/MNIST/CIFAR10 loaders.
  - MNIST is converted to RGB when needed.
  - Includes offline fallback to `torchvision.datasets.FakeData`.

### 5) Training/evaluation pipeline

- `train.py`
  - `train_one_epoch(...)`, `evaluate(...)`.
  - Supports `SGD` or `Adam`.
  - Uses `MultiStepLR` (milestones at 50% and 75% of epochs).
  - Optional adversarial training (`--adv-train`) with FGSM/PGD.
- `eval.py`
  - `extract_features(...)`: gets penultimate features.
  - `plot_umap(...)`: 2D UMAP of features.
  - Optional sample-grid export for report figures.

### 6) Utilities

- `utils.py`: seed control and checkpoint I/O.
- `trainers.py`: minimal reusable trainer helper.
- `runner.py`: lightweight entry point that calls `train.main()`.

## Quick run

```bash
cd HomeWorks/HW1/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Baseline:

```bash
python train.py --dataset svhn --epochs 80 --batch-size 128 --optimizer sgd --save-dir checkpoints/svhn_baseline
```

Adversarial training:

```bash
python train.py --dataset cifar10 --epochs 100 --adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7 --save-dir checkpoints/cifar_adv
```

Evaluation + UMAP:

```bash
python eval.py --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap
```

One-command report artifact generation (demo-safe):

```bash
python run_report_pipeline.py --epochs 3
```

Full dataset run (long):

```bash
python run_report_pipeline.py --full-run --epochs 80 --dataset svhn
```

## Outputs

- Checkpoints: `checkpoints/<experiment>/last.pth`, `best.pth`
- TensorBoard logs: same `--save-dir`
- UMAP figure: `<checkpoint>.umap.png`
- Adversarial/noise figure: `<checkpoint>.grid.png`
- Training curves + history: `<save-dir>/training_curves.png`, `training_history.{csv,json}`
