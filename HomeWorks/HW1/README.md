# HW1: Generalization and Robustness

HW1 implements image-classification experiments for Trusted AI.

## Folders

- `code`: all runnable implementations.
- `description`: official assignment text.
- `report`: LaTeX report files.

## Methods used in this project

- **Custom ResNet18** (`models/resnet18_custom.py`):
  manual `BasicBlock` implementation with optional BatchNorm.
- **Generalization methods**:
  data augmentation, optimizer comparison (`SGD` vs `Adam`), optional BatchNorm ablation, label smoothing.
- **Robustness methods**:
  FGSM and PGD adversarial attacks, adversarial training (50% adversarial batches), optional Circle Loss module.
- **Representation analysis**:
  UMAP projection of learned feature vectors for qualitative comparison.

## Quick start

```bash
cd HomeWorks/HW1/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset svhn --epochs 80 --batch-size 128 --optimizer sgd --save-dir checkpoints/svhn_baseline
python eval.py --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap
```

For experiment presets and comparison runs, see:

- `code/README.md`
- `code/README_RUNS.md`
- `code/README_experiments.md`
