# HW1 — Project structure

This file documents the folder/file layout that was added for Homework 1 (Trusted AI). Use it as a quick reference for where to find scripts, models, losses, attacks and notebooks.

```
HomeWorks/HW1/
├─ code/
│  ├─ .gitignore
│  ├─ README.md
│  ├─ README_EXPERIMENTS.md
│  ├─ README_RUNS.md
│  ├─ requirements.txt
│  ├─ azure_deploy_notes.txt
│  ├─ train.py                 # main training script (long runs + checkpoints)
│  ├─ eval.py                  # evaluation + UMAP feature visualization
│  ├─ trainers.py              # small Trainer helper (optional)
│  ├─ runner.py                # programmatic runner calling train.py
│  ├─ attacks.py               # FGSM / PGD adversarial generators
│  ├─ datasets.py              # dataset loaders & transforms (SVHN / MNIST / CIFAR10)
│  ├─ losses.py                # LabelSmoothingCrossEntropy + CircleLoss
│  ├─ utils.py                 # checkpointing, seed, helpers
│  ├─ evaluate_recourse.py     # placeholder (not used)
│  ├─ notebooks/
│  │  └─ analysis.ipynb        # analysis / visualization starter notebook
│  ├─ models/
│  │  └─ resnet18_custom.py    # ResNet18 implemented from scratch (BN optional)
│  └─ checkpoints/             # (ignored) where training checkpoints are saved
├─ description/
│  └─ README.md                # assignment description (original en.md)
└─ report/
   └─ assignment_template.tex  # LaTeX report template
```

Key run examples
- Baseline SVHN training:
  `python train.py --dataset svhn --epochs 80 --batch-size 128 --lr 0.1 --optimizer sgd --save-dir checkpoints/svhn_baseline`
- CIFAR10 adversarial training:
  `python train.py --dataset cifar10 --adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7 --epochs 100 --save-dir checkpoints/cifar_adv`
- UMAP of test features:
  `python eval.py --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap`

Notes
- `checkpoints/` is included in `.gitignore` and will be created at runtime.
- The code is structured so you can run experiments by changing flags in `train.py` (label smoothing, BN on/off, adversarial training, optimizer choice).

If you want, I can also add a rendered tree (JSON or DOT) or commit this file to git — tell me which next step you prefer. ✅