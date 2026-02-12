HW1 — Trusted AI (complete solution)

Overview
- Implements Part 1 (Generalization: ResNet18 from scratch, label smoothing, data-augmentation, pre-trained feature extractor, optimizer comparisons, fine-tuning) and Part 2 (Robustness: FGSM/PGD attacks, Circle Loss, adversarial training, UMAP visualization).

Quick start (GPU recommended)
1. Create a virtualenv and install requirements:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2. Example full training (SVHN -> test SVHN + MNIST):
   python train.py --dataset svhn --epochs 80 --batch-size 128 --lr 0.1 --optimizer sgd --save-dir checkpoints/svhn_baseline

3. Run robustness experiment (CIFAR10, adversarial training):
   python train.py --dataset cifar10 --epochs 100 --batch-size 128 --optimizer sgd --adv-train --attack pgd --epsilon 8/255 --save-dir checkpoints/cifar_adv

4. Evaluate & visualize embeddings:
   python eval.py --dataset svhn --checkpoint checkpoints/svhn_baseline/best.pth --umap

Notes
- Default scripts checkpoint automatically and save training logs in the same folder as checkpoints.
- The `--demo` flag runs short quick checks (useful on CPU). Remove it for full/long runs.

Files
- models/resnet18_custom.py  : manual ResNet18 implementation (BN optional)
- losses.py                  : LabelSmoothing + Circle Loss
- attacks.py                 : FGSM, PGD generators
- datasets.py                : dataset loaders and augmentations (handles MNIST<->RGB conversions)
- train.py                   : training loop + options for adversarial training, label-smoothing, optimizer choice
- eval.py                    : evaluation, UMAP visualizations
- utils.py                   : helpers (checkpointing, seed, metrics)
- notebooks/analysis.ipynb   : starter analysis notebook

If you want, I can run a short smoke-run or create SLURM/GPU job scripts next.