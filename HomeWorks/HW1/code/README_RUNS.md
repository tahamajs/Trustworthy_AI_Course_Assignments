Recommended hyperparameters & long-run presets

1) SVHN baseline (train on SVHN, test SVHN + MNIST)
- epochs: 80
- batch-size: 128
- optimizer: SGD, lr=0.1, momentum=0.9, weight-decay=5e-4
- label-smoothing: 0.0 (baseline)
- augmentations: random crop + horizontal flip + color jitter
- checkpoints saved to `checkpoints/svhn_baseline/`

2) SVHN + Label smoothing
- same as baseline but `--label-smoothing 0.1`

3) SVHN without BatchNorm
- `--use-bn False` (retrain to compare)

4) CIFAR10 adversarial experiments
- epochs: 100
- adv-train: use `--adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7`

Logging & checkpoints
- tensorboard logs are written to the same `--save-dir`
- best model is saved as `best.pth`

Notes
- Full training assumes a GPU with ~8+ GB memory. For multi-GPU adaptation, wrap model with `torch.nn.DataParallel` or use distributed training.
