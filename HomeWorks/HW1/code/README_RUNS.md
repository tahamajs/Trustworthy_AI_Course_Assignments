# HW1 Recommended Runs

Run these commands from `HomeWorks/HW1/code`.

## 1. Baseline (SVHN)

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --save-dir checkpoints/svhn_baseline
```

## 2. Label smoothing ablation

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --label-smoothing 0.1 \
  --save-dir checkpoints/svhn_label_smooth
```

## 3. BatchNorm ablation

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --use-bn false \
  --save-dir checkpoints/svhn_no_bn
```

## 4. Adam vs SGD

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --optimizer adam \
  --lr 1e-3 \
  --save-dir checkpoints/svhn_adam
```

## 5. Robustness: adversarial training (CIFAR10 + PGD)

```bash
python train.py \
  --dataset cifar10 \
  --epochs 100 \
  --adv-train \
  --attack pgd \
  --epsilon 8/255 \
  --alpha 2/255 \
  --iters 7 \
  --save-dir checkpoints/cifar_adv_pgd
```

## 6. Evaluation and UMAP

```bash
python eval.py \
  --dataset svhn \
  --checkpoint checkpoints/svhn_baseline/best.pth \
  --umap
```

## Notes

- Best checkpoint is always saved as `best.pth`.
- `--demo` can be used for fast smoke runs on CPU.
- Training logs are written by TensorBoard into `--save-dir`.
