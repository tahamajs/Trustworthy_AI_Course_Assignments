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

## 2. Label smoothing ablation (smoothing=0.25 per assignment)

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --label-smoothing 0.25 \
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

## 5. Augmentation preset (digits_safe: no horizontal flip)

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --augment-preset digits_safe \
  --save-dir checkpoints/svhn_augment_digits_safe
```

## 6. Pretrained ResNet18 (ImageNet) as feature extractor

```bash
python train.py \
  --dataset svhn \
  --epochs 80 \
  --pretrained imagenet \
  --save-dir checkpoints/svhn_pretrained
```

Optional: `--freeze-backbone` to train only the classifier.

## 7. Reverse training (MNIST then test on SVHN)

```bash
python train.py --dataset mnist --epochs 80 --save-dir checkpoints/mnist_baseline
python eval.py --dataset svhn --checkpoint checkpoints/mnist_baseline/best.pth --eval-datasets mnist,svhn --cross-eval-csv checkpoints/mnist_baseline/cross_eval.csv
```

## 8. Fine-tuning (MNIST → 750 SVHN, freeze conv)

```bash
# First train on MNIST
python train.py --dataset mnist --epochs 80 --save-dir checkpoints/mnist_for_finetune
# Then fine-tune classifier only on 500–1000 SVHN samples
python train.py \
  --finetune-from checkpoints/mnist_for_finetune/best.pth \
  --finetune-dataset svhn \
  --finetune-samples 750 \
  --freeze-conv \
  --epochs 20 \
  --save-dir checkpoints/mnist_finetune_svhn
```

## 9. Cross-domain evaluation (one checkpoint on multiple test sets)

```bash
python eval.py \
  --checkpoint checkpoints/svhn_baseline/best.pth \
  --eval-datasets svhn,mnist \
  --cross-eval-csv checkpoints/svhn_baseline/cross_eval.csv \
  --dataset svhn
```

For a checkpoint saved with `--pretrained imagenet`, add `--pretrained`.

## 10. Robustness: CIFAR10 with 20% train / 80% val (stratified)

```bash
python train.py \
  --dataset cifar10 \
  --cifar-split 0.2 \
  --epochs 100 \
  --save-dir checkpoints/cifar_baseline
```

## 11. Adversarial training (CIFAR10 + PGD, 50% prob)

```bash
python train.py \
  --dataset cifar10 \
  --cifar-split 0.2 \
  --epochs 100 \
  --adv-train \
  --attack pgd \
  --epsilon 8/255 \
  --alpha 2/255 \
  --iters 7 \
  --save-dir checkpoints/cifar_adv_pgd
```

## 12. Circle Loss (CIFAR10)

```bash
python train.py \
  --dataset cifar10 \
  --cifar-split 0.2 \
  --epochs 100 \
  --loss circle \
  --save-dir checkpoints/cifar_circle
```

## 13. Adversarial examples grid (FGSM epsilon 0.1 per assignment)

```bash
python eval.py \
  --dataset cifar10 \
  --checkpoint checkpoints/cifar_baseline/best.pth \
  --save-grid \
  --grid-path checkpoints/cifar_baseline/grid_eps01.png \
  --epsilon 0.1 \
  --attack fgsm
```

## 14. Evaluation and UMAP

```bash
python eval.py \
  --dataset svhn \
  --checkpoint checkpoints/svhn_baseline/best.pth \
  --umap
```

## 15. Full HW1 pipeline (all experiments + summary CSVs)

```bash
python run_full_hw1.py --demo
```

Writes `checkpoints/report_summary/training_summary.csv`, `cross_domain_summary.csv`, `robustness_summary.csv` and copies figures to `../report/figures`. Use `--no-demo` and `--epochs 80` (or more) for full dataset runs.

## Notes

- Best checkpoint is always saved as `best.pth`.
- `--demo` can be used for fast smoke runs on CPU.
- Training logs are written by TensorBoard into `--save-dir`.
- MNIST is single-channel; the pipeline converts to RGB in `datasets.py` so the same 3-channel ResNet18 is used for SVHN and MNIST.
