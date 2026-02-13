# HW1 Experiment Checklist

Use this list to cover the required comparisons in a reproducible order.

## A. Generalization experiments

1. Baseline ResNet18 on SVHN
   - `python train.py --dataset svhn --epochs 80 --optimizer sgd --save-dir checkpoints/svhn_baseline`
2. BatchNorm ablation
   - `python train.py --dataset svhn --epochs 80 --use-bn false --save-dir checkpoints/svhn_no_bn`
3. Label smoothing
   - `python train.py --dataset svhn --epochs 80 --label-smoothing 0.1 --save-dir checkpoints/svhn_label_smooth`
4. Optimizer comparison
   - `python train.py --dataset svhn --epochs 80 --optimizer adam --lr 1e-3 --save-dir checkpoints/svhn_adam`
5. Reverse domain direction (MNIST to SVHN)
   - `python train.py --dataset mnist --epochs 80 --save-dir checkpoints/mnist_train`
6. Evaluate all saved checkpoints with UMAP
   - `python eval.py --dataset svhn --checkpoint checkpoints/<exp>/best.pth --umap`

## B. Robustness experiments

1. Baseline on CIFAR10 (no adversarial training)
   - `python train.py --dataset cifar10 --epochs 100 --save-dir checkpoints/cifar_base`
2. FGSM adversarial training
   - `python train.py --dataset cifar10 --epochs 100 --adv-train --attack fgsm --epsilon 8/255 --save-dir checkpoints/cifar_adv_fgsm`
3. PGD adversarial training
   - `python train.py --dataset cifar10 --epochs 100 --adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7 --save-dir checkpoints/cifar_adv_pgd`

## C. Optional extension

- Circle Loss is implemented in `losses.py` (`CircleLoss` class).  
  To use it in training, replace the criterion in `train.py` with `CircleLoss` over embeddings.
