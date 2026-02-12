Experiment checklist (what to run)

1) Baseline ResNet18 on SVHN
   - python train.py --dataset svhn --epochs 80 --batch-size 128 --lr 0.1 --optimizer sgd --save-dir checkpoints/svhn_baseline
   - Evaluate on MNIST test: python eval.py --dataset mnist --checkpoint checkpoints/svhn_baseline/best.pth --umap

2) Remove BatchNorm
   - python train.py --dataset svhn --use-bn False --epochs 80 --batch-size 128 --lr 0.1 --save-dir checkpoints/svhn_no_bn

3) Label-smoothing
   - python train.py --dataset svhn --label-smoothing 0.1 --epochs 80 --save-dir checkpoints/svhn_label_smooth

4) Pretrained feature extractor (use torchvision's resnet18 pretrained) — code stub in notebook

5) CIFAR10 adversarial experiments
   - python train.py --dataset cifar10 --adv-train --attack pgd --epsilon 8/255 --alpha 2/255 --iters 7 --epochs 100 --save-dir checkpoints/cifar_adv

6) Circle Loss training — use notebook and modify training loop to train on embeddings (example available on request)
