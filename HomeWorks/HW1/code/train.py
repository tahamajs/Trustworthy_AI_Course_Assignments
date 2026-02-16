import argparse
import csv
import json
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloaders
from models.resnet18_custom import resnet18
from losses import LabelSmoothingCrossEntropy, CircleLoss
from attacks import fgsm_attack, pgd_attack
from utils import set_seed, save_checkpoint, load_checkpoint


def save_training_history(save_dir, history):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, 'training_history.json')
    csv_path = os.path.join(save_dir, 'training_history.csv')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(len(history['epoch'])):
            writer.writerow(
                [
                    history['epoch'][i],
                    history['train_loss'][i],
                    history['train_acc'][i],
                    history['val_loss'][i],
                    history['val_acc'][i],
                ]
            )


def save_training_plot(save_dir, history):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[WARN] Could not save training plot: {e}')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history['epoch'], history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['epoch'], history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross Entropy')
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].legend()

    axes[1].plot(history['epoch'], history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['epoch'], history['val_acc'], label='Val', linewidth=2)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Top-1 Accuracy (%)')
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].legend()

    fig.suptitle('Training Curves')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=220)
    plt.close(fig)


def train_one_epoch(model, loader, optimizer, device, epoch, loss_fn, adv_attack=None, adv_params=None, use_circle_loss=False, ce_for_adv=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # For adversarial attack we need a loss on logits (CE); circle_loss is on embeddings
    attack_loss_fn = ce_for_adv if (adv_attack and use_circle_loss) else loss_fn
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if adv_attack is not None and torch.rand(1).item() < adv_params.get('prob', 1.0):
            if adv_attack == 'fgsm':
                x = fgsm_attack(model, x, y, adv_params['epsilon'], loss_fn=attack_loss_fn, device=device)
            elif adv_attack == 'pgd':
                x = pgd_attack(model, x, y, adv_params['epsilon'], adv_params['alpha'], adv_params['iters'], loss_fn=attack_loss_fn, device=device)
        if use_circle_loss:
            logits, feat = model(x, return_features=True)
            loss = loss_fn(feat, y)
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
    return running_loss/total, 100.*correct/total


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += x.size(0)
    return running_loss/total, 100.*correct/total


def parse_args():
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        val = str(v).strip().lower()
        if val in {'1', 'true', 't', 'yes', 'y'}:
            return True
        if val in {'0', 'false', 'f', 'no', 'n'}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='svhn')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument(
        '--use-bn',
        type=str_to_bool,
        nargs='?',
        const=True,
        default=True,
        help='Enable/disable BatchNorm (supports --use-bn, --use-bn true, --use-bn false)',
    )
    parser.add_argument('--adv-train', action='store_true')
    parser.add_argument('--attack', choices=['fgsm', 'pgd'], default='fgsm')
    parser.add_argument('--epsilon', type=str, default='8/255')
    parser.add_argument('--alpha', type=str, default='2/255')
    parser.add_argument('--iters', type=int, default=7)
    parser.add_argument('--save-dir', type=str, default='checkpoints/exp')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # Pretrained ImageNet and augmentation
    parser.add_argument('--pretrained', type=str, default=None, choices=[None, 'imagenet'], help='Use ImageNet-pretrained ResNet18 (imagenet)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone when using --pretrained imagenet')
    parser.add_argument('--augment-preset', type=str, default='default', choices=['default', 'digits_safe'])
    parser.add_argument('--cifar-split', type=float, default=None, help='CIFAR10 train ratio (e.g. 0.2 for 20%% train / 80%% val), stratified')
    # Fine-tuning: load from checkpoint, train only classifier on small SVHN subset
    parser.add_argument('--finetune-from', type=str, default=None)
    parser.add_argument('--finetune-dataset', type=str, default='svhn')
    parser.add_argument('--finetune-samples', type=int, default=750)
    parser.add_argument('--freeze-conv', action='store_true', help='Freeze conv layers (only train classifier); use with --finetune-from')
    # Loss: ce (default), circle
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'circle'])
    args = parser.parse_args()
    return args


def str_to_eps(s):
    if '/' in s:
        a, b = s.split('/')
        return float(a) / float(b)
    return float(s)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset for fine-tuning: cap samples and use finetune-dataset
    use_finetune = args.finetune_from is not None
    train_dataset = args.finetune_dataset if use_finetune else args.dataset
    max_train_samples = args.finetune_samples if use_finetune else None
    cifar_ratio = args.cifar_split if train_dataset.lower() == 'cifar10' else None
    if use_finetune:
        args.epochs = min(args.epochs, 20)  # typical fine-tuning length

    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        train_dataset,
        batch_size=args.batch_size,
        augment=True,
        demo=args.demo,
        augmentation_preset=args.augment_preset,
        cifar10_train_ratio=cifar_ratio,
        max_train_samples=max_train_samples,
        seed=args.seed,
    )

    use_pretrained = args.pretrained == 'imagenet'
    if use_pretrained:
        from models.resnet18_pretrained import resnet18_imagenet
        model = resnet18_imagenet(num_classes=num_classes, in_channels=in_channels, freeze_backbone=args.freeze_backbone, pretrained=True)
    else:
        model = resnet18(num_classes=num_classes, in_channels=in_channels, use_bn=args.use_bn)

    # Load checkpoint for fine-tuning
    if use_finetune:
        ck = load_checkpoint(args.finetune_from, device)
        model.load_state_dict(ck['state_dict'], strict=False)
        if args.freeze_conv:
            for name, param in model.named_parameters():
                if use_pretrained:
                    if 'backbone.fc' not in name:
                        param.requires_grad = False
                else:
                    if 'fc' not in name:
                        param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'[Finetune] Frozen conv; training {trainable} parameters')

    model = model.to(device)

    if args.loss == 'circle':
        loss_fn = CircleLoss(m=0.25, gamma=256)
        ce_for_adv = nn.CrossEntropyLoss()
    elif args.label_smoothing > 0:
        loss_fn = LabelSmoothingCrossEntropy(args.label_smoothing)
        ce_for_adv = None
    else:
        loss_fn = nn.CrossEntropyLoss()
        ce_for_adv = None

    # Validation uses CE on logits for scalar reporting (same for circle training)
    eval_loss_fn = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    writer = SummaryWriter(log_dir=args.save_dir)
    best_acc = 0.0
    epsilon = str_to_eps(args.epsilon)
    alpha = str_to_eps(args.alpha)
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    adv_params = {'epsilon': epsilon, 'alpha': alpha, 'iters': args.iters, 'prob': 0.5}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, loss_fn,
            adv_attack=(args.attack if args.adv_train else None),
            adv_params=adv_params,
            use_circle_loss=(args.loss == 'circle'),
            ce_for_adv=ce_for_adv,
        )
        val_loss, val_acc = evaluate(model, test_loader, device, eval_loss_fn)

        scheduler.step()

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'pretrained': use_pretrained,
        }
        save_checkpoint(ckpt, is_best, args.save_dir)

        print(f"Epoch {epoch}  Train Loss {train_loss:.4f}  Train Acc {train_acc:.2f}%  Val Loss {val_loss:.4f}  Val Acc {val_acc:.2f}%  Best {best_acc:.2f}%")

    writer.close()
    save_training_history(args.save_dir, history)
    save_training_plot(args.save_dir, history)


if __name__ == '__main__':
    main()
