import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloaders
from models.resnet18_custom import resnet18
from losses import LabelSmoothingCrossEntropy
from attacks import fgsm_attack, pgd_attack
from utils import set_seed, save_checkpoint


def train_one_epoch(model, loader, optimizer, device, epoch, loss_fn, adv_attack=None, adv_params=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if adv_attack is not None and torch.rand(1).item() < adv_params.get('prob', 1.0):
            # generate adversarial example on the fly
            if adv_attack == 'fgsm':
                x = fgsm_attack(model, x, y, adv_params['epsilon'], loss_fn=loss_fn, device=device)
            elif adv_attack == 'pgd':
                x = pgd_attack(model, x, y, adv_params['epsilon'], adv_params['alpha'], adv_params['iters'], loss_fn=loss_fn, device=device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='svhn')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--use-bn', action='store_true', default=True)
    parser.add_argument('--adv-train', action='store_true')
    parser.add_argument('--attack', choices=['fgsm', 'pgd'], default='fgsm')
    parser.add_argument('--epsilon', type=str, default='8/255')
    parser.add_argument('--alpha', type=str, default='2/255')
    parser.add_argument('--iters', type=int, default=7)
    parser.add_argument('--save-dir', type=str, default='checkpoints/exp')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
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
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(args.dataset, batch_size=args.batch_size, augment=True, demo=args.demo)

    model = resnet18(num_classes=num_classes, in_channels=in_channels, use_bn=args.use_bn)
    model = model.to(device)

    if args.label_smoothing > 0:
        loss_fn = LabelSmoothingCrossEntropy(args.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr schedule: simple step-down at 50% and 75%
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    writer = SummaryWriter(log_dir=args.save_dir)
    best_acc = 0.0
    epsilon = str_to_eps(args.epsilon)
    alpha = str_to_eps(args.alpha)

    adv_params = {'epsilon': epsilon, 'alpha': alpha, 'iters': args.iters, 'prob': 0.5}

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, loss_fn, adv_attack=(args.attack if args.adv_train else None), adv_params=adv_params)
        val_loss, val_acc = evaluate(model, test_loader, device, loss_fn)
        scheduler.step()

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}, is_best, args.save_dir)

        print(f"Epoch {epoch}  Train Loss {train_loss:.4f}  Train Acc {train_acc:.2f}%  Val Loss {val_loss:.4f}  Val Acc {val_acc:.2f}%  Best {best_acc:.2f}%")

    writer.close()


if __name__ == '__main__':
    main()
