import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

from datasets import get_dataloaders
from models.resnet18_custom import resnet18
from utils import load_checkpoint
from attacks import fgsm_attack, pgd_attack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_features(model, loader, device):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out, feat = model(x, return_features=True)
            feats.append(feat.cpu().numpy())
            labels.append(y.numpy())
    import numpy as np
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


def str_to_ratio(value):
    if '/' in value:
        a, b = value.split('/')
        return float(a) / float(b)
    return float(value)


def parse_ratio_list(values):
    items = [x.strip() for x in values.split(',') if x.strip()]
    return [str_to_ratio(x) for x in items]


def parse_int_list(values):
    items = [x.strip() for x in values.split(',') if x.strip()]
    return [int(x) for x in items]


@torch.no_grad()
def collect_predictions(model, loader, device, max_batches=None):
    model.eval()
    probs_all = []
    preds_all = []
    labels_all = []
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        probs_all.append(probs.cpu().numpy())
        preds_all.append(preds.cpu().numpy())
        labels_all.append(y.numpy())
    return (
        np.concatenate(probs_all, axis=0),
        np.concatenate(preds_all, axis=0),
        np.concatenate(labels_all, axis=0),
    )


def plot_umap(feats, labels, save_path):
    method = 'UMAP'
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        emb = reducer.fit_transform(feats)
    except Exception as e:
        # Robust fallback when numba/umap is not usable in the current environment.
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        emb = reducer.fit_transform(feats)
        method = f'PCA fallback ({type(e).__name__})'

    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=4)
    plt.colorbar(scatter)
    plt.title(f'{method} of Features')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_confusion_matrix(labels, preds, num_classes, save_path, normalize=True):
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1.0
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        cm = cm / row_sum

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], fontsize=8)
    ax.set_yticklabels([str(i) for i in ticks], fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_per_class_accuracy(labels, preds, num_classes, save_path):
    labels = labels.astype(int)
    preds = preds.astype(int)
    acc = []
    for cls in range(num_classes):
        mask = labels == cls
        if np.sum(mask) == 0:
            acc.append(0.0)
        else:
            acc.append(float(np.mean(preds[mask] == labels[mask]) * 100.0))

    fig = plt.figure(figsize=(8.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(num_classes), acc, color='#2A9D8F')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def compute_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if np.sum(mask) == 0:
            stats.append((0.0, 0.0, 0))
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        bin_frac = float(np.mean(mask))
        ece += abs(bin_acc - bin_conf) * bin_frac
        stats.append((bin_acc, bin_conf, int(np.sum(mask))))
    return float(ece), stats


def save_reliability_diagram(probs, labels, save_path, n_bins=10):
    ece, stats = compute_ece(probs, labels, n_bins=n_bins)
    bin_acc = np.array([x[0] for x in stats], dtype=np.float64)
    bin_conf = np.array([x[1] for x in stats], dtype=np.float64)
    bin_counts = np.array([x[2] for x in stats], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    widths = np.diff(bins) * 0.9
    total = np.maximum(np.sum(bin_counts), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.2, label='Perfect calibration')
    ax.bar(centers, bin_acc, width=widths, color='#264653', alpha=0.8, label='Accuracy')
    ax.bar(centers, np.maximum(bin_conf - bin_acc, 0), width=widths, bottom=bin_acc, color='#E76F51', alpha=0.7, label='Gap')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy / Confidence')
    ax.set_title(f'Reliability Diagram (ECE={ece:.4f})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')

    ax_hist = axes[1]
    ax_hist.bar(centers, bin_counts / total, width=widths, color='#2A9D8F', alpha=0.85)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_xlabel('Confidence')
    ax_hist.set_ylabel('Bin Fraction')
    ax_hist.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return ece


def _metric_at_coverage(coverage_curve, value_curve, target_coverage):
    idx = int(np.searchsorted(coverage_curve, target_coverage, side='left'))
    if idx >= len(coverage_curve):
        idx = len(coverage_curve) - 1
    return float(value_curve[idx])


def save_confidence_coverage_plot(probs, labels, save_path, coverage_points=20):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(np.float64)

    n = len(correct)
    if n == 0:
        raise ValueError('No predictions available for confidence-coverage plotting.')

    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    cum_correct = np.cumsum(sorted_correct)
    ks = np.arange(1, n + 1)
    coverage_curve = ks / float(n)
    selective_acc_curve = cum_correct / ks
    risk_curve = 1.0 - selective_acc_curve
    aurc = float(np.mean(risk_curve))

    if coverage_points < 2:
        coverage_points = 2
    sample_idx = np.linspace(0, n - 1, coverage_points).astype(int)
    sample_cov = coverage_curve[sample_idx] * 100.0
    sample_acc = selective_acc_curve[sample_idx] * 100.0
    sample_risk = risk_curve[sample_idx] * 100.0

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sample_cov, sample_acc, marker='o', linewidth=2, color='#264653', label='Selective Accuracy')
    ax.plot(sample_cov, sample_risk, marker='s', linewidth=2, color='#E76F51', label='Selective Risk')
    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f'Confidence-Coverage Curve (AURC={aurc:.4f})')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

    acc_80 = _metric_at_coverage(coverage_curve, selective_acc_curve, 0.80) * 100.0
    acc_90 = _metric_at_coverage(coverage_curve, selective_acc_curve, 0.90) * 100.0
    return {
        'aurc': aurc,
        'acc_at_80_coverage': float(acc_80),
        'acc_at_90_coverage': float(acc_90),
    }


def compute_classwise_prf1(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(labels.astype(int), preds.astype(int)):
        cm[t, p] += 1.0

    precision = []
    recall = []
    f1 = []
    support = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom_p = tp + fp
        denom_r = tp + fn
        p_val = tp / denom_p if denom_p > 0 else 0.0
        r_val = tp / denom_r if denom_r > 0 else 0.0
        denom_f = p_val + r_val
        f_val = (2.0 * p_val * r_val / denom_f) if denom_f > 0 else 0.0
        precision.append(float(p_val))
        recall.append(float(r_val))
        f1.append(float(f_val))
        support.append(int(cm[cls, :].sum()))

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
    }


def save_classwise_prf1_plot(prf1, save_path):
    precision = np.array(prf1['precision']) * 100.0
    recall = np.array(prf1['recall']) * 100.0
    f1 = np.array(prf1['f1']) * 100.0
    num_classes = len(precision)
    x = np.arange(num_classes)
    width = 0.24

    fig = plt.figure(figsize=(9.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - width, precision, width=width, color='#264653', label='Precision')
    ax.bar(x, recall, width=width, color='#2A9D8F', label='Recall')
    ax.bar(x + width, f1, width=width, color='#E76F51', label='F1')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Score (%)')
    ax.set_title('Class-wise Precision / Recall / F1')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def compute_topk_accuracy(probs, labels, topk):
    scores = {}
    label_vec = labels.astype(int)
    n = len(label_vec)
    if n == 0:
        return {f'top{k}': 0.0 for k in topk}
    max_k = max(topk)
    top_indices = np.argsort(-probs, axis=1)[:, :max_k]
    for k in topk:
        correct = np.any(top_indices[:, :k] == label_vec[:, None], axis=1)
        scores[f'top{k}'] = float(np.mean(correct) * 100.0)
    return scores


def save_topk_accuracy_plot(topk_scores, save_path):
    keys = sorted(topk_scores.keys(), key=lambda x: int(x.replace('top', '')))
    vals = [topk_scores[k] for k in keys]
    labels = [k.upper().replace('TOP', 'Top-') for k in keys]

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    bars = ax.bar(np.arange(len(keys)), vals, color='#1D3557')
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.8, f'{vals[i]:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-k Accuracy Profile')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _stats_for_dataset(dataset_name):
    if dataset_name.lower() == 'svhn':
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    if dataset_name.lower() == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # MNIST is converted to RGB
    return (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)


def denormalize(batch, dataset_name):
    mean, std = _stats_for_dataset(dataset_name)
    mean_t = torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=batch.device).view(1, 3, 1, 1)
    return (batch * std_t + mean_t).clamp(0.0, 1.0)


@torch.no_grad()
def _predict(model, x):
    return model(x).argmax(dim=1)


def save_example_grid(model, loader, device, dataset_name, save_path, attack='fgsm', epsilon=8 / 255, alpha=2 / 255, iters=7, num_samples=8):
    model.eval()
    batch_x, batch_y = next(iter(loader))
    x = batch_x[:num_samples].to(device)
    y = batch_y[:num_samples].to(device)

    loss_fn = nn.CrossEntropyLoss()
    if attack == 'pgd':
        x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=alpha, iters=iters, loss_fn=loss_fn, device=device)
    else:
        x_adv = fgsm_attack(model, x, y, epsilon=epsilon, loss_fn=loss_fn, device=device)
    x_noise = (x + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)

    pred_clean = _predict(model, x)
    pred_adv = _predict(model, x_adv)
    pred_noise = _predict(model, x_noise)

    clean_img = denormalize(x, dataset_name).detach().cpu().permute(0, 2, 3, 1).numpy()
    adv_img = denormalize(x_adv, dataset_name).detach().cpu().permute(0, 2, 3, 1).numpy()
    noise_img = denormalize(x_noise, dataset_name).detach().cpu().permute(0, 2, 3, 1).numpy()
    y_np = y.detach().cpu().numpy()
    pred_clean_np = pred_clean.detach().cpu().numpy()
    pred_adv_np = pred_adv.detach().cpu().numpy()
    pred_noise_np = pred_noise.detach().cpu().numpy()

    rows = [('Clean', clean_img, pred_clean_np), (attack.upper(), adv_img, pred_adv_np), ('Noise', noise_img, pred_noise_np)]
    fig, axes = plt.subplots(3, num_samples, figsize=(2.2 * num_samples, 6.2))
    if num_samples == 1:
        axes = np.array(axes).reshape(3, 1)

    for row_idx, (row_name, row_imgs, row_pred) in enumerate(rows):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            ax.imshow(row_imgs[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=10)
            ax.set_title(f't={int(y_np[col_idx])} p={int(row_pred[col_idx])}', fontsize=8)

    fig.suptitle('Clean vs Adversarial vs Random Noise', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    """Return (mean_loss, accuracy_percent)."""
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
    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def evaluate_accuracy_under_attack(model, loader, device, attack='clean', epsilon=8 / 255, alpha=2 / 255, iters=7, max_batches=None):
    model.eval()
    total = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        if attack == 'fgsm':
            x_eval = fgsm_attack(model, x, y, epsilon=epsilon, loss_fn=loss_fn, device=device)
        elif attack == 'pgd':
            x_eval = pgd_attack(model, x, y, epsilon=epsilon, alpha=alpha, iters=iters, loss_fn=loss_fn, device=device)
        elif attack == 'noise':
            x_eval = (x + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
        else:
            x_eval = x

        with torch.no_grad():
            pred = model(x_eval).argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += x_eval.size(0)
    return 100.0 * correct / max(total, 1)


def save_attack_sweep_plot(model, loader, device, save_path, epsilons, alpha, sweep_iters=3, max_batches=None):
    clean_acc = evaluate_accuracy_under_attack(model, loader, device, attack='clean', epsilon=0.0, alpha=alpha, iters=sweep_iters, max_batches=max_batches)
    fgsm = []
    pgd = []
    noise = []
    eps_scaled = [e * 255.0 for e in epsilons]
    for eps in epsilons:
        fgsm.append(evaluate_accuracy_under_attack(model, loader, device, attack='fgsm', epsilon=eps, alpha=alpha, iters=sweep_iters, max_batches=max_batches))
        pgd.append(evaluate_accuracy_under_attack(model, loader, device, attack='pgd', epsilon=eps, alpha=alpha, iters=sweep_iters, max_batches=max_batches))
        noise.append(evaluate_accuracy_under_attack(model, loader, device, attack='noise', epsilon=eps, alpha=alpha, iters=sweep_iters, max_batches=max_batches))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(eps_scaled, fgsm, marker='o', linewidth=2, label='FGSM')
    ax.plot(eps_scaled, pgd, marker='s', linewidth=2, label='PGD')
    ax.plot(eps_scaled, noise, marker='^', linewidth=2, label='Random Noise')
    ax.axhline(clean_acc, color='black', linestyle='--', linewidth=1.2, label=f'Clean ({clean_acc:.2f}%)')
    ax.set_xlabel('Epsilon (x/255)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustness Sweep')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return {
        'clean_acc': clean_acc,
        'epsilons': eps_scaled,
        'fgsm_acc': fgsm,
        'pgd_acc': pgd,
        'noise_acc': noise,
    }


def save_pgd_iter_sweep_plot(model, loader, device, save_path, epsilon, alpha, iters_list, max_batches=None):
    clean_acc = evaluate_accuracy_under_attack(
        model, loader, device, attack='clean', epsilon=0.0, alpha=alpha, iters=1, max_batches=max_batches
    )
    fgsm_acc = evaluate_accuracy_under_attack(
        model, loader, device, attack='fgsm', epsilon=epsilon, alpha=alpha, iters=1, max_batches=max_batches
    )
    noise_acc = evaluate_accuracy_under_attack(
        model, loader, device, attack='noise', epsilon=epsilon, alpha=alpha, iters=1, max_batches=max_batches
    )

    pgd_acc = []
    for pgd_iter in iters_list:
        pgd_acc.append(
            evaluate_accuracy_under_attack(
                model,
                loader,
                device,
                attack='pgd',
                epsilon=epsilon,
                alpha=alpha,
                iters=int(pgd_iter),
                max_batches=max_batches,
            )
        )

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iters_list, pgd_acc, marker='o', linewidth=2, color='#2A9D8F', label='PGD')
    ax.axhline(clean_acc, color='black', linestyle='--', linewidth=1.2, label=f'Clean ({clean_acc:.2f}%)')
    ax.axhline(fgsm_acc, color='#E76F51', linestyle='-.', linewidth=1.2, label=f'FGSM ({fgsm_acc:.2f}%)')
    ax.axhline(noise_acc, color='#264653', linestyle=':', linewidth=1.5, label=f'Noise ({noise_acc:.2f}%)')
    ax.set_xlabel('PGD Iterations')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'PGD Strength Sweep at epsilon={epsilon * 255.0:.1f}/255')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

    return {
        'epsilon': float(epsilon * 255.0),
        'iters': [int(x) for x in iters_list],
        'pgd_acc': [float(x) for x in pgd_acc],
        'clean_acc': float(clean_acc),
        'fgsm_acc': float(fgsm_acc),
        'noise_acc': float(noise_acc),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='svhn')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--umap', action='store_true')
    p.add_argument('--save-grid', action='store_true')
    p.add_argument('--attack', choices=['fgsm', 'pgd'], default='fgsm')
    p.add_argument('--epsilon', type=str, default='8/255')
    p.add_argument('--alpha', type=str, default='2/255')
    p.add_argument('--iters', type=int, default=7)
    p.add_argument('--grid-samples', type=int, default=8)
    p.add_argument('--umap-path', type=str, default=None)
    p.add_argument('--grid-path', type=str, default=None)
    p.add_argument('--save-confusion', action='store_true')
    p.add_argument('--confusion-path', type=str, default=None)
    p.add_argument('--save-per-class', action='store_true')
    p.add_argument('--per-class-path', type=str, default=None)
    p.add_argument('--save-prf1', action='store_true')
    p.add_argument('--prf1-path', type=str, default=None)
    p.add_argument('--save-calibration', action='store_true')
    p.add_argument('--calibration-path', type=str, default=None)
    p.add_argument('--save-confidence-coverage', action='store_true')
    p.add_argument('--confidence-coverage-path', type=str, default=None)
    p.add_argument('--coverage-points', type=int, default=20)
    p.add_argument('--save-topk', action='store_true')
    p.add_argument('--topk-path', type=str, default=None)
    p.add_argument('--topk-list', type=str, default='1,2,3,5')
    p.add_argument('--save-attack-sweep', action='store_true')
    p.add_argument('--attack-sweep-path', type=str, default=None)
    p.add_argument('--sweep-epsilons', type=str, default='0/255,2/255,4/255,8/255,12/255')
    p.add_argument('--sweep-iters', type=int, default=3)
    p.add_argument('--sweep-max-batches', type=int, default=4)
    p.add_argument('--save-pgd-iter-sweep', action='store_true')
    p.add_argument('--pgd-iter-sweep-path', type=str, default=None)
    p.add_argument('--pgd-iters-list', type=str, default='1,2,4,7,10')
    p.add_argument('--iter-sweep-max-batches', type=int, default=4)
    p.add_argument('--metrics-path', type=str, default=None)
    p.add_argument('--demo', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    # Cross-domain: evaluate one checkpoint on multiple test sets (e.g. svhn,mnist)
    p.add_argument('--eval-datasets', type=str, default=None, help='Comma-separated list, e.g. svhn,mnist')
    p.add_argument('--cross-eval-csv', type=str, default=None, help='Output CSV path for cross-domain accuracies')
    p.add_argument('--pretrained', action='store_true', help='Use ImageNet-pretrained ResNet18 (for loading checkpoint)')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = load_checkpoint(args.checkpoint, device)
    num_classes = int(ck.get('num_classes', 10))
    in_channels = int(ck.get('in_channels', 3))
    use_pretrained = args.pretrained or ck.get('pretrained', False)

    if use_pretrained:
        from models.resnet18_pretrained import resnet18_imagenet
        model = resnet18_imagenet(num_classes=num_classes, in_channels=in_channels, freeze_backbone=False, pretrained=False)
    else:
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(ck['state_dict'], strict=False)
    model.to(device)

    _, test_loader, _, _ = get_dataloaders(args.dataset, batch_size=args.batch_size, demo=args.demo)
    epsilon = str_to_ratio(args.epsilon)
    alpha = str_to_ratio(args.alpha)
    metrics = {}
    loss_fn = nn.CrossEntropyLoss()

    # Cross-domain evaluation: run on multiple datasets and write CSV
    if args.eval_datasets:
        names = [s.strip().lower() for s in args.eval_datasets.split(',') if s.strip()]
        rows = [('eval_dataset', 'accuracy')]
        for dname in names:
            _, loader, _, _ = get_dataloaders(dname, batch_size=args.batch_size, demo=args.demo)
            _, acc = evaluate(model, loader, device, loss_fn)
            rows.append((dname, f'{acc:.4f}'))
            print(f'  {dname}: {acc:.2f}%')
        csv_path = args.cross_eval_csv or (args.checkpoint.rstrip('/') + '.cross_eval.csv')
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            import csv as csv_module
            w = csv_module.writer(f)
            w.writerows(rows)
        print(f'Saved cross-domain eval to: {csv_path}')
        metrics['cross_eval'] = dict(rows[1:])

    if args.umap:
        feats, labels = extract_features(model, test_loader, device)
        umap_path = args.umap_path or (args.checkpoint + '.umap.png')
        plot_umap(feats, labels, save_path=umap_path)
        print(f'Saved UMAP to: {umap_path}')

    if args.save_grid:
        grid_path = args.grid_path or (args.checkpoint + '.grid.png')
        save_example_grid(
            model=model,
            loader=test_loader,
            device=device,
            dataset_name=args.dataset,
            save_path=grid_path,
            attack=args.attack,
            epsilon=epsilon,
            alpha=alpha,
            iters=args.iters,
            num_samples=args.grid_samples,
        )
        print(f'Saved sample grid to: {grid_path}')

    needs_predictions = (
        args.save_confusion
        or args.save_per_class
        or args.save_prf1
        or args.save_calibration
        or args.save_confidence_coverage
        or args.save_topk
    )
    if needs_predictions:
        probs, preds, labels = collect_predictions(model, test_loader, device)
        num_classes = probs.shape[1]
        clean_acc = float(np.mean(preds == labels) * 100.0)
        metrics['clean_acc'] = clean_acc

        if args.save_confusion:
            confusion_path = args.confusion_path or (args.checkpoint + '.confusion.png')
            save_confusion_matrix(labels, preds, num_classes=num_classes, save_path=confusion_path, normalize=True)
            print(f'Saved confusion matrix to: {confusion_path}')

        if args.save_per_class:
            per_class_path = args.per_class_path or (args.checkpoint + '.per_class.png')
            save_per_class_accuracy(labels, preds, num_classes=num_classes, save_path=per_class_path)
            print(f'Saved per-class accuracy plot to: {per_class_path}')

        if args.save_prf1:
            prf1_path = args.prf1_path or (args.checkpoint + '.prf1.png')
            prf1 = compute_classwise_prf1(labels, preds, num_classes=num_classes)
            save_classwise_prf1_plot(prf1, save_path=prf1_path)
            metrics['classwise_prf1'] = prf1
            metrics['macro_precision'] = float(prf1['macro_precision'] * 100.0)
            metrics['macro_recall'] = float(prf1['macro_recall'] * 100.0)
            metrics['macro_f1'] = float(prf1['macro_f1'] * 100.0)
            print(f'Saved class-wise PRF1 plot to: {prf1_path}')

        if args.save_calibration:
            calibration_path = args.calibration_path or (args.checkpoint + '.calibration.png')
            ece = save_reliability_diagram(probs, labels, save_path=calibration_path, n_bins=10)
            metrics['ece'] = float(ece)
            print(f'Saved reliability diagram to: {calibration_path}')

        if args.save_confidence_coverage:
            coverage_path = args.confidence_coverage_path or (args.checkpoint + '.confidence_coverage.png')
            coverage_metrics = save_confidence_coverage_plot(
                probs,
                labels,
                save_path=coverage_path,
                coverage_points=args.coverage_points,
            )
            metrics['confidence_coverage'] = coverage_metrics
            metrics['aurc'] = float(coverage_metrics['aurc'])
            print(f'Saved confidence-coverage plot to: {coverage_path}')

        if args.save_topk:
            topk_path = args.topk_path or (args.checkpoint + '.topk.png')
            topk_list = parse_int_list(args.topk_list)
            topk_scores = compute_topk_accuracy(probs, labels, topk=topk_list)
            save_topk_accuracy_plot(topk_scores, save_path=topk_path)
            metrics['topk_accuracy'] = topk_scores
            if 'top5' in topk_scores:
                metrics['top5_acc'] = float(topk_scores['top5'])
            print(f'Saved top-k accuracy plot to: {topk_path}')

    if args.save_attack_sweep:
        sweep_path = args.attack_sweep_path or (args.checkpoint + '.robustness_sweep.png')
        epsilons = parse_ratio_list(args.sweep_epsilons)
        sweep = save_attack_sweep_plot(
            model=model,
            loader=test_loader,
            device=device,
            save_path=sweep_path,
            epsilons=epsilons,
            alpha=alpha,
            sweep_iters=args.sweep_iters,
            max_batches=args.sweep_max_batches,
        )
        metrics['robustness_sweep'] = sweep
        print(f'Saved robustness sweep plot to: {sweep_path}')

    if args.save_pgd_iter_sweep:
        iter_sweep_path = args.pgd_iter_sweep_path or (args.checkpoint + '.pgd_iter_sweep.png')
        iters_list = parse_int_list(args.pgd_iters_list)
        iter_sweep = save_pgd_iter_sweep_plot(
            model=model,
            loader=test_loader,
            device=device,
            save_path=iter_sweep_path,
            epsilon=epsilon,
            alpha=alpha,
            iters_list=iters_list,
            max_batches=args.iter_sweep_max_batches,
        )
        metrics['pgd_iter_sweep'] = iter_sweep
        print(f'Saved PGD-iteration sweep plot to: {iter_sweep_path}')

    if args.metrics_path is not None:
        os.makedirs(os.path.dirname(args.metrics_path) or '.', exist_ok=True)
        with open(args.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f'Saved metrics summary to: {args.metrics_path}')


if __name__ == '__main__':
    main()
