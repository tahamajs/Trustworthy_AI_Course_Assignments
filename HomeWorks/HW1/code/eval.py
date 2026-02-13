import argparse
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
    p.add_argument('--demo', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, test_loader, num_classes, in_channels = get_dataloaders(args.dataset, batch_size=args.batch_size, demo=args.demo)
    model = resnet18(num_classes=num_classes, in_channels=in_channels)
    ck = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    epsilon = str_to_ratio(args.epsilon)
    alpha = str_to_ratio(args.alpha)

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


if __name__ == '__main__':
    main()
