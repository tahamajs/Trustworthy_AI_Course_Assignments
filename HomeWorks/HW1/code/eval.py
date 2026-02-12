import argparse
import torch
import torch.nn as nn
from datasets import get_dataloaders
from models.resnet18_custom import resnet18
from utils import load_checkpoint
import umap
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


def plot_umap(feats, labels, save_path=None):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    emb = reducer.fit_transform(feats)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb[:,0], emb[:,1], c=labels, cmap='tab10', s=4)
    plt.colorbar(scatter)
    plt.title('UMAP of features')
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='svhn')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--umap', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(args.dataset, batch_size=args.batch_size)
    model = resnet18(num_classes=num_classes, in_channels=in_channels)
    ck = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    feats, labels = extract_features(model, test_loader, device)
    if args.umap:
        plot_umap(feats, labels, save_path=args.checkpoint + '.umap.png')


if __name__ == '__main__':
    main()
