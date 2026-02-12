"""Neural Cleanse (Q1) — reconstruction + MAD detector + unlearning scaffold.

Notes:
- If you provide an attacked model checkpoint, place it in `code/model_weights/` and pass its path to `load_model()`.
- A demo mode creates a small random CNN so you can run reconstruction without external weights.
- The reconstruction follows the Neural Cleanse idea: optimize a small `mask` and `pattern` such that when
  applied to inputs the model predicts a target label. We add L1 penalty on mask and a regularizer on pattern.

API:
- reconstruct_trigger(model, dataloader, target_label, device='cpu', steps=1000, lr=0.1)
- detect_outlier_scales(scales)  # MAD-based detection of attacked label
- unlearn_by_retraining(model, dataset, trigger_fn, fraction=0.2, epochs=1)  # simple unlearning epoch
"""
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DemoConvNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(path: Optional[str] = None, demo: bool = True, device: str = "cpu") -> nn.Module:
    if path is None and demo:
        return DemoConvNet().to(device)
    # placeholder: load your own checkpoint here
    model = DemoConvNet().to(device)
    if path is not None:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    return model


def reconstruct_trigger(model: nn.Module, dataloader: DataLoader, target_label: int, device: str = "cpu", steps: int = 500, lr: float = 0.1, lambda_l1: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Reconstruct mask and pattern for a given `target_label`.

    Returns (mask, pattern, final_loss_scale) where mask in [0,1] and pattern same shape as input
    """
    model.eval()
    # infer input shape from a batch
    x0, _ = next(iter(dataloader))
    x0 = x0.to(device)
    bs, C, H, W = x0.shape

    mask = torch.randn(1, 1, H, W, device=device, requires_grad=True)
    pattern = torch.randn(1, C, H, W, device=device, requires_grad=True)

    optimizer = optim.Adam([mask, pattern], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        m = torch.sigmoid(mask)
        p = torch.tanh(pattern)
        # apply trigger to a small random subset
        rng_idx = torch.randperm(bs)[: max(1, bs // 4)]
        x = x0[rng_idx]
        x_triggered = (1 - m) * x + m * p
        logits = model(x_triggered)
        target = torch.full((logits.size(0),), target_label, dtype=torch.long, device=device)
        ce = F.cross_entropy(logits, target)
        reg = lambda_l1 * m.abs().sum()
        # encourage small pattern magnitude
        pat_reg = 1e-3 * p.abs().sum()
        loss = ce + reg + pat_reg
        loss.backward()
        optimizer.step()
    # final scale (use L1 norm of mask as proxy for 'anomaly scale')
    final_scale = float(m.detach().abs().sum().cpu().numpy())
    return m.detach().cpu(), p.detach().cpu(), final_scale


def detect_outlier_scales(scales: List[float], threshold_mult: float = 3.5) -> int:
    """MAD-based outlier detection. Returns index of outlier (attacked label).

    Following Neural Cleanse, compute MAD and find label with smallest scale / outlier.
    """
    arr = np.array(scales)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        # fallback: return argmin
        return int(np.argmin(arr))
    modified_z = 0.6745 * (arr - med) / mad
    # attacked label is the one with a large negative modified_z (very small scale)
    return int(np.argmin(modified_z))


def unlearn_by_retraining(model: nn.Module, dataset: TensorDataset, trigger_fn, fraction: float = 0.2, epochs: int = 1, lr: float = 1e-3, device: str = "cpu") -> nn.Module:
    """Simple unlearning: apply trigger to `fraction` of training data with target label and train for one epoch

    `trigger_fn(x)` should return x_triggered and label y_triggered.
    """
    model.train()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_apply = int(len(dataset) * fraction)
    applied = 0
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # apply trigger to some fraction of the batch until we've applied n_apply
            if applied < n_apply:
                take = min(n_apply - applied, xb.size(0))
                xb_triggered, y_trigger = trigger_fn(xb[:take], yb[:take])
                xb[:take] = xb_triggered
                yb[:take] = y_trigger
                applied += take
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":
    # demo: reconstruct a trigger for each label using random inputs
    import os
    import torchvision.transforms as T
    from torchvision.datasets import MNIST

    device = "cpu"
    model = load_model(demo=True, device=device)
    # create a tiny fake MNIST loader from random noise for demo
    X = torch.rand(200, 1, 28, 28)
    y = torch.randint(0, 10, (200,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32)
    scales = []
    for t in range(10):
        _, _, s = reconstruct_trigger(model, loader, t, device=device, steps=100, lr=0.2)
        scales.append(s)
    attacked = detect_outlier_scales(scales)
    print("Scales:", scales)
    print("Detected attacked label (demo):", attacked)
