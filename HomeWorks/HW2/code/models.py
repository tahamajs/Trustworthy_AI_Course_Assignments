import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """MLP matching the assignment architecture (binary output - sigmoid).

    Layers:
    Linear(8->100) + BN + ReLU
    Linear(100->50) + ReLU + Dropout
    Linear(50->50) + ReLU
    Linear(50->20) + ReLU
    Linear(20->1) -> Sigmoid
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NAMClassifier(nn.Module):
    """Simple Neural Additive Model (one small subnetwork per feature).

    Each feature passes through a 2-layer MLP (shared activations), outputs are
    summed and passed through a sigmoid for binary classification.
    """

    def __init__(self, n_features=8, hidden=32):
        super().__init__()
        self.n_features = n_features
        self.subnets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_features)
        ])

    def forward(self, x):
        # x: (B, n_features)
        outs = [self.subnets[i](x[:, i : i + 1]) for i in range(self.n_features)]
        s = torch.stack(outs, dim=1).sum(1).squeeze(-1)
        return s
