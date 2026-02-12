import torch
from neural_cleanse import DemoConvNet, reconstruct_trigger, detect_outlier_scales
from torch.utils.data import TensorDataset, DataLoader


def test_reconstruct_trigger_shape():
    model = DemoConvNet()
    X = torch.rand(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    m, p, s = reconstruct_trigger(model, loader, target_label=3, steps=20, lr=0.3)
    assert m.shape[2:] == (28, 28)
    assert p.shape[1:] == (1, 28, 28)
    assert isinstance(s, float)


def test_detect_outlier_scales():
    scales = [0.5, 0.6, 0.55, 5.0, 0.52]
    idx = detect_outlier_scales(scales)
    assert idx == 3
