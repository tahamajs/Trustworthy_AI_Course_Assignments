from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_cleanse import (
    AttackedMNISTCNN,
    detect_outlier_scales,
    evaluate_asr,
    load_model,
    reconstruct_trigger,
    resolve_checkpoint_path,
)


def test_reconstruct_trigger_shape():
    model = AttackedMNISTCNN()
    X = torch.rand(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    m, p, s = reconstruct_trigger(model, loader, target_label=3, steps=20, lr=0.3)
    assert m.shape[2:] == (28, 28)
    assert p.shape[1:] == (1, 28, 28)
    assert isinstance(s, float)


def test_detect_outlier_scales_lower_tail():
    scales = [4.9, 5.1, 4.7, 0.2, 5.0]
    idx = detect_outlier_scales(scales)
    assert idx == 3


def test_load_model_checkpoint_compatibility(tmp_path):
    archive = Path(__file__).resolve().parents[1] / "poisened_models.rar"
    ckpt = resolve_checkpoint_path(
        student_id="810101504",
        archive_path=str(archive),
        extract_dir=str(tmp_path),
    )
    model = load_model(path=ckpt, device="cpu")
    out = model(torch.rand(2, 1, 28, 28))
    assert out.shape == (2, 10)


class _AlwaysTargetModel(nn.Module):
    def __init__(self, target_label: int = 3):
        super().__init__()
        self.target_label = target_label

    def forward(self, x):
        logits = torch.zeros(x.size(0), 10, device=x.device)
        logits[:, self.target_label] = 1.0
        return logits


def test_evaluate_asr_helper():
    X = torch.rand(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    model = _AlwaysTargetModel(target_label=4)

    def identity_trigger(x):
        return x

    asr = evaluate_asr(model, loader, identity_trigger, target_label=4)
    assert abs(asr - 1.0) < 1e-12
