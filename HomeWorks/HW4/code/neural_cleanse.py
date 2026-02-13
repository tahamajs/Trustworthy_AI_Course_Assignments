"""Neural Cleanse utilities for HW4 with real attacked checkpoints."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class TriggerResult:
    target_label: int
    mask: torch.Tensor
    pattern: torch.Tensor
    scale: float


class AttackedMNISTCNN(nn.Module):
    """Model architecture matching the provided poisoned checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(dim=-1),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Backwards-compatible alias used by earlier tests/scripts.
DemoConvNet = AttackedMNISTCNN


def _is_probability_output(outputs: torch.Tensor) -> bool:
    if outputs.ndim != 2:
        return False
    if torch.any(outputs < -1e-6):
        return False
    row_sums = outputs.sum(dim=1)
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)


def _classification_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if _is_probability_output(outputs):
        return F.nll_loss(torch.log(outputs.clamp_min(1e-12)), targets)
    return F.cross_entropy(outputs, targets)


def extract_poisoned_models_if_needed(archive_path: str, extract_dir: str) -> str:
    """Extract `poisened_models.rar` into `extract_dir` and return model directory path."""
    archive = Path(archive_path)
    out_dir = Path(extract_dir)
    if not archive.exists():
        raise FileNotFoundError(f"Poisoned-model archive not found: {archive}")

    model_dir = out_dir / "poisened_models"
    required = [model_dir / f"poisened_model_{i}.pth" for i in range(10)]
    if all(p.exists() for p in required):
        return str(model_dir)

    unar_bin = shutil.which("unar")
    if unar_bin is None:
        raise RuntimeError(
            "Could not find `unar` to extract `poisened_models.rar`. "
            "Install The Unarchiver/unar or extract manually."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [unar_bin, "-q", "-o", str(out_dir), str(archive)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Failed to extract poisoned models: {msg}") from exc

    if not all(p.exists() for p in required):
        raise RuntimeError(
            f"Extraction finished but expected checkpoints are missing under: {model_dir}"
        )
    return str(model_dir)


def resolve_checkpoint_path(
    student_id: str | int,
    model_index: Optional[int] = None,
    archive_path: Optional[str] = None,
    extract_dir: Optional[str] = None,
) -> str:
    """Resolve checkpoint path by explicit index or last student-id digit."""
    base = Path(__file__).resolve().parent
    archive = archive_path or str(base / "poisened_models.rar")
    extract = extract_dir or str(base / "model_weights")

    if model_index is None:
        sid = str(student_id).strip()
        if not sid or not sid[-1].isdigit():
            raise ValueError("student_id must end with a digit in [0, 9]")
        model_index = int(sid[-1])
    if model_index < 0 or model_index > 9:
        raise ValueError("model_index must be in [0, 9]")

    model_dir = Path(extract_poisoned_models_if_needed(archive, extract))
    path = model_dir / f"poisened_model_{model_index}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)


def load_model(
    path: Optional[str] = None,
    student_id: Optional[str | int] = None,
    model_index: Optional[int] = None,
    archive_path: Optional[str] = None,
    extract_dir: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """Load attacked model weights with explicit compatibility checks."""
    if path is None:
        if student_id is None and model_index is None:
            raise ValueError("Provide `path` or (`student_id`/`model_index`) to locate checkpoint.")
        sid = student_id if student_id is not None else "0"
        path = resolve_checkpoint_path(
            student_id=sid,
            model_index=model_index,
            archive_path=archive_path,
            extract_dir=extract_dir,
        )

    model = AttackedMNISTCNN().to(device)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {path}: {type(checkpoint)!r}")

    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with AttackedMNISTCNN: {path}\n{exc}"
        ) from exc

    model.eval()
    return model


def load_mnist_test(
    root: str,
    download: bool,
    batch_size: int,
    limit: Optional[int] = None,
    seed: int = 0,
) -> DataLoader:
    """Load MNIST test set with optional deterministic subset selection."""
    transform = transforms.ToTensor()
    try:
        dataset: Dataset = datasets.MNIST(root=root, train=False, transform=transform, download=download)
    except RuntimeError as exc:
        if download:
            raise RuntimeError(
                "MNIST download failed. This environment appears offline. "
                f"Download MNIST manually into `{root}` and rerun with `--no-download-mnist`."
            ) from exc
        raise RuntimeError(
            f"MNIST not found in `{root}`. Provide local files or rerun with `--download-mnist`."
        ) from exc

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[: min(limit, len(dataset))].tolist()
        dataset = Subset(dataset, indices)

    loader_gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=loader_gen)


def apply_trigger(x: torch.Tensor, mask: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """Apply reconstructed trigger `(mask, pattern)` to a batch of inputs."""
    m = mask.to(x.device)
    p = pattern.to(x.device)
    if m.dim() == 3:
        m = m.unsqueeze(0)
    if p.dim() == 3:
        p = p.unsqueeze(0)
    return (1.0 - m) * x + m * p


def reconstruct_trigger(
    model: nn.Module,
    dataloader: DataLoader,
    target_label: int,
    device: str = "cpu",
    steps: int = 500,
    lr: float = 0.1,
    lambda_l1: float = 0.01,
    lambda_pattern: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Reconstruct trigger mask/pattern for one target label."""
    model.eval()
    iterator = iter(dataloader)
    try:
        x0, _ = next(iterator)
    except StopIteration as exc:
        raise ValueError("dataloader is empty") from exc

    x0 = x0.to(device)
    _, channels, height, width = x0.shape
    mask_logits = torch.ones(1, 1, height, width, device=device, requires_grad=True)
    pattern_logits = torch.ones(1, channels, height, width, device=device, requires_grad=True)
    optimizer = optim.Adam([mask_logits, pattern_logits], lr=lr)

    for _ in range(steps):
        try:
            xb, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            xb, _ = next(iterator)
        xb = xb.to(device)

        optimizer.zero_grad()
        mask = torch.sigmoid(mask_logits)
        pattern = torch.sigmoid(pattern_logits)
        triggered = apply_trigger(xb, mask, pattern)
        outputs = model(triggered)
        targets = torch.full((outputs.size(0),), target_label, dtype=torch.long, device=device)
        cls = _classification_loss(outputs, targets)
        reg = lambda_l1 * mask.abs().sum() + lambda_pattern * pattern.abs().sum()
        loss = cls + reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mask = torch.sigmoid(mask_logits).detach().cpu()
        pattern = torch.sigmoid(pattern_logits).detach().cpu()
        scale = float(mask.abs().sum().item())
    return mask, pattern, scale


def reconstruct_all_labels(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int = 10,
    device: str = "cpu",
    steps: int = 500,
    lr: float = 0.1,
    lambda_l1: float = 0.01,
    lambda_pattern: float = 1e-3,
) -> Dict[int, TriggerResult]:
    results: Dict[int, TriggerResult] = {}
    for label in range(num_classes):
        mask, pattern, scale = reconstruct_trigger(
            model=model,
            dataloader=dataloader,
            target_label=label,
            device=device,
            steps=steps,
            lr=lr,
            lambda_l1=lambda_l1,
            lambda_pattern=lambda_pattern,
        )
        results[label] = TriggerResult(
            target_label=label,
            mask=mask,
            pattern=pattern,
            scale=scale,
        )
    return results


def detect_outlier_scales(scales: List[float], threshold_mult: float = 3.5) -> int:
    """MAD lower-tail outlier detector (smallest anomalous trigger scale)."""
    arr = np.asarray(scales, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("scales must be a non-empty 1D list/array")
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return int(np.argmin(arr))
    modified_z = 0.6745 * (arr - med) / mad
    idx = int(np.argmin(modified_z))
    if modified_z[idx] > -threshold_mult:
        return int(np.argmin(arr))
    return idx


def evaluate_clean_accuracy(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.numel())
    return float(correct / total) if total > 0 else 0.0


def evaluate_asr(
    model: nn.Module,
    dataloader: DataLoader,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    target_label: int,
    device: str = "cpu",
) -> float:
    model.eval()
    total = 0
    success = 0
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            triggered = trigger_fn(xb)
            preds = model(triggered).argmax(dim=1)
            success += int((preds == target_label).sum().item())
            total += int(preds.numel())
    return float(success / total) if total > 0 else 0.0


def unlearn_by_retraining(
    model: nn.Module,
    dataset: Dataset,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    fraction: float = 0.2,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 0,
) -> nn.Module:
    """Unlearning pass with triggered inputs and original labels preserved."""
    if fraction < 0 or fraction > 1:
        raise ValueError("fraction must be in [0, 1]")
    model.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_to_trigger = int(math.ceil(len(dataset) * fraction))
    triggered_so_far = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb_in = xb.clone()

            if triggered_so_far < n_to_trigger:
                take = min(xb_in.size(0), n_to_trigger - triggered_so_far)
                xb_in[:take] = trigger_fn(xb_in[:take])
                triggered_so_far += take

            optimizer.zero_grad()
            outputs = model(xb_in)
            loss = _classification_loss(outputs, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model
