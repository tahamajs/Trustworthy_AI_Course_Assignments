import os
import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, is_best: bool, save_dir: str, filename: str = 'last.pth'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(save_dir, 'best.pth'))


def load_checkpoint(path: str, device='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)
