"""PyTorch Dataset/DataLoader wrappers over the processed .npz window files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class SOCWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_npz(dataset: str) -> dict:
    path = PROCESSED_DIR / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m src.data.preprocess --dataset {dataset}` first")
    return dict(np.load(path))


def get_dataloaders(dataset: str, batch_size: int = 128, num_workers: int = 0):
    data = load_npz(dataset)
    train_ds = SOCWindowDataset(data["X_train"], data["y_train"])
    val_ds = SOCWindowDataset(data["X_val"], data["y_val"])
    test_ds = SOCWindowDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_combined_dataloaders(batch_size: int = 128, num_workers: int = 0):
    """Pool windows from both datasets. Each dataset keeps its own feature
    normalization (fit independently in preprocess.py) since NASA and Oxford
    cells differ in chemistry/capacity -- pooling happens after normalization."""
    nasa = load_npz("nasa")
    oxford = load_npz("oxford")

    def cat(key_x, key_y):
        X = np.concatenate([nasa[key_x], oxford[key_x]], axis=0)
        y = np.concatenate([nasa[key_y], oxford[key_y]], axis=0)
        return X, y

    X_train, y_train = cat("X_train", "y_train")
    X_val, y_val = cat("X_val", "y_val")
    X_test, y_test = cat("X_test", "y_test")

    train_loader = DataLoader(SOCWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(SOCWindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(SOCWindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
