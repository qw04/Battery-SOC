"""CLI entry point to train any model on any dataset.

Usage:
  python -m src.train --model baseline_svr --dataset oxford
  python -m src.train --model baseline_rf --dataset oxford
  python -m src.train --model lstm --dataset nasa
  python -m src.train --model cnn_unet --dataset both
  python -m src.train --model conv_ulsam_sru --dataset nasa

Note: "lstm_ukf" is not trained here -- it reuses a checkpoint already trained
via `--model lstm` and wraps it with the UKF at evaluation time (see
src/evaluate.py and src/models/lstm_ukf.py).
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import get_combined_dataloaders, get_dataloaders, load_npz
from src.models.baseline import BaselineRegressor
from src.models.cnn_unet import CNNUNet, total_variation_loss
from src.models.conv_ulsam_sru import ConvULSAMSRU
from src.models.lstm_direct import LSTMDirect
from src.utils.metrics import all_metrics
from src.utils.seed import set_seed

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"

NEURAL_MODELS = {
    "lstm": lambda: LSTMDirect(),
    "cnn_unet": lambda: CNNUNet(),
    "conv_ulsam_sru": lambda: ConvULSAMSRU(),
}
BASELINE_MODELS = {
    "baseline_svr": "svr",
    "baseline_rf": "random_forest",
}


def _get_loaders(dataset: str, batch_size: int):
    if dataset == "both":
        return get_combined_dataloaders(batch_size=batch_size)
    return get_dataloaders(dataset, batch_size=batch_size)


def train_neural(model_name: str, dataset: str, epochs: int, lr: float, batch_size: int, device: str, patience: int = 10):
    set_seed(0)
    train_loader, val_loader, test_loader = _get_loaders(dataset, batch_size)

    model = NEURAL_MODELS[model_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            if model_name == "cnn_unet":
                seq = model(X, return_sequence=True)
                pred = seq[:, -1]
                loss = loss_fn(pred, y) + 0.1 * total_variation_loss(seq)
            else:
                pred = model(X)
                loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        val_rmse = _evaluate_neural(model, val_loader, device, model_name)["rmse_pct"]
        print(f"[{model_name}/{dataset}] epoch {epoch:3d}  train_loss={np.mean(train_losses):.5f}  val_rmse={val_rmse:.3f}%")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"[{model_name}/{dataset}] early stopping at epoch {epoch} (best val_rmse={best_val:.3f}%)")
                break

    model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{model_name}/{dataset}] saved checkpoint -> {ckpt_path}")

    test_metrics = _evaluate_neural(model, test_loader, device, model_name)
    print(f"[{model_name}/{dataset}] TEST metrics: {test_metrics}")
    return test_metrics


@torch.no_grad()
def _evaluate_neural(model, loader, device, model_name) -> dict:
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.numpy())
    if not preds:
        return {"rmse_pct": float("nan"), "mae_pct": float("nan"), "mape_pct": float("nan"), "max_error_pct": float("nan")}
    return all_metrics(np.concatenate(trues), np.concatenate(preds))


SVR_MAX_TRAIN_SAMPLES = 20_000  # SVR's RBF kernel is O(n^2)-O(n^3); NASA's ~160k-row train set never finishes


def train_baseline(model_key: str, dataset: str):
    set_seed(0)
    data = load_npz(dataset) if dataset != "both" else _combine_npz()
    kind = BASELINE_MODELS[model_key]
    model = BaselineRegressor(kind=kind)

    X_train, y_train = data["X_train"], data["y_train"]
    if kind == "svr" and len(X_train) > SVR_MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_train), SVR_MAX_TRAIN_SAMPLES, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"[{model_key}/{dataset}] subsampled train set {len(data['X_train'])} -> {len(X_train)} rows for SVR")

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[{model_key}/{dataset}] fit in {time.time() - t0:.1f}s")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / f"{model_key}_{dataset}.pkl", "wb") as f:
        pickle.dump(model, f)

    preds = model.predict(data["X_test"])
    test_metrics = all_metrics(data["y_test"], preds)
    print(f"[{model_key}/{dataset}] TEST metrics: {test_metrics}")
    return test_metrics


def _combine_npz():
    nasa = load_npz("nasa")
    oxford = load_npz("oxford")
    out = {}
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = np.concatenate([nasa[f"X_{split}"], oxford[f"X_{split}"]], axis=0)
        out[f"y_{split}"] = np.concatenate([nasa[f"y_{split}"], oxford[f"y_{split}"]], axis=0)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(NEURAL_MODELS) + list(BASELINE_MODELS))
    parser.add_argument("--dataset", required=True, choices=["nasa", "oxford", "both"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.model in BASELINE_MODELS:
        train_baseline(args.model, args.dataset)
    else:
        train_neural(args.model, args.dataset, args.epochs, args.lr, args.batch_size, args.device)
