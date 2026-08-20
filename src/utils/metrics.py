"""SOC estimation error metrics. All inputs/outputs in SOC fraction [0,1]
unless noted; report as % when printing, matching the papers' convention."""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse_pct": rmse(y_true, y_pred) * 100,
        "mae_pct": mae(y_true, y_pred) * 100,
        "mape_pct": mape(y_true, y_pred),
        "max_error_pct": max_error(y_true, y_pred) * 100,
    }
