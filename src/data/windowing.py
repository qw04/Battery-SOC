"""Resample labeled time series onto a fixed timestep and build sliding-window
(V, I, T) -> SOC[-1] sequences, the standard seq-to-one input format used by
the LSTM / CNN-U-Net / Conv-ULSAM-SRU models (matches §2.1/§2.4/§3.2/§3.3 in
literature_review.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = ["voltage", "current", "temperature"]


def resample_uniform(df: pd.DataFrame, dt: float, value_cols: list[str]) -> pd.DataFrame:
    """Resample one contiguous (single cell/segment) time series onto a fixed dt grid via linear interp."""
    df = df.sort_values("t").drop_duplicates(subset="t")
    if len(df) < 2:
        return df.iloc[0:0]
    t0, t1 = df["t"].iloc[0], df["t"].iloc[-1]
    if t1 <= t0:
        return df.iloc[0:0]
    grid = np.arange(t0, t1, dt)
    out = {"t": grid}
    for col in value_cols:
        out[col] = np.interp(grid, df["t"].to_numpy(), df[col].to_numpy())
    return pd.DataFrame(out)


def make_windows(
    df: pd.DataFrame,
    window_len: int,
    stride: int,
    group_cols: list[str],
    dt: float,
    feature_cols: list[str] = FEATURE_COLS,
    label_col: str = "soc",
    confidence_col: str | None = None,
    max_confidence_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build sliding windows within each group (e.g. one cell, or one cell+checkpoint+phase).

    If `confidence_col` is given (e.g. NASA's `hours_since_anchor`, see
    soc_labeling.label_nasa_soc), a window is only kept if that column's value
    at the window's LAST timestep (the one the label comes from) is <=
    `max_confidence_value` -- used to discard windows whose SOC label may have
    drifted too far from the nearest hard anchor to be trustworthy.

    Returns:
      X: (N, window_len, n_features)
      y: (N,)  -- label at the last timestep of each window
      groups: list of the group-key tuple each window came from (for leave-cells-out splitting)
    """
    X_list, y_list, group_list = [], [], []
    value_cols = feature_cols + [label_col] + ([confidence_col] if confidence_col else [])

    for key, g in df.groupby(group_cols, sort=False):
        g_resampled = resample_uniform(g, dt, value_cols)
        if len(g_resampled) < window_len:
            continue
        feats = g_resampled[feature_cols].to_numpy()
        labels = g_resampled[label_col].to_numpy()
        confidence = g_resampled[confidence_col].to_numpy() if confidence_col else None
        n = len(g_resampled)
        for start in range(0, n - window_len + 1, stride):
            end = start + window_len
            if confidence is not None and confidence[end - 1] > max_confidence_value:
                continue
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])
            group_list.append(key if isinstance(key, tuple) else (key,))

    if not X_list:
        return (
            np.empty((0, window_len, len(feature_cols))),
            np.empty((0,)),
            [],
        )
    return np.stack(X_list), np.array(y_list), group_list


def normalize_features(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Per-feature z-score normalization. Fit (mean/std) on train, reuse on val/test."""
    if mean is None:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    if std is None:
        std = X.reshape(-1, X.shape[-1]).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std
