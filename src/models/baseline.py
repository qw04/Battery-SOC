"""Baseline SOC estimators: SVR and Random Forest regression on engineered
V/I/T window features (§1.1 Alvarez Anton et al. 2013 SVM; §1.6 Lipu et al.
2023 optimized Random Forest -- see literature_review.md).

Unlike the neural models, these don't consume raw (window_len, 3) sequences --
they consume a small fixed-size feature vector summarizing each window
(last value + rolling mean/std/min/max per channel), which is the standard
way to feed sequence data into classical regressors.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: (N, window_len, n_features) -> (N, n_features * 5) summary features."""
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    vmin = X.min(axis=1)
    vmax = X.max(axis=1)
    return np.concatenate([last, mean, std, vmin, vmax], axis=1)


class BaselineRegressor:
    """Thin wrapper so baseline models share the fit(X_windows, y)/predict(X_windows)
    interface used by train.py / evaluate.py for the neural models."""

    def __init__(self, kind: str = "random_forest", **kwargs):
        self.kind = kind
        if kind == "svr":
            defaults = dict(kernel="rbf", C=10.0, epsilon=0.005, gamma="scale")
            defaults.update(kwargs)
            self.model = SVR(**defaults)
        elif kind == "random_forest":
            defaults = dict(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0)
            defaults.update(kwargs)
            self.model = RandomForestRegressor(**defaults)
        else:
            raise ValueError(f"unknown baseline kind: {kind}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(extract_features(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(extract_features(X)), 0.0, 1.0)
