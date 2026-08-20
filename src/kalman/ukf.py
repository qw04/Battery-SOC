"""Generic sigma-point (Van der Merwe scaled) Unscented Kalman Filter.

Takes arbitrary f(x, u) -> x_next and h(x, u) -> z callables, so it works both:
  - standalone over the analytic 1RC ECM (ecm_model.ThreveninECM.f / .h), and
  - wrapped around a trained LSTM as the measurement function (src/models/lstm_ukf.py),
    reproducing the "LSTM as measurement equation inside a UKF" idea from the
    2020 Yang et al. LSTM+UKF paper (see literature_review.md §2.4).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class UnscentedKalmanFilter:
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        f: Callable[[np.ndarray, float], np.ndarray],
        h: Callable[[np.ndarray, float], np.ndarray],
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.n = dim_x
        self.dim_z = dim_z
        self.f = f
        self.h = h
        self.Q = process_noise
        self.R = measurement_noise

        self.lam = alpha ** 2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)

        # sigma-point weights
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - alpha ** 2 + beta)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        P = P + np.eye(self.n) * 1e-12  # numerical floor, keeps cholesky stable
        sqrt_P = np.linalg.cholesky(P)
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i + 1] = x + self.gamma * sqrt_P[:, i]
            sigmas[self.n + i + 1] = x - self.gamma * sqrt_P[:, i]
        return sigmas

    def step(self, x: np.ndarray, P: np.ndarray, u: float, z: float):
        """One predict+update cycle. Returns (x_new, P_new)."""
        sigmas = self._sigma_points(x, P)

        # --- predict ---
        sigmas_f = np.array([self.f(s, u) for s in sigmas])
        x_pred = self.Wm @ sigmas_f
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            dx = sigmas_f[i] - x_pred
            P_pred += self.Wc[i] * np.outer(dx, dx)

        # --- update ---
        sigmas_z = np.array([self.h(s, u) for s in sigmas_f]).reshape(2 * self.n + 1, self.dim_z)
        z_pred = self.Wm @ sigmas_z

        P_zz = self.R.copy()
        P_xz = np.zeros((self.n, self.dim_z))
        for i in range(2 * self.n + 1):
            dz = sigmas_z[i] - z_pred
            dx = sigmas_f[i] - x_pred
            P_zz += self.Wc[i] * np.outer(dz, dz)
            P_xz += self.Wc[i] * np.outer(dx, dz)

        K = P_xz @ np.linalg.inv(P_zz)
        innovation = np.atleast_1d(z) - z_pred
        x_new = x_pred + K @ innovation
        P_new = P_pred - K @ P_zz @ K.T

        return x_new, P_new

    def run(self, controls: np.ndarray, measurements: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        n_steps = len(controls)
        x = x0.copy()
        P = P0.copy()
        estimates = np.zeros((n_steps, self.n))
        for k in range(n_steps):
            x, P = self.step(x, P, controls[k], measurements[k])
            x[0] = np.clip(x[0], 0.0, 1.0)  # SOC is always state index 0
            estimates[k] = x
        return estimates
