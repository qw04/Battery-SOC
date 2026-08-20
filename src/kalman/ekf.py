"""Extended Kalman Filter (EKF) for SOC estimation over the 1RC ECM.

Unlike the plain linear KF, this relinearizes F and H at every timestep
around the current state estimate, which handles the nonlinear OCV(SOC)
curve far more accurately.
"""
from __future__ import annotations

import numpy as np

from src.kalman.ecm_model import ThreveninECM


class ExtendedKalmanFilter:
    def __init__(
        self,
        ecm: ThreveninECM,
        process_noise: np.ndarray | None = None,
        measurement_noise: float = 1e-3,
    ):
        self.ecm = ecm
        self.Q = process_noise if process_noise is not None else np.diag([1e-6, 1e-5])
        self.R = np.array([[measurement_noise]])

    def run(self, currents: np.ndarray, voltages: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        n = len(currents)
        x = x0.copy()
        P = P0.copy()
        estimates = np.zeros((n, 2))

        for k in range(n):
            i_k = currents[k]

            # --- predict ---
            F = self.ecm.F_jacobian(x, i_k)
            x_pred = self.ecm.f(x, i_k)
            P_pred = F @ P @ F.T + self.Q

            # --- update (relinearize H at the predicted state) ---
            H = self.ecm.H_jacobian(x_pred, i_k)
            y_pred = self.ecm.h(x_pred, i_k)
            innovation = voltages[k] - y_pred
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + (K.flatten() * innovation)
            x[0] = np.clip(x[0], 0.0, 1.0)
            P = (np.eye(2) - K @ H) @ P_pred

            estimates[k] = x

        return estimates
