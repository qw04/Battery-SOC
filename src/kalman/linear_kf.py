"""Classic (fully linear) Kalman filter for SOC estimation.

Uses the same 1RC ECM state transition as the EKF/UKF (which is already linear
in the state), but linearizes the measurement equation ONCE around a fixed
operating point (mid-charge) instead of relinearizing at every step the way
the EKF does. This is the simplest, least accurate of the three filters here --
included as the textbook baseline.
"""
from __future__ import annotations

import numpy as np

from src.kalman.ecm_model import ThreveninECM


class LinearKalmanFilter:
    def __init__(
        self,
        ecm: ThreveninECM,
        process_noise: np.ndarray | None = None,
        measurement_noise: float = 1e-3,
        linearize_at_soc: float = 0.5,
    ):
        self.ecm = ecm
        self.Q = process_noise if process_noise is not None else np.diag([1e-6, 1e-5])
        self.R = np.array([[measurement_noise]])
        # Fixed measurement Jacobian, linearized once at a representative SOC.
        self.x_lin = np.array([linearize_at_soc, 0.0])
        self.H = self.ecm.H_jacobian(self.x_lin, current=0.0)
        self.A = self.ecm.F_jacobian(self.x_lin, current=0.0)

    def run(self, currents: np.ndarray, voltages: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        """Run the filter over a sequence. Returns array of state estimates (N, 2)."""
        n = len(currents)
        x = x0.copy()
        P = P0.copy()
        estimates = np.zeros((n, 2))

        for k in range(n):
            i_k = currents[k]

            # --- predict ---
            x_pred = self.ecm.f(x, i_k)
            P_pred = self.A @ P @ self.A.T + self.Q

            # --- update ---
            # h linearized once around self.x_lin (NOT re-linearized each step, unlike the EKF).
            h0 = self.ecm.h(self.x_lin, i_k)
            y_pred = h0 + (self.H @ (x_pred - self.x_lin)).item()
            innovation = voltages[k] - y_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            x = x_pred + (K.flatten() * innovation)
            x[0] = np.clip(x[0], 0.0, 1.0)
            P = (np.eye(2) - K @ self.H) @ P_pred

            estimates[k] = x

        return estimates
