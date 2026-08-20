"""LSTM + UKF hybrid, per Yang et al. 2020 (literature_review.md §2.4): the
LSTM's SOC prediction is treated as a noisy *measurement* of the true SOC
state, and a UKF fuses it with a physics-based coulomb-counting *process*
model. This smooths out point-wise LSTM prediction noise using the fact that
SOC can't physically jump between consecutive windows.

State: x = [SOC] (scalar)
Process f(x, current):  coulomb-counting step (same physics as the ECM's SOC
                         equation in src/kalman/ecm_model.py, but without the
                         RC polarization branch -- we only need a SOC prior).
Measurement h(x, ...):  identity -- the "sensor" here is the LSTM itself, which
                         already outputs SOC directly, so its raw prediction IS
                         the measurement fed into the generic UKF.
"""
from __future__ import annotations

import numpy as np
import torch

from src.kalman.ukf import UnscentedKalmanFilter
from src.models.lstm_direct import LSTMDirect


def lstm_predict(model: LSTMDirect, X: np.ndarray, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    """Run the trained LSTM over all windows, return raw SOC predictions (N,)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def make_coulomb_counting_ukf(capacity_ah: float, dt: float, process_noise: float = 1e-6, measurement_noise: float = 1e-3):
    def f(x: np.ndarray, current: float) -> np.ndarray:
        soc = x[0] - (current * dt) / (3600.0 * capacity_ah)
        return np.array([np.clip(soc, 0.0, 1.0)])

    def h(x: np.ndarray, current: float) -> np.ndarray:
        return np.array([x[0]])  # identity -- LSTM output IS the measurement of SOC

    return UnscentedKalmanFilter(
        dim_x=1,
        dim_z=1,
        f=f,
        h=h,
        process_noise=np.array([[process_noise]]),
        measurement_noise=np.array([[measurement_noise]]),
    )


def filter_sequence(
    lstm_soc_preds: np.ndarray,
    currents: np.ndarray,
    capacity_ah: float,
    dt: float,
    x0: float | None = None,
    process_noise: float = 1e-5,
    measurement_noise: float = 3e-5,
) -> np.ndarray:
    """Run the UKF over one contiguous, chronologically-ordered sequence of
    per-window LSTM predictions for a single test cell/segment.

    `measurement_noise` defaults to (0.55%)^2, i.e. roughly the validation-set
    RMSE the direct LSTM (src/models/lstm_direct.py) achieves once trained --
    NOT the variance of the predictions themselves (SOC legitimately swings
    across the whole [0,1] range within a sequence, so using prediction
    variance as a noise proxy wildly overestimates how "noisy" the LSTM is and
    makes the UKF ignore it in favor of the process model almost entirely).
    """
    ukf = make_coulomb_counting_ukf(capacity_ah, dt, process_noise, measurement_noise)
    x0_val = float(lstm_soc_preds[0]) if x0 is None else x0
    estimates = ukf.run(
        controls=currents,
        measurements=lstm_soc_preds.reshape(-1, 1),
        x0=np.array([x0_val]),
        P0=np.array([[measurement_noise]]),
    )
    return estimates[:, 0]
