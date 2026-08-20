"""Fit an OCV(SOC) curve and 1RC ECM parameters (R0, R1, C1) from the Oxford
Battery Degradation Dataset's pseudo-OCV and 1C discharge cycles, so the
standalone linear-KF / EKF / UKF filters in src/kalman/ have a concrete
physical model to run on Oxford's test cells.

We use Oxford (not NASA) for the ECM/Kalman-filter path because Oxford is the
only one of the two datasets with dedicated low-current pseudo-OCV cycles --
the clean way to measure OCV(SOC) directly. NASA's randomized dataset has no
equivalent low-current characterization step, so a from-scratch OCV curve for
it would be far less reliable; see README.md for the discussion of this
scoping choice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.kalman.ecm_model import ECMParams, OCVCurve, ThreveninECM
from src.data import splits

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_ocv_curve(exclude_test_cells: bool = True) -> OCVCurve:
    """Fit OCV(SOC) from the OCVdc (pseudo-OCV discharge) cycles of the training cells."""
    path = PROCESSED_DIR / "oxford_ocv_cycles.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m src.data.preprocess --dataset oxford` first")
    df = pd.read_csv(path)
    df = df[df["phase"].str.lower() == "ocvdc"]
    if exclude_test_cells:
        df = df[~df["cell_id"].isin(splits.OXFORD_TEST_CELLS)]

    soc_all, v_all = [], []
    for (_cell, _cp), g in df.groupby(["cell_id", "checkpoint"]):
        g = g.sort_values("t")
        q = g["charge_mah"].to_numpy()
        q_total = np.nanmax(np.abs(q))
        if q_total < 1e-6:
            continue
        soc = np.clip(1.0 - np.abs(q) / q_total, 0.0, 1.0)  # OCVdc is a discharge: starts full
        soc_all.append(soc)
        v_all.append(g["voltage"].to_numpy())

    soc_all = np.concatenate(soc_all)
    v_all = np.concatenate(v_all)
    return OCVCurve(soc_all, v_all, degree=6)


def fit_ecm_params(
    ocv_curve: OCVCurve,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    soc_true: np.ndarray,
    dt: float,
    capacity_ah: float,
    r1_default: float = 0.03,
    c1_default: float = 2000.0,
) -> ECMParams:
    """Fit R0 from data; hold R1/C1 at typical small-Li-ion-pouch-cell defaults.

    A single full constant-current charge/discharge cycle (as opposed to a
    pulse/HPPC test with short current steps + relaxation) only very weakly
    constrains a 2-parameter RC branch -- R0 and R1*C1 trade off against each
    other almost interchangeably when the only signal is one slow monotonic
    discharge, so a free 3-parameter least-squares fit is poorly identified
    and drifts to whichever bound it's given (verified empirically: it pins
    to different degenerate corners depending on the bounds used). Since
    Oxford's dataset has no pulse test to identify R1/C1 properly, we fix
    them at literature-typical values for a small pouch cell (tau = R1*C1 of
    ~1 minute) and only fit the one parameter (R0, the instantaneous ohmic
    drop) that IS well identified by a CC discharge: it's linear in the
    residual after removing the (fixed) RC branch's contribution.
    """
    r1, c1 = r1_default, c1_default
    alpha = np.exp(-dt / (r1 * c1))
    v1 = np.empty(len(current_a))
    v1_prev = 0.0
    for k in range(len(current_a)):
        v1_prev = alpha * v1_prev + r1 * (1 - alpha) * current_a[k]
        v1[k] = v1_prev

    ocv = ocv_curve.voltage_at(soc_true)
    residual = ocv - v1 - voltage_v  # should equal current_a * r0
    # closed-form weighted least squares for the single scalar r0
    r0 = float(np.sum(residual * current_a) / np.sum(current_a ** 2))
    r0 = float(np.clip(r0, 1e-4, 1.0))

    return ECMParams(r0=r0, r1=r1, c1=c1, capacity_ah=capacity_ah, dt=dt)


def build_ecm_from_train_cells(dt: float = 20.0) -> ThreveninECM:
    """Convenience: fit the OCV curve + ECM params entirely from Oxford training
    cells (never touching test cells), ready to hand to linear_kf/ekf/ukf."""
    ocv_curve = load_ocv_curve(exclude_test_cells=True)

    # use one C1dc (1C discharge) cycle from a training cell to fit R0/R1/C1
    from src.data import parse_oxford, soc_labeling

    raw = parse_oxford.parse_main_file()
    labeled = soc_labeling.label_oxford_soc(raw)
    train_cell = sorted(set(labeled["cell_id"]) - splits.OXFORD_TEST_CELLS)[0]
    g = labeled[(labeled["cell_id"] == train_cell) & (labeled["phase"].str.lower() == "c1dc")]
    checkpoint = sorted(g["checkpoint"].unique())[0]
    g = g[g["checkpoint"] == checkpoint].sort_values("t")

    t = g["t"].to_numpy()
    grid = np.arange(t[0], t[-1], dt)
    # "1C" discharge means current (A) numerically equals capacity (Ah); use that
    # directly rather than differentiating the noisy cumulative charge signal.
    capacity_ah = g["capacity_mah"].iloc[0] / 1000.0
    current_a = np.full_like(grid, capacity_ah)
    voltage_v = np.interp(grid, t, g["voltage"].to_numpy())
    soc_true = np.interp(grid, t, g["soc"].to_numpy())

    params = fit_ecm_params(ocv_curve, current_a, voltage_v, soc_true, dt, capacity_ah)
    return ThreveninECM(params, ocv_curve)


if __name__ == "__main__":
    ecm = build_ecm_from_train_cells()
    print(f"fitted ECM: R0={ecm.p.r0:.4f} R1={ecm.p.r1:.4f} C1={ecm.p.c1:.1f} capacity_ah={ecm.p.capacity_ah:.3f}")
