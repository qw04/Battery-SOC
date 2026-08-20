"""Consolidated evaluation: runs every trained model + the standalone Kalman
filters on their held-out test cells, writes results/metrics.csv, and saves
predicted-vs-true SOC plots to results/plots/.

Usage:
  python -m src.evaluate                 # evaluate everything that has a checkpoint / is runnable
  python -m src.evaluate --skip-kalman   # skip the standalone ECM+KF/EKF/UKF path (Oxford-only, slower to set up)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import preprocess
from src.data.dataset import load_npz
from src.models.baseline import BaselineRegressor  # noqa: F401 (needed to unpickle)
from src.models.cnn_unet import CNNUNet
from src.models.conv_ulsam_sru import ConvULSAMSRU
from src.models.lstm_direct import LSTMDirect
from src.models.lstm_ukf import filter_sequence, lstm_predict
from src.train import BASELINE_MODELS, CKPT_DIR, NEURAL_MODELS
from src.utils.metrics import all_metrics

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

NOMINAL_CAPACITY_AH = {"nasa": 2.0, "oxford": 0.74, "both": 2.0}
# lstm_ukf's UKF steps once per WINDOW, not once per raw sample -- so its process
# model's dt must be (stride * per-sample dt), the actual time between one
# window's label and the next's, not preprocess.py's raw resampling dt. Using
# the raw per-sample dt here under-integrates coulomb counting by a factor of
# `stride` and was an earlier bug that made the "filtered" output far worse
# than the raw LSTM instead of smoothing it.
DT_SECONDS = {
    "nasa": preprocess.NASA_DT * preprocess.NASA_STRIDE,
    "oxford": preprocess.OXFORD_DT * preprocess.OXFORD_STRIDE,
    "both": preprocess.NASA_DT * preprocess.NASA_STRIDE,
}

rows: list[dict] = []


def _load_data(dataset: str) -> dict:
    """Like load_npz, but also handles 'both' by concatenating the nasa/oxford
    .npz files on the fly (there's no separate 'both.npz' on disk -- train.py
    pools them the same way via _combine_npz)."""
    if dataset != "both":
        return load_npz(dataset)

    nasa, oxford = load_npz("nasa"), load_npz("oxford")
    out = {}
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = np.concatenate([nasa[f"X_{split}"], oxford[f"X_{split}"]], axis=0)
        out[f"y_{split}"] = np.concatenate([nasa[f"y_{split}"], oxford[f"y_{split}"]], axis=0)
    out["X_test_raw"] = np.concatenate([nasa["X_test_raw"], oxford["X_test_raw"]], axis=0)
    out["groups_test"] = np.concatenate(
        [np.char.add("nasa::", nasa["groups_test"]), np.char.add("oxford::", oxford["groups_test"])]
    )
    return out


def _save_plot(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_true * 100, label="true SOC", linewidth=1.5)
    ax.plot(y_pred * 100, label="predicted SOC", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("window index (chronological within test cell)")
    ax.set_ylabel("SOC (%)")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def evaluate_neural(model_name: str, dataset: str, device: str):
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset}.pt"
    if not ckpt_path.exists():
        return
    data = _load_data(dataset)
    model = NEURAL_MODELS[model_name]().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds = lstm_predict(model, data["X_test"], device=device)  # batched forward pass; works for any model, not just LSTM
    metrics = all_metrics(data["y_test"], preds)
    rows.append({"model": model_name, "dataset": dataset, **metrics})
    print(f"[{model_name}/{dataset}] {metrics}")
    if len(preds) > 0:
        _save_plot(f"{model_name}_{dataset}", data["y_test"], preds)


def evaluate_baseline(model_key: str, dataset: str):
    ckpt_path = CKPT_DIR / f"{model_key}_{dataset}.pkl"
    if not ckpt_path.exists():
        return
    data = _load_data(dataset)
    with open(ckpt_path, "rb") as f:
        model: BaselineRegressor = pickle.load(f)
    preds = model.predict(data["X_test"])
    metrics = all_metrics(data["y_test"], preds)
    rows.append({"model": model_key, "dataset": dataset, **metrics})
    print(f"[{model_key}/{dataset}] {metrics}")
    if len(preds) > 0:
        _save_plot(f"{model_key}_{dataset}", data["y_test"], preds)


def evaluate_lstm_ukf(dataset: str, device: str):
    ckpt_path = CKPT_DIR / f"lstm_{dataset}.pt"
    if not ckpt_path.exists():
        return
    data = _load_data(dataset)
    if "groups_test" not in data or len(data["groups_test"]) == 0:
        return

    model = LSTMDirect().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    X_norm = data["X_test"]
    X_raw = data["X_test_raw"]
    y = data["y_test"]
    groups = data["groups_test"]

    lstm_preds_all = lstm_predict(model, X_norm, device=device)

    all_true, all_filtered = [], []
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]  # windows already chronological within a group (see windowing.py)
        currents = X_raw[idx][:, -1, 1]  # last-timestep current (A) of each window; feature order = [V, I, T]
        filtered = filter_sequence(
            lstm_soc_preds=lstm_preds_all[idx],
            currents=currents,
            capacity_ah=NOMINAL_CAPACITY_AH[dataset],
            dt=DT_SECONDS[dataset],
        )
        all_true.append(y[idx])
        all_filtered.append(filtered)

    y_true = np.concatenate(all_true)
    y_filtered = np.concatenate(all_filtered)
    metrics = all_metrics(y_true, y_filtered)
    rows.append({"model": "lstm_ukf", "dataset": dataset, **metrics})
    print(f"[lstm_ukf/{dataset}] {metrics}")
    _save_plot(f"lstm_ukf_{dataset}", all_true[0], all_filtered[0])

    # also report the raw (unfiltered) LSTM for direct before/after comparison
    raw_metrics = all_metrics(y, lstm_preds_all)
    rows.append({"model": "lstm_raw_for_ukf_comparison", "dataset": dataset, **raw_metrics})


def evaluate_standalone_kalman():
    """Linear KF / EKF / UKF over the fitted 1RC ECM, run on Oxford test cells
    (the only dataset with a pseudo-OCV curve to build the ECM from)."""
    from src.data import parse_oxford, soc_labeling, splits
    from src.kalman.ecm_model import ThreveninECM
    from src.kalman.ekf import ExtendedKalmanFilter
    from src.kalman.fit_ecm import build_ecm_from_train_cells
    from src.kalman.linear_kf import LinearKalmanFilter
    from src.kalman.ukf import UnscentedKalmanFilter

    print("[kalman] fitting ECM from Oxford training cells...")
    ecm = build_ecm_from_train_cells(dt=DT_SECONDS["oxford"])
    print(f"[kalman] R0={ecm.p.r0:.4f} R1={ecm.p.r1:.4f} C1={ecm.p.c1:.1f}")

    raw = parse_oxford.parse_main_file()
    labeled = soc_labeling.label_oxford_soc(raw)

    for test_cell in sorted(splits.OXFORD_TEST_CELLS):
        g = labeled[(labeled["cell_id"] == test_cell) & (labeled["phase"].str.lower() == "c1dc")]
        if g.empty:
            continue
        checkpoint = sorted(g["checkpoint"].unique())[0]
        g = g[g["checkpoint"] == checkpoint].sort_values("t")

        t = g["t"].to_numpy()
        dt = DT_SECONDS["oxford"]
        grid = np.arange(t[0], t[-1], dt)
        capacity_ah = g["capacity_mah"].iloc[0] / 1000.0
        currents = np.full_like(grid, capacity_ah)  # 1C constant discharge
        voltages = np.interp(grid, t, g["voltage"].to_numpy())
        soc_true = np.interp(grid, t, g["soc"].to_numpy())

        x0 = np.array([soc_true[0], 0.0])
        P0 = np.diag([1e-3, 1e-3])

        lkf = LinearKalmanFilter(ecm)
        lkf_est = lkf.run(currents, voltages, x0, P0)

        ekf = ExtendedKalmanFilter(ecm)
        ekf_est = ekf.run(currents, voltages, x0, P0)

        ukf = UnscentedKalmanFilter(
            dim_x=2, dim_z=1,
            f=ecm.f, h=lambda x, u: np.array([ecm.h(x, u)]),
            process_noise=np.diag([1e-6, 1e-5]), measurement_noise=np.array([[1e-3]]),
        )
        ukf_est = ukf.run(currents, voltages, x0, P0)

        for name, est in [("linear_kf", lkf_est), ("ekf", ekf_est), ("ukf", ukf_est)]:
            metrics = all_metrics(soc_true, est[:, 0])
            rows.append({"model": name, "dataset": f"oxford_ecm_{test_cell}", **metrics})
            print(f"[{name}/oxford/{test_cell}] {metrics}")
            _save_plot(f"{name}_oxford_{test_cell}", soc_true, est[:, 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-kalman", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for dataset in ("nasa", "oxford", "both"):
        for model_name in NEURAL_MODELS:
            evaluate_neural(model_name, dataset, args.device)
        for model_key in BASELINE_MODELS:
            evaluate_baseline(model_key, dataset)
        evaluate_lstm_ukf(dataset, args.device)

    if not args.skip_kalman:
        evaluate_standalone_kalman()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(f"\nsaved {len(df)} rows -> {RESULTS_DIR / 'metrics.csv'}")
    print(df.to_string(index=False))
