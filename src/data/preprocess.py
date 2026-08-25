"""Orchestrates parse -> label -> window -> split -> save for both datasets.

Usage:
  python -m src.data.preprocess --dataset nasa
  python -m src.data.preprocess --dataset oxford
  python -m src.data.preprocess --dataset both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import parse_nasa, parse_oxford, soc_labeling, splits, windowing

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# NASA random-walk currents and Oxford characterization cycles are both logged
# ~1Hz natively, so dt=1.0s (no downsampling) on both -- matching how the
# reviewed papers actually window their data (raw-sample windows, not
# resampled-then-windowed). window_len=50 matches the "50 lookback steps"
# convention used by USAL's dynamic-loading case study and the benchmark LSTM
# in NNGP (literature_review.md SS2.6/SS3.1) -- the most-repeated window size
# across the papers verified against full text. Stride differs between the two
# datasets purely to keep window COUNT (and training time) tractable given how
# much raw data one NASA cell contains (~150 days/cell) vs. an Oxford
# characterization cycle (hours); each stride is chosen to match its dataset's
# *_SHORT_STRIDE below, so the standard and short variants only differ in
# window_len, not dt or window spacing.
NASA_DT = 1.0             # seconds between resampled points
NASA_WINDOW_LEN = 50       # ~50s of history per window (SS2.6/SS3.1 convention)
NASA_STRIDE = 128
NASA_MAX_HOURS_SINCE_ANCHOR = 15.0  # discard windows whose SOC label may have drifted too far (see preprocess_nasa)

OXFORD_DT = 1.0
OXFORD_WINDOW_LEN = 50    # ~50s of history per window (SS2.6/SS3.1 convention)
OXFORD_STRIDE = 160

# "Short" variant matches Gong, Wang & Cheng (2022)'s own "input length"
# ablation (5-10 raw ~1Hz samples, default 10) for
# src/models/conv_ulsam_sru_paper.py, a faithful reproduction of their
# architecture (literature_review.md SS3.2). Same dt and stride as the standard
# variant above -- window_len is the only thing that changes between the two.
NASA_SHORT_DT = 1.0
NASA_SHORT_WINDOW_LEN = 10
NASA_SHORT_STRIDE = 128

OXFORD_SHORT_DT = 1.0
OXFORD_SHORT_WINDOW_LEN = 10
OXFORD_SHORT_STRIDE = 160


def preprocess_nasa(name: str = "nasa", window_len: int = NASA_WINDOW_LEN, stride: int = NASA_STRIDE, dt: float = NASA_DT):
    """Process one cell at a time (parse -> label -> window -> discard raw data)
    rather than holding all cells' multi-million-row raw DataFrames in memory
    simultaneously -- each RW cell is several million rows, and this dataset's
    all-at-once approach was observed climbing past 7-8GB RAM on a 16GB machine."""
    X_parts, y_parts, group_parts = [], [], []
    for cell_id, df in parse_nasa.iter_cells():
        if df.empty:
            continue
        print(f"[{name}] labeling {cell_id} ({len(df)} rows)...", flush=True)
        labeled = soc_labeling.label_nasa_soc(df)
        del df

        print(f"[{name}] windowing {cell_id}...", flush=True)
        # NASA's reference-cycle anchors can be up to ~5 days apart, and pure
        # coulomb-count integration drifts enough over that horizon to hit the
        # [0,1] clip and stay pinned there (verified visually) -- only keep
        # windows whose label is within NASA_MAX_HOURS_SINCE_ANCHOR of a hard
        # SOC reset, where the reconstruction is still trustworthy.
        X_cell, y_cell, groups_cell = windowing.make_windows(
            labeled, window_len=window_len, stride=stride, group_cols=["cell_id"], dt=dt,
            confidence_col="hours_since_anchor", max_confidence_value=NASA_MAX_HOURS_SINCE_ANCHOR,
        )
        del labeled
        print(f"[{name}] {cell_id}: {len(X_cell)} windows", flush=True)
        X_parts.append(X_cell)
        y_parts.append(y_cell)
        group_parts.extend(groups_cell)

    X = np.concatenate(X_parts) if X_parts else np.empty((0, window_len, 3))
    y = np.concatenate(y_parts) if y_parts else np.empty((0,))
    _split_and_save(name, X, y, group_parts, splits.NASA_TEST_CELLS)


def preprocess_oxford(name: str = "oxford", window_len: int = OXFORD_WINDOW_LEN, stride: int = OXFORD_STRIDE, dt: float = OXFORD_DT):
    print(f"[{name}] parsing raw .mat file...")
    df = parse_oxford.parse_main_file()
    print(f"[{name}] labeling {len(df)} rows...")
    labeled = soc_labeling.label_oxford_soc(df)
    # only keep the constant-current phases (C1ch/C1dc) for windowing -- OCV phases are
    # near-zero-current and used separately to fit the OCV(SOC) curve for the Kalman filters.
    cc_phases = labeled[labeled["phase"].str.contains("C1", case=False, na=False)].copy()

    # the main Oxford file logs charge (q) but not per-sample current -- reconstruct it
    # since these phases are constant-current by construction: magnitude = capacity_mah/1000
    # (amps), sign positive for discharge (C1dc) and negative for charge (C1ch), matching
    # the NASA dataset's +discharge/-charge convention used throughout this project.
    sign = np.where(cc_phases["phase"].str.lower() == "c1dc", 1.0, -1.0)
    cc_phases["current"] = sign * (cc_phases["capacity_mah"] / 1000.0)

    print(f"[{name}] windowing...")
    X, y, groups = windowing.make_windows(
        cc_phases,
        window_len=window_len,
        stride=stride,
        group_cols=["cell_id", "checkpoint", "phase"],
        dt=dt,
    )
    _split_and_save(name, X, y, groups, splits.OXFORD_TEST_CELLS)

    # also save the OCV cycles separately (uncut) for src/kalman/ecm_model.py's OCVCurve fit
    # -- only needs doing once, not per window-length variant.
    if name == "oxford":
        ocv_phases = labeled[labeled["phase"].str.contains("OCV", case=False, na=False)]
        ocv_path = PROCESSED_DIR / "oxford_ocv_cycles.csv"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ocv_phases.to_csv(ocv_path, index=False)
        print(f"[{name}] saved OCV reference cycles -> {ocv_path}")


def _split_and_save(name: str, X: np.ndarray, y: np.ndarray, groups: list, test_cells: set):
    if len(X) == 0:
        print(f"[{name}] WARNING: no windows produced -- check parsing/labeling upstream")
        return
    train_mask, val_mask, test_mask = splits.split_by_cell(groups, test_cells)

    X_train, mean, std = windowing.normalize_features(X[train_mask])
    X_val, _, _ = windowing.normalize_features(X[val_mask], mean, std)
    X_test, _, _ = windowing.normalize_features(X[test_mask], mean, std)

    # keep raw (unnormalized) test windows too -- the Kalman-filter-based
    # evaluation paths (lstm_ukf, standalone KF/EKF/UKF) need real current
    # values in amps, not z-scored ones.
    X_test_raw = X[test_mask]
    group_strs = np.array(["|".join(g) for g in groups])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{name}.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y[train_mask],
        X_val=X_val, y_val=y[val_mask],
        X_test=X_test, y_test=y[test_mask],
        X_test_raw=X_test_raw,
        groups_test=group_strs[test_mask],
        feature_mean=mean, feature_std=std,
    )
    print(f"[{name}] saved {out_path}: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["nasa", "oxford", "both", "nasa_short", "oxford_short", "both_short"],
        default="both",
    )
    args = parser.parse_args()

    if args.dataset in ("nasa", "both"):
        preprocess_nasa()
    if args.dataset in ("oxford", "both"):
        preprocess_oxford()
    if args.dataset in ("nasa_short", "both_short"):
        preprocess_nasa("nasa_short", NASA_SHORT_WINDOW_LEN, NASA_SHORT_STRIDE, NASA_SHORT_DT)
    if args.dataset in ("oxford_short", "both_short"):
        preprocess_oxford("oxford_short", OXFORD_SHORT_WINDOW_LEN, OXFORD_SHORT_STRIDE, OXFORD_SHORT_DT)
