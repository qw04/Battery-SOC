"""Parse the Oxford Battery Degradation Dataset 1 .mat file into flat per-record DataFrames.

Per the dataset's Readme.txt, the nested MATLAB struct layout is:
  Layer 1: Cell (1-8)
  Layer 2: characterisation-cycle checkpoint (e.g. 'cyc0100' = after 100 drive cycles)
  Layer 3: phase -- C1ch (1C charge), C1dc (1C discharge), OCVch (pseudo-OCV charge), OCVdc (pseudo-OCV discharge)
  Layer 4: t (s), v (V), q (mAh), T (degC)

We don't hardcode exact key spellings (case/prefix can vary across scipy
versions' simplify_cells handling) -- instead we recursively walk the parsed
dict/list tree and collect every leaf struct that has t/v/q/T arrays,
tagging it with the (cell, checkpoint, phase) path it was found at.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MAIN_FILE = RAW_DIR / "oxford_battery_degradation_dataset_1.mat"
EXAMPLE_FILE = RAW_DIR / "oxford_example_dc_c1.mat"

_LEAF_KEYS = {"t", "v", "q"}  # T (temperature) is present in the main file but optional


def _is_leaf_record(d: dict) -> bool:
    keys_lower = {k.lower() for k in d.keys()}
    return _LEAF_KEYS.issubset(keys_lower)


def _get_ci(d: dict, name: str):
    """Case-insensitive dict lookup."""
    for k, v in d.items():
        if k.lower() == name.lower():
            return v
    return None


def _walk(node, path: list[str], records: list[dict]):
    if isinstance(node, dict):
        if _is_leaf_record(node):
            t_raw = np.atleast_1d(_get_ci(node, "t")).astype(float).ravel()
            # `t` is stored as a MATLAB datenum (fractional days since year 0),
            # not elapsed seconds despite what Readme.txt says -- e.g. values
            # around 735954.86 decode to 08-Jan-2015, matching the dataset's
            # actual start date. Convert to elapsed seconds within this cycle.
            t = (t_raw - t_raw[0]) * 86400.0 if len(t_raw) else t_raw
            v = np.atleast_1d(_get_ci(node, "v")).astype(float).ravel()
            q = np.atleast_1d(_get_ci(node, "q")).astype(float).ravel()
            temp = _get_ci(node, "T")
            i = _get_ci(node, "i")  # only present in the small example-cycle file
            n = min(len(t), len(v), len(q))
            if n == 0:
                return
            record = {
                "path": "/".join(path),
                "t": t[:n],
                "voltage": v[:n],
                "charge_mah": q[:n],
                "temperature": (np.atleast_1d(temp).astype(float).ravel()[:n] if temp is not None else np.full(n, np.nan)),
                "current_ma": (np.atleast_1d(i).astype(float).ravel()[:n] if i is not None else np.full(n, np.nan)),
            }
            records.append(record)
            return
        for key, val in node.items():
            _walk(val, path + [str(key)], records)
    elif isinstance(node, (list, np.ndarray)) and node is not None and not isinstance(node, np.generic):
        try:
            iterator = list(node)
        except TypeError:
            return
        for idx, item in enumerate(iterator):
            _walk(item, path + [str(idx)], records)


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        parts = rec["path"].split("/")
        # parts look like [..., "Cell3", "cyc0200", "C1dc"] -- last 3 segments are the tags we care about.
        cell = next((p for p in parts if "cell" in p.lower()), parts[0] if parts else "unknown")
        checkpoint = next((p for p in parts if p.lower().startswith("cyc")), "cyc0000")
        phase = parts[-1]
        n = len(rec["t"])
        rows.append(
            pd.DataFrame(
                {
                    "cell_id": cell,
                    "checkpoint": checkpoint,
                    "phase": phase,
                    "t": rec["t"],
                    "voltage": rec["voltage"],
                    "charge_mah": rec["charge_mah"],
                    "temperature": rec["temperature"],
                    "current_ma": rec["current_ma"],
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["cell_id", "checkpoint", "phase", "t", "voltage", "charge_mah", "temperature", "current_ma"]
        )
    return pd.concat(rows, ignore_index=True)


def parse_main_file(path: Path = MAIN_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run src/data/download.py first")
    mat = loadmat(path, simplify_cells=True)
    records: list[dict] = []
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        _walk(val, [key], records)
    return _records_to_df(records)


def parse_example_file(path: Path = EXAMPLE_FILE) -> pd.DataFrame:
    """The small file with the one full dynamic (Artemis Urban drive-cycle) discharge."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run src/data/download.py first")
    mat = loadmat(path, simplify_cells=True)
    records: list[dict] = []
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        _walk(val, [key], records)
    return _records_to_df(records)


if __name__ == "__main__":
    df = parse_main_file()
    print(f"parsed {len(df)} rows")
    print(df.groupby(["cell_id", "phase"]).size().unstack(fill_value=0))
