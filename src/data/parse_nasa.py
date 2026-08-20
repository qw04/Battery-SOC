"""Parse NASA Randomized Battery Usage Dataset .mat files into flat per-cell DataFrames.

The downloaded archive is a zip-of-zips: the outer zip contains 7 sub-dataset
zips (different temperatures / current-skew conditions), each containing a
handful of cells' .mat files under `<subset>/data/Matlab/RW*.mat`. Each cell
totals several million samples over ~150 days, so -- to keep preprocessing
and training times reasonable -- we default to just sub-dataset #1
("Battery_Uniform_Distribution_Charge_Discharge"), which holds cells
RW9-RW12, the same 4 cells this project's leave-cells-out split
(src/data/splits.py: train on RW9-11, test on RW12) was designed around.
Pass `subsets=[...]` to pull in additional sub-datasets.

Each cell's .mat file contains a top-level `data` struct with a `.step`
array. Each step has: type ('C'harge / 'D'ischarge / 'R'est), a human-readable
`comment` (e.g. "charge (random walk)", "reference discharge" -- the latter
marks the periodic full-cycle reference points soc_labeling.py anchors SOC
to), relativeTime (s, since step start), time (s, since experiment start),
voltage (V), current (A, negative=charging/positive=discharging), temperature (C).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EXTRACT_DIR = RAW_DIR / "nasa_extracted"

DEFAULT_SUBSETS = ["1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip"]


def ensure_extracted(subsets: list[str] | None = None) -> Path:
    """Extract the outer zip's top-level listing (cheap), then extract just the
    requested inner sub-dataset zip(s) (each ~100-300MB) into EXTRACT_DIR."""
    zip_path = RAW_DIR / "nasa_randomized_battery_usage.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found -- run src/data/download.py first")
    subsets = subsets or DEFAULT_SUBSETS

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        outer_dir = next(n for n in zf.namelist() if n.endswith("/"))  # "11. Randomized Battery Usage Data Set/"
        for subset in subsets:
            marker = EXTRACT_DIR / (Path(subset).stem + ".done")
            if marker.exists():
                continue
            inner_bytes = zf.read(outer_dir + subset)
            import io
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                inner.extractall(EXTRACT_DIR)
            marker.touch()
    return EXTRACT_DIR


def _find_step_struct(mat_dict: dict):
    """Locate the top-level entry that holds a `.step` (or `.procedure`) field."""
    for key, val in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(val, dict) and "step" in val:
            return val
        if isinstance(val, dict):
            # some releases nest one level deeper, e.g. data['data']['step']
            for sub in val.values():
                if isinstance(sub, dict) and "step" in sub:
                    return sub
    raise KeyError(f"Could not find a struct with a 'step' field among keys: {list(mat_dict.keys())}")


def parse_cell_file(mat_path: Path) -> pd.DataFrame:
    """Parse one cell's .mat file into a flat DataFrame: cell_id, step_idx, step_type, t, V, I, T."""
    mat = loadmat(mat_path, simplify_cells=True)
    struct = _find_step_struct(mat)
    steps = struct["step"]
    if isinstance(steps, dict):
        steps = [steps]

    # Build flat numpy arrays across all ~10^5 steps and concatenate/repeat
    # once at the end, rather than constructing one small pandas.DataFrame per
    # step and pd.concat-ing tens of thousands of them -- that per-step
    # DataFrame-construction overhead was the dominant cost (multiple minutes
    # per cell) in an earlier version of this function.
    cell_id = mat_path.stem
    t_parts, v_parts, i_parts, temp_parts = [], [], [], []
    step_idx_parts, step_type_parts, comment_parts = [], [], []

    for step_idx, step in enumerate(steps):
        t = np.atleast_1d(step.get("time", np.array([])))
        v = np.atleast_1d(step.get("voltage", np.array([])))
        i = np.atleast_1d(step.get("current", np.array([])))
        temp = np.atleast_1d(step.get("temperature", np.array([])))
        n = min(len(t), len(v), len(i))
        if n == 0:
            continue
        t_parts.append(t[:n])
        v_parts.append(v[:n])
        i_parts.append(i[:n])
        temp_parts.append(temp[:n] if len(temp) >= n else np.full(n, np.nan))
        step_idx_parts.append(np.full(n, step_idx, dtype=np.int32))
        step_type_parts.append(np.full(n, str(step.get("type", "?")), dtype=object))
        comment_parts.append(np.full(n, str(step.get("comment", "")), dtype=object))

    cols = ["cell_id", "step_idx", "step_type", "comment", "t", "voltage", "current", "temperature"]
    if not t_parts:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(
        {
            "cell_id": cell_id,  # broadcasts to every row
            "step_idx": np.concatenate(step_idx_parts),
            "step_type": np.concatenate(step_type_parts),
            "comment": np.concatenate(comment_parts),
            "t": np.concatenate(t_parts),
            "voltage": np.concatenate(v_parts),
            "current": np.concatenate(i_parts),
            "temperature": np.concatenate(temp_parts),
        }
    )


def iter_cells(cell_files: list[str] | None = None):
    """Yield (cell_id, DataFrame) one cell at a time. Each cell is ~2-9M rows,
    so preprocess.py consumes this lazily (parse -> label -> window -> discard)
    rather than holding every cell's raw data in memory at once -- the earlier
    all-at-once `parse_all` blew past several GB of RAM on this dataset."""
    extract_dir = ensure_extracted()
    mat_paths = sorted(extract_dir.rglob("RW*.mat")) if cell_files is None else [
        p for p in extract_dir.rglob("*.mat") if p.stem in cell_files
    ]
    for p in mat_paths:
        yield p.stem, parse_cell_file(p)


def parse_all(cell_files: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Parse a set of cell .mat files (default: all RW*.mat found after extraction).
    Loads every cell into memory at once -- prefer `iter_cells` for large datasets."""
    return dict(iter_cells(cell_files))


if __name__ == "__main__":
    cells = parse_all()
    for cell_id, df in cells.items():
        print(f"{cell_id}: {len(df)} rows, {df['step_idx'].nunique()} steps, "
              f"types={sorted(df['step_type'].unique())}")
