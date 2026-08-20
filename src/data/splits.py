"""Leave-cells-out train/val/test splitting, shared by all models.

Splitting is done at the physical-cell level (not randomly across windows) so
test performance reflects generalization to a battery the model never saw
during training -- matching how the reviewed papers evaluate.
"""
from __future__ import annotations

import numpy as np


def split_by_cell(
    groups: list[tuple],
    test_cells: set[str],
    val_fraction: float = 0.15,
    seed: int = 0,
):
    """Given per-window group keys (first element of each tuple = cell_id) and a
    fixed set of held-out test cell ids, split the remaining cells' windows into
    train/val by randomly holding out `val_fraction` of the *training* windows
    (val windows still come only from training cells, never from test cells).

    Returns boolean masks (train_mask, val_mask, test_mask), each length == len(groups).
    """
    cell_ids = np.array([g[0] for g in groups])
    available = set(cell_ids.tolist())
    resolved_test_cells = test_cells & available
    if not resolved_test_cells:
        # requested test cell(s) not present under these exact ids (e.g. dataset's
        # actual cell naming differs from the hardcoded default) -- fall back to
        # holding out the alphabetically-last ~15% of cells so splitting still works.
        all_cells = sorted(available)
        n_hold = max(1, round(len(all_cells) * 0.15))
        resolved_test_cells = set(all_cells[-n_hold:])
        print(f"[splits] WARNING: none of {test_cells} found among {sorted(available)}; "
              f"falling back to test cells {sorted(resolved_test_cells)}")

    test_mask = np.isin(cell_ids, list(resolved_test_cells))
    train_val_idx = np.where(~test_mask)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(train_val_idx)
    n_val = int(len(train_val_idx) * val_fraction)
    val_idx = set(train_val_idx[:n_val].tolist())

    train_mask = np.zeros(len(groups), dtype=bool)
    val_mask = np.zeros(len(groups), dtype=bool)
    for idx in train_val_idx:
        if idx in val_idx:
            val_mask[idx] = True
        else:
            train_mask[idx] = True

    return train_mask, val_mask, test_mask


NASA_TEST_CELLS = {"RW12"}
OXFORD_TEST_CELLS = {"Cell7", "Cell8"}
