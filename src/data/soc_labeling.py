"""Derive ground-truth SOC labels for the NASA and Oxford datasets.

NASA (dynamic, randomized-current, long time series):
  Physics-based coulomb counting, periodically hard-reset at "reference
  charge"/"reference discharge" anchor points (SOC snapped to exactly 1.0 /
  0.0 at those points), using a capacity estimated from clean back-to-back
  full-cycle anchor pairs. NOTE: an earlier version of this function instead
  *rescaled* the coulomb trace to hit the next anchor exactly, proportionally
  across the whole gap -- that only works if accumulation is monotonic
  between anchors. It silently produced garbage (SOC pinned at 0 or 1 for
  the entire multi-day random-walk stretches between reference cycles),
  because the random walk's *gross* charge/discharge throughput between two
  anchors is far larger than its small *net* change, so the rescaled
  fraction constantly overshot [0,1] and got clipped flat. Forward physics
  integration with hard resets avoids that failure mode entirely.

Oxford (per-checkpoint constant-current characterization cycles):
  Within each (cell, checkpoint, phase) constant-current record, SOC(t) is
  just charge_mah(t) normalized by that phase's own total charge transferred
  (the checkpoint's measured capacity at that point in the cell's life) --
  exact for a fixed-current cycle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _last_row_of_step(df: pd.DataFrame, step_mask: pd.Series) -> np.ndarray:
    """Boolean array, True only at the last sample of each contiguous run of
    step_idx values where step_mask is True (i.e. the moment a matching step
    finishes -- the point at which the cell is actually full/empty)."""
    out = np.zeros(len(df), dtype=bool)
    if not step_mask.any():
        return out
    matching_steps = df.loc[step_mask, "step_idx"].unique()
    last_idx = df[df["step_idx"].isin(matching_steps)].groupby("step_idx").tail(1).index.to_numpy()
    out[last_idx] = True
    return out


def label_nasa_soc(df: pd.DataFrame, voltage_tol: float = 0.03, gap_seconds: float = 1200.0) -> pd.DataFrame:
    """Add a `soc` column (0-1) to a single cell's flattened NASA DataFrame.

    `df` must be sorted by `t` already and contain columns: step_idx, step_type, t, voltage, current.
    Anchors come from two sources, unioned together:
      1. If a `comment` column is present, its explicit "reference charge" /
         "reference discharge" step labels (present in the Randomized Battery
         Usage dataset) -- far more reliable than inferring full/empty from
         voltage alone, since a mid-experiment random-walk step can pass
         through a similar voltage without the cell actually being full/empty.
      2. Samples immediately following a time gap > `gap_seconds` whose
         voltage sits within `voltage_tol` of the cell's observed max/min.
         This dataset has a handful of multi-hour logging gaps where real
         (unlogged) charging clearly happened -- voltage jumps ~1V across a
         supposed zero-current "rest" step -- which silently broke pure
         comment-based coulomb-count integration across them (see git history
         / project notes: earlier version left SOC pinned at 0 for entire
         multi-day stretches because of exactly this). Re-anchoring right
         after any big gap catches those without needing to explain them.
    """
    df = df.sort_values("t").reset_index(drop=True)
    v = df["voltage"].to_numpy()
    t_arr = df["t"].to_numpy()
    # Percentile-based, not nanmax/nanmin: a handful of transient voltage
    # spikes (e.g. IR-drop overshoot during high-current pulse tests) can sit
    # ~0.4V above the true ~4.2V full-charge ceiling, which would make a
    # tolerance band around the raw max never match any real full-charge point.
    v_max, v_min = np.nanpercentile(v, 99.5), np.nanpercentile(v, 0.5)

    anchor_sets = []
    if "comment" in df.columns and df["comment"].str.contains("reference", case=False, na=False).any():
        is_ref_charge_end = _last_row_of_step(df, df["comment"].str.lower() == "reference charge")
        is_ref_discharge_end = _last_row_of_step(df, df["comment"].str.lower() == "reference discharge")
        idx = np.where(is_ref_charge_end | is_ref_discharge_end)[0]
        soc = np.where(is_ref_charge_end[idx], 1.0, 0.0)
        anchor_sets.append((idx, soc))
    else:
        # no comments available at all -- fall back to a pure voltage-threshold heuristic
        is_full = v >= (v_max - voltage_tol)
        is_empty = v <= (v_min + voltage_tol)
        idx = np.where(is_full | is_empty)[0]
        soc = np.where(is_full[idx], 1.0, 0.0)
        anchor_sets.append((idx, soc))

    gap_before = np.diff(t_arr, prepend=t_arr[0]) > gap_seconds
    is_full_after_gap = gap_before & (v >= (v_max - voltage_tol))
    is_empty_after_gap = gap_before & (v <= (v_min + voltage_tol))
    gap_idx = np.where(is_full_after_gap | is_empty_after_gap)[0]
    if len(gap_idx):
        gap_soc = np.where(is_full_after_gap[gap_idx], 1.0, 0.0)
        anchor_sets.append((gap_idx, gap_soc))

    all_idx = np.concatenate([a[0] for a in anchor_sets])
    all_soc = np.concatenate([a[1] for a in anchor_sets])
    order = np.argsort(all_idx)
    all_idx, all_soc = all_idx[order], all_soc[order]
    # de-duplicate indices that both sources flagged (keep the first / either -- they agree by construction)
    keep = np.concatenate([[True], np.diff(all_idx) > 0])
    anchor_idx, anchor_soc = all_idx[keep], all_soc[keep]

    t = df["t"].to_numpy()
    i = df["current"].to_numpy()  # amps, +discharge assumed per NASA convention
    # cumulative coulomb count (amp-seconds), trapezoidal
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, np.percentile(dt[dt > 0], 99) if np.any(dt > 0) else 1.0)  # guard against step-boundary jumps
    coulombs = np.cumsum(i * dt)  # amp-seconds discharged (positive = capacity lost)

    anchors = list(zip(anchor_idx.tolist(), anchor_soc.tolist()))
    if len(anchors) < 1:
        # no usable anchors at all -- fall back to plain coulomb counting against a nominal 2.0 Ah cell
        nominal_as = 2.0 * 3600
        df["soc"] = np.clip(1.0 - coulombs / nominal_as, 0.0, 1.0)
        return df

    # Capacity fades substantially over a cell's ~150-day life (observed ~2.1Ah
    # early to ~0.75Ah late on RW9) -- a single global capacity estimate makes
    # early-life segments deplete too fast (capacity underestimated) and
    # late-life segments too slow (capacity overestimated). Build a capacity
    # curve from clean anchor pairs and use each segment's local value.
    cap_times, cap_values = _estimate_capacity_curve(anchors, coulombs, t)

    soc = np.full(len(df), np.nan)
    hours_since_anchor = np.full(len(df), np.inf)
    # before the first anchor: only a backward integration is possible
    idx0, soc0 = anchors[0]
    cap0 = np.interp(t[idx0], cap_times, cap_values)
    soc[: idx0 + 1] = np.clip(soc0 + (coulombs[idx0] - coulombs[: idx0 + 1]) / cap0, 0.0, 1.0)
    hours_since_anchor[: idx0 + 1] = (t[idx0] - t[: idx0 + 1]) / 3600.0
    # after the last anchor: only a forward integration is possible
    idx_last, soc_last = anchors[-1]
    cap_last = np.interp(t[idx_last], cap_times, cap_values)
    soc[idx_last:] = np.clip(soc_last - (coulombs[idx_last:] - coulombs[idx_last]) / cap_last, 0.0, 1.0)
    hours_since_anchor[idx_last:] = (t[idx_last:] - t[idx_last]) / 3600.0

    # between each pair of anchors: integrate BOTH forward from the earlier one
    # and backward from the later one, and keep whichever is closer in time --
    # this halves the worst-case drift distance (the midpoint of the gap) vs.
    # only ever integrating forward from the trailing anchor.
    for (idx0, soc0), (idx1, soc1) in zip(anchors[:-1], anchors[1:]):
        if idx1 <= idx0:
            continue
        cap_fwd = np.interp(t[idx0], cap_times, cap_values)
        cap_bwd = np.interp(t[idx1], cap_times, cap_values)
        seg_t = t[idx0 : idx1 + 1]
        fwd_soc = soc0 - (coulombs[idx0 : idx1 + 1] - coulombs[idx0]) / cap_fwd
        bwd_soc = soc1 + (coulombs[idx1] - coulombs[idx0 : idx1 + 1]) / cap_bwd
        hrs_fwd = (seg_t - t[idx0]) / 3600.0
        hrs_bwd = (t[idx1] - seg_t) / 3600.0
        use_fwd = hrs_fwd <= hrs_bwd

        seg_soc = np.where(use_fwd, fwd_soc, bwd_soc)
        soc[idx0 : idx1 + 1] = np.clip(seg_soc, 0.0, 1.0)
        hours_since_anchor[idx0 : idx1 + 1] = np.minimum(hrs_fwd, hrs_bwd)

    df["soc"] = soc
    # Pure coulomb-count integration drifts over long unanchored stretches --
    # this dataset's reference-cycle anchors can be up to ~5 days apart, and a
    # multi-day integration horizon lets small systematic biases (sensor
    # offset, coulombic efficiency, timestep rounding) compound into large SOC
    # error. Expose how far each row's label is trusted to be from a hard
    # reset so preprocess.py can discard the least-trustworthy stretches
    # rather than training on labels that may have drifted.
    df["hours_since_anchor"] = hours_since_anchor
    return df


def _estimate_capacity_curve(
    anchors: list[tuple[int, float]], coulombs: np.ndarray, t: np.ndarray, max_gap_hours: float = 24.0
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cell capacity (amp-seconds) over time from clean, adjacent,
    opposite-SOC anchor pairs close together in time (genuine dedicated
    reference charge/discharge tests, not pairs separated by days of
    random-walk cycling where gross throughput vastly exceeds net capacity).
    Returns (times, capacity_as) sorted by time, for np.interp-based lookup
    of the locally-applicable capacity at any point in the cell's life."""
    min_capacity_as = 0.1 * 3600.0  # 0.1 Ah floor -- guards against degenerate near-zero-current anchor pairs
    times, caps = [], []
    for (idx0, soc0), (idx1, soc1) in zip(anchors[:-1], anchors[1:]):
        if idx1 <= idx0 or soc0 == soc1:
            continue
        gap_hours = (t[idx1] - t[idx0]) / 3600.0
        if gap_hours > max_gap_hours:
            continue
        cap = abs(coulombs[idx1] - coulombs[idx0]) / abs(soc1 - soc0)
        if cap < min_capacity_as:
            continue
        times.append((t[idx0] + t[idx1]) / 2.0)
        caps.append(cap)
    if not times:
        return np.array([t[0], t[-1]]), np.array([2.0 * 3600.0, 2.0 * 3600.0])  # fallback: nominal 2.0 Ah
    order = np.argsort(times)
    return np.array(times)[order], np.array(caps)[order]


def label_oxford_soc(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `soc` column to a single (cell, checkpoint, phase) constant-current record."""
    out = []
    for (cell, checkpoint, phase), g in df.groupby(["cell_id", "checkpoint", "phase"], sort=False):
        g = g.sort_values("t").copy()
        q = g["charge_mah"].to_numpy()
        q_total = np.nanmax(np.abs(q))
        if q_total < 1e-9:
            g["soc"] = np.nan
            out.append(g)
            continue
        if "dc" in phase.lower():  # discharge: starts full (SOC=1), drains to 0
            g["soc"] = np.clip(1.0 - np.abs(q) / q_total, 0.0, 1.0)
        else:  # charge: starts empty (SOC=0), fills to 1
            g["soc"] = np.clip(np.abs(q) / q_total, 0.0, 1.0)
        g["capacity_mah"] = q_total
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df.assign(soc=np.nan, capacity_mah=np.nan)
