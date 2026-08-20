# Battery SOC Estimation

Python implementation of several SOC-estimation approaches from
[literature_review.md](literature_review.md), trained/evaluated on the NASA
Randomized Battery Usage Dataset and the Oxford Battery Degradation Dataset 1.
See [`.claude/plans` / the approved implementation plan] for full design
rationale; this file is the practical how-to-run guide.

## Setup

```bash
pip install -r requirements.txt
```

(PyTorch is pinned loosely in `requirements.txt` -- if you have an NVIDIA GPU,
install the CUDA build instead, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu124`.)

## 1. Download the raw data

```bash
python -m src.data.download
```

Downloads (idempotent, skips files already present):
- `data/raw/nasa_randomized_battery_usage.zip` (~1.0 GB, NASA PCoE dataset #11)
- `data/raw/oxford_battery_degradation_dataset_1.mat` (~254 MB)
- `data/raw/oxford_example_dc_c1.mat` (~71 KB, single-cycle drive-cycle example)

## 2. Preprocess into windowed SOC-labeled datasets

```bash
python -m src.data.preprocess --dataset both
```

This parses the raw `.mat` files, derives SOC ground truth, resamples to a
fixed timestep, builds sliding windows, and writes `data/processed/nasa.npz`
and `data/processed/oxford.npz` (leave-cells-out train/val/test splits
already applied and feature-normalized).

**Important caveat**: the Oxford dataset's main file only contains
*characterization* cycles (1C charge/discharge + pseudo-OCV, recorded every
100 drive cycles) -- not the continuous dynamic drive-cycle current profile
itself (only `oxford_example_dc_c1.mat`, one cycle, has that). So the Oxford
windows here are built from constant-current segments, which is great for
testing robustness to *aging/degradation* but not dynamic-load
generalization. NASA's dataset covers the dynamic/randomized-load side.
Together they roughly cover what the reviewed papers tested across both axes
-- see `literature_review.md` for details per-paper.

## 3. Train models

```bash
python -m src.train --model baseline_svr --dataset oxford
python -m src.train --model baseline_rf --dataset oxford
python -m src.train --model lstm --dataset oxford
python -m src.train --model cnn_unet --dataset oxford
python -m src.train --model conv_ulsam_sru --dataset oxford
# repeat with --dataset nasa and --dataset both
```

Checkpoints land in `results/checkpoints/`. `lstm_ukf` isn't trained
separately -- it reuses the `lstm` checkpoint and wraps it with a UKF at
evaluation time.

## 4. Evaluate everything

```bash
python -m src.evaluate
```

Runs every model that has a checkpoint, plus the standalone `linear_kf` /
`ekf` / `ukf` filters over a fitted 1RC equivalent-circuit model (Oxford
only -- see below), and writes:
- `results/metrics.csv` -- one row per (model, dataset/test-cell), with
  RMSE/MAE/MAPE/max-error in % SOC
- `results/plots/*.png` -- predicted-vs-true SOC traces

## What's implemented

| Model | File | Paper (see literature_review.md) |
|---|---|---|
| SVR / Random Forest baseline | `src/models/baseline.py` | §1.1, §1.6 |
| Direct LSTM | `src/models/lstm_direct.py` | §2.1 (Chemali et al. 2018) |
| LSTM + UKF hybrid | `src/models/lstm_ukf.py` | §2.4 (Yang et al. 2020) |
| CNN U-Net (symmetric padding + total-variation loss) | `src/models/cnn_unet.py` | §3.3 (Fan et al. 2022) |
| Conv1D + ULSAM + SRU | `src/models/conv_ulsam_sru.py` | §3.2 (Gong et al. 2022) |
| Linear Kalman Filter | `src/kalman/linear_kf.py` | textbook baseline |
| Extended Kalman Filter | `src/kalman/ekf.py` | classic SOC-ECM approach |
| Unscented Kalman Filter | `src/kalman/ukf.py` | generic, reused by `lstm_ukf.py` |

## Methodology

### SOC ground truth

Neither dataset logs SOC directly -- both had to be reconstructed.

**Oxford** (`src/data/soc_labeling.py::label_oxford_soc`): each `(cell,
checkpoint, phase)` record is one constant-current 1C charge or discharge
segment, so SOC is exact within that segment:

```
Q = max(|charge_transferred(t)|)   # this segment's own measured capacity
SOC(t) = 1 - |charge_transferred(t)| / Q     # discharge (starts full)
SOC(t) =     |charge_transferred(t)| / Q     # charge     (starts empty)
```

`Q` is recomputed per checkpoint, so it automatically tracks that cell's
capacity fade over its life -- no separate aging model needed.

**NASA** (`src/data/soc_labeling.py::label_nasa_soc`) has no constant-current
structure to exploit -- it's a genuinely random bidirectional current profile
-- so SOC comes from coulomb counting, periodically hard-reset at trustworthy
anchor points:

1. **Anchors** are 0%/100% ground-truth points, taken from the union of:
   - the dataset's own `"reference charge"` / `"reference discharge"` step
     labels (periodic full-cycle calibration tests, up to ~5 days apart), and
   - any sample immediately following a >20-minute logging gap whose voltage
     sits within 0.03V of the cell's 99.5th/0.5th percentile voltage (catches
     a handful of real but unlogged charging events discovered during
     development -- see the bug note below).
2. **Capacity curve**: for every adjacent, opposite-SOC anchor pair less than
   24h apart (i.e. genuine dedicated reference tests, not two anchors either
   side of a multi-day random-walk stretch), capacity is estimated as
   `|Δcoulombs| / |Δsoc|` and stored against that pair's timestamp. A cell's
   capacity fades substantially over its ~150-day life (~2.1Ah early to
   ~0.75Ah late, on RW9), so this is interpolated over time rather than
   averaged into one global number.
3. **Integration**: for any point between two anchors, SOC is computed
   *twice* -- once integrating forward from the earlier anchor, once
   backward from the later one (`soc = soc_known - Δcoulombs / capacity(t)`,
   sign flipped for the backward direction) -- and whichever anchor is
   closer in time wins. This halves the worst-case drift distance (the
   midpoint of the gap) versus only ever integrating forward.
4. **Confidence filter**: each sample also gets `hours_since_anchor`, the
   time to its nearest anchor. `src/data/preprocess.py` only keeps windows
   whose *label* timestep is within 15h of an anchor (`NASA_MAX_HOURS_SINCE_ANCHOR`)
   -- coulomb-count drift over longer unanchored stretches was found to be
   large enough to pin SOC at 0%/100% and stay there. This discards ~1/3 of
   NASA's windows.

**Bug this methodology replaced**: the first version rescaled the coulomb
trace to hit the *next* anchor exactly, proportionally across the whole gap
-- valid only if accumulation is monotonic between anchors. It silently
produced garbage (SOC pinned flat at 0% or 100% for entire multi-day
stretches), because a random walk's gross throughput between anchors is much
larger than its small net change, so the rescaled fraction constantly
overshot [0,1] and got clipped. Caught via diagnostic plotting (SOC vs. time
next to the raw current trace) during development, not by any error/crash --
the labels were confidently wrong. See git history for the full before/after
plots.

### Windowing

Both datasets are resampled onto a fixed timestep, then cut into overlapping
sliding windows; the window's last timestep's SOC is the label
(`src/data/windowing.py`). Feature order is always `[voltage, current, temperature]`.

| Dataset | dt | window length | stride | ≈ window span |
|---|---|---|---|---|
| NASA | 4s | 64 steps | 32 steps | 4.3 min |
| Oxford | 20s | 64 steps | 8 steps | 21 min |

Features are z-score normalized using statistics fit on the training split
only (`src/data/windowing.py::normalize_features`); the same mean/std are
reused for val/test to avoid leakage.

### Models

- **Baseline** (`src/models/baseline.py`): each window is collapsed to a
  15-dim feature vector (last value + mean/std/min/max, per channel) and fed
  to `sklearn`'s `SVR` (RBF kernel) or `RandomForestRegressor` (200 trees,
  max depth 12).
- **LSTM** (`src/models/lstm_direct.py`): 2-layer LSTM (hidden size 64) reads
  the raw window, last hidden state → `Linear(64→32→1)` → sigmoid.
- **LSTM+UKF** (`src/models/lstm_ukf.py`): the trained LSTM's per-window
  prediction is treated as a noisy *measurement* of SOC; a 1-state UKF fuses
  it with a coulomb-counting *process* model (`SOC_k+1 = SOC_k - I·dt/(3600·Q)`,
  nominal Q per dataset) using the generic sigma-point filter in
  `src/kalman/ukf.py`.
- **CNN U-Net** (`src/models/cnn_unet.py`): 1D conv encoder-decoder with skip
  connections (2 downsample/upsample levels, base width 16), *symmetric*
  padding (mirrors the signal including the boundary sample, unlike
  zero/reflect padding) on every conv, trained with `MSE + 0.1 × total
  variation loss` (mean |Δ| between consecutive predicted timesteps) to
  discourage a jumpy output trace.
- **Conv+ULSAM+SRU** (`src/models/conv_ulsam_sru.py`): `Conv1d(3→32) → ULSAM
  (4-subspace channel attention) → SRU (hidden 64, parallel input
  projections + elementwise-only recurrence) → Linear(64→32→1)`.

### Kalman filters / equivalent circuit model

`src/kalman/ecm_model.py` implements a 1RC Thevenin model, state `x = [SOC,
V1]` (V1 = polarization voltage across the RC branch):

```
SOC_k+1 = SOC_k - (I_k · dt) / (3600 · Q)
V1_k+1  = α·V1_k + R1·(1-α)·I_k,        α = exp(-dt / (R1·C1))
V_term  = OCV(SOC_k) - V1_k - I_k·R0
```

`OCV(SOC)` is a degree-6 polynomial fit to Oxford's pseudo-OCV (near-zero-current)
discharge cycles (`OCVCurve` in `ecm_model.py`). `R0` is fit from a training
cell's 1C discharge cycle by least-squares once `R1`/`C1` are fixed at
literature-typical values for a small pouch cell (`src/kalman/fit_ecm.py`) --
a free 3-parameter fit was tried first and found to be poorly identified
(R0 and R1·C1 trade off almost interchangeably given only one slow discharge
curve, no pulse/HPPC test to separate them), so `R1`/`C1` are fixed and only
`R0` -- which *is* well identified by a single discharge -- is fit from data.

`linear_kf.py` / `ekf.py` / `ukf.py` all share this `f`/`h`; they differ only
in how they linearize `h` (fixed linearization point vs. relinearized every
step vs. sigma points, respectively).

## Results

Full, current numbers always live in [`results/metrics.csv`](results/metrics.csv)
(regenerate with `python -m src.evaluate`); per-model predicted-vs-true plots
are in `results/plots/`. Summary as of the last full run:

**NASA (leave-cell-out: train RW9/10/11, test RW12) -- SOC RMSE, %**

| Model | RMSE % | MAE % |
|---|---|---|
| Random Forest | 14.39 | 7.21 |
| Conv+ULSAM+SRU | 14.84 | 8.22 |
| LSTM + UKF | 14.56 | 8.12 |
| LSTM (direct) | 15.06 | 8.57 |
| CNN U-Net | 15.10 | 8.67 |
| SVR | 16.87 | 8.51 |

**Oxford (leave-cell-out: train 6 cells, test Cell7/Cell8) -- SOC RMSE, %**

| Model | RMSE % | MAE % |
|---|---|---|
| Random Forest | 0.41 | 0.28 |
| SVR | 0.56 | 0.40 |
| LSTM (direct) | 0.62 | 0.50 |
| Conv+ULSAM+SRU | 0.71 | 0.57 |
| CNN U-Net | 0.75 | 0.61 |
| LSTM + UKF | 1.84 | 1.51 |
| UKF (ECM, no NN) | 1.66-1.72 | 1.46-1.55 |
| EKF (ECM, no NN) | 1.63-1.69 | 1.46-1.51 |
| Linear KF (ECM, no NN) | 4.22-4.33 | 3.85-3.95 |

**Both (pooled train, evaluated on both test sets)** lands in between, close
to NASA's numbers (~14.1-16.4% RMSE) since NASA's test set is ~17x larger
than Oxford's and dominates the pooled metric.

**Reading these results:**
- **Oxford is easy, NASA is hard.** Oxford's windows come from clean
  constant-current characterization cycles; NASA's come from a genuinely
  randomized, bidirectional current profile on a cell the model never saw
  during training. The ~20-25x gap in RMSE between the two datasets reflects
  that difficulty gap, not a difference in model quality -- every model's
  *relative ranking* is fairly consistent across both datasets.
- **The classical Kalman filters (EKF/UKF) clearly beat the plain linear KF**
  on Oxford (~1.7% vs ~4.3% RMSE) -- exactly the result the literature would
  predict, since linearizing the nonlinear OCV(SOC) curve only once (linear
  KF) loses much more information than relinearizing every step (EKF) or
  using sigma points (UKF).
- **LSTM+UKF's effect flips between datasets.** On Oxford, where the raw LSTM
  is already very accurate (0.62% RMSE), fusing it with a fixed-capacity
  coulomb-counting prior *hurts* (1.84%) -- degradation across Oxford's many
  checkpoints means the fixed nominal capacity used by the process model is
  measurably wrong for aged cells, and there's little LSTM noise left to
  usefully filter out anyway. On NASA, where the raw LSTM is much noisier
  (15.06%), the same fusion *helps* (14.56%) -- there's real noise for the
  UKF to smooth out, and capacity drift matters less over NASA's shorter,
  more tightly-anchored windows. This matches the general Kalman-filtering
  principle: fusion only helps when the "sensor" (here, the LSTM) is
  imperfect enough that a physics prior adds real information.

## Scoping notes / known simplifications

- **Standalone Kalman filters run on Oxford only.** The 1RC ECM needs an
  OCV(SOC) curve, which we fit from Oxford's dedicated low-current pseudo-OCV
  cycles (`src/kalman/fit_ecm.py`). NASA has no equivalent low-current
  characterization step, so we didn't force-fit an OCV curve for it --
  extending this would mean either approximating with the Oxford-fit curve
  (different cell chemistry/geometry, so accuracy would suffer) or deriving
  a new curve from NASA's own reference cycles.
- **`lstm_ukf`'s process model is a simple coulomb-counting prior** (not the
  full 2-state RC model the standalone filters use) with a fixed nominal
  capacity per dataset -- it exists to smooth the LSTM's point predictions
  using physical plausibility, not to be a from-scratch ECM.
- **Leave-cells-out test split**: NASA holds out cell `RW12`; Oxford holds
  out `Cell7`/`Cell8`. Edit `src/data/splits.py` to change which cells are
  held out.
- **NASA's SOC labels are physics-reconstructed, not directly logged** (see
  `src/data/soc_labeling.py::label_nasa_soc`), and only windows within
  `NASA_MAX_HOURS_SINCE_ANCHOR` (15h, in `src/data/preprocess.py`) of a hard
  SOC reset are kept -- reference-cycle anchors in this dataset can be up to
  ~5 days apart, and pure coulomb-count integration drifts enough over that
  horizon to pin at 0%/100% and stay there (this was a real bug caught via
  diagnostic plotting during development: an earlier anchor-*rescaling*
  approach assumed monotonic current between anchors, which silently breaks
  on a random walk's small-net/large-gross throughput). Bidirectional
  integration (from both the preceding and following anchor) plus the
  confidence filter keeps only the trustworthy stretches; this discards
  roughly 1/3 of NASA's windows but was necessary for the labels to be usable
  at all. RW9-12 only (sub-dataset #1 of the NASA archive's 7) are parsed by
  default, both for tractable preprocessing time and because that's the
  specific 4-cell set the reference-cycle-based labeling was validated against.
- **SVR is capped at 20,000 training rows** (`SVR_MAX_TRAIN_SAMPLES` in
  `src/train.py`) -- its RBF kernel is O(n²)-O(n³), and NASA's ~160k-row
  training set never finished within a reasonable time uncapped.
- **NASA's very high MAPE/max-error numbers** (max-error near 100%, MAPE in
  the hundreds-to-thousands%) come from a small number of windows where the
  true SOC is at or near 0% -- MAPE's denominator blows up there, and any
  absolute miss near an SOC=0% boundary reads as "100% error." RMSE/MAE are
  the more meaningful metrics for this dataset; MAPE is included for
  completeness/comparability with the source papers, not as the headline number.
