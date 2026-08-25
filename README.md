# Battery SOC Estimation

Python implementation of several SOC-estimation approaches from the
[**Literature Review**](https://qw04.github.io/Battery-SOC/literature_review.html),
trained/evaluated on the NASA Randomized Battery Usage Dataset and the Oxford
Battery Degradation Dataset 1. This file is the practical how-to-run guide;
see the literature review for the papers, per-paper novelty summaries, and
citations behind each implemented model.

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
python -m src.train --model conv_ulsam_lstm --dataset oxford
# repeat with --dataset nasa and --dataset both

# faithful Gong et al. 2022 reproduction -- short window variant only
python -m src.train --model conv_ulsam_sru_paper --dataset nasa_short
python -m src.train --model conv_ulsam_sru_paper --dataset oxford_short
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
| Conv1D + ULSAM + LSTM | `src/models/conv_ulsam_lstm.py` | §3.2-inspired (Gong et al. 2022), LSTM in place of their SRU |
| Conv1D + ULSAM + SRU (faithful reproduction) | `src/models/conv_ulsam_sru_paper.py` | §3.2 (Gong et al. 2022), run on `nasa_short`/`oxford_short` |
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
| NASA | 1s (native) | 50 steps | 128 steps | 50s |
| Oxford | 1s (native) | 50 steps | 160 steps | 50s |
| NASA / Oxford ("short" variant) | 1s (native) | 10 steps | 128 / 160 steps | 10s |

Both datasets are windowed over raw ~1Hz samples (no resampling/downsampling) --
the 50-step window length matches the "50 lookback steps" convention used by
USAL's dynamic-loading case study and the benchmark LSTM in NNGP
(literature_review.md §2.6/§3.1). Stride differs between datasets only to keep
window count (and training time) tractable given how much raw data one NASA
cell contains (~150 days) vs. an Oxford characterization cycle (hours); each
dataset's "short" variant reuses the same dt/stride as its standard variant, so
window length is the only thing that changes between them. The short variant
(`--dataset nasa_short` / `oxford_short`) exists to match Gong, Wang & Cheng
(2022)'s own input-length ablation (§3.2) for `conv_ulsam_sru_paper.py`.

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
- **Conv+ULSAM+LSTM** (`src/models/conv_ulsam_lstm.py`): `Conv1d(3→32) → ULSAM
  (4-subspace channel attention) → LSTM (hidden 64) → Linear(64→32→1)`.
- **Conv+ULSAM+SRU** (`src/models/conv_ulsam_sru_paper.py`): faithful
  reproduction of Gong, Wang & Cheng (2022)'s own architecture -- two conv
  layers (3→10→20), a deeper residual ULSAM attention branch, and a real SRU
  (not LSTM) run over the flattened channel×time feature map as one long
  pseudo-sequence, per the paper's diagram. Paired with the short (10-step)
  window variant to match the paper's own input-length ablation.

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
| Conv+ULSAM+LSTM | 15.20 | 8.86 |
| Random Forest | 15.22 | 7.79 |
| CNN U-Net | 15.33 | 9.00 |
| LSTM + UKF | 15.38 | 8.71 |
| LSTM (direct) | 15.96 | 9.15 |
| SVR | 17.55 | 9.13 |

**Oxford (leave-cell-out: train 6 cells, test Cell7/Cell8) -- SOC RMSE, %**

| Model | RMSE % | MAE % |
|---|---|---|
| Random Forest | 0.52 | 0.28 |
| Conv+ULSAM+LSTM | 0.87 | 0.66 |
| LSTM (direct) | 0.89 | 0.66 |
| CNN U-Net | 1.21 | 0.99 |
| SVR | 1.26 | 0.90 |
| EKF (ECM, no NN) | 1.63-1.69 | 1.46-1.51 |
| UKF (ECM, no NN) | 1.66-1.72 | 1.49-1.55 |
| LSTM + UKF | 2.50 | 2.07 |
| Linear KF (ECM, no NN) | 4.22-4.33 | 3.85-3.95 |

**Both (pooled train, evaluated on both test sets)** lands close to NASA's
numbers (14.1-18.3% RMSE) since NASA's test set is ~11x larger than Oxford's
and dominates the pooled metric. Ranking is consistent with the NASA table:
Random Forest and Conv+ULSAM+LSTM lead, SVR trails.

**Short window variant (window=10, matching Gong et al. 2022's own ablation)**

| Model / dataset | RMSE % | MAE % |
|---|---|---|
| Conv+ULSAM+SRU (faithful) / oxford_short | 0.96 | 0.73 |
| Conv+ULSAM+SRU (faithful) / nasa_short | 15.91 | 9.37 |

**Reading these results:**
- **Oxford is easy, NASA is hard.** Oxford's windows come from clean
  constant-current characterization cycles; NASA's come from a genuinely
  randomized, bidirectional current profile on a cell the model never saw
  during training. The gap in RMSE between the two datasets ranges from
  ~13x to ~30x depending on the model, not a difference in model quality --
  every model's *relative ranking* is fairly consistent across both datasets.
- **The classical Kalman filters (EKF/UKF) clearly beat the plain linear KF**
  on Oxford (~1.6-1.7% vs ~4.2-4.3% RMSE) -- exactly the result the literature
  would predict, since linearizing the nonlinear OCV(SOC) curve only once
  (linear KF) loses much more information than relinearizing every step (EKF)
  or using sigma points (UKF).
- **LSTM+UKF's effect flips between datasets.** On Oxford, where the raw LSTM
  is already very accurate (0.89% RMSE), fusing it with a fixed-capacity
  coulomb-counting prior *hurts* (2.50%) -- degradation across Oxford's many
  checkpoints means the fixed nominal capacity used by the process model is
  measurably wrong for aged cells, and there's little LSTM noise left to
  usefully filter out anyway. On NASA, where the raw LSTM is much noisier
  (15.96%), the same fusion *helps slightly* (15.38%) -- there's real noise
  for the UKF to smooth out, and capacity drift matters less over NASA's
  shorter, more tightly-anchored windows. This matches the general
  Kalman-filtering principle: fusion only helps when the "sensor" (here, the
  LSTM) is imperfect enough that a physics prior adds real information.
- **The real SRU (short window) holds its own against the LSTM variant
  (standard window) on Oxford** -- 0.96% vs. 0.87% RMSE -- despite using a
  5x shorter context window (10 steps vs. 50) and the original paper's less
  GPU-friendly recurrence. On NASA the two are also close (15.91% vs. 15.20%),
  suggesting window length matters less than architecture for this task, at
  least in the 10-50 step range tested here.

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
