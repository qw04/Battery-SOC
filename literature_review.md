# Li-ion Battery SOC Estimation — Literature Review

Compiled from handwritten research notes. 16 papers identified (2013–2024), spanning classical ML, RNN/LSTM, and CNN/hybrid Kalman-filter approaches to State-of-Charge (SOC) estimation. Goal: use this to select methods + a dataset for a Python implementation.

---

## 1. Classical ML (SVM / ELM / RF / GPR)

### 1.1 Support Vector Machines Used to Estimate the Battery State of Charge (2013)
- **Authors**: J.C. Álvarez Antón, P.J. García Nieto, C. Blanco Viejo, J.A. Vilán Vilán
- **Venue**: IEEE Transactions on Power Electronics, 28(12), 5919–5926
- **Link**: [DOI 10.1109/TPEL.2013.2243918](https://ieeexplore.ieee.org/document/6423937/)
- **Dataset**: High-capacity LiFePO₄ cell (60 Ah nominal); Dynamic Stress Test (DST) per USABC standard; charge/discharge cycling at 0.3C/0.33C to 3.6V then CV.
- **What's new**: One of the earliest applications of SVM regression (as opposed to NNs or Kalman filtering) to SOC estimation, using voltage/current/temperature as inputs. Achieves <6% error across both steady-state and dynamic (DST) current profiles — establishes SVM as a viable model-free baseline.

### 1.2 Battery State-of-Charge Estimation Based on Regular/Recurrent Gaussian Process Regression (2018)
- **Authors**: Gozde O. Sahinoglu, Milutin Pajovic, Zafer Sahinoglu, Yebin Wang, Philip V. Orlik, Toshihiro Wada (Mitsubishi Electric Research Labs)
- **Venue**: IEEE Transactions on Industrial Electronics, 65(5), 4311–4321
- **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/8074792/) · [MERL technical report](https://www.merl.com/publications/docs/TR2017-124.pdf)
- **Dataset**: Three datasets, not one: (1) a **simulated** ECM (two-RC + series-resistance) dataset; (2) an **in-house** Mitsubishi Electric LiMn₂O₄/hard-carbon cell (4.93 Ah), constant 10 C-rate/h charge-discharge; (3) the **CALCE INR 18650-20R** public dataset (2 Ah, NMC/graphite) under FUDS/US06 dynamic driving — the same public cell used for validation in §3.1 (NNGP) below.
- **What's new**: Proposes two variants — "regular" GPR (V/I/T as inputs) and "recurrent" GPR (feeds the previous SOC estimate back into the input vector), plus an "autoregressive recurrent" GPR variant that also uses past V/I/T. Unlike SVM/NN point estimates, GPR is Bayesian/non-parametric and natively outputs **uncertainty bounds** alongside the SOC estimate, useful for reliability-aware BMS design. Best result: RMSE < 0.14%, MAE < 0.36% on the dynamic (FUDS/US06) test.

### 1.3 Extreme Learning Machine for SOC Estimation of Lithium-ion Battery Using GSA (2018)
- **Authors**: M.S.H. Lipu, M.A. Hannan, A. Hussain, M.H. Saad
- **Venue**: IEEE PEDS Conference 2018 (journal extension in IEEE Trans. Industry Applications, 2019)
- **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/8544607/)
- **Dataset**: Beijing Dynamic Stress Test (BJDST) + US06 drive cycle; current/voltage/temperature at 25°C and 45°C.
- **What's new**: Uses the **Gravitational Search Algorithm (GSA)** — a physics-inspired metaheuristic — to automatically find the optimal number of hidden-layer neurons for an Extreme Learning Machine, removing manual/trial-and-error tuning. No internal electrochemical battery model needed.

### 1.4 Neural Network Approach for Estimating SOC of Li-ion Battery Using Backtracking Search Algorithm (2018)
- **Authors**: M.A. Hannan, M.S.H. Lipu, A. Hussain, M.H. Saad, A. Ayob
- **Venue**: IEEE Access, 6, 10069–10079
- **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/8269299/)
- **Dataset**: 18650 NMC (LiNiMnCoO₂) cell (CALCE), 2.5–4.2V, ≤22A; DST + Federal Urban Driving Schedule (FUDS); 0°C/25°C/45°C.
- **What's new**: Uses the **Backtracking Search Algorithm (BSA)**, an evolutionary metaheuristic, to jointly optimize *both* the number of hidden-layer neurons **and** the learning rate of a BPNN simultaneously (rather than tuning each separately). MAE of 0.87%/0.59%/0.38% at the three temperatures on DST.

### 1.5 State of Charge Estimation for Li-ion Batteries Based on Improved Barnacle Mating Optimizer and SVM (2022)
- **Authors**: B. Liu, H. Wang, M.-L. Tseng, Z. Li
- **Venue**: Journal of Energy Storage, 55, 105830
- **Link**: [DOI 10.1016/j.est.2022.105830](https://www.sciencedirect.com/science/article/abs/pii/S2352152X22018187)
- **Dataset**: In-house 18650 ternary lithium-ion (NMC) cell, 2000 mAh rated / 3.7 V nominal (4.2 V charge cutoff, 2.5 V discharge cutoff). 32,851 samples at 1 Hz, ~24°C ambient, working-condition (not a standard drive-cycle) profile; randomly shuffled 50/50 train/test split. **Not a public dataset.**
- **What's new**: Introduces an **Improved Barnacle Mating Optimizer (IBMO)** — enhances the base Barnacles Mating Optimizer metaheuristic with cubic chaotic mapping, a hyperbolic-sinusoidal conditioning factor, and Gauss-Cauchy mutation — then uses IBMO to tune SVM hyperparameters. Reports RMSE 0.0042, MAPE 0.61%, R²=0.9994, beating plain BMO-SVM.

### 1.6 Real-Time SOC Estimation Using Optimized Random Forest Regression Algorithm (2023)
- **Authors**: M.S. Hossain Lipu, M.A. Hannan, A. Hussain, S.A. Ansari, S.A. Rahman, Hamdan Mohamad, Kashem M. Muttaqi
- **Venue**: IEEE Transactions on Intelligent Vehicles, 8(1), 639–648
- **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9740509/)
- **Dataset**: EV battery voltage/current sensor data (no internal chemistry model).
- **What's new**: Uses a **Differential Search Algorithm (DSA)** to optimize Random Forest hyperparameters (tree count/depth). Claims it eliminates the noise-filtering preprocessing step that earlier ML SOC methods needed, while staying scalable — pitched explicitly for real-time onboard BMS deployment.

---

## 2. LSTM / RNN-based

### 2.1 Long Short-Term Memory Networks for Accurate SOC Estimation of Li-ion Batteries (2018)
- **Authors**: Ephrem Chemali, Phillip J. Kollmeyer, Matthias Preindl, Ryan Ahmed, Ali Emadi
- **Venue**: IEEE Transactions on Industrial Electronics, 65(8), 6730–6739
- **Link**: [DOI 10.1109/TIE.2017.2787586](https://doi.org/10.1109/TIE.2017.2787586)
- **Dataset**: McMaster University Hybrid Electric Vehicle & Battery Lab (LG 18650 cell); 0°C, 25°C, 10→25°C ramp; 1 Hz sampling; SOC ground truth via coulomb counting.
- **What's new**: One of the first papers to show an LSTM can map V/I/T **directly** to SOC with *no* battery model, filter, or auxiliary inference system — outperforming AEKF+ANN, AUKF+LSSVM, fuzzy-NN+GA and RBFNN baselines. Foundational paper for the whole "pure deep-learning SOC" line of work below.

### 2.2 Online Joint-Prediction of Multi-Forward-Step Battery SOC Using LSTM and Multiple Linear Regression (2020)
- **Authors**: Jichao Hong, Zhenpo Wang, Wen Chen, Le Yao Wang, Chao Qu
- **Venue**: Journal of Energy Storage, 30, 101459
- **Link**: [DOI 10.1016/j.est.2020.101459](https://www.sciencedirect.com/science/article/abs/pii/S2352152X20300396)
- **Dataset**: Year-long real-world EV fleet data (V/I/SOC/speed/weather), tied to a Beijing EV monitoring center.
- **What's new**: LSTM does offline multi-step-ahead SOC prediction; correlation analysis strips redundant inputs (e.g. pack voltage, prior SOC); an online **multiple linear regression correction layer** jointly manages the accuracy/horizon tradeoff in real time — useful pattern for online drift-correction, not just pure deep learning.

### 2.3 Convolutional Gated Recurrent Unit–RNN for SOC Estimation of Li-ion Batteries (2019)
- **Authors**: Zhelin Huang, Fangfang Yang, Fan Xu, Xiangbao Song, Kwok-Leung Tsui
- **Venue**: IEEE Access, 7, 93139–93149
- **Link**: [DOI 10.1109/ACCESS.2019.2928037](https://doi.org/10.1109/ACCESS.2019.2928037)
- **Dataset**: BAK 18650 cell (Li(NiCoMn)O₂); FUDS + DST profiles, multiple temperatures.
- **What's new**: CNN-GRU encoder-decoder compresses sequential V/I inputs and estimates SOC end-to-end, beating plain RNN, GRU, SVM, and ELM baselines — explicitly framed as removing the need for a separate Kalman-filter stage.

### 2.4 SOC Estimation of Li-ion Batteries Using LSTM and UKF (2020)
- **Authors**: Fangfang Yang, Shaohui Zhang, Weihua Li, Qiang Miao
- **Venue**: Energy, 201, 117664
- **Link**: [DOI 10.1016/j.energy.2020.117664](https://doi.org/10.1016/j.energy.2020.117664)
- **Dataset**: DST, FUDS, US06 profiles, 0°C–50°C.
- **What's new**: Unlike 2.1–2.3 (which drop Kalman filtering entirely), this hybridizes it back in: the LSTM serves as the **measurement equation inside an Unscented Kalman Filter**, so UKF's recursive filtering cleans up LSTM output noise. Achieves RMSE < 1.1% across a wide temperature range — a good comparison point for "pure NN vs. NN+KF hybrid."

### 2.5 State of Charge Estimation Based on Temporal Convolutional Network and Transfer Learning (2021)
- **Authors**: Yuefeng Liu, Jiaqi Li, Gong Zhang, Bin Hua, Neal N. Xiong
- **Venue**: IEEE Access, 9, 34177–34187
- **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9348917/) · [DOI 10.1109/ACCESS.2021.3057371](https://doi.org/10.1109/ACCESS.2021.3057371)
- **Dataset**: **Panasonic 18650PF** (NCA) cell — the public Kollmeyer/Mendeley dataset (DOI [10.17632/WYKHT8Y7TG.1](https://doi.org/10.17632/wykht8y7tg.1)). Nine drive cycles (US06, HWFET, UDDS, LA92, 4 hybrid mixes, 1 custom "Cycle NN") at 25°C/10°C/0°C. Transfer-learning target is also public: **Turnigy Graphene 5000mAh 65C** cell (Kollmeyer/Mendeley, DOI [10.17632/4FX8CJPRXM.1](https://doi.org/10.17632/4fx8cjprxm.1)).
- **What's new**: Uses a **Temporal Convolutional Network** (causal + dilated convolutions, no lookahead leakage) instead of a recurrent architecture, direct-mapping V/I/T to SOC. Adds **transfer learning** so a model trained on one cell/condition can be fine-tuned to a new cell/condition with far less data — directly relevant to your dataset-selection question, since it addresses cross-dataset generalization. Average MAE 0.67% across all tests; TCN beats LSTM-RNN and GRU-RNN baselines on the same data.
- *Note: `09348917.pdf` in your Downloads is a duplicate of this same paper (same DOI/authors), not a distinct one.*

### 2.6 Uncorrelated Sparse Autoencoder with LSTM for SOC Estimation in Li-ion Battery Cells (2024)
- **Authors**: Mayuresh Savargaonkar, Isaiah Oyewole, Abdallah Chehade, Ala A. Hussein
- **Venue**: IEEE Transactions on Automation Science and Engineering, 21, 15–26
- **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9959882/)
- **Dataset**: Same three public accelerated-aging datasets as §3.1 (same author group): **NASA Ames** (Cell-B0005 for tuning, Cell-B0007 for test); **Severson et al.** fast-charging set (124 LFP/Graphite A123 APR18650M1A cells — Cell-11, Cell-77 used); **Diao et al.** set (192 cells, full-factorial aging design — Cell-137 used). Plus a fourth public dataset for the dynamic-loading validation: the **CALCE INR 18650-20R** (2 Ah, NMC/graphite) set under US06/FUDS/DST/BJDST — the same CALCE dataset confirmed in §1.2 and §3.1.
- **What's new**: "USAL" — a sparse autoencoder with an explicit **decorrelation penalty** (fights multicollinearity in the latent space), trained jointly with an LSTM in a multi-task setup. Learns compact, non-redundant, SOC-informative encodings while capturing long/short-term temporal correlation — aimed at long-horizon SOC estimation from limited initial cycling history (trained on only the first 5 charge-discharge cycles, tested on the final 20 near end-of-life). Outperforms DNN, plain LSTM, and a prior Autoencoder-LSTM baseline; e.g. MAE 0.88% vs. 2.1% for Autoencod-LSTM on Cell-77. Most recent/most complex paper in the set.

---

## 3. CNN / Hybrid (Kalman filter, Gaussian process, U-Net)

### 3.1 A Novel NN with Gaussian Process Feedback for Modeling the SOC of Battery Cells (2022)
- **Authors**: Mayuresh Savargaonkar, Abdallah A. Chehade, Ala A. Hussein
- **Venue**: IEEE Transactions on Industry Applications, 2022
- **Link**: [DOI 10.1109/TIA.2022.3170842](https://doi.org/10.1109/TIA.2022.3170842)
- **Dataset**: Four cells from three public accelerated-aging datasets, each a different cathode/anode chemistry: **Cell-B0007** (NASA Ames Prognostics Center); **Cell-14** and **Cell-19**, A123 Systems APR18650M1A cells (Diao et al. dataset, 45°C/60°C); **Cell-9**, from the Severson et al. fast-charging dataset. Plus a second validation on the public **CALCE INR 18650-20R** (2 Ah) dataset under US06/FUDS/DST dynamic driving.
- **What's new**: "NNGP" — a deep NN whose raw predictions get corrected by a **Gaussian process feedback loop** that models SOC trend correlation across cycles (using available energy, not time, as the covariate), plus adaptive weighted training to handle aging drift. MAE < 3.5% for 25-cycle-ahead forecasting on the aging datasets; MAE always < 0.95% on the CALCE dynamic-driving benchmark. (Same author group as 2.6 — worth reading together.)

### 3.2 A Novel Deep Neural Network Model for Estimating the SOC of Lithium-ion Battery (2022)
- **Authors**: Qingrui Gong, Ping Wang, Ze Cheng
- **Venue**: Journal of Energy Storage, 54, 105308
- **Link**: [DOI 10.1016/j.est.2022.105308](https://doi.org/10.1016/j.est.2022.105308)
- **Dataset**: NASA Randomized Battery Usage Dataset (NASA Ames) + Oxford Battery Degradation Dataset — **both public, good candidates for your implementation.**
- **What's new**: Pipeline is **Conv1D → ULSAM (Ultra-Lightweight Subspace Attention Module) → SRU (Simple Recurrent Unit) → Dense**. ULSAM sharpens the CNN's feature extraction with lightweight attention; SRU gives cheap sequence memory (faster than LSTM/GRU). Reports max error ~4.3% across both datasets despite differing degradation/temperature/discharge conditions — good robustness benchmark.

### 3.3 SOC Estimation of Li-ion Battery Using CNN with U-Net Architecture (2022)
- **Authors**: Xinyuan Fan, Weige Zhang, Caiping Zhang, Anci Chen, Fulai An
- **Venue**: Energy, 256, 124612
- **Link**: [DOI 10.1016/j.energy.2022.124612](https://doi.org/10.1016/j.energy.2022.124612)
- **Dataset**: In-house dynamic drive-cycle data at 5 constant ambient temperatures.
- **What's new**: Pure CNN, U-Net-style encoder-decoder — estimates SOC directly with no iterative/recursive convergence loop. Introduces **symmetric padding** convolutions (removes edge artifacts at the start/end of a sequence) and a **total variation loss** term that smooths the output without added model complexity. MAE ≤ 1.1%, RMSE ≤ 1.4% at constant temperature — one of the tightest error bounds in this set.

### 3.4 Deep CNN-based Closed-Loop SOC Estimation for Li-ion Batteries in Hierarchical Scenarios (2023)
- **Authors**: Qiao Wang, Min Ye, Meng Wei, Gaoqi Lian, Yan Li
- **Venue**: Energy, 263 (Part B), 125718
- **Link**: [DOI 10.1016/j.energy.2022.125718](https://doi.org/10.1016/j.energy.2022.125718)
- **Dataset**: In-house data across three cell types — **INR-18650** (ternary/NMC), **NCR-18650** (ternary/NMC, fresh + aged to 88% capacity), **LFP-26650** (LiFePO₄, fresh + aged to 88% capacity) — under DST/BJDST/US06/FUDS/UDDS at 25°C plus mixed drive cycles across a 10–40°C range. **Released publicly by the authors on Mendeley: [data.mendeley.com/datasets/rnvb47fp4t](https://data.mendeley.com/datasets/rnvb47fp4t)** — a strong, ready-to-use candidate for the implementation dataset list below.
- **What's new**: A 2D-CNN builds a compact pre-trained "universal" SOC feature extractor (conv + avg pooling); a **closed-loop correction** feeds its output into **Kalman filter measurement equations**; **transfer learning + pruning** let the pre-trained model adapt fast across the "hierarchy" (different cell types / aging states / operating scenarios). RMSE < 2.47% generally, < 1.78% under severe disturbance.
  - *Note: your handwritten notes had two entries here ("closed-loop hierarchical" and "Deep CNN used for estimating Kalman filter") — after research these appear to be the same paper described twice, not two distinct papers. Flag if you have a separate citation in mind.*

---

## Implementation Notes (for the Python build)

**Public datasets referenced across these papers** (best candidates to start with, roughly easiest → richest):
- **NASA Randomized Battery Usage Dataset** (NASA Ames) — used in 3.2; NASA's B0007 accelerated-aging cell also used in 3.1
- **Oxford Battery Degradation Dataset** — used in 3.2
- **CALCE 18650 NMC data** (incl. INR 18650-20R, 2Ah) — used in 1.4 (DST/FUDS, 3 temps), and confirmed used in 1.2 and 3.1 (US06/FUDS dynamic driving)
- **McMaster HEV & Battery Lab LG 18650 data** — used in 2.1 (the foundational LSTM paper; clean 1Hz V/I/T/SOC data, good first target)
- **Panasonic 18650PF (Kollmeyer/Mendeley)** — used in 2.5 (TCN); public, DOI 10.17632/WYKHT8Y7TG.1, 9 drive cycles × 3 temperatures — another clean, easy first target
- **A123 APR18650M1A (Diao et al.)** and **Severson et al. fast-charging dataset** — used in 3.1 (NNGP) and 2.6 (USAL), aging-focused
- **Wang et al. multi-chemistry dataset (Mendeley)** — used in 3.4; INR-18650/NCR-18650/LFP-26650 fresh + aged, wide temp range — richest of the newly-confirmed public sets, good candidate for the "push accuracy" / generalization stage

**A natural build order**, roughly matching increasing complexity:
1. **Baseline**: SVM or Random Forest on V/I/T features (§1.1, §1.6) — cheap sanity-check model.
2. **Core deep learning**: LSTM direct-mapping, replicating §2.1 (McMaster paper) — this is the most "classic," well-documented starting point with a clean public-ish dataset.
3. **Add the Kalman hybrid**: LSTM+UKF (§2.4) as a second model on the same data, to compare pure-NN vs. NN+filter.
4. **Push accuracy**: try the CNN/U-Net (§3.3) or Conv+ULSAM+SRU (§3.2) architectures on NASA/Oxford data for a tighter error bound.
5. **Stretch goal**: TCN + transfer learning (§2.5) if you want to test cross-cell/cross-condition generalization, which matters if you eventually want one model to work across multiple battery chemistries.

All 16 papers are now verified against full text. GPR §1.2, TCN §2.5, NNGP §3.1, USAL §2.6, and DCNN+Kalman §3.4 all turned out to use public datasets (several the same CALCE/NASA/Severson/Diao sets, cross-referenced above), so they're stronger candidates than originally thought. IBMO-SVM §1.5 is confirmed in-house/non-public, so it stays lower priority for direct reuse.
