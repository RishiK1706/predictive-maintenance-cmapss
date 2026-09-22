# PART B : INTERIM REPORT TEMPLATE (Phase 2, due Fri, 25 Sept 2026)

**Team No.:** [Insert Team Number]  
**Team Members & Registration Numbers:**
- **Member 1**: Mayurika Sathish [Reg. No.] — MLP Baseline & Feature Engineering Lead
- **Member 2**: Sachith V P [Reg. No.] — 1D-CNN & Data Loader Lead
- **Member 3**: Ishanvi Kaushik [Reg. No.] — LSTM Sequential Memory & Trajectory Lead
- **Member 4**: Rishi Khandelwal (RishiK1706) [Reg. No.] — Transformer Encoder Attention & Evaluation Lead

**Title of the Project:** Comparative Analysis of MLP, 1D-CNN, LSTM, and Transformer Architectures for Predictive Maintenance and Remaining Useful Life Estimation of Industrial Machinery  
**GitHub Repository Link:** https://github.com/RishiK1706/predictive-maintenance-cmapss

---

## 1. Literature Review

The following table summarizes 10 key research papers in Prognostics and Health Management (PHM) and time-series degradation modeling on the NASA C-MAPSS benchmark dataset:

| Paper (Author, Year) | Method | Dataset | Key Result | Relevance to Project |
| :--- | :--- | :--- | :--- | :--- |
| **Saxena et al. (2008)** | C-MAPSS Simulation Framework | NASA C-MAPSS (FD001–FD004) | Established run-to-failure multivariate sensor dataset standard. | Primary benchmark dataset source and physical degradation background. |
| **Zheng et al. (2017)** | Stacked Long Short-Term Memory (LSTM) | C-MAPSS FD001 | Achieved RMSE ~12–16; proved sequential memory beats feedforward nets. | Baseline reference for Member 3's LSTM sequence design. |
| **Li et al. (2019)** | Deep 1D Convolutional Neural Network (1D-CNN) | C-MAPSS FD001–FD004 | Achieved RMSE 12.61; demonstrated 1D temporal convolutions extract local features without recurrence. | Foundation for Member 2's 1D-CNN architecture. |
| **Babu et al. (2016)** | Deep 2D-CNN on Time-Window Matrix | C-MAPSS FD001 | RMSE 18.4; showed spatial-temporal filter maps capture cross-sensor correlations. | Justification for time-windowing approach in dataset preparation. |
| **Ellefsen et al. (2019)** | Genetic Algorithm + RBM-LSTM Hybrid | C-MAPSS FD001–FD004 | RMSE 12.56 on FD001; highlighted importance of piecewise RUL target capping. | Direct rationale for setting piecewise linear RUL target $RUL_{max} = 125$. |
| **Mo et al. (2021)** | Dual-Attention Transformer Network | C-MAPSS FD001 & FD004 | Outperformed LSTMs under variable operating conditions; RMSE 11.8 on FD001. | Motivation for Member 4's Transformer Encoder self-attention design. |
| **Lim et al. (2021)** | Temporal Fusion Transformer (TFT) | Time-series benchmarks | State-of-the-art multi-horizon forecasting with interpretable attention weights. | Theoretical foundation for multi-head self-attention on sensor sequences. |
| **Vaswani et al. (2017)** | Transformer Encoder with Multi-Head Attention | Sequential benchmarks | Replaced recurrence with self-attention mechanism $O(1)$ sequential path length. | Core architecture implemented by Member 4 for sequence modeling. |
| **Wang et al. (2020)** | Multi-Scale 1D-CNN with Attention | C-MAPSS FD001 | RMSE 12.1; showed multi-scale kernel sizes extract multi-frequency sensor patterns. | Guidance for convolution kernel selection in 1D-CNN model. |
| **Hochreiter & Schmidhuber (1997)** | Long Short-Term Memory Architecture | General sequential modeling | Solved vanishing gradient problem in long sequences via gating units. | Theoretical basis for Member 3's LSTM memory cells. |

---

## 2. Dataset Acquisition & Preprocessing

### Dataset Summary
The primary dataset used is **NASA C-MAPSS FD001**, consisting of 100 training engines (20,631 total cycles) and 100 test engines (13,096 partial cycles). Each cycle contains 26 attributes: Engine ID, Operational Cycle, 3 Operational Settings, and 21 Sensor Measurements.

### Preprocessing Pipeline
1. **Uninformative Feature Removal**: Calculated standard deviation per sensor across all engine cycles. Dropped 7 zero-variance / constant sensors (`s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`), reducing sensor feature dimension from 21 to 15.
2. **Engine-Wise Train/Validation Splitting**: Randomly split training unit IDs into 80% Train (80 engines, 14,241 windows) and 20% Validation (20 engines, 3,490 windows). **No random row-level splitting** was used, eliminating data leakage across adjacent sliding windows.
3. **MinMax Normalization**: Fitted `MinMaxScaler(feature_range=(-1, 1))` strictly on the 80 training engines. Applied the fitted scaler to transform validation and test sets.
4. **Piecewise RUL Target Capping**: Constructed target $RUL = \min(\text{max\_cycle} - \text{current\_cycle}, 125)$ to handle early-life steady states effectively.
5. **Sliding Window Sequence Generation**: Generated 30-cycle overlapping sequence windows. Input tensor shape: $(N, 30, 15)$.

---

## 3. Models Implemented So Far

All 4 proposed model architectures have been implemented, trained, and evaluated on NASA C-MAPSS FD001 under identical data splits:

| Model | Owner (Member) | Status | Preliminary Metrics (FD001 Test Set) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (Baseline)** | Mayurika Sathish | Completed | **MAE**: 8.92 cycles<br>**RMSE**: 12.13<br>**$R^2$**: 0.9083<br>**NASA Score**: 267.4 | Flattened 30x15 window fed into 3-layer Dense network with ReLU, BatchNorm, and Dropout (0.2). Fastest inference. |
| **1D-CNN** | Sachith V P | Completed | **MAE**: 14.35 cycles<br>**RMSE**: 18.89<br>**$R^2$**: 0.7778<br>**NASA Score**: 1184.6 | 3 Conv1D layers (32, 64, 128 filters) with MaxPool and AdaptiveAvgPool. Most parameter-efficient (41,153 params). |
| **LSTM** | Ishanvi Kaushik | Completed | **MAE**: 9.60 cycles<br>**RMSE**: 13.24<br>**$R^2$**: 0.8908<br>**NASA Score**: 334.2 | 2-layer Stacked LSTM (hidden dim 64) with recurrent dropout (0.2). **Best Early Warning F1 (0.8428)** and Critical F1 (0.8667). |
| **Transformer Encoder** | Rishi Khandelwal | Completed | **MAE**: 10.21 cycles<br>**RMSE**: 14.00<br>**$R^2$**: 0.8780<br>**NASA Score**: 412.5 | Multi-head self-attention (4 heads, d_model=64) with positional encoding and global average pooling. Strong global context capture. |

---

## 4. Individual Contribution Log (to date)

| Member Name | Reg. No. | Task(s) Completed | Signature |
| :--- | :--- | :--- | :--- |
| **Mayurika Sathish** | [Reg. No.] | • Feature variance analysis & constant sensor identification<br>• Implemented MLP baseline architecture (`src/models/mlp.py`)<br>• Hyperparameter tuning for dense layers & dropout<br>• Drafted Literature Review table & Synopsis | *Mayurika Sathish* |
| **Sachith V P** | [Reg. No.] | • Sliding-window sequence generator & data loader (`src/data_loader.py`)<br>• Implemented 1D-CNN architecture (`src/models/cnn1d.py`)<br>• Conv1D kernel size & pooling layer tuning<br>• Generated sensor degradation trajectory plots | *Sachith V P* |
| **Ishanvi Kaushik** | [Reg. No.] | • Engine-grouped train/val splitting logic (leakage prevention)<br>• Implemented Stacked LSTM sequential model (`src/models/rnn.py`)<br>• Recurrent dropout & hidden state pooling implementation<br>• Early warning operational alert classification metrics | *Ishanvi Kaushik* |
| **Rishi Khandelwal** | [Reg. No.] | • PyTorch Dataset & DataLoader module (`src/dataset.py`)<br>• Implemented Transformer Encoder architecture (`src/models/transformer.py`)<br>• Positional encoding & multi-head self-attention tuning<br>• Unified evaluation script (`src/evaluate.py`) & Streamlit demo app (`app.py`) | *Rishi Khandelwal* |

---

## 5. Risk / Plan for Remaining Work

### Potential Risks & Mitigation Strategies
1. **Overfitting on Small Benchmark Dataset**: Small training set (80 engines) can lead to memorization.
   * *Mitigation*: Implemented early stopping (patience=10), AdamW weight decay (1e-4), and dropout (0.2).
2. **Distribution Shift on Multi-Condition Data (`FD004`)**: Performance may drop under 6 operating conditions.
   * *Mitigation*: Perform zero-shot & fine-tuning evaluation on `FD004` during Phase 3.

### Timeline to Final Submission (Phase 3: 31 Oct 2026)

| Milestone / Task | Expected Completion | Responsible Member(s) |
| :--- | :---: | :--- |
| **Hyperparameter Fine-Tuning** (Grid search on window sizes 15, 30, 50) | 05 Oct 2026 | All Members |
| **Robustness Evaluation on FD004** (Multi-condition benchmark) | 12 Oct 2026 | Mayurika & Sachith |
| **Streamlit Interactive Demo Refinement** (`app.py` live sensor dashboard) | 18 Oct 2026 | Rishi & Ishanvi |
| **Final IEEE IEEE-Format Report Writing** (Part C Report, 12+ references) | 24 Oct 2026 | All Members |
| **Presentation Slides & Viva Defense Rehearsal** | 27 Oct 2026 | All Members |
| **Final Code & Report Submission** | 31 Oct 2026 | All Members |
