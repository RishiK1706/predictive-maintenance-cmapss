# PART A : SYNOPSIS TEMPLATE (Phase 1, due Tue, 25 Aug 2026)

**Team No.:** [Insert Team Number]  
**Team Members & Registration Numbers:**
- **Member 1**: Mayurika Sathish — MLP (Baseline) & Feature Engineering Lead
- **Member 2**: Sachith V P — 1D-CNN Spatial/Temporal Feature Extraction Lead
- **Member 3**: Ishanvi Kaushik — LSTM Sequential Memory & Trajectory Lead
- **Member 4**: Rishi Khandelwal — Transformer Encoder Multi-Head Attention Lead

---

### Title of the Project
**Comparative Analysis of MLP, 1D-CNN, LSTM, and Transformer Architectures for Predictive Maintenance and Remaining Useful Life Estimation of Industrial Machinery**

---

### Brief Description of the Project (max 500 words)
Industrial machinery failures cause significant economic losses and safety hazards across sectors such as manufacturing, aviation, power generation, and logistics. Unplanned downtime costs the global manufacturing industry an estimated $50 billion annually. Traditional maintenance strategies, either time-based (scheduled) or reactive (fix-on-failure), are either wasteful or dangerously late. Predictive Maintenance (PdM) addresses this by using historical sensor data to estimate when a machine is likely to fail, enabling intervention at the right time.

This project applies deep learning to the problem of Remaining Useful Life (RUL) estimation, predicting how many operational cycles a machine has left before failure, using multi-sensor time-series data recorded from industrial engines. The input to each model is a window of recent sensor readings, including temperature, pressure, fan speed, physical core speed, and other operational parameters. The output is the estimated RUL as a continuous value. This is a supervised regression task, evaluated using standard error metrics on a held-out test set.

The core contribution of this project is a systematic comparison of four architecturally distinct deep learning approaches applied to the same dataset under identical experimental conditions:
1. **MLP (Multilayer Perceptron)**: A fully connected baseline receiving statistical features over a window and establishing a performance floor.
2. **1D-CNN**: Applies convolutional filters directly over the sensor time window to automatically extract local temporal patterns.
3. **LSTM**: Processes the sensor sequence step-by-step, capturing long-range temporal dependencies in the degradation trend.
4. **Transformer Encoder**: Applies self-attention over the sensor sequence so each time step attends to all others, capturing global dependencies and representing a modern approach to time-series modeling.

All four models are trained and evaluated on the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, a benchmark widely used in prognostics research. A common train/validation/test split and common evaluation metrics, including RMSE, MAE, $R^2$, and the NASA scoring function, are used across all models to ensure a fair and meaningful comparison.

---

### Dataset(s) Identified
* **Dataset Name**: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan Engine Degradation Simulation.
* **Source**: NASA Ames Prognostics Data Repository (Kaggle: `behrad3d/nasa-cmaps`).
* **Subsets**: `FD001` (100 training engines, 100 test engines, single operating condition, HPC degradation fault mode).
* **Sensors & Features**: 21 sensor measurements + 3 operational settings per cycle (128–362 cycles per engine run-to-failure).
* **Preprocessing**: Normalization (MinMax fitted strictly on train engines), 30-cycle sliding window segmentation, and RUL label clipping at 125 cycles.

---

### Proposed Models for Comparison

| # | Model | Architecture Family | Brief Description |
| :---: | :--- | :--- | :--- |
| **1** | **MLP** | Classical / Fully-Connected Baseline | Flattened window fed into dense layers with ReLU activations, BatchNorm, and dropout. |
| **2** | **1D-CNN** | Convolutional Neural Network | Conv1D layers with 3x3 kernels over sliding window of raw sensor readings; extracts local temporal patterns followed by pooling and regression head. |
| **3** | **LSTM** | Recurrent / Sequential | Stacked LSTM layers (2 layers, hidden size 64) processing normalized sensor readings step-by-step; captures long-range degradation trends. |
| **4** | **Transformer Encoder** | Attention-Based | Multi-head self-attention encoder with positional encoding applied to the sensor sequence; output pooled and passed to regression head. |

---

### Model-to-Member Assignment

| Member | Name | Assigned Model | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | Mayurika Sathish | **MLP (Baseline)** | Feature engineering, normalization, MLP design, training, evaluation. |
| **Member 2** | Sachith V P | **1D-CNN** | Sliding window data loader, Conv1D architecture design, hyperparameter tuning. |
| **Member 3** | Ishanvi Kaushik | **LSTM** | Sequence preparation, stacked LSTM implementation, gradient clipping, tuning. |
| **Member 4** | Rishi Khandelwal | **Transformer Encoder** | Positional encoding, multi-head self-attention implementation, training, tuning. |

---

### Existing State-of-the-Art Method(s) Related to the Proposed Project
1. **Deep LSTM for RUL Estimation (Zheng et al., 2017)**: One of the earliest works applying LSTM to C-MAPSS, achieving RMSE of ~12–16 on FD001.
2. **1D-CNN for Prognostics (Li et al., 2019)**: Demonstrated that 1D-CNNs outperform LSTMs by capturing local degradation patterns efficiently.
3. **Temporal Fusion Transformer for Multi-horizon Forecasting (Lim et al., 2021)**: Proposed a Transformer-based architecture for time-series forecasting, achieving state-of-the-art results.
4. **Bidirectional LSTM + CNN Hybrid (Listou Ellefsen et al., 2019)**: Combined CNN feature extraction with LSTM sequence modeling.

---

### Objectives
1. To quantitatively compare MLP, 1D-CNN, LSTM, and Transformer encoder architectures for Remaining Useful Life estimation on a common industrial sensor dataset under identical experimental conditions.
2. To determine which deep learning architecture family offers the best trade-off between predictive accuracy and computational efficiency for time-series-based predictive maintenance.
3. To analyze the failure modes and error distributions of each architecture, identifying scenarios (early degradation vs. near-failure) where different models excel or underperform.

---

### Importance / Motivation of the Proposed Project
Predictive maintenance is one of the most economically impactful applications of AI in industry. A 1% reduction in unplanned downtime in manufacturing saves an average enterprise millions of dollars annually. As Industry 4.0 and IoT adoption accelerates, machines generate continuous streams of sensor data that create an unprecedented opportunity for data-driven health monitoring without requiring any additional hardware beyond existing sensors.

This project is motivated by a gap in the literature: while individual architectures (CNN, LSTM, Transformer) have each been applied to PdM in isolation, head-to-head comparisons on the same dataset under identical conditions are rare. Understanding which architecture is most suitable—and why—has direct implications for deployment in resource-constrained industrial environments.

---

### Preliminary Reference List
1. **Saxena, A., Goebel, K., Simon, D., & Eklund, N.** (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *IEEE International Conference on Prognostics and Health Management (PHM)*.
2. **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C.** (2017). Long short-term memory network for remaining useful life estimation. *IEEE International Conference on Prognostics and Health Management*.
3. **Li, X., Ding, Q., & Sun, J. Q.** (2019). Remaining useful life estimation in prognostics using deep convolutional neural networks. *Reliability Engineering & System Safety*, 172, 1–11.
4. **Vaswani, A., et al.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*.
5. **Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T.** (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*.
6. **Hochreiter, S., & Schmidhuber, J.** (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

---

### Work Plan

| Sl. No. | Component / Work Element / Milestone | Expected Completion Date |
| :---: | :--- | :---: |
| 1 | Dataset download, exploration, and understanding of C-MAPSS structure | 28 Aug 2026 |
| 2 | Common data preprocessing pipeline: normalization, sliding window, RUL label clipping | 05 Sep 2026 |
| 3 | Literature review: identify and summarize 8–10 papers in table format | 08 Sep 2026 |
| 4 | MLP baseline: feature engineering + model implementation + preliminary results | 12 Sep 2026 |
| 5 | 1D-CNN: data loader + Conv1D architecture + preliminary results | 15 Sep 2026 |
| 6 | LSTM: sequence model implementation + preliminary results | 18 Sep 2026 |
| 7 | Transformer Encoder: attention model + positional encoding + preliminary results | 22 Sep 2026 |
| 8 | Interim report (Part B) compilation + individual contribution log | 25 Sep 2026 |
| 9 | Hyperparameter tuning and optimization for all 4 models | 05 Oct 2026 |
| 10 | Final comparative evaluation: common metrics table, error analysis, plots | 12 Oct 2026 |
| 11 | Demo / visualization: Streamlit dashboard showing sensor input $\rightarrow$ RUL prediction | 18 Oct 2026 |
| 12 | Final report writing (all sections, per-member methodology subsections) | 24 Oct 2026 |
| 13 | Presentation slides preparation + rehearsal | 27 Oct 2026 |
| 14 | Final report + code repository submission | 31 Oct 2026 |
