# C-MAPSS Predictive Maintenance: Complete Final Demo & Presentation Guide
**Course Code:** ICT-4442 Deep Learning Mini Project  
**Team Members:**
1. Mayurika Sathish (Reg No. 230953140) — *MLP Baseline & Data Preprocessing*
2. Sachith V P (Reg No. 230953202) — *1D-CNN & Error Failure Analysis*
3. Ishanvi Kaushik (Reg No. 230953082) — *Stacked LSTM & Results Benchmark*
4. Rishi Khandelwal (Reg No. 230953494) — *Transformer Encoder & Live Dashboard Demo*

---

## ⏱️ Recommended Presentation Timing (Total: 12 – 14 Minutes)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  00:00 - 03:00  │  Member 1: Mayurika  │ Problem, Preprocessing & MLP Baseline│
│  03:00 - 06:00  │  Member 2: Sachith   │ 1D-CNN Architecture & Failure Cases │
│  06:00 - 09:30  │  Member 3: Ishanvi   │ Stacked LSTM & Comparative Results  │
│  09:30 - 13:00  │  Member 4: Rishi     │ Transformer & Live Dashboard Demo   │
│  13:00 - 15:00  │  All Members         │ Conclusion, Future Work & Q&A       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Member-by-Member Speaking Scripts & Flow

---

### PART 1: Mayurika Sathish (Member 1) — [00:00 – 03:00]
**Role:** Project Introduction, Leakage-Free Preprocessing, and MLP Baseline

#### 🎙️ Speaking Script:
> "Good morning, respected evaluators. Today, our team presents our deep learning project: **'Comparative Analysis of MLP, 1D-CNN, LSTM, and Transformer Architectures for Predictive Maintenance: Health State Classification and Remaining Useful Life Estimation of Industrial Machinery'** on the NASA C-MAPSS dataset.
> 
> Unplanned engine failures in commercial aviation cost millions in unscheduled downtime and pose extreme safety risks. Traditional maintenance is either reactive (fixing after failure) or scheduled (replacing parts too early). Our project uses multivariate sensor telemetry to predict Remaining Useful Life (RUL) and categorize engine health into three operational states: **Normal (>50 cycles)**, **Warning (20–50 cycles)**, and **Critical (≤20 cycles)**.
> 
> **Data Preprocessing Pipeline (My Contribution):**
> * **Zero-Variance Sensor Removal:** From 21 sensors and 3 operational settings, we removed 6 static sensors ($s_1, s_5, s_{10}, s_{16}, s_{18}, s_{19}$) with $\sigma \le 10^{-4}$. Sensor $s_6$ had variance and was kept, leaving **15 informative features**.
> * **Leakage-Free Engine Splitting:** Instead of random row shuffling—which causes data leakage across consecutive cycles—we split strictly by engine ID (80 training engines, 20 validation engines). MinMax scalers were fitted **only** on the training set.
> * **Piecewise Linear RUL Target:** In early life, machinery operates with zero measurable wear. Predicting an ever-changing RUL from healthy signals leads to false gradients. Following NASA literature, we capped RUL at **$R_{\text{max}} = 125$ cycles**.
> * **Sliding Windows:** Extracted $30$-cycle overlapping sequence windows with stride 1.
> 
> **My Model: Multi-Layer Perceptron (MLP Baseline)**
> * Flattens the $30 \times 15$ window into a **450-dimensional input vector**.
> * Architecture: Dense(450 $\to$ 256) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.2) $\to$ Dense(256 $\to$ 128) $\to$ Dense(128 $\to$ 64) $\to$ Dense(64 $\to$ 1).
> * **Results:** Converged very fast in **71.99 s**, achieving an **RMSE of 13.50 cycles** and **$R^2 = 0.8865$** (highest $R^2$ across all models).
> * **Trade-off:** Because it uses wide fully connected layers without temporal parameter sharing, it requires **157,569 parameters**—the largest memory footprint of all 4 models.
> 
> Now, Sachith will explain how temporal convolutions extract features with 3.8x fewer parameters."

#### 🎯 Likely Viva Questions for Mayurika:
* **Q: Why cap RUL at 125 cycles?**  
  *Answer:* Because healthy engines show no thermodynamic degradation in early life. Forcing a neural network to predict 250 vs 200 cycles from identical sensor signatures introduces high variance. A constant health threshold stabilizes training.
* **Q: Why is engine-wise splitting critical?**  
  *Answer:* If we perform row-wise random splitting, cycle $t$ of Engine #5 might be in train and cycle $t+1$ of Engine #5 in validation. Because sliding windows overlap, the model would simply memorize adjacent cycles instead of generalizing to unseen engines.

---

### PART 2: Sachith V P (Member 2) — [03:00 – 06:00]
**Role:** 1D-CNN Architecture, Receptive Field Analysis, and Verified Failure Cases

#### 🎙️ Speaking Script:
> "Thank you, Mayurika. Respected evaluators, I implemented the **1D Convolutional Neural Network (1D-CNN)** and conducted the in-depth error failure analysis across the test fleet.
> 
> **1D-CNN Architecture:**
> * While the MLP flattens the sequence and loses temporal locality, the 1D-CNN slides temporal convolution kernels across time steps.
> * Architecture: Conv1D(filters=32, kernel=3) $\to$ BatchNorm $\to$ ReLU $\to$ Conv1D(64, kernel=3) $\to$ MaxPool1D $\to$ AdaptiveAvgPool1D $\to$ Dense(64 $\to$ 32 $\to$ 1).
> * **Efficiency:** It requires only **41,153 parameters** (3.8x smaller than the MLP) and trained in just **64.46 seconds**, making it our most lightweight, edge-deployable model.
> 
> **Performance & Receptive Field Limitations:**
> * On continuous regression, the 1D-CNN had higher errors: **MAE = 14.32 cycles**, **RMSE = 20.41 cycles**, and a high **NASA score of 2215.1**.
> * *Why?* Standard kernel convolutions ($k=3$) have localized receptive fields. Without deep dilated layers, shallow convolutions cannot capture progressive, 30-cycle monotonic wear trends during early life.
> * However, on discrete early-warning alerts, it achieved a **Critical State F1 of 0.8750** and **Macro F1 of 0.8501**, correctly identifying 14 of 16 critical engines because terminal failure creates sharp, high-frequency sensor spikes that convolutional filters detect easily.
> 
> **Deep Error Failure Case Analysis (Section VI-C):**
> By inspecting our saved test predictions across all 100 engines, I isolated two major failure modes:
> 1. **Catastrophic Overestimation (Engine #41):**
>    * True RUL was **18 cycles** (Critical failure imminent).
>    * 1D-CNN predicted **87.67 cycles** (Error: **+69.67 cycles**), misclassifying it as **Normal/Safe**.
>    * Because NASA score penalizes late predictions exponentially ($s_i = \exp(d_i/10) - 1$), this single engine incurred a penalty of **$s_{41} \approx 1060.0$**, accounting for nearly **48% of the CNN's entire NASA score (2215.1)**! In real flight operations, this engine would cause an uncommanded failure.
> 2. **Premature Underestimation (Engine #93):**
>    * True RUL was **85 cycles** (Healthy), but the CNN predicted **28.38 cycles** (Error: **-56.62 cycles**), prematurely triggering a Warning alert. This causes unnecessary depot teardowns.
> 
> I now hand over to Ishanvi to present the Stacked LSTM, which overcomes these receptive field issues."

#### 🎯 Likely Viva Questions for Sachith:
* **Q: Why use 1D convolutions instead of 2D?**  
  *Answer:* Sensor telemetry is 1D across time for each channel. 2D convolutions assume spatial proximity between rows and columns (like pixels in an image). In sensor matrices, sensor #2 and sensor #3 have no physical spatial grid relationship, so 1D temporal filtering across the time axis is physically meaningful.
* **Q: Why did Engine #41 fail so badly in 1D-CNN?**  
  *Answer:* Engine #41 had subtle sensor drift without high-frequency spikes. The localized receptive field missed the gradual drop in HPC efficiency, causing prediction lag until late in the lifecycle.

---

### PART 3: Ishanvi Kaushik (Member 3) — [06:00 – 09:30]
**Role:** Stacked LSTM Gating, Parameter Optimization, and Benchmark Results Discussion

#### 🎙️ Speaking Script:
> "Thank you, Sachith. Respected evaluators, I designed and implemented the **Stacked Long Short-Term Memory (LSTM)** model, authored the Abstract, and compiled the comprehensive Results & Discussion section.
> 
> **Why LSTM for Turbofan Degradation?**
> Gas turbine degradation is an inherently cumulative, non-Markovian physical process. The current state of blade erosion depends on the entire operating history across past flight cycles. LSTMs maintain internal memory through three mathematical gates:
> 1. **Forget Gate ($f_t$):** Discards transient noise and flight turbulence.
> 2. **Input Gate ($i_t$):** Ingests persistent thermodynamic drift (e.g., rising HPC discharge temperature $T_{30}$ and fuel flow $W_{32}$).
> 3. **Cell State ($c_t$):** Acts as the continuous physical accumulator of wear.
> 4. **Output Gate ($o_t$):** Exposes current degradation to the hidden state.
> 
> **Architecture & Head Optimization:**
> * We stacked **2 unidirectional LSTM layers** (hidden dimension = 64) with inter-layer dropout (0.2).
> * The final cycle hidden state ($h_{30} \in \mathbb{R}^{64}$) summarizes the sequence.
> * To optimize memory, I implemented a **two-stage bottleneck regression head**: Linear(64 $\to$ 32) $\to$ ReLU $\to$ Dropout $\to$ Linear(32 $\to$ 1).
> * This reduced total parameters to **56,129** (a **64.4% reduction** compared to MLP) while training smoothly in **191.50 s**.
> 
> **Comprehensive Benchmark Discussion (Table III):**
> * Across all 100 unseen test turbofans, the Stacked LSTM achieved the **lowest Mean Absolute Error of 10.08 cycles**.
> * Most importantly, it achieved the **lowest NASA penalty score of 366.1** (compared to 368.6 for MLP, 416.4 for Transformer, and 2215.1 for 1D-CNN).
> * The NASA metric penalizes late predictions exponentially. The LSTM's lowest NASA score proves that its recurrent cell state acts as an effective accumulator of wear, preventing the network from dangerously overestimating engine life.
> * On early warning alerts, the LSTM correctly classified **66 of 67 Normal engines** and achieved a **Macro F1 of 0.8262**, maintaining excellent balance between regression accuracy and alert reliability.
> 
> Rishi will now demonstrate the Transformer Encoder and show our live interactive dashboard."

#### 🎯 Likely Viva Questions for Ishanvi:
* **Q: How does the LSTM avoid the vanishing gradient problem?**  
  *Answer:* In standard RNNs, gradients decay exponentially due to repeated matrix multiplications. In LSTMs, the cell state update is additive: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. This allows backpropagation gradients to flow through time without decaying to zero.
* **Q: Why is your NASA score the lowest?**  
  *Answer:* Because the recurrent cell state preserves cumulative degradation, the LSTM rarely overestimates remaining life near failure. Under the asymmetric NASA penalty ($s_i = \exp(d_i/10) - 1$), avoiding positive errors ($d_i > 0$) keeps the cumulative score low.

---

### PART 4: Rishi Khandelwal (Member 4) — [09:30 – 13:00]
**Role:** Transformer Encoder Architecture, Live Streamlit Dashboard Demo, and Conclusion

#### 🎙️ Speaking Script:
> "Thank you, Ishanvi. Respected evaluators, I implemented the **Transformer Encoder** architecture and developed our deployed **Streamlit Multi-Model Consensus Dashboard (`app.py`)**.
> 
> **Transformer Encoder Architecture:**
> * Recurrent models process cycles sequentially step-by-step ($O(W)$ path length). Transformers use **Multi-Head Self-Attention (MHSA)** to compute pairwise affinities across all 30 cycles simultaneously with an **$O(1)$ path length**.
> * Stage 1: Linear projection from 15 sensors to $d_{\text{model}} = 64$.
> * Stage 2: Injected **fixed sinusoidal positional encodings** to encode temporal cycle order.
> * Stage 3: 2 stacked Transformer Encoder layers with **4 attention heads**, feed-forward dimension 128, and post-LayerNorm.
> * Stage 4: Global average pooling over time followed by a 64 $\to$ 32 $\to$ 1 regression head (70,081 parameters).
> * **Performance:** Achieved an **MAE of 10.21 cycles**, $R^2 = 0.8740$, and tied with 1D-CNN for the highest **Critical State F1 (0.8750)**, detecting 14 of 16 critical engines. However, self-attention computation across all tokens increased CPU training time to **420.91 seconds**.
> 
> **Live Software Demonstration (`app.py`):**
> *(Screen share or show the browser window running Streamlit)*
> 
> 'I will now demonstrate our interactive dashboard running live on NASA C-MAPSS test data:
> 
> 1. **Engine Selection & Telemetry:** We can select any unseen test engine from 1 to 100 via the active fleet controller.
> 2. **Multi-Model Inference:** When an engine is selected, all 4 models run inference simultaneously in real-time.
> 3. **Case 1 — Healthy Asset (Engine #24):**
>    * Observed cycles: 186. True RUL is **82 cycles**.
>    * The dashboard shows consensus in the **🟢 Normal State (Safe)**.
> 4. **Case 2 — Critical Failure Asset (Engine #34):**
>    * True RUL is only **14 cycles**.
>    * Notice the live **🔴 Critical State Alert Badge** triggered immediately, advising maintenance crews to ground the aircraft and replace the HPC module.
> 5. **Multi-Sensor Degradation Trajectory:** Below, the plot shows real-time divergence in core sensors like $s_2, s_3, s_4, s_7,$ and $s_{11}$, giving engineers explainable visual evidence of thermodynamic wear.
> 6. **Consensus & Uncertainty:** The dashboard displays the standard deviation across all 4 models, providing operators with confidence bounds before making multi-million dollar maintenance decisions.'
> 
> **Conclusion:**
> Our findings confirm that while MLP offers fast tabular fitting and 1D-CNN provides ultra-compact parameter efficiency, the **Stacked LSTM provides the optimal balance of prognostic accuracy (MAE 10.08) and safety (NASA score 366.1)**, while the Transformer excels at critical anomaly recall.
> 
> Thank you. We are now open for your questions."

#### 🎯 Likely Viva Questions for Rishi:
* **Q: Why are sinusoidal positional encodings necessary in Transformers?**  
  *Answer:* Multi-Head Self-Attention is permutation-equivariant: shuffling the input sequence rows produces the exact same output. Because sensor degradation depends strictly on chronological time, sinusoidal positional encodings must be added to inject cycle order into the embeddings.
* **Q: Why did Transformer take the longest to train (420.91 s)?**  
  *Answer:* Computing scaled dot-product attention matrices ($Q K^T / \sqrt{d_k}$) across 4 heads, projecting through linear feed-forward blocks, and running layer normalizations requires more floating-point operations per batch than recurrent gating or convolutional kernels on CPU.

---

## 💻 Step-by-Step Instructions to Run the Demo

### Step 1: Open Terminal in Project Root
```powershell
cd "d:\DL Project"
```

### Step 2: Start Streamlit App
```powershell
python -m streamlit run app.py
```
*If a browser does not open automatically, open Chrome and navigate to:*  
`http://localhost:8501`

### Step 3: Best Test Engines to Demonstrate to Evaluators

| Demo Scenario | Engine ID | True RUL | Alert State | What to Highlight to Faculty |
| :--- | :---: | :---: | :---: | :--- |
| **Pristine / Healthy** | **Engine #24** | **82.0 cycles** | 🟢 **Normal** | Models agree ($>50$ cycles). Sensor trajectories show steady horizontal baseline. |
| **Accelerated Wear** | **Engine #91** | **38.0 cycles** | 🟡 **Warning** | Degradation detected ($20 < \text{RUL} \le 50$). Turnaround depot inspection recommended. |
| **Imminent Failure** | **Engine #34** | **14.0 cycles** | 🔴 **Critical** | True emergency ($y \le 20$). Demonstrates high Critical F1 detection and multi-sensor thermal spike ($s_2, s_4$). |
| **Failure Analysis Proof** | **Engine #41** | **18.0 cycles** | 🔴 / 🟢 Discrepancy | Proves Sachith's failure analysis: CNN overestimates at 87.67 (Normal), while LSTM correctly identifies high risk! Evaluators love this level of depth. |

---

## 📊 Quick Master Cheat Sheet for All Members

Keep this table handy during the presentation:

| Architecture | Test MAE | Test RMSE | $R^2$ Score | NASA Score | Trainable Params | Training Time | Macro $F_1$ | Critical $F_1$ | Lead Member |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MLP Baseline** | 10.21 | **13.50** | **0.8865** | 368.6 | 157,569 | 71.99 s | 0.8010 | 0.8000 | **Mayurika** |
| **1D-CNN** | 14.32 | 20.41 | 0.7407 | 2215.1 | **41,153** | **64.46 s** | **0.8501** | **0.8750** | **Sachith** |
| **Stacked LSTM** | **10.08** | 13.65 | 0.8839 | **366.1** | 56,129 | 191.50 s | 0.8262 | 0.7857 | **Ishanvi** |
| **Transformer** | 10.21 | 14.22 | 0.8740 | 416.4 | 70,081 | 420.91 s | 0.8299 | **0.8750** | **Rishi** |
