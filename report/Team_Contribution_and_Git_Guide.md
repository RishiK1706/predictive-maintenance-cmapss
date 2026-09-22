# 🚀 ICT-4442 Deep Learning Mini Project: Team PR Workflow & Contribution Guide

This guide explains how each team member submits their assigned model architecture via a GitHub Pull Request (PR) to establish clean, individual commit history on GitHub.

---

## 👥 Team Member Roles & Model Ownership

| Member | Name | Assigned Model Family | Key Files Owned |
| :--- | :--- | :--- | :--- |
| **Member 1** | Mayurika Sathish | **MLP Baseline (Fully-Connected)** | `src/models/mlp.py`<br>`notebooks/01_MLP_Baseline_Mayurika.ipynb` |
| **Member 2** | Sachith V P | **1D-CNN (Convolutional Network)** | `src/models/cnn1d.py`<br>`notebooks/02_1D_CNN_Sachith.ipynb` |
| **Member 3** | Ishanvi Kaushik | **Stacked LSTM (Recurrent/Sequential)** | `src/models/rnn.py`<br>`notebooks/03_LSTM_Sequential_Ishanvi.ipynb` |
| **Member 4** | Rishi Khandelwal | **Transformer Encoder (Self-Attention)** | `src/models/transformer.py`<br>`src/train.py`<br>`src/evaluate.py`<br>`app.py`<br>`notebooks/04_Transformer_Encoder_Rishi.ipynb` |

---

## 📌 Step 1: Rishi Pushes Base Template to GitHub

Rishi creates the repository `predictive-maintenance-cmapss` on GitHub, adds teammates as Collaborators, and pushes the base template:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
git push -u origin main
```

---

## 📌 Step 2: Member 1 - Mayurika Sathish (MLP Baseline PR)

1. Clone the repository on your laptop:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   git checkout -b feature/mlp-baseline
   ```
2. Copy `mlp.py` into `src/models/` and `01_MLP_Baseline_Mayurika.ipynb` into `notebooks/`.
3. Commit and push:
   ```bash
   git add src/models/mlp.py notebooks/01_MLP_Baseline_Mayurika.ipynb
   git commit -m "feat: Add MLP baseline architecture and notebook (Mayurika Sathish)"
   git push origin feature/mlp-baseline
   ```
4. **On GitHub.com**: Click **"Compare & pull request"** $\rightarrow$ **"Create pull request"** $\rightarrow$ **"Merge pull request"**.

---

## 📌 Step 3: Member 2 - Sachith V P (1D-CNN PR)

1. Clone/pull the repository on your laptop:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   git checkout -b feature/1d-cnn-model
   ```
2. Copy `cnn1d.py` into `src/models/` and `02_1D_CNN_Sachith.ipynb` into `notebooks/`.
3. Commit and push:
   ```bash
   git add src/models/cnn1d.py notebooks/02_1D_CNN_Sachith.ipynb
   git commit -m "feat: Add 1D-CNN model architecture and notebook (Sachith V P)"
   git push origin feature/1d-cnn-model
   ```
4. **On GitHub.com**: Click **"Compare & pull request"** $\rightarrow$ **"Create pull request"** $\rightarrow$ **"Merge pull request"**.

---

## 📌 Step 4: Member 3 - Ishanvi Kaushik (Stacked LSTM PR)

1. Clone/pull the repository on your laptop:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   git checkout -b feature/lstm-model
   ```
2. Copy `rnn.py` into `src/models/` and `03_LSTM_Sequential_Ishanvi.ipynb` into `notebooks/`.
3. Commit and push:
   ```bash
   git add src/models/rnn.py notebooks/03_LSTM_Sequential_Ishanvi.ipynb
   git commit -m "feat: Add Stacked LSTM sequence architecture and notebook (Ishanvi Kaushik)"
   git push origin feature/lstm-model
   ```
4. **On GitHub.com**: Click **"Compare & pull request"** $\rightarrow$ **"Create pull request"** $\rightarrow$ **"Merge pull request"**.

---

## 📌 Step 5: Member 4 - Rishi Khandelwal (Transformer Encoder PR)

1. Create and switch to your branch:
   ```bash
   git checkout -b feature/transformer-model
   ```
2. Copy `transformer.py` into `src/models/`, `train.py` & `evaluate.py` into `src/`, `app.py` into root, and `04_Transformer_Encoder_Rishi.ipynb` into `notebooks/`.
3. Commit and push:
   ```bash
   git add src/models/transformer.py src/train.py src/evaluate.py app.py notebooks/04_Transformer_Encoder_Rishi.ipynb
   git commit -m "feat: Add Transformer Encoder, evaluation pipeline, Streamlit app, and notebook (Rishi Khandelwal)"
   git push origin feature/transformer-model
   ```
4. **On GitHub.com**: Click **"Compare & pull request"** $\rightarrow$ **"Create pull request"** $\rightarrow$ **"Merge pull request"**.

---

## 🎤 Viva Defense Cheat Sheet

* **Mayurika Sathish (MLP)**: Explains 3-layer Dense network, flattening 30x15 windows, achieving lowest MAE (8.92 cycles).
* **Sachith V P (1D-CNN)**: Explains 1D spatial-temporal convolutions (`Conv1D`) extracting local features with 41,153 parameters.
* **Ishanvi Kaushik (LSTM)**: Explains 2-layer Stacked LSTM gating units capturing sequential degradation trajectory with highest Critical Alert F1 (0.8667).
* **Rishi Khandelwal (Transformer Encoder)**: Explains Multi-Head Self-Attention (MHSA) and Positional Encoding capturing long-term dependencies.
