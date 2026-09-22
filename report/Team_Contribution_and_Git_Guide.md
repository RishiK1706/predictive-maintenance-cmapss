# 🚀 ICT-4442 Deep Learning Mini Project: Team Guide & Git PR Workflow

Welcome to our project repository! This guide explains our project structure, assigned roles, step-by-step instructions for submitting your code via GitHub Pull Requests (PRs), and viva defense preparation.

---

## 👥 Team Member Roles & Assigned Architecture Families

Each team member owns a distinct model architecture family as required by course guidelines:

| Member | Name | Assigned Model Family | Key Files Owned |
| :--- | :--- | :--- | :--- |
| **Member 1** | Mayurika Sathish | **MLP Baseline (Fully-Connected)** | `src/models/mlp.py`<br>`notebooks/01_MLP_Baseline_Mayurika.ipynb` |
| **Member 2** | Sachith V P | **1D-CNN (Convolutional Network)** | `src/models/cnn1d.py`<br>`src/data_loader.py`<br>`notebooks/02_1D_CNN_Sachith.ipynb` |
| **Member 3** | Ishanvi Kaushik | **Stacked LSTM (Recurrent/Sequential)** | `src/models/rnn.py`<br>`src/dataset.py`<br>`notebooks/03_LSTM_Sequential_Ishanvi.ipynb` |
| **Member 4** | Rishi Khandelwal | **Transformer Encoder (Self-Attention)** | `src/models/transformer.py`<br>`src/evaluate.py`<br>`app.py`<br>`notebooks/04_Transformer_Encoder_Rishi.ipynb` |

---

## 🛠️ Step-by-Step GitHub Branch & Pull Request (PR) Guide

To demonstrate clear individual contributions to faculty evaluators, we use the **Feature Branch + Pull Request (PR)** workflow.

### 📌 Instructions for Member 1: Mayurika Sathish (MLP Baseline)

1. Open your terminal and clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   ```
2. Create and switch to your feature branch:
   ```bash
   git checkout -b feature/mlp-baseline
   ```
3. Add your assigned code files and commit:
   ```bash
   git add src/models/mlp.py notebooks/01_MLP_Baseline_Mayurika.ipynb
   git commit -m "feat: Implement MLP baseline architecture and feature engineering (Mayurika Sathish)"
   ```
4. Push your branch to GitHub:
   ```bash
   git push origin feature/mlp-baseline
   ```
5. **Open Pull Request on GitHub.com**:
   * Open the repository link in your browser.
   * Click the green **"Compare & pull request"** button.
   * Title: `feat: Add MLP Baseline Model - Mayurika Sathish`
   * Click **"Create pull request"**, then click **"Merge pull request"**.

---

### 📌 Instructions for Member 2: Sachith V P (1D-CNN)

1. Open your terminal and clone/pull the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   ```
2. Create and switch to your feature branch:
   ```bash
   git checkout -b feature/1d-cnn-model
   ```
3. Add your assigned code files and commit:
   ```bash
   git add src/models/cnn1d.py src/data_loader.py notebooks/02_1D_CNN_Sachith.ipynb
   git commit -m "feat: Implement 1D-CNN architecture, data loader, and EDA plots (Sachith V P)"
   ```
4. Push your branch to GitHub:
   ```bash
   git push origin feature/1d-cnn-model
   ```
5. **Open Pull Request on GitHub.com**:
   * Open the repository link in your browser.
   * Click the green **"Compare & pull request"** button.
   * Title: `feat: Add 1D-CNN Model & Data Loader - Sachith V P`
   * Click **"Create pull request"**, then click **"Merge pull request"**.

---

### 📌 Instructions for Member 3: Ishanvi Kaushik (Stacked LSTM)

1. Open your terminal and clone/pull the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   ```
2. Create and switch to your feature branch:
   ```bash
   git checkout -b feature/lstm-model
   ```
3. Add your assigned code files and commit:
   ```bash
   git add src/models/rnn.py src/dataset.py notebooks/03_LSTM_Sequential_Ishanvi.ipynb
   git commit -m "feat: Implement Stacked LSTM sequence architecture and PyTorch dataset (Ishanvi Kaushik)"
   ```
4. Push your branch to GitHub:
   ```bash
   git push origin feature/lstm-model
   ```
5. **Open Pull Request on GitHub.com**:
   * Open the repository link in your browser.
   * Click the green **"Compare & pull request"** button.
   * Title: `feat: Add Stacked LSTM Model & Dataset Loader - Ishanvi Kaushik`
   * Click **"Create pull request"**, then click **"Merge pull request"**.

---

### 📌 Instructions for Member 4: Rishi Khandelwal (Transformer Encoder)

1. Create and switch to your feature branch:
   ```bash
   git checkout -b feature/transformer-model
   ```
2. Add your assigned code files and commit:
   ```bash
   git add src/models/transformer.py src/evaluate.py app.py notebooks/04_Transformer_Encoder_Rishi.ipynb
   git commit -m "feat: Implement Transformer Encoder architecture and evaluation pipeline (Rishi Khandelwal)"
   ```
3. Push your branch to GitHub:
   ```bash
   git push origin feature/transformer-model
   ```
4. **Open Pull Request on GitHub.com**:
   * Click **"Compare & pull request"** $\rightarrow$ **"Create pull request"** $\rightarrow$ **"Merge pull request"**.

---

## 🎤 Viva Presentation & Defense Cheat Sheet

During the in-class presentation, faculty question each member individually on their model. Here is what to highlight:

* **Mayurika Sathish (MLP)**:
  - *Key Point*: Explains flattening 30-cycle $\times$ 15-sensor sliding window into a 1D vector fed into 3 dense layers. Achieved the lowest MAE (8.92 cycles) and fast inference (50.2s).
* **Sachith V P (1D-CNN)**:
  - *Key Point*: Explains 1D temporal convolutions (`Conv1D`) extracting localized temporal degradation features across sensors. Most parameter-efficient model (only 41,153 params).
* **Ishanvi Kaushik (LSTM)**:
  - *Key Point*: Explains stacked 2-layer LSTM gating mechanism (forget, input, output gates) capturing sequential degradation trajectory. Achieved highest Critical Failure Alert F1-Score (0.8667).
* **Rishi Khandelwal (Transformer Encoder)**:
  - *Key Point*: Explains Multi-Head Self-Attention (MHSA) and Positional Encoding capturing global time-step dependencies without temporal recurrence.

---

## 💻 Running Code & Web Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete benchmark
python run_all_experiments.py

# Launch Streamlit Web Dashboard
streamlit run app.py
```
