# GitHub Team Collaboration Guide (ICT-4442 Deep Learning Project)

This guide provides step-by-step instructions for each of the 4 team members to commit and push their assigned model architecture to the shared GitHub repository from their own GitHub accounts.

---

## Initial Setup (Done by Rishi Khandelwal)

1. Create a public/private repository on GitHub named `predictive-maintenance-cmapss`.
2. Add teammates as **Collaborators** (Repo $\rightarrow$ Settings $\rightarrow$ Collaborators $\rightarrow$ Add people):
   - Mayurika Sathish
   - Sachith V P
   - Ishanvi Kaushik
3. Push the initial repository skeleton to GitHub:
   ```bash
   git add .gitignore requirements.txt run_all_experiments.py report/ src/
   git commit -m "Initial commit: Project structure and Transformer Encoder implementation (Rishi Khandelwal)"
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   git branch -M main
   git push -u origin main
   ```

---

## Instructions for Member 1: Mayurika Sathish (MLP Baseline)

1. Clone the repository to your laptop:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-maintenance-cmapss.git
   cd predictive-maintenance-cmapss
   ```
2. Verify or add your MLP architecture file in `src/models/mlp.py`.
3. Commit and push from your GitHub account:
   ```bash
   git add src/models/mlp.py
   git commit -m "feat: Add MLP baseline architecture and feature engineering (Mayurika Sathish)"
   git push origin main
   ```

---

## Instructions for Member 2: Sachith V P (1D-CNN)

1. Pull the latest repository updates:
   ```bash
   git pull origin main
   ```
2. Verify or add your 1D-CNN architecture file in `src/models/cnn1d.py` and data loader in `src/data_loader.py`.
3. Commit and push from your GitHub account:
   ```bash
   git add src/models/cnn1d.py src/data_loader.py notebooks/eda_and_plots.py
   git commit -m "feat: Add 1D-CNN model architecture, sliding window data loader, and EDA plots (Sachith V P)"
   git push origin main
   ```

---

## Instructions for Member 3: Ishanvi Kaushik (LSTM)

1. Pull the latest repository updates:
   ```bash
   git pull origin main
   ```
2. Verify or add your Stacked LSTM architecture file in `src/models/rnn.py` and dataset wrapper in `src/dataset.py`.
3. Commit and push from your GitHub account:
   ```bash
   git add src/models/rnn.py src/dataset.py
   git commit -m "feat: Add Stacked LSTM sequence architecture and PyTorch dataset (Ishanvi Kaushik)"
   git push origin main
   ```

---

## Instructions for Member 4: Rishi Khandelwal (Transformer Encoder)

1. Pull the latest repository updates:
   ```bash
   git pull origin main
   ```
2. Verify or update your Transformer Encoder architecture file in `src/models/transformer.py`, evaluation module in `src/evaluate.py`, and Streamlit app in `app.py`.
3. Commit and push from your GitHub account:
   ```bash
   git add src/models/transformer.py src/evaluate.py app.py report/Phase2_Interim_ICT4442.md
   git commit -m "feat: Add Transformer Encoder, evaluation pipeline, Streamlit demo app, and Phase 2 Interim Report (Rishi Khandelwal)"
   git push origin main
   ```
