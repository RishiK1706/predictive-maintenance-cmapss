import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data_loader import (
    load_raw_data,
    add_piecewise_rul,
    get_informative_features,
    split_train_val_by_engine,
    scale_features,
    build_test_dataset,
)
from src.evaluate import RUL_to_early_warning_state
from src.train import get_model

st.set_page_config(
    page_title="C-MAPSS Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Maintenance & Early Warning Dashboard")
st.markdown("### Remaining Useful Life (RUL) Estimation & Health Monitoring on NASA C-MAPSS")

# Sidebar settings
st.sidebar.header("Configuration & Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Trained Deep Learning Model:",
    ["MLP Baseline", "1D-CNN", "LSTM Sequential", "Transformer Encoder"],
)

model_key_map = {
    "MLP Baseline": "mlp",
    "1D-CNN": "cnn1d",
    "LSTM Sequential": "lstm",
    "Transformer Encoder": "transformer",
}

m_key = model_key_map[model_choice]
checkpoint_path = f"results/{m_key}_FD001_best.pt"

if not os.path.exists(checkpoint_path):
    st.error(f"Checkpoint for {model_choice} not found. Please train models first using `python run_all_experiments.py`.")
    st.stop()

# Load Data
@st.cache_data
def load_and_preprocess_test_data():
    train_df, test_df, rul_df = load_raw_data(dataset_id="FD001")
    train_df = add_piecewise_rul(train_df, max_rul=125)
    feature_cols = get_informative_features(train_df, drop_constant=True)
    train_split, val_split = split_train_val_by_engine(train_df, val_ratio=0.2, seed=42)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(train_split, val_split, test_df, feature_cols)
    X_test, y_test = build_test_dataset(test_scaled, rul_df, window_size=30, feature_cols=feature_cols, max_rul=125)
    return test_df, feature_cols, X_test, y_test

test_df, feature_cols, X_test, y_test = load_and_preprocess_test_data()

# Select Test Engine
test_engine_id = st.sidebar.slider("Select Test Engine ID (1 to 100):", 1, 100, 24)

# Load Model
device = torch.device("cpu")
model = get_model(m_key, window_size=30, n_features=len(feature_cols))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Inference for selected engine
engine_sample_X = torch.tensor(X_test[test_engine_id - 1: test_engine_id], dtype=torch.float32)
with torch.no_grad():
    pred_rul = model(engine_sample_X).item()

true_rul = y_test[test_engine_id - 1]

# Operational State Classification
state_code = RUL_to_early_warning_state(np.array([pred_rul]))[0]
state_names = {0: "Normal State (Green)", 1: "Warning State (Yellow)", 2: "Critical State (Red)"}
state_colors = {0: "green", 1: "orange", 2: "red"}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Engine ID", f"Engine #{test_engine_id}")
col2.metric("Predicted RUL", f"{pred_rul:.1f} Cycles")
col3.metric("True RUL", f"{true_rul:.1f} Cycles")
col4.metric("Error |Pred - True|", f"{abs(pred_rul - true_rul):.1f} Cycles")

st.markdown("---")

st.markdown(f"#### Operational Health Alert Status: **:{state_colors[state_code]}[{state_names[state_code]}]**")

# Plot Sensor Trajectories for selected engine
st.markdown(f"### Sensor Degradation Trajectory for Test Engine #{test_engine_id}")
engine_raw = test_df[test_df["unit_nr"] == test_engine_id]

fig, ax = plt.subplots(figsize=(10, 4))
for s in ["s_2", "s_3", "s_4", "s_7", "s_11"]:
    ax.plot(engine_raw["time_cycles"], engine_raw[s], label=f"Sensor {s}")
ax.set_xlabel("Operational Cycles Observed")
ax.set_ylabel("Sensor Measurements")
ax.set_title(f"Multi-Sensor Trajectory (Engine #{test_engine_id})")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)
