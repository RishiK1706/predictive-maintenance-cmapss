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
    page_title="Predictive Maintenance and Early Warning Dashboard",
    layout="wide",
)

st.title("Predictive Maintenance and Early Warning Dashboard")
st.markdown("### Real-Time Remaining Useful Life (RUL) Inference & Fleet Health Monitoring")

# ─── Sidebar: Fleet Controls & Alert Protocol Legend ─────────────────────────
st.sidebar.header("Active Fleet Controller")
test_engine_id = st.sidebar.slider("Select Unseen Test Engine (1 to 100):", 1, 100, 24)

st.sidebar.markdown("---")
st.sidebar.subheader("Maintenance Protocol Legend")
st.sidebar.markdown(
    """
    * 🟢 **Normal (Safe)**: $\\text{RUL} > 50$ cycles  
      *Engine certified airworthy. Routine flight operations.*
    * 🟡 **Warning**: $20 < \\text{RUL} \\le 50$ cycles  
      *Degradation detected. Schedule depot inspection at next turnaround.*
    * 🔴 **Critical**: $\\text{RUL} \\le 20$ cycles  
      *Immediate failure hazard! Ground aircraft and replace HPC module.*
    """
)

# ─── Cached Data Ingestion ───────────────────────────────────────────────────
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

# ─── Cached Model Loading ────────────────────────────────────────────────────
@st.cache_resource
def load_all_models(n_features):
    models = {}
    model_keys = {
        "MLP Baseline": "mlp",
        "1D-CNN": "cnn1d",
        "Stacked LSTM": "lstm",
        "Transformer": "transformer",
    }
    device = torch.device("cpu")
    for display_name, key in model_keys.items():
        ckpt_path = f"results/{key}_FD001_best.pt"
        if os.path.exists(ckpt_path):
            m = get_model(key, window_size=30, n_features=n_features).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
            m.eval()
            models[display_name] = (m, key)
    return models

trained_models = load_all_models(len(feature_cols))

if not trained_models:
    st.error("No model checkpoints found in results/. Please train models first using run_all_experiments.py.")
    st.stop()

# ─── Live Dynamic Inference for Selected Engine ──────────────────────────────
engine_sample_X = torch.tensor(X_test[test_engine_id - 1: test_engine_id], dtype=torch.float32)
true_rul = float(y_test[test_engine_id - 1])

# Extract current operational age from raw test flight history
engine_raw = test_df[test_df["unit_nr"] == test_engine_id]
observed_cycles = int(engine_raw["time_cycles"].max())

# Run forward pass live through all 4 models
predictions = {}
with torch.no_grad():
    for name, (m, key) in trained_models.items():
        pred = m(engine_sample_X).item()
        error = abs(pred - true_rul)
        state_code = RUL_to_early_warning_state(np.array([pred]))[0]
        predictions[name] = {"pred": pred, "error": error, "state_code": state_code}

# ─── Section 1: Engine Diagnostics Summary ───────────────────────────────────
st.subheader(f"Fleet Diagnostics Overview — Engine #{test_engine_id}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Engine", f"Engine #{test_engine_id}")
col2.metric("Flights Completed", f"{observed_cycles} Cycles")
col3.metric("True Ground-Truth RUL", f"{true_rul:.1f} Cycles")

# Highlight LSTM's estimated total lifespan (Observed Age + Predicted RUL)
lstm_pred = predictions.get("Stacked LSTM", {}).get("pred", true_rul)
est_lifespan = observed_cycles + lstm_pred
col4.metric("Estimated Total Lifespan", f"{est_lifespan:.1f} Cycles")

# Actionable Maintenance Banner
state_meta = {
    0: {"name": "NORMAL OPERATIONAL STATE", "box": "success", "action": "Engine is airworthy. Certified for regular scheduled flight rotations."},
    1: {"name": "MAINTENANCE WARNING ALERT", "box": "warning", "action": "Subtle degradation detected. Schedule borescope inspection at next turnaround."},
    2: {"name": "CRITICAL FAILURE HAZARD", "box": "error", "action": "High-Pressure Compressor stall hazard! Ground aircraft immediately for engine overhaul."},
}

lstm_state = predictions.get("Stacked LSTM", {}).get("state_code", 0)
selected_meta = state_meta[lstm_state]

if selected_meta["box"] == "success":
    st.success(f"🟢 **Status: {selected_meta['name']}** — {selected_meta['action']}")
elif selected_meta["box"] == "warning":
    st.warning(f"🟡 **Status: {selected_meta['name']}** — {selected_meta['action']}")
else:
    st.error(f"🔴 **Status: {selected_meta['name']}** — {selected_meta['action']}")

st.markdown("---")

# ─── Section 2: Multi-Model Consensus (Side-by-Side) ─────────────────────────
st.subheader("Multi-Model Consensus & Predictive Benchmark")
st.markdown("*Real-time forward pass executed live across all four neural network architectures on this engine:*")

cols = st.columns(len(predictions))
state_badges = {0: "🟢 Safe", 1: "🟡 Warning", 2: "🔴 Critical"}

for col, (name, data) in zip(cols, predictions.items()):
    with col:
        st.markdown(f"#### {name}")
        st.metric(
            label="Predicted RUL",
            value=f"{data['pred']:.1f} Cyc",
            delta=f"{data['pred'] - true_rul:+.1f} Error",
            delta_color="inverse",
        )
        st.caption(f"Alert State: **{state_badges[data['state_code']]}**")

st.markdown("---")

# ─── Section 3: Diagnostic Visualizations ────────────────────────────────────
st.subheader("Diagnostic Telemetry & RUL Prediction Comparison")

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    # Model Consensus Bar Chart vs Thresholds
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    model_names = list(predictions.keys())
    pred_values = [predictions[m]["pred"] for m in model_names]
    
    # Bar colors: highlight LSTM in distinct color, others in slate
    bar_colors = ["#4A90E2" if m == "Stacked LSTM" else "#708090" for m in model_names]
    y_pos = np.arange(len(model_names))
    
    ax1.barh(y_pos, pred_values, color=bar_colors, height=0.55, label="Model Predictions")
    ax1.axvline(x=true_rul, color="black", linestyle="-", linewidth=2.0, label=f"True RUL ({true_rul:.1f})")
    ax1.axvline(x=50, color="goldenrod", linestyle="--", linewidth=1.5, label="Warning (50 cyc)")
    ax1.axvline(x=20, color="crimson", linestyle=":", linewidth=1.5, label="Critical (20 cyc)")
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("RUL (Cycles)", fontsize=10)
    ax1.set_title(f"Model Predictions vs Failure Thresholds (Engine #{test_engine_id})", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="x")
    st.pyplot(fig1)

with plot_col2:
    # Multi-Sensor Degradation Trajectory Plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sensor_labels = {
        "s_2": "LPC Outlet Temp (s_2)",
        "s_3": "HPC Outlet Temp (s_3)",
        "s_4": "LPT Outlet Temp (s_4)",
        "s_7": "HPC Static Press (s_7)",
        "s_11": "HPC Outlet Press (s_11)",
    }

    for s_key, s_label in sensor_labels.items():
        ax2.plot(engine_raw["time_cycles"], engine_raw[s_key], label=s_label, linewidth=1.5)

    ax2.set_xlabel("Flight Cycles Observed", fontsize=10)
    ax2.set_ylabel("Sensor Measurement", fontsize=10)
    ax2.set_title(f"Sensor Trajectories Across {observed_cycles} Cycles", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig2)

# ─── Explainable AI: Raw Input Matrix Heatmap ────────────────────────────────
with st.expander(f"Inspect Input Tensor Matrix for Engine #{test_engine_id} (Explainable AI)"):
    st.markdown(f"Visual representation of the **3D sequence tensor (30 cycles × 15 sensors = 450 values)** ingested by the models for Engine #{test_engine_id}:")
    st.markdown("The vertical axis shows the 30 consecutive flight cycles leading to the current flight ($t-29$ to $t$). Hotter colors indicate sensors saturated near degradation thresholds.")

    fig_xai, ax_xai = plt.subplots(figsize=(10, 3.8))
    window_matrix = engine_sample_X[0].numpy()
    im = ax_xai.imshow(window_matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax_xai.set_xticks(np.arange(len(feature_cols)))
    ax_xai.set_xticklabels(feature_cols, rotation=45, fontsize=8)
    ax_xai.set_yticks(np.arange(0, 30, 5))
    ax_xai.set_yticklabels([f"t-{29-i}" if (29-i) > 0 else "t (Current)" for i in range(0, 30, 5)], fontsize=8)
    ax_xai.set_xlabel("Informative Physical Sensor Channels", fontsize=9)
    ax_xai.set_ylabel("Flight Cycles Ingested", fontsize=9)
    ax_xai.set_title(f"Normalized Input Matrix Ingested by Deep Learning Models (Engine #{test_engine_id})", fontsize=10, fontweight="bold")
    cbar = fig_xai.colorbar(im, ax=ax_xai, pad=0.02)
    cbar.set_label("Normalized Reading [-1, 1]", fontsize=8)
    st.pyplot(fig_xai)

# ─── Section 4: Expandable Benchmark Performance Table ───────────────────────
with st.expander("View Overall Fleet Benchmark Comparison (100 Test Engines)"):
    st.markdown("Empirical benchmark results across the entire NASA C-MAPSS FD001 test fleet:")
    csv_path = "results/model_comparison_table.csv"
    if os.path.exists(csv_path):
        bench_df = pd.read_csv(csv_path)
        bench_df = bench_df.drop(columns=["Train Time (s)"], errors="ignore")
        st.dataframe(bench_df, use_container_width=True)
    else:
        st.info("Benchmark summary table will appear here once run_all_experiments.py is executed.")

    st.markdown("#### Fleet-Wide Prediction Error Distribution (All 100 Test Engines)")
    st.markdown("*Distribution of absolute prediction errors ($|\\hat{y} - y|$) across all 100 unseen aircraft engines:*")

    # Compute errors across all 100 test engines
    fleet_errors = []
    t_X = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        for name, (m, _) in trained_models.items():
            preds_all = m(t_X).numpy()
            errs = np.abs(preds_all - y_test)
            fleet_errors.append(errs)

    fig3, ax3 = plt.subplots(figsize=(10, 3.8))
    box = ax3.boxplot(
        fleet_errors,
        patch_artist=True,
        tick_labels=list(trained_models.keys()),
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors = ["#708090", "#A9A9A9", "#4A90E2", "#B0C4DE"]
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax3.set_ylabel("Absolute Error (Cycles)", fontsize=10)
    ax3.set_title("Prediction Error Spread Across 100 Unseen Test Engines", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.3, axis="y")
    st.pyplot(fig3)
