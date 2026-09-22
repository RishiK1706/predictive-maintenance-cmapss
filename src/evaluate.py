import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data_loader import (
    load_raw_data,
    add_piecewise_rul,
    get_informative_features,
    split_train_val_by_engine,
    scale_features,
    build_test_dataset,
)
from src.dataset import CMAPSSDataset, DataLoader
from src.train import calculate_metrics, compute_nasa_score, get_model


def RUL_to_early_warning_state(rul):
    """
    Convert RUL value into early warning operational alert state:
    0: Normal   (RUL > 50)
    1: Warning  (20 < RUL <= 50)
    2: Critical (RUL <= 20)
    """
    states = np.zeros_like(rul, dtype=int)
    states[(rul > 20) & (rul <= 50)] = 1
    states[rul <= 20] = 2
    return states


def evaluate_early_warning(y_true, y_pred):
    """
    Evaluate early warning alert states performance.
    """
    true_states = RUL_to_early_warning_state(y_true)
    pred_states = RUL_to_early_warning_state(y_pred)

    state_names = ["Normal", "Warning", "Critical"]
    cm = confusion_matrix(true_states, pred_states, labels=[0, 1, 2])
    f1_macro = f1_score(true_states, pred_states, average="macro")
    f1_critical = f1_score(true_states, pred_states, labels=[2], average="micro")

    print("\n--- Early Warning Alert State Evaluation ---")
    print(classification_report(true_states, pred_states, target_names=state_names, zero_division=0))
    print("Confusion Matrix:")
    print(cm)
    print(f"Macro F1-Score: {f1_macro:.4f} | Critical State F1-Score: {f1_critical:.4f}\n")

    return {
        "confusion_matrix": cm.tolist(),
        "f1_macro": float(f1_macro),
        "f1_critical": float(f1_critical),
    }


def run_full_evaluation(model_type="mlp", dataset_id="FD001", window_size=30, max_rul=125, save_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(save_dir, f"{model_type}_{dataset_id}_best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load data
    train_df, test_df, rul_df = load_raw_data(dataset_id=dataset_id)
    train_df = add_piecewise_rul(train_df, max_rul=max_rul)
    feature_cols = get_informative_features(train_df, drop_constant=True)
    n_features = len(feature_cols)

    train_split, val_split = split_train_val_by_engine(train_df, val_ratio=0.2, seed=42)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(train_split, val_split, test_df, feature_cols)
    X_test, y_test = build_test_dataset(test_scaled, rul_df, window_size=window_size, feature_cols=feature_cols, max_rul=max_rul)

    # Load model
    model = get_model(model_type, window_size, n_features).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_dataset = CMAPSSDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    y_pred_list = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            X_b = X_b.to(device)
            preds = model(X_b)
            y_pred_list.extend(preds.cpu().numpy())

    y_pred = np.array(y_pred_list)

    mae, rmse, r2, nasa_score = calculate_metrics(y_test, y_pred)
    ew_metrics = evaluate_early_warning(y_test, y_pred)

    print(f"=== [{model_type.upper()}] Final Test Metrics ===")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f} | NASA Score: {nasa_score:.1f}")

    return {
        "model_type": model_type,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "nasa_score": round(nasa_score, 1),
        "early_warning": ew_metrics,
    }


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    run_full_evaluation(model_type=model_type)
