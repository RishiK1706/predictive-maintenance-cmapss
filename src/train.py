import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import (
    load_raw_data,
    add_piecewise_rul,
    get_informative_features,
    split_train_val_by_engine,
    scale_features,
    create_sliding_windows,
    build_test_dataset,
)
from src.dataset import get_dataloaders
from src.models.mlp import MLPBaseline
from src.models.cnn1d import CNN1DModel
from src.models.rnn import RNNModel
from src.models.transformer import TransformerModel


def compute_nasa_score(y_true, y_pred):
    """
    Calculate the official NASA C-MAPSS scoring metric:
    d_i = pred - true
    s_i = exp(-d_i / 13) - 1 if d_i < 0 else exp(d_i / 10) - 1
    S = sum(s_i)
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics: MAE, RMSE, R2, and NASA Score.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    nasa_score = compute_nasa_score(y_true, y_pred)
    return float(mae), float(rmse), float(r2), float(nasa_score)


def get_model(model_type, window_size, n_features):
    model_type = model_type.lower()
    if model_type == "mlp":
        return MLPBaseline(window_size=window_size, n_features=n_features)
    elif model_type == "cnn1d":
        return CNN1DModel(window_size=window_size, n_features=n_features)
    elif model_type in ["lstm", "gru"]:
        return RNNModel(window_size=window_size, n_features=n_features, rnn_type=model_type.upper())
    elif model_type == "transformer":
        return TransformerModel(window_size=window_size, n_features=n_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model_type="mlp",
    dataset_id="FD001",
    window_size=30,
    max_rul=125,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    patience=10,
    save_dir="results",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training {model_type.upper()} on {dataset_id} (Device: {device}) ---")

    # 1. Load Data
    train_df, test_df, rul_df = load_raw_data(dataset_id=dataset_id)

    # 2. RUL Label Construction & Feature Selection
    train_df = add_piecewise_rul(train_df, max_rul=max_rul)
    feature_cols = get_informative_features(train_df, drop_constant=True)
    n_features = len(feature_cols)
    print(f"Selected {n_features} informative sensors: {feature_cols}")

    # 3. Engine-wise Split & Feature Scaling
    train_split, val_split = split_train_val_by_engine(train_df, val_ratio=0.2, seed=42)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_split, val_split, test_df, feature_cols
    )

    # 4. Generate Sliding Windows
    X_train, y_train = create_sliding_windows(train_scaled, window_size=window_size, feature_cols=feature_cols)
    X_val, y_val = create_sliding_windows(val_scaled, window_size=window_size, feature_cols=feature_cols)
    X_test, y_test = build_test_dataset(test_scaled, rul_df, window_size=window_size, feature_cols=feature_cols, max_rul=max_rul)

    print(f"Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 5. DataLoaders & Model Ingestion
    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )

    model = get_model(model_type, window_size, n_features).to(device)
    n_params = count_parameters(model)
    print(f"Model {model_type.upper()} Total Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(save_dir, f"{model_type}_{dataset_id}_best.pt")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_mse = np.mean(train_losses)

        # Validation Loop
        model.eval()
        val_losses = []
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                val_losses.append(loss.item())
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(y_b.cpu().numpy())

        val_mse = np.mean(val_losses)
        scheduler.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            val_mae, val_rmse, val_r2, val_nasa = calculate_metrics(np.array(val_targets_list), np.array(val_preds_list))
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_mse:.2f} | Val MSE: {val_mse:.2f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | NASA Score: {val_nasa:.1f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # 6. Test Set Evaluation
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    test_preds_list, test_targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            preds = model(X_b)
            test_preds_list.extend(preds.cpu().numpy())
            test_targets_list.extend(y_b.cpu().numpy())

    test_mae, test_rmse, test_r2, test_nasa = calculate_metrics(np.array(test_targets_list), np.array(test_preds_list))
    print(f"\n>>> FINAL TEST RESULTS [{model_type.upper()}]: MAE = {test_mae:.2f} | RMSE = {test_rmse:.2f} | R2 = {test_r2:.4f} | NASA Score = {test_nasa:.1f} | Time = {total_time:.2f}s <<<\n")

    results = {
        "model_type": model_type,
        "dataset_id": dataset_id,
        "window_size": window_size,
        "max_rul": max_rul,
        "parameters": n_params,
        "train_time_sec": round(total_time, 2),
        "test_mae": round(test_mae, 2),
        "test_rmse": round(test_rmse, 2),
        "test_r2": round(test_r2, 4),
        "nasa_score": round(test_nasa, 1),
    }

    metrics_file = os.path.join(save_dir, "metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}

    all_metrics[f"{model_type}_{dataset_id}"] = results
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C-MAPSS RUL Model")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn1d", "lstm", "gru", "transformer"])
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--max_rul", type=int, default=125)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train_model(
        model_type=args.model,
        dataset_id=args.dataset,
        window_size=args.window_size,
        max_rul=args.max_rul,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
