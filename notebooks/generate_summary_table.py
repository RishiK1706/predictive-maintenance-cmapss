import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import run_full_evaluation


def generate_results_table(metrics_path="results/metrics.json"):
    """
    Read metrics.json and print/save a formatted comparison table for the report.
    """
    if not os.path.exists(metrics_path):
        print("No metrics.json found yet.")
        return

    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)

    rows = []
    for key, data in metrics_dict.items():
        model_name = data["model_type"].upper()
        if model_name == "CNN1D":
            model_name = "1D-CNN"

        rows.append(
            {
                "Model Architecture": model_name,
                "MAE": data["test_mae"],
                "RMSE": data["test_rmse"],
                "R2": data["test_r2"],
                "Parameters": f"{data['parameters']:,}",
                "Train Time (s)": data["train_time_sec"],
            }
        )

    df_results = pd.DataFrame(rows)
    print("\n=================== MODEL COMPARISON TABLE (C-MAPSS FD001) ===================")
    print(df_results.to_string(index=False))
    print("=================================================================================\n")

    # Save to CSV for report inclusion
    csv_path = "results/model_comparison_table.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")
    return df_results


def plot_actual_vs_predicted_all(save_path="figures/actual_vs_predicted_comparison.png"):
    """
    Plot actual vs predicted RUL for test engines across trained models.
    """
    from src.data_loader import load_raw_data, add_piecewise_rul, get_informative_features, split_train_val_by_engine, scale_features, build_test_dataset
    from src.dataset import CMAPSSDataset, DataLoader
    from src.train import get_model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "results"

    trained_models = []
    for m in ["mlp", "cnn1d", "lstm", "transformer"]:
        if os.path.exists(os.path.join(save_dir, f"{m}_FD001_best.pt")):
            trained_models.append(m)

    if not trained_models:
        print("No trained model checkpoints found to plot.")
        return

    train_df, test_df, rul_df = load_raw_data(dataset_id="FD001")
    train_df = add_piecewise_rul(train_df, max_rul=125)
    feature_cols = get_informative_features(train_df, drop_constant=True)
    n_features = len(feature_cols)

    train_split, val_split = split_train_val_by_engine(train_df, val_ratio=0.2, seed=42)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(train_split, val_split, test_df, feature_cols)
    X_test, y_test = build_test_dataset(test_scaled, rul_df, window_size=30, feature_cols=feature_cols, max_rul=125)

    test_dataset = CMAPSSDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    plt.figure(figsize=(12, 6))
    sort_idx = np.argsort(y_test)
    y_test_sorted = y_test[sort_idx]
    plt.plot(y_test_sorted, label="True RUL", color="black", linewidth=2.5, linestyle="--")

    colors = {"mlp": "#1f77b4", "cnn1d": "#ff7f0e", "lstm": "#2ca02c", "transformer": "#d62728"}
    names = {"mlp": "MLP Baseline", "cnn1d": "1D-CNN", "lstm": "LSTM Sequential", "transformer": "Transformer Encoder"}

    for m in trained_models:
        checkpoint_path = os.path.join(save_dir, f"{m}_FD001_best.pt")
        model = get_model(m, 30, n_features).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for X_b, _ in test_loader:
                X_b = X_b.to(device)
                preds.extend(model(X_b).cpu().numpy())

        preds_sorted = np.array(preds)[sort_idx]
        plt.plot(preds_sorted, label=names.get(m, m.upper()), color=colors.get(m, "blue"), alpha=0.8, linewidth=1.8)

    plt.title("Actual vs Predicted Remaining Useful Life (RUL) on NASA C-MAPSS FD001 Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Test Engine Index (Sorted by True RUL)")
    plt.ylabel("Remaining Useful Life (Cycles)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved actual vs predicted RUL plot to {save_path}")


if __name__ == "__main__":
    generate_results_table()
    plot_actual_vs_predicted_all()
