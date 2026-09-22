import os
from src.train import train_model
from src.evaluate import run_full_evaluation
from notebooks.generate_summary_table import generate_results_table, plot_actual_vs_predicted_all

def main():
    models = ["mlp", "cnn1d", "lstm", "transformer"]
    print(f"Executing architecture comparison for all {len(models)} models on C-MAPSS FD001...\n")

    for model_type in models:
        print(f"\n=======================================================")
        print(f"          STARTING TRAINING: {model_type.upper()}")
        print(f"=======================================================")
        train_model(
            model_type=model_type,
            dataset_id="FD001",
            window_size=30,
            max_rul=125,
            epochs=25,
            batch_size=64,
            lr=1e-3,
        )
        run_full_evaluation(model_type=model_type)

    print("\n--- Generating Comparison Table and Figures ---")
    generate_results_table()
    plot_actual_vs_predicted_all()
    print("All architecture comparison experiments completed successfully!")

if __name__ == "__main__":
    main()
