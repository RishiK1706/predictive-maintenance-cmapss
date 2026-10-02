import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_raw_data, get_informative_features, add_piecewise_rul

os.makedirs("figures", exist_ok=True)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def plot_sensor_trajectories(dataset_id="FD001", sample_engines=[1, 2, 5, 10]):
    """
    Plot degradation trends of representative sensors over operational cycles.
    """
    train_df, _, _ = load_raw_data(dataset_id=dataset_id)
    informative_sensors = get_informative_features(train_df, drop_constant=True)

    # Pick top 6 sensors with highest variance / dynamic range
    selected_sensors = ["s_2", "s_3", "s_4", "s_7", "s_11", "s_12"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, sensor in enumerate(selected_sensors):
        ax = axes[i]
        for engine_id in sample_engines:
            engine_data = train_df[train_df["unit_nr"] == engine_id]
            ax.plot(engine_data["time_cycles"], engine_data[sensor], label=f"Engine {engine_id}", alpha=0.8, linewidth=1.5)

        ax.set_title(f"Sensor: {sensor}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sensor Value")
        if i >= 4:
            ax.set_xlabel("Operational Cycles")
        ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sample_engines), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    save_path = os.path.join("figures", f"sensor_degradation_{dataset_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sensor degradation plot to {save_path}")


def plot_rul_distribution(dataset_id="FD001"):
    """
    Plot distribution of total engine life and piecewise RUL target.
    """
    train_df, _, _ = load_raw_data(dataset_id=dataset_id)
    engine_lifetimes = train_df.groupby("unit_nr")["time_cycles"].max()

    plt.figure(figsize=(8, 5))
    sns.histplot(engine_lifetimes, kde=True, color="teal", bins=20)
    plt.axvline(engine_lifetimes.mean(), color="red", linestyle="--", label=f"Mean Life: {engine_lifetimes.mean():.1f} cycles")
    plt.title(f"C-MAPSS {dataset_id} Engine Lifespan Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Total Operational Cycles Until Failure")
    plt.ylabel("Engine Count")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join("figures", f"engine_life_distribution_{dataset_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved engine lifespan distribution plot to {save_path}")


if __name__ == "__main__":
    print("Generating exploratory data analysis plots...")
    plot_sensor_trajectories()
    plot_rul_distribution()
