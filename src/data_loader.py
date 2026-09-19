import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# C-MAPSS column specifications
COLUMN_NAMES = (
    ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

# Sensors known to have zero or near-zero variance in FD001
CONSTANT_SENSORS_FD001 = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]


def load_raw_data(dataset_id="FD001", data_dir="data"):
    """
    Load raw space-delimited C-MAPSS files into pandas DataFrames.
    """
    train_path = os.path.join(data_dir, f"train_{dataset_id}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset_id}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset_id}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_end"])

    return train_df, test_df, rul_df


def add_piecewise_rul(df, max_rul=125):
    """
    Calculate Remaining Useful Life (RUL) for training trajectories
    and apply piecewise linear capping at max_rul.
    """
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    raw_rul = max_cycles - df["time_cycles"]
    df["RUL"] = raw_rul.clip(upper=max_rul) if max_rul else raw_rul
    return df


def get_informative_features(df, drop_constant=True):
    """
    Identify informative sensor/setting columns by filtering zero-variance features.
    """
    all_sensor_cols = [f"s_{i}" for i in range(1, 22)]
    if drop_constant:
        # Calculate standard deviation per column
        stds = df[all_sensor_cols].std()
        informative = stds[stds > 1e-4].index.tolist()
        return informative
    return all_sensor_cols


def split_train_val_by_engine(df, val_ratio=0.2, seed=42):
    """
    Split DataFrame by unit_nr to avoid data leakage across adjacent time windows.
    """
    np.random.seed(seed)
    unique_units = df["unit_nr"].unique()
    np.random.shuffle(unique_units)

    n_val = int(len(unique_units) * val_ratio)
    val_units = set(unique_units[:n_val])
    train_units = set(unique_units[n_val:])

    train_df = df[df["unit_nr"].isin(train_units)].copy()
    val_df = df[df["unit_nr"].isin(val_units)].copy()

    return train_df, val_df


def scale_features(train_df, val_df, test_df, feature_cols):
    """
    Fit MinMaxScaler on train_df ONLY, and transform train, val, and test dataframes.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler


def create_sliding_windows(df, window_size=30, feature_cols=None, target_col="RUL"):
    """
    Convert sequence dataframe into 3D sliding window arrays:
    X shape: (N_windows, window_size, N_features)
    y shape: (N_windows,)
    """
    if feature_cols is None:
        feature_cols = [f"s_{i}" for i in range(1, 22)]

    X_list = []
    y_list = []

    for unit_id, group in df.groupby("unit_nr"):
        data = group[feature_cols].values
        targets = group[target_col].values if target_col in group.columns else None

        num_cycles = len(group)
        if num_cycles < window_size:
            continue

        for i in range(num_cycles - window_size + 1):
            X_list.append(data[i : i + window_size])
            if targets is not None:
                # Target corresponds to the end cycle of the window
                y_list.append(targets[i + window_size - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    return X, y


def build_test_dataset(test_df, rul_df, window_size=30, feature_cols=None, max_rul=125):
    """
    Build test windows and reconstruct ground truth RUL using RUL_FD001.txt.
    """
    if feature_cols is None:
        feature_cols = [f"s_{i}" for i in range(1, 22)]

    X_test_list = []
    y_test_list = []

    for unit_id, group in test_df.groupby("unit_nr"):
        true_end_rul = rul_df.iloc[unit_id - 1]["RUL_end"]
        max_observed_cycle = group["time_cycles"].max()
        total_lifetime = max_observed_cycle + true_end_rul

        # Calculate exact RUL for every row in test engine trajectory
        group = group.copy()
        raw_rul = total_lifetime - group["time_cycles"]
        group["RUL"] = raw_rul.clip(upper=max_rul) if max_rul else raw_rul

        data = group[feature_cols].values
        targets = group["RUL"].values

        num_cycles = len(group)
        if num_cycles < window_size:
            # Pad sequence if test engine trajectory is shorter than window_size
            pad_len = window_size - num_cycles
            padded_data = np.pad(data, ((pad_len, 0), (0, 0)), mode="edge")
            X_test_list.append(padded_data)
            y_test_list.append(targets[-1])
        else:
            # Take windows up to the last available cycle
            X_test_list.append(data[-window_size:])
            y_test_list.append(targets[-1])

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    return X_test, y_test
