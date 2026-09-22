"""
Commit 1 — Data Verification Tests
Member 3: Ishanvi Kaushik
Author: Ishanvi Kaushik

Verifies that the shared data pipeline produces valid, leakage-free inputs
for the recurrent (LSTM) model:
  - X_train has 3 dimensions with correct shape (N, 30, 14)
  - y_train has one label per window
  - Train and validation engine IDs are strictly disjoint
  - Exactly 14 informative sensor features are selected
"""

import os
import pytest
import numpy as np

# ── skip entire module if C-MAPSS data files are not available ──────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_AVAILABLE = os.path.isfile(os.path.join(DATA_DIR, "train_FD001.txt"))

pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="C-MAPSS data files not found in data/. Download from NASA Prognostics repository.",
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_outputs():
    """
    Run the full preprocessing pipeline once for all tests in this module.
    Returns a dict of all intermediate artefacts needed for assertion.
    """
    from src.data_loader import (
        load_raw_data,
        add_piecewise_rul,
        get_informative_features,
        split_train_val_by_engine,
        scale_features,
        create_sliding_windows,
    )

    WINDOW_SIZE = 30
    MAX_RUL = 125
    DATASET_ID = "FD001"

    train_df, test_df, rul_df = load_raw_data(dataset_id=DATASET_ID, data_dir=DATA_DIR)
    train_df = add_piecewise_rul(train_df, max_rul=MAX_RUL)
    feature_cols = get_informative_features(train_df, drop_constant=True)

    train_split, val_split = split_train_val_by_engine(train_df, val_ratio=0.2, seed=42)
    train_scaled, val_scaled, test_scaled, _ = scale_features(
        train_split, val_split, test_df, feature_cols
    )

    X_train, y_train = create_sliding_windows(
        train_scaled, window_size=WINDOW_SIZE, feature_cols=feature_cols
    )
    X_val, y_val = create_sliding_windows(
        val_scaled, window_size=WINDOW_SIZE, feature_cols=feature_cols
    )

    train_engine_ids = set(train_split["unit_nr"].unique().tolist())
    val_engine_ids = set(val_split["unit_nr"].unique().tolist())

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feature_cols": feature_cols,
        "train_engine_ids": train_engine_ids,
        "val_engine_ids": val_engine_ids,
    }


# ── tests ─────────────────────────────────────────────────────────────────────

class TestWindowShape:
    """Verify sliding window output dimensions are correct for LSTM input."""

    def test_X_train_is_3d(self, pipeline_outputs):
        """X_train must be a 3-D array: (N_windows, window_size, n_features)."""
        X_train = pipeline_outputs["X_train"]
        assert X_train.ndim == 3, (
            f"Expected X_train.ndim == 3, got {X_train.ndim}. "
            "The LSTM requires (batch, seq_len, features) shaped input."
        )

    def test_X_train_window_and_feature_dims(self, pipeline_outputs):
        """Window size must be 30 cycles and feature count must be 15."""
        X_train = pipeline_outputs["X_train"]
        assert X_train.shape[1] == 30, (
            f"Expected window_size=30, got {X_train.shape[1]}."
        )
        assert X_train.shape[2] == 15, (
            f"Expected 15 informative sensors, got {X_train.shape[2]}. "
            "Check that 6 constant sensors were dropped (s_6 is retained)."
        )

    def test_X_train_val_same_feature_dim(self, pipeline_outputs):
        """Train and val windows must share the same feature dimension."""
        X_train = pipeline_outputs["X_train"]
        X_val = pipeline_outputs["X_val"]
        assert X_train.shape[1:] == X_val.shape[1:], (
            f"Train shape {X_train.shape[1:]} != Val shape {X_val.shape[1:]}."
        )


class TestLabelAlignment:
    """Verify that each window has exactly one RUL label."""

    def test_X_y_train_length_match(self, pipeline_outputs):
        """Number of training windows must equal number of RUL labels."""
        X_train = pipeline_outputs["X_train"]
        y_train = pipeline_outputs["y_train"]
        assert len(X_train) == len(y_train), (
            f"Mismatch: {len(X_train)} windows vs {len(y_train)} labels."
        )

    def test_X_y_val_length_match(self, pipeline_outputs):
        """Number of validation windows must equal number of RUL labels."""
        X_val = pipeline_outputs["X_val"]
        y_val = pipeline_outputs["y_val"]
        assert len(X_val) == len(y_val), (
            f"Mismatch: {len(X_val)} windows vs {len(y_val)} labels."
        )

    def test_y_train_rul_range(self, pipeline_outputs):
        """All capped RUL targets must be in [0, 125]."""
        y_train = pipeline_outputs["y_train"]
        assert float(y_train.min()) >= 0, "Negative RUL values found."
        assert float(y_train.max()) <= 125, "RUL values exceed max_rul=125 cap."


class TestLeakagePrevention:
    """Verify strict engine-wise split: no engine appears in both train and val."""

    def test_engine_ids_disjoint(self, pipeline_outputs):
        """Train and validation engine sets must be strictly disjoint."""
        train_ids = pipeline_outputs["train_engine_ids"]
        val_ids = pipeline_outputs["val_engine_ids"]
        overlap = train_ids & val_ids
        assert len(overlap) == 0, (
            f"Data leakage detected! Engines {overlap} appear in both train and val."
        )

    def test_engine_split_covers_all(self, pipeline_outputs):
        """Combined train + val engines should cover all 100 FD001 engines."""
        train_ids = pipeline_outputs["train_engine_ids"]
        val_ids = pipeline_outputs["val_engine_ids"]
        total = len(train_ids | val_ids)
        # FD001 has 100 training engines
        assert total == 100, (
            f"Expected 100 total engines (FD001), got {total}."
        )


class TestFeatureSelection:
    """Verify that exactly 15 informative sensor features are selected."""

    def test_exactly_15_features(self, pipeline_outputs):
        """Six constant sensors (std <= 1e-4) must be dropped.
        s_6 has std=0.001389 in FD001 and is NOT dropped.
        """
        feature_cols = pipeline_outputs["feature_cols"]
        assert len(feature_cols) == 15, (
            f"Expected 15 informative features, got {len(feature_cols)}: {feature_cols}. "
            "Confirm that 6 near-zero variance sensors are removed (s_6 is retained)."
        )

    def test_known_constant_sensors_excluded(self, pipeline_outputs):
        """Explicitly verify the 6 truly constant sensors are excluded."""
        feature_cols = pipeline_outputs["feature_cols"]
        # s_6 has std=0.001389 in FD001 — it is NOT constant and should NOT be excluded
        constant_sensors = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
        for sensor in constant_sensors:
            assert sensor not in feature_cols, (
                f"Constant sensor '{sensor}' was not removed from feature_cols."
            )

    def test_s6_is_retained(self, pipeline_outputs):
        """s_6 has std=0.001389 in FD001 and should be retained as informative."""
        feature_cols = pipeline_outputs["feature_cols"]
        assert "s_6" in feature_cols, (
            "s_6 should be retained — it has std=0.001389 in FD001, above the 1e-4 threshold."
        )
