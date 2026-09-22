"""
Commit 2 — Recurrent Model Shape & Correctness Tests
Member 3: Ishanvi Kaushik

Verifies the stacked LSTM (RNNModel) forward pass:
  - Output shape is (batch_size,) — one RUL per sample
  - All output values are finite (no NaN / Inf)
  - Works for both LSTM and GRU variants
  - Correct behaviour when batch_size=1 (edge case)
"""

import pytest
import torch
import torch.nn as nn
from src.models.rnn import RNNModel

# ── shared model configurations ───────────────────────────────────────────────

BATCH_SIZE = 8
WINDOW_SIZE = 30
N_FEATURES = 15  # 15 informative sensors after constant-sensor removal (s_6 retained, std=0.001389)


@pytest.fixture
def lstm_model():
    """Phase 2 recommended LSTM configuration."""
    return RNNModel(
        window_size=WINDOW_SIZE,
        n_features=N_FEATURES,
        hidden_dim=64,
        num_layers=2,
        rnn_type="LSTM",
        dropout=0.2,
    ).eval()


@pytest.fixture
def gru_model():
    """GRU variant for comparison (optional extension)."""
    return RNNModel(
        window_size=WINDOW_SIZE,
        n_features=N_FEATURES,
        hidden_dim=64,
        num_layers=2,
        rnn_type="GRU",
        dropout=0.2,
    ).eval()


# ── LSTM tests ────────────────────────────────────────────────────────────────

class TestLSTMForwardPass:
    """Core shape and numerical correctness tests for the stacked LSTM."""

    def test_output_shape(self, lstm_model):
        """Output must be 1-D with length equal to batch size."""
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        with torch.no_grad():
            y = lstm_model(x)
        assert y.shape == (BATCH_SIZE,), (
            f"Expected output shape ({BATCH_SIZE},), got {y.shape}. "
            "Ensure the model returns a 1-D tensor (no sigmoid/softmax)."
        )

    def test_output_is_finite(self, lstm_model):
        """All predicted RUL values must be finite (no NaN or Inf)."""
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        with torch.no_grad():
            y = lstm_model(x)
        assert torch.isfinite(y).all(), (
            "Model produced NaN or Inf in output. Check weight initialisation."
        )

    def test_single_sample_batch(self, lstm_model):
        """Model must handle batch_size=1 without errors (edge case for inference)."""
        x = torch.randn(1, WINDOW_SIZE, N_FEATURES)
        with torch.no_grad():
            y = lstm_model(x)
        assert y.shape == (1,), f"Expected shape (1,), got {y.shape}."
        assert torch.isfinite(y).all()

    def test_input_shape_matches_spec(self, lstm_model):
        """Explicitly verify the model accepts (batch, 30 cycles, 15 features)."""
        x = torch.randn(BATCH_SIZE, 30, N_FEATURES)
        with torch.no_grad():
            y = lstm_model(x)
        assert y.shape == (BATCH_SIZE,)

    def test_no_output_sigmoid_clipping(self, lstm_model):
        """RUL regression output should NOT use Sigmoid — head must end with nn.Linear."""
        # Structural check: verify the final layer in the head is nn.Linear
        last_layer = list(lstm_model.head.children())[-1]
        assert isinstance(last_layer, nn.Linear), (
            f"Final head layer should be nn.Linear (regression), got {type(last_layer).__name__}. "
            "Make sure no Sigmoid/Softmax is applied at the output."
        )


class TestLSTMDeterminism:
    """Verify that the same input produces the same output in eval mode."""

    def test_deterministic_eval(self, lstm_model):
        """Two forward passes with the same input must produce identical outputs."""
        torch.manual_seed(0)
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        with torch.no_grad():
            y1 = lstm_model(x)
            y2 = lstm_model(x)
        assert torch.allclose(y1, y2), "Model is non-deterministic in eval mode."


# ── GRU tests ─────────────────────────────────────────────────────────────────

class TestGRUForwardPass:
    """Smoke tests for GRU variant (same interface, different cell)."""

    def test_gru_output_shape(self, gru_model):
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        with torch.no_grad():
            y = gru_model(x)
        assert y.shape == (BATCH_SIZE,)
        assert torch.isfinite(y).all()


# ── invalid config tests ──────────────────────────────────────────────────────

class TestInvalidConfig:
    """Verify that unsupported rnn_type raises a clear error."""

    def test_invalid_rnn_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported rnn_type"):
            RNNModel(rnn_type="VANILLA")
