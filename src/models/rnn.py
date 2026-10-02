"""
Member 3: Ishanvi Kaushik — Stacked LSTM / GRU Recurrent Model
Author: Ishanvi Kaushik (ICT-4442 Deep Learning Mini Project)

Architecture Overview (Phase 2 LSTM configuration):
  - Input: float32 tensor shaped (batch_size, 30 cycles, 15 sensors)
      · 6 constant sensors dropped (s_1, s_5, s_10, s_16, s_18, s_19, std <= 1e-4)
      · s_6 is retained — its std=0.001389 in FD001 is above the zero-variance threshold
  - Encoder: Unidirectional LSTM with batch_first=True
      · Hidden size: 64
      · Layers: 2 (stacked)
      · Inter-layer dropout: 0.2 (applied between LSTM layers, NOT recurrent dropout)
  - Representation: hidden state of the FINAL observed cycle (last time-step)
  - Head: Dense regression head → 64 → 32 → 1
      · ReLU activation + Dropout between layers
      · No sigmoid or softmax — pure regression
  - Output: tensor shaped (batch_size,) — one RUL estimate per sample

Key gating mechanism (viva talking point):
  LSTM uses three gates to control information flow:
    · Forget gate  : decides what to discard from cell state
    · Input gate   : decides what new information to store
    · Output gate  : decides what to expose from cell state
  This allows the model to retain long-range degradation trends across
  30-cycle windows and selectively forget irrelevant sensor fluctuations.

Reproduced FD001 test results:
  MAE: 9.98 | RMSE: 13.64 | R²: 0.8842 | NASA Score: 382.8
  Early Warning Critical F1-Score: 0.7857
  Alert states are threshold-derived from predicted RUL (no classifier head).
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    Stacked LSTM (or GRU) model for RUL regression on C-MAPSS sequences.

    Captures sequential engine degradation trajectories over sliding windows
    using recurrent gating mechanisms. The final time-step hidden state is
    passed through a small dense regression head.

    Args:
        window_size  (int) : Sequence length in cycles (default: 30).
        n_features   (int) : Number of informative sensor features (default: 14).
        hidden_dim   (int) : LSTM hidden state size (default: 64).
        num_layers   (int) : Number of stacked recurrent layers (default: 2).
        rnn_type     (str) : Recurrent cell type — 'LSTM' or 'GRU' (default: 'LSTM').
        dropout    (float) : Inter-layer dropout probability (default: 0.2).
                             Applied between stacked layers only; no recurrent dropout.
    """

    def __init__(
        self,
        window_size: int = 30,
        n_features: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.2,
    ):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type.upper()

        rnn_kwargs = dict(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # Inter-layer dropout: only active when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(
                f"Unsupported rnn_type: '{rnn_type}'. Choose 'LSTM' or 'GRU'."
            )

        # Regression head: 64 → 32 → 1
        # Two-stage reduction keeps parameter count low (58 K total)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, window_size, n_features).

        Returns:
            Predicted RUL tensor of shape (batch_size,).
        """
        # r_out: (batch_size, window_size, hidden_dim)
        r_out, _ = self.rnn(x)

        # Use the hidden state at the LAST observed cycle as the sequence summary
        last_step = r_out[:, -1, :]  # (batch_size, hidden_dim)

        out = self.head(last_step)    # (batch_size, 1)
        return out.squeeze(-1)        # (batch_size,)
