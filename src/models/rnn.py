import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    Member 3: Sequential Memory Model (LSTM or GRU).
    Captures temporal progression and long-term degradation trajectories over time windows.
    """

    def __init__(self, window_size=30, n_features=14, hidden_dim=64, num_layers=2, rnn_type="LSTM", dropout=0.2):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type

        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, n_features)
        r_out, _ = self.rnn(x)  # r_out shape: (batch_size, window_size, hidden_dim)
        # Pool last hidden state across the sequence
        last_step = r_out[:, -1, :]
        out = self.head(last_step)
        return out.squeeze(-1)
