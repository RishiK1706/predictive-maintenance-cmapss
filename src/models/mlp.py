import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Member 1: Classical Multi-Layer Perceptron Baseline.
    Flattens sequence window (window_size * n_features) into a 1D tabular vector.
    """

    def __init__(self, window_size=30, n_features=14, hidden_dims=[256, 128, 64], dropout=0.2):
        super(MLPBaseline, self).__init__()
        input_dim = window_size * n_features

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, window_size, n_features)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        out = self.net(x_flat)
        return out.squeeze(-1)
