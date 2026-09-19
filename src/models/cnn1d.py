import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    Member 2: 1D Convolutional Neural Network (1D-CNN).
    Extracts localized spatial-temporal degradation patterns across sensors.
    """

    def __init__(self, window_size=30, n_features=14, channels=[32, 64, 128], dropout=0.2):
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            nn.Linear(channels[2], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, n_features) -> permute to (batch_size, n_features, window_size)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1)  # (batch_size, channels[2])
        out = self.head(x)
        return out.squeeze(-1)
