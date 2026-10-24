"""
Sequence models for NASA POWER hourly weather and solar irradiance forecasting.

Provides:
- WeatherLSTM: Multi-layer recurrent neural network for sliding window sequence prediction.
- Causal, lookahead-free forward pass mapping past observations [t-23, ..., t] to t+1.
"""
from typing import Tuple
import torch
import torch.nn as nn


class WeatherLSTM(nn.Module):
    """
    LSTM sequence forecasting model for next-hour solar irradiance.

    Attributes:
        input_dim: Number of meteorological features per timestep (default: 4).
        hidden_dim: Number of hidden units in LSTM cell.
        num_layers: Number of stacked LSTM layers.
        dropout_rate: Dropout probability applied between LSTM layers and before projection.
        lstm: PyTorch LSTM module configured with batch_first=True.
        fc_head: Regression projection mapping last recurrent state to scalar irradiance.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.bidirectional = bool(bidirectional)

        # PyTorch LSTM dropout applies only across stacked layers (num_layers > 1)
        lstm_dropout = self.dropout_rate if self.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        direction_mult = 2 if self.bidirectional else 1
        head_in = self.hidden_dim * direction_mult

        if self.dropout_rate > 0.0:
            self.fc_head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(head_in, 1),
            )
        else:
            self.fc_head = nn.Linear(head_in, 1)

    def get_parameter_counts(self) -> Tuple[int, int]:
        """Return tuple of (trainable_parameters, total_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input feature tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Irradiance predictions of shape (batch_size, 1).
        """
        assert x.ndim == 3, f"Expected 3D input (batch, seq_len, features), got {x.shape}"
        assert x.shape[-1] == self.input_dim, f"Feature dimension mismatch: {x.shape[-1]} != {self.input_dim}"

        # lstm_out: (batch_size, seq_len, hidden_dim * directions)
        lstm_out, _ = self.lstm(x)

        # Take representation at the final timestep (forecast origin t)
        last_timestep = lstm_out[:, -1, :]

        # Project to single next-hour prediction (t + 1)
        pred = self.fc_head(last_timestep)
        return pred
