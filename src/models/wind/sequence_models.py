"""
Sequence deep learning models for wind resource forecasting.

Provides:
- WindLSTM: Stacked LSTM taking (N, sequence_length, input_dim) and predicting wind speed (N, 1).
"""
from typing import Tuple
import torch
import torch.nn as nn


class WindLSTM(nn.Module):
    """
    Stacked LSTM sequence model for causal wind speed regression.

    Architecture:
    - Input: Tensor of shape (batch_size, sequence_length, input_dim).
    - LSTM backbone: 1 to 2 stacked LSTM layers with optional dropout between layers.
    - Head: Dropout + Linear projection mapping final hidden state to scalar wind speed output.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Predictions tensor of shape (batch_size, 1).
        """
        lstm_out, (hn, cn) = self.lstm(x)
        # hn[-1] is the final layer's hidden state at the last time step
        last_hidden = hn[-1]
        out = self.head(last_hidden)
        return out

    def get_parameter_counts(self) -> Tuple[int, int]:
        """Return (trainable_parameters, total_parameters)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
