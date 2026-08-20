"""Direct V/I/T -> SOC LSTM, replicating Chemali et al. 2018 (McMaster paper,
literature_review.md §2.1): stacked LSTM layers reading a window of
[voltage, current, temperature] and mapping straight to SOC, with no battery
model or filter in the loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMDirect(nn.Module):
    def __init__(self, n_features: int = 3, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_len, n_features) -> (batch,) SOC prediction."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final timestep's hidden state
        soc = torch.sigmoid(self.head(last)).squeeze(-1)  # SOC in [0,1]
        return soc
