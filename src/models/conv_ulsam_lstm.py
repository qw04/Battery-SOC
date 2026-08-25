"""Conv1D -> ULSAM -> LSTM -> Dense, a variant on Gong, Wang & Cheng 2022's
architecture (literature_review.md §3.2) with an LSTM in place of their SRU.
The `SRU` class below is kept only because conv_ulsam_sru_paper.py imports it
for its own faithful reproduction of the original paper.

- ULSAM (Ultra-Lightweight Subspace Attention Module): splits the conv
  feature channels into groups ("subspaces"), learns a small per-group
  attention map over the sequence position, and reweights each group by it --
  cheap channel/position attention with far fewer parameters than full
  self-attention. Originally proposed for 2D CNNs; adapted here to 1D signals.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ULSAM1d(nn.Module):
    def __init__(self, channels: int, num_subspaces: int = 4):
        super().__init__()
        if channels % num_subspaces != 0:
            num_subspaces = 1  # fall back gracefully for odd channel counts
        self.g = num_subspaces
        self.group_ch = channels // self.g
        self.attn_convs = nn.ModuleList(
            [nn.Conv1d(self.group_ch, 1, kernel_size=3, padding=1) for _ in range(self.g)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C, L), each channel-subspace reweighted by its own attention."""
        groups = torch.chunk(x, self.g, dim=1)
        outs = []
        for conv, g in zip(self.attn_convs, groups):
            attn = torch.softmax(conv(g), dim=-1) * g.shape[-1]  # (B,1,L), scaled so mean weight ~= 1
            outs.append(g * attn)
        return torch.cat(outs, dim=1)


class SRU(nn.Module):
    """Simplified Simple Recurrent Unit (Lei et al. 2017): parallel input
    projections, elementwise-only recurrence, highway output connection."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size)
        self.proj = None if input_size == hidden_size else nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, input_size) -> (B, L, hidden_size)."""
        B, L, _ = x.shape
        x_tilde = self.W(x)
        f = torch.sigmoid(self.W_f(x))
        r = torch.sigmoid(self.W_r(x))
        x_skip = x if self.proj is None else self.proj(x)

        c = x.new_zeros(B, self.hidden_size)
        cs = []
        for t in range(L):
            c = f[:, t] * c + (1 - f[:, t]) * x_tilde[:, t]
            cs.append(c)
        c_seq = torch.stack(cs, dim=1)
        h = r * torch.tanh(c_seq) + (1 - r) * x_skip
        return h


class ConvULSAMLSTM(nn.Module):
    def __init__(self, n_features: int = 3, conv_channels: int = 32, hidden_size: int = 64, num_subspaces: int = 4, num_layers: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(n_features, conv_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(conv_channels)
        self.ulsam = ULSAM1d(conv_channels, num_subspaces=num_subspaces)
        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_len, n_features) -> (batch,) SOC prediction."""
        x = x.transpose(1, 2)               # (B, n_features, L)
        x = torch.relu(self.bn(self.conv(x)))
        x = self.ulsam(x)                    # (B, C, L)
        x = x.transpose(1, 2)                # (B, L, C)
        out, _ = self.lstm(x)                # (B, L, H)
        last = out[:, -1, :]
        soc = torch.sigmoid(self.head(last)).squeeze(-1)
        return soc
