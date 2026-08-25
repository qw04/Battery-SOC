"""Faithful reproduction of Gong, Wang & Cheng (2022)'s Conv+ULSAM+SRU
architecture, per the exact layer diagram (see literature_review.md §3.2 and
the internship report cross-referenced during project discussion) -- as
opposed to `conv_ulsam_lstm.py`'s looser from-scratch interpretation built
before those diagrams were available.

Differences from `conv_ulsam_lstm.py`:
  - Conv channels 3 -> 10 -> 20 (two conv layers, not one 3 -> 32 layer).
  - ULSAM includes a residual connection the paper states explicitly:
    Output = (Input (x) Atten) (+) Input -- ours previously only multiplied.
  - ULSAM's attention branch is deeper: Conv1x1 -> BN -> Mish ->
    MaxPool(k=3, stride=1) -> Conv1x1 -> BN -> Mish -> Softmax, not a single
    conv straight into softmax.
  - SRU runs over the FLATTENED (channels * time) feature map treated as one
    long pseudo-sequence of scalars, not over the natural time axis with
    channels as the per-step feature vector -- an unusual choice, but it's
    what the paper's diagram shows ("Flatten -> 200x1 -> SRU,100 -> 100x1"
    for a 20-channel, 10-timestep input).
  - SRU hidden size 100 (not 64); FC head 100 -> 50 -> 1; Mish activation
    throughout (not ReLU).
  - Meant to pair with a SHORT input window (5-10 timesteps, the paper's own
    "input length" ablation) rather than this project's usual 64-step
    windows, though the forward pass works for any window length.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv_ulsam_lstm import SRU


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class ULSAM1dResidual(nn.Module):
    """Per-subspace attention with the paper's deeper branch and explicit
    residual connection: Output = (Input (x) Atten) (+) Input."""

    def __init__(self, channels: int, num_subspaces: int = 4):
        super().__init__()
        if channels % num_subspaces != 0:
            num_subspaces = 1
        self.g = num_subspaces
        self.group_ch = channels // self.g
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.group_ch, self.group_ch, kernel_size=1),
                    nn.BatchNorm1d(self.group_ch),
                    Mish(),
                    nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                    nn.Conv1d(self.group_ch, 1, kernel_size=1),
                    nn.BatchNorm1d(1),
                    Mish(),
                )
                for _ in range(self.g)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C, L)."""
        groups = torch.chunk(x, self.g, dim=1)
        outs = []
        for branch, g in zip(self.branches, groups):
            attn = torch.softmax(branch(g), dim=-1)  # (B, 1, L)
            outs.append(g * attn + g)  # residual connection
        return torch.cat(outs, dim=1)


class ConvULSAMSRUPaper(nn.Module):
    def __init__(self, n_features: int = 3, num_subspaces: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(10)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(20)
        self.mish = Mish()
        self.ulsam = ULSAM1dResidual(20, num_subspaces=num_subspaces)
        self.sru = SRU(input_size=1, hidden_size=100)
        self.head = nn.Sequential(nn.Linear(100, 50), Mish(), nn.Linear(50, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_len, n_features) -> (batch,) SOC prediction."""
        x = x.transpose(1, 2)  # (B, n_features, L)
        x = self.mish(self.bn1(self.conv1(x)))
        x = self.mish(self.bn2(self.conv2(x)))
        x = self.ulsam(x)  # (B, 20, L)

        b = x.shape[0]
        flat = x.reshape(b, -1, 1)  # (B, 20*L, 1) -- flatten channels*time into one long pseudo-sequence
        h = self.sru(flat)  # (B, 20*L, 100)
        last = h[:, -1, :]
        soc = torch.sigmoid(self.head(last)).squeeze(-1)
        return soc
