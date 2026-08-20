"""1D CNN with U-Net (encoder-decoder + skip connections) architecture for
SOC estimation, per Fan et al. 2022 (literature_review.md §3.3). Two ideas
from that paper are reproduced:

  1. Symmetric padding on every conv layer (mirrors the signal including the
     boundary sample, unlike PyTorch's built-in zero/reflect padding) to avoid
     the edge artifacts a short window's start/end otherwise introduces.
  2. A total-variation loss term (see `total_variation_loss` below), added to
     the training objective in train.py, that penalizes jumpiness in the
     model's per-timestep SOC trace without adding any extra parameters.

The network outputs a full per-timestep SOC trace (seq2seq, true to the U-Net
paper); we supervise/report the last timestep to stay consistent with the
seq-to-one label produced by src/data/windowing.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Mirror-pad including the boundary sample itself (true 'symmetric' padding,
    distinct from torch's 'reflect' mode which excludes the boundary sample)."""
    if pad == 0:
        return x
    left = x[:, :, :pad].flip(dims=[2])
    right = x[:, :, -pad:].flip(dims=[2])
    return torch.cat([left, x, right], dim=2)


class SymConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=0)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(symmetric_pad1d(x, self.pad))))
        x = F.relu(self.bn2(self.conv2(symmetric_pad1d(x, self.pad))))
        return x


class CNNUNet(nn.Module):
    def __init__(self, n_features: int = 3, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = SymConvBlock(n_features, c)
        self.enc2 = SymConvBlock(c, c * 2)
        self.bottleneck = SymConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool1d(2)

        self.up2 = nn.ConvTranspose1d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = SymConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose1d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = SymConvBlock(c * 2, c)

        self.head = nn.Conv1d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """x: (batch, window_len, n_features) -> (batch,) SOC at last timestep
        (or (batch, window_len) full trace if return_sequence=True, used by
        the total-variation loss during training)."""
        x = x.transpose(1, 2)  # (B, n_features, L)

        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        b = self.bottleneck(p2)

        d2 = self.up2(b)
        d2 = _match_length(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = _match_length(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seq = torch.sigmoid(self.head(d1)).squeeze(1)  # (B, L)
        if return_sequence:
            return seq
        return seq[:, -1]


def _match_length(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Odd window lengths can shift the up-sampled length off by one vs. the
    skip connection; trim/pad the trailing edge to match."""
    diff = ref.shape[-1] - x.shape[-1]
    if diff > 0:
        x = F.pad(x, (0, diff))
    elif diff < 0:
        x = x[:, :, :ref.shape[-1]]
    return x


def total_variation_loss(seq: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between consecutive timesteps -- penalizes a
    jumpy predicted SOC trace, per Fan et al. 2022's proposed TV regularizer."""
    return torch.mean(torch.abs(seq[:, 1:] - seq[:, :-1]))
