"""CNN front-end for the audio branch (Phase 4 baseline).

Takes a log-mel spectrogram and produces a sequence of temporal tokens
``[B, T', D]`` rather than collapsing the clip to a single vector - the pooling
step lives in the task head, so this encoder can be reused unchanged by the
audio-only spoof classifier (Phase 4), the audio Transformer (Phase 5), and the
audio branch of the AV sync model.

Each ``Conv-BN-ReLU x2`` block halves both the mel and time axes, so
``T' = ceil(T / 2**num_blocks)`` and the effective token hop is
``mel.hop_length * 2**num_blocks`` samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.config import ModelConfig

__all__ = ["AudioCNNEncoder", "AudioEncoderOutput"]


@dataclass
class AudioEncoderOutput:
    tokens: torch.Tensor      # [B, T', D]
    time_downsample: int      # T -> T' factor, and mel-frame -> token hop multiplier


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AudioCNNEncoder(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_mels: int) -> None:
        super().__init__()
        channels = model_cfg.audio_cnn_channels
        self.n_mels = n_mels
        self.time_downsample = 2 ** len(channels)
        self.output_dim = model_cfg.audio_embedding_dim

        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(_ConvBlock(in_ch, out_ch, model_cfg.audio_cnn_dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Collapse the (downsampled) frequency axis, then project each frame to D.
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(channels[-1], self.output_dim)

    def forward(self, mel: torch.Tensor) -> AudioEncoderOutput:
        """``mel``: ``[B, n_mels, T]`` or ``[B, 1, n_mels, T]`` -> tokens ``[B, T', D]``."""

        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        if mel.dim() != 4 or mel.shape[1] != 1:
            raise ValueError(f"expected [B, n_mels, T] or [B, 1, n_mels, T], got {tuple(mel.shape)}")

        x = self.blocks(mel)                 # [B, C, n_mels', T']
        x = self.freq_pool(x).squeeze(2)     # [B, C, T']
        x = x.transpose(1, 2)                # [B, T', C]
        tokens = self.proj(x)               # [B, T', D]
        return AudioEncoderOutput(tokens=tokens, time_downsample=self.time_downsample)
