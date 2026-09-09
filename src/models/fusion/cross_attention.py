"""Bidirectional audio-visual cross-attention (Phase 10).

Consumes the Phase-9 outputs ``audio_aligned [B, T, 256]`` and
``visual_tokens [B, T, 256]`` (both already time-aligned, 1:1, same length ``T``)
and models the **relationship** between the two modalities with two independent
directional cross-attention stacks:

* **audio queries visual** - ``Q = audio_aligned``, ``K,V = visual_tokens``
  -> ``audio_query_visual`` (``a2v``): visual context selected per timestep by
  the audio stream.
* **visual queries audio** - ``Q = visual_tokens``, ``K,V = audio_aligned``
  -> ``visual_query_audio`` (``v2a``): audio context selected per timestep by
  the visual stream.

The two directional outputs are fused (``concat_proj`` by default; ``gated`` /
``add`` are configurable ablations) into a single per-timestep AV representation
``fused [B, T, 256]`` for the Phase-11 sync head.

This module does **not** pool, classify, produce a sync score, or apply any
contrastive loss - those are Phases 11/12. It is fully trainable; Phase 9's
frozen encoders are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from src.models.common.cross_attention import CrossAttentionBlock
from src.models.common.temporal_transformer import SinusoidalPositionalEncoding

__all__ = [
    "CrossAttentionConfig",
    "BidirectionalCrossAttentionOutput",
    "BidirectionalCrossAttention",
]

_FUSION_MODES = ("concat_proj", "gated", "add")


@dataclass(frozen=True)
class CrossAttentionConfig:
    """The ``av_align.cross_attention`` block of ``configs/av_align.yaml``."""

    dim: int = 256
    num_heads: int = 4
    n_layers: int = 1
    ff_dim: int = 1024
    dropout: float = 0.1
    fusion: str = "concat_proj"          # production default
    add_positional_encoding: bool = False

    def __post_init__(self) -> None:
        if self.fusion not in _FUSION_MODES:
            raise ValueError(f"fusion must be one of {_FUSION_MODES}, got {self.fusion!r}")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CrossAttentionConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        blk = raw.get("av_align", raw)
        ca: dict[str, Any] = dict(blk.get("cross_attention", {}))
        allowed = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        unknown = set(ca) - allowed
        if unknown:
            raise ValueError(f"unknown cross_attention key(s): {sorted(unknown)}")
        return cls(**ca)


@dataclass
class BidirectionalCrossAttentionOutput:
    fused: Tensor               # [B, T, 256]  per-timestep fused AV representation (-> Phase 11)
    audio_query_visual: Tensor  # [B, T, 256]  a2v
    visual_query_audio: Tensor  # [B, T, 256]  v2a


class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 1,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        fusion: str = "concat_proj",
        add_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        if fusion not in _FUSION_MODES:
            raise ValueError(f"fusion must be one of {_FUSION_MODES}, got {fusion!r}")
        self.dim = dim
        self.fusion = fusion

        self.pos_encoding = (
            SinusoidalPositionalEncoding(dim, dropout=0.0) if add_positional_encoding else None
        )

        blk_kwargs = dict(dim=dim, num_heads=num_heads, n_layers=n_layers,
                          ff_dim=ff_dim, dropout=dropout)
        self.audio_to_visual = CrossAttentionBlock(**blk_kwargs)   # Q=audio, K/V=visual
        self.visual_to_audio = CrossAttentionBlock(**blk_kwargs)   # Q=visual, K/V=audio

        if fusion == "concat_proj":
            self.fuse_proj = nn.Linear(2 * dim, dim)
        elif fusion == "gated":
            self.fuse_gate = nn.Linear(2 * dim, dim)
        # "add" needs no extra parameters
        self.ln_fuse = nn.LayerNorm(dim)

    @classmethod
    def from_config(cls, cfg: CrossAttentionConfig) -> "BidirectionalCrossAttention":
        return cls(
            dim=cfg.dim, num_heads=cfg.num_heads, n_layers=cfg.n_layers,
            ff_dim=cfg.ff_dim, dropout=cfg.dropout, fusion=cfg.fusion,
            add_positional_encoding=cfg.add_positional_encoding,
        )

    def _fuse(self, a2v: Tensor, v2a: Tensor) -> Tensor:
        if self.fusion == "concat_proj":
            fused = self.fuse_proj(torch.cat([a2v, v2a], dim=-1))
        elif self.fusion == "gated":
            g = torch.sigmoid(self.fuse_gate(torch.cat([a2v, v2a], dim=-1)))
            fused = g * a2v + (1.0 - g) * v2a
        else:  # "add"
            fused = a2v + v2a
        return self.ln_fuse(fused)

    def forward(
        self,
        audio_aligned: Tensor,                       # [B, T, 256]
        visual_tokens: Tensor,                       # [B, T, 256]
        *,
        audio_key_padding_mask: Tensor | None = None,   # [B, T] bool; True -> ignore that audio key
        visual_key_padding_mask: Tensor | None = None,  # [B, T] bool; True -> ignore that visual key
    ) -> BidirectionalCrossAttentionOutput:
        if audio_aligned.shape != visual_tokens.shape:
            raise ValueError(
                f"audio_aligned {tuple(audio_aligned.shape)} and visual_tokens "
                f"{tuple(visual_tokens.shape)} must have the same shape"
            )
        if audio_aligned.dim() != 3 or audio_aligned.shape[-1] != self.dim:
            raise ValueError(f"expected [B, T, {self.dim}] inputs, got {tuple(audio_aligned.shape)}")

        if self.pos_encoding is not None:
            audio_aligned = self.pos_encoding(audio_aligned)
            visual_tokens = self.pos_encoding(visual_tokens)

        # direction 1: audio queries visual  (K/V = visual -> mask is on visual keys)
        a2v = self.audio_to_visual(
            audio_aligned, visual_tokens, key_padding_mask=visual_key_padding_mask
        )
        # direction 2: visual queries audio  (K/V = audio -> mask is on audio keys)
        v2a = self.visual_to_audio(
            visual_tokens, audio_aligned, key_padding_mask=audio_key_padding_mask
        )

        fused = self._fuse(a2v, v2a)                 # [B, T, 256]
        return BidirectionalCrossAttentionOutput(
            fused=fused, audio_query_visual=a2v, visual_query_audio=v2a
        )
