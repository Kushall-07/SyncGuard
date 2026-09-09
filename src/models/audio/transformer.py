"""Transformer encoder over audio temporal tokens (Phase 5).

Sits on top of the Phase 4 CNN front-end: it takes the token sequence
``[B, T', D]`` the CNN emits, adds sinusoidal positional information, and refines
it with a small ``nn.TransformerEncoder`` stack, returning ``[B, T', D]``
unchanged in shape so the same :class:`~src.models.heads.spoof_head.SpoofHead`
(and, later, the cross-attention fusion) consumes it.

The mechanics now live in the modality-agnostic
:class:`~src.models.common.temporal_transformer.TemporalTransformerEncoder`
(shared with the Phase 7 visual branch); this class is just the audio-config
adapter over it. Subclassing keeps the submodule names (``pos_encoding``,
``encoder``) identical, so existing Phase 5 checkpoints still load.
"""

from __future__ import annotations

from src.config import ModelConfig
from src.models.common.temporal_transformer import (
    SinusoidalPositionalEncoding,
    TemporalTransformerEncoder,
)

__all__ = ["SinusoidalPositionalEncoding", "AudioTransformerEncoder"]


class AudioTransformerEncoder(TemporalTransformerEncoder):
    """``[B, T, D]`` -> ``[B, T, D]``, configured from the audio ``ModelConfig``."""

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__(
            d_model=model_cfg.audio_embedding_dim,
            n_heads=model_cfg.num_heads,
            ff_dim=model_cfg.audio_tf_ff_dim,
            dropout=model_cfg.audio_tf_dropout,
            num_layers=model_cfg.audio_tf_layers,
        )
