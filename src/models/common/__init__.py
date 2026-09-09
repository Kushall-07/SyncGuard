"""Modality-agnostic building blocks shared by the audio and visual branches."""

from src.models.common.temporal_transformer import (
    SinusoidalPositionalEncoding,
    TemporalTransformerEncoder,
)

__all__ = ["SinusoidalPositionalEncoding", "TemporalTransformerEncoder"]
