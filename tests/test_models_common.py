"""Phase 7 unit tests: the shared TemporalTransformerEncoder and its audio adapter."""

from __future__ import annotations

import torch

from src.config import ModelConfig
from src.models.audio.transformer import AudioTransformerEncoder
from src.models.common import SinusoidalPositionalEncoding, TemporalTransformerEncoder


def test_positional_encoding_adds_and_preserves_shape() -> None:
    pe = SinusoidalPositionalEncoding(16, max_len=32)
    x = torch.zeros(2, 10, 16)
    out = pe(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)          # something was added


def test_temporal_encoder_shape_roundtrip() -> None:
    enc = TemporalTransformerEncoder(d_model=32, n_heads=4, ff_dim=64, dropout=0.0, num_layers=2)
    x = torch.randn(3, 12, 32)
    assert enc(x).shape == (3, 12, 32)


def test_audio_adapter_matches_shared_encoder_bitwise() -> None:
    cfg = ModelConfig(audio_embedding_dim=32, num_heads=4, audio_tf_ff_dim=64,
                      audio_tf_dropout=0.0, audio_tf_layers=2)
    adapter = AudioTransformerEncoder(cfg).eval()
    shared = TemporalTransformerEncoder(
        d_model=32, n_heads=4, ff_dim=64, dropout=0.0, num_layers=2
    ).eval()
    shared.load_state_dict(adapter.state_dict())   # identical submodule names
    x = torch.randn(2, 9, 32)
    assert torch.allclose(adapter(x), shared(x), atol=1e-6)
