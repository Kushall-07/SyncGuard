"""Unit tests for explicit checkpoint-compatibility validation (Phase 2 of the
production AV-inference correctness audit).

Covers `load_state_dict_validated` directly (valid checkpoint, missing key,
unexpected key, allowed-exception normalization) and, at the
`SyncGuardPredictor` integration level, the specific failure modes Phase 2
asks for: wrong embedding dimension, wrong configuration (cross-attention
layer count), and a genuinely compatible checkpoint loading cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.inference.checkpoint_validation import (
    CheckpointCompatibilityError,
    load_state_dict_validated,
)
from src.models.fusion.cross_attention import BidirectionalCrossAttention, CrossAttentionConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig


class _TinyModule(nn.Module):
    def __init__(self, extra_param: bool = False) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        if extra_param:
            self.c = nn.Linear(4, 4)


def test_valid_checkpoint_loads_without_error() -> None:
    module = _TinyModule()
    load_state_dict_validated(module, _TinyModule().state_dict(), context="tiny module")


def test_missing_key_raises_compatibility_error() -> None:
    module = _TinyModule(extra_param=True)  # has a.*, b.*, c.*
    source = _TinyModule(extra_param=False).state_dict()  # only a.*, b.*
    with pytest.raises(CheckpointCompatibilityError, match="missing_keys"):
        load_state_dict_validated(module, source, context="tiny module missing c")


def test_unexpected_key_raises_compatibility_error() -> None:
    module = _TinyModule(extra_param=False)  # only a.*, b.*
    source = _TinyModule(extra_param=True).state_dict()  # a.*, b.*, c.* (extra)
    with pytest.raises(CheckpointCompatibilityError, match="unexpected_keys"):
        load_state_dict_validated(module, source, context="tiny module with extra c")


def test_allowed_missing_key_is_tolerated() -> None:
    module = _TinyModule(extra_param=True)
    source = _TinyModule(extra_param=False).state_dict()
    # Explicitly documenting that "c" is allowed to be absent must let this load.
    load_state_dict_validated(
        module, source, context="tiny module", allowed_missing=["c.weight", "c.bias"]
    )


def test_allowed_unexpected_prefix_is_tolerated() -> None:
    module = _TinyModule(extra_param=False)
    source = _TinyModule(extra_param=True).state_dict()
    load_state_dict_validated(
        module, source, context="tiny module", allowed_unexpected_prefixes=["c."]
    )


def test_wrong_shape_raises_regardless_of_allow_lists() -> None:
    """Shape mismatches for a key present in both source and target are never
    tolerated - `strict`/allow-lists only govern key presence, not shape, so a
    wrong embedding dimension must still raise even with maximal allow-lists."""
    module = _TinyModule()
    bad_source = {"a.weight": torch.zeros(8, 8), "a.bias": torch.zeros(8), **{
        k: v for k, v in _TinyModule().state_dict().items() if k.startswith("b.")
    }}
    with pytest.raises(RuntimeError):
        load_state_dict_validated(
            module, bad_source, context="tiny module wrong shape",
            allowed_missing=["a.weight", "a.bias"], allowed_unexpected_prefixes=["a."],
        )


# --------------------------------------------------------------------------- integration: cross_attention / sync_head


def test_compatible_cross_attention_checkpoint_loads_cleanly() -> None:
    cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=512, dropout=0.1)
    source = BidirectionalCrossAttention.from_config(cfg)
    target = BidirectionalCrossAttention.from_config(cfg)
    load_state_dict_validated(target, source.state_dict(), context="cross_attention")


def test_wrong_embedding_dimension_raises_compatibility_error() -> None:
    """A checkpoint trained with dim=256 loaded into a module built for a
    different dim: PyTorch raises a shape-mismatch RuntimeError (not our
    CheckpointCompatibilityError, since this is a shape issue, not a key-set
    issue) - either way inference must not proceed silently."""
    source_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=512, dropout=0.1)
    target_cfg = CrossAttentionConfig(dim=128, num_heads=4, n_layers=1, ff_dim=512, dropout=0.1)
    source = BidirectionalCrossAttention.from_config(source_cfg)
    target = BidirectionalCrossAttention.from_config(target_cfg)
    with pytest.raises(RuntimeError):
        load_state_dict_validated(target, source.state_dict(), context="cross_attention wrong dim")


def test_wrong_layer_count_configuration_raises_compatibility_error() -> None:
    """A checkpoint trained with n_layers=2 loaded into a module configured
    for n_layers=1: the extra layer's keys become 'unexpected', which must
    raise rather than silently discard a trained layer."""
    source_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=2, ff_dim=512, dropout=0.1)
    target_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=512, dropout=0.1)
    source = BidirectionalCrossAttention.from_config(source_cfg)
    target = BidirectionalCrossAttention.from_config(target_cfg)
    with pytest.raises(CheckpointCompatibilityError, match="unexpected_keys"):
        load_state_dict_validated(target, source.state_dict(), context="cross_attention wrong n_layers")


def test_compatible_sync_head_checkpoint_loads_cleanly() -> None:
    cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    source = SyncHead.from_config(cfg, input_dim=256)
    target = SyncHead.from_config(cfg, input_dim=256)
    load_state_dict_validated(target, source.state_dict(), context="sync_head")


def test_wrong_sync_head_input_dim_raises() -> None:
    cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    source = SyncHead.from_config(cfg, input_dim=256)
    target = SyncHead.from_config(cfg, input_dim=64)
    with pytest.raises(RuntimeError):
        load_state_dict_validated(target, source.state_dict(), context="sync_head wrong input_dim")


# --------------------------------------------------------------------------- integration: SyncGuardPredictor end-to-end


def test_predictor_raises_compatibility_error_for_mismatched_sync_config(
    tmp_path: Path,
) -> None:
    """End-to-end: build a real sync_model checkpoint with n_layers=2, but point
    the predictor's sync_config_path at n_layers=1 - this exact mismatch used to
    be silently swallowed by `strict=False` (leaving cross_attention's layer-1
    weights randomly initialized); it must now raise
    CheckpointCompatibilityError instead."""
    from src.config import AudioConfig, ModelConfig, VideoDataConfig
    from src.inference.predictor import SyncGuardPredictor
    from src.models.audio.encoder import AudioEncoder, export_audio_encoder
    from src.models.video.visual_encoder import VisualEncoder, export_visual_encoder

    model_cfg = ModelConfig(
        audio_embedding_dim=256, visual_embedding_dim=256, num_heads=4, dropout=0.1,
        audio_cnn_channels=(32, 64, 128), audio_cnn_dropout=0.1,
        spoof_head_hidden=128, spoof_head_pooling="attentive",
        audio_encoder="cnn_transformer", audio_tf_layers=1, audio_tf_ff_dim=512, audio_tf_dropout=0.1,
        visual_encoder="transformer", visual_regions="face_mouth", landmark_coords=3,
        visual_embed_hidden=256, visual_tf_layers=1, visual_tf_ff_dim=512, visual_tf_dropout=0.1,
    )
    audio_cfg = AudioConfig(sample_rate=16000, mono=True, normalize="peak")
    video_cfg = VideoDataConfig(num_frames=32, regions="face_mouth")

    audio_encoder_path = tmp_path / "audio_encoder.pt"
    export_audio_encoder(
        audio_encoder_path, encoder=AudioEncoder(model_cfg, n_mels=80),
        model_cfg=model_cfg, audio_cfg=audio_cfg, n_mels=80,
    )
    visual_encoder_path = tmp_path / "visual_encoder.pt"
    export_visual_encoder(
        visual_encoder_path, encoder=VisualEncoder(model_cfg), model_cfg=model_cfg, video_cfg=video_cfg,
    )

    # Checkpoint trained with n_layers=2
    trained_ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=2, ff_dim=512, dropout=0.1)
    trained_ca = BidirectionalCrossAttention.from_config(trained_ca_cfg)
    trained_sh = SyncHead.from_config(SyncHeadConfig(hidden_dim=128, dropout=0.1), input_dim=256)
    sync_model_path = tmp_path / "sync_model.pt"
    torch.save(
        {"model": {
            **{f"cross_attention.{k}": v for k, v in trained_ca.state_dict().items()},
            **{f"sync_head.{k}": v for k, v in trained_sh.state_dict().items()},
        }},
        sync_model_path,
    )

    # Predictor's sync_config_path declares n_layers=1 - a wrong configuration
    sync_config_path = tmp_path / "av_align.yaml"
    sync_config_path.write_text(
        """
av_align:
  cross_attention:
    dim: 256
    num_heads: 4
    n_layers: 1
    ff_dim: 512
    dropout: 0.1
    fusion: concat_proj
    add_positional_encoding: false
  sync_head:
    hidden_dim: 128
    dropout: 0.1
    aggregation: mean
""",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointCompatibilityError, match="unexpected_keys"):
        SyncGuardPredictor(
            audio_encoder_path=audio_encoder_path,
            visual_encoder_path=visual_encoder_path,
            sync_model_path=sync_model_path,
            sync_config_path=sync_config_path,
            device="cpu",
        )
