"""Regression tests for the audio-token timing bug (Phase 1 of the production
AV-inference correctness audit).

Root cause: `export_audio_encoder` (src/models/audio/encoder.py) never stores a
`time_downsample` scalar in its payload. Several call sites nonetheless read
`audio_payload.get("time_downsample", 1)`, which therefore ALWAYS silently
returns the default `1` - never the encoder's real value
(`AudioCNNEncoder.time_downsample = 2 ** len(audio_cnn_channels)`, e.g. `8` for
the production `[32, 64, 128]` encoder). `src/inference/predictor.py`'s
Transformer-checkpoint branch hardcoded the same wrong `1` directly.

These tests pin `audio_token_seconds_from_encoder` (the single correct
derivation, used by production inference, `scripts/train_sync_physical.py`, and
`scripts/evaluate_physical_sync.py`) against the encoder's actual architecture,
and prove runtime (`SyncGuardPredictor`) timing agrees with it end to end. Any
future change that reintroduces a hardcoded `time_downsample` (`1` or any other
constant not derived from the loaded encoder) must fail these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.config import AudioConfig, ModelConfig, VideoDataConfig
from src.inference.predictor import SyncGuardPredictor
from src.models.audio.encoder import (
    AudioEncoder,
    audio_token_seconds_from_encoder,
    export_audio_encoder,
)
from src.models.audio.spoof_classifier import SpoofClassifier
from src.models.fusion.cross_attention import BidirectionalCrossAttention, CrossAttentionConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.video.visual_encoder import VisualEncoder, export_visual_encoder

_AUDIO_CFG_DICT = {"mel": {"hop_length": 160}, "sample_rate": 16000}

# The production audio encoder's actual training config
# (outputs/runs/spoof-transformer-20260906-123646/config.yaml): 3 CNN blocks ->
# time_downsample = 2**3 = 8 -> 160 * 8 / 16000 = 0.08s/token.
_PRODUCTION_CHANNELS = (32, 64, 128)
_PRODUCTION_EXPECTED_SECONDS = 0.08


# --------------------------------------------------------------------------- unit: the shared derivation


@pytest.mark.parametrize(
    "channels,expected_downsample,expected_seconds",
    [
        ((32, 64, 128), 8, 0.08),       # production audio_cnn_channels
        ((32, 64), 4, 0.04),
        ((32,), 2, 0.02),
        ((32, 64, 128, 256), 16, 0.16),
    ],
)
def test_audio_token_seconds_from_encoder_matches_cnn_architecture(
    channels: tuple[int, ...], expected_downsample: int, expected_seconds: float
) -> None:
    """`time_downsample` must come from `2 ** len(channels)` (the CNN's real
    depth), not a constant - this is the exact quantity the old
    `payload.get("time_downsample", 1)` pattern never actually read."""
    model_cfg = ModelConfig(audio_cnn_channels=channels, audio_encoder="cnn")
    encoder = AudioEncoder(model_cfg, n_mels=80)

    assert encoder.time_downsample == expected_downsample
    seconds = audio_token_seconds_from_encoder(encoder, _AUDIO_CFG_DICT)
    assert seconds == pytest.approx(expected_seconds)

    # A hardcoded time_downsample=1 (the historical bug) would always give
    # 160/16000 = 0.01s regardless of architecture - guard that we actually
    # diverge from that whenever the real downsample isn't 1.
    hardcoded_bug_value = 160 * 1 / 16000
    if expected_downsample != 1:
        assert seconds != pytest.approx(hardcoded_bug_value)


def test_audio_token_seconds_from_encoder_uses_cnn_transformer_variant_too() -> None:
    """The Transformer variant wraps the same CNN front-end and must report the
    identical time_downsample - this is the variant SyncGuardPredictor's
    `spoof_head_checkpoint` branch actually loads in production."""
    model_cfg = ModelConfig(
        audio_cnn_channels=_PRODUCTION_CHANNELS,
        audio_encoder="cnn_transformer",
        audio_tf_layers=3,
    )
    encoder = AudioEncoder(model_cfg, n_mels=80)
    assert encoder.time_downsample == 8
    seconds = audio_token_seconds_from_encoder(encoder, _AUDIO_CFG_DICT)
    assert seconds == pytest.approx(_PRODUCTION_EXPECTED_SECONDS)


def test_export_audio_encoder_payload_never_carries_time_downsample() -> None:
    """Documents exactly why `payload.get("time_downsample", 1)` is dead code:
    the key is never written by `export_audio_encoder`. If this test ever
    starts failing because the key IS present, `audio_token_seconds_from_encoder`
    (and every one of its callers) must still be reading `encoder.time_downsample`
    directly rather than trusting the payload, since the payload is a static
    snapshot of config, not the authoritative architecture source."""
    model_cfg = ModelConfig(audio_cnn_channels=_PRODUCTION_CHANNELS, audio_encoder="cnn")
    encoder = AudioEncoder(model_cfg, n_mels=80)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "enc.pt"
        export_audio_encoder(
            path,
            encoder=encoder,
            model_cfg=model_cfg,
            audio_cfg=AudioConfig(),
            n_mels=80,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "time_downsample" not in payload


# --------------------------------------------------------------------------- integration: runtime == exported encoder timing


@pytest.fixture()
def production_shaped_checkpoints(tmp_path: Path) -> dict[str, Path]:
    """Build encoder/sync-model checkpoints with the PRODUCTION cnn_transformer
    architecture (audio_cnn_channels=[32, 64, 128]) so `time_downsample` is 8,
    not the trivial 1 that would mask this bug."""
    model_cfg = ModelConfig(
        audio_embedding_dim=256,
        visual_embedding_dim=256,
        num_heads=4,
        dropout=0.1,
        audio_cnn_channels=_PRODUCTION_CHANNELS,
        audio_cnn_dropout=0.1,
        spoof_head_hidden=128,
        spoof_head_pooling="attentive",
        audio_encoder="cnn_transformer",
        audio_tf_layers=1,
        audio_tf_ff_dim=512,
        audio_tf_dropout=0.1,
        visual_encoder="transformer",
        visual_regions="face_mouth",
        landmark_coords=3,
        visual_embed_hidden=256,
        visual_tf_layers=1,
        visual_tf_ff_dim=512,
        visual_tf_dropout=0.1,
    )
    audio_cfg = AudioConfig(sample_rate=16000, mono=True, normalize="peak")
    video_cfg = VideoDataConfig(num_frames=32, regions="face_mouth")

    audio_encoder_path = tmp_path / "audio_encoder.pt"
    export_audio_encoder(
        audio_encoder_path,
        encoder=AudioEncoder(model_cfg, n_mels=80),
        model_cfg=model_cfg,
        audio_cfg=audio_cfg,
        n_mels=80,
    )

    visual_encoder_path = tmp_path / "visual_encoder.pt"
    export_visual_encoder(
        visual_encoder_path,
        encoder=VisualEncoder(model_cfg),
        model_cfg=model_cfg,
        video_cfg=video_cfg,
    )

    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=512, dropout=0.1)
    sync_head = SyncHead.from_config(SyncHeadConfig(hidden_dim=128, dropout=0.1), input_dim=256)
    sync_model_path = tmp_path / "sync_model.pt"
    torch.save(
        {
            "model": {
                **{f"cross_attention.{k}": v for k, v in BidirectionalCrossAttention.from_config(ca_cfg).state_dict().items()},
                **{f"sync_head.{k}": v for k, v in sync_head.state_dict().items()},
            }
        },
        sync_model_path,
    )

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

    # Full "spoof-transformer" style checkpoint + sibling config.yaml, exactly the
    # layout SyncGuardPredictor's spoof_head_checkpoint branch expects
    # (Path(spoof_head_checkpoint).parent.parent / "config.yaml") - this is the
    # ONLY branch production (backend/main.py) actually uses.
    run_dir = tmp_path / "spoof-transformer-run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    spoof_head_path = ckpt_dir / "best.pt"

    full_model = SpoofClassifier(model_cfg, n_mels=80)
    torch.save({"model": full_model.state_dict()}, spoof_head_path)

    run_config = {
        "experiment": {"name": "test-spoof-transformer", "seed": 1337},
        "audio": {
            "sample_rate": 16000, "mono": True, "normalize": "peak",
            "mel": {"n_fft": 400, "hop_length": 160, "win_length": 400, "n_mels": 80,
                     "f_min": 0, "f_max": 8000, "power": 2.0, "center": True, "log": True, "log_top_db": 80.0},
        },
        "model": {
            "audio_embedding_dim": 256, "visual_embedding_dim": 256, "num_heads": 4, "dropout": 0.1,
            "audio_cnn_channels": list(_PRODUCTION_CHANNELS), "audio_cnn_dropout": 0.1,
            "spoof_head_hidden": 128, "spoof_head_pooling": "attentive",
            "audio_encoder": "cnn_transformer", "audio_tf_layers": 1, "audio_tf_ff_dim": 512, "audio_tf_dropout": 0.1,
        },
    }
    (run_dir / "config.yaml").write_text(yaml.dump(run_config), encoding="utf-8")

    return {
        "audio_encoder_path": audio_encoder_path,
        "visual_encoder_path": visual_encoder_path,
        "sync_model_path": sync_model_path,
        "sync_config_path": sync_config_path,
        "spoof_head_path": spoof_head_path,
    }


def test_predictor_production_path_derives_true_time_downsample(
    production_shaped_checkpoints: dict[str, Path],
) -> None:
    """The EXACT branch backend/main.py uses (spoof_head_checkpoint, no
    audio_encoder_path): this used to hardcode `time_downsample = 1`, giving
    0.01s/token instead of the true 0.08s/token for this 3-block CNN. This is
    the regression the whole Phase 1 fix targets."""
    c = production_shaped_checkpoints
    predictor = SyncGuardPredictor(
        visual_encoder_path=c["visual_encoder_path"],
        sync_model_path=c["sync_model_path"],
        sync_config_path=c["sync_config_path"],
        spoof_head_checkpoint=c["spoof_head_path"],
        device="cpu",
    )

    assert predictor.audio_encoder.time_downsample == 8
    assert predictor.audio_token_seconds == pytest.approx(_PRODUCTION_EXPECTED_SECONDS)
    assert predictor.audio_token_seconds != pytest.approx(0.01)


def test_predictor_legacy_path_derives_true_time_downsample(
    production_shaped_checkpoints: dict[str, Path],
) -> None:
    """The legacy `audio_encoder_path` branch must derive the identical value as
    the production branch above - both read `self.audio_encoder.time_downsample`,
    never a payload scalar."""
    c = production_shaped_checkpoints
    predictor = SyncGuardPredictor(
        audio_encoder_path=c["audio_encoder_path"],
        visual_encoder_path=c["visual_encoder_path"],
        sync_model_path=c["sync_model_path"],
        sync_config_path=c["sync_config_path"],
        device="cpu",
    )

    assert predictor.audio_encoder.time_downsample == 8
    assert predictor.audio_token_seconds == pytest.approx(_PRODUCTION_EXPECTED_SECONDS)


def test_runtime_timing_matches_exported_encoder_timing_matches_training_derivation(
    production_shaped_checkpoints: dict[str, Path],
) -> None:
    """End-to-end parity check: (1) runtime/production (`SyncGuardPredictor`),
    (2) the exported checkpoint's own architecture
    (`audio_token_seconds_from_encoder` on a freshly `load_audio_encoder`-loaded
    copy, exactly as `scripts/train_sync_physical.py` and
    `scripts/evaluate_physical_sync.py` compute it), and (3) the raw
    `2 ** len(channels) * hop_length / sample_rate` formula must all agree."""
    from src.models.audio.encoder import load_audio_encoder

    c = production_shaped_checkpoints

    predictor = SyncGuardPredictor(
        visual_encoder_path=c["visual_encoder_path"],
        sync_model_path=c["sync_model_path"],
        sync_config_path=c["sync_config_path"],
        spoof_head_checkpoint=c["spoof_head_path"],
        device="cpu",
    )
    runtime_seconds = predictor.audio_token_seconds

    # (2) training/evaluation-style: load_audio_encoder + audio_token_seconds_from_encoder,
    # the exact calls scripts/train_sync_physical.py and scripts/evaluate_physical_sync.py make.
    training_encoder, training_payload = load_audio_encoder(c["audio_encoder_path"], map_location="cpu")
    training_seconds = audio_token_seconds_from_encoder(training_encoder, training_payload["audio_cfg"])

    # (3) raw formula, independent of any helper function
    raw_seconds = 160 * (2 ** len(_PRODUCTION_CHANNELS)) / 16000

    assert runtime_seconds == pytest.approx(training_seconds)
    assert runtime_seconds == pytest.approx(raw_seconds)
    assert training_seconds == pytest.approx(_PRODUCTION_EXPECTED_SECONDS)
