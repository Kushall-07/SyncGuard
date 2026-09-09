"""Phase 9: AVEncoder = frozen Phase-5 audio encoder + frozen Phase-8 visual
encoder + deterministic time-aware alignment. Tiny encoders are built and
exported to a tmp dir (no dependency on outputs/)."""

from __future__ import annotations

import torch

from src.config import AudioConfig, MelConfig, ModelConfig, VideoDataConfig
from src.models.audio.encoder import AudioEncoder, export_audio_encoder
from src.models.fusion.av_encoder import AVEncoder, AVEncoderOutput
from src.models.video.visual_encoder import VisualEncoder, export_visual_encoder
from src.preprocessing.landmarks import region_point_count

N_MELS = 80
T_MEL = 1 + 16000 // 160          # 101  -> T_a = ceil(101/8) = 13
T_V = 32
N_PTS = region_point_count("face_mouth")   # 142


def _model_cfg(audio_variant: str = "cnn_transformer") -> ModelConfig:
    return ModelConfig(
        audio_embedding_dim=256, visual_embedding_dim=256, num_heads=4, dropout=0.0,
        audio_cnn_channels=(8, 16, 32), audio_cnn_dropout=0.0,
        audio_encoder=audio_variant, audio_tf_layers=1, audio_tf_ff_dim=64, audio_tf_dropout=0.0,
        visual_encoder="transformer", visual_regions="face_mouth", landmark_coords=3,
        visual_embed_hidden=64, visual_tf_layers=1, visual_tf_ff_dim=64, visual_tf_dropout=0.0,
        spoof_head_hidden=16,
    )


def _export_pair(tmp_path, *, audio_variant="cnn_transformer", hop_length=160):
    mc = _model_cfg(audio_variant)
    a_enc = AudioEncoder(mc, n_mels=N_MELS)
    a_cfg = AudioConfig(sample_rate=16000, mel=MelConfig(hop_length=hop_length))
    a_pt = tmp_path / "audio_encoder.pt"
    export_audio_encoder(a_pt, encoder=a_enc, model_cfg=mc, audio_cfg=a_cfg, n_mels=N_MELS)

    v_enc = VisualEncoder(mc)
    v_pt = tmp_path / "visual_encoder.pt"
    export_visual_encoder(v_pt, encoder=v_enc, model_cfg=mc, video_cfg=VideoDataConfig())
    return a_pt, v_pt


def _inputs(b=2, fps=(30.0, 29.97)):
    mel = torch.randn(b, N_MELS, T_MEL)
    lm = torch.randn(b, T_V, N_PTS, 3)
    return mel, lm, torch.tensor(fps[:b], dtype=torch.float32)


# ------------------------------------------------------------------------ tests

def test_output_shapes_and_dims(tmp_path):
    av = AVEncoder(*_export_pair(tmp_path))
    mel, lm, fps = _inputs()
    out = av(mel, lm, fps)

    assert isinstance(out, AVEncoderOutput)
    assert out.audio_tokens.shape == (2, 13, 256)          # T_a = ceil(101/8)
    assert out.visual_tokens.shape == (2, T_V, 256)
    assert out.audio_aligned.shape == out.visual_tokens.shape   # aligned 1:1 with visual
    assert out.bucket_counts.shape == (2, T_V)
    assert out.bucket_counts.dtype == torch.int64
    assert out.video_fps.shape == (2,)
    assert av.audio_dim == 256 and av.visual_dim == 256
    assert not torch.isnan(out.audio_aligned).any()


def test_scalar_fps_is_broadcast(tmp_path):
    av = AVEncoder(*_export_pair(tmp_path))
    mel, lm, _ = _inputs(b=3)
    out = av(mel, lm, 30.0)
    assert out.video_fps.shape == (3,)
    assert torch.allclose(out.video_fps, torch.full((3,), 30.0))


def test_encoders_are_frozen_and_eval(tmp_path):
    av = AVEncoder(*_export_pair(tmp_path))
    for enc in (av.audio_encoder, av.visual_encoder):
        assert all(not p.requires_grad for p in enc.parameters())
        assert not enc.training
    # stay in eval even when the parent is switched to train()
    av.train()
    assert not av.audio_encoder.training and not av.visual_encoder.training
    assert list(av.aligner.parameters()) == []             # aligner has no params


def test_audio_token_seconds_derived_from_payload_not_hardcoded(tmp_path):
    av_160 = AVEncoder(*_export_pair(tmp_path / "a", hop_length=160))
    assert av_160.audio_token_seconds == 160 * 8 / 16000    # == 0.08

    av_320 = AVEncoder(*_export_pair(tmp_path / "b", hop_length=320))
    assert av_320.audio_token_seconds == 320 * 8 / 16000    # == 0.16  (tracks the checkpoint)
    assert av_320.aligner.audio_token_seconds == av_320.audio_token_seconds


def test_rejects_non_cnn_transformer_audio_encoder(tmp_path):
    a_pt, v_pt = _export_pair(tmp_path, audio_variant="cnn")
    import pytest
    with pytest.raises(ValueError, match="cnn_transformer"):
        AVEncoder(a_pt, v_pt)


def test_deterministic_output(tmp_path):
    av = AVEncoder(*_export_pair(tmp_path))
    mel, lm, fps = _inputs()
    a = av(mel, lm, fps)
    b = av(mel, lm, fps)
    assert torch.equal(a.audio_tokens, b.audio_tokens)
    assert torch.equal(a.visual_tokens, b.visual_tokens)
    assert torch.equal(a.audio_aligned, b.audio_aligned)
    assert torch.equal(a.bucket_counts, b.bucket_counts)


def test_per_item_fps_changes_alignment(tmp_path):
    av = AVEncoder(*_export_pair(tmp_path))
    mel, lm, _ = _inputs(b=2)
    # same audio+video for both items, different fps -> different aligned audio
    mel2 = mel[:1].expand(2, -1, -1).contiguous()
    lm2 = lm[:1].expand(2, -1, -1, -1).contiguous()
    out = av(mel2, lm2, torch.tensor([30.0, 24.0]))
    assert not torch.allclose(out.audio_aligned[0], out.audio_aligned[1])
    assert torch.allclose(out.visual_tokens[0], out.visual_tokens[1], atol=1e-5)  # visual unaffected
    assert torch.allclose(out.audio_tokens[0], out.audio_tokens[1], atol=1e-5)    # audio tokens unaffected
