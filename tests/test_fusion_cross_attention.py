"""Phase 10: bidirectional audio-visual cross-attention.

Covers shape preservation, genuinely-bidirectional behaviour, gradient flow,
trainable params, eval determinism / train stochasticity, the concat_proj /
gated / add fusion variants, n_layers 1 vs 2, head divisibility, optional
key-padding masks, config round-trip, Phase-9 compatibility, a parameter-count
guard, and that Phase 9's public API is unchanged.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.models.fusion import (
    AVEncoder,
    AVEncoderOutput,
    AudioToVideoAligner,
    BidirectionalCrossAttention,
    BidirectionalCrossAttentionOutput,
    CrossAttentionConfig,
    align_audio_to_video,
)

D = 256
CFG_PATH = "configs/av_align.yaml"


def _module(**kw) -> BidirectionalCrossAttention:
    base = dict(dim=D, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1, fusion="concat_proj")
    base.update(kw)
    return BidirectionalCrossAttention(**base)


def _pair(b=2, t=32, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(b, t, d, generator=g)
    v = torch.randn(b, t, d, generator=g)
    return a, v


def _nparams(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- shape / typing

@pytest.mark.parametrize("b,t", [(1, 8), (3, 16), (2, 32), (2, 50)])
def test_shapes_preserved(b, t):
    m = _module().eval()
    a, v = _pair(b, t)
    out = m(a, v)
    assert isinstance(out, BidirectionalCrossAttentionOutput)
    assert out.fused.shape == (b, t, D)
    assert out.audio_query_visual.shape == (b, t, D)
    assert out.visual_query_audio.shape == (b, t, D)


def test_not_a_classifier():
    m = _module()
    a, v = _pair()
    out = m(a, v)
    assert out.fused.dim() == 3 and out.fused.shape[-1] == D          # no [B, n_classes]
    for banned in ("classifier", "head", "n_classes", "sync_score", "logits"):
        assert not hasattr(m, banned)


# ------------------------------------------------------- genuinely bidirectional

def test_bidirectional_uses_both_inputs():
    m = _module().eval()
    a, v = _pair(seed=1)
    da, dv = _pair(seed=2)
    base = m(a, v)

    # perturbing visual (context for a2v, query for v2a) moves both branches + fused
    pv = m(a, v + 0.5 * dv)
    assert not torch.allclose(pv.audio_query_visual, base.audio_query_visual)
    assert not torch.allclose(pv.visual_query_audio, base.visual_query_audio)
    assert not torch.allclose(pv.fused, base.fused)

    # perturbing audio (query for a2v, context for v2a) moves both branches + fused
    pa = m(a + 0.5 * da, v)
    assert not torch.allclose(pa.audio_query_visual, base.audio_query_visual)
    assert not torch.allclose(pa.visual_query_audio, base.visual_query_audio)
    assert not torch.allclose(pa.fused, base.fused)


def test_directions_are_distinct_not_symmetric():
    m = _module().eval()
    a, v = _pair(seed=3)
    swapped = m(v, a)                       # audio<-v, visual<-a
    normal = m(a, v)
    assert not torch.allclose(normal.fused, swapped.fused)
    # a2v(a,v) is a different computation from v2a(a,v)
    assert not torch.allclose(normal.audio_query_visual, normal.visual_query_audio)


# --------------------------------------------------------------- trainability

def test_trainable_parameters_exist_and_require_grad():
    m = _module()
    params = list(m.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_gradient_flows_to_inputs_and_params():
    m = _module().train()
    a, v = _pair(seed=4)
    a.requires_grad_(True)
    v.requires_grad_(True)
    out = m(a, v)
    out.fused.pow(2).sum().backward()

    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


# --------------------------------------------------- eval determinism / dropout

def test_eval_is_deterministic():
    m = _module().eval()
    a, v = _pair(seed=5)
    assert torch.equal(m(a, v).fused, m(a, v).fused)


def test_train_dropout_is_stochastic():
    m = _module(dropout=0.2).train()
    a, v = _pair(seed=6)
    torch.manual_seed(0)
    o1 = m(a, v).fused
    o2 = m(a, v).fused                      # RNG advanced -> different dropout masks
    assert not torch.allclose(o1, o2, atol=1e-5)


# --------------------------------------------------------------- fusion variants

@pytest.mark.parametrize("fusion", ["concat_proj", "gated", "add"])
def test_fusion_variants_shape_and_grad(fusion):
    m = _module(fusion=fusion).train()
    a, v = _pair(seed=7)
    a.requires_grad_(True)
    out = m(a, v)
    assert out.fused.shape == (2, 32, D)
    out.fused.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


def test_default_fusion_is_concat_proj():
    assert _module().fusion == "concat_proj"


def test_add_fusion_has_fewer_params_than_concat_proj():
    assert _nparams(_module(fusion="add")) < _nparams(_module(fusion="concat_proj"))
    assert _nparams(_module(fusion="gated")) > _nparams(_module(fusion="add"))


def test_invalid_fusion_rejected():
    with pytest.raises(ValueError):
        _module(fusion="attention")


# --------------------------------------------------------------------- n_layers

def test_n_layers_1_and_2_build_and_run():
    a, v = _pair(seed=8)
    p1 = _nparams(_module(n_layers=1))
    p2 = _nparams(_module(n_layers=2))
    assert p2 > p1
    # the extra layer costs ~2 * (per-direction layer) params (~1.58 M for dim 256)
    assert 1.4e6 < (p2 - p1) < 1.8e6
    for n in (1, 2):
        out = _module(n_layers=n).eval()(a, v)
        assert out.fused.shape == (2, 32, D)


def test_n_layers_must_be_positive():
    with pytest.raises(ValueError):
        _module(n_layers=0)


# ---------------------------------------------------------- head divisibility

def test_head_divisibility_guard():
    with pytest.raises(ValueError):
        _module(num_heads=6)                # 256 % 6 != 0
    for h in (4, 8):
        assert _module(num_heads=h).eval()(*_pair()).fused.shape == (2, 32, D)


# --------------------------------------------------- optional key-padding masks

def test_key_padding_masks_run_and_change_output():
    m = _module().eval()
    a, v = _pair(seed=9)
    b, t = a.shape[:2]
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, -2:] = True                     # ignore the last 2 key positions (never a whole row)

    base = m(a, v)
    with_a = m(a, v, audio_key_padding_mask=mask)      # affects v2a (K/V = audio)
    with_v = m(a, v, visual_key_padding_mask=mask)     # affects a2v (K/V = visual)

    for out in (with_a, with_v):
        assert not torch.isnan(out.fused).any()
        assert out.fused.shape == (b, t, D)
    assert not torch.allclose(with_a.visual_query_audio, base.visual_query_audio)
    assert not torch.allclose(with_v.audio_query_visual, base.audio_query_visual)


# ------------------------------------------------------- positional-encoding knob

def test_positional_encoding_knob():
    assert _module().pos_encoding is None
    m = _module(add_positional_encoding=True)
    assert m.pos_encoding is not None
    out = m.eval()(*_pair(seed=10))
    assert out.fused.shape == (2, 32, D)


# ------------------------------------------------------------- config round-trip

def test_config_round_trip_from_av_align_yaml():
    cfg = CrossAttentionConfig.from_yaml(CFG_PATH)
    assert (cfg.dim, cfg.num_heads, cfg.n_layers, cfg.ff_dim) == (256, 4, 1, 1024)
    assert cfg.dropout == pytest.approx(0.1)
    assert cfg.fusion == "concat_proj"
    assert cfg.add_positional_encoding is False

    m = BidirectionalCrossAttention.from_config(cfg).eval()
    out = m(*_pair())
    assert out.fused.shape == (2, 32, D) and m.fusion == "concat_proj"


def test_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("av_align:\n  cross_attention:\n    dim: 256\n    bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown cross_attention key"):
        CrossAttentionConfig.from_yaml(p)


def test_config_validates_values():
    with pytest.raises(ValueError):
        CrossAttentionConfig(fusion="lerp")
    with pytest.raises(ValueError):
        CrossAttentionConfig(dim=256, num_heads=7)
    with pytest.raises(ValueError):
        CrossAttentionConfig(n_layers=0)


# ------------------------------------------------------------ parameter-count guard

def test_parameter_count_within_expected_band():
    total = _nparams(_module())            # dim256 / heads4 / n_layers1 / ff1024 / concat_proj
    assert 1_600_000 < total < 1_850_000, total   # proposal estimate ~1.71 M


# ------------------------------------------------------------- Phase 9 compat

def _tiny_av_encoder(tmp_path):
    from src.config import AudioConfig, MelConfig, ModelConfig, VideoDataConfig
    from src.models.audio.encoder import AudioEncoder, export_audio_encoder
    from src.models.video.visual_encoder import VisualEncoder, export_visual_encoder

    mc = ModelConfig(
        audio_embedding_dim=256, visual_embedding_dim=256, num_heads=4, dropout=0.0,
        audio_cnn_channels=(8, 16, 32), audio_cnn_dropout=0.0,
        audio_encoder="cnn_transformer", audio_tf_layers=1, audio_tf_ff_dim=64, audio_tf_dropout=0.0,
        visual_encoder="transformer", visual_regions="face_mouth", landmark_coords=3,
        visual_embed_hidden=64, visual_tf_layers=1, visual_tf_ff_dim=64, visual_tf_dropout=0.0,
        spoof_head_hidden=16,
    )
    a_pt = tmp_path / "audio_encoder.pt"
    export_audio_encoder(a_pt, encoder=AudioEncoder(mc, n_mels=80), model_cfg=mc,
                         audio_cfg=AudioConfig(sample_rate=16000, mel=MelConfig(hop_length=160)),
                         n_mels=80)
    v_pt = tmp_path / "visual_encoder.pt"
    export_visual_encoder(v_pt, encoder=VisualEncoder(mc), model_cfg=mc, video_cfg=VideoDataConfig())
    return AVEncoder(a_pt, v_pt)


def test_consumes_phase9_avencoder_output_and_trains(tmp_path):
    from src.preprocessing.landmarks import region_point_count

    av = _tiny_av_encoder(tmp_path)
    t_mel = 1 + 16000 // 160
    mel = torch.randn(2, 80, t_mel)
    lm = torch.randn(2, 32, region_point_count("face_mouth"), 3)
    enc = av(mel, lm, torch.tensor([30.0, 29.97]))

    # Phase 9 outputs arrive as non-grad tensors (encoders run under no_grad)
    assert not enc.audio_aligned.requires_grad and not enc.visual_tokens.requires_grad
    assert enc.audio_aligned.shape == enc.visual_tokens.shape == (2, 32, 256)

    m = _module().train()
    out = m(enc.audio_aligned, enc.visual_tokens)
    assert out.fused.shape == (2, 32, 256)
    out.fused.sum().backward()
    assert all(p.grad is not None for p in m.parameters())   # module trains on frozen-encoder features


# --------------------------------------------------- Phase 9 public API unchanged

def test_phase9_public_api_unchanged():
    sig = set(inspect.signature(align_audio_to_video).parameters)
    assert sig == {"audio_tokens", "n_video_tokens", "audio_token_seconds", "video_fps",
                   "window_seconds", "empty_bucket", "audio_valid_len"}
    assert set(AVEncoderOutput.__dataclass_fields__) == {
        "audio_tokens", "visual_tokens", "audio_aligned", "bucket_counts", "video_fps"}
    assert issubclass(AudioToVideoAligner, torch.nn.Module)
    av_sig = set(inspect.signature(AVEncoder.forward).parameters)
    assert {"mel_window", "landmarks", "video_fps", "window_seconds", "audio_valid_len"} <= av_sig
