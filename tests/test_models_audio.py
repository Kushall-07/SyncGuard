"""Phase 4 unit tests: config additions, augmentation, CNN encoder, spoof head."""

from __future__ import annotations

import pytest
import torch

from src.config import AugmentConfig, DataConfig, ModelConfig, SpecAugmentConfig, load_config
from src.models.audio.cnn import AudioCNNEncoder
from src.models.audio.spoof_classifier import SpoofClassifier
from src.models.heads.spoof_head import SpoofHead, temporal_pool
from src.preprocessing.augment import SpecAugment, WaveformAugment

MODEL_CFG = ModelConfig(audio_cnn_channels=(8, 16), audio_embedding_dim=32, spoof_head_hidden=16)
N_MELS = 80


# ------------------------------------------------------------------------ config


def test_spoof_cnn_config_loads_and_resolves_pointers() -> None:
    cfg = load_config("configs/spoof_cnn.yaml")
    assert cfg.audio.sample_rate == 16000          # audio: -> configs/audio.yaml
    assert cfg.data.feature == "logmel"            # data: -> configs/asvspoof.yaml
    assert cfg.model.audio_cnn_channels == (32, 64, 128)
    assert cfg.training.monitor == "val_eer"
    assert cfg.augment.specaugment.freq_masks == 2


def test_model_config_rejects_bad_pooling_and_channels() -> None:
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"spoof_head_pooling": "loud"})
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"audio_cnn_channels": [32, -4]})


def test_model_config_coerces_channel_list_to_tuple() -> None:
    cfg = ModelConfig.from_dict({"audio_cnn_channels": [16, 32]})
    assert cfg.audio_cnn_channels == (16, 32)


def test_augment_config_validates_ranges() -> None:
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"noise_snr_db": [30.0, 10.0]})
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"gain_prob": 1.5})
    ok = AugmentConfig.from_dict({"noise_snr_db": [5, 20], "specaugment": {"time_masks": 1}})
    assert ok.noise_snr_db == (5, 20) and ok.specaugment.time_masks == 1


def test_data_config_unknown_key_raises() -> None:
    with pytest.raises(ValueError):
        DataConfig.from_dict({"feature": "logmel", "bogus": 1})


# ------------------------------------------------------------------- augmentation


def test_waveform_augment_disabled_is_identity() -> None:
    cfg = AugmentConfig(enabled=False)
    wav = torch.randn(1, 1600)
    assert torch.equal(WaveformAugment(cfg)(wav), wav)


def test_waveform_augment_is_seed_deterministic_and_bounded() -> None:
    cfg = AugmentConfig(enabled=True, noise_prob=1.0, gain_prob=1.0)
    wav = torch.randn(1, 1600).clamp(-1, 1)
    a = WaveformAugment(cfg, seed=5)(wav)
    b = WaveformAugment(cfg, seed=5)(wav)
    assert torch.equal(a, b)
    assert a.abs().max() <= 1.0
    assert not torch.equal(a, wav)


def test_specaugment_shape_and_masking() -> None:
    spec = SpecAugment(SpecAugmentConfig(freq_masks=2, freq_mask_width=10,
                                         time_masks=2, time_mask_width=10), mask_value=0.0)
    x = torch.ones(4, N_MELS, 50)
    out = spec(x)
    assert out.shape == x.shape
    assert (out == 0.0).any() and (out == 1.0).any()
    # 4-D [B, 1, n_mels, T] path
    out4 = spec(torch.ones(2, 1, N_MELS, 50))
    assert out4.shape == (2, 1, N_MELS, 50)


def test_specaugment_zero_masks_is_identity() -> None:
    spec = SpecAugment(SpecAugmentConfig(freq_masks=0, time_masks=0))
    x = torch.randn(3, N_MELS, 40)
    assert torch.equal(spec(x), x)


# ---------------------------------------------------------------------- cnn encoder


@pytest.mark.parametrize("t_in", [61, 100, 401])
def test_encoder_token_shape(t_in: int) -> None:
    enc = AudioCNNEncoder(MODEL_CFG, N_MELS)
    out = enc(torch.randn(2, N_MELS, t_in))
    t_out = -(-t_in // enc.time_downsample)  # ceil, matches MaxPool2d(ceil_mode=True)
    assert out.tokens.shape == (2, t_out, MODEL_CFG.audio_embedding_dim)
    assert out.time_downsample == 2 ** len(MODEL_CFG.audio_cnn_channels)


def test_encoder_accepts_4d_input() -> None:
    enc = AudioCNNEncoder(MODEL_CFG, N_MELS)
    a = enc(torch.randn(2, N_MELS, 80)).tokens
    b = enc(torch.randn(2, 1, N_MELS, 80)).tokens
    assert a.shape == b.shape


def test_encoder_rejects_bad_shape() -> None:
    enc = AudioCNNEncoder(MODEL_CFG, N_MELS)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 3, N_MELS, 80))


# ------------------------------------------------------------------------ head


@pytest.mark.parametrize("pooling", ["attentive", "mean", "meanmax"])
def test_spoof_head_pooling_variants(pooling: str) -> None:
    head = SpoofHead(32, hidden=16, pooling=pooling)
    logits = head(torch.randn(5, 7, 32))
    assert logits.shape == (5, 2)


def test_temporal_pool_meanmax_doubles_dim() -> None:
    pooled = temporal_pool(torch.randn(4, 9, 12), "meanmax")
    assert pooled.shape == (4, 24)


def test_attentive_weights_sum_to_one() -> None:
    head = SpoofHead(8, pooling="attentive")
    tokens = torch.randn(3, 6, 8)
    w = torch.softmax(head.attn(tokens), dim=1)
    assert torch.allclose(w.sum(dim=1), torch.ones(3, 1), atol=1e-5)


# ---------------------------------------------------------------- full classifier


def test_spoof_classifier_forward_and_backward() -> None:
    model = SpoofClassifier(MODEL_CFG, n_mels=N_MELS)
    mel = torch.randn(4, N_MELS, 120, requires_grad=True)
    logits = model(mel)
    assert logits.shape == (4, 2)
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_spoof_classifier_encode_matches_encoder_dim() -> None:
    model = SpoofClassifier(MODEL_CFG, n_mels=N_MELS)
    tokens = model.encode(torch.randn(2, N_MELS, 64))
    assert tokens.shape[-1] == MODEL_CFG.audio_embedding_dim
    assert tokens.dim() == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spoof_classifier_on_cuda() -> None:
    model = SpoofClassifier(MODEL_CFG, n_mels=N_MELS).cuda()
    logits = model(torch.randn(2, N_MELS, 100, device="cuda"))
    assert logits.is_cuda and logits.shape == (2, 2)
