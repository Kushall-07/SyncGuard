"""Phase 5 unit tests: positional encoding, audio Transformer, shared encoder, export."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.config import ModelConfig, load_config
from src.models.audio.cnn import AudioCNNEncoder
from src.models.audio.encoder import AudioEncoder, export_audio_encoder, load_audio_encoder
from src.models.audio.spoof_classifier import SpoofClassifier
from src.models.audio.transformer import AudioTransformerEncoder, SinusoidalPositionalEncoding

CNN_CFG = ModelConfig(audio_cnn_channels=(8, 16), audio_embedding_dim=32, num_heads=4,
                      spoof_head_hidden=16, audio_encoder="cnn")
TF_CFG = replace(CNN_CFG, audio_encoder="cnn_transformer", audio_tf_layers=2, audio_tf_ff_dim=64)
N_MELS = 80
AUDIO_CFG = load_config("configs/spoof_transformer.yaml").audio


# ------------------------------------------------------------------------ config


def test_spoof_transformer_config_loads() -> None:
    cfg = load_config("configs/spoof_transformer.yaml")
    assert cfg.model.audio_encoder == "cnn_transformer"
    assert cfg.model.audio_tf_layers == 3
    assert cfg.audio.mel.n_mels == 80


def test_model_config_rejects_unknown_encoder() -> None:
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"audio_encoder": "rnn"})


def test_model_config_rejects_negative_tf_layers() -> None:
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"audio_tf_layers": -1})


# ------------------------------------------------------------- positional encoding


def test_positional_encoding_is_additive_and_shape_preserving() -> None:
    pe = SinusoidalPositionalEncoding(16, max_len=64, dropout=0.0)
    x = torch.zeros(2, 10, 16)
    out = pe(x)
    assert out.shape == (2, 10, 16)
    assert torch.allclose(out, pe.pe[:, :10].expand(2, -1, -1))


def test_positional_encoding_deterministic_in_eval() -> None:
    pe = SinusoidalPositionalEncoding(16, dropout=0.5).eval()
    x = torch.randn(1, 8, 16)
    assert torch.equal(pe(x), pe(x))


def test_positional_encoding_rejects_too_long_sequence() -> None:
    pe = SinusoidalPositionalEncoding(16, max_len=12)
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 20, 16))


# --------------------------------------------------------------- transformer stack


def test_transformer_preserves_shape() -> None:
    tf = AudioTransformerEncoder(TF_CFG)
    x = torch.randn(3, 15, TF_CFG.audio_embedding_dim)
    assert tf(x).shape == x.shape


def test_transformer_rejects_non_3d() -> None:
    tf = AudioTransformerEncoder(TF_CFG)
    with pytest.raises(ValueError):
        tf(torch.randn(4, 32))


def test_transformer_key_padding_mask_runs() -> None:
    tf = AudioTransformerEncoder(TF_CFG).eval()
    x = torch.randn(2, 6, TF_CFG.audio_embedding_dim)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 4:] = True
    assert tf(x, key_padding_mask=mask).shape == x.shape


# ------------------------------------------------------------------ shared encoder


def test_audio_encoder_variant_wiring() -> None:
    cnn = AudioEncoder(CNN_CFG, N_MELS)
    tf = AudioEncoder(TF_CFG, N_MELS)
    assert cnn.transformer is None
    assert isinstance(tf.transformer, AudioTransformerEncoder)
    assert cnn.output_dim == tf.output_dim == CNN_CFG.audio_embedding_dim
    assert cnn.time_downsample == tf.time_downsample


def test_audio_encoder_token_shapes_match_across_variants() -> None:
    mel = torch.randn(2, N_MELS, 100)
    a = AudioEncoder(CNN_CFG, N_MELS)(mel).tokens
    b = AudioEncoder(TF_CFG, N_MELS)(mel).tokens
    assert a.shape == b.shape == (2, AudioCNNEncoder(CNN_CFG, N_MELS)(mel).tokens.shape[1],
                                  CNN_CFG.audio_embedding_dim)


def test_cnn_transformer_with_zero_layers_falls_back_to_cnn() -> None:
    cfg0 = replace(TF_CFG, audio_tf_layers=0)
    enc = AudioEncoder(cfg0, N_MELS)
    assert enc.transformer is None


# --------------------------------------------------------------------- export / io


def test_export_and_reload_encoder_round_trip(tmp_path) -> None:
    enc = AudioEncoder(TF_CFG, N_MELS).eval()
    path = export_audio_encoder(tmp_path / "enc.pt", encoder=enc, model_cfg=TF_CFG,
                                audio_cfg=AUDIO_CFG, n_mels=N_MELS, extra={"note": "test"})
    reloaded, payload = load_audio_encoder(path)
    reloaded.eval()

    mel = torch.randn(2, N_MELS, 96)
    with torch.no_grad():
        assert torch.equal(enc(mel).tokens, reloaded(mel).tokens)
    assert payload["variant"] == "cnn_transformer"
    assert payload["n_mels"] == N_MELS
    assert payload["extra"]["note"] == "test"


def test_load_audio_encoder_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_audio_encoder(tmp_path / "nope.pt")


def test_load_audio_encoder_rejects_bad_format(tmp_path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"format": 999}, bad)
    with pytest.raises(ValueError, match="unsupported export format"):
        load_audio_encoder(bad)


# ---------------------------------------------------------------- classifier both


@pytest.mark.parametrize("cfg", [CNN_CFG, TF_CFG], ids=["cnn", "cnn_transformer"])
def test_spoof_classifier_both_variants_forward_backward(cfg: ModelConfig) -> None:
    model = SpoofClassifier(cfg, n_mels=N_MELS)
    mel = torch.randn(4, N_MELS, 110)
    logits = model(mel)
    assert logits.shape == (4, 2)
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_transformer_variant_has_more_params() -> None:
    from src.training.utils import count_parameters

    assert count_parameters(SpoofClassifier(TF_CFG, N_MELS)) > count_parameters(
        SpoofClassifier(CNN_CFG, N_MELS)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_audio_encoder_transformer_on_cuda() -> None:
    enc = AudioEncoder(TF_CFG, N_MELS).cuda()
    out = enc(torch.randn(2, N_MELS, 100, device="cuda"))
    assert out.tokens.is_cuda and out.tokens.shape[-1] == TF_CFG.audio_embedding_dim
