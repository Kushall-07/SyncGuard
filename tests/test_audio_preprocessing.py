"""Phase 2A unit tests: audio preprocessing + log-mel feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import AudioConfig, load_audio_config
from src.features.mel_spectrogram import (
    MelSpectrogramExtractor,
    compute_log_mel,
    expected_num_frames,
)
from src.preprocessing.audio import normalize, preprocess_audio, resample, to_mono
from src.preprocessing.synthetic import chirp, sine, white_noise, write_wav

CONFIG_PATH = "configs/audio.yaml"


@pytest.fixture(scope="module")
def cfg() -> AudioConfig:
    return load_audio_config(CONFIG_PATH)


# --------------------------------------------------------------------------- config


def test_config_matches_spec_defaults(cfg: AudioConfig) -> None:
    assert cfg.sample_rate == 16000
    assert cfg.mono is True
    assert cfg.normalize == "peak"
    assert (cfg.mel.n_mels, cfg.mel.n_fft, cfg.mel.hop_length) == (80, 400, 160)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        AudioConfig.from_dict({"sample_rate": 16000, "bogus": 1})


def test_config_rejects_fmax_above_nyquist() -> None:
    with pytest.raises(ValueError):
        AudioConfig.from_dict({"sample_rate": 8000, "mel": {"f_max": 8000}})


# ----------------------------------------------------------------------- transforms


def test_to_mono_collapses_stereo_and_preserves_length() -> None:
    stereo = torch.randn(2, 1000)
    mono = to_mono(stereo)
    assert mono.shape == (1, 1000)
    torch.testing.assert_close(mono[0], stereo.mean(dim=0))


def test_to_mono_passthrough_when_already_mono() -> None:
    mono = torch.randn(1, 500)
    assert to_mono(mono) is mono


def test_resample_hits_target_length_for_one_second() -> None:
    one_sec_44k = torch.randn(1, 44100)
    out = resample(one_sec_44k, 44100, 16000)
    assert out.shape == (1, 16000)


def test_resample_is_noop_when_rates_equal() -> None:
    wav = torch.randn(1, 16000)
    assert resample(wav, 16000, 16000) is wav


def test_normalize_peak_scales_to_target() -> None:
    wav = torch.randn(1, 4000) * 0.03
    out = normalize(wav, "peak", 1.0)
    assert out.abs().max().item() == pytest.approx(1.0, abs=1e-5)


def test_normalize_rms_scales_to_target() -> None:
    wav = torch.randn(1, 8000) * 5.0
    out = normalize(wav, "rms", 0.1)
    rms = out.pow(2).mean().sqrt().item()
    assert rms == pytest.approx(0.1, rel=1e-4)


def test_normalize_silent_input_is_safe() -> None:
    silent = torch.zeros(1, 1000)
    out = normalize(silent, "peak", 1.0)
    assert torch.equal(out, silent)


# ---------------------------------------------------------------- preprocess_audio


def test_preprocess_from_array_matches_preprocess_from_file(tmp_path, cfg: AudioConfig) -> None:
    samples, sr = sine(240.0, sample_rate=22050)
    wav_path = write_wav(tmp_path / "tone.wav", samples, sr)

    from_file = preprocess_audio(wav_path, cfg)
    from_array = preprocess_audio(samples, cfg, source_sr=sr)

    assert from_file.shape == from_array.shape == (1, cfg.sample_rate)
    # WAV round-trips through 16-bit PCM, so allow a small quantization tolerance.
    torch.testing.assert_close(from_file, from_array, atol=2e-4, rtol=0)


def test_preprocess_accepts_callable_source(cfg: AudioConfig) -> None:
    out = preprocess_audio(lambda: chirp(100.0, 3000.0), cfg)
    assert out.shape == (1, cfg.sample_rate)
    assert out.dtype == torch.float32


def test_preprocess_raw_array_requires_source_sr(cfg: AudioConfig) -> None:
    with pytest.raises(ValueError):
        preprocess_audio(np.zeros(1000, dtype=np.float32), cfg)


def test_preprocess_output_is_peak_normalized(cfg: AudioConfig) -> None:
    out = preprocess_audio(lambda: white_noise(seed=7), cfg)
    assert out.abs().max().item() == pytest.approx(cfg.norm_target, abs=1e-4)


# --------------------------------------------------------------------- mel features


def test_expected_num_frames_one_second(cfg: AudioConfig) -> None:
    assert expected_num_frames(cfg.sample_rate, cfg.mel) == 101


def test_log_mel_shape_single_clip(cfg: AudioConfig) -> None:
    wav = preprocess_audio(lambda: sine(220.0), cfg)
    mel = compute_log_mel(wav, cfg)
    assert mel.shape == (cfg.mel.n_mels, 101)
    assert torch.isfinite(mel).all()


def test_log_mel_shape_batched(cfg: AudioConfig) -> None:
    batch = torch.stack([preprocess_audio(lambda: sine(f), cfg)[0] for f in (110, 220, 440, 880)])
    assert batch.shape == (4, cfg.sample_rate)
    mel = compute_log_mel(batch, cfg)
    assert mel.shape == (4, cfg.mel.n_mels, 101)


def test_extractor_moves_with_module_to_device(cfg: AudioConfig) -> None:
    extractor = MelSpectrogramExtractor(cfg)
    # mel filterbank buffer lives inside the MelSpectrogram submodule
    fb = extractor.mel_spectrogram.mel_scale.fb
    assert fb.device.type == "cpu"


def test_amplitude_to_db_disabled_when_log_false(cfg: AudioConfig) -> None:
    from dataclasses import replace

    linear_cfg = replace(cfg, mel=replace(cfg.mel, log=False))
    extractor = MelSpectrogramExtractor(linear_cfg)
    assert extractor.to_db is None
    mel = extractor(preprocess_audio(lambda: sine(220.0), linear_cfg))
    assert (mel >= 0).all()  # power spectrogram is non-negative


# --------------------------------------------------------------------- determinism


def test_synthetic_white_noise_is_seed_deterministic() -> None:
    a, _ = white_noise(seed=42)
    b, _ = white_noise(seed=42)
    c, _ = white_noise(seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------- cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_runs_on_cuda(cfg: AudioConfig) -> None:
    wav = preprocess_audio(lambda: chirp(80.0, 4000.0), cfg, device="cuda")
    assert wav.is_cuda
    mel = compute_log_mel(wav, cfg, device="cuda")
    assert mel.is_cuda
    assert mel.shape == (cfg.mel.n_mels, 101)
    assert torch.isfinite(mel).all()
