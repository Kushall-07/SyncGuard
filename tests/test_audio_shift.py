"""Tests for src/data/audio_shift.py (physical waveform-domain temporal shift).

Covers the mandatory regression list from the sync-detection fix: zero shift,
+/-0.5s shift, duration preservation, sample-rate preservation (by construction -
the function never resamples), and no source mutation.
"""

from __future__ import annotations

import torch

from src.data.audio_shift import shift_waveform

SR = 16000


def _tone(n_samples: int) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / SR
    return torch.sin(2 * torch.pi * 220.0 * t).unsqueeze(0)  # [1, N]


def test_zero_shift_equals_original() -> None:
    wav = _tone(SR * 2)
    out = shift_waveform(wav, SR, 0.0)
    assert torch.equal(out, wav)
    assert out.data_ptr() != wav.data_ptr()  # a copy, not the same storage


def test_positive_half_second_shift_delays_by_exactly_half_second() -> None:
    wav = _tone(SR * 2)
    out = shift_waveform(wav, SR, 0.5)
    expected_silence = int(round(0.5 * SR))
    assert torch.all(out[..., :expected_silence] == 0.0)
    assert out[..., expected_silence] != 0.0 or torch.any(out[..., expected_silence:] != 0.0)
    # the shifted body matches the original's first (N - shift) samples
    n = wav.shape[-1]
    assert torch.equal(out[..., expected_silence:], wav[..., : n - expected_silence])


def test_negative_half_second_shift_advances_by_exactly_half_second() -> None:
    wav = _tone(SR * 2)
    out = shift_waveform(wav, SR, -0.5)
    expected_silence = int(round(0.5 * SR))
    n = wav.shape[-1]
    assert torch.all(out[..., n - expected_silence :] == 0.0)
    assert torch.equal(out[..., : n - expected_silence], wav[..., expected_silence:])


def test_duration_is_preserved_for_all_magnitudes() -> None:
    wav = _tone(SR * 3)
    for shift in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        out = shift_waveform(wav, SR, shift)
        assert out.shape == wav.shape


def test_sample_rate_is_unchanged_in_contract() -> None:
    # shift_waveform never resamples: verify by checking a fixed-frequency tone's
    # unshifted region retains identical sample values (i.e. no interpolation/resampling
    # artifact was introduced).
    wav = _tone(SR)
    out = shift_waveform(wav, SR, 0.25)
    k = int(round(0.25 * SR))
    assert torch.equal(out[..., k:], wav[..., : SR - k])


def test_source_waveform_is_not_mutated() -> None:
    wav = _tone(SR)
    original = wav.clone()
    for shift in (-0.5, 0.0, 0.5):
        shift_waveform(wav, SR, shift)
    assert torch.equal(wav, original)


def test_deterministic_repeated_calls() -> None:
    wav = _tone(SR)
    a = shift_waveform(wav, SR, 0.3)
    b = shift_waveform(wav, SR, 0.3)
    assert torch.equal(a, b)


def test_shift_larger_than_clip_yields_all_silence() -> None:
    wav = _tone(SR // 4)  # 0.25s clip
    out = shift_waveform(wav, SR, 10.0)
    assert torch.all(out == 0.0)
    assert out.shape == wav.shape


def test_accepts_1d_waveform() -> None:
    wav = _tone(SR).squeeze(0)
    out = shift_waveform(wav, SR, 0.25)
    assert out.ndim == 1
    assert out.shape == wav.shape
