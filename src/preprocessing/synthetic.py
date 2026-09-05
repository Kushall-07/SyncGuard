"""Deterministic synthetic audio generators for dataset-independent development.

Phase 2A exercises the whole audio preprocessing pipeline without any downloaded
dataset. These generators produce simple, reproducible waveforms with known
spectral content so the mel-spectrogram output can be checked by eye:

* ``sine``       - single tone -> one horizontal band
* ``chirp``      - linear frequency sweep -> a rising diagonal ridge
* ``white_noise``- broadband energy -> uniform fill
* ``multi_tone`` - several tones -> stacked horizontal bands
* ``silence``    - zeros -> flat noise floor

Every generator returns ``(samples, sample_rate)`` where ``samples`` is a 1-D
``float32`` numpy array in roughly [-1, 1]. Passing the same ``seed`` yields a
bit-identical array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

__all__ = [
    "sine",
    "chirp",
    "white_noise",
    "multi_tone",
    "silence",
    "write_wav",
    "SYNTHETIC_SIGNALS",
]

_DEFAULT_SR = 16000
_DEFAULT_DURATION_S = 1.0


def _time_axis(duration_s: float, sample_rate: int) -> np.ndarray:
    n_samples = int(round(duration_s * sample_rate))
    if n_samples <= 0:
        raise ValueError("duration_s * sample_rate must be >= 1 sample")
    # endpoint=False keeps the sample count exact and the signal loop-friendly.
    return np.arange(n_samples, dtype=np.float64) / sample_rate


def sine(
    freq: float = 220.0,
    *,
    duration_s: float = _DEFAULT_DURATION_S,
    sample_rate: int = _DEFAULT_SR,
    amplitude: float = 0.9,
    seed: int = 0,  # unused; kept for a uniform generator signature
) -> tuple[np.ndarray, int]:
    """A single sine tone at ``freq`` Hz."""

    del seed
    t = _time_axis(duration_s, sample_rate)
    wave = amplitude * np.sin(2.0 * np.pi * freq * t)
    return wave.astype(np.float32), sample_rate


def chirp(
    f0: float = 80.0,
    f1: float = 4000.0,
    *,
    duration_s: float = _DEFAULT_DURATION_S,
    sample_rate: int = _DEFAULT_SR,
    amplitude: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """A linear frequency sweep from ``f0`` to ``f1`` Hz."""

    del seed
    t = _time_axis(duration_s, sample_rate)
    total = t[-1] if t[-1] > 0 else 1.0
    inst_phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / total * t**2)
    wave = amplitude * np.sin(inst_phase)
    return wave.astype(np.float32), sample_rate


def white_noise(
    *,
    duration_s: float = _DEFAULT_DURATION_S,
    sample_rate: int = _DEFAULT_SR,
    amplitude: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Zero-mean Gaussian white noise, reproducible for a given ``seed``."""

    n_samples = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    wave = amplitude * rng.standard_normal(n_samples)
    return wave.astype(np.float32), sample_rate


def multi_tone(
    freqs: Sequence[float] = (220.0, 660.0, 1760.0),
    *,
    duration_s: float = _DEFAULT_DURATION_S,
    sample_rate: int = _DEFAULT_SR,
    amplitude: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """A sum of equal-amplitude sine tones, peak-scaled to ``amplitude``."""

    del seed
    if not freqs:
        raise ValueError("freqs must contain at least one frequency")
    t = _time_axis(duration_s, sample_rate)
    wave = np.sum([np.sin(2.0 * np.pi * f * t) for f in freqs], axis=0)
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * amplitude
    return wave.astype(np.float32), sample_rate


def silence(
    *,
    duration_s: float = _DEFAULT_DURATION_S,
    sample_rate: int = _DEFAULT_SR,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """All-zero signal - used to check the pipeline handles silent input safely."""

    del seed
    n_samples = int(round(duration_s * sample_rate))
    return np.zeros(n_samples, dtype=np.float32), sample_rate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write ``samples`` to a 16-bit PCM WAV file, creating parent dirs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_16")
    return path


# Named catalogue used by the smoke test and the unit tests. Each entry is a
# zero-argument callable returning ``(samples, sample_rate)`` at the defaults above.
SYNTHETIC_SIGNALS: dict[str, "callable[[], tuple[np.ndarray, int]]"] = {
    "sine_220hz": lambda: sine(220.0),
    "chirp_80_4000hz": lambda: chirp(80.0, 4000.0),
    "white_noise": lambda: white_noise(seed=1234),
    "multi_tone": lambda: multi_tone((220.0, 660.0, 1760.0)),
    "silence": lambda: silence(),
}
