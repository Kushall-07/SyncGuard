"""Source-agnostic audio preprocessing for SyncGuard (Phase 2A).

Pipeline: ``load -> mono -> resample to 16 kHz -> amplitude normalization``.

The single entry point :func:`preprocess_audio` accepts a file path, a raw
array / tensor, or a zero-argument callable. This means the synthetic generators
used now and the real ASVspoof ``.flac`` / ``.wav`` files used in Phase 3 flow
through exactly the same code without any rewrite.

All functions operate on ``torch.Tensor`` waveforms shaped ``[channels, num_samples]``
(or ``[1, num_samples]`` after mono downmix) and are deterministic and free of
side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import AudioConfig

__all__ = ["load_audio", "to_mono", "resample", "normalize", "preprocess_audio"]

# Anything preprocess_audio() knows how to turn into a waveform.
AudioSource = Union[
    str,
    Path,
    torch.Tensor,
    np.ndarray,
    Callable[[], "tuple[np.ndarray | torch.Tensor, int]"],
]

_EPS = 1e-8


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load an audio file to a float32 tensor ``[channels, num_samples]``.

    Uses :mod:`soundfile` (libsndfile), which handles ``.wav`` now and ASVspoof
    ``.flac`` in Phase 3. ``torchaudio``'s own loader is not used because
    torchaudio 2.11 delegates decoding to the optional ``torchcodec`` package.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)  # [frames, channels]
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))               # [channels, frames]
    return waveform.to(torch.float32), int(sample_rate)


def _as_2d_waveform(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Coerce a raw array/tensor into a float32 ``[channels, num_samples]`` tensor."""

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(data))
    elif isinstance(data, torch.Tensor):
        tensor = data
    else:  # pragma: no cover - guarded by preprocess_audio
        raise TypeError(f"unsupported waveform type: {type(data)!r}")

    tensor = tensor.to(torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)          # [num_samples] -> [1, num_samples]
    elif tensor.ndim == 2:
        # Accept [num_samples, channels] from soundfile-style arrays and transpose
        # to the [channels, num_samples] convention when it is unambiguous.
        if tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.t()
    else:
        raise ValueError(f"expected a 1-D or 2-D waveform, got shape {tuple(tensor.shape)}")
    return tensor.contiguous()


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Downmix ``[channels, num_samples]`` to ``[1, num_samples]`` by averaging."""

    if waveform.ndim != 2:
        raise ValueError(f"expected [channels, num_samples], got shape {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample ``waveform`` from ``orig_sr`` to ``target_sr`` (no-op if equal)."""

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)


def normalize(waveform: torch.Tensor, mode: str = "peak", target: float = 1.0) -> torch.Tensor:
    """Amplitude-normalize a waveform.

    ``peak`` scales so ``max(|x|) == target``; ``rms`` scales so the RMS equals
    ``target``; ``none`` returns the input unchanged. Silent / all-zero input is
    returned unchanged instead of dividing by zero.
    """

    if mode == "none":
        return waveform
    if target <= 0:
        raise ValueError("target must be positive")

    if mode == "peak":
        scale = waveform.abs().max()
    elif mode == "rms":
        scale = waveform.pow(2).mean().sqrt()
    else:
        raise ValueError(f"unknown normalize mode: {mode!r}")

    if scale < _EPS:
        return waveform
    return waveform * (target / scale)


def _resolve_source(
    source: AudioSource,
    source_sr: int | None,
) -> tuple[torch.Tensor, int]:
    """Turn any accepted source into a ``([channels, num_samples] tensor, sample_rate)``."""

    if callable(source) and not isinstance(source, (str, Path)):
        produced = source()
        if not (isinstance(produced, tuple) and len(produced) == 2):
            raise ValueError("callable source must return (samples, sample_rate)")
        data, sr = produced
        return _as_2d_waveform(data), int(sr)

    if isinstance(source, (str, Path)):
        return load_audio(source)

    if isinstance(source, (np.ndarray, torch.Tensor)):
        if source_sr is None:
            raise ValueError("source_sr is required when passing a raw array or tensor")
        return _as_2d_waveform(source), int(source_sr)

    raise TypeError(f"unsupported audio source type: {type(source)!r}")


def preprocess_audio(
    source: AudioSource,
    cfg: AudioConfig,
    *,
    source_sr: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Run the full preprocessing pipeline and return a waveform tensor.

    Parameters
    ----------
    source:
        A file path, a raw ``np.ndarray`` / ``torch.Tensor`` (then ``source_sr``
        must be given), or a zero-argument callable returning ``(samples, sr)``.
    cfg:
        Audio configuration (see :class:`src.config.AudioConfig`).
    source_sr:
        Sample rate of a raw array/tensor ``source``. Ignored otherwise.
    device:
        Optional target device for the returned tensor. Defaults to CPU.

    Returns
    -------
    torch.Tensor
        Float32 waveform shaped ``[1, num_samples]`` (mono when ``cfg.mono``),
        resampled to ``cfg.sample_rate`` and amplitude-normalized per ``cfg``.
    """

    waveform, sr = _resolve_source(source, source_sr)

    if cfg.mono:
        waveform = to_mono(waveform)
    waveform = resample(waveform, sr, cfg.sample_rate)
    waveform = normalize(waveform, cfg.normalize, cfg.norm_target)
    waveform = waveform.to(torch.float32).contiguous()

    if device is not None:
        waveform = waveform.to(device)
    return waveform
