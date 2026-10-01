"""Physical (waveform-domain) temporal shifting for audio-visual sync training.

This is the corrected negative-construction primitive described in the SyncGuard
sync-detection fix: unlike `src.data.sync_pairs.compute_shifted_alignment` (which
re-buckets *already-encoded* audio tokens onto a shifted time grid - a purely
token-space operation that never touches the waveform or the audio encoder), this
module shifts the raw waveform itself, *before* the mel spectrogram / audio encoder
ever see it. That is the same physical operation used to build
`demo/video/desync/*` (prepend/trim silence on the real WAV file), so a model
trained on these negatives is trained on the same kind of signal production
inference actually receives when a user supplies a desynchronized external WAV.

The shift is duration-preserving (output has exactly the same number of samples
as the input) and never mutates the input tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["shift_waveform"]


def shift_waveform(waveform: Tensor, sample_rate: int, shift_seconds: float) -> Tensor:
    """Physically shift ``waveform`` in time by ``shift_seconds``, preserving duration.

    Args:
        waveform: ``[channels, num_samples]`` (or ``[num_samples]``) float tensor.
        sample_rate: Sample rate in Hz, used only to convert seconds -> samples.
        shift_seconds: Positive = audio delayed relative to video (silence is
            prepended and an equal-length tail is dropped). Negative = audio
            advanced (leading samples are dropped and silence is appended).
            Zero returns a duration- and content-identical copy.

    Returns:
        A new tensor, same shape and dtype as ``waveform``. The input tensor is
        never modified in place.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    squeeze = waveform.ndim == 1
    wav = waveform.unsqueeze(0) if squeeze else waveform
    if wav.ndim != 2:
        raise ValueError(f"waveform must be 1-D or [channels, num_samples], got {tuple(waveform.shape)}")

    n = wav.shape[-1]
    shift_samples = int(round(shift_seconds * sample_rate))

    if shift_samples == 0:
        out = wav.clone()
    elif shift_samples > 0:
        # Delay: prepend silence, drop an equal-length tail.
        k = min(shift_samples, n)
        silence = wav.new_zeros(wav.shape[0], k)
        body = wav[..., : n - k]
        out = torch.cat([silence, body], dim=-1)
    else:
        # Advance: drop leading samples, append an equal-length silence tail.
        k = min(-shift_samples, n)
        silence = wav.new_zeros(wav.shape[0], k)
        body = wav[..., k:]
        out = torch.cat([body, silence], dim=-1)

    if out.shape[-1] != n:
        raise AssertionError("shift_waveform must preserve sample count")  # pragma: no cover - invariant
    return out.squeeze(0) if squeeze else out
