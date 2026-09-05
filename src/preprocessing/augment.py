"""Training-time audio augmentation (Phase 4).

Two stages, both **training only** and both no-ops at eval:

* :class:`WaveformAugment` - per-sample CPU perturbations (additive noise at a
  sampled SNR, random gain). Plugged into :class:`ManifestAudioDataset` via its
  ``waveform_transform`` hook so it runs inside DataLoader workers.
* :class:`SpecAugment` - batched frequency/time masking of a log-mel tensor
  ``[B, n_mels, T]`` (Park et al., 2019). Runs on-device inside the trainer's
  ``compute_loss`` so it costs almost nothing.

Both take a per-call/-construction ``seed`` or ``torch.Generator`` for
reproducibility.
"""

from __future__ import annotations

import random

import torch

from src.config import AugmentConfig, SpecAugmentConfig

__all__ = ["WaveformAugment", "SpecAugment"]

_EPS = 1e-8


class WaveformAugment:
    """Additive-noise + random-gain waveform augmentation for one sample ``[1, N]``."""

    def __init__(self, cfg: AugmentConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._torch_gen = torch.Generator().manual_seed(seed)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return waveform
        out = waveform

        if self._rng.random() < self.cfg.noise_prob:
            snr_db = self._rng.uniform(*self.cfg.noise_snr_db)
            signal_power = out.pow(2).mean().clamp_min(_EPS)
            noise = torch.randn(out.shape, generator=self._torch_gen, dtype=out.dtype)
            noise_power = noise.pow(2).mean().clamp_min(_EPS)
            scale = (signal_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
            out = out + scale * noise

        if self._rng.random() < self.cfg.gain_prob:
            gain_db = self._rng.uniform(*self.cfg.gain_db)
            out = out * (10 ** (gain_db / 20))

        return out.clamp(-1.0, 1.0)


class SpecAugment:
    """Frequency + time masking of a log-mel batch ``[B, n_mels, T]`` (or ``[B, 1, n_mels, T]``)."""

    def __init__(self, cfg: SpecAugmentConfig, *, mask_value: float | None = None) -> None:
        self.cfg = cfg
        self.mask_value = mask_value  # None -> per-sample mean

    def __call__(
        self,
        mel: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        squeeze = mel.dim() == 4 and mel.shape[1] == 1
        x = mel.squeeze(1) if squeeze else mel
        if x.dim() != 3:
            raise ValueError(f"expected [B, n_mels, T] (or [B, 1, n_mels, T]), got {tuple(mel.shape)}")

        b, n_mels, t = x.shape
        x = x.clone()
        fill = self.mask_value

        for i in range(b):
            sample_fill = x[i].mean() if fill is None else fill
            for width, axis_len, take_axis in (
                (self.cfg.freq_mask_width, n_mels, "f"),
                (self.cfg.time_mask_width, t, "t"),
            ):
                n_masks = self.cfg.freq_masks if take_axis == "f" else self.cfg.time_masks
                for _ in range(n_masks):
                    span = int(torch.randint(0, max(1, min(width, axis_len)) + 1,
                                             (1,), generator=generator).item())
                    if span == 0:
                        continue
                    start = int(torch.randint(0, axis_len - span + 1,
                                              (1,), generator=generator).item())
                    if take_axis == "f":
                        x[i, start : start + span, :] = sample_fill
                    else:
                        x[i, :, start : start + span] = sample_fill

        return x.unsqueeze(1) if squeeze else x
