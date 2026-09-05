"""Log-mel spectrogram feature extraction for SyncGuard (Phase 2A).

Wraps :class:`torchaudio.transforms.MelSpectrogram` (and, optionally,
:class:`torchaudio.transforms.AmplitudeToDB`) in an ``nn.Module`` so the mel
filterbank is a registered buffer that moves with ``.to(device)`` and runs on
CUDA. The same extractor feeds the CNN / Transformer audio front-ends of later
phases.

Shape convention for :meth:`MelSpectrogramExtractor.forward`:

======================  ============================
input                   output
======================  ============================
``[num_samples]``       ``[n_mels, T]``
``[1, num_samples]``    ``[n_mels, T]``
``[C>1, num_samples]``  ``[C, n_mels, T]``
``[B, 1, num_samples]`` ``[B, n_mels, T]``
``[B, C, num_samples]`` ``[B, C, n_mels, T]``
======================  ============================

With ``center=True`` a signal of ``N`` samples yields ``T = 1 + N // hop_length``
frames (e.g. 1.0 s @ 16 kHz with ``hop_length=160`` -> ``T = 101``).
"""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

from src.config import AudioConfig, MelConfig

__all__ = ["MelSpectrogramExtractor", "compute_log_mel", "expected_num_frames"]


def expected_num_frames(num_samples: int, mel: MelConfig) -> int:
    """Number of mel frames produced for ``num_samples`` input samples."""

    if mel.center:
        return 1 + num_samples // mel.hop_length
    return 1 + max(0, (num_samples - mel.win_length)) // mel.hop_length


class MelSpectrogramExtractor(nn.Module):
    """Configurable (log-)mel spectrogram extractor.

    Parameters
    ----------
    cfg:
        Either a full :class:`~src.config.AudioConfig` or just its
        :class:`~src.config.MelConfig`. When an ``AudioConfig`` is given its
        ``sample_rate`` is used; a bare ``MelConfig`` needs ``sample_rate``.
    sample_rate:
        Required only when ``cfg`` is a bare :class:`MelConfig`.
    """

    def __init__(
        self,
        cfg: AudioConfig | MelConfig,
        *,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__()

        if isinstance(cfg, AudioConfig):
            mel = cfg.mel
            sample_rate = cfg.sample_rate
        else:
            mel = cfg
            if sample_rate is None:
                raise ValueError("sample_rate is required when cfg is a bare MelConfig")

        self.sample_rate = int(sample_rate)
        self.mel_cfg = mel

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=mel.n_fft,
            win_length=mel.win_length,
            hop_length=mel.hop_length,
            f_min=mel.f_min,
            f_max=mel.f_max,
            n_mels=mel.n_mels,
            power=mel.power,
            center=mel.center,
        )
        self.to_db = (
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=mel.log_top_db)
            if mel.log
            else None
        )

    def expected_num_frames(self, num_samples: int) -> int:
        return expected_num_frames(num_samples, self.mel_cfg)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 0:
            raise ValueError("waveform must have at least 1 dimension")

        # Remember whether to drop a singleton channel axis from the result.
        squeeze_channel = waveform.ndim >= 2 and waveform.shape[-2] == 1

        mel = self.mel_spectrogram(waveform)          # (..., n_mels, T)
        if self.to_db is not None:
            mel = self.to_db(mel)

        if squeeze_channel:
            mel = mel.squeeze(-3)
        return mel


def compute_log_mel(
    waveform: torch.Tensor,
    cfg: AudioConfig | MelConfig,
    *,
    sample_rate: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Functional one-shot wrapper around :class:`MelSpectrogramExtractor`.

    Applies dB compression when ``cfg`` enables it (``mel.log``), despite the
    name. Runs under ``torch.no_grad`` - Phase 2A only needs the features.
    """

    extractor = MelSpectrogramExtractor(cfg, sample_rate=sample_rate)
    if device is not None:
        extractor = extractor.to(device)
        waveform = waveform.to(device)
    extractor.eval()
    with torch.no_grad():
        return extractor(waveform)
