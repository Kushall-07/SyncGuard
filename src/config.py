"""Typed configuration objects for SyncGuard.

Phase 2A only needs the audio preprocessing / mel-spectrogram configuration.
Later phases (2B onwards) extend this module with model and training sections.

The single entry point is :func:`load_audio_config`, which reads a YAML file such
as ``configs/audio.yaml`` and returns a validated :class:`AudioConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["MelConfig", "AudioConfig", "load_audio_config"]

_VALID_NORMALIZE = ("peak", "rms", "none")


def _check_unknown_keys(data: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in '{where}'. "
            f"Allowed keys: {sorted(allowed)}"
        )


@dataclass(frozen=True)
class MelConfig:
    """Mel-spectrogram parameters (see ``configs/audio.yaml``)."""

    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 2.0
    center: bool = True
    log: bool = True
    log_top_db: float = 80.0

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("n_fft, hop_length and win_length must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.f_min < 0 or self.f_max <= self.f_min:
            raise ValueError("require 0 <= f_min < f_max")
        if self.power not in (1.0, 2.0):
            raise ValueError("power must be 1.0 (magnitude) or 2.0 (power)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MelConfig":
        data = dict(data or {})
        _check_unknown_keys(data, tuple(f.name for f in fields(cls)), "audio.mel")
        return cls(**data)


@dataclass(frozen=True)
class AudioConfig:
    """Audio loading / normalization parameters plus the nested mel config."""

    sample_rate: int = 16000
    mono: bool = True
    normalize: str = "peak"
    norm_target: float = 1.0
    mel: MelConfig = field(default_factory=MelConfig)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.normalize not in _VALID_NORMALIZE:
            raise ValueError(
                f"normalize must be one of {_VALID_NORMALIZE}, got {self.normalize!r}"
            )
        if self.norm_target <= 0:
            raise ValueError("norm_target must be positive")
        if self.mel.f_max > self.sample_rate / 2:
            raise ValueError(
                f"mel.f_max ({self.mel.f_max}) exceeds the Nyquist frequency "
                f"({self.sample_rate / 2})"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AudioConfig":
        data = dict(data)
        scalar_keys = tuple(f.name for f in fields(cls) if f.name != "mel")
        _check_unknown_keys(data, scalar_keys + ("mel",), "audio")
        mel = MelConfig.from_dict(data.pop("mel", None))
        return cls(mel=mel, **data)


def load_audio_config(path: str | Path) -> AudioConfig:
    """Load and validate an :class:`AudioConfig` from a YAML file.

    The file must contain a top-level ``audio:`` mapping, e.g. ``configs/audio.yaml``.
    Unknown keys raise :class:`ValueError` so typos fail loudly.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if "audio" not in raw:
        raise ValueError(f"{path}: missing top-level 'audio:' section")
    if not isinstance(raw["audio"], Mapping):
        raise ValueError(f"{path}: 'audio' section must be a mapping")

    return AudioConfig.from_dict(raw["audio"])
