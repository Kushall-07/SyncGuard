"""Typed configuration objects for SyncGuard.

Phase 2A added the audio preprocessing / mel-spectrogram configuration
(:class:`AudioConfig`). Phase 2B adds the experiment / model / training sections
and a composed :class:`Config` plus :func:`load_config` for full experiment YAML
files such as ``configs/default.yaml``.

Entry points
------------
* :func:`load_audio_config` - just the ``audio:`` block (used by the Phase 2A
  scripts and tests).
* :func:`load_config` - a full experiment config (``experiment`` + ``audio`` +
  ``model`` + ``training``).

Every dataclass is frozen and validates itself in ``__post_init__``; unknown YAML
keys raise :class:`ValueError` so typos fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "MelConfig",
    "AudioConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "Config",
    "load_audio_config",
    "load_config",
]

_VALID_NORMALIZE = ("peak", "rms", "none")
_VALID_MONITOR_MODE = ("min", "max")


def _check_unknown_keys(data: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in '{where}'. "
            f"Allowed keys: {sorted(allowed)}"
        )


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(f.name for f in fields(cls))


# --------------------------------------------------------------------------- audio


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
        _check_unknown_keys(data, _field_names(cls), "audio.mel")
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


# ---------------------------------------------------------------------- experiment


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment identity and reproducibility settings."""

    name: str = "syncguard-default"
    seed: int = 1337
    deterministic: bool = False
    output_root: str = "outputs/runs"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment.name must be non-empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ExperimentConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "experiment")
        return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    """Shared model hyper-parameters (starting values from spec section 20)."""

    audio_embedding_dim: int = 256
    visual_embedding_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("audio_embedding_dim", "visual_embedding_dim", "num_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.audio_embedding_dim % self.num_heads:
            raise ValueError("model.audio_embedding_dim must be divisible by model.num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0.0, 1.0)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ModelConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "model")
        return cls(**data)


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop / optimization settings.

    ``amp`` and ``grad_accum_steps`` exist for the 6 GB laptop-GPU constraint
    (spec section 21). ``monitor`` / ``monitor_mode`` drive best-checkpoint
    selection and early stopping in :class:`src.training.trainer.Trainer`.
    """

    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True
    grad_accum_steps: int = 1
    grad_clip_norm: float = 5.0
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    early_stopping_patience: int = 5

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("training.batch_size and training.epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be >= 0")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be >= 0")
        if self.grad_accum_steps < 1:
            raise ValueError("training.grad_accum_steps must be >= 1")
        if self.grad_clip_norm < 0:
            raise ValueError("training.grad_clip_norm must be >= 0 (0 disables clipping)")
        if self.monitor_mode not in _VALID_MONITOR_MODE:
            raise ValueError(f"training.monitor_mode must be one of {_VALID_MONITOR_MODE}")
        if self.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be >= 0 (0 disables)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrainingConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "training")
        return cls(**data)


@dataclass(frozen=True)
class Config:
    """Full experiment configuration: experiment + audio + model + training."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path | None = None) -> "Config":
        data = dict(data)
        _check_unknown_keys(data, _field_names(cls), "<root>")
        return cls(
            experiment=ExperimentConfig.from_dict(data.get("experiment")),
            audio=_resolve_audio_section(data.get("audio"), base_dir),
            model=ModelConfig.from_dict(data.get("model")),
            training=TrainingConfig.from_dict(data.get("training")),
        )


# ------------------------------------------------------------------------- loaders


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_audio_section(
    section: Any,
    base_dir: Path | None,
) -> AudioConfig:
    """Accept an inline ``audio:`` mapping or a path string to an audio YAML file."""

    if section is None:
        return AudioConfig()
    if isinstance(section, str):
        candidates = [Path(section)]
        if base_dir is not None:
            candidates.append(base_dir / section)
        for candidate in candidates:
            if candidate.is_file():
                return load_audio_config(candidate)
        raise FileNotFoundError(f"audio config file not found: tried {list(map(str, candidates))}")
    if isinstance(section, Mapping):
        return AudioConfig.from_dict(section)
    raise ValueError("'audio' must be a mapping or a path string to an audio YAML file")


def load_audio_config(path: str | Path) -> AudioConfig:
    """Load and validate an :class:`AudioConfig` from a YAML file with an ``audio:`` block."""

    path = Path(path)
    raw = _read_yaml(path)
    if "audio" not in raw:
        raise ValueError(f"{path}: missing top-level 'audio:' section")
    if not isinstance(raw["audio"], Mapping):
        raise ValueError(f"{path}: 'audio' section must be a mapping")
    return AudioConfig.from_dict(raw["audio"])


def load_config(path: str | Path) -> Config:
    """Load and validate a full experiment :class:`Config` from a YAML file.

    The ``audio:`` section may be an inline mapping or a path string pointing at a
    separate audio YAML file (resolved relative to CWD, then to ``path``'s dir).
    """

    path = Path(path)
    raw = _read_yaml(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    return Config.from_dict(raw, base_dir=path.parent)
