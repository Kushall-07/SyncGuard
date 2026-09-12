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
    "SpecAugmentConfig",
    "AugmentConfig",
    "DataConfig",
    "VideoDataConfig",
    "VideoAugmentConfig",
    "Config",
    "load_audio_config",
    "load_config",
]

_VALID_NORMALIZE = ("peak", "rms", "none")
_VALID_MONITOR_MODE = ("min", "max")
_VALID_SCHEDULER = ("none", "cosine")
_VALID_CLASS_WEIGHT = ("balanced", "none")
_VALID_POOLING = ("attentive", "mean", "meanmax")
_VALID_FEATURE = ("logmel", "waveform")
_VALID_AUDIO_ENCODER = ("cnn", "cnn_transformer")
_VALID_VISUAL_ENCODER = ("transformer", "mlp_baseline")
_VALID_VISUAL_REGIONS = ("face", "mouth", "face_mouth")
_VALID_LANDMARK_NORMALIZE = ("interocular", "none")


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
    """Shared model hyper-parameters (starting values from spec section 20).

    The ``audio_cnn_*`` and ``spoof_head_*`` fields configure the Phase 4 CNN
    audio baseline; ``audio_embedding_dim`` is the temporal-token dimension the
    encoder emits and that later phases (Transformer, cross-attention) consume.
    """

    audio_embedding_dim: int = 256
    visual_embedding_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1

    audio_cnn_channels: tuple[int, ...] = (32, 64, 128)
    audio_cnn_dropout: float = 0.1
    spoof_head_hidden: int = 128
    spoof_head_pooling: str = "attentive"

    # Audio encoder variant (Phase 5): "cnn" = CNN front-end only,
    # "cnn_transformer" = CNN front-end + Transformer encoder over its tokens.
    audio_encoder: str = "cnn"
    audio_tf_layers: int = 3
    audio_tf_ff_dim: int = 1024
    audio_tf_dropout: float = 0.1

    # Visual branch (Phase 7): the classifier consumes normalised MediaPipe
    # face/mouth landmark sequences, not pixels. "transformer" = landmark
    # embedding + positional encoding + temporal Transformer + deepfake head;
    # "mlp_baseline" = landmark embedding + order-free temporal pooling + MLP
    # (the non-temporal reference used to measure the temporal contribution).
    visual_encoder: str = "transformer"
    visual_regions: str = "face_mouth"
    landmark_coords: int = 3
    visual_embed_hidden: int = 256
    visual_tf_layers: int = 3
    visual_tf_ff_dim: int = 1024
    visual_tf_dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("audio_embedding_dim", "visual_embedding_dim", "num_heads",
                     "spoof_head_hidden", "audio_tf_ff_dim", "visual_embed_hidden",
                     "visual_tf_ff_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.audio_embedding_dim % self.num_heads:
            raise ValueError("model.audio_embedding_dim must be divisible by model.num_heads")
        if self.visual_embedding_dim % self.num_heads:
            raise ValueError("model.visual_embedding_dim must be divisible by model.num_heads")
        for name in ("dropout", "audio_cnn_dropout", "audio_tf_dropout", "visual_tf_dropout"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"model.{name} must be in [0.0, 1.0)")
        if not self.audio_cnn_channels or any(c <= 0 for c in self.audio_cnn_channels):
            raise ValueError("model.audio_cnn_channels must be a non-empty list of positive ints")
        if self.spoof_head_pooling not in _VALID_POOLING:
            raise ValueError(f"model.spoof_head_pooling must be one of {_VALID_POOLING}")
        if self.audio_encoder not in _VALID_AUDIO_ENCODER:
            raise ValueError(f"model.audio_encoder must be one of {_VALID_AUDIO_ENCODER}")
        if self.audio_tf_layers < 0:
            raise ValueError("model.audio_tf_layers must be >= 0")
        if self.visual_encoder not in _VALID_VISUAL_ENCODER:
            raise ValueError(f"model.visual_encoder must be one of {_VALID_VISUAL_ENCODER}")
        if self.visual_regions not in _VALID_VISUAL_REGIONS:
            raise ValueError(f"model.visual_regions must be one of {_VALID_VISUAL_REGIONS}")
        if self.landmark_coords not in (2, 3):
            raise ValueError("model.landmark_coords must be 2 or 3")
        if self.visual_tf_layers < 0:
            raise ValueError("model.visual_tf_layers must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ModelConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "model")
        if "audio_cnn_channels" in data:
            data["audio_cnn_channels"] = tuple(data["audio_cnn_channels"])
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
    # Per-epoch LR schedule (stepped once per epoch by src.training.trainer.Trainer).
    # "none" keeps the flat LR (existing behaviour); "cosine" does an optional
    # `warmup_epochs` linear ramp then cosine decay to ~0 - see
    # src.training.utils.build_scheduler.
    scheduler: str = "none"
    warmup_epochs: int = 0
    # Only read by the video branch (scripts/train_deepfake_landmark.py):
    # "balanced" = inverse-frequency class weights (default, unchanged);
    # "none" = ordinary cross-entropy.
    class_weight: str = "balanced"
    # Phase 11: probability of using negative (time-shifted) pairs during training
    negative_pair_probability: float = 0.5
    # Phase 12: weight for contrastive loss (0 = sync-only, >0 = combined)
    lambda_contrastive: float = 0.0

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
        if self.scheduler not in _VALID_SCHEDULER:
            raise ValueError(f"training.scheduler must be one of {_VALID_SCHEDULER}")
        if not 0 <= self.negative_pair_probability <= 1:
            raise ValueError("training.negative_pair_probability must be in [0.0, 1.0]")
        if self.warmup_epochs < 0:
            raise ValueError("training.warmup_epochs must be >= 0")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("training.warmup_epochs must be < training.epochs")
        if self.class_weight not in _VALID_CLASS_WEIGHT:
            raise ValueError(f"training.class_weight must be one of {_VALID_CLASS_WEIGHT}")
        if self.lambda_contrastive < 0:
            raise ValueError("training.lambda_contrastive must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrainingConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "training")
        return cls(**data)


@dataclass(frozen=True)
class SpecAugmentConfig:
    """SpecAugment masking applied to log-mel batches during training only."""

    freq_masks: int = 2
    freq_mask_width: int = 12       # max mel bins per mask
    time_masks: int = 2
    time_mask_width: int = 16       # max frames per mask

    def __post_init__(self) -> None:
        for name in _field_names(type(self)):
            if getattr(self, name) < 0:
                raise ValueError(f"augment.specaugment.{name} must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SpecAugmentConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "augment.specaugment")
        return cls(**data)


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time audio augmentation (waveform perturbations + SpecAugment)."""

    enabled: bool = True
    noise_prob: float = 0.5
    noise_snr_db: tuple[float, float] = (10.0, 30.0)
    gain_prob: float = 0.5
    gain_db: tuple[float, float] = (-6.0, 6.0)
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)

    def __post_init__(self) -> None:
        for name in ("noise_prob", "gain_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"augment.{name} must be in [0.0, 1.0]")
        for name in ("noise_snr_db", "gain_db"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"augment.{name} must be (low, high) with low <= high")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AugmentConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "augment")
        spec = SpecAugmentConfig.from_dict(data.pop("specaugment", None))
        for name in ("noise_snr_db", "gain_db"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(specaugment=spec, **data)


@dataclass(frozen=True)
class DataConfig:
    """Where the audio dataset lives and how samples are framed for batching."""

    source: str = "synthetic"
    root: str = "data/asvspoof/LA"
    manifest_csv: str = "data/asvspoof/manifest.csv"
    feature: str = "logmel"
    fixed_seconds: float = 4.0
    splits: tuple[str, ...] = ("train", "dev", "eval")
    subset_size: int | None = None
    subset_seed: int = 0
    subset_per_speaker_cap: int | None = None

    def __post_init__(self) -> None:
        if self.feature not in _VALID_FEATURE:
            raise ValueError(f"data.feature must be one of {_VALID_FEATURE}")
        if self.fixed_seconds <= 0:
            raise ValueError("data.fixed_seconds must be positive")
        if not self.splits:
            raise ValueError("data.splits must be non-empty")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("data.subset_size must be positive or null")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DataConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "data")
        if "splits" in data:
            data["splits"] = tuple(data["splits"])
        return cls(**data)


@dataclass(frozen=True)
class VideoDataConfig:
    """Where the Celeb-DF video dataset and its cached landmarks live (Phase 7).

    The model consumes ``<landmarks_dir>/<sample_id>.npz`` (raw MediaPipe FaceMesh
    points, normalised in the dataset); ``crops_dir`` only holds optional face
    JPEGs for preprocessing QA and is never read by the classifier.
    """

    source: str = "celeb-df-v2"
    root: str = "data/celebdf"
    landmarks_dir: str = "data/celebdf/landmarks"
    crops_dir: str = "data/celebdf/crops"
    manifest_csv: str = "data/celebdf/manifest.csv"
    num_frames: int = 16
    regions: str = "face_mouth"
    normalize: str = "interocular"
    align_rotation: bool = True
    dev_identity_frac: float = 0.15
    min_valid_frames: int = 4
    splits: tuple[str, ...] = ("train", "dev", "eval")
    subset_size: int | None = None
    subset_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("video.num_frames must be positive")
        if self.regions not in _VALID_VISUAL_REGIONS:
            raise ValueError(f"video.regions must be one of {_VALID_VISUAL_REGIONS}")
        if self.normalize not in _VALID_LANDMARK_NORMALIZE:
            raise ValueError(f"video.normalize must be one of {_VALID_LANDMARK_NORMALIZE}")
        if not 0.0 <= self.dev_identity_frac < 1.0:
            raise ValueError("video.dev_identity_frac must be in [0.0, 1.0)")
        if self.min_valid_frames < 0:
            raise ValueError("video.min_valid_frames must be >= 0")
        if not self.splits:
            raise ValueError("video.splits must be non-empty")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("video.subset_size must be positive or null")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "VideoDataConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "video")
        if "splits" in data:
            data["splits"] = tuple(data["splits"])
        return cls(**data)


@dataclass(frozen=True)
class VideoAugmentConfig:
    """Training-time landmark-space augmentation for the visual branch (Phase 7)."""

    enabled: bool = True
    coord_jitter_std: float = 0.01      # gaussian noise added to normalised coords
    coord_jitter_prob: float = 0.5
    time_mask_frames: int = 2           # up to this many frames zero-velocity held
    time_mask_prob: float = 0.5
    hflip_prob: float = 0.5             # mirror via landmarks.MIRROR_MAP
    scale_jitter: float = 0.05         # +/- fractional global scale
    rot_jitter_deg: float = 5.0        # +/- global in-plane rotation

    def __post_init__(self) -> None:
        for name in ("coord_jitter_prob", "time_mask_prob", "hflip_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"video_augment.{name} must be in [0.0, 1.0]")
        for name in ("coord_jitter_std", "scale_jitter", "rot_jitter_deg"):
            if getattr(self, name) < 0:
                raise ValueError(f"video_augment.{name} must be >= 0")
        if self.time_mask_frames < 0:
            raise ValueError("video_augment.time_mask_frames must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "VideoAugmentConfig":
        data = dict(data or {})
        _check_unknown_keys(data, _field_names(cls), "video_augment")
        return cls(**data)


@dataclass(frozen=True)
class Config:
    """Full experiment configuration.

    ``data`` / ``audio`` / ``augment`` drive the audio branch (Phases 2-6);
    ``video`` / ``video_augment`` drive the visual branch (Phase 7) and are
    ``None`` for audio-only experiments.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    video: VideoDataConfig | None = None
    video_augment: VideoAugmentConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path | None = None) -> "Config":
        data = dict(data)
        _check_unknown_keys(data, _field_names(cls), "<root>")
        video_raw = _resolve_pointer(data.get("video"), base_dir, "video")
        return cls(
            experiment=ExperimentConfig.from_dict(data.get("experiment")),
            data=_resolve_section(data.get("data"), base_dir, "data", DataConfig.from_dict),
            audio=_resolve_audio_section(data.get("audio"), base_dir),
            model=ModelConfig.from_dict(data.get("model")),
            training=TrainingConfig.from_dict(data.get("training")),
            augment=AugmentConfig.from_dict(data.get("augment")),
            video=VideoDataConfig.from_dict(video_raw) if video_raw is not None else None,
            video_augment=(
                VideoAugmentConfig.from_dict(data.get("video_augment"))
                if data.get("video_augment") is not None
                else None
            ),
        )


# ------------------------------------------------------------------------- loaders


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_pointer(section: Any, base_dir: Path | None, key: str) -> Any:
    """If ``section`` is a path string, load that YAML and return its ``key`` sub-mapping."""

    if not isinstance(section, str):
        return section
    candidates = [Path(section)]
    if base_dir is not None:
        candidates.append(base_dir / section)
    for candidate in candidates:
        if candidate.is_file():
            sub = _read_yaml(candidate)
            if key not in sub:
                raise ValueError(f"{candidate}: missing top-level '{key}:' section")
            return sub[key]
    raise FileNotFoundError(
        f"'{key}' config file not found: tried {list(map(str, candidates))}"
    )


def _resolve_section(section: Any, base_dir: Path | None, key: str, builder):
    """Build a sub-config from an inline mapping, a pointer file, or defaults."""

    section = _resolve_pointer(section, base_dir, key)
    return builder(section)


def _resolve_audio_section(section: Any, base_dir: Path | None) -> AudioConfig:
    """Accept an inline ``audio:`` mapping or a path string to an audio YAML file."""

    section = _resolve_pointer(section, base_dir, "audio")
    if section is None:
        return AudioConfig()
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
