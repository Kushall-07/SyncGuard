"""Shared audio encoder (Phase 5).

:class:`AudioEncoder` is the single audio representation network reused across
SyncGuard: the CNN front-end from Phase 4, optionally followed by the Phase 5
Transformer, emitting temporal tokens ``[B, T', D]``. Both the audio-only spoof
classifier and (from Phase 9) the audio branch of the AV sync model consume it.

:func:`export_audio_encoder` / :func:`load_audio_encoder` persist just the encoder
weights alongside the ``ModelConfig`` and ``AudioConfig`` needed to rebuild it, so
a spoof-detection run can hand its trained encoder to the sync branch.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.config import AudioConfig, ModelConfig
from src.models.audio.cnn import AudioCNNEncoder, AudioEncoderOutput
from src.models.audio.transformer import AudioTransformerEncoder

__all__ = ["AudioEncoder", "export_audio_encoder", "load_audio_encoder"]

_EXPORT_FORMAT = 1


class AudioEncoder(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_mels: int) -> None:
        super().__init__()
        self.variant = model_cfg.audio_encoder
        self.cnn = AudioCNNEncoder(model_cfg, n_mels)
        self.output_dim = self.cnn.output_dim
        self.time_downsample = self.cnn.time_downsample

        use_tf = self.variant == "cnn_transformer" and model_cfg.audio_tf_layers > 0
        self.transformer = AudioTransformerEncoder(model_cfg) if use_tf else None

    def forward(
        self,
        mel: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> AudioEncoderOutput:
        out = self.cnn(mel)
        if self.transformer is not None:
            tokens = self.transformer(out.tokens, key_padding_mask=key_padding_mask)
            out = AudioEncoderOutput(tokens=tokens, time_downsample=out.time_downsample)
        return out


def export_audio_encoder(
    path: str | Path,
    *,
    encoder: AudioEncoder,
    model_cfg: ModelConfig,
    audio_cfg: AudioConfig,
    n_mels: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save the encoder weights + the config needed to rebuild it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _EXPORT_FORMAT,
            "state_dict": encoder.state_dict(),
            "model_cfg": asdict(model_cfg),
            "audio_cfg": asdict(audio_cfg),
            "n_mels": n_mels,
            "variant": encoder.variant,
            "extra": extra or {},
        },
        path,
    )
    return path


def load_audio_encoder(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[AudioEncoder, dict[str, Any]]:
    """Rebuild an :class:`AudioEncoder` from an :func:`export_audio_encoder` file.

    Returns ``(encoder, payload)`` where ``payload`` carries ``model_cfg`` /
    ``audio_cfg`` dicts, ``n_mels`` and any ``extra`` metadata.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio encoder export not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != _EXPORT_FORMAT:
        raise ValueError(f"{path}: unsupported export format {payload.get('format')!r}")

    model_cfg = ModelConfig.from_dict(payload["model_cfg"])
    encoder = AudioEncoder(model_cfg, payload["n_mels"])
    encoder.load_state_dict(payload["state_dict"], strict=strict)
    encoder.to(map_location)
    return encoder, payload
