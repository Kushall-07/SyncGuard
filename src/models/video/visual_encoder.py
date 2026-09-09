"""Shared visual encoder (Phase 7).

:class:`VisualEncoder` is the single visual representation network: landmark
embedding + sinusoidal positional encoding + a temporal Transformer over the
frame tokens, emitting ``[B, T, D]``. The Phase 7 deepfake classifier consumes it
through :class:`~src.models.heads.spoof_head.SpoofHead`, and (from Phase 9) the
visual branch of the AV-sync model consumes the same encoder.

:func:`export_visual_encoder` / :func:`load_visual_encoder` persist just the
encoder weights alongside the ``ModelConfig`` / ``VideoDataConfig`` needed to
rebuild it, mirroring :func:`src.models.audio.encoder.export_audio_encoder`, so a
Phase 7 training run can hand its trained encoder to the sync branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.config import ModelConfig, VideoDataConfig
from src.models.common.temporal_transformer import TemporalTransformerEncoder
from src.models.video.landmark_embedding import LandmarkEmbedding

__all__ = [
    "VisualEncoderOutput",
    "VisualEncoder",
    "export_visual_encoder",
    "load_visual_encoder",
]

_EXPORT_FORMAT = 1


@dataclass
class VisualEncoderOutput:
    tokens: torch.Tensor      # [B, T, D]
    frame_stride: int = 1     # sampled-frame -> token factor (1 for landmarks)


class VisualEncoder(nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.variant = model_cfg.visual_encoder
        self.embed = LandmarkEmbedding(model_cfg)
        self.output_dim = self.embed.output_dim

        use_tf = model_cfg.visual_tf_layers > 0
        self.transformer = (
            TemporalTransformerEncoder(
                d_model=model_cfg.visual_embedding_dim,
                n_heads=model_cfg.num_heads,
                ff_dim=model_cfg.visual_tf_ff_dim,
                dropout=model_cfg.visual_tf_dropout,
                num_layers=model_cfg.visual_tf_layers,
            )
            if use_tf
            else None
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> VisualEncoderOutput:
        tokens = self.embed(landmarks)                     # [B, T, D]
        if self.transformer is not None:
            tokens = self.transformer(tokens, key_padding_mask=key_padding_mask)
        return VisualEncoderOutput(tokens=tokens)


def export_visual_encoder(
    path: str | Path,
    *,
    encoder: VisualEncoder,
    model_cfg: ModelConfig,
    video_cfg: VideoDataConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save the visual encoder weights + the config needed to rebuild it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _EXPORT_FORMAT,
            "state_dict": encoder.state_dict(),
            "model_cfg": asdict(model_cfg),
            "video_cfg": asdict(video_cfg),
            "regions": model_cfg.visual_regions,
            "landmark_coords": model_cfg.landmark_coords,
            "variant": encoder.variant,
            "extra": extra or {},
        },
        path,
    )
    return path


def load_visual_encoder(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[VisualEncoder, dict[str, Any]]:
    """Rebuild a :class:`VisualEncoder` from an :func:`export_visual_encoder` file.

    Returns ``(encoder, payload)`` where ``payload`` carries the ``model_cfg`` /
    ``video_cfg`` dicts, ``regions``, ``landmark_coords`` and any ``extra``.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"visual encoder export not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != _EXPORT_FORMAT:
        raise ValueError(f"{path}: unsupported export format {payload.get('format')!r}")

    model_cfg = ModelConfig.from_dict(payload["model_cfg"])
    encoder = VisualEncoder(model_cfg)
    encoder.load_state_dict(payload["state_dict"], strict=strict)
    encoder.to(map_location)
    return encoder, payload
