"""AV encoder for Phase 9: frozen shared audio encoder + frozen visual encoder +
deterministic time-aware temporal alignment.

* Audio side: the **Phase-5 shared CNN+Transformer audio encoder**
  (:class:`src.models.audio.encoder.AudioEncoder`, ``variant="cnn_transformer"``),
  loaded from its exported checkpoint. This is the shared *representation* encoder
  — **not** the Phase-4/5 score-level CNN+Transformer ensemble.
* Visual side: the **Phase-8 visual Transformer encoder**
  (:class:`src.models.video.visual_encoder.VisualEncoder`), loaded from its
  exported checkpoint.
* Alignment: :class:`src.models.fusion.temporal_align.AudioToVideoAligner`.
  ``audio_token_seconds`` is **derived from the exported audio checkpoint's
  payload** (``hop_length * time_downsample / sample_rate``), never hard-coded.

Both encoders are frozen for Phase 9 — ``eval()`` and ``requires_grad_(False)`` —
and stay in eval mode even when the parent module is put in ``train()``. Their
architecture and checkpoint contents are not modified.

No cross-attention, no sync head, no contrastive loss — those are Phases 10/11/12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from src.models.audio.encoder import AudioEncoder, load_audio_encoder
from src.models.fusion.temporal_align import AudioToVideoAligner
from src.models.video.visual_encoder import VisualEncoder, load_visual_encoder

__all__ = ["AVAlignConfig", "load_av_align_config", "AVEncoderOutput", "AVEncoder"]

_REQUIRED_AUDIO_VARIANT = "cnn_transformer"


@dataclass(frozen=True)
class AVAlignConfig:
    """Parsed ``configs/av_align.yaml`` (the ``av_align:`` block)."""

    audio_encoder_pt: str
    visual_encoder_pt: str
    empty_bucket: str = "nearest"
    window_policy: str = "per_clip_fps"       # "per_clip_fps" | a positive float (seconds)
    n_video_tokens: int = 32
    audio_variant_required: str = _REQUIRED_AUDIO_VARIANT

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AVAlignConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        blk = raw.get("av_align", raw)
        align = blk.get("alignment", {})
        window = align.get("window_seconds", "per_clip_fps")
        return cls(
            audio_encoder_pt=str(blk["audio_encoder_pt"]),
            visual_encoder_pt=str(blk["visual_encoder_pt"]),
            empty_bucket=str(align.get("empty_bucket", "nearest")),
            window_policy=str(window),
            n_video_tokens=int(blk.get("video", {}).get("num_frames", 32)),
            audio_variant_required=str(blk.get("audio_variant_required", _REQUIRED_AUDIO_VARIANT)),
        )


def load_av_align_config(path: str | Path) -> AVAlignConfig:
    return AVAlignConfig.from_yaml(path)


@dataclass
class AVEncoderOutput:
    audio_tokens: Tensor      # [B, T_a, D_a]  (D_a = 256 for the Phase-5 encoder)
    visual_tokens: Tensor     # [B, T_v, D_v]  (D_v = 256 for the Phase-8 encoder)
    audio_aligned: Tensor     # [B, T_v, D_a]  (index-aligned to visual_tokens)
    bucket_counts: Tensor     # [B, T_v] int64  (0 -> fallback-filled bucket)
    video_fps: Tensor         # [B] float32


def _audio_token_seconds_from_payload(encoder: AudioEncoder, payload: dict[str, Any]) -> float:
    audio_cfg = payload["audio_cfg"]
    hop_length = int(audio_cfg["mel"]["hop_length"])
    sample_rate = int(audio_cfg["sample_rate"])
    time_downsample = int(encoder.time_downsample)
    return hop_length * time_downsample / sample_rate


class AVEncoder(nn.Module):
    def __init__(
        self,
        audio_encoder_pt: str | Path,
        visual_encoder_pt: str | Path,
        *,
        empty_bucket: str = "nearest",
        map_location: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.audio_encoder, self._audio_payload = load_audio_encoder(
            audio_encoder_pt, map_location=map_location
        )
        self.visual_encoder, self._visual_payload = load_visual_encoder(
            visual_encoder_pt, map_location=map_location
        )

        self.audio_variant = self._audio_payload["variant"]
        if self.audio_variant != _REQUIRED_AUDIO_VARIANT:
            raise ValueError(
                f"Phase 9 requires the shared {_REQUIRED_AUDIO_VARIANT!r} audio encoder "
                f"(not the score-level ensemble), got variant {self.audio_variant!r}"
            )

        self.audio_token_seconds = _audio_token_seconds_from_payload(
            self.audio_encoder, self._audio_payload
        )
        self.n_mels = int(self._audio_payload["n_mels"])
        self.audio_dim = int(self.audio_encoder.output_dim)
        self.visual_dim = int(self.visual_encoder.output_dim)

        self.aligner = AudioToVideoAligner(
            audio_token_seconds=self.audio_token_seconds, empty_bucket=empty_bucket
        )

        for enc in (self.audio_encoder, self.visual_encoder):
            enc.eval()
            enc.requires_grad_(False)

    # keep the frozen encoders in eval() no matter what the parent does
    def train(self, mode: bool = True) -> "AVEncoder":
        super().train(mode)
        self.audio_encoder.eval()
        self.visual_encoder.eval()
        return self

    @classmethod
    def from_config(cls, cfg: AVAlignConfig, *, map_location: str | torch.device = "cpu") -> "AVEncoder":
        return cls(
            cfg.audio_encoder_pt,
            cfg.visual_encoder_pt,
            empty_bucket=cfg.empty_bucket,
            map_location=map_location,
        )

    @torch.no_grad()
    def _encode(self, mel_window: Tensor, landmarks: Tensor) -> tuple[Tensor, Tensor]:
        audio_tokens = self.audio_encoder(mel_window).tokens        # [B, T_a, D_a]
        visual_tokens = self.visual_encoder(landmarks).tokens        # [B, T_v, D_v]
        return audio_tokens, visual_tokens

    def forward(
        self,
        mel_window: Tensor,                 # [B, n_mels, T_mel]  (encoder's own AudioConfig)
        landmarks: Tensor,                  # [B, T_v, N, C]
        video_fps: float | Tensor,          # scalar or [B]
        *,
        window_seconds: float | Tensor | None = None,
        audio_valid_len: Tensor | None = None,
    ) -> AVEncoderOutput:
        audio_tokens, visual_tokens = self._encode(mel_window, landmarks)
        t_v = visual_tokens.shape[1]

        audio_aligned, bucket_counts = self.aligner(
            audio_tokens,
            n_video_tokens=t_v,
            video_fps=video_fps,
            window_seconds=window_seconds,
            audio_valid_len=audio_valid_len,
        )

        fps = torch.as_tensor(video_fps, dtype=torch.float32, device=visual_tokens.device)
        if fps.ndim == 0:
            fps = fps.expand(visual_tokens.shape[0])

        return AVEncoderOutput(
            audio_tokens=audio_tokens,
            visual_tokens=visual_tokens,
            audio_aligned=audio_aligned,
            bucket_counts=bucket_counts,
            video_fps=fps.contiguous(),
        )
