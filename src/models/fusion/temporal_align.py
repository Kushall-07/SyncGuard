"""Time-aware temporal-correspondence alignment of audio tokens onto a video grid
(Phase 9).

This is **not** interpolation and **not** ``F.adaptive_avg_pool1d``. Those map by
*index proportion* and ignore the real temporal extent of each token and the
per-clip frame rate; their error varies clip-to-clip. Here every token carries an
explicit wall-clock interval and buckets are assembled by **interval overlap**:

* audio token ``j`` occupies ``[j * dt_a, (j+1) * dt_a)`` seconds, where
  ``dt_a = audio_token_seconds`` (for the Phase-5 encoder,
  ``hop_length * time_downsample / sample_rate = 0.08 s``);
* video token ``k`` defines the bucket ``[k / fps, (k+1) / fps)`` seconds;
* both are measured from the start of the aligned window and, when
  ``window_seconds`` is given, clipped to ``[0, window_seconds)``;
* a bucket's value is the **mean of every audio token whose interval overlaps
  it**; a bucket with no overlap is filled by the **nearest audio token by
  interval centre** (``empty_bucket="nearest"``) or by zeros
  (``empty_bucket="zero"``);
* ``bucket_counts[b, k]`` reports how many audio tokens were averaged into bucket
  ``k`` — ``0`` marks a fallback-filled bucket.

The operator is a fixed linear map ``aligned = W @ audio_tokens`` whose weights
``W`` depend only on the timing scalars (``dt_a``, ``fps``, ``T_a``,
``window_seconds``), never on the token values, so it is differentiable w.r.t.
``audio_tokens`` and introduces no NaNs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["align_audio_to_video", "AudioToVideoAligner"]

_EMPTY_BUCKET_MODES = ("nearest", "zero")


def _as_fps_vector(video_fps: float | Tensor, batch: int, device, dtype) -> Tensor:
    fps = torch.as_tensor(video_fps, device=device, dtype=dtype)
    if fps.ndim == 0:
        fps = fps.expand(batch)
    if fps.shape != (batch,):
        raise ValueError(f"video_fps must be a scalar or shape [B={batch}], got {tuple(fps.shape)}")
    if torch.any(fps <= 0):
        raise ValueError("video_fps must be positive")
    return fps.contiguous()


def align_audio_to_video(
    audio_tokens: Tensor,                       # [B, T_a, D]
    *,
    n_video_tokens: int,
    audio_token_seconds: float,
    video_fps: float | Tensor,                  # scalar or [B]
    window_seconds: float | Tensor | None = None,
    empty_bucket: str = "nearest",
    audio_valid_len: Tensor | None = None,      # [B] real T_a per item (right-padded batches)
) -> tuple[Tensor, Tensor]:
    """Deterministic time-aware temporal-correspondence alignment.

    Returns ``(aligned_audio [B, T_v, D], bucket_counts [B, T_v] int64)``.
    ``T_v = n_video_tokens``. See the module docstring for the exact rule.
    """

    if audio_tokens.dim() != 3:
        raise ValueError(f"audio_tokens must be [B, T_a, D], got {tuple(audio_tokens.shape)}")
    if n_video_tokens <= 0:
        raise ValueError("n_video_tokens must be positive")
    if audio_token_seconds <= 0:
        raise ValueError("audio_token_seconds must be positive")
    if empty_bucket not in _EMPTY_BUCKET_MODES:
        raise ValueError(f"empty_bucket must be one of {_EMPTY_BUCKET_MODES}, got {empty_bucket!r}")

    B, T_a, D = audio_tokens.shape
    T_v = int(n_video_tokens)
    device = audio_tokens.device
    work_dtype = audio_tokens.dtype if audio_tokens.is_floating_point() else torch.float32

    fps = _as_fps_vector(video_fps, B, device, work_dtype)                       # [B]
    dt_a = float(audio_token_seconds)

    if window_seconds is None:
        win = (T_v / fps).to(work_dtype)                                        # [B]
    else:
        win = torch.as_tensor(window_seconds, device=device, dtype=work_dtype)
        if win.ndim == 0:
            win = win.expand(B)
        if win.shape != (B,):
            raise ValueError(f"window_seconds must be scalar or [B={B}], got {tuple(win.shape)}")
    if torch.any(win <= 0):
        raise ValueError("window_seconds must be positive")

    # validity mask over audio tokens (padding excluded from overlap + fallback)
    j = torch.arange(T_a, device=device, dtype=work_dtype)                       # [T_a]
    if audio_valid_len is None:
        valid = torch.ones(B, T_a, dtype=torch.bool, device=device)
    else:
        vlen = torch.as_tensor(audio_valid_len, device=device).reshape(-1)
        if vlen.shape != (B,):
            raise ValueError(f"audio_valid_len must be [B={B}], got {tuple(vlen.shape)}")
        valid = torch.arange(T_a, device=device)[None, :] < vlen[:, None]       # [B, T_a]

    # token / bucket intervals, clipped to [0, window_seconds) per item
    a_lo = torch.minimum(j[None, :] * dt_a, win[:, None])                       # [B, T_a]
    a_hi = torch.minimum((j[None, :] + 1.0) * dt_a, win[:, None])              # [B, T_a]
    k = torch.arange(T_v, device=device, dtype=work_dtype)                      # [T_v]
    b_lo = torch.minimum(k[None, :] / fps[:, None], win[:, None])              # [B, T_v]
    b_hi = torch.minimum((k[None, :] + 1.0) / fps[:, None], win[:, None])      # [B, T_v]

    # half-open interval overlap: a_hi > b_lo AND a_lo < b_hi
    overlap = (a_hi[:, None, :] > b_lo[:, :, None]) & (a_lo[:, None, :] < b_hi[:, :, None])
    overlap = overlap & valid[:, None, :]                                       # [B, T_v, T_a]

    counts = overlap.sum(dim=-1)                                                # [B, T_v] int
    W = overlap.to(work_dtype) / counts.clamp(min=1).unsqueeze(-1).to(work_dtype)

    empty = counts == 0                                                         # [B, T_v]
    any_valid = valid.any(dim=1)                                                # [B]
    if empty_bucket == "nearest":
        c_a = (j + 0.5) * dt_a                                                  # [T_a]
        c_k = (k[None, :] + 0.5) / fps[:, None]                                 # [B, T_v]
        dist = (c_k[:, :, None] - c_a[None, None, :]).abs()                     # [B, T_v, T_a]
        dist = dist.masked_fill(~valid[:, None, :], float("inf"))
        nearest = dist.argmin(dim=-1)                                           # [B, T_v]  (ties -> lowest j)
        onehot = F.one_hot(nearest, num_classes=T_a).to(work_dtype)            # [B, T_v, T_a]
        fill = empty & any_valid[:, None]
        W = torch.where(fill.unsqueeze(-1), onehot, W)
    # else "zero": W rows for empty buckets stay all-zero -> aligned row is 0

    aligned = torch.matmul(W, audio_tokens.to(work_dtype))                      # [B, T_v, D]
    if aligned.dtype != audio_tokens.dtype and audio_tokens.is_floating_point():
        aligned = aligned.to(audio_tokens.dtype)
    return aligned, counts.to(torch.int64)


class AudioToVideoAligner(nn.Module):
    """``nn.Module`` wrapper around :func:`align_audio_to_video`. No parameters.

    ``audio_token_seconds`` is fixed at construction (it is a property of the
    audio encoder, not of any single clip). ``video_fps`` / ``window_seconds`` /
    ``audio_valid_len`` are per-call.
    """

    def __init__(self, *, audio_token_seconds: float, empty_bucket: str = "nearest") -> None:
        super().__init__()
        if audio_token_seconds <= 0:
            raise ValueError("audio_token_seconds must be positive")
        if empty_bucket not in _EMPTY_BUCKET_MODES:
            raise ValueError(f"empty_bucket must be one of {_EMPTY_BUCKET_MODES}")
        self.audio_token_seconds = float(audio_token_seconds)
        self.empty_bucket = empty_bucket

    def forward(
        self,
        audio_tokens: Tensor,
        *,
        n_video_tokens: int,
        video_fps: float | Tensor,
        window_seconds: float | Tensor | None = None,
        audio_valid_len: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return align_audio_to_video(
            audio_tokens,
            n_video_tokens=n_video_tokens,
            audio_token_seconds=self.audio_token_seconds,
            video_fps=video_fps,
            window_seconds=window_seconds,
            empty_bucket=self.empty_bucket,
            audio_valid_len=audio_valid_len,
        )

    def extra_repr(self) -> str:
        return f"audio_token_seconds={self.audio_token_seconds}, empty_bucket={self.empty_bucket!r}"
