"""Temporal target and pair construction for audio-visual synchronization (Phase 11).

Utilities for constructing positive (aligned) and negative (time-shifted) audio-video
pairs, and generating corresponding per-window sync targets.

IMPORTANT: Manipulation is not equivalent to temporal desynchronization.
LAV-DF fake_periods are parsed but NOT used as sync labels in the baseline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "SyncPairConfig",
    "SyncPair",
    "compute_shifted_alignment",
    "compute_shifted_targets",
    "create_positive_pair",
    "create_negative_pair",
    "parse_fake_periods",
    "sample_shift_seconds",
    "build_sync_pair",
]


@dataclass
class SyncPairConfig:
    """Configuration for sync pair construction."""

    shift_seconds: list[float] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    shift_strategy: Literal["uniform", "random"] = "uniform"
    audio_token_seconds: float = 0.08  # From Phase 5 audio encoder

    def __post_init__(self) -> None:
        if not self.shift_seconds:
            raise ValueError("shift_seconds must not be empty")
        if self.audio_token_seconds <= 0:
            raise ValueError("audio_token_seconds must be positive")


@dataclass
class SyncPair:
    """A single audio-video sync pair with targets."""

    audio_aligned: Tensor  # [T, D] - audio tokens aligned to video timeline
    visual_tokens: Tensor  # [T, D] - visual tokens
    targets: Tensor  # [T] - per-window sync targets (0 or 1)
    mask: Tensor  # [T] - valid positions (True = valid)
    shift_seconds: float  # applied shift (0.0 for positive pairs)
    is_positive: bool  # True = aligned pair, False = shifted pair


def compute_shifted_alignment(
    audio_tokens: Tensor,
    n_video_tokens: int,
    video_fps: float,
    shift_seconds: float,
    audio_token_seconds: float = 0.08,
    window_seconds: float | None = None,
    empty_bucket: str = "nearest",
) -> tuple[Tensor, Tensor]:
    """Compute shifted audio-to-video alignment.

    The shift is implemented at the temporal audio-token level. When audio is shifted
    by Δt seconds, the audio token timeline is offset by Δt relative to the video
    timeline. A video token at position k covers interval [k/fps, (k+1)/fps).
    The shifted audio token that originally covered [j*dt_a, (j+1)*dt_a) now covers
    [j*dt_a + Δt, (j+1)*dt_a + Δt).

    This function recomputes the audio-to-video correspondence using the shifted
    timeline, producing shifted_audio_aligned [T_v, D] from the original audio tokens.

    Args:
        audio_tokens: Original audio tokens [T_a, D]
        n_video_tokens: Number of video tokens T_v
        video_fps: Video frames per second
        shift_seconds: Audio shift in seconds (positive = audio delayed)
        audio_token_seconds: Duration of one audio token in seconds
        window_seconds: Optional window duration for clipping
        empty_bucket: How to fill empty buckets ("nearest" or "zero")

    Returns:
        shifted_audio_aligned: Shifted audio aligned to video timeline [T_v, D]
        bucket_counts: Number of audio tokens averaged per bucket [T_v]
    """
    if audio_tokens.dim() != 2:
        raise ValueError(f"audio_tokens must be [T_a, D], got {tuple(audio_tokens.shape)}")
    if n_video_tokens <= 0:
        raise ValueError("n_video_tokens must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if audio_token_seconds <= 0:
        raise ValueError("audio_token_seconds must be positive")

    T_a, D = audio_tokens.shape
    T_v = int(n_video_tokens)
    device = audio_tokens.device
    work_dtype = audio_tokens.dtype if audio_tokens.is_floating_point() else torch.float32

    dt_a = float(audio_token_seconds)
    fps = float(video_fps)

    # Determine window duration
    if window_seconds is None:
        win = T_v / fps
    else:
        win = float(window_seconds)

    win_t = torch.tensor(win, device=device, dtype=work_dtype)

    # Audio token intervals with shift: [j*dt_a + Δt, (j+1)*dt_a + Δt)
    j = torch.arange(T_a, device=device, dtype=work_dtype)
    a_lo = j * dt_a + shift_seconds  # [T_a]
    a_hi = (j + 1.0) * dt_a + shift_seconds

    # Video token intervals: [k/fps, (k+1)/fps)
    k = torch.arange(T_v, device=device, dtype=work_dtype)
    b_lo = k / fps  # [T_v]
    b_hi = (k + 1.0) / fps

    # Half-open interval overlap: a_hi > b_lo AND a_lo < b_hi  -> [T_v, T_a]
    overlap = (a_hi[None, :] > b_lo[:, None]) & (a_lo[None, :] < b_hi[:, None])

    counts = overlap.sum(dim=-1)  # [T_v]
    W = overlap.to(work_dtype) / counts.clamp(min=1).unsqueeze(-1).to(work_dtype)

    empty = counts == 0  # [T_v]
    if empty_bucket == "nearest":
        c_a = (j + 0.5) * dt_a + shift_seconds  # [T_a] shifted centres
        c_k = (k + 0.5) / fps  # [T_v]
        dist = (c_k[:, None] - c_a[None, :]).abs()  # [T_v, T_a]
        nearest = dist.argmin(dim=-1)  # [T_v]
        onehot = torch.nn.functional.one_hot(nearest, num_classes=T_a).to(work_dtype)
        W = torch.where(empty.unsqueeze(-1), onehot, W)
    # else "zero": W rows for empty buckets stay all-zero

    shifted_aligned = torch.matmul(W, audio_tokens.to(work_dtype))  # [T_v, D]
    if shifted_aligned.dtype != audio_tokens.dtype and audio_tokens.is_floating_point():
        shifted_aligned = shifted_aligned.to(audio_tokens.dtype)

    return shifted_aligned, counts.to(torch.int64)


def compute_shifted_targets(
    n_video_tokens: int,
    video_fps: float,
    shift_seconds: float,
    audio_token_seconds: float = 0.08,
    window_seconds: float | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute per-window sync targets for a time-shifted audio-video pair.

    DECISION DOCUMENTATION:
    For the baseline, we use a simple rule:
    - If shift_seconds == 0, the pair is positive (target=1)
    - If shift_seconds != 0, the pair is negative (target=0)

    This avoids the complexity of threshold-based decisions. Any non-zero shift
    is considered a deliberate negative example for self-supervised learning.

    NOTE: The mask returned by this function is always all-True. For negative
    pairs, the actual valid mask should be computed from bucket_counts (overlap
    check) in create_negative_pair to mask out positions with no audio overlap.

    Args:
        n_video_tokens: Number of video tokens T
        video_fps: Video frames per second
        shift_seconds: Audio shift in seconds (positive = audio delayed)
        audio_token_seconds: Duration of one audio token in seconds
        window_seconds: Optional window duration for clipping
        device: Device to place tensors on (defaults to CPU)

    Returns:
        targets: Per-window sync targets [T] (0 or 1)
        mask: Valid positions mask [T] (all True here; actual masking done elsewhere)
    """
    if n_video_tokens <= 0:
        raise ValueError("n_video_tokens must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")

    # Simple rule: zero shift = positive, non-zero shift = negative
    if shift_seconds == 0.0:
        targets = torch.ones(n_video_tokens, dtype=torch.float32, device=device)
    else:
        targets = torch.zeros(n_video_tokens, dtype=torch.float32, device=device)

    mask = torch.ones(n_video_tokens, dtype=torch.bool, device=device)

    return targets, mask


def parse_fake_periods(fake_periods_str: str) -> list[list[float]]:
    """Parse LAV-DF fake_periods from manifest JSON string.

    Args:
        fake_periods_str: JSON string like "[[4.1, 5.044]]" or "[[2.8, 3.464], [4.964, 6.048]]"

    Returns:
        List of [start, end] periods in seconds

    Note:
        This function parses fake_periods for potential future use in analysis,
        but does NOT use them as sync labels. Manipulation is not equivalent to
        temporal desynchronization.
    """
    if not fake_periods_str or fake_periods_str == "[]":
        return []

    try:
        periods = json.loads(fake_periods_str)
        if not isinstance(periods, list):
            raise ValueError("fake_periods must be a list")
        for period in periods:
            if not isinstance(period, list) or len(period) != 2:
                raise ValueError("each fake_period must be [start, end]")
            if not all(isinstance(x, (int, float)) for x in period):
                raise ValueError("fake_period values must be numeric")
        return periods
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in fake_periods: {e}")


def create_positive_pair(
    audio_aligned: Tensor,
    visual_tokens: Tensor,
    bucket_counts: Tensor | None = None,
) -> SyncPair:
    """Create a positive (aligned) sync pair.

    Args:
        audio_aligned: Audio tokens aligned to video timeline [T, D]
        visual_tokens: Visual tokens [T, D]
        bucket_counts: Optional bucket counts [T] for masking invalid positions

    Returns:
        SyncPair with all targets=1 (aligned)
    """
    if audio_aligned.shape != visual_tokens.shape:
        raise ValueError(
            f"audio_aligned {tuple(audio_aligned.shape)} must match visual_tokens {tuple(visual_tokens.shape)}"
        )

    T = audio_aligned.shape[0]

    # All positions are aligned for positive pairs
    targets = torch.ones(T, dtype=torch.float32, device=audio_aligned.device)

    # Use bucket_counts to create mask if provided
    if bucket_counts is not None:
        mask = bucket_counts > 0
    else:
        mask = torch.ones(T, dtype=torch.bool, device=audio_aligned.device)

    return SyncPair(
        audio_aligned=audio_aligned,
        visual_tokens=visual_tokens,
        targets=targets,
        mask=mask,
        shift_seconds=0.0,
        is_positive=True,
    )


def create_negative_pair(
    audio_tokens: Tensor,
    visual_tokens: Tensor,
    shift_seconds: float,
    video_fps: float,
    audio_token_seconds: float = 0.08,
    window_seconds: float | None = None,
    empty_bucket: str = "nearest",
) -> SyncPair:
    """Create a negative (time-shifted) sync pair.

    The shift is implemented by recomputing audio-to-video alignment with a shifted
    timeline. This produces a different aligned audio representation from the same
    original audio tokens.

    Args:
        audio_tokens: Original audio tokens [T_a, D] (before alignment)
        visual_tokens: Visual tokens [T_v, D]
        shift_seconds: Audio shift in seconds (positive = audio delayed)
        video_fps: Video frames per second
        audio_token_seconds: Duration of one audio token in seconds
        window_seconds: Optional window duration for clipping
        empty_bucket: How to fill empty buckets ("nearest" or "zero")

    Returns:
        SyncPair with shifted audio alignment and negative targets
    """
    if audio_tokens.dim() != 2:
        raise ValueError(f"audio_tokens must be [T_a, D], got {tuple(audio_tokens.shape)}")
    if visual_tokens.dim() != 2:
        raise ValueError(f"visual_tokens must be [T_v, D], got {tuple(visual_tokens.shape)}")

    T_v = visual_tokens.shape[0]

    # Compute shifted alignment
    shifted_audio_aligned, bucket_counts = compute_shifted_alignment(
        audio_tokens=audio_tokens,
        n_video_tokens=T_v,
        video_fps=video_fps,
        shift_seconds=shift_seconds,
        audio_token_seconds=audio_token_seconds,
        window_seconds=window_seconds,
        empty_bucket=empty_bucket,
    )

    # Compute targets (non-zero shift = negative)
    targets, _ = compute_shifted_targets(
        n_video_tokens=T_v,
        video_fps=video_fps,
        shift_seconds=shift_seconds,
        audio_token_seconds=audio_token_seconds,
        window_seconds=window_seconds,
        device=audio_tokens.device,
    )

    # Use bucket_counts to create mask (valid positions have overlapping audio)
    mask = bucket_counts > 0

    # Apply bucket_counts mask
    mask = mask & (bucket_counts > 0)

    return SyncPair(
        audio_aligned=shifted_audio_aligned,
        visual_tokens=visual_tokens,
        targets=targets,
        mask=mask,
        shift_seconds=shift_seconds,
        is_positive=False,
    )


def sample_shift_seconds(
    config: SyncPairConfig,
    rng: random.Random | None = None,
    *,
    include_positive: bool = False,
) -> float:
    """Sample a temporal shift value from ``SyncPairConfig``.

    ``shift_strategy``:
    - ``uniform``: cycle through ``shift_seconds`` in order (deterministic without rng)
    - ``random``: uniform random choice from ``shift_seconds``

    When ``include_positive`` is False, zero is excluded from the candidate pool.
    """
    rng = rng or random.Random()
    candidates = list(config.shift_seconds)
    if not include_positive:
        candidates = [s for s in candidates if s != 0.0]
        if not candidates:
            raise ValueError("shift_seconds must contain a non-zero value when include_positive=False")

    if config.shift_strategy == "random":
        return float(rng.choice(candidates))
    # "uniform": deterministic round-robin via rng for reproducibility in tests
    idx = rng.randrange(len(candidates))
    return float(candidates[idx])


def build_sync_pair(
    audio_tokens: Tensor,
    visual_tokens: Tensor,
    *,
    video_fps: float,
    shift_seconds: float = 0.0,
    audio_token_seconds: float = 0.08,
    window_seconds: float | None = None,
    empty_bucket: str = "nearest",
    positive_audio_aligned: Tensor | None = None,
    positive_bucket_counts: Tensor | None = None,
) -> SyncPair:
    """Build a sync pair from encoder outputs with optional temporal audio shift.

    Positive (``shift_seconds == 0``): uses ``positive_audio_aligned`` when provided
    (the normal Phase-9 alignment); otherwise recomputes unshifted alignment.

    Negative (``shift_seconds != 0``): recomputes audio-to-video correspondence on
    a shifted audio-token timeline, producing a different ``audio_aligned`` tensor
    from the same original ``audio_tokens``.
    """
    if audio_tokens.dim() != 2:
        raise ValueError(f"audio_tokens must be [T_a, D], got {tuple(audio_tokens.shape)}")
    if visual_tokens.dim() != 2:
        raise ValueError(f"visual_tokens must be [T_v, D], got {tuple(visual_tokens.shape)}")

    if shift_seconds == 0.0:
        if positive_audio_aligned is not None:
            audio_aligned = positive_audio_aligned
            bucket_counts = positive_bucket_counts
        else:
            audio_aligned, bucket_counts = compute_shifted_alignment(
                audio_tokens=audio_tokens,
                n_video_tokens=visual_tokens.shape[0],
                video_fps=video_fps,
                shift_seconds=0.0,
                audio_token_seconds=audio_token_seconds,
                window_seconds=window_seconds,
                empty_bucket=empty_bucket,
            )
        return create_positive_pair(
            audio_aligned=audio_aligned,
            visual_tokens=visual_tokens,
            bucket_counts=bucket_counts,
        )

    return create_negative_pair(
        audio_tokens=audio_tokens,
        visual_tokens=visual_tokens,
        shift_seconds=shift_seconds,
        video_fps=video_fps,
        audio_token_seconds=audio_token_seconds,
        window_seconds=window_seconds,
        empty_bucket=empty_bucket,
    )
