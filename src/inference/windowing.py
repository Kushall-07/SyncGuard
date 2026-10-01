"""Temporal window construction for windowed audio-visual sync inference.

Production `predict_audio_visual` historically fed an *entire* variable-length
clip's landmark sequence through the frozen visual encoder as one sequence (see
`SyncGuardPredictor._predict_audio_visual_legacy`), aligned a matching whole-clip
audio timeline, and mean-pooled the SyncHead's per-frame logits into a single
video-level score. That is a different temporal regime from the one the
cross-attention + SyncHead were trained/evaluated on: `LAVDFSyncDataset` always
builds a fixed `av_align.video.num_frames` (32) -token sample per training
example, and `scripts/evaluate_sync.py` never evaluates a sequence longer than
that.

This module splits a video's frame timeline into consecutive, fixed-size windows
of `window_frames` (default 32, matching `av_align.video.num_frames`) so that
production inference on a long clip runs the exact same frozen
encoders -> alignment -> cross-attention -> SyncHead pipeline once per window,
each window shaped like the samples the model was actually evaluated on, instead
of once over the whole clip.

Note on "the same temporal regime" as training: `LAVDFSyncDataset` builds its
32-token sample by *sparsely* subsampling a whole (multi-second) clip via
`np.linspace` across all available frames, then aligns only the clip's first
`32 / fps` seconds of audio to those 32 tokens (see `LAVDFSyncDataset.timing`).
That means a training "token k" is not generally frame k of a contiguous
1.28s span, and the audio it gets aligned to is not the audio actually
co-occurring with the (sparsely sampled) video frame it represents. Production
windows here instead use *consecutive* frames per window, so window token k is
truly video frame `start_frame + k` and the aligned audio interval
`[start_frame/fps, (start_frame + window_frames)/fps)` genuinely co-occurs with
it - a dense window of the same *size* (32) the SyncHead was trained on, applied
in a way that keeps `align_audio_to_video`'s "token k covers real time k/fps"
assumption actually true. This is a deliberate, documented choice (see
docs/windowed_av_inference.md), not an attempt to bit-for-bit replicate LAV-DF's
sparse-sampling quirk in a production setting where clips are not curated to a
handful of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WindowSpec", "select_frame_indices", "generate_windows"]


@dataclass(frozen=True)
class WindowSpec:
    """One temporal window over a video's frame timeline."""

    index: int
    start_frame: int    # first real frame index covered by this window
    end_frame: int       # exclusive; min(start_frame + window_frames, n_frames)
    start_time: float    # start_frame / fps, seconds
    end_time: float       # end_frame / fps, seconds (real elapsed time; excludes any padding)
    n_valid_frames: int   # end_frame - start_frame (<= window_frames)
    window_frames: int    # nominal window size (target token count fed to the visual encoder)
    valid_fraction: float  # n_valid_frames / window_frames; 1.0 unless this is a short final window


def select_frame_indices(n_available: int, n_want: int) -> np.ndarray:
    """Deterministically select `n_want` frame indices from `[0, n_available)`.

    Uses ``np.linspace(0, n_available - 1, n_want).round()``: evenly spaced
    indices, repeated when `n_available < n_want`. This is the exact formula
    `LAVDFSyncDataset._select_frames` uses for its dense-sampling branch
    (`n_available >= n_want`, the only branch real LAV-DF clips exercise, since
    every clip has far more than 32 frames); it degenerates to
    `arange(n_want)` when `n_available == n_want` (a window with a full
    complement of frames, the common case here), and generalizes cleanly to
    `n_available < n_want` (a short final window) by deterministically
    repeating frames rather than the training dataset's seeded-random-choice
    fallback for that case. Determinism matters here because production
    inference must be reproducible; the training fallback's randomness is
    fine there because that branch is never actually hit.
    """
    if n_available <= 0:
        raise ValueError("n_available must be positive")
    if n_want <= 0:
        raise ValueError("n_want must be positive")
    if n_available == 1:
        return np.zeros(n_want, dtype=np.int64)
    return np.linspace(0, n_available - 1, n_want).round().astype(np.int64)


def generate_windows(
    n_frames: int,
    fps: float,
    *,
    window_frames: int = 32,
    stride_frames: int = 32,
) -> list[WindowSpec]:
    """Tile ``[0, n_frames)`` into consecutive windows of ``window_frames`` frames,
    stepping by ``stride_frames``.

    Always covers the complete usable timeline: a final window shorter than
    ``window_frames`` (``n_frames`` not divisible by ``window_frames``, or
    ``n_frames < window_frames``) is still emitted, never silently dropped.
    A video with fewer frames than ``window_frames`` yields exactly one
    (short) window.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")

    windows: list[WindowSpec] = []
    start = 0
    index = 0
    while start < n_frames:
        end = min(start + window_frames, n_frames)
        n_valid = end - start
        windows.append(
            WindowSpec(
                index=index,
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
                n_valid_frames=n_valid,
                window_frames=window_frames,
                valid_fraction=n_valid / window_frames,
            )
        )
        index += 1
        start += stride_frames
    return windows
