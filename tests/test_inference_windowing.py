"""Tests for src/inference/windowing.py (production windowed AV inference, Phase 22).

Covers window generation for exact/short/long/non-divisible clip lengths and the
deterministic frame-selection formula windows use to build a fixed-size landmark
tensor per window.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.windowing import WindowSpec, generate_windows, select_frame_indices


# --------------------------------------------------------------------------- select_frame_indices


def test_select_frame_indices_identity_when_exact_match() -> None:
    idx = select_frame_indices(32, 32)
    assert np.array_equal(idx, np.arange(32))


def test_select_frame_indices_downsamples_dense_case() -> None:
    idx = select_frame_indices(120, 32)
    assert idx.shape == (32,)
    assert idx[0] == 0
    assert idx[-1] == 119
    assert np.all(np.diff(idx) >= 0)  # monotonically non-decreasing


def test_select_frame_indices_upsamples_short_case() -> None:
    idx = select_frame_indices(5, 32)
    assert idx.shape == (32,)
    assert idx.min() == 0
    assert idx.max() == 4
    assert len(set(idx.tolist())) <= 5  # necessarily repeats


def test_select_frame_indices_single_frame() -> None:
    idx = select_frame_indices(1, 32)
    assert np.all(idx == 0)


def test_select_frame_indices_is_deterministic() -> None:
    a = select_frame_indices(17, 32)
    b = select_frame_indices(17, 32)
    assert np.array_equal(a, b)


def test_select_frame_indices_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        select_frame_indices(0, 32)
    with pytest.raises(ValueError):
        select_frame_indices(10, 0)


# --------------------------------------------------------------------------- generate_windows


def test_generate_windows_exact_multiple() -> None:
    windows = generate_windows(64, 25.0, window_frames=32, stride_frames=32)
    assert len(windows) == 2
    assert [w.start_frame for w in windows] == [0, 32]
    assert [w.end_frame for w in windows] == [32, 64]
    assert all(w.valid_fraction == 1.0 for w in windows)
    assert all(w.n_valid_frames == 32 for w in windows)


def test_generate_windows_non_divisible_covers_full_timeline() -> None:
    """117 frames at 32-frame windows -> 0-31, 32-63, 64-95, 96-116 (final partial window kept)."""
    windows = generate_windows(117, 25.0, window_frames=32, stride_frames=32)
    assert len(windows) == 4
    assert [(w.start_frame, w.end_frame) for w in windows] == [(0, 32), (32, 64), (64, 96), (96, 117)]
    assert windows[-1].n_valid_frames == 21
    assert windows[-1].valid_fraction == pytest.approx(21 / 32)
    assert windows[-1].end_time == pytest.approx(117 / 25.0)
    # Every frame index up to n_frames is covered by exactly one window.
    covered = set()
    for w in windows:
        covered.update(range(w.start_frame, w.end_frame))
    assert covered == set(range(117))


def test_generate_windows_short_video_single_window() -> None:
    windows = generate_windows(20, 25.0, window_frames=32, stride_frames=32)
    assert len(windows) == 1
    assert windows[0].start_frame == 0
    assert windows[0].end_frame == 20
    assert windows[0].n_valid_frames == 20
    assert windows[0].valid_fraction == pytest.approx(20 / 32)


def test_generate_windows_exact_32_frames_single_full_window() -> None:
    windows = generate_windows(32, 25.0, window_frames=32, stride_frames=32)
    assert len(windows) == 1
    assert windows[0].n_valid_frames == 32
    assert windows[0].valid_fraction == 1.0


def test_generate_windows_timestamps_use_real_fps() -> None:
    windows = generate_windows(64, 30.0, window_frames=32, stride_frames=32)
    assert windows[0].start_time == pytest.approx(0.0)
    assert windows[0].end_time == pytest.approx(32 / 30.0)
    assert windows[1].start_time == pytest.approx(32 / 30.0)
    assert windows[1].end_time == pytest.approx(64 / 30.0)


def test_generate_windows_supports_overlap_when_stride_smaller() -> None:
    windows = generate_windows(64, 25.0, window_frames=32, stride_frames=16)
    starts = [w.start_frame for w in windows]
    assert starts == [0, 16, 32, 48]


def test_generate_windows_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        generate_windows(0, 25.0, window_frames=32, stride_frames=32)
    with pytest.raises(ValueError):
        generate_windows(64, 0.0, window_frames=32, stride_frames=32)
    with pytest.raises(ValueError):
        generate_windows(64, 25.0, window_frames=0, stride_frames=32)
    with pytest.raises(ValueError):
        generate_windows(64, 25.0, window_frames=32, stride_frames=0)


def test_window_spec_is_frozen() -> None:
    w = WindowSpec(
        index=0, start_frame=0, end_frame=32, start_time=0.0, end_time=1.28,
        n_valid_frames=32, window_frames=32, valid_fraction=1.0,
    )
    with pytest.raises(Exception):
        w.index = 1  # type: ignore[misc]
