"""Phase 8A: dense contiguous landmark-window selection in the extraction script.

Covers frame selection (contiguous / center / stride / start / end), short-video
handling, dense output shape, audit reflection, and that the legacy sparse path
is byte-for-byte unchanged. No MediaPipe model, no video, no training.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "extract_celebdf_landmarks.py"
_spec = importlib.util.spec_from_file_location("extract_celebdf_landmarks", _SCRIPT)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)  # type: ignore[union-attr]


# ---- fakes -----------------------------------------------------------------

class _FakePoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, i: int) -> None:
        self.x = 0.001 * i
        self.y = 0.002 * i
        self.z = 0.0


class _FakeResult:
    face_landmarks = [[_FakePoint(i) for i in range(ex.N_FACEMESH_POINTS)]]


class _FakeLandmarker:
    def detect(self, image):  # noqa: ARG002 - image ignored
        return _FakeResult()


def _patch_frames(monkeypatch, n: int, fps: float = 30.0):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]
    monkeypatch.setattr(ex, "read_frames", lambda p, max_frames=600: (frames, fps))


# ---- dense_window_indices ------------------------------------------------

def test_window_is_contiguous_and_correct_length() -> None:
    idx = ex.dense_window_indices(500, window=32, stride=1, window_start="start")
    assert idx.tolist() == list(range(0, 32))
    assert np.all(np.diff(idx) == 1)
    assert idx.dtype == np.int64


def test_center_window_is_deterministic_and_centred() -> None:
    # span = 32, n = 500 -> start = (500-32)//2 = 234
    idx = ex.dense_window_indices(500, 32, 1, "center")
    assert idx[0] == 234 and idx[-1] == 265
    # symmetric margins (floor)
    left, right = idx[0], 500 - 1 - idx[-1]
    assert left == right == 234
    # deterministic
    assert np.array_equal(idx, ex.dense_window_indices(500, 32, 1, "center"))


def test_window_start_variants() -> None:
    assert ex.dense_window_indices(100, 32, 1, "start")[0] == 0
    assert ex.dense_window_indices(100, 32, 1, "end")[-1] == 99
    assert ex.dense_window_indices(100, 32, 1, "end")[0] == 100 - 32


def test_stride_expands_the_span_but_not_the_count() -> None:
    idx = ex.dense_window_indices(200, window=8, stride=3, window_start="start")
    assert idx.tolist() == [0, 3, 6, 9, 12, 15, 18, 21]
    assert len(idx) == 8
    assert np.all(np.diff(idx) == 3)
    # centred with stride: span = (8-1)*3+1 = 22, n=200 -> start=(200-22)//2=89
    c = ex.dense_window_indices(200, 8, 3, "center")
    assert c[0] == 89 and c[-1] == 89 + 21


def test_short_clip_returns_none() -> None:
    assert ex.dense_window_indices(31, 32, 1, "center") is None      # one frame short
    assert ex.dense_window_indices(32, 32, 1, "center") is not None  # exactly enough
    assert ex.dense_window_indices(21, 8, 3, "center") is None       # span 22 > 21
    assert ex.dense_window_indices(0, 32, 1, "center") is None


def test_bad_window_start_raises() -> None:
    with pytest.raises(ValueError):
        ex.dense_window_indices(100, 32, 1, "random")


# ---- extract_one: dense ------------------------------------------------

def test_extract_one_dense_shape_and_indices(monkeypatch) -> None:
    _patch_frames(monkeypatch, n=469)
    res = ex.extract_one(_FakeLandmarker(), Path("x.mp4"), 16,
                         window=32, stride=1, window_start="center")
    assert res["short"] is False
    assert res["points"].shape == (32, ex.N_FACEMESH_POINTS, 3)
    assert res["valid"].shape == (32,)
    assert res["valid"].all()                       # fake landmarker always detects
    assert res["frame_idx"].tolist() == list(range((469 - 32) // 2, (469 - 32) // 2 + 32))
    assert np.all(np.diff(res["frame_idx"]) == 1)
    assert res["n_frames_decoded"] == 469
    assert float(res["fps"]) == 30.0


def test_extract_one_dense_short_video_is_recorded_not_crashed(monkeypatch) -> None:
    _patch_frames(monkeypatch, n=20)
    res = ex.extract_one(_FakeLandmarker(), Path("x.mp4"), 16, window=32)
    assert res["short"] is True
    assert res["points"] is None and res["frame_idx"] is None
    assert res["n_frames_decoded"] == 20


# ---- extract_one: sparse path unchanged ------------------------------

def test_sample_indices_regression_pins() -> None:
    assert ex.sample_indices(469, 16).tolist() == [
        0, 31, 62, 94, 125, 156, 187, 218, 250, 281, 312, 343, 374, 406, 437, 468]
    assert ex.sample_indices(0, 16).tolist() == []


def test_extract_one_sparse_path_is_unchanged(monkeypatch) -> None:
    _patch_frames(monkeypatch, n=100)
    res = ex.extract_one(_FakeLandmarker(), Path("x.mp4"), 16)     # no window kwarg
    assert res["short"] is False
    exp = ex.sample_indices(100, 16)
    assert res["frame_idx"].tolist() == exp.tolist()
    assert res["points"].shape == (len(exp), ex.N_FACEMESH_POINTS, 3)


# ---- audit reflects dense info + short-video reason ------------------

def test_audit_carries_window_and_short_reason(tmp_path) -> None:
    lm = tmp_path / "landmarks_dense"
    lm.mkdir()
    # one good dense sample
    valid = np.ones(32, bool)
    np.savez_compressed(lm / "good.npz",
                        points=np.zeros((32, 478, 3), np.float32), valid=valid,
                        fps=np.float32(30), frame_idx=np.arange(32))
    (lm / "good.meta.json").write_text(json.dumps({
        "sample_id": "good", "n_frames_decoded": 400, "n_detected": 32,
        "n_requested": 32, "mode": "dense", "window_requested": 32,
    }), encoding="utf-8")
    # one short-video stub: meta present, no .npz
    (lm / "short.meta.json").write_text(json.dumps({
        "sample_id": "short", "n_frames_decoded": 12, "n_detected": 0, "n_requested": 0,
        "mode": "dense", "window_requested": 32,
        "excluded": True, "exclude_reason": "insufficient_decoded_frames",
    }), encoding="utf-8")

    audit = ex.build_cache_audit(lm, ["good", "short"], min_valid_frames=4, window=32)
    assert audit["window"] == 32
    assert audit["n_excluded"] == 1
    (e,) = audit["excluded"]
    assert e["sample_id"] == "short"
    assert e["reason"] == "insufficient_decoded_frames"
    assert e["n_frames_decoded"] == 12
    assert audit["excluded_by_reason"] == {"insufficient_decoded_frames": 1}


def test_audit_window_is_none_for_sparse(tmp_path) -> None:
    lm = tmp_path / "landmarks"
    lm.mkdir()
    np.savez_compressed(lm / "s.npz", points=np.zeros((16, 478, 3), np.float32),
                        valid=np.ones(16, bool), fps=np.float32(30), frame_idx=np.arange(16))
    (lm / "s.meta.json").write_text(json.dumps(
        {"sample_id": "s", "n_frames_decoded": 300, "n_detected": 16}), encoding="utf-8")
    audit = ex.build_cache_audit(lm, ["s"], min_valid_frames=4)   # no window
    assert audit["window"] is None
    assert audit["n_excluded"] == 0
