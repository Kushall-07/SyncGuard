"""Phase 7 unit tests: landmark geometry (normalisation, mirror map, interpolation)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.landmarks import (
    MIRROR_MAP,
    N_FACEMESH_POINTS,
    REGIONS,
    hflip_landmarks,
    interpolate_invalid,
    normalize_landmarks,
    region_indices,
    region_point_count,
)


def _fake_sequence(t: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 40.0, size=(N_FACEMESH_POINTS, 3)).astype(np.float32) + 200.0
    return base[None] + rng.normal(0.0, 1.5, size=(t, N_FACEMESH_POINTS, 3)).astype(np.float32)


def test_regions_are_ordered_and_sized() -> None:
    for name in ("face", "mouth", "face_mouth"):
        idx = region_indices(name)
        assert idx.ndim == 1 and idx.dtype == np.int64
        assert len(idx) == region_point_count(name)
        assert idx.max() < N_FACEMESH_POINTS
    assert region_point_count("face_mouth") == region_point_count("face") + region_point_count("mouth")
    assert len(set(REGIONS["face"])) == len(REGIONS["face"])  # no dupes


def test_normalize_is_translation_and_scale_invariant() -> None:
    seq = _fake_sequence()
    a = normalize_landmarks(seq)
    b = normalize_landmarks(seq * 3.0 + 17.0)
    assert np.abs(a - b).max() < 1e-4


def test_normalize_makes_interocular_distance_unit() -> None:
    from src.preprocessing.landmarks import LEFT_EYE_OUTER, RIGHT_EYE_OUTER

    out = normalize_landmarks(_fake_sequence(), align_rotation=True)
    for frame in out:
        d = np.linalg.norm(frame[LEFT_EYE_OUTER, :2] - frame[RIGHT_EYE_OUTER, :2])
        assert d == pytest.approx(1.0, abs=1e-3)


def test_normalize_none_passthrough_and_coords() -> None:
    seq = _fake_sequence()
    out = normalize_landmarks(seq, method="none", coords=2)
    assert out.shape == (seq.shape[0], N_FACEMESH_POINTS, 2)
    assert np.allclose(out, seq[..., :2])


def test_mirror_map_is_an_involution_and_covers_regions() -> None:
    for idx in set(REGIONS["face_mouth"]):
        assert idx in MIRROR_MAP
        assert MIRROR_MAP[MIRROR_MAP[idx]] == idx


def test_hflip_twice_is_identity() -> None:
    seq = _fake_sequence()
    once = hflip_landmarks(seq)
    twice = hflip_landmarks(once)
    assert np.abs(seq - twice).max() < 1e-5
    perm = [MIRROR_MAP.get(i, i) for i in range(seq.shape[1])]
    assert np.allclose(once[..., 0], -seq[:, perm, 0])


def test_interpolate_invalid_fills_interior_and_edges() -> None:
    seq = np.arange(6, dtype=np.float32).reshape(6, 1, 1).repeat(2, axis=1)
    valid = np.array([False, True, False, False, True, False])
    out = interpolate_invalid(seq, valid)
    assert out[0, 0, 0] == pytest.approx(1.0)      # leading gap -> nearest valid
    assert out[2, 0, 0] == pytest.approx(2.0)      # interior linear between idx 1 and 4
    assert out[3, 0, 0] == pytest.approx(3.0)
    assert out[5, 0, 0] == pytest.approx(4.0)      # trailing gap -> nearest valid


def test_interpolate_invalid_all_invalid_is_noop() -> None:
    seq = _fake_sequence(t=3)
    out = interpolate_invalid(seq, np.zeros(3, dtype=bool))
    assert np.array_equal(seq, out)
