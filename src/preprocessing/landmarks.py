"""MediaPipe FaceMesh landmark geometry for the visual branch (Phase 7).

This module is deliberately free of any ``mediapipe`` import: it only defines the
FaceMesh landmark *index* sets SyncGuard uses, plus the pure-NumPy geometry that
turns a raw ``[T, 478, C]`` landmark sequence into the translation- / scale- /
rotation-normalised sequence the model consumes. Keeping it import-light means the
normalisation can be unit-tested and reused without the heavyweight MediaPipe
runtime (which only the extraction script needs).

Landmark indexing follows MediaPipe FaceMesh with ``refine_landmarks=True``:
468 face-mesh points plus 10 iris points (indices 468-477) = 478 total.

Region sets
-----------
``"face"``       - face oval + eyebrows + eyes + nose bridge/tip (structural pose)
``"mouth"``      - outer and inner lip contours (the sync-relevant articulators)
``"face_mouth"`` - the two concatenated, face points first

The exact point membership is an engineering choice, not a standard; it is
recorded in ``docs/decisions/0002-video-deepfake-detection.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "N_FACEMESH_POINTS",
    "FACE_OVAL",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_BROW",
    "RIGHT_BROW",
    "NOSE",
    "LIPS_OUTER",
    "LIPS_INNER",
    "REGIONS",
    "region_indices",
    "region_point_count",
    "MIRROR_MAP",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_OUTER",
    "NOSE_TIP",
    "normalize_landmarks",
    "interpolate_invalid",
    "hflip_landmarks",
]

# refine_landmarks=True -> 468 mesh points + 10 iris points.
N_FACEMESH_POINTS = 478

# --- MediaPipe FaceMesh canonical index sets (from face_mesh_connections) ------

FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

LEFT_EYE: tuple[int, ...] = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
)
RIGHT_EYE: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
)

LEFT_BROW: tuple[int, ...] = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
RIGHT_BROW: tuple[int, ...] = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)

# Nose bridge (top -> tip) then the two nostril wings.
NOSE: tuple[int, ...] = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358)

LIPS_OUTER: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
)
LIPS_INNER: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
)

# Stable single-point references used by the normaliser.
RIGHT_EYE_OUTER = 33     # subject's right eye, outer canthus
LEFT_EYE_OUTER = 263     # subject's left eye, outer canthus
NOSE_TIP = 1


def _dedup(seq: tuple[int, ...]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in seq:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


_FACE_POINTS = _dedup(FACE_OVAL + LEFT_BROW + RIGHT_BROW + LEFT_EYE + RIGHT_EYE + NOSE)
_MOUTH_POINTS = _dedup(LIPS_OUTER + LIPS_INNER)

REGIONS: dict[str, list[int]] = {
    "face": _FACE_POINTS,
    "mouth": _MOUTH_POINTS,
    "face_mouth": _FACE_POINTS + _MOUTH_POINTS,
}


def region_indices(region: str) -> np.ndarray:
    """The ordered landmark indices for a region name as an ``int64`` array."""

    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; choose from {sorted(REGIONS)}")
    return np.asarray(REGIONS[region], dtype=np.int64)


def region_point_count(region: str) -> int:
    return len(region_indices(region))


# --- left/right mirror map for the horizontal-flip augmentation ----------------
#
# Maps a FaceMesh index to the index of its mirror-image point. Points on (or
# near) the facial midline map to themselves. Coverage is complete for every
# index that appears in any REGIONS entry; other indices default to identity in
# :func:`hflip_landmarks`. Pairs come from MediaPipe's symmetric connection lists
# (the L/R eye rings, brows and lip halves are defined in corresponding order).

def _pairs_from(a: tuple[int, ...], b: tuple[int, ...]) -> list[tuple[int, int]]:
    if len(a) != len(b):
        raise ValueError("mirror pair lists must be equal length")
    return list(zip(a, b))


_MIRROR_PAIRS: list[tuple[int, int]] = []
_MIRROR_PAIRS += _pairs_from(LEFT_EYE, RIGHT_EYE)
_MIRROR_PAIRS += _pairs_from(LEFT_BROW, RIGHT_BROW)
# Face oval: index k mirrors index (len - k) about the vertical (10 = top,
# 152 = chin are the fixed points).
_oval = list(FACE_OVAL)
for k in range(1, len(_oval) // 2 + 1):
    _MIRROR_PAIRS.append((_oval[k], _oval[len(_oval) - k]))
# Lip contours: the 11-point "corner..corner" arc then the 9-point return arc are
# each left-right symmetric about their centre index.
for lip in (LIPS_OUTER, LIPS_INNER):
    arc1, arc2 = lip[:11], lip[11:]
    for k in range(len(arc1) // 2):
        _MIRROR_PAIRS.append((arc1[k], arc1[len(arc1) - 1 - k]))
    for k in range(len(arc2) // 2):
        _MIRROR_PAIRS.append((arc2[k], arc2[len(arc2) - 1 - k]))
# Nose: bridge + tip points are midline (self-mapped); nostril wings pair up.
_MIRROR_PAIRS += [(98, 327), (129, 358)]

MIRROR_MAP: dict[int, int] = {}
for _l, _r in _MIRROR_PAIRS:
    MIRROR_MAP[_l] = _r
    MIRROR_MAP[_r] = _l
for _idx in set(REGIONS["face_mouth"]):
    MIRROR_MAP.setdefault(_idx, _idx)  # midline / unpaired -> itself


# --- geometry ----------------------------------------------------------------

def _interocular(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return ``(centre, scale, angle)`` for one ``[478, C]`` frame.

    ``centre`` is the nose-tip point, ``scale`` the distance between the two eye
    outer corners, ``angle`` the roll of the eye line (radians).
    """

    right = frame[RIGHT_EYE_OUTER, :2]
    left = frame[LEFT_EYE_OUTER, :2]
    centre = frame[NOSE_TIP].copy()
    delta = left - right
    scale = float(np.hypot(delta[0], delta[1]))
    angle = float(np.arctan2(delta[1], delta[0]))
    return centre, scale, angle


def normalize_landmarks(
    points: np.ndarray,
    *,
    method: str = "interocular",
    align_rotation: bool = True,
    coords: int = 3,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalise a raw landmark sequence to a canonical pose.

    Parameters
    ----------
    points
        ``[T, N, C]`` array of raw FaceMesh coordinates (C = 2 or 3), in image /
        normalised-image units. ``N`` may be the full 478 or a region subset, but
        ``method="interocular"`` needs the full mesh (it indexes eye/nose points).
    method
        ``"interocular"`` - per-frame: translate the nose tip to the origin,
        rotate the eye line horizontal (if ``align_rotation``), scale so the
        inter-ocular distance is 1. ``"none"`` - return ``points[..., :coords]``
        unchanged.
    coords
        Number of coordinate channels to keep in the output (2 or 3).

    Returns
    -------
    ``[T, N, coords]`` float32, normalised.
    """

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected [T, N, C], got {arr.shape}")
    if coords not in (2, 3):
        raise ValueError("coords must be 2 or 3")
    if method == "none":
        return np.ascontiguousarray(arr[..., :coords])
    if method != "interocular":
        raise ValueError(f"unknown method {method!r}")
    if arr.shape[1] < N_FACEMESH_POINTS:
        raise ValueError(
            "interocular normalisation needs the full 478-point mesh; "
            f"got N={arr.shape[1]}"
        )

    out = np.empty((arr.shape[0], arr.shape[1], coords), dtype=np.float32)
    for t in range(arr.shape[0]):
        frame = arr[t]
        centre, scale, angle = _interocular(frame)
        xy = frame[:, :2] - centre[:2]
        if align_rotation:
            c, s = np.cos(-angle), np.sin(-angle)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            xy = xy @ rot.T
        xy = xy / max(scale, eps)
        out[t, :, 0:2] = xy
        if coords == 3:
            out[t, :, 2] = (frame[:, 2] - frame[NOSE_TIP, 2]) / max(scale, eps)
    return out


def interpolate_invalid(seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill frames marked invalid in a ``[T, ...]`` sequence.

    ``valid`` is a ``[T]`` boolean mask. Interior gaps are linearly interpolated
    between the nearest valid frames; leading / trailing gaps take the nearest
    valid frame. If no frame is valid the sequence is returned unchanged.
    """

    seq = np.asarray(seq, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if valid.shape[0] != seq.shape[0]:
        raise ValueError("valid mask length must match sequence length")
    idx = np.nonzero(valid)[0]
    if idx.size == 0 or idx.size == seq.shape[0]:
        return seq

    out = seq.copy()
    flat = out.reshape(out.shape[0], -1)
    src = flat[idx]
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(np.arange(flat.shape[0]), idx, src[:, j])
    return flat.reshape(seq.shape)


def hflip_landmarks(points: np.ndarray, *, full_mesh: bool = True) -> np.ndarray:
    """Mirror a ``[T, 478, C]`` sequence left<->right.

    Negates the x channel and swaps each point with its :data:`MIRROR_MAP`
    partner (points without a mapped partner keep their slot). Intended for the
    full 478-point mesh *before* a region subset is selected.
    """

    arr = np.asarray(points, dtype=np.float32).copy()
    if arr.ndim != 3:
        raise ValueError(f"expected [T, N, C], got {arr.shape}")
    if full_mesh and arr.shape[1] != N_FACEMESH_POINTS:
        raise ValueError(f"expected the full {N_FACEMESH_POINTS}-point mesh")

    perm = np.arange(arr.shape[1])
    for i in range(arr.shape[1]):
        perm[i] = MIRROR_MAP.get(int(i), i)
    arr = arr[:, perm, :]
    arr[..., 0] = -arr[..., 0]
    return arr
