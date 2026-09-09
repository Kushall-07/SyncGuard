"""Training-time landmark-space augmentation for the visual branch (Phase 7).

:class:`FrameAugment` perturbs a *normalised* full-mesh landmark sequence
``[T, 478, C]`` (as produced by
:func:`src.preprocessing.landmarks.normalize_landmarks`) before the region subset
is selected in :class:`~src.data.celebdf_dataset.CelebDFLandmarkDataset`. It is
the visual analogue of :class:`src.preprocessing.augment.WaveformAugment`: a
per-sample CPU transform plugged in via the dataset's ``frame_transform`` hook,
and a no-op at evaluation (the dataset only attaches it to the ``train`` split).

All operations are geometry-only (mirror, small global scale / rotation, coord
jitter, frame hold) so they stay consistent with the interocular normalisation.
"""

from __future__ import annotations

import numpy as np

from src.config import VideoAugmentConfig
from src.preprocessing.landmarks import hflip_landmarks

__all__ = ["FrameAugment"]


class FrameAugment:
    """Landmark-space augmentation for one normalised sequence ``[T, 478, C]``."""

    def __init__(self, cfg: VideoAugmentConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if not cfg.enabled:
            return points

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 3:
            raise ValueError(f"expected [T, N, C], got {pts.shape}")
        rng = self._rng

        if rng.random() < cfg.hflip_prob:
            pts = hflip_landmarks(pts, full_mesh=True)

        if cfg.scale_jitter > 0:
            s = 1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter)
            pts = pts * np.float32(s)

        if cfg.rot_jitter_deg > 0:
            theta = np.deg2rad(rng.uniform(-cfg.rot_jitter_deg, cfg.rot_jitter_deg))
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            xy = pts[..., :2] @ rot.T
            pts = pts.copy()
            pts[..., :2] = xy

        if cfg.coord_jitter_std > 0 and rng.random() < cfg.coord_jitter_prob:
            pts = pts + rng.normal(0.0, cfg.coord_jitter_std, size=pts.shape).astype(np.float32)

        if cfg.time_mask_frames > 0 and pts.shape[0] > 1 and rng.random() < cfg.time_mask_prob:
            pts = pts.copy()
            k = int(rng.integers(1, cfg.time_mask_frames + 1))
            for _ in range(k):
                t = int(rng.integers(1, pts.shape[0]))
                pts[t] = pts[t - 1]

        return np.ascontiguousarray(pts, dtype=np.float32)
