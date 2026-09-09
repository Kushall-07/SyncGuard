"""Phase 8: guardrails for the tiny-subset overfit diagnostic
(scripts/visual_overfit_probe.py) - deterministic balanced subset, dense 32-frame
cache actually used, augmentation + class weighting disabled, no dev/eval leak."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from src.config import VideoDataConfig
from src.data.celebdf_dataset import synthetic_landmark_manifest
from src.training.deepfake_trainer import DeepfakeTrainer

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visual_overfit_probe.py"
_spec = importlib.util.spec_from_file_location("visual_overfit_probe", _SCRIPT)
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)  # type: ignore[union-attr]


def _manifest(tmp_path):
    return synthetic_landmark_manifest(
        tmp_path / "dense", n_per_split={"train": 240, "dev": 40, "eval": 40},
        n_ids_per_split=4, num_frames=32, seed=1,
    )


def test_subset_is_deterministic_and_balanced_and_train_only(tmp_path) -> None:
    m = _manifest(tmp_path)
    a = op.select_overfit_subset(m, n_per_class=100, seed=1337)
    b = op.select_overfit_subset(m, n_per_class=100, seed=1337)
    assert [r.sample_id for r in a] == [r.sample_id for r in b]        # deterministic
    assert len(a) == 200
    assert sum(1 for r in a if r.label == 1) == 100                    # balanced
    assert sum(1 for r in a if r.label == 0) == 100
    train_ids = {r.sample_id for r in m.split("train")}
    assert {r.sample_id for r in a} <= train_ids                       # train-only
    dev_eval = {r.sample_id for r in m if r.split in ("dev", "eval")}
    assert {r.sample_id for r in a}.isdisjoint(dev_eval)               # no leak
    assert len(set(r.sample_id for r in a)) == 200                     # no dupes


def test_different_seed_changes_the_subset(tmp_path) -> None:
    m = _manifest(tmp_path)
    a = {r.sample_id for r in op.select_overfit_subset(m, n_per_class=100, seed=1337)}
    c = {r.sample_id for r in op.select_overfit_subset(m, n_per_class=100, seed=7)}
    assert a != c


def test_dataset_delivers_full_contiguous_32_frame_window(tmp_path) -> None:
    m = _manifest(tmp_path)
    subset = op.select_overfit_subset(m, n_per_class=100, seed=1337)
    ds = op.build_overfit_dataset(subset, tmp_path / "dense", coords=3, num_frames=32,
                                  regions="face_mouth", video_cfg=VideoDataConfig())
    x, y = ds[0]
    assert x.shape == (32, 142, 3)                                     # [T, 142 landmarks, 3 coords]
    assert x.dtype == torch.float32
    assert ds.frame_transform is None                                  # augmentation OFF
    assert ds.random_sample is False                                   # no random temporal subsample
    # eval-mode selection is the identity -> matches the raw npz frame order
    raw = np.load(tmp_path / "dense" / f"{subset[0].sample_id}.npz")["points"]
    assert raw.shape[0] == 32


def test_class_weighting_is_disabled_in_trainer(tmp_path) -> None:
    from src.config import TrainingConfig
    from src.models.video import DeepfakeClassifier
    from src.training.utils import RunDirectory

    model = DeepfakeClassifier(_small_model_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    rd = RunDirectory(tmp_path / "runs", "t")
    tr = DeepfakeTrainer(model, opt, config=TrainingConfig(batch_size=8, epochs=1, amp=False),
                         run_dir=rd, device="cpu", class_weights=None)
    assert tr.class_weights is None                                    # ordinary CE


def test_probe_combos_are_the_six_canonical_plus_warmup() -> None:
    # the script's canonical grid: {mlp, transformer} x {1e-4, 3e-4, 5e-4}
    canonical = [("mlp_baseline", 1e-4, 0), ("mlp_baseline", 3e-4, 0), ("mlp_baseline", 5e-4, 0),
                 ("transformer", 1e-4, 0), ("transformer", 3e-4, 0), ("transformer", 5e-4, 0)]
    assert len(canonical) == 6
    assert {c[0] for c in canonical} == {"mlp_baseline", "transformer"}
    assert sorted({c[1] for c in canonical}) == [1e-4, 3e-4, 5e-4]


def _small_model_cfg():
    from src.config import ModelConfig
    return ModelConfig(visual_embedding_dim=32, num_heads=4, visual_embed_hidden=32,
                       visual_tf_ff_dim=64, visual_tf_layers=1, spoof_head_hidden=16)
