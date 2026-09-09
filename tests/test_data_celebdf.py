"""Phase 7 unit tests: Celeb-DF protocol parsing, strict identity-disjoint
manifest partition, min-valid-frames exclusion, and the landmark dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import VideoDataConfig
from src.data.celebdf_dataset import (
    CelebDFLandmarkDataset,
    build_celebdf_datasets,
    build_celebdf_manifest,
    identities_of_sample,
    parse_testing_list,
    synthetic_landmark_manifest,
    video_identities,
    video_identity,
    video_sample_id,
)
from src.data.manifests import LeakageError


# --------------------------------------------------------------------- protocol

def test_video_identity_helpers() -> None:
    assert video_identity("Celeb-real/id3_0009.mp4") == "id3"
    assert video_identity("Celeb-synthesis/id0_id1_0003.mp4") == "id1"           # target
    assert video_identities("Celeb-synthesis/id0_id1_0003.mp4") == {"id0", "id1"}  # src + tgt
    assert video_identities("Celeb-real/id3_0009.mp4") == {"id3"}
    assert video_identities("YouTube-real/00170.mp4") == {"yt-00170"}
    assert video_sample_id("Celeb-synthesis/id0_id1_0003.mp4") == "Celeb-synthesis__id0_id1_0003"
    assert identities_of_sample("Celeb-synthesis__id0_id1_0003") == {"id0", "id1"}


def _write_tree(root, real_ids, synth_pairs, n_youtube, testing):
    for d in ("Celeb-real", "Celeb-synthesis", "YouTube-real"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i in real_ids:
        for c in range(2):
            (root / "Celeb-real" / f"id{i}_{c:04d}.mp4").write_bytes(b"")
    for a, b in synth_pairs:
        for c in range(3):
            (root / "Celeb-synthesis" / f"id{a}_id{b}_{c:04d}.mp4").write_bytes(b"")
    for i in range(n_youtube):
        (root / "YouTube-real" / f"{i:05d}.mp4").write_bytes(b"")
    (root / "List_of_testing_videos.txt").write_text(
        "".join(f"{lbl} {p}\n" for lbl, p in testing), encoding="utf-8"
    )


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "celebdf"
    # two disjoint identity groups, each an internally-connected synthesis chain
    groupA = list(range(0, 6))      # id0..id5
    groupB = list(range(10, 16))    # id10..id15
    chainA = [(groupA[k], groupA[k + 1]) for k in range(len(groupA) - 1)]
    chainB = [(groupB[k], groupB[k + 1]) for k in range(len(groupB) - 1)]
    _write_tree(
        root, groupA + groupB, chainA + chainB, n_youtube=20,
        testing=[
            (1, "Celeb-real/id0_0000.mp4"),
            (0, "Celeb-synthesis/id0_id1_0000.mp4"),
            (1, "YouTube-real/00000.mp4"),
        ],
    )
    return root


def test_parse_testing_list(fake_root) -> None:
    entries = parse_testing_list(fake_root / "List_of_testing_videos.txt")
    assert len(entries) == 3
    assert entries[0] == {"rel_path": "Celeb-real/id0_0000.mp4", "label": 1}
    assert entries[1]["label"] == 0


# ------------------------------------------------------- strict identity split

def test_partition_is_strictly_identity_disjoint(fake_root) -> None:
    manifest, report = build_celebdf_manifest(
        fake_root, dev_identity_frac=0.3, seed=1, return_report=True
    )
    assert report["train_dev_identity_overlap"] == 0

    train_ids = {i for r in manifest.split("train") for i in identities_of_sample(r.sample_id)}
    dev_ids = {i for r in manifest.split("dev") for i in identities_of_sample(r.sample_id)}
    assert train_ids.isdisjoint(dev_ids)

    # both splits carry both classes
    for split in ("train", "dev"):
        labels = {r.label for r in manifest.split(split)}
        assert labels == {0, 1}


def test_synthesis_source_identity_cannot_straddle_split(tmp_path) -> None:
    """Every synthesised clip's *source* and *target* identity land in the same
    split - a source id can never leak into the split that holds none of its
    other clips."""

    root = tmp_path / "celebdf"
    # two independent fake-bearing components: A={id1,id2,id3}, B={id11,id12,id13}
    groupA, groupB = [1, 2, 3], [11, 12, 13]
    synth = [(1, 2), (2, 3), (3, 1)] + [(11, 12), (12, 13), (13, 11)]
    _write_tree(root, groupA + groupB, synth, n_youtube=12,
                testing=[(1, "Celeb-real/id1_0000.mp4"),
                         (0, "Celeb-synthesis/id1_id2_0000.mp4")])

    manifest, report = build_celebdf_manifest(root, dev_identity_frac=0.4, seed=3,
                                              return_report=True)
    assert report["train_dev_identity_overlap"] == 0

    split_of: dict[str, str] = {}
    for split in ("train", "dev"):
        for r in manifest.split(split):
            for ident in identities_of_sample(r.sample_id):
                split_of.setdefault(ident, split)
                assert split_of[ident] == split, f"{ident} straddles splits"

    # every synthesised (source, target) pair shares one split, incl. the pool
    # rows whose `speaker` (target) differs from the source
    for r in manifest:
        if r.attack == "celebdf-fs" and r.split != "eval":
            src, tgt = sorted(identities_of_sample(r.sample_id))
            assert split_of[src] == split_of[tgt] == r.split
    # groups A and B are whole and opposite
    assert {split_of["id1"], split_of["id2"], split_of["id3"]} == {split_of["id1"]}
    assert {split_of["id11"], split_of["id12"], split_of["id13"]} == {split_of["id11"]}


def test_partition_is_deterministic(fake_root) -> None:
    a = build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=99)
    b = build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=99)
    assert [(r.sample_id, r.split) for r in a] == [(r.sample_id, r.split) for r in b]


def test_eval_split_is_the_testing_list_verbatim(fake_root) -> None:
    tl = parse_testing_list(fake_root / "List_of_testing_videos.txt")
    manifest = build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=1)
    ev = [r.sample_id for r in manifest.split("eval")]
    assert ev == [video_sample_id(e["rel_path"]) for e in tl]           # same order
    pool_ids = {r.sample_id for r in manifest if r.split != "eval"}
    assert pool_ids.isdisjoint(set(ev))


def test_single_fake_component_is_rejected(tmp_path) -> None:
    root = tmp_path / "celebdf"
    _write_tree(root, [1, 2, 3], [(1, 2), (2, 3), (3, 1)], n_youtube=5,
                testing=[(1, "Celeb-real/id1_0000.mp4"), (0, "Celeb-synthesis/id1_id2_0000.mp4")])
    with pytest.raises(ValueError, match="fake-bearing identity component"):
        build_celebdf_manifest(root, dev_identity_frac=0.3, seed=1)


# --------------------------------------------------- min_valid_frames exclusion

def _write_landmarks(lm_dir, sample_id, n_valid, n_frames=16):
    lm_dir.mkdir(parents=True, exist_ok=True)
    valid = np.zeros(n_frames, bool)
    valid[:n_valid] = True
    np.savez_compressed(lm_dir / f"{sample_id}.npz",
                        points=np.random.default_rng(0).normal(0.2, 0.1, (n_frames, 478, 3)).astype(np.float32),
                        valid=valid, fps=np.float32(25), frame_idx=np.arange(n_frames))


def test_min_valid_frames_excludes_low_quality_videos(fake_root, tmp_path) -> None:
    lm_dir = tmp_path / "lm"
    manifest_full = build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=1)
    # give every video a full-quality cache, then knock two below threshold
    low = {manifest_full[0].sample_id, manifest_full[5].sample_id}
    for r in manifest_full:
        _write_landmarks(lm_dir, r.sample_id, n_valid=(2 if r.sample_id in low else 16))

    manifest, report = build_celebdf_manifest(
        fake_root, dev_identity_frac=0.3, seed=1,
        landmarks_dir=lm_dir, min_valid_frames=4, return_report=True,
    )
    kept = {r.sample_id for r in manifest}
    assert low.isdisjoint(kept)
    assert report["n_excluded"] == 2
    assert {e["sample_id"] for e in report["excluded"]} == low
    assert all(e["reason"] == "insufficient_valid_frames" for e in report["excluded"])
    assert set(report["excluded_by_split"]) <= {"train", "dev", "eval"}


def test_min_valid_frames_keeps_videos_with_occasional_gaps(fake_root, tmp_path) -> None:
    lm_dir = tmp_path / "lm"
    manifest_full = build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=1)
    for r in manifest_full:                    # 10 of 16 valid -> above threshold, kept
        _write_landmarks(lm_dir, r.sample_id, n_valid=10)

    manifest = build_celebdf_manifest(
        fake_root, dev_identity_frac=0.3, seed=1,
        landmarks_dir=lm_dir, min_valid_frames=4,
    )
    assert len(manifest) == len(manifest_full)         # nothing excluded

    # and the gaps are still interpolated at load time (no NaN, finite tensor)
    cfg = VideoDataConfig(landmarks_dir=str(lm_dir), num_frames=8)
    ds = build_celebdf_datasets(manifest, cfg, splits=("train",), landmarks_dir=lm_dir)["train"]
    feat, _ = ds[0]
    assert torch.isfinite(feat).all()


def test_missing_landmarks_raise_without_min_valid_frames(fake_root, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no landmark .npz"):
        build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=1,
                               landmarks_dir=tmp_path / "empty")


def test_missing_landmarks_are_excluded_with_min_valid_frames(fake_root, tmp_path) -> None:
    lm_dir = tmp_path / "lm"
    manifest_full = build_celebdf_manifest(fake_root, dev_identity_frac=0.3, seed=1)
    # cache all but one video
    for r in list(manifest_full)[1:]:
        _write_landmarks(lm_dir, r.sample_id, n_valid=16)
    gone = manifest_full[0].sample_id

    manifest, report = build_celebdf_manifest(
        fake_root, dev_identity_frac=0.3, seed=1,
        landmarks_dir=lm_dir, min_valid_frames=4, return_report=True,
    )
    assert gone not in {r.sample_id for r in manifest}
    assert any(e["sample_id"] == gone and e["reason"] == "missing_landmarks"
               for e in report["excluded"])


# -------------------------------------------------------------------- synthetic

def test_synthetic_landmark_manifest_round_trips(tmp_path) -> None:
    manifest = synthetic_landmark_manifest(
        tmp_path / "lm", n_per_split={"train": 12, "dev": 6, "eval": 6},
        n_ids_per_split=3, num_frames=8, seed=0,
    )
    assert manifest.split_sizes() == {"train": 12, "dev": 6, "eval": 6}
    manifest.assert_splits_speaker_disjoint()
    npz = np.load(tmp_path / "lm" / f"{manifest[0].sample_id}.npz")
    assert npz["points"].shape == (8, 478, 3)
    assert npz["valid"].all()


def test_landmark_dataset_shapes_and_weights(tmp_path) -> None:
    manifest = synthetic_landmark_manifest(
        tmp_path / "lm", n_per_split={"train": 10, "dev": 4, "eval": 4},
        n_ids_per_split=2, num_frames=8, seed=1,
    )
    cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "lm"), num_frames=12, regions="face_mouth")
    datasets = build_celebdf_datasets(manifest, cfg, coords=3, seed=1, landmarks_dir=tmp_path / "lm")
    from src.preprocessing.landmarks import region_point_count

    feat, label = datasets["train"][0]
    assert feat.shape == (12, region_point_count("face_mouth"), 3)
    assert feat.dtype == torch.float32
    assert label in (0, 1)
    w = datasets["train"].label_weights()
    assert w.shape == (2,) and torch.isfinite(w).all()


def test_landmark_dataset_eval_sampling_is_deterministic(tmp_path) -> None:
    manifest = synthetic_landmark_manifest(
        tmp_path / "lm", n_per_split={"eval": 4}, n_ids_per_split=2, num_frames=8, seed=2,
    )
    cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "lm"), num_frames=6)
    ds = build_celebdf_datasets(manifest, cfg, splits=("eval",), landmarks_dir=tmp_path / "lm")["eval"]
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_dense_window_reaches_model_contiguously(tmp_path) -> None:
    """Phase 8 plumbing: with num_frames == cached window length, the dense
    32-frame contiguous window is passed through unchanged for BOTH train
    (random_sample) and eval - frame selection collapses to the identity."""

    from src.preprocessing.landmarks import region_point_count

    manifest = synthetic_landmark_manifest(
        tmp_path / "dense", n_per_split={"train": 6, "dev": 4, "eval": 4},
        n_ids_per_split=2, num_frames=32, seed=5,
    )
    cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "dense"), num_frames=32, regions="face_mouth")
    ds = build_celebdf_datasets(manifest, cfg, splits=("train", "eval"),
                                coords=3, seed=5, landmarks_dir=tmp_path / "dense")
    n = region_point_count("face_mouth")

    tr, _ = ds["train"][0]          # random_sample=True
    ev, _ = ds["eval"][0]           # random_sample=False
    assert tr.shape == ev.shape == (32, n, 3)

    # both must select every frame in order (identity) -> against the raw npz
    import numpy as np
    row = ds["train"].manifest[0]
    raw = np.load(tmp_path / "dense" / f"{row.sample_id}.npz")["points"][:, :, :3]
    from src.preprocessing.landmarks import normalize_landmarks, region_indices
    want = normalize_landmarks(raw, method="interocular", align_rotation=True, coords=3)
    want = want[:, region_indices("face_mouth"), :]
    assert np.allclose(tr.numpy(), want, atol=1e-5)   # no re-subsampling, no reorder


def test_shorter_num_frames_resubsamples_the_window(tmp_path) -> None:
    """Guard the failure mode: num_frames < window length re-subsamples the dense
    cache (random for train, strided for eval) - so Phase 8 must set num_frames=32."""

    manifest = synthetic_landmark_manifest(
        tmp_path / "d", n_per_split={"train": 4, "dev": 2, "eval": 2},
        n_ids_per_split=2, num_frames=32, seed=1,
    )
    cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "d"), num_frames=16)
    ds = build_celebdf_datasets(manifest, cfg, splits=("train", "eval"), landmarks_dir=tmp_path / "d")
    tr, _ = ds["train"][0]
    ev, _ = ds["eval"][0]
    assert tr.shape[0] == ev.shape[0] == 16          # window silently halved


def test_timing_accessor_is_non_breaking(tmp_path) -> None:
    """Phase 9: CelebDFLandmarkDataset.timing(index) exposes per-clip fps /
    frame_idx0 / window_seconds; __getitem__ still returns (features, label)."""

    manifest = synthetic_landmark_manifest(
        tmp_path / "t", n_per_split={"train": 6, "dev": 2, "eval": 2},
        n_ids_per_split=2, num_frames=32, seed=3,           # synth npz: fps=25.0, frame_idx=arange
    )
    cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "t"), num_frames=32)
    ds = build_celebdf_datasets(manifest, cfg, splits=("train",), landmarks_dir=tmp_path / "t")["train"]

    feat, label = ds[0]                                 # signature unchanged
    assert feat.ndim == 3 and label in (0, 1)

    t = ds.timing(0)
    assert set(t) == {"fps", "frame_idx0", "window_seconds"}
    assert t["fps"] == 25.0
    assert t["frame_idx0"] == 0
    assert t["window_seconds"] == 32 / 25.0
    assert ds.timing(1) == ds.timing(1)                 # deterministic
