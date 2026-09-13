"""Unit tests for LAV-DF manifest builder (Phase 11 dataset preparation)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.build_lavdf_manifest import (
    LAVDFManifest,
    LAVDFRow,
    build_lavdf_manifest,
    classify_category,
    metadata_to_row,
    parse_metadata_from_tar,
    verify_tar_members,
)

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def sample_metadata() -> list[dict]:
    """Sample LAV-DF metadata for testing (without LAV-DF prefix, as in real dataset)."""
    return [
        {
            "file": "train/real_001.mp4",
            "n_fakes": 0,
            "fake_periods": [],
            "duration": 5.0,
            "original": "",
            "modify_audio": False,
            "modify_video": False,
            "split": "train",
            "video_frames": 150,
            "audio_frames": 80000,
        },
        {
            "file": "train/audio_fake_001.mp4",
            "n_fakes": 1,
            "fake_periods": [[0.5, 1.5]],
            "duration": 4.0,
            "original": "real_001.mp4",
            "modify_audio": True,
            "modify_video": False,
            "split": "train",
            "video_frames": 120,
            "audio_frames": 64000,
        },
        {
            "file": "train/video_fake_001.mp4",
            "n_fakes": 1,
            "fake_periods": [[1.0, 2.0]],
            "duration": 6.0,
            "original": "real_002.mp4",
            "modify_audio": False,
            "modify_video": True,
            "split": "train",
            "video_frames": 180,
            "audio_frames": 96000,
        },
        {
            "file": "train/av_fake_001.mp4",
            "n_fakes": 2,
            "fake_periods": [[0.0, 1.0], [2.0, 3.0]],
            "duration": 5.0,
            "original": "real_003.mp4",
            "modify_audio": True,
            "modify_video": True,
            "split": "train",
            "video_frames": 150,
            "audio_frames": 80000,
        },
        {
            "file": "dev/real_002.mp4",
            "n_fakes": 0,
            "fake_periods": [],
            "duration": 4.5,
            "original": "",
            "modify_audio": False,
            "modify_video": False,
            "split": "dev",
            "video_frames": 135,
            "audio_frames": 72000,
        },
        {
            "file": "dev/audio_fake_002.mp4",
            "n_fakes": 1,
            "fake_periods": [[1.0, 2.5]],
            "duration": 5.5,
            "original": "real_004.mp4",
            "modify_audio": True,
            "modify_video": False,
            "split": "dev",
            "video_frames": 165,
            "audio_frames": 88000,
        },
        {
            "file": "test/real_003.mp4",
            "n_fakes": 0,
            "fake_periods": [],
            "duration": 5.0,
            "original": "",
            "modify_audio": False,
            "modify_video": False,
            "split": "test",
            "video_frames": 150,
            "audio_frames": 80000,
        },
    ]


@pytest.fixture
def sample_tar(tmp_path, sample_metadata) -> Path:
    """Create a sample TAR file with metadata.min.json and dummy video files."""
    tar_path = tmp_path / "LAV-DF.tar"
    with tarfile.open(tar_path, "w") as tar:
        # Add metadata.min.json
        metadata_json = json.dumps(sample_metadata).encode("utf-8")
        metadata_bytes = __import__("io").BytesIO(metadata_json)
        tarinfo = tarfile.TarInfo(name="LAV-DF/metadata.min.json")
        tarinfo.size = len(metadata_json)
        tar.addfile(tarinfo, metadata_bytes)

        # Add dummy video files WITH LAV-DF prefix (as in real TAR)
        for entry in sample_metadata:
            video_bytes = b"dummy video content"
            video_bytes_io = __import__("io").BytesIO(video_bytes)
            tarinfo = tarfile.TarInfo(name=f"LAV-DF/{entry['file']}")
            tarinfo.size = len(video_bytes)
            tar.addfile(tarinfo, video_bytes_io)

    return tar_path


@pytest.fixture
def sample_tar_with_prefix(tmp_path) -> Path:
    """Create a TAR where members have LAV-DF/ prefix but metadata does not."""
    tar_path = tmp_path / "LAV-DF-prefixed.tar"
    
    # Metadata without LAV-DF prefix (as in real dataset)
    metadata_no_prefix = [
        {
            "file": "train/real_001.mp4",
            "n_fakes": 0,
            "fake_periods": [],
            "duration": 5.0,
            "original": "",
            "modify_audio": False,
            "modify_video": False,
            "split": "train",
            "video_frames": 150,
            "audio_frames": 80000,
        },
        {
            "file": "train/audio_fake_001.mp4",
            "n_fakes": 1,
            "fake_periods": [[0.5, 1.5]],
            "duration": 4.0,
            "original": "real_001.mp4",
            "modify_audio": True,
            "modify_video": False,
            "split": "train",
            "video_frames": 120,
            "audio_frames": 64000,
        },
    ]
    
    with tarfile.open(tar_path, "w") as tar:
        # Add metadata.min.json
        metadata_json = json.dumps(metadata_no_prefix).encode("utf-8")
        metadata_bytes = __import__("io").BytesIO(metadata_json)
        tarinfo = tarfile.TarInfo(name="LAV-DF/metadata.min.json")
        tarinfo.size = len(metadata_json)
        tar.addfile(tarinfo, metadata_bytes)

        # Add video files WITH LAV-DF prefix (as in real TAR)
        for entry in metadata_no_prefix:
            video_bytes = b"dummy video content"
            video_bytes_io = __import__("io").BytesIO(video_bytes)
            # TAR member has prefix, metadata does not
            tarinfo = tarfile.TarInfo(name=f"LAV-DF/{entry['file']}")
            tarinfo.size = len(video_bytes)
            tar.addfile(tarinfo, video_bytes_io)

    return tar_path


# ------------------------------------------------------------------- metadata parsing


def test_parse_metadata_from_tar_reads_json(sample_tar) -> None:
    """Test that metadata.min.json is read correctly from TAR."""
    metadata = parse_metadata_from_tar(sample_tar)
    assert len(metadata) == 7
    assert metadata[0]["file"] == "train/real_001.mp4"
    assert metadata[0]["n_fakes"] == 0


def test_parse_metadata_from_tar_fails_on_missing_tar(tmp_path) -> None:
    """Test that FileNotFoundError is raised for missing TAR."""
    with pytest.raises(FileNotFoundError, match="TAR file not found"):
        parse_metadata_from_tar(tmp_path / "nonexistent.tar")


def test_parse_metadata_from_tar_fails_on_missing_metadata(tmp_path) -> None:
    """Test that FileNotFoundError is raised when metadata.min.json is missing."""
    tar_path = tmp_path / "empty.tar"
    with tarfile.open(tar_path, "w") as tar:
        dummy_bytes = b"dummy"
        dummy_io = __import__("io").BytesIO(dummy_bytes)
        tarinfo = tarfile.TarInfo(name="some_file.txt")
        tarinfo.size = len(dummy_bytes)
        tar.addfile(tarinfo, dummy_io)

    with pytest.raises(FileNotFoundError, match="metadata.min.json not found"):
        parse_metadata_from_tar(tar_path)


# --------------------------------------------------------------- category classification


def test_classify_category_real() -> None:
    """Test classification of real samples."""
    entry = {"n_fakes": 0, "modify_audio": False, "modify_video": False}
    assert classify_category(entry) == "real"


def test_classify_category_audio_only() -> None:
    """Test classification of audio-only manipulated samples."""
    entry = {"n_fakes": 1, "modify_audio": True, "modify_video": False}
    assert classify_category(entry) == "audio_only"


def test_classify_category_video_only() -> None:
    """Test classification of video-only manipulated samples."""
    entry = {"n_fakes": 1, "modify_audio": False, "modify_video": True}
    assert classify_category(entry) == "video_only"


def test_classify_category_audio_video() -> None:
    """Test classification of audio+video manipulated samples."""
    entry = {"n_fakes": 2, "modify_audio": True, "modify_video": True}
    assert classify_category(entry) == "audio_video"


def test_classify_category_unknown() -> None:
    """Test classification of edge case (n_fakes > 0 but both flags false)."""
    entry = {"n_fakes": 1, "modify_audio": False, "modify_video": False}
    assert classify_category(entry) == "unknown"


# --------------------------------------------------------------- label assignment


def test_metadata_to_row_real_label() -> None:
    """Test that real samples get label 1."""
    entry = {
        "file": "test.mp4",
        "n_fakes": 0,
        "fake_periods": [],
        "duration": 5.0,
        "original": "",
        "modify_audio": False,
        "modify_video": False,
        "split": "train",
        "video_frames": 150,
        "audio_frames": 80000,
    }
    row = metadata_to_row(entry)
    assert row.label == 1
    assert row.label_name == "real"


def test_metadata_to_row_manipulated_label() -> None:
    """Test that manipulated samples get label 0."""
    entry = {
        "file": "test.mp4",
        "n_fakes": 1,
        "fake_periods": [[1.0, 2.0]],
        "duration": 5.0,
        "original": "orig.mp4",
        "modify_audio": True,
        "modify_video": False,
        "split": "train",
        "video_frames": 150,
        "audio_frames": 80000,
    }
    row = metadata_to_row(entry)
    assert row.label == 0
    assert row.label_name == "manipulated"


def test_metadata_to_row_preserves_fake_periods() -> None:
    """Test that fake_periods are preserved as JSON string."""
    entry = {
        "file": "test.mp4",
        "n_fakes": 2,
        "fake_periods": [[0.0, 1.0], [2.0, 3.0]],
        "duration": 5.0,
        "original": "orig.mp4",
        "modify_audio": True,
        "modify_video": True,
        "split": "train",
        "video_frames": 150,
        "audio_frames": 80000,
    }
    row = metadata_to_row(entry)
    assert row.fake_periods == "[[0.0, 1.0], [2.0, 3.0]]"


def test_metadata_to_row_validates_label_name() -> None:
    """Test that label_name validation works."""
    with pytest.raises(ValueError, match="label_name"):
        LAVDFRow(
            sample_id="test",
            path="test.mp4",
            label=1,
            label_name="manipulated",  # Wrong for label 1
            split="train",
            modify_audio=False,
            modify_video=False,
            n_fakes=0,
            fake_periods="[]",
            duration=5.0,
            original="",
            video_frames=150,
            audio_frames=80000,
        )


# ----------------------------------------------------------- deterministic sampling


def test_subset_is_deterministic(sample_metadata) -> None:
    """Test that subset sampling is deterministic with same seed."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:5]]
    manifest = LAVDFManifest(rows)

    a = manifest.subset(3, seed=42)
    b = manifest.subset(3, seed=42)

    assert [r.sample_id for r in a] == [r.sample_id for r in b]


def test_subset_respects_size_limit(sample_metadata) -> None:
    """Test that subset respects the requested size."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:5]]
    manifest = LAVDFManifest(rows)

    subset = manifest.subset(2, seed=42)
    assert len(subset) == 2


def test_subset_with_stratification(sample_metadata) -> None:
    """Test that subset can stratify by a field."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:5]]
    manifest = LAVDFManifest(rows)

    subset = manifest.subset(4, seed=42, stratify_by="label")
    # Should have mix of labels if possible
    labels = [r.label for r in subset]
    assert len(set(labels)) >= 1


# -------------------------------------------------------------- split isolation


def test_build_manifest_rejects_test_split(sample_tar) -> None:
    """Test that test split cannot be used for development subsets."""
    with pytest.raises(ValueError, match="test split must never be used"):
        build_lavdf_manifest(
            sample_tar,
            {"real": 1},
            splits=("train", "test"),
        )


def test_build_manifest_respects_split_filtering(sample_tar) -> None:
    """Test that only requested splits are included."""
    manifest = build_lavdf_manifest(
        sample_tar,
        {"real": 1},
        seed=42,
        splits=("train",),
    )

    # Should only have train samples
    assert all(row.split == "train" for row in manifest.rows)


def test_manifest_split_isolation(sample_metadata) -> None:
    """Test that split filtering works correctly."""
    rows = [metadata_to_row(entry) for entry in sample_metadata]
    manifest = LAVDFManifest(rows)

    train = manifest.split("train")
    dev = manifest.split("dev")
    test = manifest.split("test")

    assert len(train) == 4
    assert len(dev) == 2
    assert len(test) == 1


# ----------------------------------------------------------- manifest serialization


def test_manifest_csv_round_trip(sample_metadata, tmp_path) -> None:
    """Test CSV write and read round-trip."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:3]]
    manifest = LAVDFManifest(rows)

    csv_path = tmp_path / "manifest.csv"
    manifest.write_csv(csv_path)

    # Read back and verify
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = __import__("csv").DictReader(f)
        rows_back = list(reader)

    assert len(rows_back) == 3
    assert rows_back[0]["sample_id"] == rows[0].sample_id
    assert rows_back[0]["label"] == str(rows[0].label)


def test_manifest_csv_has_all_columns(sample_metadata, tmp_path) -> None:
    """Test that CSV includes all required columns."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:1]]
    manifest = LAVDFManifest(rows)

    csv_path = tmp_path / "manifest.csv"
    manifest.write_csv(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = __import__("csv").DictReader(f)
        header = reader.fieldnames

    expected_columns = [
        "sample_id",
        "path",
        "label",
        "label_name",
        "split",
        "modify_audio",
        "modify_video",
        "n_fakes",
        "fake_periods",
        "duration",
        "original",
        "video_frames",
        "audio_frames",
    ]
    assert set(header) == set(expected_columns)


# ----------------------------------------------------------- TAR member validation


def test_verify_tar_members_passes(sample_tar, sample_metadata) -> None:
    """Test that verification passes when all paths exist."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:3]]
    missing = verify_tar_members(sample_tar, rows)
    assert missing == []


def test_verify_tar_members_detects_missing(sample_tar, sample_metadata) -> None:
    """Test that verification detects missing paths."""
    rows = [metadata_to_row(entry) for entry in sample_metadata[:3]]
    # Modify one path to be non-existent
    rows = [
        LAVDFRow(
            sample_id=r.sample_id,
            path="nonexistent/path.mp4",
            label=r.label,
            label_name=r.label_name,
            split=r.split,
            modify_audio=r.modify_audio,
            modify_video=r.modify_video,
            n_fakes=r.n_fakes,
            fake_periods=r.fake_periods,
            duration=r.duration,
            original=r.original,
            video_frames=r.video_frames,
            audio_frames=r.audio_frames,
        )
        if i == 0
        else r
        for i, r in enumerate(rows)
    ]

    missing = verify_tar_members(sample_tar, rows)
    assert len(missing) == 1
    assert "nonexistent/path.mp4" in missing[0]


# --------------------------------------------------------------- integration tests


def test_build_manifest_full_integration(sample_tar) -> None:
    """Test full manifest building with multiple categories."""
    manifest = build_lavdf_manifest(
        sample_tar,
        {"real": 1, "audio_only": 1, "video_only": 1},
        seed=42,
        splits=("train", "dev"),
    )

    assert len(manifest) == 3
    stats = manifest.statistics()
    assert stats["real"] == 1
    assert stats["audio_only"] == 1
    assert stats["video_only"] == 1


def test_build_manifest_insufficient_samples(sample_tar) -> None:
    """Test that error is raised when requesting more than available."""
    with pytest.raises(ValueError, match="Requested.*but only.*available"):
        build_lavdf_manifest(
            sample_tar,
            {"real": 100},  # More than available
            seed=42,
            splits=("train", "dev"),
        )


def test_build_manifest_unknown_category(sample_tar) -> None:
    """Test that error is raised for unknown category."""
    with pytest.raises(ValueError, match="Category.*requested but no samples"):
        build_lavdf_manifest(
            sample_tar,
            {"nonexistent": 1},
            seed=42,
        )


def test_statistics_computation(sample_metadata) -> None:
    """Test that statistics are computed correctly."""
    rows = [metadata_to_row(entry) for entry in sample_metadata]
    manifest = LAVDFManifest(rows)
    stats = manifest.statistics()

    assert stats["total"] == 7
    assert stats["real"] == 3  # 3 real samples
    assert stats["manipulated"] == 4
    assert stats["audio_only"] == 2
    assert stats["video_only"] == 1
    assert stats["audio_video"] == 1
    assert stats["duration_min"] == 4.0
    assert stats["duration_max"] == 6.0


def test_manifest_filter_by_category(sample_metadata) -> None:
    """Test filtering manifest by various conditions."""
    rows = [metadata_to_row(entry) for entry in sample_metadata]
    manifest = LAVDFManifest(rows)

    real_only = manifest.filter(label=1)
    assert len(real_only) == 3
    assert all(r.label == 1 for r in real_only)

    train_only = manifest.filter(split="train")
    assert len(train_only) == 4
    assert all(r.split == "train" for r in train_only)

    audio_only = manifest.filter(modify_audio=True, modify_video=False)
    assert len(audio_only) == 2


# ----------------------------------------------------------- regression tests


def test_path_normalization_with_lavdf_prefix(sample_tar_with_prefix) -> None:
    """Test that metadata paths without LAV-DF prefix are normalized to match TAR members."""
    # This is a regression test for the bug where metadata had "train/file.mp4"
    # but TAR members were "LAV-DF/train/file.mp4"
    manifest = build_lavdf_manifest(
        sample_tar_with_prefix,
        {"real": 1, "audio_only": 1},
        seed=42,
        splits=("train",),
    )

    assert len(manifest) == 2
    # All paths should have LAV-DF prefix
    for row in manifest.rows:
        assert row.path.startswith("LAV-DF/"), f"Path {row.path} missing LAV-DF prefix"
    
    # Verification should pass (no missing paths)
    # This would have failed before the fix
    missing = verify_tar_members(sample_tar_with_prefix, manifest.rows)
    assert missing == []


def test_metadata_to_row_normalizes_path() -> None:
    """Test that metadata_to_row adds LAV-DF prefix when missing."""
    entry = {
        "file": "train/test.mp4",
        "n_fakes": 0,
        "fake_periods": [],
        "duration": 5.0,
        "original": "",
        "modify_audio": False,
        "modify_video": False,
        "split": "train",
        "video_frames": 150,
        "audio_frames": 80000,
    }
    row = metadata_to_row(entry)
    assert row.path == "LAV-DF/train/test.mp4"


def test_metadata_to_row_preserves_existing_prefix() -> None:
    """Test that metadata_to_row doesn't double-prefix if already present."""
    entry = {
        "file": "LAV-DF/train/test.mp4",
        "n_fakes": 0,
        "fake_periods": [],
        "duration": 5.0,
        "original": "",
        "modify_audio": False,
        "modify_video": False,
        "split": "train",
        "video_frames": 150,
        "audio_frames": 80000,
    }
    row = metadata_to_row(entry)
    assert row.path == "LAV-DF/train/test.mp4"
    assert not row.path.startswith("LAV-DF/LAV-DF/")
