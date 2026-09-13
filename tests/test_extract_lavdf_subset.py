"""Unit tests for LAV-DF subset extractor (Phase 11 dataset preparation)."""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import pytest

from scripts.extract_lavdf_subset import (
    extract_subset,
    print_statistics,
    read_manifest_paths,
    validate_tar_path,
)

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def sample_manifest(tmp_path) -> Path:
    """Create a sample manifest CSV file."""
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "real_001",
                "path": "LAV-DF/train/real_001.mp4",
                "label": "1",
                "label_name": "real",
                "split": "train",
                "modify_audio": "False",
                "modify_video": "False",
                "n_fakes": "0",
                "fake_periods": "[]",
                "duration": "5.0",
                "original": "",
                "video_frames": "150",
                "audio_frames": "80000",
            }
        )
        writer.writerow(
            {
                "sample_id": "fake_001",
                "path": "LAV-DF/train/fake_001.mp4",
                "label": "0",
                "label_name": "manipulated",
                "split": "train",
                "modify_audio": "True",
                "modify_video": "False",
                "n_fakes": "1",
                "fake_periods": "[[0.5, 1.5]]",
                "duration": "4.0",
                "original": "real_001.mp4",
                "video_frames": "120",
                "audio_frames": "64000",
            }
        )
        writer.writerow(
            {
                "sample_id": "real_002",
                "path": "LAV-DF/dev/real_002.mp4",
                "label": "1",
                "label_name": "real",
                "split": "dev",
                "modify_audio": "False",
                "modify_video": "False",
                "n_fakes": "0",
                "fake_periods": "[]",
                "duration": "4.5",
                "original": "",
                "video_frames": "135",
                "audio_frames": "72000",
            }
        )
    return manifest_path


@pytest.fixture
def sample_tar(tmp_path) -> Path:
    """Create a sample TAR file with LAV-DF structure."""
    tar_path = tmp_path / "LAV-DF.tar"
    with tarfile.open(tar_path, "w") as tar:
        # Add train files
        for name in ["real_001.mp4", "fake_001.mp4"]:
            content = f"video content {name}".encode("utf-8")
            content_io = __import__("io").BytesIO(content)
            tarinfo = tarfile.TarInfo(name=f"LAV-DF/train/{name}")
            tarinfo.size = len(content)
            tar.addfile(tarinfo, content_io)

        # Add dev file
        content = b"video content real_002.mp4"
        content_io = __import__("io").BytesIO(content)
        tarinfo = tarfile.TarInfo(name="LAV-DF/dev/real_002.mp4")
        tarinfo.size = len(content)
        tar.addfile(tarinfo, content_io)

        # Add test file (should not be extracted)
        content = b"video content test_001.mp4"
        content_io = __import__("io").BytesIO(content)
        tarinfo = tarfile.TarInfo(name="LAV-DF/test/test_001.mp4")
        tarinfo.size = len(content)
        tar.addfile(tarinfo, content_io)

    return tar_path


# ------------------------------------------------------------------- manifest reading


def test_read_manifest_paths(sample_manifest) -> None:
    """Test reading paths from manifest CSV."""
    paths = read_manifest_paths(sample_manifest)
    assert len(paths) == 3
    assert "LAV-DF/train/real_001.mp4" in paths
    assert "LAV-DF/train/fake_001.mp4" in paths
    assert "LAV-DF/dev/real_002.mp4" in paths


def test_read_manifest_handles_empty_file(tmp_path) -> None:
    """Test reading from empty manifest."""
    manifest_path = tmp_path / "empty.csv"
    manifest_path.write_text("sample_id,path\n", encoding="utf-8")
    paths = read_manifest_paths(manifest_path)
    assert paths == set()


# --------------------------------------------------------------- path validation


def test_validate_tar_path_accepts_valid_train() -> None:
    """Test that valid train paths are accepted."""
    assert validate_tar_path("LAV-DF/train/file.mp4") == "LAV-DF/train/file.mp4"


def test_validate_tar_path_accepts_valid_dev() -> None:
    """Test that valid dev paths are accepted."""
    assert validate_tar_path("LAV-DF/dev/file.mp4") == "LAV-DF/dev/file.mp4"


def test_validate_tar_path_rejects_test_split() -> None:
    """Test that test split paths are rejected."""
    assert validate_tar_path("LAV-DF/test/file.mp4") is None


def test_validate_tar_path_rejects_path_traversal() -> None:
    """Test that path traversal attempts are rejected."""
    assert validate_tar_path("../etc/passwd") is None
    assert validate_tar_path("LAV-DF/../../../etc/passwd") is None


def test_validate_tar_path_rejects_absolute_path() -> None:
    """Test that absolute paths are rejected."""
    assert validate_tar_path("/absolute/path.mp4") is None
    assert validate_tar_path("LAV-DF//absolute/path.mp4") is None


def test_validate_tar_path_rejects_missing_prefix() -> None:
    """Test that paths without LAV-DF prefix are rejected."""
    assert validate_tar_path("train/file.mp4") is None
    assert validate_tar_path("dev/file.mp4") is None


def test_validate_tar_path_rejects_invalid_split() -> None:
    """Test that invalid split names are rejected."""
    assert validate_tar_path("LAV-DF/invalid/file.mp4") is None
    assert validate_tar_path("LAV-DF/eval/file.mp4") is None


# ----------------------------------------------------------- selective extraction


def test_extract_subset_selective_extraction(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that only files in manifest are extracted."""
    paths = read_manifest_paths(sample_manifest)
    out_dir = tmp_path / "extracted"

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert stats["requested"] == 3
    assert stats["extracted"] == 3
    assert stats["failed"] == 0

    # Check that correct files were extracted
    assert (out_dir / "train" / "real_001.mp4").exists()
    assert (out_dir / "train" / "fake_001.mp4").exists()
    assert (out_dir / "dev" / "real_002.mp4").exists()

    # Check that test file was NOT extracted
    assert not (out_dir / "test" / "test_001.mp4").exists()


def test_extract_subset_preserves_split_directories(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that split directory structure is preserved."""
    paths = read_manifest_paths(sample_manifest)
    out_dir = tmp_path / "extracted"

    extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert (out_dir / "train").is_dir()
    assert (out_dir / "dev").is_dir()


def test_extract_subset_skips_already_present(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that existing non-empty files are skipped."""
    paths = read_manifest_paths(sample_manifest)
    out_dir = tmp_path / "extracted"

    # Create one file in advance
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "real_001.mp4").write_bytes(b"existing content")

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert stats["already_present"] == 1
    assert stats["extracted"] == 2

    # Verify the pre-existing file was not overwritten
    content = (out_dir / "train" / "real_001.mp4").read_bytes()
    assert content == b"existing content"


def test_extract_subset_overwrites_empty_file(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that empty files are overwritten."""
    paths = read_manifest_paths(sample_manifest)
    out_dir = tmp_path / "extracted"

    # Create an empty file in advance
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "real_001.mp4").write_bytes(b"")

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert stats["already_present"] == 0
    assert stats["extracted"] == 3

    # Verify the empty file was overwritten
    content = (out_dir / "train" / "real_001.mp4").read_bytes()
    assert content == b"video content real_001.mp4"


def test_extract_subset_handles_missing_tar_member(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that missing TAR members are reported as failures."""
    # Add a path that doesn't exist in TAR
    paths = read_manifest_paths(sample_manifest)
    paths.add("LAV-DF/train/nonexistent.mp4")

    out_dir = tmp_path / "extracted"

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert stats["requested"] == 4
    assert stats["extracted"] == 3
    assert stats["failed"] == 1
    assert "LAV-DF/train/nonexistent.mp4" in stats["missing_from_tar"]


def test_extract_subset_dry_run(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that dry-run doesn't write files."""
    paths = read_manifest_paths(sample_manifest)
    out_dir = tmp_path / "extracted"

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=True)

    assert stats["extracted"] == 3
    assert stats["total_bytes"] > 0

    # Verify no files were actually written
    assert not (out_dir / "train" / "real_001.mp4").exists()
    assert not (out_dir / "dev" / "real_002.mp4").exists()


def test_extract_subset_rejects_test_split(sample_tar, tmp_path) -> None:
    """Test that test split paths are rejected."""
    paths = {"LAV-DF/test/test_001.mp4"}
    out_dir = tmp_path / "extracted"

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert stats["requested"] == 1
    assert stats["extracted"] == 0
    assert stats["failed"] == 1


def test_extract_subset_rejects_path_traversal(tmp_path) -> None:
    """Test that path traversal attempts are rejected."""
    # Create a TAR with a malicious path
    tar_path = tmp_path / "malicious.tar"
    with tarfile.open(tar_path, "w") as tar:
        content = b"malicious"
        content_io = __import__("io").BytesIO(content)
        tarinfo = tarfile.TarInfo(name="../../../etc/passwd")
        tarinfo.size = len(content)
        tar.addfile(tarinfo, content_io)

    paths = {"../../../etc/passwd"}
    out_dir = tmp_path / "extracted"

    stats = extract_subset(tar_path, paths, out_dir, dry_run=False)

    assert stats["requested"] == 1
    assert stats["extracted"] == 0
    assert stats["failed"] == 1


def test_extract_subset_verifies_non_zero_size(sample_tar, sample_manifest, tmp_path) -> None:
    """Test that extracted files are verified to have non-zero size."""
    paths = read_manifest_paths(sample_manifest)
    out_dir = tmp_path / "extracted"

    stats = extract_subset(sample_tar, paths, out_dir, dry_run=False)

    assert stats["extracted"] == 3

    # Verify all extracted files have non-zero size
    for file_path in out_dir.rglob("*.mp4"):
        assert file_path.stat().st_size > 0


def test_extract_subset_handles_non_regular_files(tmp_path) -> None:
    """Test that non-regular files (directories, symlinks) are skipped."""
    # Create a TAR with a directory
    tar_path = tmp_path / "with_dir.tar"
    with tarfile.open(tar_path, "w") as tar:
        tarinfo = tarfile.TarInfo(name="LAV-DF/train/some_dir")
        tarinfo.type = tarfile.DIRTYPE
        tar.addfile(tarinfo)

    paths = {"LAV-DF/train/some_dir"}
    out_dir = tmp_path / "extracted"

    stats = extract_subset(tar_path, paths, out_dir, dry_run=False)

    assert stats["requested"] == 1
    assert stats["extracted"] == 0
    assert stats["failed"] == 1


def test_extract_subset_train_dev_isolation(sample_tar, tmp_path) -> None:
    """Test that train and dev are properly isolated."""
    train_paths = {"LAV-DF/train/real_001.mp4"}
    dev_paths = {"LAV-DF/dev/real_002.mp4"}
    out_dir = tmp_path / "extracted"

    # Extract train
    stats_train = extract_subset(sample_tar, train_paths, out_dir, dry_run=False)
    assert stats_train["extracted"] == 1

    # Extract dev
    stats_dev = extract_subset(sample_tar, dev_paths, out_dir, dry_run=False)
    assert stats_dev["extracted"] == 1

    # Verify isolation
    assert (out_dir / "train" / "real_001.mp4").exists()
    assert (out_dir / "dev" / "real_002.mp4").exists()
    assert not (out_dir / "train" / "real_002.mp4").exists()
    assert not (out_dir / "dev" / "real_001.mp4").exists()


# ----------------------------------------------------------- statistics reporting


def test_print_statistics(capsys) -> None:
    """Test statistics printing."""
    stats = {
        "requested": 10,
        "extracted": 8,
        "already_present": 1,
        "failed": 1,
        "total_bytes": 1024000,
        "missing_from_tar": ["LAV-DF/train/missing.mp4"],
    }

    print_statistics(stats, dry_run=False)

    captured = capsys.readouterr()
    assert "Requested: 10" in captured.out
    assert "Extracted: 8" in captured.out
    assert "Already present: 1" in captured.out
    assert "Failed: 1" in captured.out
    assert "Total bytes: 1,024,000" in captured.out


def test_print_statistics_dry_run(capsys) -> None:
    """Test statistics printing with dry-run."""
    stats = {
        "requested": 5,
        "extracted": 5,
        "already_present": 0,
        "failed": 0,
        "total_bytes": 512000,
        "missing_from_tar": [],
    }

    print_statistics(stats, dry_run=True)

    captured = capsys.readouterr()
    assert "Dry run: no files extracted" in captured.out


def test_print_statistics_with_missing_files(capsys) -> None:
    """Test statistics printing with missing files."""
    stats = {
        "requested": 10,
        "extracted": 7,
        "already_present": 0,
        "failed": 3,
        "total_bytes": 700000,
        "missing_from_tar": ["LAV-DF/train/a.mp4", "LAV-DF/train/b.mp4", "LAV-DF/train/c.mp4"],
    }

    print_statistics(stats, dry_run=False)

    captured = capsys.readouterr()
    assert "Missing from TAR: 3" in captured.out
    assert "LAV-DF/train/a.mp4" in captured.out
