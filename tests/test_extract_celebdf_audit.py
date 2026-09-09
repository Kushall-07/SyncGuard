"""Phase 7: extraction_audit.json is a pure function of the current landmark
cache and the current --min-valid-frames threshold (no cross-run/cross-threshold
contamination)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "extract_celebdf_landmarks.py"

_spec = importlib.util.spec_from_file_location("extract_celebdf_landmarks", _SCRIPT)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)  # type: ignore[union-attr]


def _write_cache(lm_dir: Path, sample_id: str, *, n_valid: int, n_frames: int,
                 n_frames_decoded: int | None, meta_min_valid: int = 4,
                 meta_excluded: bool = False) -> None:
    lm_dir.mkdir(parents=True, exist_ok=True)
    valid = np.zeros(n_frames, dtype=bool)
    valid[:n_valid] = True
    np.savez_compressed(
        lm_dir / f"{sample_id}.npz",
        points=np.zeros((n_frames, 478, 3), dtype=np.float32),
        valid=valid, fps=np.float32(30), frame_idx=np.arange(n_frames),
    )
    meta = {"sample_id": sample_id, "n_requested": n_frames, "n_detected": n_valid,
            "detection_rate": (n_valid / n_frames) if n_frames else 0.0,
            "fps": 30.0, "min_valid_frames": meta_min_valid,
            "excluded": meta_excluded, "exclude_reason": None}
    if n_frames_decoded is not None:
        meta["n_frames_decoded"] = n_frames_decoded
    (lm_dir / f"{sample_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------- pure-function audit

def test_stale_threshold8_audit_does_not_contaminate_threshold4(tmp_path) -> None:
    lm = tmp_path / "lm"
    # id0_0000-like: 7 valid frames, healthy source
    _write_cache(lm, "Celeb-real__id0_0000", n_valid=7, n_frames=16,
                 n_frames_decoded=469, meta_min_valid=8, meta_excluded=True)
    # a genuinely bad video
    _write_cache(lm, "Celeb-real__idbad", n_valid=1, n_frames=16, n_frames_decoded=300)

    # a leftover audit from a previous --min-valid-frames 8 run
    (lm / "extraction_audit.json").write_text(json.dumps({
        "min_valid_frames": 8, "n_excluded": 1,
        "excluded": [{"sample_id": "Celeb-real__id0_0000", "n_detected": 7,
                      "reason": "insufficient_valid_frames"}],
    }), encoding="utf-8")

    audit = ex.build_cache_audit(lm, ["Celeb-real__id0_0000", "Celeb-real__idbad"],
                                 min_valid_frames=4)
    ids = {e["sample_id"] for e in audit["excluded"]}
    assert "Celeb-real__id0_0000" not in ids          # 7 >= 4 -> kept, stale entry gone
    assert ids == {"Celeb-real__idbad"}
    assert audit["n_excluded"] == 1


def test_seven_valid_frames_retained_at_threshold_4(tmp_path) -> None:
    lm = tmp_path / "lm"
    _write_cache(lm, "s", n_valid=7, n_frames=16, n_frames_decoded=469)
    audit = ex.build_cache_audit(lm, ["s"], min_valid_frames=4)
    assert audit["n_excluded"] == 0
    assert audit["excluded"] == []


def test_seven_valid_frames_excluded_at_threshold_8(tmp_path) -> None:
    lm = tmp_path / "lm"
    _write_cache(lm, "s", n_valid=7, n_frames=16, n_frames_decoded=469)
    audit = ex.build_cache_audit(lm, ["s"], min_valid_frames=8)
    assert audit["n_excluded"] == 1
    (e,) = audit["excluded"]
    assert e["sample_id"] == "s" and e["n_detected"] == 7
    assert e["reason"] == "face_not_detected"          # decoded 469 frames, faces just sparse
    assert e["n_frames_decoded"] == 469


def test_audit_count_equals_list_length(tmp_path) -> None:
    lm = tmp_path / "lm"
    _write_cache(lm, "keep", n_valid=10, n_frames=16, n_frames_decoded=300)
    _write_cache(lm, "drop1", n_valid=0, n_frames=16, n_frames_decoded=300)
    _write_cache(lm, "drop2", n_valid=2, n_frames=16, n_frames_decoded=300)
    audit = ex.build_cache_audit(lm, ["keep", "drop1", "drop2"], min_valid_frames=4)
    assert audit["n_excluded"] == len(audit["excluded"]) == 2


def test_truncated_one_frame_source_excluded_as_corrupt(tmp_path) -> None:
    lm = tmp_path / "lm"
    # id27_0005-like: source decoded to a single frame, no face
    _write_cache(lm, "Celeb-real__id27_0005", n_valid=0, n_frames=1, n_frames_decoded=1)
    audit = ex.build_cache_audit(lm, ["Celeb-real__id27_0005"], min_valid_frames=4)
    assert audit["n_excluded"] == 1
    (e,) = audit["excluded"]
    assert e["reason"] == "corrupt_or_truncated_source"
    assert e["n_detected"] == 0 and e["n_frames_decoded"] == 1


def test_missing_npz_is_reported_as_missing_landmarks(tmp_path) -> None:
    lm = tmp_path / "lm"
    lm.mkdir()
    audit = ex.build_cache_audit(lm, ["never_extracted"], min_valid_frames=4)
    assert audit["n_excluded"] == 1
    assert audit["excluded"][0]["reason"] == "missing_landmarks"
    assert audit["n_uncached"] == 1 and audit["n_cached"] == 0


def test_refresh_meta_restamps_current_threshold(tmp_path) -> None:
    lm = tmp_path / "lm"
    _write_cache(lm, "s", n_valid=7, n_frames=16, n_frames_decoded=469,
                 meta_min_valid=8, meta_excluded=True)          # stale meta
    ex.build_cache_audit(lm, ["s"], min_valid_frames=4, refresh_meta=True)
    meta = json.loads((lm / "s.meta.json").read_text(encoding="utf-8"))
    assert meta["min_valid_frames"] == 4
    assert meta["excluded"] is False
    assert meta["exclude_reason"] is None
    # npz untouched
    assert int(np.load(lm / "s.npz")["valid"].sum()) == 7

    ex.build_cache_audit(lm, ["s"], min_valid_frames=8, refresh_meta=True)
    meta8 = json.loads((lm / "s.meta.json").read_text(encoding="utf-8"))
    assert meta8["min_valid_frames"] == 8 and meta8["excluded"] is True
    assert meta8["exclude_reason"] == "face_not_detected"


def test_no_refresh_meta_leaves_meta_untouched(tmp_path) -> None:
    lm = tmp_path / "lm"
    _write_cache(lm, "s", n_valid=7, n_frames=16, n_frames_decoded=469,
                 meta_min_valid=8, meta_excluded=True)
    ex.build_cache_audit(lm, ["s"], min_valid_frames=4, refresh_meta=False)
    meta = json.loads((lm / "s.meta.json").read_text(encoding="utf-8"))
    assert meta["min_valid_frames"] == 8            # unchanged


# ------------------------------------------------- end-to-end: terminal == json

def test_rescan_only_terminal_count_matches_json(tmp_path) -> None:
    from src.data.manifests import Manifest, ManifestRow

    lm = tmp_path / "lm"
    rows = []
    for sid, nv, dec in [("Celeb-real__idA", 16, 300), ("Celeb-real__idB", 2, 300),
                         ("Celeb-synthesis__id1_id2", 0, 1), ("YouTube-real__000", 10, 250)]:
        _write_cache(lm, sid, n_valid=nv, n_frames=16 if dec > 1 else 1, n_frames_decoded=dec)
        rows.append(ManifestRow.make(sample_id=sid, path=str(tmp_path / f"{sid}.mp4"),
                                     label=1 if "synthesis" not in sid else 0,
                                     speaker=sid, split="train"))
    # a stale audit that must be ignored
    (lm / "extraction_audit.json").write_text(json.dumps(
        {"min_valid_frames": 8, "excluded": [{"sample_id": "Celeb-real__idA"}]}), encoding="utf-8")

    csv = tmp_path / "m.csv"
    Manifest(rows).write_csv(csv)

    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--manifest", str(csv),
         "--landmarks-dir", str(lm), "--rescan-only", "--min-valid-frames", "4"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    import re
    m = re.search(r"cache-wide exclusions \(< --min-valid-frames=4\): (\d+)", out.stdout)
    assert m, out.stdout
    terminal_n = int(m.group(1))

    audit = json.loads((lm / "extraction_audit.json").read_text(encoding="utf-8"))
    assert terminal_n == audit["n_excluded"] == len(audit["excluded"])
    ids = {e["sample_id"] for e in audit["excluded"]}
    assert ids == {"Celeb-real__idB", "Celeb-synthesis__id1_id2"}   # idA (16 valid) kept
    reasons = {e["sample_id"]: e["reason"] for e in audit["excluded"]}
    assert reasons["Celeb-synthesis__id1_id2"] == "corrupt_or_truncated_source"
    assert reasons["Celeb-real__idB"] == "face_not_detected"
