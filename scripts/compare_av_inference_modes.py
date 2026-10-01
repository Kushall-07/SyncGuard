#!/usr/bin/env python3
"""Phase 14/16/24: A/B comparison of `legacy_full_clip` vs `windowed` AV inference.

Runs the existing, frozen SyncGuardPredictor (frozen audio/visual encoders +
trained cross-attention + trained SyncHead - no weights touched) in both
`mode="legacy_full_clip"` and `mode="windowed"` over the SAME inputs, and reports:

1. LAV-DF dev subset (native timing) - DESCRIPTIVE ONLY. LAV-DF's real/fake labels
   are content-manipulation labels, not synchronization ground truth (manipulation
   != desynchronization; see src/data/sync_pairs.py). This section never reports
   AUC/accuracy of "sync" against those labels - only score distributions.

2. Demo sync set (demo/video/sync/) - DESCRIPTIVE ONLY. There is no independent
   confirmation these clips are genuinely aligned beyond being shipped as the
   "sync" demo; treated as informal, not as validated ground truth.

3. Demo desync set (demo/video/desync/) - the one case here with real, verifiable
   ground truth: each pair is constructed by physically prepending silence to the
   audio track (see the leading-silence check this script runs per file), which is
   a genuine, physical temporal desynchronization - NOT the token-level controlled
   shift `src/data/sync_pairs.py` / `scripts/evaluate_sync.py` use. Both mechanisms
   are real but distinct; see docs/windowed_av_inference.md.

4. Latency: mean wall-clock time per clip for each mode.

This script does NOT decide a production threshold from these numbers, and does
NOT retrain, retune, or modify any checkpoint. It also does not use filenames or
folder names as model inputs anywhere in the inference path - "sync"/"desync"
directory names are used only to select which files to *read*, exactly like any
other file-loading code, and are never passed to the model or used to bias a score.

Usage:
    .venv/Scripts/python.exe scripts/compare_av_inference_modes.py \
        --demo-dir demo/video --lavdf-manifest data/lavdf/manifest_dev.csv \
        --lavdf-video-dir data/lavdf/extracted/dev \
        --lavdf-audio-dir data/lavdf/processed/audio \
        --lavdf-landmarks-dir data/lavdf/processed/landmarks \
        --n-lavdf 40 --n-demo 25 --output-dir outputs/av_mode_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.predictor import SyncGuardPredictor  # noqa: E402

CHECKPOINT_DIR = REPO_ROOT / "outputs" / "runs"
CONFIG_DIR = REPO_ROOT / "configs"

VISUAL_ENCODER_PATH = CHECKPOINT_DIR / "deepfake-transformer-final-20260908-210034" / "checkpoints" / "visual_encoder.pt"
SYNC_MODEL_PATH = CHECKPOINT_DIR / "sync-phase12-lambda01-20260913-115101" / "checkpoints" / "best.pt"
SYNC_CONFIG_PATH = CONFIG_DIR / "av_align_lambda01.yaml"
SPOOF_HEAD_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "best.pt"


def count_leading_silence_seconds(wav_path: Path) -> float:
    """Independently verify a WAV's leading-silence duration (exact-zero samples),
    to confirm a demo/desync pair really is a physically shifted sample rather than
    trusting the folder name."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    nonzero = np.flatnonzero(mono != 0.0)
    n_leading_zero = int(nonzero[0]) if nonzero.size else len(mono)
    return n_leading_zero / sr


def build_predictor(mode: str, device: str) -> SyncGuardPredictor:
    return SyncGuardPredictor(
        visual_encoder_path=VISUAL_ENCODER_PATH,
        sync_model_path=SYNC_MODEL_PATH,
        sync_config_path=SYNC_CONFIG_PATH,
        spoof_head_checkpoint=SPOOF_HEAD_PATH,
        device=device,
        av_inference_mode=mode,
    )


def run_one(
    predictor: SyncGuardPredictor,
    video: Path,
    audio: Path | None,
    mode: str,
    landmarker: Any = None,
    landmarks_path: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = predictor.predict_audio_visual(
        video, audio_path=audio, landmarks_path=landmarks_path, landmarker=landmarker, mode=mode
    )
    elapsed = time.perf_counter() - t0
    return {
        "predicted_label": result.predicted_label,
        "aggregate_sync_score": result.aggregate_sync_score,
        "num_windows": (result.timing_metadata or {}).get("num_windows"),
        "min_window_score": min(result.per_window_sync_scores) if result.per_window_sync_scores else None,
        "max_window_score": max(result.per_window_sync_scores) if result.per_window_sync_scores else None,
        "elapsed_seconds": elapsed,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    scores = [r["aggregate_sync_score"] for r in rows]
    n_sync = sum(1 for r in rows if r["predicted_label"] == "sync")
    return {
        "n": len(rows),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "n_predicted_sync": n_sync,
        "n_predicted_desync": len(rows) - n_sync,
        "mean_latency_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
    }


def get_landmarker() -> Any:
    global _LANDMARKER
    if _LANDMARKER is None:
        from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker

        model_path = ensure_model(DEFAULT_MODEL)
        _LANDMARKER = make_landmarker(model_path)
    return _LANDMARKER


_LANDMARKER: Any = None


def run_demo_set(demo_dir: Path, subdir: str, n: int, device: str) -> dict[str, Any]:
    folder = demo_dir / subdir
    pairs: list[tuple[Path, Path]] = []
    for video_path in sorted(folder.glob("*_video.mp4"))[:n]:
        stem = video_path.name[: -len("_video.mp4")]
        audio_path = folder / f"{stem}_audio.wav"
        if audio_path.is_file():
            pairs.append((video_path, audio_path))

    verified_shift_seconds = []
    if subdir == "desync":
        for _video, audio in pairs:
            verified_shift_seconds.append(round(count_leading_silence_seconds(audio), 3))

    results: dict[str, list[dict[str, Any]]] = {"legacy_full_clip": [], "windowed": []}
    per_clip: list[dict[str, Any]] = []
    for i, (video, audio) in enumerate(pairs):
        clip_row: dict[str, Any] = {"sample": video.stem}
        if subdir == "desync":
            clip_row["verified_leading_silence_seconds"] = verified_shift_seconds[i]
        for mode in ("legacy_full_clip", "windowed"):
            predictor = _PREDICTORS[(mode, device)]
            r = run_one(predictor, video, audio, mode, landmarker=get_landmarker())
            results[mode].append(r)
            clip_row[mode] = r
        per_clip.append(clip_row)

    return {
        "n_pairs": len(pairs),
        "legacy_full_clip_summary": summarize(results["legacy_full_clip"]),
        "windowed_summary": summarize(results["windowed"]),
        "per_clip": per_clip,
    }


def run_lavdf_subset(
    manifest_path: Path, video_dir: Path, audio_dir: Path, landmarks_dir: Path, n: int, device: str
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows = rows[:n]

    by_category: dict[str, list[dict[str, Any]]] = {"REAL": [], "AUDIO-ONLY": [], "VIDEO-ONLY": [], "AUDIO+VIDEO": []}
    latencies = {"legacy_full_clip": [], "windowed": []}
    all_rows = {"legacy_full_clip": [], "windowed": []}

    for row in rows:
        video_path = video_dir / Path(row["path"]).name
        audio_path = audio_dir / f"{Path(row['path']).stem}.wav"
        landmarks_path = landmarks_dir / f"{row['sample_id']}.npz"
        if not video_path.is_file() or not audio_path.is_file() or not landmarks_path.is_file():
            continue
        if row["label_name"] == "real":
            category = "REAL"
        elif row["modify_audio"] == "True" and row["modify_video"] != "True":
            category = "AUDIO-ONLY"
        elif row["modify_audio"] != "True" and row["modify_video"] == "True":
            category = "VIDEO-ONLY"
        else:
            category = "AUDIO+VIDEO"

        for mode in ("legacy_full_clip", "windowed"):
            predictor = _PREDICTORS[(mode, device)]
            try:
                r = run_one(predictor, video_path, audio_path, mode, landmarks_path=landmarks_path)
            except Exception as e:  # noqa: BLE001 - keep the sweep going over a bad sample
                print(f"  [skip] {video_path.name} ({mode}): {e}")
                continue
            r["category"] = category
            all_rows[mode].append(r)
            latencies[mode].append(r["elapsed_seconds"])
            if mode == "windowed":
                by_category[category].append(r)

    return {
        "n_samples_attempted": len(rows),
        "legacy_full_clip_summary": summarize(all_rows["legacy_full_clip"]),
        "windowed_summary": summarize(all_rows["windowed"]),
        "windowed_by_category": {k: summarize(v) for k, v in by_category.items()},
    }


_PREDICTORS: dict[tuple[str, str], SyncGuardPredictor] = {}


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B compare legacy_full_clip vs windowed AV inference")
    parser.add_argument("--demo-dir", default=str(REPO_ROOT / "demo" / "video"))
    parser.add_argument("--lavdf-manifest", default=str(REPO_ROOT / "data" / "lavdf" / "manifest_dev.csv"))
    parser.add_argument("--lavdf-video-dir", default=str(REPO_ROOT / "data" / "lavdf" / "extracted" / "dev"))
    parser.add_argument("--lavdf-audio-dir", default=str(REPO_ROOT / "data" / "lavdf" / "processed" / "audio"))
    parser.add_argument("--lavdf-landmarks-dir", default=str(REPO_ROOT / "data" / "lavdf" / "processed" / "landmarks"))
    parser.add_argument("--n-lavdf", type=int, default=40)
    parser.add_argument("--n-demo", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "av_mode_comparison"))
    args = parser.parse_args()

    device = args.device
    print(f"Loading predictors on device={device} (legacy_full_clip + windowed)...")
    _PREDICTORS[("legacy_full_clip", device)] = build_predictor("legacy_full_clip", device)
    _PREDICTORS[("windowed", device)] = build_predictor("windowed", device)

    results: dict[str, Any] = {}

    print(f"\n=== Demo sync set (n<={args.n_demo}) ===")
    results["demo_sync"] = run_demo_set(Path(args.demo_dir), "sync", args.n_demo, device)
    print(json.dumps({k: v for k, v in results["demo_sync"].items() if k != "per_clip"}, indent=2))

    print(f"\n=== Demo desync set (n<={args.n_demo}) ===")
    results["demo_desync"] = run_demo_set(Path(args.demo_dir), "desync", args.n_demo, device)
    print(json.dumps({k: v for k, v in results["demo_desync"].items() if k != "per_clip"}, indent=2))

    lavdf_manifest = Path(args.lavdf_manifest)
    if lavdf_manifest.is_file():
        print(f"\n=== LAV-DF dev subset (n<={args.n_lavdf}, descriptive only) ===")
        results["lavdf_dev_subset"] = run_lavdf_subset(
            lavdf_manifest,
            Path(args.lavdf_video_dir),
            Path(args.lavdf_audio_dir),
            Path(args.lavdf_landmarks_dir),
            args.n_lavdf,
            device,
        )
        print(json.dumps(results["lavdf_dev_subset"], indent=2))
    else:
        print(f"\n[skip] LAV-DF manifest not found at {lavdf_manifest}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
