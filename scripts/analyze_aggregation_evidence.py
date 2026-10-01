#!/usr/bin/env python3
"""Analysis-only investigation of AV video-level aggregation (Tasks 2-8 of the
final aggregation investigation).

This script does NOT modify predictor.py, windowing.py, sync_head.py,
cross_attention.py, sync_pairs.py, or any checkpoint. It only runs the existing,
frozen inference pipeline and the existing controlled temporal-shift mechanism
(`src/data/sync_pairs.build_sync_pair`, the same primitive `scripts/evaluate_sync.py`
and `SyncGuardPredictor.predict_sync_lab` use) to collect score distributions and
compare candidate video-level aggregation statistics.

Four data sources, kept explicitly separate (manipulation != desynchronization):

1. demo/video/sync   - descriptive only, no independent ground truth beyond the
   demo-set label.
2. demo/video/desync - the one set with real, physically verified ground truth:
   each pair has a WAV with verified leading digital silence (a genuine temporal
   shift), NOT a token-level controlled shift.
3. CONTROLLED TEMPORAL-SHIFT VALIDATION - LAV-DF dev clips at scripted shifts
   {0, +0.5, +1.0, +2.0}s using the dense 32-frame training-matched sampling
   (`build_lavdf_datasets` + `build_sync_pair`), exactly as `evaluate_sync.py`
   does. This is a controlled experiment, not real-world deepfake ground truth.
4. Native LAV-DF descriptive subset - unshifted (shift=0) LAV-DF dev clips run
   through the production windowed predictor at native timing, used ONLY to check
   whether candidate statistics produce excessive false "desync" evidence on
   ordinary synchronized-timing clips (real or manipulated-content, since
   manipulation category is irrelevant to timing).

Output: outputs/aggregation_analysis/report.json (machine-readable) plus a
concise console summary.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_dataset import build_lavdf_datasets
from src.data.sync_pairs import SyncPairConfig, build_sync_pair
from src.inference.predictor import SyncGuardPredictor
from scripts.evaluate_sync import load_trained_model

CHECKPOINT_DIR = REPO_ROOT / "outputs" / "runs"
CONFIG_DIR = REPO_ROOT / "configs"

VISUAL_ENCODER_PATH = CHECKPOINT_DIR / "deepfake-transformer-final-20260908-210034" / "checkpoints" / "visual_encoder.pt"
SYNC_MODEL_PATH = CHECKPOINT_DIR / "sync-phase12-lambda01-20260913-115101" / "checkpoints" / "best.pt"
SYNC_CONFIG_PATH = CONFIG_DIR / "av_align_lambda01.yaml"
SPOOF_HEAD_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "best.pt"
AUDIO_ENCODER_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "audio_encoder.pt"

DEMO_DIR = REPO_ROOT / "demo" / "video"
LAVDF_MANIFEST = REPO_ROOT / "data" / "lavdf" / "manifest_dev.csv"
LAVDF_VIDEO_DIR = REPO_ROOT / "data" / "lavdf" / "extracted" / "dev"
LAVDF_AUDIO_DIR = REPO_ROOT / "data" / "lavdf" / "processed" / "audio"
LAVDF_LANDMARKS_DIR = REPO_ROOT / "data" / "lavdf" / "processed" / "landmarks"

OUTPUT_DIR = REPO_ROOT / "outputs" / "aggregation_analysis"

SHIFTS = [0.0, 0.5, 1.0, 2.0]
N_LAVDF_SHIFT = 60      # samples for controlled temporal-shift validation
N_LAVDF_NATIVE = 60     # samples for native descriptive subset (windowed, shift=0)
N_DEMO = 25


# ---------------------------------------------------------------------------
# Shared statistics helpers
# ---------------------------------------------------------------------------

def score_distribution_stats(scores: list[float]) -> dict[str, float]:
    """Descriptive statistics for one clip's per-window/per-frame score array."""
    arr = np.asarray(scores, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {}
    sorted_arr = np.sort(arr)
    # simple monotonic trend: slope of a linear fit against window index,
    # plus a first-vs-last-quarter delta (robust to n<4)
    idx = np.arange(n, dtype=np.float64)
    if n >= 2:
        slope = float(np.polyfit(idx, arr, 1)[0])
    else:
        slope = 0.0
    q = max(1, n // 4)
    first_q_mean = float(arr[:q].mean())
    last_q_mean = float(arr[-q:].mean())
    return {
        "n_windows": int(n),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "frac_below_0.5": float((arr < 0.5).mean()),
        "frac_below_0.6": float((arr < 0.6).mean()),
        "frac_below_0.7": float((arr < 0.7).mean()),
        "trend_slope": slope,
        "trend_first_quarter_mean": first_q_mean,
        "trend_last_quarter_mean": last_q_mean,
        "trend_last_minus_first": last_q_mean - first_q_mean,
    }


def candidate_aggregations(scores: list[float], valid_fractions: list[float] | None = None) -> dict[str, float]:
    """Compute candidate video-level aggregation statistics (ANALYSIS ONLY).

    None of these are wired into production; they exist purely to compare
    against the current `mean_of_window_scores` rule.
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {}
    mean = float(arr.mean())
    median = float(np.median(arr))
    p10 = float(np.percentile(arr, 10))
    p25 = float(np.percentile(arr, 25))
    minimum = float(arr.min())
    frac_below_05 = float((arr < 0.5).mean())
    std = float(arr.std())

    if valid_fractions is not None and len(valid_fractions) == n and sum(valid_fractions) > 0:
        w = np.asarray(valid_fractions, dtype=np.float64)
        duration_weighted_mean = float((arr * w).sum() / w.sum())
    else:
        duration_weighted_mean = mean

    return {
        "mean": mean,
        "median": median,
        "min": minimum,
        "p10": p10,
        "p25": p25,
        "frac_below_0.5": frac_below_05,
        "duration_weighted_mean": duration_weighted_mean,
        "mean_plus_lower_tail": 0.5 * mean + 0.5 * p10,
        "temporal_consistency": std,
    }


# ---------------------------------------------------------------------------
# Section A/B/D/6: demo sync, demo desync, native LAV-DF descriptive
# (all via the production windowed predictor - the actual website code path)
# ---------------------------------------------------------------------------

def count_leading_silence_seconds(wav_path: Path) -> float:
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    nonzero = np.flatnonzero(mono != 0.0)
    n_leading_zero = int(nonzero[0]) if nonzero.size else len(mono)
    return n_leading_zero / sr


def run_demo_set(predictor: SyncGuardPredictor, landmarker: Any, subdir: str, n: int) -> dict[str, Any]:
    folder = DEMO_DIR / subdir
    pairs: list[tuple[Path, Path]] = []
    for video_path in sorted(folder.glob("*_video.mp4"))[:n]:
        stem = video_path.name[: -len("_video.mp4")]
        audio_path = folder / f"{stem}_audio.wav"
        if audio_path.is_file():
            pairs.append((video_path, audio_path))

    clips = []
    for video, audio in pairs:
        result = predictor.predict_audio_visual(video, audio_path=audio, landmarker=landmarker, mode="windowed")
        windows_meta = (result.timing_metadata or {}).get("windows", [])
        valid_fracs = [w["valid_fraction"] for w in windows_meta] if windows_meta else None
        record: dict[str, Any] = {
            "sample": video.stem,
            "predicted_label": result.predicted_label,
            "aggregate_sync_score": result.aggregate_sync_score,
            "per_window_scores": result.per_window_sync_scores,
            "distribution": score_distribution_stats(result.per_window_sync_scores or []),
            "candidates": candidate_aggregations(result.per_window_sync_scores or [], valid_fracs),
        }
        if subdir == "desync":
            record["verified_leading_silence_seconds"] = round(count_leading_silence_seconds(audio), 3)
        clips.append(record)

    return {"n_pairs": len(pairs), "clips": clips}


def run_native_lavdf_subset(predictor: SyncGuardPredictor, n: int) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with LAVDF_MANIFEST.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    clips = []
    for row in rows:
        if len(clips) >= n:
            break
        video_path = LAVDF_VIDEO_DIR / Path(row["path"]).name
        audio_path = LAVDF_AUDIO_DIR / f"{Path(row['path']).stem}.wav"
        landmarks_path = LAVDF_LANDMARKS_DIR / f"{row['sample_id']}.npz"
        if not (video_path.is_file() and audio_path.is_file() and landmarks_path.is_file()):
            continue

        if row["label_name"] == "real":
            category = "REAL"
        elif row["modify_audio"] == "True" and row["modify_video"] != "True":
            category = "AUDIO-ONLY"
        elif row["modify_audio"] != "True" and row["modify_video"] == "True":
            category = "VIDEO-ONLY"
        else:
            category = "AUDIO+VIDEO"

        try:
            result = predictor.predict_audio_visual(
                video_path, audio_path=audio_path, landmarks_path=landmarks_path, mode="windowed"
            )
        except Exception as e:  # noqa: BLE001 - keep the sweep going over a bad sample
            print(f"  [skip] {video_path.name}: {e}")
            continue

        windows_meta = (result.timing_metadata or {}).get("windows", [])
        valid_fracs = [w["valid_fraction"] for w in windows_meta] if windows_meta else None
        clips.append(
            {
                "sample_id": row["sample_id"],
                "category": category,
                "predicted_label": result.predicted_label,
                "aggregate_sync_score": result.aggregate_sync_score,
                "distribution": score_distribution_stats(result.per_window_sync_scores or []),
                "candidates": candidate_aggregations(result.per_window_sync_scores or [], valid_fracs),
            }
        )

    return {"n_clips": len(clips), "clips": clips}


# ---------------------------------------------------------------------------
# Section C: controlled temporal-shift validation
# (same primitive as scripts/evaluate_sync.py / predict_sync_lab: build_sync_pair,
# on the dense 32-frame training-matched sampling from build_lavdf_datasets)
# ---------------------------------------------------------------------------

def run_controlled_shift_validation(device: torch.device, n: int) -> dict[str, Any]:
    model, audio_token_seconds = load_trained_model(
        str(SYNC_MODEL_PATH), str(AUDIO_ENCODER_PATH), str(VISUAL_ENCODER_PATH), str(SYNC_CONFIG_PATH), device
    )

    with open(SYNC_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    video_dict = cfg.get("av_align", {}).get("video", {})
    audio_dict = cfg.get("audio", {})

    video_cfg = VideoDataConfig(num_frames=video_dict.get("num_frames", 32), regions=video_dict.get("regions", "face_mouth"))
    audio_cfg = AudioConfig.from_dict(audio_dict)

    sp_cfg_dict = cfg.get("av_align", {}).get("sync_pairs", {})
    sp_cfg = replace(SyncPairConfig(**sp_cfg_dict), audio_token_seconds=audio_token_seconds)

    dev_dataset = build_lavdf_datasets(
        manifest_path=str(LAVDF_MANIFEST),
        video_dir=str(LAVDF_VIDEO_DIR),
        audio_dir=str(LAVDF_AUDIO_DIR),
        landmarks_dir=str(LAVDF_LANDMARKS_DIR),
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=("dev",),
        n_video_tokens=video_dict.get("num_frames", 32),
        sync_pair_config=sp_cfg,
        use_negative_pairs=False,
        negative_pair_probability=0.0,
        random_sample=False,
        seed=42,
    )["dev"]

    n = min(n, len(dev_dataset))
    per_shift: dict[str, Any] = {}

    for shift in SHIFTS:
        video_level_candidates: list[dict[str, float]] = []
        per_sample_records = []
        with torch.no_grad():
            for i in range(n):
                sample = dev_dataset[i]
                mel = sample["mel_window"].unsqueeze(0).to(device)
                landmarks = sample["landmarks"].unsqueeze(0).to(device)
                fps = sample["fps"]
                window_seconds = sample["window_seconds"]

                audio_out = model.audio_encoder(mel)
                audio_tokens = audio_out.tokens[0]
                visual_out = model.visual_encoder(landmarks)
                visual_tokens = visual_out.tokens[0]

                pair = build_sync_pair(
                    audio_tokens, visual_tokens, video_fps=fps, shift_seconds=shift,
                    audio_token_seconds=audio_token_seconds, window_seconds=window_seconds,
                )
                fused_out = model.cross_attention(pair.audio_aligned.unsqueeze(0), visual_tokens.unsqueeze(0))
                logits = model.sync_head(fused_out.fused).squeeze(0)
                probs = torch.sigmoid(logits)
                valid_probs = probs[pair.mask].cpu().numpy().tolist()

                if not valid_probs:
                    continue
                cand = candidate_aggregations(valid_probs)
                video_level_candidates.append(cand)
                per_sample_records.append(
                    {"sample_id": sample.get("sample_id", f"idx_{i}"), "label_name": sample.get("label_name", "unknown"), **cand}
                )

        # Aggregate each candidate statistic ACROSS videos for this shift
        stat_names = ["mean", "median", "min", "p10", "p25", "frac_below_0.5", "duration_weighted_mean", "mean_plus_lower_tail", "temporal_consistency"]
        summary = {}
        for stat in stat_names:
            values = np.asarray([c[stat] for c in video_level_candidates], dtype=np.float64)
            if values.size == 0:
                continue
            summary[stat] = {
                "video_mean": float(values.mean()),
                "video_std": float(values.std()),
                "video_min": float(values.min()),
                "video_max": float(values.max()),
                # For threshold-facing stats (everything except frac_below_0.5 and
                # temporal_consistency), fraction of videos this statistic would
                # push below 0.5 - i.e. flip to "desync" under a naive same-threshold rule.
                "frac_videos_below_0.5": float((values < 0.5).mean()) if stat not in ("frac_below_0.5", "temporal_consistency") else None,
            }

        per_shift[str(shift)] = {
            "n_videos": len(video_level_candidates),
            "summary": summary,
            "per_sample": per_sample_records,
        }

    return {
        "note": "CONTROLLED TEMPORAL-SHIFT VALIDATION - scripted audio shift via build_sync_pair on LAV-DF dev, "
        "dense 32-frame training-matched sampling. NOT real-world deepfake ground truth.",
        "n_samples": n,
        "shifts": per_shift,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    print("\n=== Controlled temporal-shift validation (LAV-DF dev, dense 32-frame regime) ===")
    results["controlled_shift_validation"] = run_controlled_shift_validation(device, N_LAVDF_SHIFT)

    print("\n=== Loading windowed production predictor (for demo sets + native descriptive) ===")
    predictor = SyncGuardPredictor(
        visual_encoder_path=VISUAL_ENCODER_PATH,
        sync_model_path=SYNC_MODEL_PATH,
        sync_config_path=SYNC_CONFIG_PATH,
        spoof_head_checkpoint=SPOOF_HEAD_PATH,
        device=str(device),
        av_inference_mode="windowed",
    )

    from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker
    landmarker = make_landmarker(ensure_model(DEFAULT_MODEL))

    print("\n=== Demo sync set (physical demo, descriptive only) ===")
    results["demo_sync"] = run_demo_set(predictor, landmarker, "sync", N_DEMO)

    print("\n=== Demo desync set (physical +0.5s verified desync) ===")
    results["demo_desync"] = run_demo_set(predictor, landmarker, "desync", N_DEMO)

    print("\n=== Native LAV-DF descriptive subset (windowed, native timing, shift=0) ===")
    results["native_lavdf_descriptive"] = run_native_lavdf_subset(predictor, N_LAVDF_NATIVE)

    out_path = OUTPUT_DIR / "report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
