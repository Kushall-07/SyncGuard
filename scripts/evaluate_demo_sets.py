#!/usr/bin/env python3
"""Evaluate a given AV sync checkpoint against the demo sync/desync sets (section 20).

Runs the real, frozen `SyncGuardPredictor` (windowed mode - the production
default) over every pair in `demo/video/sync/` and `demo/video/desync/`, using
whichever `--sync-model-path` / `--sync-config-path` checkpoint is passed in
(old Phase 12 token-shift checkpoint or the new physical-shift checkpoint -
same script, so results are directly comparable). Folder names ("sync"/
"desync") are used only to select which files to *read from disk*, exactly
like any other file-loading code - never as a model input or a score
adjustment.

This script does NOT tune anything based on these results (no threshold
search, no aggregation change, no special-casing of any file) - it only
reports the honest sync/desync split the checkpoint under test produces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.predictor import SyncGuardPredictor  # noqa: E402

CHECKPOINT_DIR = REPO_ROOT / "outputs" / "runs"
CONFIG_DIR = REPO_ROOT / "configs"
VISUAL_ENCODER_PATH = CHECKPOINT_DIR / "deepfake-transformer-final-20260908-210034" / "checkpoints" / "visual_encoder.pt"
SPOOF_HEAD_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "best.pt"

_LANDMARKER: Any = None


def get_landmarker() -> Any:
    global _LANDMARKER
    if _LANDMARKER is None:
        from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker

        _LANDMARKER = make_landmarker(ensure_model(DEFAULT_MODEL))
    return _LANDMARKER


def evaluate_set(predictor: SyncGuardPredictor, folder: Path, n: int) -> dict[str, Any]:
    pairs: list[tuple[Path, Path]] = []
    for video_path in sorted(folder.glob("*_video.mp4"))[:n]:
        stem = video_path.name[: -len("_video.mp4")]
        audio_path = folder / f"{stem}_audio.wav"
        if audio_path.is_file():
            pairs.append((video_path, audio_path))

    per_clip: list[dict[str, Any]] = []
    for video, audio in pairs:
        result = predictor.predict_audio_visual(video, audio_path=audio, landmarker=get_landmarker(), mode="windowed")
        per_clip.append({
            "sample": video.stem,
            "predicted_label": result.predicted_label,
            "aggregate_sync_score": result.aggregate_sync_score,
            "num_windows": (result.timing_metadata or {}).get("num_windows"),
        })

    scores = [c["aggregate_sync_score"] for c in per_clip]
    n_sync = sum(1 for c in per_clip if c["predicted_label"] == "sync")
    return {
        "n_pairs": len(pairs),
        "n_predicted_sync": n_sync,
        "n_predicted_desync": len(per_clip) - n_sync,
        "mean_score": float(np.mean(scores)) if scores else None,
        "median_score": float(np.median(scores)) if scores else None,
        "std_score": float(np.std(scores)) if scores else None,
        "per_clip": per_clip,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", default=str(REPO_ROOT / "demo" / "video"))
    parser.add_argument("--sync-model-path", required=True)
    parser.add_argument("--sync-config-path", required=True)
    parser.add_argument("--visual-encoder-path", default=str(VISUAL_ENCODER_PATH))
    parser.add_argument("--spoof-head-path", default=str(SPOOF_HEAD_PATH))
    parser.add_argument("--n-demo", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default="checkpoint_under_test", help="Free-text tag for the output JSON (e.g. 'old' or 'new')")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "demo_evaluation"))
    args = parser.parse_args()

    predictor = SyncGuardPredictor(
        visual_encoder_path=args.visual_encoder_path,
        sync_model_path=args.sync_model_path,
        sync_config_path=args.sync_config_path,
        spoof_head_checkpoint=args.spoof_head_path,
        device=args.device,
        av_inference_mode="windowed",
    )

    results: dict[str, Any] = {
        "label": args.label,
        "sync_model_path": str(args.sync_model_path),
        "sync_config_path": str(args.sync_config_path),
    }
    print(f"\n=== demo/video/sync (n<={args.n_demo}) [{args.label}] ===")
    results["demo_sync"] = evaluate_set(predictor, Path(args.demo_dir) / "sync", args.n_demo)
    print(json.dumps({k: v for k, v in results["demo_sync"].items() if k != "per_clip"}, indent=2))

    print(f"\n=== demo/video/desync (n<={args.n_demo}) [{args.label}] ===")
    results["demo_desync"] = evaluate_set(predictor, Path(args.demo_dir) / "desync", args.n_demo)
    print(json.dumps({k: v for k, v in results["demo_desync"].items() if k != "per_clip"}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"demo_evaluation_{args.label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
