#!/usr/bin/env python3
"""Audio-only regression check against the demo bonafide/spoof sets (Phase 9 of
the production AV-inference correctness audit).

Runs the real, frozen `SyncGuardPredictor.predict_audio` over every file in
`demo/audio/bonafide/` and `demo/audio/spoof/` through the production API,
exactly as deployed (no filename/folder-based logic feeds the model - the
folder name is only used here, after the fact, to report whether the
prediction matched the known ground truth). This script does not modify
audio scoring, thresholds, or the spoofing pipeline in any way - it only
reports the existing model's behavior on the existing demo set.
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


def evaluate_set(predictor: SyncGuardPredictor, folder: Path, expected_label: str) -> dict[str, Any]:
    files = sorted(p for p in folder.iterdir() if p.is_file())
    per_file: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            result = predictor.predict_audio(path)
            per_file.append({
                "sample": path.name,
                "predicted_label": result.predicted_label,
                "expected_label": expected_label,
                "correct": result.predicted_label == expected_label,
                "bonafide_probability": result.bonafide_probability,
                "spoof_probability": result.spoof_probability,
            })
        except Exception as e:  # noqa: BLE001 - reporting script, must not hide failures
            errors.append({"sample": path.name, "error": f"{type(e).__name__}: {e}"})

    n_correct = sum(1 for c in per_file if c["correct"])
    scores = [c["bonafide_probability"] for c in per_file]
    return {
        "n_total": len(files),
        "n_scored": len(per_file),
        "n_errors": len(errors),
        "n_correct": n_correct,
        "n_incorrect": len(per_file) - n_correct,
        "accuracy": (n_correct / len(per_file)) if per_file else None,
        "mean_bonafide_probability": float(np.mean(scores)) if scores else None,
        "per_file": per_file,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", default=str(REPO_ROOT / "demo" / "audio"))
    parser.add_argument(
        "--spoof-head-path",
        default=str(CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "best.pt"),
    )
    parser.add_argument(
        "--visual-encoder-path",
        default=str(CHECKPOINT_DIR / "deepfake-transformer-final-20260908-210034" / "checkpoints" / "visual_encoder.pt"),
    )
    parser.add_argument(
        "--sync-model-path",
        default=str(CHECKPOINT_DIR / "sync-physical-v2-20260927-020548" / "checkpoints" / "best.pt"),
    )
    parser.add_argument("--sync-config-path", default=str(CONFIG_DIR / "av_align_physical.yaml"))
    parser.add_argument(
        "--cnn-checkpoint",
        default=str(CHECKPOINT_DIR / "spoof-cnn-baseline-20260906-104508" / "checkpoints" / "best.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "demo_evaluation"))
    args = parser.parse_args()

    # The sync model/config/visual-encoder are required by SyncGuardPredictor's
    # constructor even for audio-only use; they have no effect on predict_audio.
    predictor = SyncGuardPredictor(
        visual_encoder_path=args.visual_encoder_path,
        sync_model_path=args.sync_model_path,
        sync_config_path=args.sync_config_path,
        spoof_head_checkpoint=args.spoof_head_path,
        cnn_checkpoint=args.cnn_checkpoint,
        device=args.device,
    )

    results: dict[str, Any] = {}
    print("\n=== demo/audio/bonafide ===")
    results["bonafide"] = evaluate_set(predictor, Path(args.demo_dir) / "bonafide", "bonafide")
    print(json.dumps({k: v for k, v in results["bonafide"].items() if k not in ("per_file", "errors")}, indent=2))

    print("\n=== demo/audio/spoof ===")
    results["spoof"] = evaluate_set(predictor, Path(args.demo_dir) / "spoof", "spoof")
    print(json.dumps({k: v for k, v in results["spoof"].items() if k not in ("per_file", "errors")}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "audio_demo_evaluation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
