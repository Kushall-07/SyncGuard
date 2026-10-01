#!/usr/bin/env python3
"""Held-out PHYSICAL desynchronization evaluation (sync-detection fix, section 19).

Evaluates a trained `PhysicalSyncModel` checkpoint (see
`scripts/train_sync_physical.py`) against the SAME set of held-out LAV-DF dev
windows under nine conditions:

    A. native (shift = 0.0s, the clip's own unshifted audio)
    B/C. +0.25s / -0.25s
    D/E. +0.50s / -0.50s
    F/G. +0.75s / -0.75s
    H/I. +1.00s / -1.00s

Every condition reuses `LAVDFPhysicalSyncDataset.build_fixed`, so all nine
conditions are evaluated on the exact same (clip, window) set - the only thing
that changes is the caller-specified physical shift applied to that window's
own audio before the mel spectrogram / audio encoder, exactly mirroring how
`demo/video/desync/*` was constructed. Cross-clip negatives are NOT part of
this specific breakdown (they are evaluated separately by the training-time
validation metrics, since section 19's own category list is native + 8 shift
magnitudes only).

Reports, per category: n, mean/median score, accuracy, precision, recall, F1,
ROC-AUC, and a confusion matrix - at both the per-window level and the
per-video level (windows of the same clip mean-pooled, matching production's
`mean_of_window_scores` aggregation). Also reports one pooled binary
confusion matrix (native vs. all shift conditions combined) at the video level.

This script never reads a manipulation label (`label_name`/`modify_audio`/
`modify_video`) to decide sync/desync - only the shift this script itself
applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_physical_sync_dataset import (
    LAVDFPhysicalSyncConfig,
    LAVDFPhysicalSyncDataset,
    PhysicalSyncPairConfig,
    physical_sync_collate_fn,
)
from src.models.audio.encoder import audio_token_seconds_from_encoder, load_audio_encoder
from src.models.fusion.cross_attention import BidirectionalCrossAttention, CrossAttentionConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.video.visual_encoder import load_visual_encoder
from src.training.physical_sync_trainer import PhysicalSyncModel

CATEGORIES: list[tuple[str, float]] = [
    ("native", 0.0),
    ("+0.25s", 0.25), ("-0.25s", -0.25),
    ("+0.50s", 0.50), ("-0.50s", -0.50),
    ("+0.75s", 0.75), ("-0.75s", -0.75),
    ("+1.00s", 1.00), ("-1.00s", -1.00),
]


def load_physical_model(
    checkpoint_path: str, audio_encoder_pt: str, visual_encoder_pt: str, config_yaml: str, device: torch.device,
) -> tuple[PhysicalSyncModel, float]:
    print(f"Loading audio encoder from {audio_encoder_pt}")
    audio_encoder, audio_payload = load_audio_encoder(audio_encoder_pt, map_location=device)
    print(f"Loading visual encoder from {visual_encoder_pt}")
    visual_encoder, _ = load_visual_encoder(visual_encoder_pt, map_location=device)

    audio_token_seconds = audio_token_seconds_from_encoder(audio_encoder, audio_payload["audio_cfg"])

    with open(config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ca_cfg = CrossAttentionConfig(**cfg.get("av_align", {}).get("cross_attention", {}))
    sh_cfg = SyncHeadConfig(**cfg.get("av_align", {}).get("sync_head", {}))
    cross_attention = BidirectionalCrossAttention.from_config(ca_cfg)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=ca_cfg.dim)

    model = PhysicalSyncModel(audio_encoder, visual_encoder, cross_attention, sync_head)
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # The OLD Phase 11/12 checkpoint (src.training.sync_trainer.SyncModel) has
        # extra Phase-12 contrastive-adapter/projection keys this model has no
        # parameters for (never used at inference either way - see
        # src/inference/predictor.py, which also never references them). Filter
        # to only the submodules PhysicalSyncModel actually has, so the SAME
        # checkpoint-loading path works for both old and new checkpoints and can
        # be used for a like-for-like old-vs-new comparison.
        filtered = {k: v for k, v in state_dict.items() if k.split(".")[0] in ("audio_encoder", "visual_encoder", "cross_attention", "sync_head")}
        model.load_state_dict(filtered, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, audio_token_seconds


def _confusion(preds: np.ndarray, targets: np.ndarray) -> dict[str, int]:
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _binary_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    preds = (scores >= 0.5).astype(int)
    cm = _confusion(preds, targets)
    n = len(targets)
    accuracy = (cm["tp"] + cm["tn"]) / n if n else 0.0
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = None
    if len(set(targets.tolist())) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(targets, scores))
        except Exception:
            auc = None
    return {
        "n": n,
        "mean_score": float(scores.mean()) if n else 0.0,
        "median_score": float(np.median(scores)) if n else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def score_category(
    model: PhysicalSyncModel, dataset: LAVDFPhysicalSyncDataset, shift_seconds: float,
    audio_token_seconds: float, device: torch.device, batch_size: int, limit: int | None,
) -> tuple[list[dict[str, Any]]]:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    records: list[dict[str, Any]] = []
    batch: list[dict] = []

    def flush(batch_items: list[dict]) -> None:
        collated = physical_sync_collate_fn(batch_items)
        mel = collated["mel_window"].to(device)
        landmarks = collated["landmarks"].to(device)
        logits, mask = model(mel, landmarks, collated["fps"], collated["window_seconds"], audio_token_seconds)
        probs = torch.sigmoid(logits)
        for b in range(len(batch_items)):
            valid = probs[b][mask[b]]
            score = valid.mean().item() if valid.numel() > 0 else float("nan")
            records.append({
                "sample_id": batch_items[b]["sample_id"],
                "window_index": batch_items[b]["window_index"],
                "score": score,
                "target": batch_items[b]["label"],
            })

    for i in range(n):
        batch.append(dataset.build_fixed(i, shift_seconds=shift_seconds))
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    return records


def video_level(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_sample: dict[str, list[float]] = {}
    target_by_sample: dict[str, float] = {}
    for r in records:
        if r["score"] != r["score"]:  # NaN guard (all-silence window with no valid tokens)
            continue
        by_sample.setdefault(r["sample_id"], []).append(r["score"])
        target_by_sample[r["sample_id"]] = r["target"]
    scores = np.array([np.mean(v) for v in by_sample.values()])
    targets = np.array([target_by_sample[k] for k in by_sample])
    return scores, targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio-encoder", required=True)
    parser.add_argument("--visual-encoder", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--split", default="dev", help="'dev' for validation; pass 'test' only once, deliberately, for a final held-out number")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of windows evaluated per category (debug)")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    model, audio_token_seconds = load_physical_model(
        args.checkpoint, args.audio_encoder, args.visual_encoder, args.config, device
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    video_dict = cfg.get("av_align", {}).get("video", {})
    audio_cfg = AudioConfig.from_dict(cfg.get("audio", {}))
    video_cfg = VideoDataConfig(num_frames=video_dict.get("num_frames", 32), regions=video_dict.get("regions", "face_mouth"))

    ds_cfg = LAVDFPhysicalSyncConfig(
        manifest_path=args.manifest, video_dir=args.video_dir, audio_dir=args.audio_dir,
        landmarks_dir=args.landmarks_dir, split=args.split, pair_config=PhysicalSyncPairConfig(),
    )
    dataset = LAVDFPhysicalSyncDataset(ds_cfg, audio_cfg, video_cfg)
    print(f"Loaded {len(dataset)} windows from split={args.split!r}")

    results: dict[str, Any] = {"split": args.split, "n_windows_total": len(dataset), "categories": {}}
    all_video_scores: dict[str, np.ndarray] = {}
    all_video_targets: dict[str, np.ndarray] = {}

    for name, shift in CATEGORIES:
        print(f"\n=== Category: {name} ===")
        records = score_category(model, dataset, shift, audio_token_seconds, device, args.batch_size, args.limit)
        window_scores = np.array([r["score"] for r in records if r["score"] == r["score"]])
        window_targets = np.array([r["target"] for r in records if r["score"] == r["score"]])
        window_metrics = _binary_metrics(window_scores, window_targets)

        video_scores, video_targets = video_level(records)
        video_metrics = _binary_metrics(video_scores, video_targets)
        all_video_scores[name] = video_scores
        all_video_targets[name] = video_targets

        print(f"  window-level: n={window_metrics['n']} mean={window_metrics['mean_score']:.4f} "
              f"acc={window_metrics['accuracy']:.4f} f1={window_metrics['f1']:.4f} "
              f"auc={window_metrics['roc_auc']}")
        print(f"  video-level:  n={video_metrics['n']} mean={video_metrics['mean_score']:.4f} "
              f"acc={video_metrics['accuracy']:.4f} f1={video_metrics['f1']:.4f} "
              f"auc={video_metrics['roc_auc']}")

        results["categories"][name] = {"shift_seconds": shift, "window_level": window_metrics, "video_level": video_metrics}

    # Pooled binary confusion matrix: native (SYNC) vs. every shift condition combined (DESYNC),
    # at the video level - the single number closest to "does this checkpoint separate sync from
    # physical desync overall".
    pooled_scores = np.concatenate([all_video_scores[name] for name, _ in CATEGORIES])
    pooled_targets = np.concatenate([all_video_targets[name] for name, _ in CATEGORIES])
    results["pooled_video_level"] = _binary_metrics(pooled_scores, pooled_targets)
    print("\n=== Pooled video-level (native vs. all shifts combined) ===")
    print(json.dumps(results["pooled_video_level"], indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "physical_sync_evaluation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
