#!/usr/bin/env python3
"""Phase 11.5: Evaluate trained audio-visual synchronization model.

Evaluates a trained Phase 11 model on LAV-DF dev data to determine
whether it learned meaningful audio-visual synchronization.

Key evaluations:
1. Standard evaluation on dev split (shift=0)
2. Breakdown by LAV-DF category
3. Temporal shift sensitivity experiment
4. Per-window sync-score timeline output

IMPORTANT LIMITATIONS:
- The LAV-DF dev split contains 1000 samples (32 frames each, ~1.28 seconds at 25 FPS).
- Negative temporal shifts (audio advanced relative to video) may produce zero valid
  windows because the short audio duration does not overlap with the video window.
- This is expected behavior for short clips and does not indicate a model bug.
- Positive shifts (audio delayed) show sensitivity in the evaluation results.
- AUC is not reported when all windows have the same target (all positive at shift=0,
  all negative at non-zero shifts).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_dataset import build_lavdf_datasets
from src.data.sync_pairs import SyncPairConfig, build_sync_pair
from src.evaluation.sync_metrics import SyncMetrics, compute_sync_metrics
from src.models.audio.encoder import load_audio_encoder
from src.models.video.visual_encoder import load_visual_encoder
from src.models.fusion.cross_attention import BidirectionalCrossAttention, CrossAttentionConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.training.sync_trainer import SyncModel


def load_trained_model(
    checkpoint_path: str,
    audio_encoder_pt: str,
    visual_encoder_pt: str,
    config_yaml: str,
    device: torch.device,
) -> tuple[SyncModel, float]:
    """Load trained Phase 11 model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config
    with open(config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)

    # Load encoders
    print(f"Loading audio encoder from {audio_encoder_pt}")
    audio_encoder, audio_payload = load_audio_encoder(audio_encoder_pt, map_location=device)
    print(f"Loading visual encoder from {visual_encoder_pt}")
    visual_encoder, visual_payload = load_visual_encoder(visual_encoder_pt, map_location=device)

    # Get audio token seconds
    hop_length = audio_payload["audio_cfg"]["mel"]["hop_length"]
    time_downsample = audio_payload.get("time_downsample", 1)
    sample_rate = audio_payload["audio_cfg"]["sample_rate"]
    audio_token_seconds = (hop_length * time_downsample) / sample_rate
    print(f"Audio token seconds: {audio_token_seconds:.4f}")

    # Load model components from config
    ca_cfg_dict = config_dict.get("av_align", {}).get("cross_attention", {})
    ca_cfg = CrossAttentionConfig(**ca_cfg_dict)

    sh_cfg_dict = config_dict.get("av_align", {}).get("sync_head", {})
    sh_cfg = SyncHeadConfig(**sh_cfg_dict)

    # Initialize components
    cross_attention = BidirectionalCrossAttention.from_config(ca_cfg)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=ca_cfg.dim)

    # Create model
    model = SyncModel(
        audio_encoder=audio_encoder,
        visual_encoder=visual_encoder,
        cross_attention=cross_attention,
        sync_head=sync_head,
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Freeze encoders
    model.audio_encoder.eval()
    for param in model.audio_encoder.parameters():
        param.requires_grad = False

    model.visual_encoder.eval()
    for param in model.visual_encoder.parameters():
        param.requires_grad = False

    print("Model loaded successfully")

    return model, audio_token_seconds


def evaluate_with_shift(
    model: SyncModel,
    dataset,
    device: torch.device,
    shift_seconds: float,
    audio_token_seconds: float,
    batch_size: int = 32,
) -> tuple[SyncMetrics, dict, dict]:
    """Evaluate model with a specific temporal shift.

    Returns:
        metrics: SyncMetrics object
        stats: Dictionary with detailed statistics (mean score, std, etc.)
        category_stats: Dictionary with per-category statistics
    """
    model.eval()
    all_logits = []
    all_targets = []
    all_masks = []
    all_video_ids = []
    all_categories = []

    def collate_fn(batch):
        max_mel_len = max(item["mel_window"].shape[1] for item in batch)
        mel_windows = []
        for item in batch:
            mel = item["mel_window"]
            if mel.shape[1] < max_mel_len:
                padding = max_mel_len - mel.shape[1]
                mel = torch.nn.functional.pad(mel, (0, padding), mode='constant', value=0)
            mel_windows.append(mel)
        return {
            "mel_window": torch.stack(mel_windows),
            "landmarks": torch.stack([item["landmarks"] for item in batch]),
            "fps": [item["fps"] for item in batch],
            "window_seconds": [item["window_seconds"] for item in batch],
            "sample_id": [item.get("sample_id", f"idx_{i}") for i, item in enumerate(batch)],
            "label_name": [item.get("label_name", "unknown") for i, item in enumerate(batch)],
            "modify_audio": [item.get("modify_audio", False) for i, item in enumerate(batch)],
            "modify_video": [item.get("modify_video", False) for i, item in enumerate(batch)],
        }

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        for batch in loader:
            mel = batch["mel_window"].to(device)
            landmarks = batch["landmarks"].to(device)
            fps = batch["fps"]
            window_seconds = batch["window_seconds"]
            sample_ids = batch["sample_id"]
            label_names = batch["label_name"]
            modify_audios = batch["modify_audio"]
            modify_videos = batch["modify_video"]

            B = mel.shape[0]

            # Encode batch
            audio_out = model.audio_encoder(mel)
            audio_tokens = audio_out.tokens  # [B, T_a, D]
            visual_out = model.visual_encoder(landmarks)
            visual_tokens = visual_out.tokens  # [B, T_v, D]

            T_v = visual_tokens.shape[1]

            # Process each sample in batch
            for b in range(B):
                sample_fps = fps[b] if isinstance(fps, list) else fps
                sample_window = window_seconds[b] if isinstance(window_seconds, list) else window_seconds
                sample_id = sample_ids[b] if isinstance(sample_ids, list) else sample_ids
                label_name = label_names[b] if isinstance(label_names, list) else label_names
                modify_audio = modify_audios[b] if isinstance(modify_audios, list) else modify_audios
                modify_video = modify_videos[b] if isinstance(modify_videos, list) else modify_videos

                # Determine category
                if label_name == "real":
                    category = "REAL"
                elif modify_audio and not modify_video:
                    category = "AUDIO-ONLY"
                elif not modify_audio and modify_video:
                    category = "VIDEO-ONLY"
                elif modify_audio and modify_video:
                    category = "AUDIO+VIDEO"
                else:
                    category = "UNKNOWN"

                # Build sync pair with shift
                pair = build_sync_pair(
                    audio_tokens[b],  # [T_a, D]
                    visual_tokens[b],  # [T_v, D]
                    video_fps=sample_fps,
                    shift_seconds=shift_seconds,
                    audio_token_seconds=audio_token_seconds,
                    window_seconds=sample_window,
                )

                # Forward through cross-attention and sync head
                fused_out = model.cross_attention(pair.audio_aligned.unsqueeze(0), visual_tokens[b:b+1])
                fused = fused_out.fused  # [1, T_v, D]
                logits = model.sync_head(fused).squeeze(0)  # [T_v]

                all_logits.append(logits.unsqueeze(0).cpu())  # [1, T_v]
                all_targets.append(pair.targets.unsqueeze(0).cpu())  # [1, T_v]
                all_masks.append(pair.mask.unsqueeze(0).cpu())  # [1, T_v]
                all_video_ids.append(sample_id)
                all_categories.append(category)

    # Concatenate all
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_masks = torch.cat(all_masks)

    # Compute metrics
    metrics = compute_sync_metrics(all_logits, all_targets, all_masks, aggregation="mean")

    # Compute detailed statistics
    probs = torch.sigmoid(all_logits)
    valid_probs = probs[all_masks]
    valid_logits = all_logits[all_masks]
    valid_targets = all_targets[all_masks]

    # Video-level score distribution (for debugging non-monotonic accuracy)
    video_scores = []
    video_preds = []
    video_targets_list = []
    video_categories = []

    for i in range(all_logits.shape[0]):
        video_mask = all_masks[i]
        video_logits = all_logits[i]
        video_target = all_targets[i]
        video_category = all_categories[i]

        if video_mask.any():
            video_prob = torch.sigmoid(video_logits[video_mask]).mean().item()
            video_scores.append(video_prob)
            video_preds.append(1.0 if video_prob >= 0.5 else 0.0)
            video_targets_list.append(video_target[video_mask].mean().item())
            video_categories.append(video_category)

    stats = {
        "shift_seconds": shift_seconds,
        "n_videos": len(all_video_ids),
        "n_valid_windows": valid_logits.numel(),
        "n_invalid_windows": all_masks.numel() - valid_logits.numel(),
        "mean_score": valid_probs.mean().item() if valid_logits.numel() > 0 else 0.0,
        "median_score": valid_probs.median().item() if valid_logits.numel() > 0 else 0.0,
        "std_score": valid_probs.std().item() if valid_logits.numel() > 0 else 0.0,
        "min_score": valid_probs.min().item() if valid_logits.numel() > 0 else 0.0,
        "max_score": valid_probs.max().item() if valid_logits.numel() > 0 else 0.0,
        "n_positive_targets": (valid_targets == 1).sum().item(),
        "n_negative_targets": (valid_targets == 0).sum().item(),
        "video_score_mean": sum(video_scores) / len(video_scores) if video_scores else 0.0,
        "video_score_std": (sum((s - sum(video_scores)/len(video_scores))**2 for s in video_scores) / len(video_scores))**0.5 if video_scores else 0.0,
    }

    # Per-category statistics
    category_stats = {}
    for category in ["REAL", "AUDIO-ONLY", "VIDEO-ONLY", "AUDIO+VIDEO"]:
        cat_video_scores = [s for s, c in zip(video_scores, video_categories) if c == category]
        if cat_video_scores:
            category_stats[category] = {
                "n_videos": len(cat_video_scores),
                "mean_score": sum(cat_video_scores) / len(cat_video_scores),
                "std_score": (sum((s - sum(cat_video_scores)/len(cat_video_scores))**2 for s in cat_video_scores) / len(cat_video_scores))**0.5,
                "min_score": min(cat_video_scores),
                "max_score": max(cat_video_scores),
            }
        else:
            category_stats[category] = {
                "n_videos": 0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }

    return metrics, stats, category_stats


def extract_timeline(
    model: SyncModel,
    dataset,
    device: torch.device,
    sample_indices: list[int],
    audio_token_seconds: float,
    shifts: list[float],
) -> dict:
    """Extract per-window sync-score timelines for representative clips.

    Returns:
        Dictionary mapping sample_id to timeline data for each shift
    """
    model.eval()
    timelines = {}

    for idx in sample_indices:
        sample_data = dataset[idx]
        sample_id = sample_data["sample_id"]
        mel = sample_data["mel_window"].unsqueeze(0).to(device)
        landmarks = sample_data["landmarks"].unsqueeze(0).to(device)
        sample_fps = sample_data["fps"]
        sample_window = sample_data["window_seconds"]

        timelines[sample_id] = {
            "label": sample_data["label_name"],
            "duration": sample_data["duration"],
            "fps": sample_fps,
            "window_seconds": sample_window,
            "shifts": {},
        }

        with torch.no_grad():
            audio_out = model.audio_encoder(mel)
            audio_tokens = audio_out.tokens[0]  # [T_a, D]
            visual_out = model.visual_encoder(landmarks)
            visual_tokens = visual_out.tokens[0]  # [T_v, D]

        for shift in shifts:
            pair = build_sync_pair(
                audio_tokens,
                visual_tokens,
                video_fps=sample_fps,
                shift_seconds=shift,
                audio_token_seconds=audio_token_seconds,
                window_seconds=sample_window,
            )

            fused_out = model.cross_attention(pair.audio_aligned.unsqueeze(0), visual_tokens.unsqueeze(0))
            fused = fused_out.fused
            logits = model.sync_head(fused).squeeze(0)
            probs = torch.sigmoid(logits)

            # Extract timeline data
            valid_mask = pair.mask.cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            targets_np = pair.targets.cpu().numpy()

            # Window timestamps
            window_times = [(i / sample_fps) for i in range(len(probs_np))]

            timelines[sample_id]["shifts"][shift] = {
                "window_times": window_times,
                "scores": probs_np.tolist(),
                "targets": targets_np.tolist(),
                "valid_mask": valid_mask.tolist(),
                "n_valid_windows": int(valid_mask.sum()),
            }

    return timelines


def compute_audio_baseline(
    mel: Tensor,
    shift_seconds: float,
    audio_token_seconds: float,
    sample_rate: int = 16000,
) -> float:
    """Compute a simple temporal audio baseline correlation.

    This is a very simple baseline: compute the normalized audio energy correlation
    between the original and shifted audio windows. Higher correlation = more similar.

    This is NOT a synchronization model, just a sanity check to see if the task
    is trivially solvable by simple audio statistics.
    """
    # Compute audio energy (log-Mel magnitude)
    energy = mel.abs().mean(dim=0).detach().cpu().numpy()  # [T_mel]

    # Shift indices
    shift_samples = int(shift_seconds * sample_rate / (160 * 2))  # Approximate mel shift
    if shift_samples == 0:
        return 1.0  # Perfect correlation with itself

    # Simple correlation between original and shifted
    if abs(shift_samples) >= len(energy):
        return 0.0  # No overlap

    # Normalize
    energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)

    if shift_samples > 0:
        shifted = energy_norm[shift_samples:]
        original = energy_norm[:-shift_samples]
    else:
        shifted = energy_norm[:shift_samples]
        original = energy_norm[-shift_samples:]

    min_len = min(len(original), len(shifted))
    original = original[:min_len]
    shifted = shifted[:min_len]

    # Pearson correlation
    correlation = np.corrcoef(original, shifted)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 11.5: Evaluate trained sync model")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--audio-encoder", required=True, help="Path to audio encoder export")
    parser.add_argument("--visual-encoder", required=True, help="Path to visual encoder export")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--dev-manifest", required=True, help="LAV-DF dev manifest")
    parser.add_argument("--video-dir", required=True, help="Video directory")
    parser.add_argument("--audio-dir", required=True, help="Audio directory")
    parser.add_argument("--landmarks-dir", required=True, help="Landmarks directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, audio_token_seconds = load_trained_model(
        args.checkpoint,
        args.audio_encoder,
        args.visual_encoder,
        args.config,
        device,
    )

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    video_dict = cfg.get("av_align", {}).get("video", {})
    audio_dict = cfg.get("audio", {})

    # Build dev dataset
    video_cfg = VideoDataConfig(
        num_frames=video_dict.get("num_frames", 32),
        regions=video_dict.get("regions", "face_mouth"),
    )
    audio_cfg = AudioConfig.from_dict(audio_dict)

    sp_cfg_dict = cfg.get("av_align", {}).get("sync_pairs", {})
    sp_cfg = SyncPairConfig(**sp_cfg_dict)
    sp_cfg = replace(sp_cfg, audio_token_seconds=audio_token_seconds)

    print(f"Video window size: {video_cfg.num_frames} frames")
    print(f"Expected window duration at 25 FPS: {video_cfg.num_frames / 25.0:.2f}s")

    dev_dataset = build_lavdf_datasets(
        manifest_path=args.dev_manifest,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        landmarks_dir=args.landmarks_dir,
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

    print(f"Dev samples: {len(dev_dataset)}")

    # Report actual clip durations from metadata
    durations = [s.duration for s in dev_dataset.samples]
    print(f"Clip duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"Mean clip duration: {sum(durations)/len(durations):.2f}s")

    # Standard evaluation (shift=0)
    print("\n=== Standard Evaluation (shift=0) ===")
    overall_metrics, overall_stats, overall_category_stats = evaluate_with_shift(
        model, dev_dataset, device, shift_seconds=0.0, audio_token_seconds=audio_token_seconds, batch_size=64
    )

    if overall_metrics.video_auc is not None:
        print(f"Video-level AUC: {overall_metrics.video_auc:.4f}")
    else:
        print(f"Video-level AUC: N/A")
    print(f"Video-level Accuracy: {overall_metrics.video_accuracy:.4f}")
    if overall_metrics.auc is not None:
        print(f"Per-window AUC: {overall_metrics.auc:.4f}")
    else:
        print(f"Per-window AUC: N/A")
    print(f"Per-window Accuracy: {overall_metrics.accuracy:.4f}")
    print(f"Samples: {overall_metrics.n_samples}")
    print(f"Windows: {overall_metrics.n_windows}")
    print(f"Valid windows: {overall_stats['n_valid_windows']}")
    print(f"Mean sync score: {overall_stats['mean_score']:.4f}")
    print(f"Median sync score: {overall_stats['median_score']:.4f}")
    print(f"Std sync score: {overall_stats['std_score']:.4f}")

    # Category-wise analysis (shift=0)
    print("\n=== Category-Wise Analysis (shift=0) ===")
    for category, cat_stats in overall_category_stats.items():
        if cat_stats["n_videos"] > 0:
            print(f"{category}:")
            print(f"  Videos: {cat_stats['n_videos']}")
            print(f"  Mean sync score: {cat_stats['mean_score']:.4f}")
            print(f"  Std sync score: {cat_stats['std_score']:.4f}")
            print(f"  Min sync score: {cat_stats['min_score']:.4f}")
            print(f"  Max sync score: {cat_stats['max_score']:.4f}")

    # Category breakdown
    print("\n=== Category Breakdown ===")
    category_results = {}
    for sample in dev_dataset.samples:
        category = sample.label_name
        if sample.modify_audio and not sample.modify_video:
            category = "AUDIO-ONLY"
        elif not sample.modify_audio and sample.modify_video:
            category = "VIDEO-ONLY"
        elif sample.modify_audio and sample.modify_video:
            category = "AUDIO+VIDEO"

        if category not in category_results:
            category_results[category] = {"samples": [], "indices": []}
        category_results[category]["samples"].append(sample)
        category_results[category]["indices"].append(
            next(i for i, s in enumerate(dev_dataset.samples) if s.sample_id == sample.sample_id)
        )

    for category, data in category_results.items():
        print(f"{category}: {len(data['samples'])} samples")

    # Shift sensitivity experiment
    print("\n=== Temporal Shift Sensitivity ===")
    shifts = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
    shift_results = {}
    shift_stats_dict = {}
    shift_category_stats_dict = {}
    for shift in shifts:
        print(f"Shift={shift}s")
        metrics, stats, category_stats = evaluate_with_shift(
            model, dev_dataset, device, shift_seconds=shift, audio_token_seconds=audio_token_seconds, batch_size=64
        )
        shift_results[shift] = asdict(metrics)
        shift_stats_dict[shift] = stats
        shift_category_stats_dict[shift] = category_stats
        if metrics.video_auc is not None:
            print(f"  Video-level AUC: {metrics.video_auc:.4f}")
        else:
            print(f"  Video-level AUC: N/A")
        print(f"  Video-level Accuracy: {metrics.video_accuracy:.4f}")
        print(f"  Valid windows: {stats['n_valid_windows']}/{metrics.n_windows}")
        print(f"  Mean sync score: {stats['mean_score']:.4f}")
        print(f"  Median sync score: {stats['median_score']:.4f}")
        print(f"  Std sync score: {stats['std_score']:.4f}")
        print(f"  Video-level score mean: {stats['video_score_mean']:.4f}")
        print(f"  Video-level score std: {stats['video_score_std']:.4f}")

    # Representative clip analysis: same clips at different shifts
    print("\n=== Representative Clip Analysis (First 10 Clips) ===")
    representative_results = {}
    n_representative = min(10, len(dev_dataset))
    shifts_for_table = [0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0]

    for i in range(n_representative):
        sample_id = dev_dataset.samples[i].sample_id
        label_name = dev_dataset.samples[i].label_name
        duration = dev_dataset.samples[i].duration
        fps = dev_dataset.timing(i)["fps"]
        representative_results[sample_id] = {
            "label": label_name,
            "duration": duration,
            "fps": fps,
            "shifts": {},
        }

        # Get single sample data
        sample_data = dev_dataset[i]
        mel = sample_data["mel_window"].unsqueeze(0).to(device)
        landmarks = sample_data["landmarks"].unsqueeze(0).to(device)
        sample_fps = sample_data["fps"]
        sample_window = sample_data["window_seconds"]

        with torch.no_grad():
            audio_out = model.audio_encoder(mel)
            audio_tokens = audio_out.tokens[0]  # [T_a, D]
            visual_out = model.visual_encoder(landmarks)
            visual_tokens = visual_out.tokens[0]  # [T_v, D]

        for shift in shifts_for_table:
            pair = build_sync_pair(
                audio_tokens,
                visual_tokens,
                video_fps=sample_fps,
                shift_seconds=shift,
                audio_token_seconds=audio_token_seconds,
                window_seconds=sample_window,
            )

            fused_out = model.cross_attention(pair.audio_aligned.unsqueeze(0), visual_tokens.unsqueeze(0))
            fused = fused_out.fused
            logits = model.sync_head(fused).squeeze(0)
            probs = torch.sigmoid(logits)

            valid_probs = probs[pair.mask]
            if valid_probs.numel() > 0:
                representative_results[sample_id]["shifts"][shift] = {
                    "mean_score": valid_probs.mean().item(),
                    "median_score": valid_probs.median().item(),
                    "n_valid_windows": valid_probs.numel(),
                }
            else:
                representative_results[sample_id]["shifts"][shift] = {
                    "mean_score": None,
                    "median_score": None,
                    "n_valid_windows": 0,
                }

    # Print representative table
    print(f"{'Sample ID':<20} {'Label':<15} {'Duration(s)':<12} {'FPS':<6}", end="")
    for shift in shifts_for_table:
        print(f" {shift:+.1f}s", end="")
    print()
    print("-" * (20 + 15 + 12 + 6 + len(shifts_for_table) * 8))

    for sample_id, data in representative_results.items():
        print(f"{sample_id:<20} {data['label']:<15} {data['duration']:<12.2f} {data['fps']:<6.1f}", end="")
        for shift in shifts_for_table:
            shift_data = data["shifts"][shift]
            if shift_data["n_valid_windows"] > 0:
                print(f" {shift_data['mean_score']:.3f}", end="")
            else:
                print(f" {'N/A':<5}", end="")
        print()

    # Per-window timeline extraction
    print("\n=== Per-Window Timeline Extraction ===")
    timeline_indices = list(range(min(5, len(dev_dataset))))
    timeline_shifts = [0.0, 0.5, 1.0, 2.0]
    timelines = extract_timeline(
        model, dev_dataset, device, timeline_indices, audio_token_seconds, timeline_shifts
    )
    print(f"Extracted timelines for {len(timelines)} clips at shifts {timeline_shifts}")

    # Audio-only baseline comparison
    print("\n=== Audio-Only Baseline Comparison ===")
    audio_baseline_results = {}
    for shift in [0.0, 0.5, 1.0, 2.0]:
        baseline_correlations = []
        for i in range(min(100, len(dev_dataset))):
            sample_data = dev_dataset[i]
            mel = sample_data["mel_window"]
            corr = compute_audio_baseline(mel, shift, audio_token_seconds)
            baseline_correlations.append(corr)
        
        audio_baseline_results[shift] = {
            "mean_correlation": sum(baseline_correlations) / len(baseline_correlations),
            "std_correlation": (sum((c - sum(baseline_correlations)/len(baseline_correlations))**2 for c in baseline_correlations) / len(baseline_correlations))**0.5,
        }
        print(f"Shift={shift}s: mean correlation = {audio_baseline_results[shift]['mean_correlation']:.4f}, std = {audio_baseline_results[shift]['std_correlation']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "overall": asdict(overall_metrics),
        "overall_stats": overall_stats,
        "overall_category_stats": overall_category_stats,
        "category_breakdown": {k: len(v["samples"]) for k, v in category_results.items()},
        "shift_sensitivity": shift_results,
        "shift_stats": shift_stats_dict,
        "shift_category_stats": shift_category_stats_dict,
        "representative_clips": representative_results,
        "timelines": timelines,
        "audio_baseline": audio_baseline_results,
    }

    results_path = output_dir / "evaluation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Save timelines separately as CSV for easier analysis
    timelines_csv_path = output_dir / "timelines.csv"
    with open(timelines_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "shift", "window_index", "window_time", "score", "target", "valid"])
        for sample_id, data in timelines.items():
            for shift, shift_data in data["shifts"].items():
                for i, (score, target, valid) in enumerate(zip(
                    shift_data["scores"], shift_data["targets"], shift_data["valid_mask"]
                )):
                    writer.writerow([
                        sample_id,
                        shift,
                        i,
                        shift_data["window_times"][i],
                        score,
                        target,
                        valid,
                    ])

    print(f"\nResults saved to {results_path}")
    print(f"Timelines saved to {timelines_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
