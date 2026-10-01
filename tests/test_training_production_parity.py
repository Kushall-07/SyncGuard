"""Training-vs-production parity test (Phase 4 of the production AV-inference
correctness audit).

Runs the SAME real LAV-DF dev clip and window through two independent code
paths:

A. training/evaluation-style inference: `LAVDFPhysicalSyncDataset.build_fixed`
   (the exact window-construction code `scripts/train_sync_physical.py` and
   `scripts/evaluate_physical_sync.py` use) feeding
   `src.training.physical_sync_trainer.PhysicalSyncModel.forward` directly.
B. production inference: `SyncGuardPredictor._predict_audio_visual_windowed`
   (via the public `predict_audio_visual(..., mode="windowed")`), using the
   same clip's video, external audio, and precomputed landmarks.

Both paths load the identical frozen audio encoder, frozen visual encoder,
cross-attention, and SyncHead checkpoints, and (as of the Phase 1 fix) the
identical `audio_token_seconds` derivation. If they disagree by more than
floating-point noise, something in the two pipelines has silently diverged -
exactly the class of bug this whole audit targets - so this test intentionally
uses a tight tolerance and does not accept an unexplained discrepancy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

_SAMPLE_ID = "035559"  # first row of manifest_dev.csv: real, 120 frames @ 25fps, window 0 = frames[0:32)
_CHECKPOINT = REPO_ROOT / "outputs/runs/sync-physical-v2-20260927-020548/checkpoints/best.pt"
_AUDIO_ENCODER = REPO_ROOT / "outputs/runs/spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt"
_VISUAL_ENCODER = REPO_ROOT / "outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt"
_SPOOF_HEAD = REPO_ROOT / "outputs/runs/spoof-transformer-20260906-123646/checkpoints/best.pt"
_CONFIG = REPO_ROOT / "configs/av_align_physical.yaml"
_MANIFEST = REPO_ROOT / "data/lavdf/manifest_dev.csv"
_VIDEO_DIR = REPO_ROOT / "data/lavdf/extracted/dev"
_AUDIO_DIR = REPO_ROOT / "data/lavdf/processed/audio"
_LANDMARKS_DIR = REPO_ROOT / "data/lavdf/processed/landmarks"

_REQUIRED = [_CHECKPOINT, _AUDIO_ENCODER, _VISUAL_ENCODER, _SPOOF_HEAD, _CONFIG, _MANIFEST,
             _VIDEO_DIR / f"{_SAMPLE_ID}.mp4", _AUDIO_DIR / f"{_SAMPLE_ID}.wav", _LANDMARKS_DIR / f"{_SAMPLE_ID}.npz"]
_missing = [str(p) for p in _REQUIRED if not Path(p).exists()]

pytestmark = pytest.mark.skipif(bool(_missing), reason=f"real checkpoints/dataset not present: {_missing}")


def _training_style_score() -> float:
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
    from src.inference.checkpoint_validation import load_state_dict_validated
    from src.training.physical_sync_trainer import PhysicalSyncModel

    device = torch.device("cpu")
    audio_encoder, audio_payload = load_audio_encoder(_AUDIO_ENCODER, map_location=device)
    visual_encoder, _ = load_visual_encoder(_VISUAL_ENCODER, map_location=device)
    audio_token_seconds = audio_token_seconds_from_encoder(audio_encoder, audio_payload["audio_cfg"])

    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    ca_cfg = CrossAttentionConfig(**cfg["av_align"]["cross_attention"])
    sh_cfg = SyncHeadConfig(**cfg["av_align"]["sync_head"])
    cross_attention = BidirectionalCrossAttention.from_config(ca_cfg)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=ca_cfg.dim)

    model = PhysicalSyncModel(audio_encoder, visual_encoder, cross_attention, sync_head)
    # Load ONLY cross_attention.*/sync_head.* from the checkpoint - exactly what
    # SyncGuardPredictor does. The checkpoint's "model" dict also contains
    # "audio_encoder.*"/"visual_encoder.*" keys (state_dict() saves the whole
    # PhysicalSyncModel, frozen submodules included), but those are a SNAPSHOT
    # taken during training, not the canonical encoder this test (and
    # production) must use - loading them here would silently substitute a
    # different audio/visual encoder than production actually runs.
    full_state_dict = torch.load(_CHECKPOINT, map_location=device, weights_only=False)["model"]
    ca_keys = {k[len("cross_attention."):]: v for k, v in full_state_dict.items() if k.startswith("cross_attention.")}
    sh_keys = {k[len("sync_head."):]: v for k, v in full_state_dict.items() if k.startswith("sync_head.")}
    load_state_dict_validated(cross_attention, ca_keys, context="parity test cross_attention")
    load_state_dict_validated(sync_head, sh_keys, context="parity test sync_head")
    model.eval()

    video_dict = cfg["av_align"].get("video", {})
    audio_cfg = AudioConfig.from_dict(cfg["audio"])
    video_cfg = VideoDataConfig(num_frames=video_dict.get("num_frames", 32), regions=video_dict.get("regions", "face_mouth"))

    ds_cfg = LAVDFPhysicalSyncConfig(
        manifest_path=_MANIFEST, video_dir=_VIDEO_DIR, audio_dir=_AUDIO_DIR, landmarks_dir=_LANDMARKS_DIR,
        split="dev", pair_config=PhysicalSyncPairConfig(),
    )
    dataset = LAVDFPhysicalSyncDataset(ds_cfg, audio_cfg, video_cfg)
    assert dataset.samples[0].sample_id == _SAMPLE_ID, (
        f"manifest_dev.csv's first row changed - expected {_SAMPLE_ID}, got {dataset.samples[0].sample_id}"
    )
    item = dataset.build_fixed(0, shift_seconds=0.0)  # first window (window_index=0) of the first sample
    assert item["window_index"] == 0
    assert item["sample_id"] == _SAMPLE_ID

    batch = physical_sync_collate_fn([item])
    with torch.no_grad():
        logits, mask = model(batch["mel_window"], batch["landmarks"], batch["fps"], batch["window_seconds"], audio_token_seconds)
    valid = torch.sigmoid(logits[0])[mask[0]]
    return valid.mean().item()


def _production_score() -> float:
    from src.inference.predictor import SyncGuardPredictor

    predictor = SyncGuardPredictor(
        visual_encoder_path=_VISUAL_ENCODER,
        sync_model_path=_CHECKPOINT,
        sync_config_path=_CONFIG,
        spoof_head_checkpoint=_SPOOF_HEAD,
        device="cpu",
        av_inference_mode="windowed",
        window_frames=32,
        stride_frames=32,
    )
    result = predictor.predict_audio_visual(
        _VIDEO_DIR / f"{_SAMPLE_ID}.mp4",
        audio_path=_AUDIO_DIR / f"{_SAMPLE_ID}.wav",
        landmarks_path=_LANDMARKS_DIR / f"{_SAMPLE_ID}.npz",
        mode="windowed",
    )
    assert result.timing_metadata["windows"][0]["index"] == 0
    return result.per_window_sync_scores[0]


def test_training_and_production_scores_match_for_the_same_window() -> None:
    training_score = _training_style_score()
    production_score = _production_score()

    abs_diff = abs(training_score - production_score)
    rel_diff = abs_diff / max(abs(training_score), 1e-9)

    print(f"\ntraining-style score:  {training_score:.8f}")
    print(f"production score:      {production_score:.8f}")
    print(f"absolute difference:   {abs_diff:.8f}")
    print(f"relative difference:   {rel_diff:.6%}")

    # Both paths run the identical frozen encoders / cross-attention / SyncHead
    # over the identical mel+landmarks window with the identical
    # audio_token_seconds; any remaining difference should be pure floating-
    # point noise (batch-of-1 vs collate, op ordering), not a pipeline
    # divergence, so this tolerance is intentionally tight.
    assert abs_diff < 1e-4, (
        f"training-style score ({training_score}) and production score "
        f"({production_score}) disagree by more than floating-point noise "
        f"(abs diff {abs_diff}) - a pipeline divergence, not proven accuracy, "
        "is the failure mode this test exists to catch."
    )
