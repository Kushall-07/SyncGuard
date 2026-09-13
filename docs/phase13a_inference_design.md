# Phase 13A: Dual-Mode Inference API Design Notes

## Overview

Phase 13A implements a clean dual-mode inference API that separates prediction from training, enabling deployment without training dependencies.

## Architecture Decisions

### Why Inference is Separated from Training

1. **Deployment Simplicity**: Inference requires only the trained checkpoints and preprocessing, not the full training infrastructure (optimizers, schedulers, data loaders, etc.)

2. **Clean API Surface**: The predictor provides a simple interface (`predict_audio()`, `predict_audio_visual()`) that hides the complexity of the multi-stage pipeline.

3. **Resource Efficiency**: Inference uses `torch.inference_mode()` to disable gradient computation, reducing memory and improving speed.

4. **Safety**: Explicit separation prevents accidental training during inference and ensures models remain frozen.

### Checkpoints Used

The inference API uses the following trained checkpoints:

1. **Audio Encoder** (Phase 5):
   - Path: `outputs/runs/spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt`
   - Architecture: CNN front-end + Transformer encoder (cnn_transformer variant)
   - Output: Temporal audio tokens [B, T, 256]
   - Audio token duration: 0.01 seconds (hop_length * time_downsample / sample_rate)

2. **Visual Encoder** (Phase 8):
   - Path: `outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt`
   - Architecture: Landmark embedding + positional encoding + Transformer encoder
   - Output: Temporal visual tokens [B, T, 256]

3. **Sync Model** (Phase 12):
   - Path: `outputs/runs/sync-phase12-lambda01-20260913-115101/checkpoints/best.pt`
   - Components: Bidirectional cross-attention + sync head
   - Training: λ=0.1 contrastive learning combined with sync loss
   - Best validation metric: val_video_sync_auc = 1.0000 @ epoch 7

4. **Spoof Head** (Phase 5, optional for audio-only mode):
   - Path: `outputs/runs/spoof-transformer-20260906-123646/checkpoints/best.pt`
   - Architecture: Pooling + classifier head
   - Output: Binary logits [spoof, bonafide]

### Audio-Only vs Audio-Visual Flow

#### Audio-Only Mode

**Pipeline:**
```
Audio File → Preprocessing (16kHz mono, peak norm) → Log-Mel Spectrogram
→ Phase-5 AudioEncoder (frozen) → SpoofHead (frozen) → Binary Classification
```

**Output Fields:**
- `mode`: "audio_only"
- `predicted_label`: "bonafide" or "spoof"
- `spoof_probability`: Probability of synthetic/cloned speech
- `bonafide_probability`: Probability of genuine speech
- `confidence`: Maximum of the two probabilities
- `audio_encoder_checkpoint`: Path to audio encoder checkpoint
- `spoof_head_checkpoint`: Path to spoof head checkpoint

**Use Case:** Detect synthetic, cloned, or spoofed speech without visual context.

#### Audio-Visual Mode

**Pipeline:**
```
Video File → Audio Extraction (16kHz mono) + Landmark Extraction (MediaPipe)
→ Audio Preprocessing → Log-Mel Spectrogram → Phase-5 AudioEncoder (frozen)
→ Landmark Preprocessing → Phase-8 VisualEncoder (frozen)
→ Phase-9: Temporal Alignment (deterministic time-aware correspondence)
→ Phase-10: Bidirectional Cross-Attention (fused AV representation)
→ Phase-11: SyncHead (per-window sync logits)
→ Aggregation (mean) → Video-level sync score
```

**Output Fields:**
- `mode`: "audio_visual"
- `predicted_label`: "sync" or "desync"
- `sync_probability`: Probability of temporal synchronization
- `desync_probability`: Probability of temporal desynchronization
- `aggregate_sync_score`: Video-level sync score (mean of per-window scores)
- `per_window_sync_scores`: List of per-window sync probabilities
- `timing_metadata`: Dictionary with fps, window_seconds, num_frames, etc.
- `audio_encoder_checkpoint`: Path to audio encoder checkpoint
- `visual_encoder_checkpoint`: Path to visual encoder checkpoint
- `sync_model_checkpoint`: Path to sync model checkpoint

**Use Case:** Detect temporal misalignment between audio and video streams.

### Important Limitations

#### AV SyncHead is NOT a Universal Deepfake Detector

The Phase 11 SyncHead is specifically designed to measure **temporal synchronization** between audio and video streams. It:

- **Detects**: Temporal misalignment (audio desynced from video)
- **Does NOT Detect**: 
  - Visual-only deepfakes (face swaps, reenactment) with properly synced audio
  - Audio-only deepfakes (voice cloning) with properly synced video
  - Content manipulation that preserves temporal alignment
  - Compression artifacts or quality degradation

The sync head was trained on LAV-DF with **controlled temporal shifts** (audio advanced/delayed by -2.0 to +2.0 seconds). It learns to recognize when audio and video are temporally misaligned, not when content is manipulated.

#### Controlled-Shift Validation ≠ Real-World Deepfake Detection

The Phase 12 validation results (val_video_sync_auc = 1.0000) were obtained on:

- **LAV-DF dev split** with artificial temporal shifts
- **Controlled negative pairs**: Audio shifted by -2.0, -1.0, -0.5, 0.5, 1.0, 2.0 seconds
- **Binary task**: Classify aligned (shift=0) vs misaligned (shift≠0) pairs

This validation demonstrates that the model learned the temporal synchronization task, but:

- **Does NOT generalize** to real-world deepfake detection
- **Does NOT account** for sophisticated attacks that preserve timing
- **Does NOT measure** detection accuracy on actual manipulated content
- **Is NOT equivalent** to performance on deepfake benchmarks like FaceForensics++ or Celeb-DF

For real-world deployment, the sync head should be used as **one component** of a multi-modal detection system, not as a standalone deepfake detector.

### Design Constraints

The predictor implementation adheres to the following constraints:

1. **No Architecture Modifications**: Reuses existing model components without changes
2. **No Retraining**: Uses frozen checkpoints without fine-tuning
3. **Preserves Conventions**: 
   - 16 kHz audio sample rate
   - MediaPipe 478-point landmarks
   - 0.01-second audio token timing
   - Label conventions (index 1 = bonafide)
4. **Error Handling**: Fails with clear, actionable errors for invalid inputs
5. **Device Flexibility**: Auto-selects CPU/CUDA with explicit override option
6. **Variable-Length Support**: Handles variable-length audio/video safely
7. **Missing Data Handling**: Uses repository's existing behavior for missing/invalid landmarks

### API Surface

```python
from src.inference import SyncGuardPredictor

# Initialize predictor
predictor = SyncGuardPredictor(
    audio_encoder_path="path/to/audio_encoder.pt",
    visual_encoder_path="path/to/visual_encoder.pt",
    sync_model_path="path/to/sync_model.pt",
    sync_config_path="path/to/av_align_lambda01.yaml",
    spoof_head_checkpoint="path/to/spoof_head.pt",  # Optional for audio-only mode
    device="auto",  # "auto", "cpu", or "cuda"
)

# Audio-only inference
audio_result = predictor.predict_audio("path/to/audio.wav")
print(f"Label: {audio_result.predicted_label}")
print(f"Confidence: {audio_result.confidence:.2f}")

# Audio-visual inference
av_result = predictor.predict_audio_visual(
    video_path="path/to/video.mp4",
    # audio_path="path/to/audio.wav",  # Optional, extracts from video if None
    # landmarks_path="path/to/landmarks.npz",  # Optional, extracts if None
    # landmarker=mediapipe_landmarker,  # Required if landmarks_path not provided
)
print(f"Label: {av_result.predicted_label}")
print(f"Sync score: {av_result.aggregate_sync_score:.2f}")
print(f"Per-window scores: {av_result.per_window_sync_scores}")
```

### Testing Strategy

The test suite (`tests/test_predictor.py`) covers:

1. **Import Tests**: Verifies predictor and result classes import successfully
2. **Checkpoint Loading**: Tests initialization with and without spoof head
3. **Audio-Only Inference**: Tests valid results, missing spoof head, missing files
4. **AV Inference**: Tests valid results (with synthetic data), missing video
5. **Schema Validation**: Tests result field validation and constraints
6. **Device Handling**: Tests CPU, CUDA, and auto-selection
7. **Gradient-Free**: Verifies inference produces no parameter gradients
8. **Per-Window Scores**: Tests score range and validation

### Future Extensions

Potential enhancements for Phase 13B+:

1. **Batch Inference**: Support processing multiple files in a single call
2. **Streaming Inference**: Support real-time processing of live streams
3. **Ensemble Mode**: Combine audio-only and AV predictions
4. **Confidence Calibration**: Improve probability estimates with temperature scaling
5. **Custom Thresholds**: Allow user-defined decision thresholds
6. **Detailed Logging**: Add optional verbose logging for debugging
7. **Export Formats**: Support JSON/CSV output for batch processing
