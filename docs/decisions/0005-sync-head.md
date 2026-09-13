# Decision 0005: Phase 11 Audio-Visual Sync Head

## Context

Phase 11 implements a self-supervised audio-visual synchronization head for SyncGuard. This phase builds on the frozen Phase 9 AVEncoder and trainable Phase 10 bidirectional cross-attention to learn temporal correspondence between audio and video streams.

The sync head consumes the fused AV representation `[B, T, 256]` from Phase 10 and produces per-window sync logits `[B, T]` for frame-level synchronization assessment.

## Key Principle: Manipulation ≠ Desynchronization

**CRITICAL: Manipulation is not equivalent to temporal desynchronization.**

LAV-DF dataset categories (real, audio-only manipulated, video-only manipulated, audio+video manipulated) must NOT automatically receive sync=0 labels. A manipulated modality can still be temporally synchronized with the other modality.

Therefore, **LAV-DF fake_periods are NOT used as sync labels** in the baseline implementation. They are parsed and preserved as metadata only.

## Decisions

### D1: Positive Pair Construction

**Decision:** Positive pairs are constructed from naturally aligned audio and video from the same clip.

**Implementation:**
- Original audio tokens → normal Phase-9 time-aware alignment → `audio_aligned [T_v, D]`
- All valid video-token windows receive target=1
- No temporal shift applied (`shift_seconds = 0.0`)
- Works for all LAV-DF categories (real and manipulated)

### D2: Negative Pair Construction

**Decision:** Negative pairs are constructed by deliberately time-shifting audio relative to video within the same clip.

**Implementation:**
- Same original audio tokens as the positive pair
- Temporal shift applied at the **audio-token timeline level** before cross-attention
- Audio token interval `j` is shifted: `[j*dt_a + Δt, (j+1)*dt_a + Δt)`
- Audio-to-video correspondence is recomputed using the shifted timeline
- Produces `shifted_audio_aligned [T_v, D]` — a different tensor from the positive pair
- Target=0 for all valid windows (see D3)
- Shift values: `[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]` seconds (configurable)
- **No MP4/audio file rewriting, no duplicate datasets, no changes to frozen encoders**

**Where shift is applied:**
```
audio_tokens [T_a, D]  (from frozen AudioEncoder)
        │
        ├─ shift=0  → align_audio_to_video()        → audio_aligned_pos [T_v, D] → target 1
        └─ shift=Δt → compute_shifted_alignment() → audio_aligned_neg [T_v, D] → target 0
                              │
                              ▼
              BidirectionalCrossAttention(audio_aligned_*, visual_tokens)
                              │
                              ▼
                         SyncHead → BCE loss
```

### D3: Target Semantics (Threshold Rejected)

**Decision:** Use a simple zero/non-zero shift rule. **Do not use a 0.04s alignment threshold.**

**Rationale:**
- Once negative pairs are explicitly constructed with a configured non-zero shift, a separate overlap threshold is redundant and ambiguous
- A deliberately shifted pair (e.g. 0.03s) should be labeled negative, not positive
- The model input (`audio_aligned`) already reflects the shift via recomputed correspondence; the target should match that construction

**Implementation:**
- `shift_seconds == 0.0` → target=1 (positive)
- `shift_seconds != 0.0` → target=0 (negative)
- Applies uniformly to all valid (unmasked) video tokens

**Rejected alternative:** threshold `dt_a/2 = 0.04s` where small shifts remain positive. This conflates "small deliberate shift" with "naturally aligned" and is not needed once alignment itself is shifted.

### D4: fake_periods Handling

**Decision:** LAV-DF fake_periods are parsed but NOT used as sync labels.

**Implementation:**
- `parse_fake_periods()` parses JSON string to list of `[start, end]` periods
- Stored in dataset metadata (`fake_periods` field)
- No sync labels derived from fake_periods
- Manipulation label (`label=0/1`) is also NOT used as sync label

### D5: Per-Window vs Video-Level Targets

**Decision:** Primary targets are per-window (per-token) sync labels.

**Implementation:**
- SyncHead outputs logits `[B, T]` for each video token
- Loss computed per-window with masking for invalid positions
- Video-level metrics via configurable aggregation (mean/max/median)

### D6: Masking

**Decision:** Mask video tokens with no valid audio correspondence after alignment.

**Implementation:**
- `bucket_counts[k] == 0` → mask position `k` as invalid
- Large shifts can push audio tokens entirely outside the video window
- Mask applied in both loss and metrics
- Cross-attention receives unmasked inputs; loss/metrics exclude invalid positions

### D7: Temporal Windowing

**Decision:** Fixed `n_video_tokens` window with deterministic frame selection.

**Implementation:**
- `n_video_tokens` (default 32, from `configs/av_align.yaml` → `video.num_frames`)
- Frame indices selected via `linspace(0, n_frames-1, n_video_tokens)` (deterministic)
- Optional `random_sample=True` for training augmentation
- `window_seconds = n_video_tokens / fps` (per-clip, not hard-coded 25 FPS)
- Audio cropped to `[frame_idx0/fps, frame_idx0/fps + window_seconds)` before mel extraction
- Landmarks subsampled to the same frame indices

**Audio/video timing convention:**
- Time origin: start of the aligned window (`frame_idx0` in source video)
- Video bucket `k`: `[k/fps, (k+1)/fps)` seconds from window start
- Audio token `j`: `[j*dt_a, (j+1)*dt_a)` seconds from window start (unshifted)
- Shifted audio token `j`: `[j*dt_a + Δt, (j+1)*dt_a + Δt)`

### D8: FPS / Timing

**Decision:** FPS is obtained from landmark NPZ metadata, not assumed.

**Implementation:**
- Landmark extraction stores `fps` in each `.npz` file
- `LAVDFSyncDataset.timing()` reads FPS from NPZ
- Passed explicitly to alignment code (`video_fps` argument)
- Manifest `video_frames` / `duration` are metadata only; not used for alignment timing

### D9: Train/Dev Split Isolation

**Decision:** Training pipeline must explicitly load `split=train` for training and `split=dev` for validation.

**Implementation:**
- `LAVDFSyncConfig.split` filters manifest rows
- `build_lavdf_datasets(splits=("train", "dev"))` creates isolated datasets
- Init-time validation ensures no split mismatch
- Tests verify: train-only rows, dev-only rows, no path/sample_id overlap

### D10: Sync Head Architecture

Lightweight temporal prediction head: `[B, T, 256] → Linear → LayerNorm → ReLU → Dropout → Linear → [B, T]`

### D11: Loss Function

Binary cross-entropy with optional mask and `pos_weight`. Mean reduction over valid positions only.

### D12: Training Pipeline

**Decision:** Implement SyncTrainer as a subclass of the base Trainer class, following existing patterns.

**Implementation:**
- `SyncModel` wrapper combines frozen encoders + trainable cross-attention + sync head
- `SyncTrainer` extends `Trainer` with sync-specific compute_loss and compute_metrics
- Encoders are frozen in eval mode (deterministic dropout/batchnorm)
- Only cross-attention and sync head parameters receive gradients
- Supports AMP, gradient clipping, checkpointing, and early stopping
- Video-level sync metrics computed via configurable aggregation

**Data flow:**
```
batch → frozen encoders → audio_tokens, visual_tokens
       → Phase 9 alignment (positive reference)
       → build_sync_pair() with shift_seconds
       → Phase 10 cross-attention (trainable)
       → Phase 11 sync head (trainable)
       → masked BCE loss
       → sync metrics
```

### D13: Scope Limitations

**Excluded in baseline:**
- Contrastive learning (InfoNCE)
- Cross-sample audio/video pairing
- Using fake_periods or manipulation labels as sync supervision
- LAV-DF audio/landmark extraction scripts (prerequisite data prep)
- Phase 11.5 broader AV evaluation (manipulation + sync)

## Files

| File | Purpose |
|------|---------|
| `src/data/sync_pairs.py` | Shifted alignment, target construction, pair builders |
| `src/data/lavdf_dataset.py` | LAV-DF training data path (mel, landmarks, timing, shift metadata) |
| `src/models/heads/sync_head.py` | Sync head module |
| `src/losses/sync_loss.py` | BCE sync loss |
| `src/evaluation/sync_metrics.py` | Sync evaluation metrics |
| `src/training/sync_trainer.py` | SyncTrainer and SyncModel for Phase 11 training |
| `scripts/train_sync.py` | CLI entry point for sync training |
| `tests/test_sync_head.py` | Unit tests (head, loss, targets, shift alignment) |
| `tests/test_sync_integration.py` | End-to-end integration test |
| `tests/test_sync_trainer.py` | Unit tests for sync trainer |
| `scripts/test_sync_trainer.py` | Smoke test for sync trainer |
| `tests/test_lavdf_dataset.py` | Dataset + split isolation tests |
| `configs/av_align.yaml` | Phase 11 configuration |

## Known Limitations

1. **Temporal granularity:** 32 video tokens per clip (~1.3s at 25 FPS)
2. **Audio-only shifts:** Video timeline is fixed; only audio correspondence is shifted
3. **No cross-sample negatives:** Same-clip pairing only
4. **Prerequisite extraction:** LAV-DF requires pre-extracted WAV + landmark NPZ files
5. **Frozen encoders:** Phase 11 baseline freezes Phase 5 audio and Phase 8 visual encoders; only cross-attention and sync head are trained

## Consequences

### Positive
- Valid self-supervised sync training: model receives different temporal correspondence for pos/neg
- Clear separation between manipulation and synchronization
- Reuses Phase 2A audio preprocessing and Phase 7 landmark preprocessing
- Deterministic windowing compatible with fixed-sequence model

### Negative
- Simple binary targets (no per-window overlap-based labels)
- Requires pre-extracted audio and landmarks for LAV-DF
- No training script in this phase
