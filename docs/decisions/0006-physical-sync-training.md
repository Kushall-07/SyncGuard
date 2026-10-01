# Decision 0006: Physical-Shift Synchronization Training (Sync-Detection Fix)

## Context

Production `SyncGuardPredictor` (windowed mode, see `docs/windowed_av_inference.md`)
frequently predicted SYNC for deliberately desynchronized video/audio pairs
(`demo/video/desync/*`, each carrying a real, physical, 0.500s prepended-silence
shift). `docs/av_aggregation_investigation.md` traced this to the SyncHead
itself, not to aggregation: under the production windowed regime, the trained
model's per-window scores barely separate native (assumed-synchronized) clips
from verified physically-desynchronized ones (Section 2A/2B/2D of that doc).

## Root cause

Traced in full in this decision's companion audit (see conversation / final
report): the Phase 11/12 SyncHead (`docs/decisions/0005-sync-head.md`) was
trained on a **token-space** negative construction
(`src.data.sync_pairs.compute_shifted_alignment`) that re-buckets
*already-encoded* audio tokens onto a shifted time grid. The frozen audio
encoder's input (the mel spectrogram, and the waveform beneath it) is **never
actually shifted** - only the token-to-video-frame correspondence map is. This
is architecturally different from the production condition (`demo/video/desync/*`:
same video, a physically time-shifted WAV file, encoded and aligned normally).
Compounding this, `LAVDFSyncDataset` builds its 32-token training sample by
**sparsely** subsampling frames across an entire multi-second clip
(`np.linspace`), while production's windowed inference (`src/inference/windowing.py`)
uses **consecutive** 32-frame windows - a second, independent train/inference
regime mismatch (documented in `src/inference/windowing.py`'s own module
docstring and re-confirmed empirically in `docs/av_aggregation_investigation.md`
Section 3/4: the token-shift experiment shows near-perfect sensitivity on
LAV-DF's sparse-sampling regime, but that sensitivity does not transfer to the
real, consecutive-frame, physically-shifted production condition).

## Decision

Train a new SyncHead + cross-attention checkpoint (architecture unchanged) on
corrected data:

1. **Physical waveform shift** (`src/data/audio_shift.shift_waveform`) applied
   to the raw audio *before* the mel spectrogram / frozen audio encoder, not a
   token-timeline reassignment after encoding. Duration- and sample-rate
   preserving, deterministic, never mutates the source waveform.
2. **Consecutive, production-style 32-frame/stride-32 windows**
   (`src/inference/windowing.generate_windows` / `select_frame_indices` -
   reused, not reimplemented) instead of LAV-DF's sparse whole-clip sampling.
3. **Cross-clip negatives** (video from clip A + native audio from a different
   clip B, same split) as an additional, secondary hard-negative category
   alongside physical-shift, tracked via `negative_type` metadata
   (`physical_shift` | `cross_clip`), never fed to the model.
4. **Multiple shift magnitudes and both directions**: {0.25, 0.5, 0.75, 1.0}s,
   sign randomized, plus zero-shift positives - so the model cannot solve the
   task by learning "any leading silence => desync" alone.
5. Manipulation labels (`label_name`/`modify_audio`/`modify_video`/
   `fake_periods`) remain metadata-only, exactly as Decision 0005 requires -
   never used to derive the sync/desync target.

New code lives alongside the original Phase 11/12 implementation, which is
**left unmodified** so the two can be A/B compared:

| Concern | Original (Phase 11/12, unchanged) | Physical-shift fix (new) |
|---|---|---|
| Negative construction | `src.data.sync_pairs.compute_shifted_alignment` (token-space) | `src.data.audio_shift.shift_waveform` (waveform-space) |
| Dataset | `src.data.lavdf_dataset.LAVDFSyncDataset` (sparse whole-clip sampling) | `src.data.lavdf_physical_sync_dataset.LAVDFPhysicalSyncDataset` (consecutive production windows) |
| Model/trainer | `src.training.sync_trainer.SyncModel` / `SyncTrainer` | `src.training.physical_sync_trainer.PhysicalSyncModel` / `PhysicalSyncTrainer` |
| Config | `configs/av_align_lambda01.yaml` | `configs/av_align_physical.yaml` |
| Train script | `scripts/train_sync.py` | `scripts/train_sync_physical.py` |
| Eval script | `scripts/evaluate_sync.py` (token-shift) | `scripts/evaluate_physical_sync.py` (physical-shift, section 19 categories) |

Architecture is **unchanged**: `BidirectionalCrossAttention` and `SyncHead`
(same classes, same configs `dim=256, num_heads=4, hidden_dim=128`), the audio
and visual encoders remain frozen and untouched, and the loss remains
`SyncLoss` (masked BCE). Only the training data construction and the resulting
checkpoint change.

## Data availability limitation

Only LAV-DF `train` (4000 clips) and `dev` (1000 clips) are present locally
(`data/lavdf/manifest_{train,dev}.csv`); no `manifest_test.csv` / held-out LAV-DF
`test` split was downloaded. Training uses `train` only; all validation and the
physical-shift evaluation in this decision use `dev`, which the training
pipeline never sees gradients from - a genuine held-out set for this purpose,
though not the official LAV-DF `test` split. This is documented, not
papered over, per the task's own "do not hide failures" requirement.

## Consequences

See the final report (delivered separately) for the actual validation results,
physical-shift evaluation, demo evaluation, and old-vs-new comparison this
decision produced.
