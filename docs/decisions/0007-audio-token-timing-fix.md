# Decision 0007: Audio-Token Timing Fix (Production AV-Inference Correctness Audit)

## Context

A production AV-inference correctness audit investigated the claim that
`src/inference/predictor.py` hardcoded `time_downsample = 1` for the audio
encoder's temporal downsampling factor, while `AudioCNNEncoder` (see
`src/models/audio/cnn.py`) actually performs `2 ** len(audio_cnn_channels)`
downsampling - `8` for the production encoder (`audio_cnn_channels: [32, 64, 128]`,
trained in run `spoof-transformer-20260906-123646`).

## Root cause (confirmed)

`export_audio_encoder` (`src/models/audio/encoder.py`) never stores a
`time_downsample` key in its payload. Five call sites nonetheless read
`audio_payload.get("time_downsample", 1)`, which is dead code that always
silently returns the default `1` regardless of the encoder's real depth:

- `src/inference/predictor.py` (production; one branch also hardcoded the
  literal `1` directly, with no payload lookup at all)
- `scripts/train_sync_physical.py` (trains the current production checkpoint)
- `scripts/evaluate_physical_sync.py` (the held-out evaluation script)
- `scripts/train_sync.py` / `scripts/evaluate_sync.py` (the old Phase 11/12
  token-shift pipeline, left unmodified - see below)

With `hop_length=160`, `sample_rate=16000`, the bug computes
`audio_token_seconds = 160 * 1 / 16000 = 0.01s`; the architecturally correct
value is `160 * 8 / 16000 = 0.08s` (matching `AudioEncoder.time_downsample`
and the value already documented correctly in
`src/models/fusion/temporal_align.py`'s module docstring and correctly
*implemented* in `src/models/fusion/av_encoder.py`'s
`_audio_token_seconds_from_payload`, which reads `encoder.time_downsample`
directly rather than the payload).

**Critical nuance**: this is not a production-only bug. Because
`scripts/train_sync_physical.py` and `scripts/evaluate_physical_sync.py` use
the identical broken pattern, the currently deployed checkpoint
(`outputs/runs/sync-physical-v2-20260927-020548`) was trained AND evaluated
AND served in production under the SAME wrong `0.01s` convention - training
and production agreed with each other, just on a value that didn't reflect
either input's real timing. Fixing only `predictor.py` in isolation would
have broken that agreement rather than restored it.

## Fix

Added `audio_token_seconds_from_encoder(encoder, audio_cfg)` to
`src/models/audio/encoder.py` as the single source of truth (derives
`time_downsample` from the live, loaded encoder module's own attribute -
`2 ** len(channels)` - never from a stored scalar). Applied uniformly to:

- `src/inference/predictor.py` (production, both loading branches)
- `src/models/fusion/av_encoder.py` (refactored its existing private
  duplicate to reuse the shared helper - already correct, now non-duplicated)
- `scripts/evaluate_physical_sync.py`
- `scripts/train_sync_physical.py`

`scripts/train_sync.py` / `scripts/evaluate_sync.py` (the old Phase 11/12
token-shift pipeline, explicitly preserved for old-vs-new comparison per
Decision 0006) were left untouched: that pipeline pairs with a different,
older checkpoint (`sync-phase12-lambda01-20260913-115101`) outside this
audit's scope, and the project's own convention is to keep it frozen for
comparison.

## Empirical validation (held-out LAV-DF dev set, `sync-physical-v2`, 7413 windows)

| | Before fix (0.01s/token) | After fix (0.08s/token) |
|---|---|---|
| Pooled video-level ROC-AUC (native vs. all shifts) | 0.8222 (reproduces the previously reported ≈0.822) | see final report |
| Demo `video/sync` (25 pairs) | 21/25 predicted sync (84%) | 23/25 (92%) |
| Demo `video/desync` (25 pairs) | 24/25 predicted desync (96%) | 25/25 (100%) |

The fix did not degrade held-out or demo-set discrimination - see the
session's final report for the complete before/after numbers and the
training-vs-production parity result.

## Separate finding surfaced during parity testing (not fixed here)

`src/training/physical_sync_trainer.py`'s `PhysicalSyncModel` has no `train()`
override to keep `audio_encoder`/`visual_encoder` pinned to `eval()` mode
(unlike `src/models/fusion/av_encoder.py`'s `AVEncoder`, which explicitly does
this). The base `Trainer._train_epoch` calls `self.model.train()` every
epoch, which - absent that override - flips the "frozen" encoders back into
train mode for every training batch: their `BatchNorm2d` running statistics
drift away from the canonical exported checkpoint, and internal `Dropout2d`
becomes stochastically active, for the entire training run. Confirmed
empirically: `sync-physical-v2/checkpoints/best.pt`'s embedded
`audio_encoder.*` weights differ from the canonical
`spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt` export by a
max absolute difference of ~11,925 (validation/eval-time forward passes are
unaffected - `_val_epoch` does call `self.model.eval()` - and production
correctly never loads the encoder from the sync checkpoint at all, so this
does not affect production correctness). It does mean `cross_attention` and
`sync_head` were optimized against noisier, drifting encoder features than
production ever actually serves them at inference. Not fixed in this audit
per its explicit no-retraining-without-justification scope; recorded here as
the specific defect a future retraining pass should fix first (add a
`train()` override to `PhysicalSyncModel` mirroring `AVEncoder`'s pattern).
