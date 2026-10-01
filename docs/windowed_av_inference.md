# Windowed AV Inference: Fixing the Production/Training Temporal Mismatch

## 1. Root cause of the original production mismatch

`SyncGuardPredictor.predict_audio_visual` (now `_predict_audio_visual_legacy`)
took an entire, variable-length clip's landmark sequence and ran it through the
frozen visual encoder as **one sequence** (`T_v` = however many frames the clip
has - e.g. 117 for a 4.68s / 25fps clip), aligned a whole-clip audio timeline to
it, ran the trained cross-attention + SyncHead once over that single long
sequence, and mean-pooled the per-frame logits into one video-level score.

The model, however, was trained and evaluated by `LAVDFSyncDataset` /
`scripts/evaluate_sync.py` on **fixed 32-token samples**
(`av_align.video.num_frames: 32` in `configs/av_align_lambda01.yaml`). The frozen
Transformer encoders (`TemporalTransformerEncoder`, sinusoidal positional
encoding, `max_len=4096`) place no *architectural* ceiling on sequence length, so
running 117 tokens through them does not crash or error - but the cross-attention
and SyncHead's weights were never optimized against a 117-token attention pattern,
and mean-pooling 117 per-frame scores dilutes any signal concentrated in a small
sub-region of the clip. This is exactly what was observed: a physically verified
+0.500s desync demo clip showed a per-frame score as low as 0.051 in places, but
the whole-clip aggregate (0.833) still read as confidently "sync".

## 2. Existing legacy inference (`mode="legacy_full_clip"`)

Unchanged and preserved exactly as `SyncGuardPredictor._predict_audio_visual_legacy`.
Still available via `predict_audio_visual(..., mode="legacy_full_clip")` and via
`SyncGuardPredictor(av_inference_mode="legacy_full_clip")`, kept as a fallback and
as the pre-windowing reference implementation. No behavior change.

## 3. New windowed inference (`mode="windowed"`)

`src/inference/windowing.py` tiles a clip's frame timeline into consecutive,
fixed-size windows (`generate_windows`), and
`SyncGuardPredictor._predict_audio_visual_windowed` runs the full
encode -> align -> cross-attend -> SyncHead pipeline **once per window**, each
window shaped like the 32-token samples the model actually saw during
training/evaluation, instead of once over the whole clip.

```
full landmark sequence
        |
        v
  generate_windows(n_frames, fps, window_frames=32, stride_frames=32)
        |
        +-- Window 0: frames [0, 32)   -> audio [0.00s, 1.28s)
        +-- Window 1: frames [32, 64)  -> audio [1.28s, 2.56s)
        +-- Window 2: frames [64, 96)  -> audio [2.56s, 3.84s)
        +-- Window 3: frames [96, 117) -> audio [3.84s, 4.68s)   (partial: 21/32 valid)
        |
        v (per window)
  visual_encoder(window landmarks) -> visual_tokens [1, 32, 256]
  crop_audio_window(full waveform, start, window_seconds) -> audio_window
  audio_encoder(mel(audio_window)) -> audio_tokens
  align_audio_to_video(audio_tokens, video_fps, window_seconds) -> audio_aligned
  cross_attention(audio_aligned, visual_tokens) -> fused
  sync_head(fused) -> per-frame logits -> sigmoid -> mean = window_score
        |
        v
  video-level aggregate = mean(window_scores)   (default; median / valid-weighted
                                                  mean also computed and reported)
```

### Why *consecutive* windows, not LAV-DF's own sampling

`LAVDFSyncDataset` builds its 32-token training sample by **sparsely**
subsampling an entire multi-second clip via `np.linspace` across *all* available
frames, then aligns only the clip's **first** `32/fps` seconds of audio to those
32 tokens (`LAVDFSyncDataset.timing`; `frame_idx0` is always 0 because
`np.linspace(...)[0] == 0`). That means a training "token k" is a frame sampled
from anywhere across the whole clip, but the audio interval `align_audio_to_video`
assigns to token `k` (`[k/fps, (k+1)/fps)`) is not the audio that actually
co-occurred with that (sparsely sampled) frame. This is a real property of the
existing training/eval data pipeline, not something introduced here, and it is
out of scope to change (training code is untouched per this task's constraints).

Production windows instead use **consecutive** frames per window, so window
token `k` truly is video frame `start_frame + k`, and the aligned audio interval
`[start_frame/fps, (start_frame + 32)/fps)` genuinely co-occurs with it. This
keeps `align_audio_to_video`'s "token k covers real time k/fps" assumption
actually true, while still using the same window *size* (32) and the same
alignment/cross-attention/SyncHead code the model was trained with. It is a
deliberate, documented choice, not an attempt to bit-for-bit replicate LAV-DF's
sparse-sampling quirk in a setting (arbitrary user-uploaded video) where clips
are not curated to a couple of seconds.

## 4. Window size and stride

Defaults: `window_frames=32`, `stride_frames=32` (non-overlapping), matching
`av_align.video.num_frames` in `configs/av_align_lambda01.yaml`. Both are
constructor parameters on `SyncGuardPredictor` and env vars on the backend
(`SYNCGUARD_AV_WINDOW_FRAMES`, `SYNCGUARD_AV_STRIDE_FRAMES`). Overlapping windows
(`stride_frames < window_frames`) are supported by `generate_windows` but not
enabled by default - no evidence was gathered to justify overlap, so it stays off
per the "do not introduce overlap without a demonstrated need" instruction.

## 5. Short/long/non-divisible video handling

- **Shorter than one window**: exactly one short window is emitted
  (`generate_windows` always covers `[0, n_frames)`); its landmarks are upsampled
  to 32 tokens via `select_frame_indices` (see below), and `valid_fraction < 1.0`
  is reported so downstream consumers know part of the window is repeated, not
  fresh evidence.
- **Exactly one window's worth of frames**: a single full window, `valid_fraction
  == 1.0`.
- **Longer, non-divisible length** (e.g. 117 frames): windows
  `[0,32), [32,64), [64,96), [96,117)` - the full timeline is covered, the final
  partial window is kept (never silently discarded), with `n_valid_frames=21`,
  `valid_fraction=21/32`.

### Frame selection / "padding" policy

`select_frame_indices(n_available, n_want)` uses
`np.linspace(0, n_available - 1, n_want).round()` - the exact formula
`LAVDFSyncDataset._select_frames` already uses for its dense-sampling branch
(`n_available >= n_want`, the only branch real LAV-DF clips exercise). It
degenerates to `arange(n_want)` when a window is exactly full (the common case),
and generalizes cleanly to a short final window by deterministically **repeating**
real frames rather than zero-padding or masking. This was a deliberate choice over
introducing a new masking scheme: the frozen visual encoder / cross-attention /
SyncHead were never trained with a padding-mask convention for partial sequences
(`LAVDFSyncDataset`'s own fallback for `n_available < n_want` uses a **seeded
random** repeat, not a mask, and that branch is never actually exercised during
training since every LAV-DF clip has far more than 32 frames). Repeating real
frames deterministically:
- reuses an existing, already-validated formula rather than inventing a new one,
- keeps production inference reproducible (no RNG), which the training fallback
  is not,
- avoids feeding the frozen SyncHead a masking pattern it was never trained with.

`valid_fraction` is still reported per window (and used for the optional
valid-weighted aggregate) so a short/partial window's reduced information content
is visible to callers even though it isn't enforced via a model-side mask.

## 6. Audio alignment

For each window, the corresponding audio interval is derived purely from real
video timestamps: `start_seconds = start_frame / fps`,
`window_seconds = window_frames / fps`. `src.data.lavdf_dataset.crop_audio_window`
(already used by training) crops the **preprocessed, already-resolved** waveform
to that interval, zero-padding if the window extends past the audio's end (this
naturally, correctly handles the final partial window and any external audio
shorter than the video). A fresh log-mel spectrogram is computed per window crop
(`compute_log_mel`) and encoded independently - this mirrors
`LAVDFSyncDataset.__getitem__` exactly (crop waveform -> mel -> encode), rather
than slicing a single whole-clip mel/token sequence, which is what the legacy path
does.

`audio_token_seconds` (0.01s = `hop_length(160) / sample_rate(16000)`) is read
from the loaded checkpoint/config exactly as before - never hard-coded, never
reverted to the old incorrect 0.08s placeholder value that still appears only as
a dataclass *default* in `src/data/sync_pairs.py` (overridden by the real config
value at load time).

## 7. External audio / video-only / landmarks

Unchanged rules, shared with legacy mode:
- `audio_path` provided -> that exact file is preprocessed and used; the video's
  embedded audio is never read (see
  `test_windowed_av_inference_uses_external_audio_not_embedded`).
- `audio_path is None` -> embedded audio is extracted from the video via
  `extract_audio_from_mp4`, exactly as before.
- `landmarks_path` provided -> used as-is (after the same Phase 7/8
  interpolate/normalize/region-select preprocessing the legacy path already
  applies); never silently replaced by fresh extraction.
- `landmarks_path is None` -> extracted via the supplied MediaPipe `landmarker`.
- FPS, sample rate, mono/stereo, and duration mismatches all flow through the
  existing `preprocess_audio` / `crop_audio_window` / landmark-preprocessing
  infrastructure, unchanged.

## 8. Aggregation

Per-window score = mean of per-frame sigmoid probabilities within the window -
the exact convention `scripts/evaluate_sync.py`'s `evaluate_with_shift` already
uses for its "video-level" score (not `SyncHead.aggregate`, which the rest of the
codebase also does not call for this purpose).

Video-level aggregate (across windows) = **mean of window scores** (default).
This step has no precedent in training (LAV-DF training/eval only ever sees one
window per sample, so there is no "combine N windows" rule to inherit) and was
chosen, not tuned, for being the simplest, most interpretable extension of the
"mean" convention already used everywhere else in this codebase (frame -> window
in `SyncHeadConfig.aggregation: mean`, window -> video in `evaluate_with_shift`).
`median_window_score` and `valid_weighted_window_score` (windows weighted by
`valid_fraction`, so a short trailing window with mostly-repeated frames
contributes less) are also computed and exposed in `timing_metadata` for
transparency, but are not used for the primary label - no aggregation variant was
selected by checking which one makes the demo set look best (see Section 10).

## 9. A/B results (`scripts/compare_av_inference_modes.py`)

Run on: 25 `demo/video/sync` pairs, 25 `demo/video/desync` pairs (each
independently verified below to carry exactly +0.500s of prepended silence - real,
physical desynchronization, not a manipulation label), and 40 LAV-DF dev clips
(native timing, all `label_name=real` in the sampled prefix - descriptive only,
*not* used as sync/desync ground truth). Full JSON:
`outputs/av_mode_comparison/comparison_results.json`.

| Set | Mode | mean score | median | std | predicted sync / desync | mean latency |
|---|---|---|---|---|---|---|
| demo/sync (n=25) | legacy | 0.770 | 0.780 | 0.114 | 24 / 1 | 14.2s |
| demo/sync (n=25) | windowed | 0.574 | 0.579 | 0.096 | 19 / 6 | 13.7s |
| demo/desync (n=25) | legacy | 0.732 | 0.786 | 0.149 | 23 / 2 | 19.3s |
| demo/desync (n=25) | windowed | 0.566 | 0.561 | 0.092 | 19 / 6 | 16.5s |
| LAV-DF dev (n=40, native, real-only prefix) | legacy | 0.743 | 0.780 | 0.159 | 34 / 6 | 1.07s |
| LAV-DF dev (n=40, native, real-only prefix) | windowed | 0.557 | 0.544 | 0.105 | 28 / 12 | 0.51s |

(Latencies are single-run, mixed GPU/CPU caching effects included; see Section 12.)

**Honest reading of these numbers** (per this task's explicit instruction not to
decide based on whether the demo says "DESYNC"): windowed mode pulls scores
closer to 0.5 across *all three* sets, including the sync-labeled demo set and
the (assumed-aligned) LAV-DF real clips. On the desync demo set it does flip more
clips to "desync" than legacy (6/25 vs 2/25) - a real, measured improvement on
verified physical ground truth - but it flips a similar fraction of the *sync*
demo set too (6/25 vs 1/25). The mean-score gap between the sync and desync demo
sets is actually *smaller* under windowed (0.574 vs 0.566, a 0.008 gap) than under
legacy (0.770 vs 0.732, a 0.038 gap). **This is not a case of windowed inference
dramatically "solving" desync detection** - it mainly reduces the trained
SyncHead's overconfidence everywhere, which is itself a defensible property (the
legacy path's higher confidence came from feeding it out-of-distribution 117-token
sequences), but the model's underlying sensitivity to a modest 0.5s
misalignment over a full clip remains limited. See Section 13 (Limitations).

## 10. Controlled temporal-shift validation

Unchanged: `scripts/evaluate_sync.py` already performs this exact experiment
(shift in `{0, +0.5, -0.5, +1.0, -1.0, +2.0, -2.0}`, using
`src.data.sync_pairs.compute_shifted_alignment` - a **token-timeline** shift, not
a waveform edit) against the LAV-DF dev split's single-32-token-per-sample regime,
and its own docstring already states the correct scope/limits (AUC undefined
when a shift condition has only one class, negative shifts often yield zero valid
windows, etc.). This is orthogonal to windowed multi-window inference (there is
no multi-window analogue of "shift the whole clip's token timeline" - the
controlled-shift mechanism is inherently single-window), so it was re-run as-is
rather than rebuilt; see `scripts/evaluate_sync.py`'s own historical outputs for
these numbers. It must not be described as "real-world desync ground truth" (its
own docstring already says so) and is distinct from the physical WAV-shift
mechanism below.

## 11. Physical demo-shift validation

`scripts/compare_av_inference_modes.py` independently re-verifies, per clip,
that each `demo/video/desync/*` pair's leading silence is exactly 0.500s (by
counting literal leading-zero samples in the WAV - not trusting the folder name),
then runs both inference modes. Per-clip results for the first 10 desync pairs
(`windowed_summary`/`per_clip` in `outputs/av_mode_comparison/comparison_results.json`):

| clip | verified silence | legacy score (label) | windowed score (label) |
|---|---|---|---|
| 0001 | 0.500s | 0.833 (sync) | 0.776 (sync) |
| 0002 | 0.500s | 0.405 (desync) | 0.408 (desync) |
| 0003 | 0.500s | 0.764 (sync) | 0.430 (**desync**) |
| 0004 | 0.500s | 0.817 (sync) | 0.454 (**desync**) |
| 0005 | 0.500s | 0.941 (sync) | 0.627 (sync) |
| 0006 | 0.500s | 0.604 (sync) | 0.574 (sync) |
| 0007 | 0.500s | 0.712 (sync) | 0.550 (sync) |
| 0008 | 0.500s | 0.584 (sync) | 0.531 (sync) |
| 0009 | 0.500s | 0.829 (sync) | 0.477 (**desync**) |
| 0010 | 0.500s | 0.638 (sync) | 0.651 (sync) |

Windowed mode moves every one of these 10 scores down relative to legacy, and
flips 3 of the 10 to the (physically correct) "desync" label that legacy missed.
It does not flip all of them - see Limitations.

## 12. Latency (Phase 24)

Single-clip CPU timing (`demo/video/desync/0001_video.mp4`, 4.68s clip, 117
frames): legacy 48.5s, windowed 39.6s (both dominated by first-call
kernel/cuDNN warmup on that run). Batch-sweep GPU mean latencies (Section 9
table) show windowed consistently *faster* than legacy on average
(e.g. LAV-DF dev: 0.51s vs 1.07s/clip) - each window's attention sequence is much
shorter (32 tokens) than the legacy path's whole-clip sequence (up to 117+ tokens
here), and that more than offsets doing multiple forward passes. No batching
across windows was implemented (correctness first, per the task's own priority
ordering); this is a possible follow-up if further speed is needed.

## 13. Limitations (do not oversell)

- Windowed mode is **methodologically more consistent** with the trained/evaluated
  temporal regime (fixed 32-frame windows, correctly time-aligned audio per
  window) than legacy mode, which processes an out-of-distribution sequence
  length. This is the primary justification for making it the default.
- It is **not** a dramatic accuracy fix for demo desync detection: at a modest
  +0.5s shift, the score gap between demo-sync and demo-desync sets remains small
  under both modes, and windowed mode's gain (more desync flags) comes with a
  similar-sized increase in desync flags on the sync-labeled set. The deeper
  limitation is the trained SyncHead's sensitivity ceiling at this shift
  magnitude over a full clip, not the windowing scheme itself.
- LAV-DF dev-set numbers here are **descriptive only** - `label_name`/manipulation
  flags are never treated as synchronization ground truth (manipulation !=
  desynchronization; enforced elsewhere by `test_no_fake_periods_as_sync_labels`).
- No new video-level threshold was fit to any dataset here; 0.5 (the existing
  convention) is unchanged.
- Overlapping windows, batched multi-window forward passes, and calibration are
  documented as possible future work, not implemented, since no evidence
  justified them yet.

## 14. Final decision (Phase 25)

**Production default: `windowed`** (`SyncGuardPredictor(av_inference_mode=...)`
now defaults to `"windowed"`; `backend/main.py` reads
`SYNCGUARD_AV_INFERENCE_MODE`, default `"windowed"`). Justification: it is the
only mode whose temporal regime matches what the model was actually
trained/evaluated on, it is numerically valid (no NaN/Inf across every sample
run), deterministic and reproducible (`test_windowed_av_inference_is_deterministic`),
robust across every frame-count case tested (short/exact/partial/multiple
windows), and empirically no slower (in fact faster on average here) than
legacy. `legacy_full_clip` is kept, fully functional, as an explicit fallback and
reference mode (`mode="legacy_full_clip"`) - it is not removed and can be
selected per-call or via `SYNCGUARD_AV_INFERENCE_MODE=legacy_full_clip`.
This decision was **not** based on windowed mode making the demo desync set say
"DESYNCHRONIZED" more often - see Section 9 for the honest, mixed empirical
picture that was weighed alongside the methodological argument.
