# AV Video-Level Aggregation Investigation

**Scope:** whether the current video-level aggregation rule (mean of window scores,
threshold 0.5) is appropriate, and whether the trained SyncHead's per-window output
contains evidence to support a better rule. Analysis only. No model, training,
checkpoint, SyncHead, cross-attention, or audio/visual encoder change was made.

## 1. Current aggregation implementation (as-built, unmodified)

Traced in `src/inference/predictor.py`, `src/inference/windowing.py`,
`scripts/evaluate_sync.py`, `src/models/heads/sync_head.py`, `backend/main.py`,
`frontend/src/lib/timeline.ts`, `frontend/src/pages/Analyze.tsx`.

- **Per-frame -> window score**: `SyncHead` outputs one logit per video token in the
  window; `_predict_audio_visual_windowed` takes `sigmoid(logits).mean()` over the
  window's tokens (`predictor.py:836`). This is the same "mean of per-frame sigmoid
  probabilities" convention `scripts/evaluate_sync.py`'s `evaluate_with_shift` uses,
  and is *not* `SyncHead.aggregate()` (that method exists, defaults to `"mean"`, but
  is not called anywhere in the inference path).
- **Window -> video score**: `aggregate_sync_score = mean(per_window_scores)`, an
  **unweighted arithmetic mean across windows** (`predictor.py:852`). A
  `median_window_score` and a `valid_weighted_window_score` (windows weighted by
  `valid_fraction`, so a short trailing window counts less) are also computed and
  placed in `timing_metadata` for transparency, but **neither is used for the
  label** — confirmed by reading `predictor.py:871-873`, where `sync_prob =
  aggregate_score` (the plain mean) directly.
- **Threshold**: `predicted_label = "sync" if aggregate_score >= 0.5 else "desync"`
  (`predictor.py:873`), applied once, in Python, directly to the plain mean.
- **Valid-window weighting**: computed but unused for the decision (see above).
- **Window duration**: all windows are nominally the same duration
  (`window_frames/fps`) except a possible short final window; that shorter window's
  reduced information content is *not* down-weighted in the actual decision, only
  in the unused `valid_weighted_window_score`.
- **UI parity**: `backend/main.py`'s `/api/analyze/av` calls
  `predictor.predict_audio_visual(...)` and returns `asdict(result)` verbatim — no
  threshold or label logic in the backend. The frontend (`Analyze.tsx`,
  `Results.tsx`) displays `result.predicted_label` / `result.aggregate_sync_score`
  exactly as received. `frontend/src/lib/timeline.ts` separately defines
  `SYNC_THRESHOLD = 0.5`, but only to color/annotate the timeline chart — it does
  not feed back into the label. This is a second, independent hardcoded copy of
  the same 0.5 value; currently consistent with the backend but would silently
  drift if the backend threshold ever changed without a matching frontend edit.

## 2-8. Empirical evidence (real inference, three explicitly separate mechanisms)

Full machine-readable data: `outputs/aggregation_analysis/report.json` (this
investigation's new analysis script, `scripts/analyze_aggregation_evidence.py`,
analysis-only, not called from production) plus the pre-existing
`outputs/av_mode_comparison/comparison_results.json` and
`outputs/analysis/av_independent_native_dev/independent_av_results.json`.

### 2A/2B/6. Demo sets — production windowed predictor, real inference

25 `demo/video/sync` pairs and 25 `demo/video/desync` pairs (each independently
re-verified here to carry exactly 0.500s of leading digital silence — genuine
physical desynchronization, not a manipulation label or a token-level shift), run
through the actual `mode="windowed"` predictor.

| candidate statistic | demo/sync mean (std) | demo/desync mean (std) | separation |
|---|---|---|---|
| mean (current rule) | 0.574 (0.096) | 0.566 (0.092) | none |
| median | 0.567 (0.110) | 0.549 (0.109) | none |
| min | 0.461 (0.112) | 0.426 (0.094) | weak |
| p10 | 0.487 (0.107) | 0.450 (0.094) | weak |
| p25 | 0.525 (0.108) | 0.484 (0.102) | weak |
| duration-weighted mean | 0.571 (0.097) | 0.569 (0.096) | none |
| mean + lower-tail (0.5·mean+0.5·p10) | 0.531 (0.100) | 0.508 (0.091) | weak |
| frac. windows < 0.5 | 0.278 | 0.420 | weak |

The one clip in the prompt's worked example (`0001_video`) reproduces exactly:
windows `[0.825, 0.819, 0.796, 0.664]` -> mean `0.7762` -> `sync`, matching
`docs/windowed_av_inference.md` Section 11's own number for this clip.

### 2D/5. Native LAV-DF descriptive subset — production windowed predictor, shift=0

60 native-timing LAV-DF dev clips (mixed manipulation categories — manipulation
label is irrelevant here, only used to confirm this isn't accidentally the same
subset as a labeled desync set):

| candidate statistic | native mean (std) | frac. of *native* clips below 0.5 |
|---|---|---|
| mean (current rule) | 0.548 (0.101) | 0.350 |
| median | 0.538 (0.107) | 0.450 |
| min | 0.417 (0.108) | **0.817** |
| p10 | 0.446 (0.104) | **0.767** |
| p25 | 0.485 (0.106) | 0.600 |
| frac. windows < 0.5 | 0.404 | 0.517 |

A separate, previously-generated 1000-sample native-timing evaluation
(`outputs/analysis/av_independent_native_dev/independent_av_results.json`, a
different, finer-grained windowing config) shows the *legacy-style* aggregate is
extremely tight at native alignment (mean ≈0.99, std ≈0.001-0.003, all four
manipulation categories indistinguishable) — i.e. the mismatch above is specific to
the production **windowed** regime's per-window noise floor, not a property of the
frozen SyncHead in general.

**This is the central finding.** Under the actual `mode="windowed"` production
path, min/p10 (lower-tail candidates) flag a *larger* fraction of assumed-
synchronized native clips as "desync" (76.7-81.7%) than they correctly flag
verified-physical-desync demo clips as "desync" (60-76%, from the table above).
None of the nine candidate statistics tested separates demo-sync from demo-desync
from native-descriptive in a way that would support a better binary rule than the
current mean.

### 3/4. Candidate aggregation vs. CONTROLLED TEMPORAL-SHIFT VALIDATION

Using the existing controlled-shift mechanism (`src/data/sync_pairs.build_sync_pair`
/ `compute_shifted_alignment`, the same primitive `scripts/evaluate_sync.py` and
`predict_sync_lab` use), on LAV-DF dev's dense 32-token training-matched sampling
(`build_lavdf_datasets`, `n_video_tokens=32`), 60 samples, shifts `{0, +0.5, +1.0,
+2.0}`s:

| shift | mean (video-mean, std) | frac. videos with mean<0.5 | frac. videos with p10<0.5 |
|---|---|---|---|
| 0.0s | 0.9912 (0.0013) | **0.000** | 0.000 |
| +0.5s | 0.2352 (0.1997) | **0.900** | 0.950 |
| +1.0s | 0.1476 (0.1762) | **0.933** | 0.967 |
| +2.0s | n/a | n/a (0 valid videos — see note) | |

At `+2.0s`, every sample produced zero overlapping audio/video buckets (the
sample-level audio window here is only `32/fps ≈ 1.28s` long, per
`LAVDFSyncDataset.timing`, so a 2.0s shift pushes the audio entirely outside the
video window regardless of `empty_bucket="nearest"`). This is the same documented
limitation `scripts/evaluate_sync.py`'s own docstring already states for large
shifts, just triggered here for a positive shift because this sample-level audio
window is short, not because of a bug.

**Every candidate statistic** (mean, median, min, p10, p25, duration-weighted mean,
mean+lower-tail) shows the same pattern here: essentially zero false "desync"
evidence at shift=0, and 90-97% of videos crossing below 0.5 at shift≥0.5s. Taken
at face value this looks like strong, clean, monotonic sensitivity — dramatically
stronger than anything seen in Section 2A/2B/2D above.

**This strength does not transfer to production.** `LAVDFSyncDataset` builds its
32-token sample by *sparsely* subsampling frames across the *entire* clip
(`np.linspace` over all available frames — a token can be a frame from anywhere in
a ~5s clip) while aligning only the clip's *first* `32/fps` seconds of audio to
those 32 tokens (documented in `src/inference/windowing.py`'s module docstring,
lines 20-35, and in `docs/windowed_av_inference.md` Section 3, written during the
prior windowing work — not a new finding here, but empirically confirmed by this
investigation's numbers). A "shift" of that alignment therefore does not model the
same real-world event as the physical demo desync (real, consecutive frames, real
time-aligned audio, evaluated through `mode="windowed"`) — it is a large,
somewhat artificial jolt to an already-nonstandard sparse alignment. The
controlled-shift experiment is measuring the trained SyncHead's sensitivity to
*that specific token-timeline construction*, not its sensitivity to real
audio-video desynchronization as encountered by the deployed windowed pipeline.
This is exactly the "manipulation != desynchronization" / "controlled shift !=
real-world ground truth" separation this task requires, extended to a second
axis: *controlled-shift ground truth != production-windowed-regime ground truth*.

### 7. Does the 0.81/0.80/0.78/0.68 -> 0.7762 distribution contain enough evidence?

No, not reliably. That distribution (demo/desync clip `0001_video`) has one lower
window (0.664/0.68) but the other three are 0.78-0.83 — a downward trend
(`trend_last_minus_first ≈ -0.16`), but comparable downward trends and comparably
low minimum windows appear routinely in the *native* descriptive set (Section 2D:
mean native min = 0.417, i.e. native clips regularly bottom out well below 0.5 in
their worst window). A single low window, or even a downward trend, is not unusual
enough in assumed-synchronized data to be trustworthy desync evidence on its own.

### 8. Recommendation on forcing a binary label

Per the instructions, since no candidate aggregation cleanly separates verified
physical desync from native/demo-sync under the production windowed regime
(Section 2D), **no new binary rule is justified**. See Section 9/10 for the
resulting decision.

## 9. Candidate aggregation selection

**No candidate aggregation statistic was adopted.** All nine candidates tested
(mean, median, min, p10, p25, frac-below-threshold, duration-weighted mean,
mean+lower-tail, temporal-consistency/std) were evaluated purely as analysis (see
`scripts/analyze_aggregation_evidence.py`; never imported by `predictor.py`,
`backend/main.py`, or the frontend). None satisfies the required validation
criteria (Section 9 of the task): all of them separate the controlled-shift
condition cleanly but *none* separates demo-sync from demo-desync from native
LAV-DF under the actual deployed windowed pipeline (Section 2A/2B/2D) — several
(min, p10) produce *more* false "desync" flags on native data than true "desync"
flags on verified desync data. Per the explicit instruction not to invent a rule
absent validation, **production aggregation remains unchanged**: mean of window
scores, threshold 0.5.

## 10. Threshold analysis

Not performed as a deployment-threshold-selection exercise, per the explicit
instruction not to fit a threshold using only `demo/video/sync` +
`demo/video/desync`, and because no properly labeled, independent desync
validation set exists beyond those demo pairs and the (separately-mechanismed)
controlled-shift experiment. What Section 2D does show: under the current mean
rule, native LAV-DF (assumed synchronized) already crosses below 0.5 in 35% of
sampled clips (21/60 here; 12/40 in the prior `compare_av_inference_modes.py` run,
`docs/windowed_av_inference.md` Section 9) — i.e. the windowed pipeline's own
native false-"desync"-flag rate under the *existing* threshold is already
substantial. Moving the threshold in either direction trades this native
false-positive rate against demo-desync sensitivity along a curve that was not
fit here, per instruction; a scientifically validated deployment threshold cannot
be established from the data available.

## 11. UI

No UI change. Aggregation remains mean; per the task's own instruction ("If
aggregation remains mean: keep current behavior"), `Analyze.tsx` / `Results.tsx` /
`SyncTimeline.tsx` are unmodified. The existing hedge text ("AV mode detects
temporal synchronization inconsistencies... does not directly detect content
manipulation") already avoids "definitely synchronized/fake" language and needs no
change.

## 12. Regression

No production file was changed by this investigation, so this is a sanity check
on the pre-existing (uncommitted) windowed-inference work, not a regression check
on new behavior:

- `pytest tests/test_predictor.py tests/test_inference_windowing.py
  tests/test_backend_main.py`: **59 passed, 1 skipped** (the skip requires a video
  file for a full AV inference test — expected/pre-existing, not caused here).
  Backend tests include `/api/health` readiness (`test_health_ready`,
  `test_health_unavailable_when_predictor_fails`).
- `npm test -- --run` (frontend): **45 passed, 1 failed** in the full-suite run
  (`App.test.tsx`, a `findByRole` heading lookup whose own comment notes it can
  exceed the default 1000ms timeout under load). Re-run in isolation:
  **8/8 passed**. Not a regression — no frontend file was touched by this
  investigation, and this test is unrelated to AV aggregation.
- `npm run build` (frontend): **succeeded**, `tsc -b && vite build` clean, all
  chunks emitted.

## 13-14. Backend E2E / frontend build

No backend or frontend production code changed, so no new E2E risk was
introduced. `/api/health` readiness after repeated AV requests, external-audio
handling, and embedded-audio fallback are exercised by the existing
`tests/test_backend_main.py` / `tests/test_predictor.py` suites (all passing
above); a live-server manual E2E was not run since there is nothing new in that
path to validate.

## 15. Final recommendation and limitations

**Answer to the central question: NO** — the existing SyncHead's per-window output,
as consumed by the actual production windowed pipeline, does not contain enough
validated evidence to support a better video-level desynchronization decision than
simple mean pooling. Leave production inference unchanged.

Distinguishing the three layers explicitly, as required:

- **MODEL BEHAVIOR**: the frozen cross-attention + SyncHead *can* be highly
  sensitive to audio-video timeline shifts — but only in the sparse-sampled,
  short-audio-window regime it was trained/evaluated on (Section 3/4). Whether it
  is *inherently* incapable of resolving a real 0.5s consecutive-frame shift, or
  whether some other representation of the same frozen weights could do better, is
  not resolved by this investigation and would require retraining/architecture
  work explicitly out of scope here.
- **INFERENCE/AGGREGATION BEHAVIOR**: given the model as-is, no post-hoc
  aggregation rule investigated here recovers reliable separation on real,
  windowed, consecutive-frame inference. The mean rule is not worse than any
  tested alternative on the evidence gathered; several alternatives (min, p10)
  are measurably worse (higher native false-positive rate).
- **DEMO DATASET BEHAVIOR**: the demo desync set is genuinely physically
  desynchronized (independently re-verified leading-silence check); the website
  reporting "SYNCHRONIZED" on some of these clips is a real, honestly-measured
  limitation of the current model+aggregation combination on a modest 0.5s shift,
  not a bug in the website, not evidence the demo file is malformed, and not
  something a demo-tuned threshold should paper over.

**Remaining limitations**: (1) no independently-labeled, non-demo desync
validation set exists, so any future threshold or aggregation work needs one
before it can be called validated; (2) the gap between controlled-shift and
physical-demo-shift sensitivity (Section 3/4) is itself worth its own follow-up —
understanding *why* the sparse-sampled regime is so much more shift-sensitive than
the consecutive-frame regime could point toward a genuine future improvement (e.g.
whether training on consecutive-frame windows, which is out of scope here, would
transfer that sensitivity into production), but that is a training-side question,
not an aggregation-side one; (3) `frontend/src/lib/timeline.ts`'s `SYNC_THRESHOLD`
constant duplicates the backend's 0.5 and should be treated as a single source of
truth if the threshold is ever revisited.
