# SyncGuard — Experiment Results

This document is the single reference for what has actually been measured in
this project, on which data, and what those numbers do and do not support. It
complements the per-phase decision records under `docs/decisions/`, which carry
the full methodological detail; this page is the researcher-facing summary.

Nothing here is invented or silently recomputed. Every number is traceable to a
result file under `outputs/` (paths given inline) and, where relevant, to a
decision record.

---

## 1. Audio-only spoof detection (ASVspoof 2019 LA)

Trained and evaluated on the official ASVspoof 2019 LA logical-access protocol
(train 25,380 / dev 24,844 / eval 71,237 utterances, speaker-disjoint, eval
attacks A07–A19 unseen during training). See
`docs/decisions/0001-audio-spoof-detection.md` for the full audit trail.

| Model | AUC | EER |
|---|---|---|
| CNN | 0.9881 | 4.98% |
| CNN + Transformer | 0.9928 | 3.74% |
| CNN + Transformer ensemble (mean, 0.5/0.5) | 0.9978 | 1.93% |

Source: `outputs/analysis/ensemble/ensemble.json`.

The ensemble's worst per-attack EER on the eval set is A17 (voice-conversion
attack) at **≈4.50%**; every other attack is below that. All 13 unseen attacks
are individually below 5% EER under the ensemble. Per-attack EER is computed
against the pooled bonafide set, per ASVspoof convention.

**What these metrics mean.** EER (equal error rate) is the operating point
where false-accept and false-reject rates are equal — lower is better, and it
is threshold-free in the sense that it summarizes the whole ROC curve at its
crossing point. AUC (area under the ROC curve) summarizes ranking quality
across all thresholds; 1.0 is perfect separation, 0.5 is chance.

**What is not reported.** ASVspoof's official primary metric for the LA track
is **min t-DCF** (minimum tandem detection cost function), which additionally
requires automatic speaker verification (ASV) scores to model the cost of
spoof attacks against a speaker verification system. This project does not
implement an ASV subsystem, so min t-DCF is not computed and is not claimed.
EER and AUC are reported instead, consistent with how they are defined in the
official protocol, but they are not a substitute for min t-DCF as the
challenge's headline metric.

Argmax (threshold-0.5) accuracy is not used as a headline metric: on the eval
set it is uncalibrated (see `docs/decisions/0001-audio-spoof-detection.md`,
D4).

---

## 2. Visual landmark representation (Celeb-DF v2)

The visual branch consumes MediaPipe face/mouth **landmarks**, not raw face
crops or CNN image features — see `docs/decisions/0002-video-deepfake-detection.md`.
It was evaluated standalone on Celeb-DF v2 twice, with two different modeling
approaches, to characterize how much deepfake-discriminative signal the
landmark representation alone carries:

| Approach | AUC | EER |
|---|---|---|
| Linear/simple features (pooled coords, feature set A, logistic regression) | 0.7337 | 34.61% |
| Landmark Transformer (dense 32-frame windows, face_mouth, 3-seed aggregate) | 0.660 ± 0.003 | 37.33% ± 0.06pp |

Sources: `outputs/analysis/visual_feature_ceiling/*.json` (linear result);
`docs/decisions/0002-video-deepfake-detection.md` D9–D13 (Transformer result,
Phase 8 full-data run).

**Role in the architecture.** These numbers are **weak** as a standalone
deepfake classifier and are reported without qualification. The visual branch
is retained in SyncGuard **only as an auxiliary representation** feeding the
audio-visual cross-attention and Sync Head (Section 3) — it is never presented
as, and is not used as, a standalone deepfake detector. A tiny-subset overfit
diagnostic (train AUC ≈ 0.99) shows the training pipeline itself can fit the
data, so the weak generalization result is attributed to limited transferable
landmark signal at this scale, not a broken implementation. Whether this
representation is nonetheless useful for multimodal *synchronization* is a
separate question addressed in Section 3 — that use case has not been proven
superior by a controlled ablation against alternative visual features, so this
explanation is a design rationale, not an experimentally established claim.

---

## 3. Audio-visual synchronization (LAV-DF)

### 3.1 Architecture recap

Video → MediaPipe face/mouth landmarks → visual landmark Transformer → visual
tokens. Audio → mel-spectrogram → CNN+Transformer encoder → audio tokens →
deterministic temporal alignment to video-rate buckets. Aligned audio tokens +
visual tokens → bidirectional cross-attention → Sync Head → per-window
synchronization score → mean-aggregated to a video-level score.

### 3.2 Controlled temporal-shift evaluation

The primary AV-sync evidence comes from an **artificial, controlled-shift**
experiment (`scripts/evaluate_sync.py`, LAV-DF dev split, 1,000 clips, Phase 11
model): the audio track of a clip is either left aligned (shift = 0.0s) or
offset by a fixed amount before being run through the frozen pipeline.

- **Positive class:** shift = 0.0s (native alignment).
- **Negative class:** a non-zero *controlled* temporal shift introduced by
  this codebase — **not** a naturally occurring deepfake artifact, and not
  "real deepfake negatives."

| Audio shift | Mean sync score |
|---|---|
| 0.0s | 0.9909 |
| +0.5s | 0.2545 |
| +1.0s | 0.1552 |
| +2.0s | 0.0000 |

Source: `outputs/runs/sync-phase11-20260912-172237/evaluation_results.json`
(`overall_stats.mean_score` at shift 0.0, `shift_stats["0.5"/"1.0"/"2.0"].mean_score`
otherwise).

**Interpretation.** The Phase 11 model demonstrates strong temporal sensitivity
to controlled audio delays on unseen LAV-DF clips. This is the scientifically
supported statement. It does **not** support "the model detects all
deepfakes," "100% deepfake detection," "the model is universally robust,"
"the model detects every desynchronization," or SOTA-style claims, and none of
those claims are made anywhere in this project.

**Negative shifts.** Shifts of −0.5s, −1.0s, and −2.0s produce **zero valid
overlapping windows** under this alignment scheme for the LAV-DF dev clips
(each ~1.28s at 25 FPS with a 32-frame window) — this is a property of the
short clip length and the alignment bucketing, not a model bug. AUC is
reported as `N/A` (not a misleading single-class number) wherever a shift
condition has only one class of window present, which is every non-zero shift
condition here (0-vs-1 class only) as well as shift 0 itself (all-positive).

### 3.3 Independent evaluation: manipulation category vs. native-timing sync score

Section 3.2 never uses LAV-DF's own real/fake manipulation labels as a
synchronization target — see `docs/decisions/0005-lavdf-dataset.md` and the
regression test `test_no_fake_periods_as_sync_labels`. To investigate the AV
system on an axis *other* than artificial shifts, `scripts/evaluate_av_independent.py`
runs the same frozen pipeline (Phase 12, λ=0.1 checkpoint — the one actually
deployed in the app) at **native, unmodified timing** (no shift at all) over
LAV-DF dev, grouped by manipulation category:

| Category | n | Mean sync score | Std | Min | Max |
|---|---|---|---|---|---|
| REAL | 250 | 0.9915 | 0.0013 | 0.9767 | 0.9923 |
| AUDIO-ONLY fake | 250 | 0.9913 | 0.0026 | 0.9686 | 0.9923 |
| VIDEO-ONLY fake | 250 | 0.9915 | 0.0012 | 0.9809 | 0.9923 |
| AUDIO+VIDEO fake | 250 | 0.9916 | 0.0011 | 0.9787 | 0.9923 |

Source: `outputs/analysis/av_independent_native_dev/independent_av_results.json`.

**What this shows.** At native timing, all four categories — including the
three manipulated ones — score almost identically (~0.991), consistent with
the project's core scientific premise: **manipulation is not equivalent to
desynchronization**. LAV-DF's manipulated clips (face-swap, audio replacement)
are constructed to remain temporally plausible, so the Sync Head correctly
does not flag them as desynchronized on this axis alone; content manipulation
without temporal misalignment is outside what the Sync Head is designed to
detect. This descriptive result is a check that the model does not spuriously
conflate manipulation with desync, not a demonstration that the AV pipeline
detects manipulated content.

**What this does not show.** These are descriptive statistics by manipulation
label, not a synchronization classification result: no AUC/EER is computed
against manipulation category, because manipulation labels are not
synchronization ground truth (see "Independent AV evaluation" below). No
dataset used in this project provides annotated, ground-truth desynchronization
labels independent of controlled shifts.

### 3.4 Independent AV evaluation: dataset investigation

A central open question for this phase was whether a **genuinely independent**
audio-visual desynchronization ground truth exists, separate from (a)
controlled temporal shifts and (b) manipulation-content labels. Investigated
candidates:

| Dataset | Video | Audio | Real/fake AV pairs | Manipulation metadata | Explicit desync labels | Available here |
|---|---|---|---|---|---|---|
| LAV-DF | Yes | Yes | Yes | Yes (`fake_periods`, `modify_audio`, `modify_video`) | No | Yes (local, train/dev extracted; test held out) |
| Celeb-DF v2 | Yes | No (official release has no audio track used by this project) | N/A | No | No | Yes (local, video only) |
| ASVspoof 2019 LA | No | Yes | N/A | N/A | N/A | Yes (local, audio-only dataset) |
| FakeAVCeleb | Yes | Yes | Yes | Yes (real / fake-audio / fake-video / fake-both) | No (documented) | Not downloaded |
| DFDC (Deepfake Detection Challenge) | Yes | Yes (mostly preserved) | Yes | Real/fake video label only | No | Not downloaded (empty `data/dfdc/` placeholder) |
| KoDF | Yes | Yes | Yes | Real/fake label only | No | Not downloaded |

**Conclusion.** No dataset identified — local or publicly available — ships an
explicit, annotated audio-visual synchronization ground truth distinct from
content-manipulation labels. This matches the wider literature: even the
original lip-sync-error detection line of work (e.g. SyncNet-style training)
uses the same paradigm as this project — genuine pairs are the aligned
original, and negatives are constructed by artificially shifting audio — rather
than a dataset with manually annotated "this clip is desynchronized" labels.
Section "Dataset documentation" (`docs/datasets.md`) records source, license,
and access details for the three not-yet-downloaded candidates so they can be
evaluated later with `scripts/evaluate_av_independent.py` without downloading
anything automatically in this pass.

The LAV-DF **test split** (26,100 samples, extracted and preprocessing never
run — see `docs/decisions/0005-lavdf-dataset.md` D2, "test split must remain
completely untouched") is the closest thing to a *held-out* independent
evaluation already available in this project. It was **not** exercised in this
phase, deliberately: spending it changes its status from "never used" to
"used once," which should be a decision made when a final, retrained or
otherwise finalized model is ready to be reported against it — not spent on an
intermediate script test. `scripts/evaluate_av_independent.py` accepts
`--split test` and will run against it once a small test manifest and
extraction subset are prepared (see "Exact commands" in the final report / the
top-level README's reproducibility section for how).

---

## 4. Contrastive ablation (Phase 12)

Phase 12 adds an optional symmetric InfoNCE contrastive objective on top of the
Phase 11 sync-only baseline: `L = L_sync + λ · L_contrastive`, active only when
λ > 0 (see `docs/decisions/0012-phase12-contrastive-learning.md`).

| Configuration | Best val. video-level sync AUC | Best epoch |
|---|---|---|
| λ = 0 (Phase 11 baseline) | 1.0000 | 7 / 120 |
| λ = 0.1 (contrastive) | 1.0000 | 7 / 120 |

Source: `outputs/runs/sync-phase12-lambda0-20260913-014828/summary.json`,
`outputs/runs/sync-phase12-lambda01-20260913-115101/summary.json`.

**Interpretation.** Both configurations saturate at the same ceiling AUC on
the controlled-shift validation split, which was already separable at that
ceiling under the Phase 11 baseline. **This does not demonstrate that
contrastive learning improves performance** — the comparison is inconclusive
at this ceiling, not a measured benefit. The λ = 0.1 checkpoint is used in the
deployed app (see `backend/main.py`) for architectural/representation-learning
reasons (it is the more general Phase 12 configuration with adapters engaged),
not because it was shown to outperform λ = 0 on this metric.

---

## 5. Summary of what is and is not claimed

| Claim | Status |
|---|---|
| Audio spoof detection generalizes to unseen ASVspoof attacks at EER ≤ ~5% | Supported (Section 1) |
| Audio spoof detection achieves official ASVspoof min t-DCF results | Not measured / not claimed |
| Visual landmarks alone are a strong deepfake detector | Not supported — explicitly weak (Section 2) |
| Visual landmarks are useful as an auxiliary AV representation | Design rationale; not proven by a controlled ablation |
| The Sync Head is sensitive to controlled temporal shifts | Supported (Section 3.2) |
| The Sync Head does not spuriously flag native-timing manipulated clips as desynced | Supported, descriptively (Section 3.3) |
| The Sync Head has been validated against real, annotated desynchronization | **Not supported — no such ground truth was available or found** |
| Contrastive learning (λ>0) improves sync performance | Not supported — inconclusive at ceiling (Section 4) |
| SyncGuard is a universal / SOTA / 100%-accurate deepfake detector | Not claimed anywhere in this project |
