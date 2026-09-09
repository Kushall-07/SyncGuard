# 2. Video deepfake detection — track decisions (Phases 7–8)

- Status: **Accepted.** Phase 7 methodology + Phase 8 experimental results recorded.
- Date: 2026-09-06 (Phase 7), updated 2026-09-08 (Phase 8 results + final decision)
- Scope: video-only deepfake detection on Celeb-DF-v2, landmark-based
- Supersedes: none
- Related: `0001-audio-spoof-detection.md` (the audio branch this mirrors)

This note records the decisions made building the visual branch: the landmark
preprocessing pipeline, the Celeb-DF split protocol, the two models (non-temporal
baseline and temporal Transformer), the shared visual-encoder hand-off to the
audio-visual sync phase (D1–D8, Phase 7), and the Phase 8 controlled experiments
that measured what the landmark representation actually achieves (D9–D13). It is
append-only for *decisions*; stale placeholders (TBD tables, "pending" notes) are
replaced in place rather than left to contradict later sections.

**Bottom line (D13):** the landmark visual Transformer is retained as an
*auxiliary* visual representation for multimodal audio-visual synchronization, not
as a standalone deepfake detector.

---

## Context

SyncGuard needs a video-only deepfake detector and a shared visual *encoder* that
the Phase 9 audio-visual sync model will reuse, exactly as the audio branch
produced a shared audio encoder. Phase 7 is standalone — no audio, no fusion.

The visual branch follows the frozen SyncGuard architecture: the classifier
consumes **normalised MediaPipe face/mouth landmark sequences**, not pixels:

    video → face + mouth MediaPipe landmarks → normalised temporal landmark
    sequence → landmark embedding → positional encoding → temporal Transformer
    → visual tokens → deepfake head

Dataset: Celeb-DF-v2 (590 Celeb-real, 5,639 Celeb-synthesis face swaps, 300
YouTube-real; `List_of_testing_videos.txt` = 518-video official test list).

---

## Decisions

### D1 — Dataset and split protocol
Use Celeb-DF-v2 with the official test list as the `eval` split **verbatim** — the
file `List_of_testing_videos.txt` is only ever *read*, never rewritten, and the
eval rows keep its exact order (518 videos).

The remaining videos are split `train` / `dev` **strictly identity-disjoint** by
*identity co-occurrence component* (`src/data/celebdf_dataset.py::_identity_components`):

1. Every video contributes its full identity set — for a Celeb-synthesis clip
   `id{src}_id{tgt}_*` that is **both** `id{src}` (the face rendered in) and
   `id{tgt}` (the head/pose shown); for a real clip its one identity; for a
   YouTube-real clip a synthetic per-clip identity.
2. Identities that ever co-occur in a pool video are unioned into a connected
   component. A component is **atomic**: all its videos go to one split.
3. Components are assigned deterministically (seeded RNG): the **smallest
   fake-bearing component seeds `dev`**, then real-only (singleton) components are
   added until `dev` reaches `dev_identity_frac` of the pool identities. Further
   fake-bearing components are never added to `dev`. `train` takes the rest.
4. The build **asserts** `train_identities ∩ dev_identities = ∅` over the full
   (source + target) identity sets and that both splits carry real *and* fake
   videos, raising `LeakageError` / `ValueError` otherwise.

Because whole components move together, **no identity — source identities of
synthesised clips included — can appear in both `train` and `dev`.** The choice
is fully deterministic in `seed` (default 1337).

Manifest as built (`seed=1337`, `dev_identity_frac=0.15`):

| split | videos (no exclusion) | videos @ `min_valid_frames=4` | real | fake | identities |
|---|---|---|---|---|---|
| train | 5,196 | 5,145 | 584 | 4,561 | 246 |
| dev   |   815 |   804 | 108 |   696 |  43 |
| eval  |   518 |   511 | 173 |   338 | 126 |

The `min_valid_frames=4` column drops the 69 landmark-quality exclusions
(51 / 11 / 7 by split); real/fake counts shown are post-exclusion.

Pool identity co-occurrence components: sizes `[38, 10, 10, 1×231]`. `dev` is one
10-identity fake-bearing component plus 33 YouTube-real singletons.
**`train ∩ dev` identity intersection = 0** (verified for source and target ids).
Full report is written to `data/celebdf/manifest.csv.audit.json`.

### D2 — Known limitations of the protocol
- **Train/eval share identities.** Celeb-DF reuses the same ~59 celebrities in
  its test list, so `train` and `eval` are *not* identity-disjoint (only `train`
  vs `dev` is). This is inherent to the dataset. "Generalization" in this branch
  means unseen *clips* of seen identities and the seen synthesis method — it is
  **not** a claim about unseen identities, unseen manipulation methods, or other
  corpora.
- **`train`/`dev` source-identity leakage is eliminated** by the component
  partition above (this replaces the earlier target-only split, under which ~88 %
  of `dev` fakes had a source identity present in `train`).
- **Class imbalance.** Fakes outnumber reals ~7.6× in `train` (Celeb-DF's
  natural distribution). Phase 7 build runs used inverse-frequency class weights;
  the Phase 8 audit found the heavy (~7×) real-class weight destabilised the
  argmax decision boundary epoch-to-epoch, so the **Phase 8 controlled run uses
  plain cross-entropy** (`training.class_weight: none`; see D11). EER/AUC are
  threshold-free and the reported operating point is the EER threshold, never
  argmax — so the imbalance affects F1/accuracy, not the headline AUC/EER.

### D3 — Why landmarks, not pixels
The classifier consumes landmark geometry, not face crops, because (a) it aligns
with the frozen AV-sync architecture — the Phase 9 fusion reasons over lip/face
*motion* against audio, and a landmark encoder is directly reusable there, where
a texture CNN is not; (b) face/mouth *motion* is the sync-relevant signal;
(c) landmark sequences are tiny, so training is CPU-cheap and fits the 6 GB
laptop constraint with head-room. Face-crop JPEGs are cached only when
`--save-crops` is passed, for preprocessing QA, and are never read by the model.

### D4 — Preprocessing (`scripts/extract_celebdf_landmarks.py`)
- Decode with PyAV. Two frame-selection modes:
  - **sparse** (Phase 7 build, `--num-frames`, default 16): indices spread across
    the whole clip. Output → `data/celebdf/landmarks/`.
  - **dense** (Phase 8, `--window 32 --stride 1 --window-start center`): one
    *contiguous* 32-frame window (deterministic, centred). Output →
    `data/celebdf/landmarks_dense/`; manifest → `data/celebdf/manifest_dense.csv`
    (`configs/celebdf_dense.yaml`). The sparse ~1 Hz sampling aliases mouth
    articulation (2–7 Hz); the dense window is what the Phase 8 experiments use,
    with `video.num_frames: 32` so frame selection is the identity and the window
    reaches the model contiguously.
- Dense manifest as built (`manifest_dense.csv`, `seed=1337`,
  `min_valid_frames=4`): **6,429 usable clips — train 5,117 / dev 804 / eval
  508** (fewer than the sparse manifest's 6,460: the contiguous-window face
  detector rejects ~31 additional clips). `train ∩ dev` identity intersection = 0.
- MediaPipe Tasks `FaceLandmarker` (`face_landmarker.task`, float16, 478-point
  mesh), one face per frame, downloaded once to `checkpoints/mediapipe/`.
- Cache `<landmarks_dir>/<sample_id>.npz` — `points[T,478,3]` (raw, un-normalised),
  `valid[T]`, `fps`, `frame_idx` — plus a `.meta.json` carrying the detection
  rate, `n_detected`, `min_valid_frames`, and an `excluded` flag.
- Normalisation is deferred to the dataset (`src/preprocessing/landmarks.py`,
  pure NumPy) so it stays configurable: per frame, translate the nose tip to the
  origin, rotate the eye line horizontal (this removes head-roll *dynamics*, not
  just static roll), scale so the inter-ocular distance is 1.
- **Occasional** missing frames in an otherwise-valid video are linearly
  interpolated from their neighbours at load time (leading/trailing gaps take the
  nearest valid frame).
- **`min_valid_frames` (default 4) is enforced, not advisory.** A video with
  fewer detected-face frames than the threshold — or with no landmark cache at
  all — is **excluded from every split** (`train`, `dev` *and* `eval`) rather
  than interpolated in. Excluding eval videos changes the eval denominator
  (`518 − n_excluded`); the *list file itself is never modified*.
- **`extraction_audit.json` is a pure function of the current cache + current
  `--min-valid-frames`.** `scripts/extract_celebdf_landmarks.py` rebuilds it from
  scratch on every run (no merge with a prior audit, so a stale threshold cannot
  contaminate it); `--rescan-only` regenerates it — and re-stamps each
  `meta.json` exclusion flag — without touching any `.npz`. Each excluded record
  carries `n_detected`, `n_frames_decoded`, and a reason:
  `corrupt_or_truncated_source` (the source decoded to fewer frames than the
  threshold — it can never pass) vs `face_not_detected` (enough frames decoded,
  MediaPipe found faces in too few) vs `missing_landmarks` (no cache).
  `scripts/build_celebdf_manifest.py --min-valid-frames N` independently
  re-derives the same exclusions from the `.npz` `valid` arrays and writes the
  per-split / per-reason counts into `manifest.csv.audit.json`. On the current
  full cache both agree: **69 excluded** (68 `face_not_detected`, 1
  `corrupt_or_truncated_source` — `id27_0005`, a truncated 17 KB / 1-frame
  source), split 51 train / 11 dev / 7 eval.
- Region sets: `face` (oval + brows + eyes + nose, **102 pts**), `mouth` (outer +
  inner lip contours, **40 pts**), `face_mouth` (both, **142 pts** = exact
  concatenation, no overlap). Default `face_mouth`.
**Status: Accepted.** A 5-video probe gave a mean per-frame detection rate of
0.89 (min 0.44); real exclusion counts follow the full extraction.

### D5 — Two models
`src/models/video/DeepfakeClassifier` selects a variant via
`model.visual_encoder`:
- **`mlp_baseline`** — landmark embedding → order-free temporal pooling
  (mean ⊕ std over frames) → MLP. Discards frame order, so it is the
  no-temporal-modelling reference.
- **`transformer`** — landmark embedding → sinusoidal positional encoding →
  temporal Transformer (shared `src/models/common/temporal_transformer.py`, the
  same stack the audio branch uses) → reused
  `src/models/heads/spoof_head.py::SpoofHead` (attentive temporal pooling → 2
  logits `[fake, real]`).

`scripts/compare_visual_models.py` trains both on identical data/seed/augmentation;
the EER/AUC gap is the measured temporal contribution — the visual analogue of
the Phase 5 CNN-vs-Transformer experiment.
**Status: Accepted.** Phase 8 measured results are in D9–D11; the temporal-gap
comparison is **not** re-run as a headline (D13 — standalone visual tuning is not
pursued further at this stage).

### D6 — Label convention and headline metrics
`1 = real` (positive class), `0 = fake`. Score = `softmax(logits)[:,1]` =
`P(real)`. One **video** is one sample; the head pools over its sampled frames,
so EER / ROC-AUC / DET are video-level. Per-synthesis-method EER (Celeb-DF has
one method, `celebdf-fs`, vs the pooled real set) is reported for parity with the
audio branch's per-attack breakdown and to leave room for future datasets.
**Status: Accepted**, mirrors audio D2. Reuses `src/evaluation/metrics.py`
unchanged.

### D7 — Encoder handed to the AV-sync branch
Phase 9 consumes a shared visual *encoder*, not the classifier. A `transformer`
training run exports `<run>/checkpoints/visual_encoder.pt` via
`src/models/video/visual_encoder.py::export_visual_encoder` (weights + the
`ModelConfig` / `VideoDataConfig` needed to rebuild it), mirroring the audio
branch's D6. The `mlp_baseline` is a comparison artefact and is never exported.
**Status: Accepted.**

### D8 — Deferred (superseded by the D13 decision)
- **Ablation grid** (`--regions {face,mouth,face_mouth}` × `--num-frames`) — still
  wired, still not run. Not pursued: D13 concludes further standalone visual
  tuning is not justified at this stage.
- Fake-class-weight rebalancing — **resolved**: the Phase 8 run uses plain CE
  (D11), which removed the argmax instability the audit flagged.
- Multi-seed run — **done**: the Phase 8 controlled experiment reports 3 seeds
  (D11).
**Status: Closed by D13.**

---

## Phase 8 — experimental evidence

Phase 8 ran three experiments to answer one question: *can the landmark
representation reach the ~0.734 eval-AUC linear-feature result under a sensible
training setup?* All use the dense 32-frame cache, `face_mouth`, 3-D landmarks,
`manifest_dense.csv`, and the D6 metric convention (`1 = real`, video-level
AUC / EER).

### D9 — Tiny-subset overfit diagnostic (`scripts/visual_overfit_probe.py`)

**Setup:** 200 balanced training clips (100 real + 100 fake, deterministic,
seed 1337), dense 32-frame input, augmentation OFF, class weights OFF,
weight_decay 0, no early stopping, 250 epochs; 6 canonical runs
(`{MLP, Transformer} × LR {1e-4, 3e-4, 5e-4}`) + 3 Transformer warmup runs.
"Eval-mode loss" = the same 200 clips scored with dropout off.

| run | max train AUC | min eval-mode loss |
|---|---|---|
| MLP @ 5e-4 | **0.9985** | **0.074** |
| Transformer @ 3e-4 | **0.995** | **0.097** |
| MLP @ 3e-4 | 0.986 | 0.166 |
| Transformer @ 1e-4 | 0.993 | 0.115 |
| Transformer @ 1e-4 + warmup | 0.986 | 0.149 |
| Transformer @ 3e-4 + warmup | 0.992 | 0.125 |
| Transformer @ 5e-4 (± warmup) | 0.97–0.97 | 0.22–0.23 |
| MLP @ 1e-4 | 0.850 | 0.510 (LR too low — did not converge in budget) |

**What this establishes:** the implementation and training pipeline **can fit**
the subset — both architectures drive train AUC to ≈ 0.99–1.0 and loss well below
chance, with no NaNs and monotone descent. The single "failure" (MLP @ 1e-4) is an
LR/epoch-budget artefact; the same architecture at 3e-4 / 5e-4 fits fully. Warmup
was neither needed nor helpful; the order-free MLP memorised as well as the
Transformer.

**What this does NOT establish:** it does **not** validate the visual
representation. Memorising 200 specific clips only shows learning/memorisation
capability; it says nothing about whether the features transfer. Full details:
`outputs/analysis/visual_overfit/summary.{md,json}` + `history/*.csv`.

### D10 — Feature-ceiling probe (`scripts/visual_feature_ceiling.py`)

**Setup:** no neural training. Logistic Regression and Gradient-Boosted Trees on
hand-pooled features of the dense normalised landmark sequences; three feature
sets — (A) pooled coordinates (mean & std over time), (B) A + velocity /
acceleration / global jitter statistics, (C) B + articulation geometry
time-series (mouth-aspect-ratio, lip thickness, eye-aspect-ratio L/R, jaw
opening, brow raise L/R). Train on 4,000 clips, evaluate on the full eval split.

**Best observed linear-feature result:** Logistic Regression on set A —
**eval AUC ≈ 0.734, eval EER ≈ 0.346**.

- Adding velocity / acceleration / jitter (B) and handcrafted articulation
  geometry (C) **did not improve** the result (logreg: A 0.734 → B 0.669 →
  C 0.662; GBT stayed ≤ 0.65). The exploitable signal is in the static pooled
  pose statistics, not in these motion / articulation features **as constructed**.
- **~0.734 is the best *observed* linear-feature result on these features — it is
  NOT an absolute theoretical ceiling of the landmark representation.** A
  different featurisation, detector, or dataset could differ.

Full details: `outputs/analysis/visual_feature_ceiling/ceiling.json`.

### D11 — Controlled full-data Transformer experiment

**Config (`configs/deepfake_transformer_final.yaml`):** existing `transformer`
architecture unchanged; dense 32-frame windows; `face_mouth`; 3-D coordinates;
**plain cross-entropy** (`class_weight: none`); LR **3e-4**; **cosine** scheduler
with **8-epoch linear warmup**; **coordinate jitter OFF**, **time masking OFF**
(hflip / scale / rotation kept — temporally consistent); **no early stopping**;
**checkpoint selection by dev AUC** (`monitor: val_roc_auc`, mode `max`);
120 epochs; `weight_decay 1e-4`, `batch_size 32`, AMP on.

Three seeds:

| seed | best dev AUC (@ epoch) | eval AUC | eval EER | eval acc | eval F1 | confusion `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|
| 1337 | 0.5965 (@96) | 0.664 | 0.374 | 0.679 | 0.138 | `[[332, 5], [158, 13]]` |
| 2024 | 0.5894 (@62) | 0.658 | 0.373 | 0.689 | 0.141 | `[[337, 0], [158, 13]]` |
| 42   | 0.5894 (@62) | 0.658 | 0.373 | 0.689 | 0.141 | `[[337, 0], [158, 13]]` |

**Aggregate: eval AUC = 0.660 ± 0.003, eval EER = 37.33 % ± 0.06 pp.**

Observations:
- The controlled Transformer lands at **~0.66 eval AUC** — *below* the ~0.734
  linear-feature result (D10) and far from a usable standalone detector. Argmax
  F1 ≈ 0.14 (near-constant "fake" prediction; two seeds predict zero false
  positives on eval) — expected under plain CE on a 7.6:1 imbalance, and why the
  headline metric is threshold-free AUC/EER, not F1/accuracy.
- Seeds 2024 and 42 produced **identical** eval metrics and confusion matrices.
  With `deterministic: false` this indicates the pipeline converged to
  effectively the same solution and selected the same dev-AUC epoch — i.e. low
  seed sensitivity here, not a seeding bug.
- Best dev AUC across all seeds is ≈ 0.59 — barely above chance — which both
  confirms the weak signal and flags the dev split as a poor selector (D12).

Full details: `outputs/analysis/` run reports + per-run `report.json`
(`selected_by: val_roc_auc`).

### D12 — Interpretation

**Ruled out / substantially reduced by D9 + D11:**
- **Broken implementation / broken training loop** — the pipeline fits 200 clips
  to AUC ≈ 0.99 (D9) and trains 120 epochs on full data with monotone loss
  descent and a working cosine schedule (D11).
- **Inability to optimise the model / insufficient model capacity** — both
  architectures reach train AUC ≈ 0.99–1.0 on the subset.
- **Simple LR / configuration failure** — a deliberate, sensible setup (plain CE,
  LR 3e-4, cosine + warmup, jitter/time-mask off, dev-AUC selection, 3 seeds)
  moved the full-data result only marginally (0.66 vs the earlier ~0.67), i.e.
  configuration was not the dominant cause of the weak result.

**Still plausible / supported:**
- **Weak transferable landmark signal** — a linear model on the same dense
  features tops out at ~0.734 eval AUC (D10); explicit motion / articulation
  features do not help; Celeb-DF face swaps largely preserve landmark geometry.
- **Poor generalization from Celeb-DF landmark geometry** — capacity that
  memorises 200 clips (D9) will memorise identity- and clip-specific quirks of
  5,117 training clips without learning a transferable real/fake boundary.
- **Dataset / identity distribution effects** — `train` and `eval` share
  identities by Celeb-DF design (D2); "generalization" here is only to unseen
  *clips* of seen identities and the one seen synthesis method.
- **Train / dev / eval distribution differences** — see the dev-set limitation
  below; dev AUC (~0.59 best) tracks eval AUC (~0.66) poorly.

**Dev-set limitation (affects model selection, not the code):**
- `dev` contains only **108 real clips**.
- Its fake clips are concentrated in the **id39–id48 synthesis clique** (one
  10-identity co-occurrence component; ~696 fakes).
- Therefore **dev AUC is a weak, noisy model-selection signal** — the max-dev-AUC
  checkpoint need not be the max-eval-AUC checkpoint, and `report.json`'s
  `selected_metric_value` is a dev number, not an eval estimate. `metrics.csv`
  logs `val_roc_auc` every epoch so the whole trajectory can be inspected.

### D13 — Final architectural decision

> **The visual landmark Transformer is retained as an auxiliary visual
> representation for multimodal audio-visual synchronization, not as a standalone
> deepfake detector.**

Rationale, stated precisely:

> The experiments show that the landmark stream contains limited standalone
> discriminative signal and does not generalize strongly enough for standalone
> detection. Its remaining value will be evaluated in the multimodal
> synchronization setting, where visual information is combined with audio rather
> than required to independently classify every clip.

This decision does **not** claim: that the representation is useless or has zero
signal; that ~0.734 is a hard upper bound; that the visual branch is
state-of-the-art; that temporal modelling has been proven useless; or that the
architecture is fundamentally bad.

**Further standalone visual tuning is not justified at this stage.** The
deferred D8 ablations are not pursued. The `transformer` encoder still exports
via D7 for Phase 9 to consume as one input to the fusion model; Phase 9 evaluates
its contribution *in combination with audio*, not as an independent classifier.

**Status: Accepted.**

---

## Limitations (summary)

- Celeb-DF-v2 only; `train`/`eval` share identities (D2). Not a claim about unseen
  identities, unseen manipulation methods, or other corpora.
- `dev` is small (108 real) and distribution-shifted (id39–id48 fake clique), so
  it is a poor model-selection signal (D12).
- The ~0.734 linear-feature number (D10) is the *best observed* result with the
  featurisations tried, not a proven ceiling.
- The tiny-subset diagnostic (D9) establishes learning capability only; it does
  not validate the representation.
- Phase 8 evaluated the `transformer` variant; the `mlp_baseline` vs
  `transformer` temporal-gap comparison was not run as a headline (D13).

---

## Consequences

- The visual branch is a **weak/auxiliary stream**. Phase 9 fusion must not
  require it to independently classify each clip; audio carries the decision.
- The Phase 9 AV-sync branch is unblocked and takes the `transformer` visual
  encoder (D7), now understood as an auxiliary input.
- All results are Celeb-DF-v2-only. See D2 / D12 for what "generalization" means.
- `src/models/audio/transformer.py` delegates to the shared
  `TemporalTransformerEncoder`; it is a thin config adapter and keeps the same
  submodule names, so existing Phase 5 checkpoints still load.

---

## Evidence

- Code: `src/preprocessing/landmarks.py`, `src/preprocessing/video_augment.py`,
  `src/data/celebdf_dataset.py`, `src/models/video/`,
  `src/models/common/temporal_transformer.py`,
  `src/training/deepfake_trainer.py`, `src/evaluation/deepfake_eval.py`,
  `src/training/utils.py::build_scheduler`
- Configs: `configs/celebdf.yaml`, `configs/celebdf_dense.yaml`,
  `configs/deepfake_transformer.yaml`, `configs/deepfake_landmark_baseline.yaml`,
  `configs/deepfake_transformer_final.yaml` (the D11 controlled run)
- Scripts: `scripts/extract_celebdf_landmarks.py`,
  `scripts/build_celebdf_manifest.py`, `scripts/train_deepfake_landmark.py`,
  `scripts/evaluate_deepfake.py`, `scripts/compare_visual_models.py`,
  `scripts/visual_feature_ceiling.py` (D10), `scripts/visual_overfit_probe.py` (D9)
- Analysis outputs: `outputs/analysis/visual_overfit/` (D9),
  `outputs/analysis/visual_feature_ceiling/` (D10), Phase 8 training runs under
  `outputs/runs/` (D11)
- Tests: `tests/test_preprocessing_landmarks.py`, `tests/test_data_celebdf.py`,
  `tests/test_models_common.py`, `tests/test_models_video.py`,
  `tests/test_deepfake_eval.py`, `tests/test_extract_celebdf_audit.py`,
  `tests/test_extract_celebdf_dense.py`, `tests/test_visual_overfit_probe.py`,
  `tests/test_scheduler_and_final_config.py`
- Manifests: `data/celebdf/manifest.csv` (sparse) and
  `data/celebdf/manifest_dense.csv` (dense, used by Phase 8) — neither committed;
  rebuild with `scripts/build_celebdf_manifest.py`
