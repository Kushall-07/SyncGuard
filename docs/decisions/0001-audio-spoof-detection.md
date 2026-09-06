# 1. Audio spoof detection — track decisions (Phases 4–6)

- Status: Accepted
- Date: 2026-09-06
- Scope: audio-only spoof detection on ASVspoof 2019 LA
- Supersedes: none

This note records every decision made while building the audio spoof detector,
from the Phase 4 CNN baseline through the Phase 5 model comparison and the
selected ensemble. It is append-only; a later reversal gets a new note.

---

## Context

SyncGuard needs an audio-only spoof/deepfake detector, and a shared audio
*encoder* that a later audio-visual sync model will reuse. The work ran in three
phases: Phase 4 a CNN baseline, Phase 5 a Transformer variant plus a head-to-head
comparison, Phase 6 the evaluation pipeline. A full leakage / protocol /
implementation audit was run on the Phase 4 baseline before any result was
trusted.

Dataset: ASVspoof 2019 LA, official countermeasure (CM) protocol. Splits are
speaker-disjoint and the evaluation split uses spoofing systems (A07–A19) that
never appear in train/dev (A01–A06) — i.e. generalization is measured against
*unseen* attacks by design.

---

## Decisions

### D1 — Dataset and protocol
Use ASVspoof 2019 LA with the official CM protocol files, unmodified. Splits:
train 25,380 / dev 24,844 / eval 71,237 utterances. No custom re-splitting, no
subsetting for the headline runs.
**Status: Accepted.** Manifest verified row-for-row against the protocol files
(0 mismatches) and against published dataset statistics (exact match).

### D2 — Label convention and headline metric
`1 = bonafide` (positive class), `0 = spoof`. Model emits logits ordered
`[spoof, bonafide]`; score = `softmax(logits)[:, 1]` = `P(bonafide)`.
The headline metrics are **EER, ROC-AUC, and the DET curve** (threshold-free),
plus per-attack EER against the pooled bonafide set (ASVspoof convention).
**Status: Accepted.** Implementation of every metric was traced and independently
reproduced.

### D3 — CNN baseline accepted as the Phase 4 reference
The log-mel CNN (`spoof-cnn-baseline-20260906-104508`, 353,315 params, best
epoch 25) is a valid baseline: **eval EER 4.98%, AUC 0.988**. It is competitive
with the official ASVspoof baselines (LFCC-GMM 8.09%, CQCC-GMM 9.57%).
**Status: Accepted**, conditional on the audit below, which passed.

### D4 — How results are reported
- dev EER (0.41%) is a **seen-attack** number and is never presented as the
  project's performance.
- Argmax (threshold-0.5) accuracy and F1 are **not** headline metrics. On the
  eval set they are uncalibrated (the class weight and the unseen-attack score
  drift push the 0.5 boundary off the optimal operating point). If reported at
  all, they are computed at the EER threshold and labelled "uncalibrated".
- No SOTA claims, no in-the-wild deepfake claims, no general-robustness claims.
**Status: Accepted.**

### D5 — Selected audio spoof detector: CNN + Transformer score ensemble
The audio spoof detector is the **mean of the CNN and Transformer `P(bonafide)`
scores (0.5 / 0.5)** over identical inputs. No retraining, no weight changes, no
threshold fitting on eval. Reproduced by `scripts/ensemble_experiment.py`.

| Model | Eval EER | Eval AUC | EER threshold |
|---|---|---|---|
| CNN baseline | 4.98% | 0.9881 | 0.906 |
| CNN + Transformer | 3.74% | 0.9928 | 0.999 |
| **Ensemble (mean)** | **1.93%** | **0.9978** | 0.668 |

**Status: Accepted.** Selection rule: adopt the ensemble only if pooled EER
falls *and* AUC does not drop versus the Transformer — both held.

### D6 — Encoder handed to the AV-sync branch
The audio-visual sync model consumes a shared audio *encoder*, not the spoof
classifier. Use the **Transformer's** `audio_encoder.pt`
(`spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt`) — one encoder,
and the stronger single model. The ensemble stays a classifier-level construct
and does not propagate into the encoder hand-off.
**Status: Accepted.**

### D7 — Deferred
Not done now, revisit if a future result or requirement calls for it:
- a jointly trained / retrained fusion model instead of a score average;
- softening the bonafide class weight (currently 8.83×) to improve argmax
  calibration on eval;
- a second-seed Transformer run to confirm the A17 regression (below).
**Status: Deferred.**

---

## Audit findings (Phase 4 baseline)

**Leakage — PASS.** Over the full manifest and all 121,461 `.flac` files:
speaker overlap 0/0/0, path overlap 0/0/0, sample-id overlap 0/0/0, and
SHA-1-of-bytes finds **0** identical-content groups across splits and **0**
within any split. Eval is never touched during training, model selection, or
preprocessing (peak-norm is per-file; `AmplitudeToDB` is per-sample — no fitted
statistics).

**Protocol — PASS.** Manifest counts match official ASVspoof 2019 LA exactly for
all three splits; attacks are A01–A06 (train/dev) and A07–A19 (eval), disjoint by
design.

**Label & metric implementation — PASS.** Label mapping, weighted cross-entropy,
confusion matrix (`[[TN, FP], [FN, TP]]` with bonafide positive),
precision/recall/F1, ROC-AUC and EER all traced through code and reproduced.

**dev → eval gap — explained, not a bug.** The bonafide score distribution is
essentially unchanged dev→eval (mean 0.976 → 0.974); the spoof distribution
grows an upper tail on 4 unseen attacks (A10, A12, A13, A15), which produced
86% of the CNN's false accepts. The argmax accuracy/F1 collapse
(0.997 → 0.856 / 0.987 → 0.585) is an operating-point artifact — at the EER
threshold, eval accuracy recovers to 0.950 and F1 to 0.798. The residual EER
gap (0.41% → 4.98%) and AUC drop (0.9998 → 0.988) are genuine ranking loss on
unseen attacks.

---

## Phase 5 model comparison

The Transformer (`cnn_transformer` encoder, 3 layers) improved pooled eval EER
4.98% → 3.74% and AUC 0.988 → 0.993, but **not uniformly**: it eliminated the
CNN's four worst attacks and regressed hard on A17.

| Attack | CNN EER | Transformer EER | Ensemble EER |
|---|---|---|---|
| A10 | 7.83% | 0.84% | 1.52% |
| A12 | 11.97% | 0.93% | 1.85% |
| A13 | 5.13% | 0.63% | 0.73% |
| A15 | 7.22% | 1.35% | 2.59% |
| A17 | 4.56% | **12.54%** | 4.50% |
| A18 | 3.17% | 5.40% | 2.92% |

The ensemble keeps most of the Transformer's gains and repairs A17; **every
eval attack is below 5% EER** for the first time. Weight sweep is flat across
w_transformer 0.4–0.7 (EER 1.88–2.09%), so the plain mean is used.

---

## Consequences

- Selected model runs **two encoders at inference** (~2× audio compute). The CNN
  is the cheap half; drop it only if a later single model matches 1.93% EER.
- The ensemble also lands a usable operating point as a side effect: EER
  threshold 0.999 → 0.67.
- All results are ASVspoof-2019-LA-only. "Generalization" here means unseen
  synthesis systems A07–A19; it is not a claim about other corpora or
  real-world deepfakes.
- The AV-sync branch is unblocked and takes the Transformer encoder (D6).

---

## Open threads (non-blocking)

1. **A17 regression** — confirm the Transformer's 4.6% → 12.5% jump is real and
   not seed variance (one second-seed run). The ensemble neutralizes it (4.5%),
   so this does not block anything.
2. **fp32 consistency** — the training script's end-of-run eval runs under AMP;
   Phase 6 runs fp32. This flips ~10 borderline samples in the confusion matrix
   (EER/AUC identical to 4 digits). Quote the Phase 6 numbers; optionally make
   the training-end eval fp32.
3. **Dead config** — `data.subset_size` exists in `configs/asvspoof.yaml` but is
   not wired into any training script. Wire it in or remove it.
4. **Class weight** — the 8.83× bonafide weight is what pushes the argmax
   boundary off; a softer weight (√-inverse-freq or ~3×) may improve eval
   calibration without hurting EER. Experiment, not a fix.

---

## Evidence

- Runs: `outputs/runs/spoof-cnn-baseline-20260906-104508`,
  `outputs/runs/spoof-transformer-20260906-123646`
- Phase 6 reports: `<run>/eval/report.{json,txt}` + DET / ROC / confusion /
  score-distribution / training-curve PNGs
- Baseline audit: `outputs/analysis/spoof-cnn-baseline/{leakage_protocol.json,
  analysis.json, scores.npz}` — scripts `scripts/investigate_baseline.py`,
  `scripts/investigate_scores.py`
- Ensemble: `outputs/analysis/ensemble/{ensemble.json,ensemble.txt,
  ensemble_scores.npz}` — script `scripts/ensemble_experiment.py`
- Comparison page (Artifact):
  https://claude.ai/code/artifact/7d8e5232-dd71-4e07-974f-88a3fc3b895c
