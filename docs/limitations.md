# SyncGuard — Limitations

This page collects, in one place, everything SyncGuard's current evaluation
does **not** establish. It is meant to be read alongside `docs/experiments.md`
(what was measured) rather than instead of it.

1. **Audio-only evaluation is based on ASVspoof 2019 LA only.** Generalization
   to other spoofing/TTS/voice-conversion systems, languages, recording
   conditions, or codecs released after or excluded from ASVspoof 2019 LA is
   untested.

2. **AV synchronization validation currently relies primarily on controlled
   temporal shifts**, not on naturally occurring deepfake desynchronization
   (`docs/experiments.md`, Section 3.2).

3. **Controlled shifts are not equivalent to naturally occurring deepfake
   manipulation.** A model that is sensitive to an artificial constant-offset
   shift is not thereby shown to be sensitive to the more subtle, localized,
   or content-driven timing artifacts that a real manipulation pipeline might
   introduce.

4. **Celeb-DF v2, as used in this project, lacks the audio needed for native
   AV-synchronization evaluation.** It was used only for the standalone visual
   landmark representation development (`docs/experiments.md`, Section 2).

5. **Visual landmarks are an auxiliary representation with limited standalone
   classification performance in the current experiments** (AUC ≈0.66–0.73 on
   Celeb-DF v2, well below a strong standalone deepfake detector). They are
   not presented as, and should not be used as, a standalone deepfake
   detector.

6. **The Sync Head detects synchronization consistency, not every form of
   manipulation.** A manipulated clip that remains temporally synchronized
   (e.g. a face-swap that preserves lip timing, or an audio replacement timed
   to match the original) is not guaranteed to be flagged. Section 3.3 of
   `docs/experiments.md` shows exactly this: at native timing, manipulated and
   real LAV-DF clips score almost identically on the Sync Head.

7. **No independent, annotated audio-visual desynchronization ground truth was
   found or used.** Every dataset investigated (LAV-DF, FakeAVCeleb, DFDC,
   KoDF) ships manipulation-content labels, not synchronization labels — see
   `docs/datasets.md` and `docs/experiments.md` Section 3.4. Manipulation
   labels are never converted into synchronization targets anywhere in this
   codebase (enforced by `test_no_fake_periods_as_sync_labels` in
   `tests/test_evaluate_sync.py`).

8. **No claim of universal deepfake robustness.** Results are reported only on
   the specific datasets, splits, and attack sets described in
   `docs/experiments.md`. Performance on unseen manipulation methods, unseen
   speakers/identities, adversarially crafted inputs, or degraded/compressed
   media in the wild is not measured.

9. **No claim of state-of-the-art (SOTA) status.** No head-to-head comparison
   against other published deepfake-detection or AV-sync systems on a common
   benchmark and protocol has been run.

10. **No frame-level manipulation localization is claimed.** LAV-DF's
    `fake_periods` metadata is preserved in manifests for potential future
    temporal-localization work, but no component of SyncGuard currently
    predicts *where* within a clip a manipulation occurs — only a per-window
    (Sync Head) or whole-clip (spoof head) score.

11. **Raw model scores are not calibrated probabilities**, unless calibration
    has actually been performed. `sync_probability`, `desync_probability`,
    `bonafide_probability`, `spoof_probability`, and `aggregate_sync_score`
    are model outputs after a sigmoid/softmax, not the result of a calibration
    procedure (e.g. temperature scaling or isotonic regression) validated
    against held-out reliability diagrams. Treat them as ranking/confidence
    signals, not as literal probabilities of ground truth.

12. **ASVspoof's official primary metric (min t-DCF) is not implemented.**
    This project reports EER and AUC, which are well-defined under the
    official protocol, but does not implement the ASV subsystem min t-DCF
    additionally requires (`docs/experiments.md`, Section 1).

13. **The contrastive learning ablation (λ=0 vs λ=0.1) is inconclusive.** Both
    settings saturate at the same ceiling AUC on the controlled-shift
    validation split; no measured improvement from contrastive learning is
    claimed (`docs/experiments.md`, Section 4).

14. **The LAV-DF test split (26,100 samples) has not been evaluated.** It
    remains held out per `docs/decisions/0005-lavdf-dataset.md`. Any result
    reported in this project comes from the train/dev splits only, or from
    controlled experiments built on top of dev.

15. **Latency/throughput has not been benchmarked.** No "real-time" claim is
    made anywhere in this project; end-to-end inference latency on
    representative hardware has not been measured and published.
