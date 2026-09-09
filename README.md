# SyncGuard

Audio-visual deepfake / spoof detection with a shared encoder feeding a later
audio-visual synchronization model.

## Project status

| Track | State | Decision record |
|---|---|---|
| Audio spoof detection (Phases 4–6) | Complete — CNN + Transformer ensemble, 1.93 % EER on ASVspoof 2019 LA | `docs/decisions/0001-audio-spoof-detection.md` |
| Visual landmark branch (Phases 7–8) | **Complete.** Retained as an **auxiliary** visual stream for multimodal sync, **not** a standalone deepfake detector | `docs/decisions/0002-video-deepfake-detection.md` |
| Audio-visual sync / fusion (Phase 9) | Not started | — |

### Phase 8 status (2026-09-08)

Controlled full-data training of the landmark visual Transformer (dense 32-frame
windows, `face_mouth`, plain CE, LR 3e-4 + cosine + warmup, dev-AUC selection,
3 seeds) reaches **eval AUC 0.660 ± 0.003 / EER 37.3 %** on Celeb-DF-v2 — below
the ~0.734 best linear-feature result. A tiny-subset overfit diagnostic confirms
the training pipeline can fit the data (train AUC ≈ 0.99), so the weak result is
attributed to limited transferable landmark signal and generalization, not a
broken implementation or optimization failure. **Decision:** keep the landmark
Transformer as an auxiliary input to Phase 9 fusion; further standalone visual
tuning is not justified at this stage. Details and the full A/B/C interpretation
are in `docs/decisions/0002-video-deepfake-detection.md` (D9–D13).
