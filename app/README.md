# SyncGuard App Launcher

`app/app.py` is the one-command launcher for the SyncGuard demo: it starts the
existing FastAPI backend (`backend/main.py`) and the existing React/Vite
frontend (`frontend/`) as child processes, waits for both to become reachable,
prints their URLs, opens a browser, and shuts both down cleanly on Ctrl+C. It
contains no ML/inference logic and no API routes of its own.

> An earlier prototype of this demo used a single-process Gradio UI on port
> 7860. That has been replaced by the FastAPI + React/Vite stack described
> here; `tests/test_app.py` still references the old Gradio module and is
> skipped (not failing) in the current test suite for that reason — see
> `docs/limitations.md`.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

cd frontend
npm install
cd ..
```

## Launching

```powershell
python .\app\app.py
```

- Frontend: http://127.0.0.1:5173
- Backend: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/api/health
- Stop: press Ctrl+C

The launcher prefers `.venv\Scripts\python.exe` for the backend and reuses an
already-running backend/frontend instead of starting a duplicate. If port 8000
or 5173 is occupied by something other than SyncGuard, it prints an error
instead of touching that process.

To run backend and frontend manually instead, see the root
[README.md](../README.md#advanced-run-backend-and-frontend-manually).

## Usage

### Audio-Only mode (Analyze page)

Upload an audio file (WAV, FLAC, etc.) and get a bonafide/spoof classification
with per-class scores. Detects whether the speech signal is likely bonafide or
synthetic/spoofed, based on acoustic features learned from ASVspoof 2019 LA.

### Audio-Visual mode (Analyze page)

Upload a video file (MP4, etc.) and get a synchronization report: aggregate
sync score, per-window timeline, and sync/desync classification. Detects
temporal synchronization inconsistencies between the audio and visual
streams — it does not directly detect content manipulation (see Limitations).

### Synchronization Lab

A research/educational page that lets you pick a precomputed LAV-DF sample and
an explicit audio offset (0.0s / 0.5s / 1.0s / 2.0s) and see how the trained
Sync Head responds. This reuses the same frozen weights as normal inference;
only the audio-to-video timing is changed. Negative shifts are not exposed
because they produce zero valid overlapping windows under the current
alignment scheme (verified in the Phase 11 controlled shift-sensitivity
evaluation — see `docs/experiments.md`).

## Important limitations

### Audio-only mode
- Trained and evaluated on ASVspoof 2019 LA only; generalization to other
  spoofing/TTS systems is untested.
- Does not detect all types of audio manipulation.

### Audio-visual mode
- **This is not a universal deepfake detector.**
- It specifically measures temporal synchronization consistency between audio
  and video.
- A synchronized manipulated video can still be a deepfake.
- An audio-only or video-only manipulated clip can remain synchronized (see
  `docs/experiments.md` Section 3.3 for a measured example of this).
- The Phase 12 controlled-shift validation (best val. AUC = 1.0) was on
  artificial temporal shifts, not real deepfake content, and should not be
  interpreted as 100% real-world deepfake detection accuracy.

Full detail: [docs/experiments.md](../docs/experiments.md) and
[docs/limitations.md](../docs/limitations.md).

## Model architecture

### Audio-only pipeline
```
Audio → Log-Mel Spectrogram → CNN + Transformer Audio Encoder → Spoof Head → BONAFIDE / SPOOF
```

### Audio-visual pipeline
```
Video → Face/Mouth Landmarks → Visual Transformer → Visual Tokens
Audio → CNN + Transformer → Audio Tokens
Audio + Visual → Temporal Alignment → Bidirectional Cross-Attention → Sync Head → SYNC / DESYNC
```

## Checkpoints used (see `backend/main.py` for the exact paths)

- **Audio encoder:** Phase 5 CNN + Transformer (`spoof-transformer-20260906-123646`)
- **Visual encoder:** Phase 8 Landmark Transformer (`deepfake-transformer-final-20260908-210034`)
- **AV sync model:** Phase 12 λ=0.1 contrastive learning (`sync-phase12-lambda01-20260913-115101`)
- **Spoof head:** Phase 5 spoof classifier (`spoof-transformer-20260906-123646`)
- **CNN ensemble member:** Phase 4 CNN baseline (`spoof-cnn-baseline-20260906-104508`)

## Technical details

- Audio sample rate: 16 kHz
- Audio token timing: 0.01 seconds
- Landmark format: MediaPipe 478-point face landmarks
- Device: auto-selects CUDA if available, otherwise CPU
- Inference mode: `torch.inference_mode()` for efficiency
- Scores returned by the API are raw model outputs (post sigmoid/softmax), not
  calibrated probabilities (`docs/limitations.md`, item 11)

## Troubleshooting

### "Failed to initialize predictor" / backend fails to start
- Ensure checkpoint files exist under `outputs/runs/` at the paths listed in
  `backend/main.py`.
- Check that `configs/av_align_lambda01.yaml` exists.
- Verify PyTorch and dependencies are correctly installed (`pip list` inside
  `.venv`).

### "File not found" errors
- Ensure uploaded files are accessible and in a supported format.

### CUDA not available
- The predictor falls back to CPU automatically; inference is slower but
  functional. No latency benchmark has been published for either path.

### MediaPipe errors
- Check that the video contains a detectable face; try a different clip if
  landmark extraction fails.

## Development

Run the backend test suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_backend_main.py tests/test_predictor.py -v
```

## License

This is a research demonstration. Results should not be used for
security-critical decisions without further, independent validation.
