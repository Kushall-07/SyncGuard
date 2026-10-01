# SyncGuard

**Multimodal Deepfake Detection**

SyncGuard is a multimodal deepfake detection system that analyzes audio-visual
synchronization to identify manipulated videos and independently detects
synthetic or cloned speech when only audio is available.

## Problem

Two related but distinct problems: (1) is this speech real or synthetic/cloned
(audio-only), and (2) do the audio and video streams of a clip stay temporally
consistent with each other (audio-visual). SyncGuard treats these as two
separate, explicit inference modes rather than one blended "deepfake score" —
see [Key idea](#key-idea) for why.

## Key idea

**Manipulation is not the same thing as desynchronization.** A face-swapped or
audio-replaced clip can still be perfectly time-aligned; a genuine clip can be
desynchronized by a bad video codec or subtitle timing shift. SyncGuard's
audio-visual branch therefore measures *temporal synchronization consistency*
between audio and video streams — not "is this a deepfake" directly — and the
project documents that distinction explicitly rather than blurring it (see
[Results](#experimental-results) and [Limitations](#limitations)).

## Architecture

**Audio-only mode**

```
Audio → Mel-Spectrogram → CNN Front-End → Audio Transformer → Spoof Head → BONAFIDE / SPOOF
```

**Audio-visual mode**

```
Video ─→ MediaPipe Face/Mouth Landmarks ─→ Visual Landmark Transformer ─┐
                                                                          ├─→ Bidirectional Cross-Attention ─→ Sync Head ─→ per-window score ─→ aggregate SYNC / DESYNC
Audio ─→ Mel-Spectrogram ─→ CNN Front-End ─→ Audio Transformer ─────────┘
         (audio tokens deterministically time-aligned to video-rate buckets before fusion)
```

An optional symmetric InfoNCE contrastive objective (λ-weighted) can be enabled
on top of the sync loss as a representation-learning ablation; see
[docs/experiments.md](docs/experiments.md) Section 4 for why this is not
claimed to improve performance.

The visual branch is **landmark-based** (MediaPipe face/mouth points), not a
CNN over raw face-crop images, and it is retained as an **auxiliary**
multimodal representation for the sync pipeline — not a standalone deepfake
classifier (its standalone numbers are weak; see below).

## Datasets

| Dataset | Used for |
|---|---|
| ASVspoof 2019 LA | Audio-only spoof detection (training + evaluation) |
| Celeb-DF v2 | Visual landmark representation development (video only, no audio used) |
| LAV-DF | Audio-visual synchronization (controlled-shift + independent category evaluation) |

None of these datasets are redistributed in this repository. See
[docs/datasets.md](docs/datasets.md) for source, license, and access details
per dataset, including three additional datasets (FakeAVCeleb, DFDC, KoDF)
investigated as independent AV-sync validation candidates and documented —
but not downloaded — for future use.

## Experimental results

Full detail, sources, and exact interpretation caveats:
[docs/experiments.md](docs/experiments.md). Headline numbers:

**Audio spoof detection (ASVspoof 2019 LA, eval split, unseen attacks)**

| Model | AUC | EER |
|---|---|---|
| CNN | 0.9881 | 4.98% |
| CNN + Transformer | 0.9928 | 3.74% |
| CNN + Transformer ensemble | 0.9978 | 1.93% |

**AV synchronization (LAV-DF dev, controlled temporal shift, Phase 11 model)**

| Audio shift | Mean sync score |
|---|---|
| 0.0s | 0.9909 |
| +0.5s | 0.2545 |
| +1.0s | 0.1552 |
| +2.0s | 0.0000 |

This demonstrates strong sensitivity to **artificial, controlled** audio
delays on unseen clips — it does not demonstrate detection of real-world
deepfakes in general, and no such claim is made. A separate independent check
(native timing, grouped by manipulation category, not by shift) confirms the
model does not spuriously flag manipulated-but-synchronized clips as desynced
— see [docs/experiments.md](docs/experiments.md) Section 3.3.

The standalone visual landmark classifier is intentionally reported as weak
(AUC ≈0.66–0.73 on Celeb-DF v2) because that is the honest result; it is used
only as an auxiliary AV representation, never as a standalone detector.

## Limitations

Full list: [docs/limitations.md](docs/limitations.md). In short: no
independent, annotated audio-visual desynchronization ground truth was found
or used anywhere (every candidate dataset ships manipulation labels, not sync
labels); AV validation is currently controlled-shift-based; the visual branch
is weak standalone; raw scores are not calibrated probabilities; no SOTA or
universal-robustness claim is made.

## Setup

- Python 3.11 (this project was built and tested against 3.11.9)
- PyTorch 2.13 (CUDA 13.0 build used in development; CPU-only also works, slower)
- Node.js 22 / npm (for the frontend)
- A CUDA-capable GPU is recommended for training and for batch evaluation, but
  is not required for interactive inference via the app

```powershell
# Backend Python environment
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Frontend dependencies (first time only)
cd frontend
npm install
cd ..
```

Dataset preparation, checkpoint locations, and full reproducibility notes
(what each script expects, expected outputs, exact commands) are in
[app/README.md](app/README.md), [frontend/README.md](frontend/README.md), and
the frontend's in-app **Reproducibility** page. Model checkpoints used by the
app are under `outputs/runs/<run-name>/checkpoints/` — see the paths hardcoded
in `backend/main.py` for the exact ones currently deployed.

## One-command launch

From the project root, in PowerShell:

```powershell
python .\app\app.py
```

This starts the FastAPI backend and the React/Vite frontend, waits for both to
become ready, prints their URLs, and opens the app in your default browser.

- Frontend: http://127.0.0.1:5173
- Backend: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/api/health
- Stop: press Ctrl+C

The launcher automatically prefers the project's `.venv\Scripts\python.exe`
for the backend, and reuses an already-running backend/frontend instead of
starting a duplicate. If port 8000 or 5173 is occupied by something other
than SyncGuard, it prints an error instead of touching that process.

### Advanced: run backend and frontend manually

```powershell
# Backend (FastAPI), from the project root
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend (Vite dev server), in a separate terminal
cd frontend
npm install   # first time only
npm run dev
```

## Deployment

This is a **research/demo deployment**, not a production-scale inference
service: no autoscaling, no load testing, no GPU in the deployed path. It
lets the existing, frozen SyncGuard model be reached over the internet rather
than only run locally.

### Architecture

```
React/Vite frontend
        |
Vercel / static hosting
        | HTTPS (fetch, relative or VITE_API_BASE_URL)
        v
FastAPI backend  (Docker container, e.g. Render/Railway/Fly.io/Cloud Run)
        |
CPU PyTorch + MediaPipe (torch/torchaudio CPU wheels, no CUDA)
        |
SyncGuard checkpoints  (curated ~90MB subset baked into the image)
```

- The frontend is a static build (`frontend/dist/`); it holds no model state
  and makes no decisions — every score comes from the backend.
- The backend performs all inference, on CPU. **CPU inference is slower than
  local RTX 3050 inference** — see "Observed latency" below for actual
  measured numbers from this deployment prep, not an estimate.
- No dataset (ASVspoof, Celeb-DF, LAV-DF) is deployed anywhere. The
  Synchronization Lab feature specifically depends on cached LAV-DF sample
  clips and is therefore unavailable in this deployment by design — `GET
  /api/lab/samples` returns an empty list, and the frontend shows an
  explanatory message instead of erroring. The Analyze page (upload your own
  audio/video) is unaffected and is the primary demo path.
- Cold-start/idle behavior (whether the container sleeps and takes time to
  wake up) depends entirely on the hosting provider, not on this codebase.

### Required environment variables

| Variable | Where | Purpose | Default |
|---|---|---|---|
| `PORT` | Backend container | Port uvicorn binds to; set automatically by most cloud platforms (Render, Railway, Fly.io, Cloud Run) | `8000` |
| `SYNCGUARD_ALLOWED_ORIGINS` | Backend | Comma-separated production frontend origin(s) for CORS, e.g. `https://syncguard.example.com`. Local dev origins (`localhost:5173`/`127.0.0.1:5173`) are always allowed in addition to this. | unset (no extra origins) |
| `SYNCGUARD_CHECKPOINT_DIR` | Backend | Override where checkpoints are read from (e.g. a mounted volume instead of what's baked into the image) | `<repo root>/outputs/runs` |
| `SYNCGUARD_CONFIG_DIR` | Backend | Override where the sync model's YAML config is read from | `<repo root>/configs` |
| `SYNCGUARD_DATA_DIR` | Backend | Override where Synchronization Lab sample assets are read from (not shipped by default — see above) | `<repo root>/data` |
| `SYNCGUARD_FRONTEND_DIST` | Backend | Only relevant for optional single-container mode (below) | `<repo root>/frontend/dist` |
| `VITE_API_BASE_URL` | Frontend build | Backend origin, e.g. `https://syncguard-api.example.com`. Leave unset for local dev (Vite proxy) or single-container mode (same origin). | unset (relative `/api/...`) |

None of these change model behavior — only where files are read from and
which origins the API accepts requests from.

### Frontend deployment (Vercel or equivalent static hosting)

1. Set `VITE_API_BASE_URL` in the hosting provider's environment variables to
   your deployed backend's URL (skip this only if using single-container
   mode).
2. Build settings: root directory `frontend/`, build command `npm run build`,
   output directory `dist`.
3. Deploy. Vercel (or Netlify/Cloudflare Pages) auto-detects the Vite project;
   no server-side code runs here.

### Backend deployment (Docker-compatible CPU host)

```powershell
# Build (from the repo root)
docker build -t syncguard-backend .

# Run locally to verify
docker run --rm -p 8000:8000 `
  -e SYNCGUARD_ALLOWED_ORIGINS=https://your-frontend.example.com `
  syncguard-backend

# Verify
curl http://127.0.0.1:8000/api/health
```

Push the built image to any Docker-compatible CPU host (Render, Railway,
Fly.io, Google Cloud Run, etc.) per that provider's own deploy instructions —
this Dockerfile has no provider-specific assumptions beyond honoring `$PORT`.
**Hugging Face Docker Spaces is an optional, paid deployment target** for this
project (creating a Docker/compute Space currently requires a paid plan even
though CPU Basic hardware itself is free) — not assumed as a free default.

### Docker build/run reference

- `Dockerfile` — production image (Python 3.11.9-slim, CPU-only, no CUDA).
- `.dockerignore` — excludes datasets, full `outputs/`, dev tooling, and the
  frontend source/build tooling from the build context entirely.
- `requirements.deploy.txt` — pruned runtime dependencies (see file header for
  exactly what was excluded from `requirements.txt` and why).

### Model asset requirements

The image bakes in a curated ~90MB subset of `outputs/runs/`, not the whole
directory:

| File | Used for |
|---|---|
| `spoof-transformer-20260906-123646/checkpoints/best.pt` + its `config.yaml` | Audio-only mode: Transformer spoof classifier (also the frozen audio encoder for AV mode) |
| `spoof-cnn-baseline-20260906-104508/checkpoints/best.pt` + its `config.yaml` | Audio-only mode: CNN half of the ensemble |
| `deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt` | AV mode: frozen visual landmark encoder (self-contained export, no separate config needed) |
| `sync-physical-v2-20260927-020548/checkpoints/best.pt` | AV mode (production default): physically-shift-trained cross-attention + Sync Head, paired with `configs/av_align_physical.yaml` (already in the repo) — see `docs/decisions/0006-physical-sync-training.md` |
| `sync-phase12-lambda01-20260913-115101/checkpoints/best.pt` | AV mode (old, kept for comparison only — set `SYNCGUARD_SYNC_MODEL_PATH`/`SYNCGUARD_SYNC_CONFIG_PATH` to use it instead): the original token-shift-trained cross-attention + Sync Head, paired with `configs/av_align_lambda01.yaml` |
| `checkpoints/mediapipe/face_landmarker.task` | AV mode: on-the-fly landmark extraction when a client uploads a video without precomputed landmarks |

### CPU inference limitation and observed latency

This deployment runs PyTorch on CPU (`torch==2.13.0+cpu` /
`torchaudio==2.11.0+cpu` — confirmed via `torch.cuda.is_available() == False`
inside the built container). Measured against the actual built image on this
machine (Docker Desktop, CPU-only, single uvicorn worker, no concurrency), not
estimated:

| Request | Steady-state wall time |
|---|---|
| Audio-only (`/api/analyze/audio`) | ~0.07–0.08s |
| Audio-visual (`/api/analyze/av`), ~5s clip, on-the-fly MediaPipe extraction | ~2.0s |

These are single-request, single-machine measurements, not a load-tested
benchmark, and will vary by host CPU. **This is not described as "real-time"
anywhere in this project.**

### Optional single-container mode

`backend/main.py` will additionally serve a built frontend from the same
FastAPI process if `frontend/dist/` (or `SYNCGUARD_FRONTEND_DIST`) exists —
`/` and all client-side routes fall back to `index.html`, static assets are
served as files, and `/api/*` is untouched. This is off by default (the
primary target is separate frontend/backend deployment) and requires two
manual steps to enable in Docker, since `frontend/dist/` isn't built inside
the image by default:

```dockerfile
# Add to Dockerfile, after `npm run build` has produced frontend/dist/ locally:
COPY frontend/dist/ frontend/dist/
```

Leave `VITE_API_BASE_URL` unset when building the frontend for this mode,
since the frontend and API then share one origin.

### Dataset non-distribution

No dataset (ASVspoof 2019 LA, Celeb-DF v2, LAV-DF) is included in the Docker
image, the frontend build, or any deployment artifact — verified by
`.dockerignore` and by inspecting `SyncGuardPredictor`'s actual checkpoint
requirements before deciding what to `COPY`. See
[docs/datasets.md](docs/datasets.md).

## Project structure

```
SyncGuard/
├── Dockerfile                    Production backend image (CPU-only)
├── .dockerignore                 Keeps datasets/dev tooling out of the build context
├── requirements.deploy.txt       Pruned production Python dependencies
├── app/app.py                  One-command launcher (backend + frontend)
├── backend/main.py              FastAPI wrapper around SyncGuardPredictor
├── frontend/                    React/Vite web app (Home, Analyze, Results,
│                                 Technology, Research, About, Synchronization
│                                 Lab, Reproducibility)
├── src/
│   ├── features/                Mel-spectrogram extraction
│   ├── preprocessing/           Audio/landmark preprocessing, LAV-DF helpers
│   ├── models/                  Audio/video encoders, fusion, heads (frozen
│   │                             at inference; not modified in this phase)
│   ├── losses/                  Sync loss, contrastive loss
│   ├── training/                Trainers (not run in this phase)
│   ├── data/                    Dataset loaders + manifests
│   ├── evaluation/               Metrics (sync, spoof, deepfake)
│   └── inference/predictor.py   SyncGuardPredictor — the single inference API
├── scripts/                     Manifest builders, training/eval scripts,
│                                 including evaluate_sync.py (controlled-shift
│                                 experiment) and evaluate_av_independent.py
│                                 (native-timing, manipulation-category check)
├── configs/                     YAML configs per model/experiment
├── outputs/runs/                Checkpoints + training/eval logs (not in Git)
├── data/                        Local dataset copies (not in Git; see
│                                 docs/datasets.md)
├── docs/
│   ├── experiments.md           What was measured, where, with what caveats
│   ├── limitations.md           Everything the evaluation does not establish
│   ├── datasets.md              Dataset source/license/redistribution notes
│   └── decisions/                Append-only per-phase decision records
└── tests/                       pytest suite (backend, models, data, eval)
```

## Contributing / more detail

This is a research project, not a commercial product — see the in-app
**About** page. Per-phase design rationale and the full audit trail for every
headline number live in `docs/decisions/`.
