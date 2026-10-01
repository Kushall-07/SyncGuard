# Production image for the SyncGuard FastAPI backend — CPU inference only.
#
# The ML pipeline is frozen: this file installs dependencies and copies source
# + a curated set of already-trained checkpoints. It does not train, retrain,
# or regenerate anything, and does not include any dataset (ASVspoof, Celeb-DF,
# LAV-DF) — see docs/datasets.md.
FROM python:3.11.9-slim

# Runtime system libraries required by mediapipe (OpenCV + GLES-based face
# landmarker native library) and soundfile. No CUDA/NVIDIA packages: this
# image is CPU-only by design. libegl1/libgles2 were confirmed required by an
# actual container run (mediapipe's FaceLandmarker.create_from_options failed
# with "OSError: libGLESv2.so.2: cannot open shared object file" without them).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across source-only changes.
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

# Application source. The lazy MediaPipe-landmarker import in
# scripts/extract_celebdf_landmarks.py (used only when a client uploads a video
# without precomputed landmarks) pulls in a couple of src/data modules, so the
# whole (small, source-only) src/ and scripts/ trees are copied rather than
# hand-picking files.
COPY backend/ backend/
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY checkpoints/ checkpoints/

# Curated frozen model checkpoints actually required by SyncGuardPredictor as
# configured in backend/main.py — NOT the full outputs/ directory (which also
# holds training logs and experiment/analysis artifacts irrelevant to serving
# inference). See README.md "Model asset requirements" for what each file is.
COPY outputs/runs/spoof-transformer-20260906-123646/checkpoints/best.pt \
     outputs/runs/spoof-transformer-20260906-123646/checkpoints/best.pt
COPY outputs/runs/spoof-transformer-20260906-123646/config.yaml \
     outputs/runs/spoof-transformer-20260906-123646/config.yaml
COPY outputs/runs/spoof-cnn-baseline-20260906-104508/checkpoints/best.pt \
     outputs/runs/spoof-cnn-baseline-20260906-104508/checkpoints/best.pt
COPY outputs/runs/spoof-cnn-baseline-20260906-104508/config.yaml \
     outputs/runs/spoof-cnn-baseline-20260906-104508/config.yaml
COPY outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt \
     outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt
COPY outputs/runs/sync-phase12-lambda01-20260913-115101/checkpoints/best.pt \
     outputs/runs/sync-phase12-lambda01-20260913-115101/checkpoints/best.pt
# Physically-shift-trained AV sync checkpoint (production default as of the
# sync-detection fix; see docs/decisions/0006-physical-sync-training.md). The
# old checkpoint above is kept in the image too, purely for comparison via
# SYNCGUARD_SYNC_MODEL_PATH/SYNCGUARD_SYNC_CONFIG_PATH env overrides.
COPY outputs/runs/sync-physical-v2-20260927-020548/checkpoints/best.pt \
     outputs/runs/sync-physical-v2-20260927-020548/checkpoints/best.pt
COPY outputs/runs/sync-physical-v2-20260927-020548/config.yaml \
     outputs/runs/sync-physical-v2-20260927-020548/config.yaml

# SYNCGUARD_CHECKPOINT_DIR / SYNCGUARD_CONFIG_DIR / SYNCGUARD_DATA_DIR default to
# repo-relative paths (see backend/main.py) which already resolve correctly here
# (WORKDIR /app mirrors the copies above); set them only to point at a different
# checkpoint source (e.g. a mounted volume) without rebuilding this image.
# SYNCGUARD_ALLOWED_ORIGINS: comma-separated production frontend origin(s) for CORS.
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# ${PORT} is provided by most cloud platforms (Render, Railway, Fly.io, Cloud
# Run) and overrides the default above; shell form so it actually expands.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
