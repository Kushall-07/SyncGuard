"""Visual branch models (Phase 7): landmark embedding, non-temporal baseline,
shared temporal encoder, and the deepfake classifier."""

from src.models.video.deepfake_classifier import DeepfakeClassifier
from src.models.video.landmark_baseline import LandmarkMLPBaseline
from src.models.video.landmark_embedding import LandmarkEmbedding
from src.models.video.visual_encoder import (
    VisualEncoder,
    VisualEncoderOutput,
    export_visual_encoder,
    load_visual_encoder,
)

__all__ = [
    "LandmarkEmbedding",
    "LandmarkMLPBaseline",
    "VisualEncoder",
    "VisualEncoderOutput",
    "export_visual_encoder",
    "load_visual_encoder",
    "DeepfakeClassifier",
]
