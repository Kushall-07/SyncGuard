"""Audio branch models: CNN front-end (Phase 4), shared encoder + Transformer (Phase 5)."""

from src.models.audio.cnn import AudioCNNEncoder, AudioEncoderOutput
from src.models.audio.encoder import AudioEncoder, export_audio_encoder, load_audio_encoder
from src.models.audio.spoof_classifier import SpoofClassifier
from src.models.audio.transformer import AudioTransformerEncoder, SinusoidalPositionalEncoding

__all__ = [
    "AudioCNNEncoder",
    "AudioEncoderOutput",
    "AudioEncoder",
    "export_audio_encoder",
    "load_audio_encoder",
    "AudioTransformerEncoder",
    "SinusoidalPositionalEncoding",
    "SpoofClassifier",
]
