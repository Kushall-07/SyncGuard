"""Dual-mode inference API for SyncGuard (Phase 13A).

Provides two explicit inference paths:
1. Audio-only: frozen Phase-5 AudioEncoder + SpoofHead for synthetic/cloned/spoofed speech detection
2. Audio-visual: frozen encoders + Phase 9-11 pipeline for temporal synchronization detection

IMPORTANT LIMITATIONS:
- The AV SyncHead measures temporal synchronization between audio and video streams.
- It is NOT a universal deepfake detector - it detects temporal misalignment.
- Controlled-shift validation results (e.g., LAV-DF with artificial temporal shifts) are not
  equivalent to real-world deepfake detection accuracy.
- This inference layer separates prediction from training to enable clean deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.config import AudioConfig, ModelConfig, VideoDataConfig, load_config
from src.features.mel_spectrogram import compute_log_mel
from src.models.audio.encoder import load_audio_encoder
from src.models.audio import SpoofClassifier
from src.models.fusion.cross_attention import (
    BidirectionalCrossAttention,
    CrossAttentionConfig,
)
from src.models.fusion.temporal_align import align_audio_to_video
from src.models.heads.spoof_head import SpoofHead
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.video.visual_encoder import load_visual_encoder
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.landmarks import (
    interpolate_invalid,
    normalize_landmarks,
    region_indices,
    region_point_count,
)
from src.preprocessing.lavdf import extract_audio_from_mp4, extract_landmarks_from_mp4
from src.training.utils import get_device, set_seed

__all__ = [
    "SyncGuardPredictor",
    "AudioOnlyResult",
    "AudioVisualResult",
    "SyncLabResult",
]


@dataclass
class AudioOnlyResult:
    """Structured result for audio-only spoof detection."""

    mode: str = "audio_only"
    predicted_label: str = "unknown"  # "bonafide" or "spoof"
    spoof_probability: float = 0.0
    bonafide_probability: float = 0.0
    confidence: float = 0.0  # max of the two probabilities
    audio_encoder_checkpoint: str = ""
    spoof_head_checkpoint: str = ""
    cnn_checkpoint: str = ""  # CNN checkpoint if ensemble used
    use_ensemble: bool = False  # Whether ensemble was used

    def __post_init__(self) -> None:
        if self.predicted_label not in ("bonafide", "spoof", "unknown"):
            raise ValueError(f"predicted_label must be 'bonafide', 'spoof', or 'unknown', got {self.predicted_label!r}")
        if not 0.0 <= self.spoof_probability <= 1.0:
            raise ValueError(f"spoof_probability must be in [0, 1], got {self.spoof_probability}")
        if not 0.0 <= self.bonafide_probability <= 1.0:
            raise ValueError(f"bonafide_probability must be in [0, 1], got {self.bonafide_probability}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class SyncLabResult:
    """Result of a controlled temporal-shift synchronization experiment.

    Used by the Synchronization Lab, which demonstrates how the trained SyncHead
    responds to an explicit, user-chosen audio-video offset. This is a research
    / educational tool, not a deepfake detector: shifting audio relative to video
    is a controlled manipulation of timing only, distinct from content manipulation.
    """

    mode: str = "sync_lab"
    sample_id: str = ""
    shift_seconds: float = 0.0
    aggregate_sync_score: float = 0.0
    per_window_sync_scores: list[float] | None = None
    timing_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.aggregate_sync_score <= 1.0:
            raise ValueError(f"aggregate_sync_score must be in [0, 1], got {self.aggregate_sync_score}")
        if self.per_window_sync_scores is not None:
            if not all(0.0 <= s <= 1.0 for s in self.per_window_sync_scores):
                raise ValueError("all per_window_sync_scores must be in [0, 1]")


@dataclass
class AudioVisualResult:
    """Structured result for audio-visual synchronization detection."""

    mode: str = "audio_visual"
    predicted_label: str = "unknown"  # "sync" or "desync"
    sync_probability: float = 0.0
    desync_probability: float = 0.0
    aggregate_sync_score: float = 0.0  # video-level sync score (after aggregation)
    per_window_sync_scores: list[float] | None = None  # per-window sync probabilities
    timing_metadata: dict[str, Any] | None = None  # fps, window_seconds, num_frames, etc.
    audio_encoder_checkpoint: str = ""
    visual_encoder_checkpoint: str = ""
    sync_model_checkpoint: str = ""

    def __post_init__(self) -> None:
        if self.predicted_label not in ("sync", "desync", "unknown"):
            raise ValueError(f"predicted_label must be 'sync', 'desync', or 'unknown', got {self.predicted_label!r}")
        if not 0.0 <= self.sync_probability <= 1.0:
            raise ValueError(f"sync_probability must be in [0, 1], got {self.sync_probability}")
        if not 0.0 <= self.desync_probability <= 1.0:
            raise ValueError(f"desync_probability must be in [0, 1], got {self.desync_probability}")
        if not 0.0 <= self.aggregate_sync_score <= 1.0:
            raise ValueError(f"aggregate_sync_score must be in [0, 1], got {self.aggregate_sync_score}")
        if self.per_window_sync_scores is not None:
            if not all(0.0 <= s <= 1.0 for s in self.per_window_sync_scores):
                raise ValueError("all per_window_sync_scores must be in [0, 1]")


class SyncGuardPredictor:
    """Dual-mode inference predictor for SyncGuard.

    Provides separate inference paths for:
    1. Audio-only spoof detection using Phase-5 AudioEncoder + SpoofHead
    2. Audio-visual synchronization detection using frozen encoders + Phase 9-11 pipeline

    Checkpoints used:
    - Audio encoder: Phase-5 shared CNN + Transformer encoder
    - Visual encoder: Phase-8 landmark embedding + Transformer encoder
    - Sync model: Phase-12 trained cross-attention + sync head (λ=0.1 contrastive)
    - CNN model (optional): Phase-4 CNN for ensemble (if cnn_checkpoint provided)

    IMPORTANT:
    - Encoders are frozen and run in eval() mode
    - Inference uses torch.inference_mode() for efficiency
    - Audio is preprocessed to 16 kHz mono with peak normalization
    - Video landmarks are expected to be MediaPipe 478-point face landmarks
    - Temporal alignment uses 0.01-second audio tokens (from Phase-5 encoder)
    - When CNN checkpoint is provided, audio-only uses CNN+Transformer mean ensemble (Phase 6)
    """

    def __init__(
        self,
        *,
        audio_encoder_path: str | Path | None = None,
        visual_encoder_path: str | Path,
        sync_model_path: str | Path,
        sync_config_path: str | Path,
        spoof_head_checkpoint: str | Path | None = None,
        cnn_checkpoint: str | Path | None = None,
        device: str | torch.device = "auto",
        audio_config: AudioConfig | None = None,
        video_config: VideoDataConfig | None = None,
    ) -> None:
        """Initialize the dual-mode predictor.

        Args:
            audio_encoder_path: Optional path to Phase-5 audio encoder checkpoint (legacy, not needed if spoof_head_checkpoint is a full model)
            visual_encoder_path: Path to Phase-8 visual encoder checkpoint
            sync_model_path: Path to Phase-12 sync model checkpoint (best.pt)
            sync_config_path: Path to sync training config YAML (av_align_lambda01.yaml)
            spoof_head_checkpoint: Path to Transformer model checkpoint (full model with encoder + head)
            cnn_checkpoint: Optional path to Phase-4 CNN checkpoint for ensemble (if using ensemble)
            device: Device to use ("auto", "cpu", or "cuda")
            audio_config: Optional audio config (defaults to 16 kHz mono)
            video_config: Optional video config (defaults to 32 frames face_mouth)
        """
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load configs
        self.audio_config = audio_config or AudioConfig()
        self.video_config = video_config or VideoDataConfig(
            num_frames=32,
            regions="face_mouth",
            normalize="interocular",
            align_rotation=True,
        )
        
        # Cache region indices for landmark selection
        self._region_indices = region_indices(self.video_config.regions)
        self._landmark_coords = 3  # Fixed to 3 for xyz coordinates

        # Load sync config
        with open(sync_config_path, "r", encoding="utf-8") as f:
            sync_config_dict = yaml.safe_load(f)

        # Load full Transformer model for audio-only mode (preferred approach)
        self.transformer_model = None
        self.spoof_head = None
        self.spoof_head_checkpoint = ""
        audio_payload = None  # Initialize to None
        if spoof_head_checkpoint is not None:
            print(f"Loading Transformer model from {spoof_head_checkpoint}")
            
            # The checkpoint is a full Transformer model (encoder + head)
            # Load it as a full model to match the ensemble experiment's approach
            spoof_ckpt = torch.load(spoof_head_checkpoint, map_location=self.device, weights_only=False)
            
            # Load the config from the Transformer run directory
            tf_config_path = Path(spoof_head_checkpoint).parent.parent / "config.yaml"
            if tf_config_path.exists():
                tf_cfg = load_config(tf_config_path)
                set_seed(tf_cfg.experiment.seed)
                device = get_device()
                
                # Load full Transformer model with correct config
                self.transformer_model = SpoofClassifier(tf_cfg.model, n_mels=self.audio_config.mel.n_mels).to(self.device)
                
                # Load state dict
                if "model" in spoof_ckpt:
                    self.transformer_model.load_state_dict(spoof_ckpt["model"], strict=False)
                else:
                    self.transformer_model.load_state_dict(spoof_ckpt, strict=False)
                
                self.transformer_model.eval()
                for param in self.transformer_model.parameters():
                    param.requires_grad = False
                
                # Use the encoder from the full model for AV path
                self.audio_encoder = self.transformer_model.encoder
                self.spoof_head = self.transformer_model.head
            else:
                # Fallback: load encoder separately and try to load head weights
                print("Warning: Could not load Transformer config, using separate encoder + head")
                # Keep the existing behavior as fallback
                if audio_encoder_path is None:
                    raise ValueError("audio_encoder_path must be provided when spoof_head_checkpoint config is not available")
                
                print(f"Loading audio encoder from {audio_encoder_path}")
                self.audio_encoder, audio_payload = load_audio_encoder(
                    audio_encoder_path, map_location=self.device
                )
                self.audio_encoder.eval()
                for param in self.audio_encoder.parameters():
                    param.requires_grad = False
                
                embed_dim = self.audio_encoder.output_dim
                self.spoof_head = SpoofHead(
                    in_dim=embed_dim,
                    hidden=128,
                    n_classes=2,
                    dropout=0.1,
                    pooling="attentive",
                )
                self.spoof_head.to(self.device)
                self.spoof_head.eval()
                
                # Try to load head weights from the full checkpoint
                if "model" in spoof_ckpt:
                    head_weights = {k.replace("head.", ""): v for k, v in spoof_ckpt["model"].items() if "head." in k}
                    if head_weights:
                        self.spoof_head.load_state_dict(head_weights, strict=False)
                
                for param in self.spoof_head.parameters():
                    param.requires_grad = False
            
            self.spoof_head_checkpoint = str(spoof_head_checkpoint)
        elif audio_encoder_path is not None:
            # Legacy mode: load encoder separately (no spoof head)
            print(f"Loading audio encoder from {audio_encoder_path}")
            self.audio_encoder, audio_payload = load_audio_encoder(
                audio_encoder_path, map_location=self.device
            )
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Either spoof_head_checkpoint or audio_encoder_path must be provided")

        # Compute audio token seconds from payload or config
        if audio_encoder_path is not None:
            # We loaded the encoder separately, so we have audio_payload
            hop_length = audio_payload["audio_cfg"]["mel"]["hop_length"]
            time_downsample = int(audio_payload.get("time_downsample", 1))
            sample_rate = audio_payload["audio_cfg"]["sample_rate"]
        else:
            # We loaded the full Transformer model, use config from audio_config
            hop_length = self.audio_config.mel.hop_length
            time_downsample = 1  # Default for Transformer encoder
            sample_rate = self.audio_config.sample_rate
        
        self.audio_token_seconds = (hop_length * time_downsample) / sample_rate
        print(f"Audio token seconds: {self.audio_token_seconds:.4f}")

        # Load visual encoder
        print(f"Loading visual encoder from {visual_encoder_path}")
        self.visual_encoder, visual_payload = load_visual_encoder(
            visual_encoder_path, map_location=self.device
        )
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # Load sync model components from config
        ca_cfg_dict = sync_config_dict.get("av_align", {}).get("cross_attention", {})
        self.ca_cfg = CrossAttentionConfig(**ca_cfg_dict)
        self.cross_attention = BidirectionalCrossAttention.from_config(self.ca_cfg)
        self.cross_attention.to(self.device)
        self.cross_attention.eval()

        sh_cfg_dict = sync_config_dict.get("av_align", {}).get("sync_head", {})
        self.sh_cfg = SyncHeadConfig(**sh_cfg_dict)
        self.sync_head = SyncHead.from_config(self.sh_cfg, input_dim=self.ca_cfg.dim)
        self.sync_head.to(self.device)
        self.sync_head.eval()

        # Load sync model checkpoint
        print(f"Loading sync model from {sync_model_path}")
        sync_checkpoint = torch.load(sync_model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint structures
        if "model" in sync_checkpoint:
            # Phase 12 trainer checkpoint structure
            state_dict = sync_checkpoint["model"]
            # Try to load cross_attention and sync_head with various key prefixes
            ca_keys = {k.replace("cross_attention.", "").replace("model.cross_attention.", ""): v 
                      for k, v in state_dict.items() if "cross_attention" in k}
            sh_keys = {k.replace("sync_head.", "").replace("model.sync_head.", ""): v 
                      for k, v in state_dict.items() if "sync_head" in k}
            
            if ca_keys:
                self.cross_attention.load_state_dict(ca_keys, strict=False)
            if sh_keys:
                self.sync_head.load_state_dict(sh_keys, strict=False)
        else:
            # Direct state dict (fallback)
            # Try to load with various key patterns
            ca_keys = {k.replace("cross_attention.", ""): v for k, v in sync_checkpoint.items() if "cross_attention" in k}
            sh_keys = {k.replace("sync_head.", ""): v for k, v in sync_checkpoint.items() if "sync_head" in k}
            
            if ca_keys:
                self.cross_attention.load_state_dict(ca_keys, strict=False)
            if sh_keys:
                self.sync_head.load_state_dict(sh_keys, strict=False)

        # Load CNN model for ensemble (optional)
        self.cnn_model = None
        self.cnn_checkpoint = ""
        self.use_ensemble = False
        if cnn_checkpoint is not None:
            print(f"Loading CNN model from {cnn_checkpoint} for ensemble")
            
            # Load CNN using the full model checkpoint with correct config
            cnn_path = Path(cnn_checkpoint)
            
            # Try to load the config from the CNN run directory
            cnn_config_path = cnn_path.parent.parent / "config.yaml"
            if cnn_config_path.exists():
                cnn_cfg = load_config(cnn_config_path)
                set_seed(cnn_cfg.experiment.seed)
                device = get_device()
                
                # Load CNN model with correct config
                self.cnn_model = SpoofClassifier(cnn_cfg.model, n_mels=self.audio_config.mel.n_mels).to(self.device)
                
                # Load state dict
                cnn_ckpt = torch.load(cnn_path, map_location=self.device, weights_only=False)
                if "model" in cnn_ckpt:
                    self.cnn_model.load_state_dict(cnn_ckpt["model"], strict=False)
                else:
                    self.cnn_model.load_state_dict(cnn_ckpt, strict=False)
                
                self.cnn_model.eval()
                for param in self.cnn_model.parameters():
                    param.requires_grad = False
            else:
                # Fallback to hardcoded config
                cnn_cfg = ModelConfig(
                    audio_encoder="cnn",
                    audio_cnn_channels=[32, 64, 128],
                    audio_cnn_dropout=0.1,
                    audio_embedding_dim=256,
                    spoof_head_hidden=128,
                    spoof_head_pooling="attentive",
                )
                
                self.cnn_model = SpoofClassifier(cnn_cfg, n_mels=self.audio_config.mel.n_mels).to(self.device)
                
                cnn_ckpt = torch.load(cnn_path, map_location=self.device, weights_only=False)
                if "model" in cnn_ckpt:
                    self.cnn_model.load_state_dict(cnn_ckpt["model"], strict=False)
                else:
                    self.cnn_model.load_state_dict(cnn_ckpt, strict=False)
                
                self.cnn_model.eval()
                for param in self.cnn_model.parameters():
                    param.requires_grad = False
            
            self.cnn_checkpoint = str(cnn_path)
            self.use_ensemble = True
            print("CNN model loaded successfully, ensemble mode enabled")

        # Store checkpoint paths for metadata
        self.audio_encoder_checkpoint = str(audio_encoder_path)
        self.visual_encoder_checkpoint = str(visual_encoder_path)
        self.sync_model_checkpoint = str(sync_model_path)

        print(f"Predictor initialized on device: {self.device}")

    def predict_audio(
        self,
        audio_path: str | Path,
    ) -> AudioOnlyResult:
        """Run audio-only spoof detection.

        Args:
            audio_path: Path to audio file (WAV, FLAC, etc.)

        Returns:
            AudioOnlyResult with spoof/bonafide classification

        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If spoof head not loaded or audio processing fails
        """
        if self.spoof_head is None:
            raise ValueError(
                "Spoof head not loaded. Provide spoof_head_checkpoint during initialization."
            )

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with torch.inference_mode():
            # Preprocess audio
            waveform = preprocess_audio(audio_path, self.audio_config, device=self.device)

            # Crop/pad to fixed duration (matching evaluation pipeline)
            # The evaluation pipeline uses fixed_seconds=4.0 for ASVspoof
            target_len = int(round(4.0 * self.audio_config.sample_rate))
            n = waveform.shape[-1]
            if n > target_len:
                # Center crop (matching eval pipeline's deterministic behavior)
                start = (n - target_len) // 2
                waveform = waveform[..., start : start + target_len]
            elif n < target_len:
                # Right-pad with zeros
                pad = waveform.new_zeros(*waveform.shape[:-1], target_len - n)
                waveform = torch.cat([waveform, pad], dim=-1)

            # Compute log-mel spectrogram
            mel = compute_log_mel(waveform, self.audio_config, device=self.device)

            # Add batch dimension
            mel = mel.unsqueeze(0)  # [1, n_mels, T]

            # Get Transformer scores using the full model (matching ensemble experiment)
            if self.transformer_model is not None:
                # Use full Transformer model (encoder + head together)
                tf_logits = self.transformer_model(mel)  # [1, 2]
                tf_probs = torch.softmax(tf_logits, dim=-1)[0]  # [2]
                tf_bonafide_prob = tf_probs[1].item()
                tf_spoof_prob = tf_probs[0].item()
            else:
                # Fallback to separate encoder + head
                audio_out = self.audio_encoder(mel)
                audio_tokens = audio_out.tokens  # [1, T, D]
                tf_logits = self.spoof_head(audio_tokens)  # [1, 2]
                tf_probs = torch.softmax(tf_logits, dim=-1)[0]  # [2]
                tf_bonafide_prob = tf_probs[1].item()
                tf_spoof_prob = tf_probs[0].item()

            # Get CNN scores if ensemble is enabled
            if self.use_ensemble:
                if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
                    # Use separate CNN encoder + head
                    cnn_out = self.cnn_encoder(mel)
                    cnn_tokens = cnn_out.tokens  # [1, T, D]
                    cnn_logits = self.cnn_head(cnn_tokens)  # [1, 2]
                    cnn_probs = torch.softmax(cnn_logits, dim=-1)[0]  # [2]
                    cnn_bonafide_prob = cnn_probs[1].item()
                    cnn_spoof_prob = cnn_probs[0].item()
                elif self.cnn_model is not None:
                    # Use full CNN model
                    cnn_logits = self.cnn_model(mel)  # [1, 2]
                    cnn_probs = torch.softmax(cnn_logits, dim=-1)[0]  # [2]
                    cnn_bonafide_prob = cnn_probs[1].item()
                    cnn_spoof_prob = cnn_probs[0].item()
                else:
                    # Fallback to Transformer only
                    cnn_bonafide_prob = tf_bonafide_prob
                    cnn_spoof_prob = tf_spoof_prob
                
                # Mean ensemble (matching Phase 6 ensemble experiment)
                bonafide_prob = 0.5 * tf_bonafide_prob + 0.5 * cnn_bonafide_prob
                spoof_prob = 0.5 * tf_spoof_prob + 0.5 * cnn_spoof_prob
            else:
                bonafide_prob = tf_bonafide_prob
                spoof_prob = tf_spoof_prob

            predicted_label = "bonafide" if bonafide_prob >= spoof_prob else "spoof"
            confidence = max(bonafide_prob, spoof_prob)

        return AudioOnlyResult(
            mode="audio_only",
            predicted_label=predicted_label,
            spoof_probability=spoof_prob,
            bonafide_probability=bonafide_prob,
            confidence=confidence,
            audio_encoder_checkpoint=self.audio_encoder_checkpoint,
            spoof_head_checkpoint=self.spoof_head_checkpoint,
            cnn_checkpoint=self.cnn_checkpoint if self.use_ensemble else "",
            use_ensemble=self.use_ensemble,
        )

    def predict_audio_visual(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        landmarks_path: str | Path | None = None,
        landmarker: Any = None,
    ) -> AudioVisualResult:
        """Run audio-visual synchronization detection.

        Args:
            video_path: Path to video file (MP4, etc.)
            audio_path: Optional path to separate audio file. If None, extracts from video.
            landmarks_path: Optional path to pre-computed landmarks NPZ file.
            landmarker: Optional MediaPipe FaceLandmarker instance for landmark extraction.

        Returns:
            AudioVisualResult with sync/desync classification and per-window scores

        Raises:
            FileNotFoundError: If video/audio/landmarks file not found
            ValueError: If processing fails or required components missing
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        timing_metadata: dict[str, Any] = {}

        with torch.inference_mode():
            # Extract or load audio
            if audio_path is not None:
                audio_path = Path(audio_path)
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                waveform = preprocess_audio(audio_path, self.audio_config, source_sr=None, device="cpu")
            else:
                # Extract audio from video
                waveform, sr = extract_audio_from_mp4(video_path, target_sr=self.audio_config.sample_rate)
                waveform = preprocess_audio(waveform, self.audio_config, source_sr=sr, device="cpu")

            waveform = waveform.to(self.device)

            # Extract or load landmarks
            if landmarks_path is not None:
                landmarks_path = Path(landmarks_path)
                if not landmarks_path.is_file():
                    raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")

                data = np.load(landmarks_path)
                landmarks_np = np.asarray(data["points"], dtype=np.float32)  # [T, N, 3]
                fps = float(data["fps"])
                valid = np.asarray(data.get("valid", np.ones(len(landmarks_np), dtype=bool)), dtype=bool)
            else:
                # Extract landmarks from video
                if landmarker is None:
                    raise ValueError(
                        "landmarker required when landmarks_path not provided. "
                        "Initialize MediaPipe FaceLandmarker and pass to this method."
                    )
                lm_res = extract_landmarks_from_mp4(video_path, landmarker)
                landmarks_np = np.asarray(lm_res["points"], dtype=np.float32)  # [T, N, 3]
                fps = lm_res["fps"]
                valid = np.asarray(lm_res["valid"], dtype=bool)

            # Apply canonical Phase 7/8 landmark preprocessing
            # 1. Interpolate invalid frames
            landmarks_np = interpolate_invalid(landmarks_np, valid)
            
            # 2. Normalize using interocular method (matches training)
            landmarks_np = normalize_landmarks(
                landmarks_np,
                method=self.video_config.normalize,
                align_rotation=self.video_config.align_rotation,
                coords=self._landmark_coords,
            )
            
            # 3. Select face_mouth region (142 landmarks from 478 MediaPipe points)
            landmarks_np = landmarks_np[:, self._region_indices, :]  # [T, 142, 3]
            
            # Convert to tensor and move to device
            landmarks = torch.from_numpy(landmarks_np).float().to(self.device)

            # Compute log-mel spectrogram
            mel = compute_log_mel(waveform, self.audio_config, device=self.device)

            # Add batch dimension
            mel = mel.unsqueeze(0)  # [1, n_mels, T]
            landmarks = landmarks.unsqueeze(0)  # [1, T, N, 3]

            # Encode with encoders
            audio_out = self.audio_encoder(mel)
            audio_tokens = audio_out.tokens  # [1, T_a, D]

            visual_out = self.visual_encoder(landmarks)
            visual_tokens = visual_out.tokens  # [1, T_v, D]

            T_v = visual_tokens.shape[1]
            window_seconds = T_v / fps

            # Phase 9: Temporal alignment (no shift for inference)
            audio_aligned, bucket_counts = align_audio_to_video(
                audio_tokens,  # [1, T_a, D]
                n_video_tokens=T_v,
                audio_token_seconds=self.audio_token_seconds,
                video_fps=fps,
                window_seconds=window_seconds,
                empty_bucket="nearest",
            )

            # Phase 10: Cross-attention
            fused_out = self.cross_attention(audio_aligned, visual_tokens)
            fused = fused_out.fused  # [1, T_v, D]

            # Phase 11: Sync head
            logits = self.sync_head(fused).squeeze(0)  # [T_v]
            probs = torch.sigmoid(logits)  # [T_v]

            # Aggregate to video-level score
            # Use mean aggregation (matching training config)
            valid_mask = torch.from_numpy(valid).bool().to(self.device)
            if valid_mask.any():
                aggregate_score = probs[valid_mask].mean().item()
            else:
                aggregate_score = probs.mean().item()

            # Convert to CPU for output
            per_window_scores = probs.cpu().tolist()

            # Determine label
            sync_prob = aggregate_score
            desync_prob = 1.0 - sync_prob
            predicted_label = "sync" if sync_prob >= 0.5 else "desync"

            # Timing metadata
            timing_metadata = {
                "fps": fps,
                "num_frames": T_v,
                "window_seconds": window_seconds,
                "audio_token_seconds": self.audio_token_seconds,
                "num_valid_landmark_frames": int(valid.sum()),
                "num_windows": len(per_window_scores),
            }

        return AudioVisualResult(
            mode="audio_visual",
            predicted_label=predicted_label,
            sync_probability=sync_prob,
            desync_probability=desync_prob,
            aggregate_sync_score=aggregate_score,
            per_window_sync_scores=per_window_scores,
            timing_metadata=timing_metadata,
            audio_encoder_checkpoint=self.audio_encoder_checkpoint,
            visual_encoder_checkpoint=self.visual_encoder_checkpoint,
            sync_model_checkpoint=self.sync_model_checkpoint,
        )

    def predict_sync_lab(
        self,
        audio_path: str | Path,
        landmarks_path: str | Path,
        shift_seconds: float,
        sample_id: str = "",
    ) -> SyncLabResult:
        """Run a controlled temporal-shift synchronization experiment.

        This powers the Synchronization Lab. It reuses the same frozen audio/visual
        encoders, the same trained cross-attention, and the same trained sync head as
        `predict_audio_visual` — no weights are modified or retrained. The only
        difference is the audio-to-video temporal correspondence: instead of the
        zero-shift alignment used at normal inference time, it is recomputed with an
        explicit offset using `compute_shifted_alignment`, the same deterministic
        shifting utility Phase 12 uses to construct negative (misaligned) training
        pairs.

        Args:
            audio_path: Path to a precomputed audio file for a known sample.
            landmarks_path: Path to a precomputed MediaPipe landmarks .npz file
                (points/fps/valid) for the same sample.
            shift_seconds: Audio offset in seconds (positive = audio delayed
                relative to video). Only non-negative shifts are guaranteed to
                produce valid overlapping windows with this alignment scheme.
            sample_id: Optional identifier echoed back in the result for the caller.

        Returns:
            SyncLabResult with the aggregate and per-window sync scores for the
            requested shift.

        Raises:
            FileNotFoundError: If the audio or landmarks file is not found.
        """
        from src.data.sync_pairs import compute_shifted_alignment

        audio_path = Path(audio_path)
        landmarks_path = Path(landmarks_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not landmarks_path.is_file():
            raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")

        with torch.inference_mode():
            waveform = preprocess_audio(audio_path, self.audio_config, source_sr=None, device="cpu")
            waveform = waveform.to(self.device)

            data = np.load(landmarks_path)
            landmarks_np = np.asarray(data["points"], dtype=np.float32)  # [T, N, 3]
            fps = float(data["fps"])
            valid = np.asarray(data.get("valid", np.ones(len(landmarks_np), dtype=bool)), dtype=bool)

            landmarks_np = interpolate_invalid(landmarks_np, valid)
            landmarks_np = normalize_landmarks(
                landmarks_np,
                method=self.video_config.normalize,
                align_rotation=self.video_config.align_rotation,
                coords=self._landmark_coords,
            )
            landmarks_np = landmarks_np[:, self._region_indices, :]
            landmarks = torch.from_numpy(landmarks_np).float().to(self.device)

            mel = compute_log_mel(waveform, self.audio_config, device=self.device)
            mel = mel.unsqueeze(0)  # [1, n_mels, T]
            landmarks = landmarks.unsqueeze(0)  # [1, T, N, 3]

            audio_out = self.audio_encoder(mel)
            audio_tokens = audio_out.tokens  # [1, T_a, D]

            visual_out = self.visual_encoder(landmarks)
            visual_tokens = visual_out.tokens  # [1, T_v, D]

            T_v = visual_tokens.shape[1]
            window_seconds = T_v / fps

            audio_aligned, _ = compute_shifted_alignment(
                audio_tokens[0],  # [T_a, D]
                n_video_tokens=T_v,
                video_fps=fps,
                shift_seconds=shift_seconds,
                audio_token_seconds=self.audio_token_seconds,
                window_seconds=window_seconds,
                empty_bucket="nearest",
            )
            audio_aligned = audio_aligned.unsqueeze(0)  # [1, T_v, D]

            fused_out = self.cross_attention(audio_aligned, visual_tokens)
            fused = fused_out.fused  # [1, T_v, D]

            logits = self.sync_head(fused).squeeze(0)  # [T_v]
            probs = torch.sigmoid(logits)

            valid_mask = torch.from_numpy(valid).bool().to(self.device)
            if valid_mask.any():
                aggregate_score = probs[valid_mask].mean().item()
            else:
                aggregate_score = probs.mean().item()

            per_window_scores = probs.cpu().tolist()

            timing_metadata = {
                "fps": fps,
                "num_frames": T_v,
                "window_seconds": window_seconds,
                "audio_token_seconds": self.audio_token_seconds,
                "num_valid_landmark_frames": int(valid.sum()),
                "num_windows": len(per_window_scores),
                "shift_seconds": shift_seconds,
            }

        return SyncLabResult(
            mode="sync_lab",
            sample_id=sample_id,
            shift_seconds=shift_seconds,
            aggregate_sync_score=aggregate_score,
            per_window_sync_scores=per_window_scores,
            timing_metadata=timing_metadata,
        )
