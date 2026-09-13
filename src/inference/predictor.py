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

import torch
import yaml

from src.config import AudioConfig, VideoDataConfig
from src.features.mel_spectrogram import compute_log_mel
from src.models.audio.encoder import load_audio_encoder
from src.models.fusion.cross_attention import (
    BidirectionalCrossAttention,
    CrossAttentionConfig,
)
from src.models.fusion.temporal_align import align_audio_to_video
from src.models.heads.spoof_head import SpoofHead
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.video.visual_encoder import load_visual_encoder
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.landmarks import region_point_count
from src.preprocessing.lavdf import extract_audio_from_mp4, extract_landmarks_from_mp4

__all__ = [
    "SyncGuardPredictor",
    "AudioOnlyResult",
    "AudioVisualResult",
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

    IMPORTANT:
    - Encoders are frozen and run in eval() mode
    - Inference uses torch.inference_mode() for efficiency
    - Audio is preprocessed to 16 kHz mono with peak normalization
    - Video landmarks are expected to be MediaPipe 478-point face landmarks
    - Temporal alignment uses 0.01-second audio tokens (from Phase-5 encoder)
    """

    def __init__(
        self,
        *,
        audio_encoder_path: str | Path,
        visual_encoder_path: str | Path,
        sync_model_path: str | Path,
        sync_config_path: str | Path,
        spoof_head_checkpoint: str | Path | None = None,
        device: str | torch.device = "auto",
        audio_config: AudioConfig | None = None,
        video_config: VideoDataConfig | None = None,
    ) -> None:
        """Initialize the dual-mode predictor.

        Args:
            audio_encoder_path: Path to Phase-5 audio encoder checkpoint
            visual_encoder_path: Path to Phase-8 visual encoder checkpoint
            sync_model_path: Path to Phase-12 sync model checkpoint (best.pt)
            sync_config_path: Path to sync training config YAML (av_align_lambda01.yaml)
            spoof_head_checkpoint: Optional path to spoof head checkpoint for audio-only mode
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
            num_frames=32, regions="face_mouth"
        )

        # Load sync config
        with open(sync_config_path, "r", encoding="utf-8") as f:
            sync_config_dict = yaml.safe_load(f)

        # Load audio encoder
        print(f"Loading audio encoder from {audio_encoder_path}")
        self.audio_encoder, audio_payload = load_audio_encoder(
            audio_encoder_path, map_location=self.device
        )
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # Compute audio token seconds from payload
        hop_length = audio_payload["audio_cfg"]["mel"]["hop_length"]
        time_downsample = int(audio_payload.get("time_downsample", 1))
        sample_rate = audio_payload["audio_cfg"]["sample_rate"]
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

        # Load spoof head for audio-only mode
        self.spoof_head = None
        self.spoof_head_checkpoint = ""
        if spoof_head_checkpoint is not None:
            print(f"Loading spoof head from {spoof_head_checkpoint}")
            spoof_ckpt = torch.load(spoof_head_checkpoint, map_location=self.device, weights_only=False)
            
            # Get embedding dimension from audio encoder
            embed_dim = self.audio_encoder.output_dim
            
            # Create spoof head with default configuration
            self.spoof_head = SpoofHead(
                in_dim=embed_dim,
                hidden=128,
                n_classes=2,
                dropout=0.1,
                pooling="attentive",
            )
            self.spoof_head.to(self.device)
            self.spoof_head.eval()
            
            # Load weights - handle different checkpoint structures
            if "spoof_head" in spoof_ckpt:
                self.spoof_head.load_state_dict(spoof_ckpt["spoof_head"], strict=False)
            elif "head" in spoof_ckpt:
                self.spoof_head.load_state_dict(spoof_ckpt["head"], strict=False)
            else:
                # Try direct load
                self.spoof_head.load_state_dict(spoof_ckpt, strict=False)
            
            for param in self.spoof_head.parameters():
                param.requires_grad = False
            
            self.spoof_head_checkpoint = str(spoof_head_checkpoint)

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

            # Compute log-mel spectrogram
            mel = compute_log_mel(waveform, self.audio_config, device=self.device)

            # Add batch dimension
            mel = mel.unsqueeze(0)  # [1, n_mels, T]

            # Encode with audio encoder
            audio_out = self.audio_encoder(mel)
            audio_tokens = audio_out.tokens  # [1, T, D]

            # Pool and classify with spoof head
            logits = self.spoof_head(audio_tokens)  # [1, 2]
            probs = torch.softmax(logits, dim=-1)[0]  # [2]

            # Index 1 = bonafide (matching label convention)
            bonafide_prob = probs[1].item()
            spoof_prob = probs[0].item()

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
                waveform, sr = preprocess_audio(audio_path, self.audio_config, source_sr=None, device="cpu")
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
                import numpy as np

                data = np.load(landmarks_path)
                landmarks = torch.from_numpy(data["points"]).float()  # [T, N, 3]
                fps = float(data["fps"])
                valid = data.get("valid", np.ones(len(landmarks), dtype=bool))
            else:
                # Extract landmarks from video
                if landmarker is None:
                    raise ValueError(
                        "landmarker required when landmarks_path not provided. "
                        "Initialize MediaPipe FaceLandmarker and pass to this method."
                    )
                lm_res = extract_landmarks_from_mp4(video_path, landmarker)
                landmarks = torch.from_numpy(lm_res["points"]).float()  # [T, N, 3]
                fps = lm_res["fps"]
                valid = lm_res["valid"]

            # Move landmarks to device
            landmarks = landmarks.to(self.device)

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
                audio_tokens[0],  # [T_a, D]
                n_video_tokens=T_v,
                audio_token_seconds=self.audio_token_seconds,
                video_fps=fps,
                window_seconds=window_seconds,
                empty_bucket="nearest",
            )
            audio_aligned = audio_aligned.unsqueeze(0)  # [1, T_v, D]

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
