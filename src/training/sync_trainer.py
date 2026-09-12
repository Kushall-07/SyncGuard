"""Trainer subclass for audio-visual synchronization (Phase 11/12).

Implements the training loop for the sync head, with support for:
- Frozen Phase 5 audio encoder and Phase 8 visual encoder
- Trainable Phase 10 bidirectional cross-attention
- Trainable Phase 11 sync head
- Trainable Phase 12 contrastive head (optional)
- Positive (aligned) and negative (time-shifted) pair construction
- Masked binary cross-entropy loss
- InfoNCE contrastive loss (optional)
- Video-level sync metrics
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.sync_pairs import SyncPairConfig, build_sync_pair
from src.evaluation.sync_metrics import compute_sync_metrics
from src.losses.sync_loss import SyncLoss
from src.losses.contrastive_loss import ContrastiveLoss, ContrastiveLossConfig
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.fusion.temporal_align import align_audio_to_video
from src.models.heads.sync_head import SyncHead
from src.models.heads.contrastive_head import AudioAdapter, VisualAdapter, AudioProjectionHead, VisualProjectionHead, ContrastiveHeadConfig
from src.training.trainer import Trainer

__all__ = ["SyncTrainer", "SyncModel"]


class SyncModel(nn.Module):
    """Wrapper model for audio-visual synchronization training (Phase 11/12).

    Contains all components: frozen encoders + trainable cross-attention + sync head + optional contrastive adapters/projections.
    The forward method implements the full pipeline for training.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        visual_encoder: nn.Module,
        cross_attention: BidirectionalCrossAttention,
        sync_head: SyncHead,
        audio_adapter: AudioAdapter | None = None,
        visual_adapter: VisualAdapter | None = None,
        audio_projection: AudioProjectionHead | None = None,
        visual_projection: VisualProjectionHead | None = None,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.cross_attention = cross_attention
        self.sync_head = sync_head
        self.audio_adapter = audio_adapter
        self.visual_adapter = visual_adapter
        self.audio_projection = audio_projection
        self.visual_projection = visual_projection

    def forward(
        self,
        mel: torch.Tensor,
        landmarks: torch.Tensor,
        fps: float | list[float],
        window_seconds: float | list[float],
        shift_seconds: float | list[float],
        audio_token_seconds: float = 0.08,
        return_contrastive: bool = False,
        use_adapters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the sync pipeline.

        Args:
            mel: [B, n_mels, T_mel]
            landmarks: [B, T, N_landmarks, 3]
            fps: float or [B]
            window_seconds: float or [B]
            shift_seconds: float or [B]
            audio_token_seconds: Duration of one audio token in seconds
            return_contrastive: If True, also return contrastive projections
            use_adapters: If True, use trainable adapters before cross-attention (lambda>0)

        Returns:
            If return_contrastive=False:
                logits: [B, T] sync logits
                targets: [B, T] sync targets
                mask: [B, T] valid positions mask
            If return_contrastive=True:
                logits: [B, T] sync logits
                targets: [B, T] sync targets
                mask: [B, T] valid positions mask
                audio_proj: [B, T, D] audio projections (L2-normalized, from adapted audio)
                visual_proj: [B, T, D] visual projections (L2-normalized, from adapted visual)
        """
        B = mel.shape[0]

        # Encode with encoders
        audio_out = self.audio_encoder(mel)
        audio_tokens = audio_out.tokens  # [B, T_a, D]
        visual_out = self.visual_encoder(landmarks)
        visual_tokens = visual_out.tokens  # [B, T_v, D]

        T_v = visual_tokens.shape[1]

        # Construct sync pairs
        audio_aligned_list = []
        targets_list = []
        mask_list = []

        for b in range(B):
            # Extract per-sample values (handle both list/tensor and scalar)
            if isinstance(fps, (list, tuple)):
                sample_fps = fps[b]
            elif torch.is_tensor(fps):
                sample_fps = fps[b].item() if fps.ndim > 0 else fps.item()
            else:
                sample_fps = fps

            if isinstance(window_seconds, (list, tuple)):
                sample_window_seconds = window_seconds[b]
            elif torch.is_tensor(window_seconds):
                sample_window_seconds = window_seconds[b].item() if window_seconds.ndim > 0 else window_seconds.item()
            else:
                sample_window_seconds = window_seconds

            if isinstance(shift_seconds, (list, tuple)):
                sample_shift_seconds = shift_seconds[b]
            elif torch.is_tensor(shift_seconds):
                sample_shift_seconds = shift_seconds[b].item() if shift_seconds.ndim > 0 else shift_seconds.item()
            else:
                sample_shift_seconds = shift_seconds

            # Phase 9 alignment for positive reference
            pos_aligned, pos_counts = align_audio_to_video(
                audio_tokens[b : b + 1],
                n_video_tokens=T_v,
                audio_token_seconds=audio_token_seconds,
                video_fps=sample_fps,
                window_seconds=sample_window_seconds,
            )

            # Build sync pair with shift
            pair = build_sync_pair(
                audio_tokens[b],
                visual_tokens[b],
                video_fps=sample_fps,
                shift_seconds=sample_shift_seconds,
                audio_token_seconds=audio_token_seconds,
                window_seconds=sample_window_seconds,
                positive_audio_aligned=pos_aligned[0],
                positive_bucket_counts=pos_counts[0],
            )

            audio_aligned_list.append(pair.audio_aligned)
            targets_list.append(pair.targets)
            mask_list.append(pair.mask)

        audio_aligned = torch.stack(audio_aligned_list)  # [B, T_v, D]
        targets = torch.stack(targets_list)  # [B, T_v]
        mask = torch.stack(mask_list)  # [B, T_v]

        # Phase 12: Apply trainable adapters if use_adapters=True (lambda>0)
        # For lambda=0, bypass adapters to preserve exact Phase 11 baseline
        if use_adapters and self.audio_adapter is not None and self.visual_adapter is not None:
            audio_for_ca = self.audio_adapter(audio_aligned)  # [B, T_v, D]
            visual_for_ca = self.visual_adapter(visual_tokens)  # [B, T_v, D]
        else:
            # Phase 11 baseline: use original representations
            audio_for_ca = audio_aligned
            visual_for_ca = visual_tokens

        # Phase 10 cross-attention
        fused_out = self.cross_attention(audio_for_ca, visual_for_ca)
        fused = fused_out.fused  # [B, T_v, D]

        # Phase 11 sync head
        logits = self.sync_head(fused)  # [B, T_v]

        # Phase 12 contrastive projections (optional)
        # Use adapted representations if adapters are active
        if return_contrastive and self.audio_projection is not None and self.visual_projection is not None:
            if use_adapters and self.audio_adapter is not None and self.visual_adapter is not None:
                # Project adapted representations
                audio_proj = self.audio_projection(audio_for_ca)  # [B, T_v, D_proj]
                visual_proj = self.visual_projection(visual_for_ca)  # [B, T_v, D_proj]
            else:
                # Project original representations (for lambda=0 baseline)
                audio_proj = self.audio_projection(audio_aligned)  # [B, T_v, D_proj]
                visual_proj = self.visual_projection(visual_tokens)  # [B, T_v, D_proj]
            return logits, targets, mask, audio_proj, visual_proj
        else:
            return logits, targets, mask


class SyncTrainer(Trainer):
    """Trainer for audio-visual synchronization (Phase 11/12).

    Manages the full training pipeline:
    1. Audio and visual encoding (frozen encoders)
    2. Temporal alignment (Phase 9)
    3. Positive/negative pair construction with temporal shifts
    4. Bidirectional cross-attention (Phase 10)
    5. Sync head prediction (Phase 11)
    6. Masked binary cross-entropy loss
    7. InfoNCE contrastive loss (Phase 12, optional)
    8. Video-level sync metrics
    """

    def __init__(
        self,
        model: SyncModel,
        sync_loss: SyncLoss,
        sync_pair_config: SyncPairConfig,
        contrastive_loss: ContrastiveLoss | None = None,
        lambda_contrastive: float = 0.0,
        *args,
        audio_token_seconds: float = 0.08,
        aggregation: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__(model, *args, **kwargs)

        # Loss functions
        self.sync_loss = sync_loss.to(self.device)
        self.contrastive_loss = contrastive_loss.to(self.device) if contrastive_loss is not None else None
        self.lambda_contrastive = lambda_contrastive

        # Configuration
        self.sync_pair_config = sync_pair_config
        self.audio_token_seconds = audio_token_seconds
        self.aggregation = aggregation

        # Freeze encoders
        self._freeze_encoders()

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Trainable parameters: %s", f"{trainable_params:,}")
        self.logger.info("Lambda contrastive: %s", self.lambda_contrastive)

    def _freeze_encoders(self) -> None:
        """Freeze audio and visual encoders for Phase 11/12 baseline."""
        self.model.audio_encoder.eval()
        self.model.visual_encoder.eval()

        for param in self.model.audio_encoder.parameters():
            param.requires_grad = False

        for param in self.model.visual_encoder.parameters():
            param.requires_grad = False

        self.logger.info("Frozen audio and visual encoders")

    def compute_loss(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compute sync loss and optional contrastive loss for a batch.

        Args:
            batch: Dictionary containing:
                - mel_window: [B, n_mels, T_mel]
                - landmarks: [B, T, N_landmarks, 3]
                - fps: float or [B]
                - window_seconds: float or [B]
                - shift_seconds: float or [B]
                - is_positive: bool or [B]

        Returns:
            Dictionary with loss and detached outputs for metrics
        """
        # Move inputs to device
        mel = batch["mel_window"].to(self.device, non_blocking=True)
        landmarks = batch["landmarks"].to(self.device, non_blocking=True)
        fps = batch["fps"]
        window_seconds = batch["window_seconds"]
        shift_seconds = batch["shift_seconds"]

        # Forward pass through model (with contrastive if lambda > 0)
        if self.lambda_contrastive > 0 and self.contrastive_loss is not None:
            # use_adapters=True for lambda>0 to allow contrastive gradients to influence cross-attention
            logits, targets, mask, audio_proj, visual_proj = self.model(
                mel, landmarks, fps, window_seconds, shift_seconds, self.audio_token_seconds, 
                return_contrastive=True, use_adapters=True
            )
            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(audio_proj, visual_proj, mask)
        else:
            logits, targets, mask = self.model(
                mel, landmarks, fps, window_seconds, shift_seconds, self.audio_token_seconds, return_contrastive=False
            )
            contrastive_loss = torch.tensor(0.0, device=self.device)

        # Compute masked sync loss
        sync_loss = self.sync_loss(logits, targets, mask)

        # Combined loss
        loss = sync_loss + self.lambda_contrastive * contrastive_loss

        return {
            "loss": loss,
            "sync_loss": sync_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "logits": logits.detach().float(),
            "targets": targets.detach(),
            "mask": mask.detach(),
        }

    def compute_metrics(self, step_outputs: list[Mapping[str, Any]]) -> dict[str, float]:
        """Compute sync metrics from validation outputs.

        Args:
            step_outputs: List of dictionaries from compute_loss

        Returns:
            Dictionary of metric names to values
        """
        if not step_outputs:
            return {}

        # Concatenate all outputs
        logits = torch.cat([o["logits"] for o in step_outputs]).cpu()
        targets = torch.cat([o["targets"] for o in step_outputs]).cpu()
        mask = torch.cat([o["mask"] for o in step_outputs]).cpu()

        # Compute sync metrics
        metrics = compute_sync_metrics(logits, targets, mask, aggregation=self.aggregation)

        result = {
            "sync_accuracy": metrics.accuracy,
            "sync_precision": metrics.precision,
            "sync_recall": metrics.recall,
            "sync_f1": metrics.f1,
            "sync_auc": metrics.auc if metrics.auc is not None else 0.0,
            "video_sync_accuracy": metrics.video_accuracy,
            "video_sync_auc": metrics.video_auc if metrics.video_auc is not None else 0.0,
        }

        # Add contrastive loss tracking if applicable
        if self.lambda_contrastive > 0:
            sync_losses = [o["sync_loss"] for o in step_outputs]
            contrastive_losses = [o["contrastive_loss"] for o in step_outputs]
            result["sync_loss"] = torch.stack(sync_losses).mean().item()
            result["contrastive_loss"] = torch.stack(contrastive_losses).mean().item()

        return result
