"""Trainer for the PHYSICALLY-shift-trained audio-visual SyncHead.

This implements the corrected pipeline from the sync-detection fix (see
`docs/decisions/0006-physical-sync-training.md`):

    original / physically-shifted / cross-clip waveform
        -> mel spectrogram
        -> frozen AudioEncoder                  (audio_tokens)
        -> frozen VisualEncoder                  (visual_tokens)
        -> align_audio_to_video (single, UNSHIFTED token-space alignment -
           the physical shift already happened upstream, in the waveform, so
           no token-timeline reassignment is needed or performed here)
        -> BidirectionalCrossAttention (trainable)
        -> SyncHead (trainable)
        -> masked BCE loss against the dataset's SYNC/DESYNC label

This deliberately does NOT use `src.data.sync_pairs.build_sync_pair` /
`compute_shifted_alignment` (the original Phase 11 token-reassignment
mechanism) - see `src/training/sync_trainer.py`'s `SyncModel`, kept unmodified
for old-vs-new comparison. Because `src.data.lavdf_physical_sync_dataset`
already bakes the desired temporal relationship into the *waveform* the audio
encoder consumes, `PhysicalSyncModel.forward` only ever needs the ordinary,
always-zero-shift `align_audio_to_video` call - exactly the alignment
production inference (`SyncGuardPredictor._predict_audio_visual_windowed`)
performs per window.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.evaluation.sync_metrics import compute_sync_metrics
from src.losses.sync_loss import SyncLoss
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.fusion.temporal_align import align_audio_to_video
from src.models.heads.sync_head import SyncHead
from src.training.trainer import Trainer

__all__ = ["PhysicalSyncModel", "PhysicalSyncTrainer"]


def _per_sample_scalar(value: float | list[float] | Tensor, b: int) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[b])
    if torch.is_tensor(value):
        return float(value[b].item() if value.ndim > 0 else value.item())
    return float(value)


class PhysicalSyncModel(nn.Module):
    """Frozen encoders + trainable cross-attention + sync head, no token-shift path."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        visual_encoder: nn.Module,
        cross_attention: BidirectionalCrossAttention,
        sync_head: SyncHead,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.cross_attention = cross_attention
        self.sync_head = sync_head

    def forward(
        self,
        mel: Tensor,
        landmarks: Tensor,
        fps: float | list[float],
        window_seconds: float | list[float],
        audio_token_seconds: float,
    ) -> tuple[Tensor, Tensor]:
        """Returns (logits [B, T_v], valid_mask [B, T_v] bool)."""
        audio_tokens = self.audio_encoder(mel).tokens  # [B, T_a, D]
        visual_tokens = self.visual_encoder(landmarks).tokens  # [B, T_v, D]
        B = mel.shape[0]
        T_v = visual_tokens.shape[1]

        aligned_list: list[Tensor] = []
        mask_list: list[Tensor] = []
        for b in range(B):
            sample_fps = _per_sample_scalar(fps, b)
            sample_ws = _per_sample_scalar(window_seconds, b)
            aligned, counts = align_audio_to_video(
                audio_tokens[b : b + 1],
                n_video_tokens=T_v,
                audio_token_seconds=audio_token_seconds,
                video_fps=sample_fps,
                window_seconds=sample_ws,
            )
            aligned_list.append(aligned[0])
            mask_list.append(counts[0] > 0)

        audio_aligned = torch.stack(aligned_list)  # [B, T_v, D]
        valid_mask = torch.stack(mask_list)  # [B, T_v] bool

        fused = self.cross_attention(audio_aligned, visual_tokens).fused
        logits = self.sync_head(fused)  # [B, T_v]
        return logits, valid_mask


class PhysicalSyncTrainer(Trainer):
    """Trainer for `PhysicalSyncModel` against `LAVDFPhysicalSyncDataset` batches."""

    def __init__(
        self,
        model: PhysicalSyncModel,
        sync_loss: SyncLoss,
        *args,
        audio_token_seconds: float = 0.01,
        aggregation: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__(model, *args, **kwargs)
        self.sync_loss = sync_loss.to(self.device)
        self.audio_token_seconds = audio_token_seconds
        self.aggregation = aggregation

        self.model.audio_encoder.eval()
        self.model.visual_encoder.eval()
        for param in self.model.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.model.visual_encoder.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Trainable parameters: %s", f"{trainable:,}")

    def compute_loss(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        mel = batch["mel_window"].to(self.device, non_blocking=True)
        landmarks = batch["landmarks"].to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)  # [B] float 0/1

        logits, valid_mask = self.model(
            mel, landmarks, batch["fps"], batch["window_seconds"], self.audio_token_seconds
        )
        targets = label.unsqueeze(1).expand_as(logits)
        loss = self.sync_loss(logits, targets, valid_mask)

        return {
            "loss": loss,
            "logits": logits.detach().float(),
            "targets": targets.detach(),
            "mask": valid_mask.detach(),
            "negative_type": list(batch.get("negative_type", [])),
            "shift_seconds": list(batch.get("shift_seconds", [])),
        }

    def compute_metrics(self, step_outputs: list[Mapping[str, Any]]) -> dict[str, float]:
        if not step_outputs:
            return {}
        logits = torch.cat([o["logits"] for o in step_outputs]).cpu()
        targets = torch.cat([o["targets"] for o in step_outputs]).cpu()
        mask = torch.cat([o["mask"] for o in step_outputs]).cpu()
        metrics = compute_sync_metrics(logits, targets, mask, aggregation=self.aggregation)
        return {
            "sync_accuracy": metrics.accuracy,
            "sync_precision": metrics.precision,
            "sync_recall": metrics.recall,
            "sync_f1": metrics.f1,
            "sync_auc": metrics.auc if metrics.auc is not None else 0.0,
            "video_sync_accuracy": metrics.video_accuracy,
            "video_sync_auc": metrics.video_auc if metrics.video_auc is not None else 0.0,
        }
