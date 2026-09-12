"""InfoNCE Contrastive Loss (Phase 12).

Implements the InfoNCE (Information Noise Contrastive Estimation) loss for
contrastive learning on audio-visual representations.

The loss encourages the model to pull together aligned (positive) pairs and
push apart misaligned (negative) pairs in the representation space.

This implementation uses symmetric InfoNCE:
- Audio-to-visual: treat each audio query, match with positive visual at same (b,t)
- Visual-to-audio: treat each visual query, match with positive audio at same (b,t)
- Final loss: 0.5 * (L_audio_to_visual + L_visual_to_audio)

Negatives are in-batch: all other positions in the flattened [B*T] batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["ContrastiveLossConfig", "ContrastiveLoss"]


@dataclass(frozen=True)
class ContrastiveLossConfig:
    """Configuration for InfoNCE contrastive loss."""

    temperature: float = 0.07
    reduction: Literal["none", "mean", "sum"] = "mean"
    symmetric: bool = True  # Use symmetric InfoNCE (audio->visual + visual->audio)

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of ('none', 'mean', 'sum'), got {self.reduction!r}")


class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE contrastive loss for audio-visual learning.

    The loss is computed as:
    L_audio_to_visual = -log(exp(sim(audio_i, visual_i) / tau) / sum_k exp(sim(audio_i, visual_k) / tau))
    L_visual_to_audio = -log(exp(sim(visual_i, audio_i) / tau) / sum_k exp(sim(visual_i, audio_k) / tau))
    L = 0.5 * (L_audio_to_visual + L_visual_to_audio)

    where:
    - audio_i, visual_i are positive pair projections (same (b,t) position)
    - visual_k, audio_k are all projections in the batch (including the positive)
    - sim is cosine similarity
    - tau is the temperature parameter

    In this implementation:
    - Positive pairs: aligned_audio[b,t] <-> visual_tokens[b,t] (same temporal position)
    - Negatives: all other positions in the batch (in-batch negatives)
    - Masked positions are excluded from loss computation
    - Symmetric InfoNCE is used by default
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: Literal["none", "mean", "sum"] = "mean",
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.symmetric = symmetric

    def _compute_directional_loss(
        self,
        query_proj: Tensor,
        key_proj: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute directional InfoNCE loss (query -> key).

        Args:
            query_proj: Query projections [B, T, D] (L2-normalized)
            key_proj: Key projections [B, T, D] (L2-normalized)
            mask: Valid positions mask [B, T] (True = valid)

        Returns:
            Scalar loss
        """
        if query_proj.shape != key_proj.shape:
            raise ValueError(
                f"query_proj and key_proj must have same shape, "
                f"got {tuple(query_proj.shape)} and {tuple(key_proj.shape)}"
            )
        if query_proj.dim() != 3:
            raise ValueError(f"expected [B, T, D] projections, got {tuple(query_proj.shape)}")
        if mask.shape != query_proj.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match projection batch/time dims {query_proj.shape[:2]}"
            )

        B, T, D = query_proj.shape

        # Flatten to [B*T, D] for batch-wise computation
        query_flat = query_proj.view(B * T, D)  # [B*T, D]
        key_flat = key_proj.view(B * T, D)  # [B*T, D]
        mask_flat = mask.view(B * T)  # [B*T]

        # Only compute loss for valid positions
        valid_indices = mask_flat.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            # No valid positions, return zero loss
            return torch.tensor(0.0, device=query_proj.device, dtype=query_proj.dtype)

        query_valid = query_flat[valid_indices]  # [N_valid, D]
        key_valid = key_flat[valid_indices]  # [N_valid, D]

        # Compute similarity matrix [N_valid, N_valid]
        # sim[i, j] = cosine_similarity(query_valid[i], key_valid[j])
        similarity = torch.mm(query_valid, key_valid.t())  # [N_valid, N_valid]

        # Scale by temperature
        similarity = similarity / self.temperature

        # Diagonal elements are positive pairs (same (b,t) position)
        # Off-diagonal elements are negatives (in-batch negatives)
        # This uses "all other samples in batch" as negatives, including
        # both same-clip different-time and different-clip samples.

        # Labels: diagonal indices
        labels = torch.arange(similarity.shape[0], device=similarity.device)

        # Compute cross-entropy loss
        # Note: We use negative log likelihood: -log(exp(sim_i,i) / sum_j exp(sim_i,j))
        loss = nn.functional.cross_entropy(similarity, labels, reduction="none")

        # Apply reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # else "none": keep per-sample loss

        return loss

    def forward(
        self,
        audio_proj: Tensor,
        visual_proj: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            audio_proj: Audio projections [B, T, D] (L2-normalized)
            visual_proj: Visual projections [B, T, D] (L2-normalized)
            mask: Valid positions mask [B, T] (True = valid)

        Returns:
            Scalar loss (or per-sample loss if reduction="none")
        """
        if self.symmetric:
            # Audio-to-visual: treat audio as query, visual as key
            loss_av = self._compute_directional_loss(audio_proj, visual_proj, mask)
            # Visual-to-audio: treat visual as query, audio as key
            loss_va = self._compute_directional_loss(visual_proj, audio_proj, mask)
            # Symmetric loss
            loss = 0.5 * (loss_av + loss_va)
        else:
            # Audio-to-visual only
            loss = self._compute_directional_loss(audio_proj, visual_proj, mask)

        return loss

    @classmethod
    def from_config(cls, cfg: ContrastiveLossConfig) -> "ContrastiveLoss":
        return cls(
            temperature=cfg.temperature,
            reduction=cfg.reduction,
            symmetric=cfg.symmetric,
        )
