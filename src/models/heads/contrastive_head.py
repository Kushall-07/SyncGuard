"""Contrastive Projection Heads and Adapters for InfoNCE Loss (Phase 12).

Implements separate projection heads for audio and visual representations
for contrastive learning BEFORE cross-attention, plus trainable modality adapters.

The contrastive objective encourages the model to learn better multimodal
representations by pulling together aligned (positive) pairs and pushing apart
misaligned (negative) pairs in the representation space.

This implementation uses:
- Trainable AudioAdapter and VisualAdapter: [B,T,D] -> [B,T,D] transformations
- Separate projection heads for:
  - Aligned audio tokens from Phase 9 temporal alignment
  - Visual tokens from Phase 8 visual encoder

Positive pairs: aligned_audio[b,t] <-> visual_tokens[b,t] (same temporal position)
Negatives: all other positions in the batch (in-batch negatives)

The adapters are used to allow contrastive gradients to influence the cross-attention
representation while keeping encoders frozen. For lambda=0, adapters are bypassed
to preserve the exact Phase 11 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["ContrastiveHeadConfig", "AudioAdapter", "VisualAdapter", "AudioProjectionHead", "VisualProjectionHead"]


@dataclass(frozen=True)
class ContrastiveHeadConfig:
    """Configuration for the contrastive projection heads and adapters."""

    projection_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    use_bn: bool = True
    adapter_hidden_dim: int = 256  # Hidden dimension for adapters

    def __post_init__(self) -> None:
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.adapter_hidden_dim <= 0:
            raise ValueError("adapter_hidden_dim must be positive")


class AudioAdapter(nn.Module):
    """Trainable adapter for audio tokens before cross-attention (Phase 12).

    Projects aligned audio tokens [B, T, D] -> [B, T, D] with a bottleneck.
    Allows contrastive gradients to influence the cross-attention representation
    while keeping the encoder frozen.

    Architecture:
    - Linear projection: [B, T, D] -> [B, T, adapter_hidden_dim]
    - ReLU activation
    - Linear projection: [B, T, adapter_hidden_dim] -> [B, T, D]
    - Residual connection (optional, to prevent large shifts)

    Initialization: near-identity to avoid large representation shifts.
    """

    def __init__(
        self,
        input_dim: int = 256,
        *,
        hidden_dim: int = 256,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, input_dim)

        # Initialize near-identity: proj2 ~ 0, proj1 as identity projection
        nn.init.xavier_uniform_(self.proj1.weight, gain=0.1)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.1)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, audio_tokens: Tensor) -> Tensor:
        """Forward pass.

        Args:
            audio_tokens: Aligned audio tokens [B, T, D]

        Returns:
            Adapted audio tokens [B, T, D]
        """
        if audio_tokens.dim() != 3:
            raise ValueError(f"expected [B, T, D] input, got {tuple(audio_tokens.shape)}")
        if audio_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"input last dim must be {self.input_dim}, got {audio_tokens.shape[-1]}"
            )

        # [B, T, D] -> [B, T, hidden_dim]
        x = self.proj1(audio_tokens)
        x = torch.relu(x)

        # [B, T, hidden_dim] -> [B, T, D]
        adapted = self.proj2(x)

        # Residual connection (optional)
        if self.use_residual:
            adapted = adapted + audio_tokens

        return adapted

    @classmethod
    def from_config(cls, cfg: ContrastiveHeadConfig, input_dim: int = 256) -> "AudioAdapter":
        return cls(
            input_dim=input_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            use_residual=True,
        )


class VisualAdapter(nn.Module):
    """Trainable adapter for visual tokens before cross-attention (Phase 12).

    Projects visual tokens [B, T, D] -> [B, T, D] with a bottleneck.
    Allows contrastive gradients to influence the cross-attention representation
    while keeping the encoder frozen.

    Architecture:
    - Linear projection: [B, T, D] -> [B, T, adapter_hidden_dim]
    - ReLU activation
    - Linear projection: [B, T, adapter_hidden_dim] -> [B, T, D]
    - Residual connection (optional, to prevent large shifts)

    Initialization: near-identity to avoid large representation shifts.
    """

    def __init__(
        self,
        input_dim: int = 256,
        *,
        hidden_dim: int = 256,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, input_dim)

        # Initialize near-identity: proj2 ~ 0, proj1 as identity projection
        nn.init.xavier_uniform_(self.proj1.weight, gain=0.1)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.1)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        """Forward pass.

        Args:
            visual_tokens: Visual tokens [B, T, D]

        Returns:
            Adapted visual tokens [B, T, D]
        """
        if visual_tokens.dim() != 3:
            raise ValueError(f"expected [B, T, D] input, got {tuple(visual_tokens.shape)}")
        if visual_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"input last dim must be {self.input_dim}, got {visual_tokens.shape[-1]}"
            )

        # [B, T, D] -> [B, T, hidden_dim]
        x = self.proj1(visual_tokens)
        x = torch.relu(x)

        # [B, T, hidden_dim] -> [B, T, D]
        adapted = self.proj2(x)

        # Residual connection (optional)
        if self.use_residual:
            adapted = adapted + visual_tokens

        return adapted

    @classmethod
    def from_config(cls, cfg: ContrastiveHeadConfig, input_dim: int = 256) -> "VisualAdapter":
        return cls(
            input_dim=input_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            use_residual=True,
        )


class AudioProjectionHead(nn.Module):
    """Projection head for audio tokens in contrastive learning (InfoNCE).

    Projects aligned audio tokens [B, T, D] to a lower-dimensional
    embedding space [B, T, projection_dim] for contrastive learning.

    Architecture:
    - Linear projection: [B, T, D] -> [B, T, hidden_dim]
    - Batch normalization (optional)
    - ReLU activation
    - Dropout
    - Linear projection: [B, T, hidden_dim] -> [B, T, projection_dim]
    - L2 normalization (for InfoNCE)

    The head is applied per-timestep to preserve temporal resolution.
    """

    def __init__(
        self,
        input_dim: int = 256,
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.use_bn = use_bn

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, audio_tokens: Tensor) -> Tensor:
        """Forward pass.

        Args:
            audio_tokens: Aligned audio tokens [B, T, D]

        Returns:
            L2-normalized audio projections [B, T, projection_dim]
        """
        if audio_tokens.dim() != 3:
            raise ValueError(f"expected [B, T, D] input, got {tuple(audio_tokens.shape)}")
        if audio_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"input last dim must be {self.input_dim}, got {audio_tokens.shape[-1]}"
            )

        B, T, D = audio_tokens.shape

        # [B, T, D] -> [B, T, hidden_dim]
        x = self.proj1(audio_tokens)
        
        # Apply batch norm (reshaping for batch norm)
        if self.use_bn:
            x = x.view(B * T, self.hidden_dim)
            x = self.bn(x)
            x = x.view(B, T, self.hidden_dim)
        
        x = torch.relu(x)
        x = self.dropout(x)

        # [B, T, hidden_dim] -> [B, T, projection_dim]
        projections = self.proj2(x)

        # L2 normalize along the projection dimension
        projections = nn.functional.normalize(projections, p=2, dim=-1)

        return projections

    @classmethod
    def from_config(cls, cfg: ContrastiveHeadConfig, input_dim: int = 256) -> "AudioProjectionHead":
        return cls(
            input_dim=input_dim,
            projection_dim=cfg.projection_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_bn=cfg.use_bn,
        )


class VisualProjectionHead(nn.Module):
    """Projection head for visual tokens in contrastive learning (InfoNCE).

    Projects visual tokens [B, T, D] to a lower-dimensional
    embedding space [B, T, projection_dim] for contrastive learning.

    Architecture:
    - Linear projection: [B, T, D] -> [B, T, hidden_dim]
    - Batch normalization (optional)
    - ReLU activation
    - Dropout
    - Linear projection: [B, T, hidden_dim] -> [B, T, projection_dim]
    - L2 normalization (for InfoNCE)

    The head is applied per-timestep to preserve temporal resolution.
    """

    def __init__(
        self,
        input_dim: int = 256,
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.use_bn = use_bn

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        """Forward pass.

        Args:
            visual_tokens: Visual tokens [B, T, D]

        Returns:
            L2-normalized visual projections [B, T, projection_dim]
        """
        if visual_tokens.dim() != 3:
            raise ValueError(f"expected [B, T, D] input, got {tuple(visual_tokens.shape)}")
        if visual_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"input last dim must be {self.input_dim}, got {visual_tokens.shape[-1]}"
            )

        B, T, D = visual_tokens.shape

        # [B, T, D] -> [B, T, hidden_dim]
        x = self.proj1(visual_tokens)
        
        # Apply batch norm (reshaping for batch norm)
        if self.use_bn:
            x = x.view(B * T, self.hidden_dim)
            x = self.bn(x)
            x = x.view(B, T, self.hidden_dim)
        
        x = torch.relu(x)
        x = self.dropout(x)

        # [B, T, hidden_dim] -> [B, T, projection_dim]
        projections = self.proj2(x)

        # L2 normalize along the projection dimension
        projections = nn.functional.normalize(projections, p=2, dim=-1)

        return projections

    @classmethod
    def from_config(cls, cfg: ContrastiveHeadConfig, input_dim: int = 256) -> "VisualProjectionHead":
        return cls(
            input_dim=input_dim,
            projection_dim=cfg.projection_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_bn=cfg.use_bn,
        )
