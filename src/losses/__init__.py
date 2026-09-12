"""Loss functions: sync loss (Phase 11), contrastive loss (Phase 12)."""

from src.losses.sync_loss import SyncLoss, SyncLossConfig
from src.losses.contrastive_loss import ContrastiveLoss, ContrastiveLossConfig

__all__ = ["SyncLoss", "SyncLossConfig", "ContrastiveLoss", "ContrastiveLossConfig"]