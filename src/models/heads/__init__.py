"""Task heads: spoof classifier (Phase 4), sync head (Phase 11), contrastive adapters/projections (Phase 12)."""

from src.models.heads.spoof_head import SpoofHead, temporal_pool
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.heads.contrastive_head import AudioAdapter, VisualAdapter, AudioProjectionHead, VisualProjectionHead, ContrastiveHeadConfig

__all__ = ["SpoofHead", "temporal_pool", "SyncHead", "SyncHeadConfig", "AudioAdapter", "VisualAdapter", "AudioProjectionHead", "VisualProjectionHead", "ContrastiveHeadConfig"]
