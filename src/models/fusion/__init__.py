"""Audio-visual fusion.

* Phase 9 - deterministic time-aware temporal-correspondence aligner + the
  frozen-encoder AV wrapper.
* Phase 10 - bidirectional audio-visual cross-attention producing a per-timestep
  fused AV representation.

The sync head (Phase 11) and the contrastive loss (Phase 12) are not here.
"""

from src.models.fusion.av_encoder import (
    AVAlignConfig,
    AVEncoder,
    AVEncoderOutput,
    load_av_align_config,
)
from src.models.fusion.cross_attention import (
    BidirectionalCrossAttention,
    BidirectionalCrossAttentionOutput,
    CrossAttentionConfig,
)
from src.models.fusion.temporal_align import AudioToVideoAligner, align_audio_to_video

__all__ = [
    "align_audio_to_video",
    "AudioToVideoAligner",
    "AVEncoder",
    "AVEncoderOutput",
    "AVAlignConfig",
    "load_av_align_config",
    "BidirectionalCrossAttention",
    "BidirectionalCrossAttentionOutput",
    "CrossAttentionConfig",
]
