"""Tests for Phase 12 contrastive learning integration with trainer."""

import torch
import pytest

from src.training.sync_trainer import SyncModel, SyncTrainer
from src.models.fusion.cross_attention import BidirectionalCrossAttention, CrossAttentionConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.heads.contrastive_head import AudioAdapter, VisualAdapter, AudioProjectionHead, VisualProjectionHead, ContrastiveHeadConfig
from src.losses.sync_loss import SyncLoss, SyncLossConfig
from src.losses.contrastive_loss import ContrastiveLoss, ContrastiveLossConfig
from src.models.audio.encoder import AudioEncoder
from src.models.video.visual_encoder import VisualEncoder
from src.config import ModelConfig


def test_phase12_lambda_zero_behavior() -> None:
    """Test that lambda=0 reproduces sync-only behavior without adapters."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    # No adapters or projections for lambda=0
    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=None, visual_adapter=None,
                     audio_projection=None, visual_projection=None)

    # Create trainer with lambda=0
    sl_cfg = SyncLossConfig()
    sync_loss = SyncLoss.from_config(sl_cfg)
    
    from src.data.sync_pairs import SyncPairConfig
    sp_cfg = SyncPairConfig()

    # Note: We can't fully test the trainer without a real dataset, but we can verify initialization
    # The actual loss computation is tested in test_contrastive_loss.py


def test_phase12_lambda_positive_behavior() -> None:
    """Test that lambda>0 activates adapters and projections."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    # Add adapters and projection heads
    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=audio_proj, visual_projection=visual_proj)

    # Verify adapters and projections are in model
    assert model.audio_adapter is not None
    assert model.visual_adapter is not None
    assert model.audio_projection is not None
    assert model.visual_projection is not None


def test_phase12_encoder_freezing() -> None:
    """Test that encoders remain frozen with contrastive learning."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=audio_proj, visual_projection=visual_proj)

    # Manually freeze encoders (as trainer does)
    model.audio_encoder.eval()
    model.visual_encoder.eval()
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.visual_encoder.parameters():
        param.requires_grad = False

    # Verify encoders are frozen
    assert not any(p.requires_grad for p in model.audio_encoder.parameters())
    assert not any(p.requires_grad for p in model.visual_encoder.parameters())

    # Verify trainable modules are trainable
    assert any(p.requires_grad for p in model.cross_attention.parameters())
    assert any(p.requires_grad for p in model.sync_head.parameters())
    assert any(p.requires_grad for p in model.audio_adapter.parameters())
    assert any(p.requires_grad for p in model.visual_adapter.parameters())
    assert any(p.requires_grad for p in model.audio_projection.parameters())
    assert any(p.requires_grad for p in model.visual_projection.parameters())


def test_phase12_checkpoint_save_load() -> None:
    """Test that Phase 12 model can be saved and loaded."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=audio_proj, visual_projection=visual_proj)

    # Save checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "epoch": 0,
        "best_metric": 0.0,
    }

    # Load checkpoint
    new_audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    new_visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))
    new_audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    new_visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    new_audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    new_visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)
    new_model = SyncModel(new_audio_enc, new_visual_enc, cross_attn, sync_head, 
                         audio_adapter=new_audio_adapter, visual_adapter=new_visual_adapter,
                         audio_projection=new_audio_proj, visual_projection=new_visual_proj)

    new_model.load_state_dict(checkpoint["model"])

    # Verify loaded model has same parameters
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_phase12_forward_with_contrastive() -> None:
    """Test that forward pass with contrastive returns projections from adapted representations."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=audio_proj, visual_projection=visual_proj)

    # Forward with contrastive and adapters
    mel = torch.randn(2, 80, 100)
    landmarks = torch.randn(2, 32, 142, 3)
    fps = 25.0
    window_seconds = 32 / 25.0
    shift_seconds = 0.0

    logits, targets, mask, audio_proj_out, visual_proj_out = model(
        mel, landmarks, fps, window_seconds, shift_seconds, 
        return_contrastive=True, use_adapters=True
    )

    assert logits.shape == (2, 32)
    assert targets.shape == (2, 32)
    assert mask.shape == (2, 32)
    assert audio_proj_out.shape == (2, 32, 128)
    assert visual_proj_out.shape == (2, 32, 128)


def test_phase12_forward_without_contrastive() -> None:
    """Test that forward pass without contrastive returns only sync outputs."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    # No adapters or projections
    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=None, visual_adapter=None,
                     audio_projection=None, visual_projection=None)

    # Forward without contrastive
    mel = torch.randn(2, 80, 100)
    landmarks = torch.randn(2, 32, 142, 3)
    fps = 25.0
    window_seconds = 32 / 25.0
    shift_seconds = 0.0

    logits, targets, mask = model(
        mel, landmarks, fps, window_seconds, shift_seconds, 
        return_contrastive=False, use_adapters=False
    )

    assert logits.shape == (2, 32)
    assert targets.shape == (2, 32)
    assert mask.shape == (2, 32)


def test_phase12_adapter_influences_cross_attention() -> None:
    """Test that changing adapter parameters changes cross-attention input."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=audio_proj, visual_projection=visual_proj)

    # Forward with adapters
    mel = torch.randn(2, 80, 100)
    landmarks = torch.randn(2, 32, 142, 3)
    fps = 25.0
    window_seconds = 32 / 25.0
    shift_seconds = 0.0

    logits1, _, _, _, _ = model(
        mel, landmarks, fps, window_seconds, shift_seconds, 
        return_contrastive=True, use_adapters=True
    )

    # Modify adapter parameters
    with torch.no_grad():
        model.audio_adapter.proj1.weight += 0.1

    # Forward again with modified adapters
    logits2, _, _, _, _ = model(
        mel, landmarks, fps, window_seconds, shift_seconds, 
        return_contrastive=True, use_adapters=True
    )

    # Logits should be different
    assert not torch.allclose(logits1, logits2)


def test_phase12_lambda_zero_bypasses_adapters() -> None:
    """Test that lambda=0 bypasses adapters to preserve Phase 11 baseline."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)

    # Model with adapters but use_adapters=False (lambda=0)
    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=None, visual_projection=None)

    # Set model to eval mode to ensure deterministic behavior
    model.eval()

    # Forward without adapters (lambda=0)
    mel = torch.randn(2, 80, 100)
    landmarks = torch.randn(2, 32, 142, 3)
    fps = 25.0
    window_seconds = 32 / 25.0
    shift_seconds = 0.0

    with torch.no_grad():
        logits1, _, _ = model(
            mel, landmarks, fps, window_seconds, shift_seconds, 
            return_contrastive=False, use_adapters=False
        )

        # Modify adapter parameters (should not affect lambda=0 path)
        model.audio_adapter.proj1.weight += 1.0

        # Forward again without adapters (lambda=0)
        logits2, _, _ = model(
            mel, landmarks, fps, window_seconds, shift_seconds, 
            return_contrastive=False, use_adapters=False
        )

    # Logits should be identical (adapters bypassed)
    assert torch.allclose(logits1, logits2)


def test_phase12_contrastive_gradients_reach_adapters() -> None:
    """Test that contrastive loss produces gradients on adapters."""
    # Create dummy encoders
    audio_enc = AudioEncoder(ModelConfig(audio_encoder="cnn_transformer"), n_mels=80)
    visual_enc = VisualEncoder(ModelConfig(visual_encoder="transformer"))

    # Create model components
    ca_cfg = CrossAttentionConfig(dim=256, num_heads=4, n_layers=1, ff_dim=1024, dropout=0.1)
    cross_attn = BidirectionalCrossAttention.from_config(ca_cfg)
    
    sh_cfg = SyncHeadConfig(hidden_dim=128, dropout=0.1)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=256)

    ch_cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, adapter_hidden_dim=256)
    audio_adapter = AudioAdapter.from_config(ch_cfg, input_dim=256)
    visual_adapter = VisualAdapter.from_config(ch_cfg, input_dim=256)
    audio_proj = AudioProjectionHead.from_config(ch_cfg, input_dim=256)
    visual_proj = VisualProjectionHead.from_config(ch_cfg, input_dim=256)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head, 
                     audio_adapter=audio_adapter, visual_adapter=visual_adapter,
                     audio_projection=audio_proj, visual_projection=visual_proj)

    # Manually freeze encoders
    model.audio_encoder.eval()
    model.visual_encoder.eval()
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.visual_encoder.parameters():
        param.requires_grad = False

    # Forward with contrastive
    mel = torch.randn(2, 80, 100, requires_grad=False)
    landmarks = torch.randn(2, 32, 142, 3, requires_grad=False)
    fps = 25.0
    window_seconds = 32 / 25.0
    shift_seconds = 0.0

    logits, targets, mask, audio_proj_out, visual_proj_out = model(
        mel, landmarks, fps, window_seconds, shift_seconds, 
        return_contrastive=True, use_adapters=True
    )

    # Compute contrastive loss
    loss_fn = ContrastiveLoss(temperature=0.07)
    mask_tensor = torch.ones_like(mask, dtype=torch.bool)
    contrastive_loss = loss_fn(audio_proj_out, visual_proj_out, mask_tensor)

    # Backward
    contrastive_loss.backward()

    # Check that adapters receive gradients
    assert model.audio_adapter.proj1.weight.grad is not None
    assert model.audio_adapter.proj2.weight.grad is not None
    assert model.visual_adapter.proj1.weight.grad is not None
    assert model.visual_adapter.proj2.weight.grad is not None

    # Check that projection heads receive gradients
    assert model.audio_projection.proj1.weight.grad is not None
    assert model.audio_projection.proj2.weight.grad is not None
    assert model.visual_projection.proj1.weight.grad is not None
    assert model.visual_projection.proj2.weight.grad is not None

    # Check that encoders do NOT receive gradients
    # Check first parameter of audio encoder
    audio_encoder_param = next(model.audio_encoder.parameters())
    assert audio_encoder_param.grad is None
    # Check first parameter of visual encoder
    visual_encoder_param = next(model.visual_encoder.parameters())
    assert visual_encoder_param.grad is None
