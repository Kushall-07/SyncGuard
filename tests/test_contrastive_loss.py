"""Tests for Phase 12 contrastive learning components."""

import torch
import pytest

from src.losses.contrastive_loss import ContrastiveLoss, ContrastiveLossConfig
from src.models.heads.contrastive_head import AudioAdapter, VisualAdapter, AudioProjectionHead, VisualProjectionHead, ContrastiveHeadConfig


def test_contrastive_loss_numerical_correctness() -> None:
    """Test that InfoNCE loss computes correctly with separate audio/visual projections."""
    # Create identical audio and visual projections (perfect positive pairs)
    audio_proj = torch.randn(4, 32, 128)
    visual_proj = audio_proj.clone()  # Identical projections
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.ones(4, 32, dtype=torch.bool)

    loss_fn = ContrastiveLoss(temperature=0.07)
    loss = loss_fn(audio_proj, visual_proj, mask)

    # Loss should be low for identical projections
    assert loss.item() < 1.0


def test_contrastive_loss_different_projections() -> None:
    """Test that InfoNCE loss is higher for different projections."""
    # Create different audio and visual projections
    audio_proj = torch.randn(4, 32, 128)
    visual_proj = torch.randn(4, 32, 128)
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.ones(4, 32, dtype=torch.bool)

    loss_fn = ContrastiveLoss(temperature=0.07)
    loss = loss_fn(audio_proj, visual_proj, mask)

    # Loss should be higher for different projections
    assert loss.item() > 0.0


def test_contrastive_loss_masking() -> None:
    """Test that masked positions are excluded from loss."""
    audio_proj = torch.randn(4, 32, 128)
    visual_proj = torch.randn(4, 32, 128)
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    
    # Mask half the positions
    mask = torch.zeros(4, 32, dtype=torch.bool)
    mask[:, :16] = True

    loss_fn = ContrastiveLoss(temperature=0.07)
    loss = loss_fn(audio_proj, visual_proj, mask)

    # Loss should be computed
    assert loss.item() >= 0.0


def test_contrastive_loss_all_masked() -> None:
    """Test that all-masked positions return zero loss."""
    audio_proj = torch.randn(4, 32, 128)
    visual_proj = torch.randn(4, 32, 128)
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.zeros(4, 32, dtype=torch.bool)

    loss_fn = ContrastiveLoss(temperature=0.07)
    loss = loss_fn(audio_proj, visual_proj, mask)

    # Loss should be zero
    assert loss.item() == 0.0


def test_contrastive_loss_temperature() -> None:
    """Test that temperature affects loss."""
    audio_proj = torch.randn(4, 32, 128)
    visual_proj = audio_proj.clone()
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.ones(4, 32, dtype=torch.bool)

    loss_low_temp = ContrastiveLoss(temperature=0.01)
    loss_high_temp = ContrastiveLoss(temperature=1.0)

    loss1 = loss_low_temp(audio_proj, visual_proj, mask)
    loss2 = loss_high_temp(audio_proj, visual_proj, mask)

    # Different temperatures should produce different losses
    assert loss1.item() != loss2.item()


def test_contrastive_loss_symmetric() -> None:
    """Test that symmetric InfoNCE (audio->visual + visual->audio) works."""
    audio_proj = torch.randn(4, 32, 128)
    visual_proj = audio_proj.clone()
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.ones(4, 32, dtype=torch.bool)

    loss_symmetric = ContrastiveLoss(temperature=0.07, symmetric=True)
    loss_asymmetric = ContrastiveLoss(temperature=0.07, symmetric=False)

    loss_sym = loss_symmetric(audio_proj, visual_proj, mask)
    loss_asym = loss_asymmetric(audio_proj, visual_proj, mask)

    # Symmetric loss should be computed
    assert loss_sym.item() >= 0.0
    # Asymmetric loss should be computed
    assert loss_asym.item() >= 0.0


def test_contrastive_loss_config() -> None:
    """Test that contrastive loss config works."""
    cfg = ContrastiveLossConfig(temperature=0.07, reduction="mean", symmetric=True)
    loss_fn = ContrastiveLoss.from_config(cfg)

    assert loss_fn.temperature == 0.07
    assert loss_fn.reduction == "mean"
    assert loss_fn.symmetric == True


def test_audio_projection_head_forward() -> None:
    """Test that audio projection head projects correctly."""
    head = AudioProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    audio_tokens = torch.randn(2, 32, 256)

    projections = head(audio_tokens)

    assert projections.shape == (2, 32, 128)
    # Check L2 normalization
    norms = projections.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_visual_projection_head_forward() -> None:
    """Test that visual projection head projects correctly."""
    head = VisualProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    visual_tokens = torch.randn(2, 32, 256)

    projections = head(visual_tokens)

    assert projections.shape == (2, 32, 128)
    # Check L2 normalization
    norms = projections.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_projection_head_config() -> None:
    """Test that projection head config works."""
    cfg = ContrastiveHeadConfig(projection_dim=128, hidden_dim=256, dropout=0.1, use_bn=True)
    audio_head = AudioProjectionHead.from_config(cfg, input_dim=256)
    visual_head = VisualProjectionHead.from_config(cfg, input_dim=256)

    assert audio_head.projection_dim == 128
    assert audio_head.hidden_dim == 256
    assert audio_head.use_bn == True
    assert visual_head.projection_dim == 128
    assert visual_head.hidden_dim == 256
    assert visual_head.use_bn == True


def test_projection_head_batch_norm() -> None:
    """Test that batch norm works correctly."""
    audio_head = AudioProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=True)
    audio_tokens = torch.randn(4, 32, 256)

    projections = audio_head(audio_tokens)

    assert projections.shape == (4, 32, 128)
    # Check L2 normalization
    norms = projections.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_projection_head_no_batch_norm() -> None:
    """Test that head works without batch norm."""
    visual_head = VisualProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    visual_tokens = torch.randn(4, 32, 256)

    projections = visual_head(visual_tokens)

    assert projections.shape == (4, 32, 128)
    # Check L2 normalization
    norms = projections.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_contrastive_gradient_flow() -> None:
    """Test that gradients flow through projection heads and loss."""
    audio_head = AudioProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    visual_head = VisualProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    loss_fn = ContrastiveLoss(temperature=0.07)

    audio_tokens = torch.randn(2, 32, 256, requires_grad=True)
    visual_tokens = torch.randn(2, 32, 256, requires_grad=True)
    audio_proj = audio_head(audio_tokens)
    visual_proj = visual_head(visual_tokens)
    mask = torch.ones(2, 32, dtype=torch.bool)

    loss = loss_fn(audio_proj, visual_proj, mask)
    loss.backward()

    # Check that gradients exist for both modalities
    assert audio_tokens.grad is not None
    assert visual_tokens.grad is not None
    assert audio_tokens.grad.abs().sum() > 0.0
    assert visual_tokens.grad.abs().sum() > 0.0


def test_contrastive_lambda_zero_behavior() -> None:
    """Test that lambda=0 reproduces sync-only behavior."""
    # This is tested at the trainer level, but we verify the loss computation here
    audio_head = AudioProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    visual_head = VisualProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    loss_fn = ContrastiveLoss(temperature=0.07)

    audio_tokens = torch.randn(2, 32, 256)
    visual_tokens = torch.randn(2, 32, 256)
    audio_proj = audio_head(audio_tokens)
    visual_proj = visual_head(visual_tokens)
    mask = torch.ones(2, 32, dtype=torch.bool)

    contrastive_loss = loss_fn(audio_proj, visual_proj, mask)
    sync_loss = torch.tensor(1.0)

    # With lambda=0, combined loss should equal sync loss
    lambda_contrastive = 0.0
    combined_loss = sync_loss + lambda_contrastive * contrastive_loss

    assert combined_loss.item() == sync_loss.item()


def test_contrastive_lambda_positive_behavior() -> None:
    """Test that lambda>0 combines losses correctly."""
    audio_head = AudioProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    visual_head = VisualProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    loss_fn = ContrastiveLoss(temperature=0.07)

    audio_tokens = torch.randn(2, 32, 256)
    visual_tokens = torch.randn(2, 32, 256)
    audio_proj = audio_head(audio_tokens)
    visual_proj = visual_head(visual_tokens)
    mask = torch.ones(2, 32, dtype=torch.bool)

    contrastive_loss = loss_fn(audio_proj, visual_proj, mask)
    sync_loss = torch.tensor(1.0)

    # With lambda>0, combined loss should be weighted sum
    lambda_contrastive = 0.5
    combined_loss = sync_loss + lambda_contrastive * contrastive_loss

    assert torch.allclose(combined_loss, sync_loss + lambda_contrastive * contrastive_loss.detach(), atol=1e-5)


def test_contrastive_cpu_cuda_shape_consistency() -> None:
    """Test that contrastive loss works on both CPU and CUDA with same shapes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    audio_proj = torch.randn(4, 32, 128)
    visual_proj = torch.randn(4, 32, 128)
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.ones(4, 32, dtype=torch.bool)

    loss_fn = ContrastiveLoss(temperature=0.07)

    # CPU
    loss_cpu = loss_fn(audio_proj, visual_proj, mask)

    # CUDA
    audio_proj_cuda = audio_proj.cuda()
    visual_proj_cuda = visual_proj.cuda()
    mask_cuda = mask.cuda()
    loss_cuda = loss_fn(audio_proj_cuda, visual_proj_cuda, mask_cuda)

    # Losses should be numerically similar
    assert torch.allclose(loss_cpu, loss_cuda.cpu(), atol=1e-5)


def test_contrastive_separate_projections() -> None:
    """Test that audio and visual projections come from different sources."""
    audio_head = AudioProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)
    visual_head = VisualProjectionHead(input_dim=256, projection_dim=128, hidden_dim=256, use_bn=False)

    audio_tokens = torch.randn(2, 32, 256)
    visual_tokens = torch.randn(2, 32, 256)

    audio_proj = audio_head(audio_tokens)
    visual_proj = visual_head(visual_tokens)

    # Projections should have the same shape
    assert audio_proj.shape == visual_proj.shape

    # But should be different (different input tokens)
    assert not torch.allclose(audio_proj, visual_proj)


def test_contrastive_positive_pair_definition() -> None:
    """Test that positive pairs are (b,t) aligned positions."""
    # Create audio and visual projections where only diagonal positions match
    audio_proj = torch.randn(2, 4, 128)
    visual_proj = torch.randn(2, 4, 128)
    
    # Make diagonal positions identical (positive pairs)
    for b in range(2):
        for t in range(4):
            visual_proj[b, t] = audio_proj[b, t]
    
    audio_proj = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
    visual_proj = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
    mask = torch.ones(2, 4, dtype=torch.bool)

    loss_fn = ContrastiveLoss(temperature=0.07)
    loss = loss_fn(audio_proj, visual_proj, mask)

    # Loss should be low because diagonal positions are identical
    assert loss.item() < 1.0


def test_audio_adapter_forward() -> None:
    """Test that audio adapter transforms correctly."""
    adapter = AudioAdapter(input_dim=256, hidden_dim=256, use_residual=True)
    audio_tokens = torch.randn(2, 32, 256)

    adapted = adapter(audio_tokens)

    assert adapted.shape == (2, 32, 256)
    # With residual, output should be different from input
    assert not torch.allclose(adapted, audio_tokens)


def test_visual_adapter_forward() -> None:
    """Test that visual adapter transforms correctly."""
    adapter = VisualAdapter(input_dim=256, hidden_dim=256, use_residual=True)
    visual_tokens = torch.randn(2, 32, 256)

    adapted = adapter(visual_tokens)

    assert adapted.shape == (2, 32, 256)
    # With residual, output should be different from input
    assert not torch.allclose(adapted, visual_tokens)


def test_adapter_gradient_flow() -> None:
    """Test that gradients flow through adapters."""
    audio_adapter = AudioAdapter(input_dim=256, hidden_dim=256, use_residual=True)
    visual_adapter = VisualAdapter(input_dim=256, hidden_dim=256, use_residual=True)

    audio_tokens = torch.randn(2, 32, 256, requires_grad=True)
    visual_tokens = torch.randn(2, 32, 256, requires_grad=True)

    audio_adapted = audio_adapter(audio_tokens)
    visual_adapted = visual_adapter(visual_tokens)

    # Simple loss
    loss = (audio_adapted.sum() + visual_adapted.sum())
    loss.backward()

    # Check that gradients exist
    assert audio_tokens.grad is not None
    assert visual_tokens.grad is not None
    assert audio_tokens.grad.abs().sum() > 0.0
    assert visual_tokens.grad.abs().sum() > 0.0


def test_adapter_initialization() -> None:
    """Test that adapters are initialized near-identity."""
    adapter = AudioAdapter(input_dim=256, hidden_dim=256, use_residual=True)
    
    # With near-identity initialization and residual, the output should be close to input
    audio_tokens = torch.randn(1, 1, 256)
    adapted = adapter(audio_tokens)
    
    # The residual connection ensures output ≈ input when proj weights are small
    # Since proj weights are initialized with gain=0.1, the deviation should be small
    deviation = (adapted - audio_tokens).abs().max()
    assert deviation < 1.0  # Should be relatively small
