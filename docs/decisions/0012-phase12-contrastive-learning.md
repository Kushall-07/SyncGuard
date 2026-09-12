# Phase 12: Contrastive Learning Decision Document

## Context

Phase 11 implemented a SyncHead with masked BCE loss for audio-visual synchronization, achieving perfect video-level AUC (1.0000) on the LAV-DF dev split. However, the model was trained on artificial shifted pairs (shift=0 → positive, shift≠0 → negative), which raises the question: did the model learn genuine temporal correspondence or merely memorize the artificial task?

Phase 12 adds an InfoNCE contrastive objective to encourage the model to learn better multimodal representations by pulling together aligned pairs and pushing apart misaligned pairs in the representation space.

## Why InfoNCE?

InfoNCE (Information Noise Contrastive Estimation) is a well-established contrastive learning objective that:

1. **Encourages representation learning:** By maximizing similarity between positive pairs and minimizing similarity with negatives, the model learns better embeddings that capture the underlying structure of the data.

2. **Complements the sync objective:** The sync head uses BCE loss on binary sync labels (aligned vs. shifted), while InfoNCE operates on the continuous representation space. The two objectives can work together: the sync head provides direct supervision for the temporal alignment task, while InfoNCE encourages the model to learn more discriminative multimodal representations.

3. **Standard in multimodal learning:** InfoNCE has been successfully used in audio-visual contrastive learning (e.g., CLIP, Audio-Visual CLIP variants).

## Why Contrastive Learning BEFORE Cross-Attention?

**Critical architectural decision:** Phase 12 contrastive learning operates on pre-cross-attention representations, NOT the fused representation.

**Rationale:**
- Using the fused representation for both audio and visual projections would make the InfoNCE objective trivial (both sides come from the same source)
- Pre-cross-contrastive learning encourages the encoders to produce aligned representations that are already synchronized before fusion
- This provides complementary learning signal to the cross-attention fusion, which learns to combine modalities
- The contrastive objective operates on the raw aligned audio and visual tokens, ensuring they learn to match at the representation level

## Why Trainable Modality Adapters?

**Critical issue with projection heads alone:** With frozen encoders, projection heads alone cannot influence the cross-attention representation. Gradients from contrastive loss would only update the projection heads, leaving the representations used by cross-attention unchanged. This would make λ>0 ineffective for improving the main SyncGuard model.

**Solution:** Add trainable modality adapters (AudioAdapter, VisualAdapter) before cross-attention:
- Adapters transform the pre-cross-attention representations: [B,T,D] -> [B,T,D]
- Adapter outputs are used for BOTH contrastive learning AND cross-attention
- This allows contrastive gradients to influence the cross-attention representation
- λ=0 bypasses adapters to preserve the exact Phase 11 baseline

**Architecture with adapters:**
```
Phase 5 frozen audio encoder -> audio tokens [B, T_a, D]
Phase 8 frozen visual encoder -> visual tokens [B, T_v, D]
Phase 9 temporal alignment -> aligned_audio [B, T_v, D]

For lambda>0:
  aligned_audio -> AudioAdapter -> adapted_audio [B, T_v, D]
  visual_tokens -> VisualAdapter -> adapted_visual [B, T_v, D]
  
  adapted_audio -> AudioProjectionHead -> audio_proj [B, T_v, D_proj]
  adapted_visual -> VisualProjectionHead -> visual_proj [B, T_v, D_proj]
  
  Positive: adapted_audio[b,t] <-> adapted_visual[b,t]
  Loss: symmetric InfoNCE
  
  adapted_audio + adapted_visual -> Phase 10 cross-attention -> fused [B, T_v, D]

For lambda=0:
  aligned_audio + visual_tokens -> Phase 10 cross-attention -> fused [B, T_v, D]
  (adapters bypassed, exact Phase 11 path)

Phase 11 sync head -> sync logits [B, T_v]
```

## Adapter Architecture

**AudioAdapter and VisualAdapter:**
- Architecture: Linear -> ReLU -> Linear -> Residual
- Input: 256-D (aligned audio or visual tokens)
- Hidden: 256-D (configurable via adapter_hidden_dim)
- Output: 256-D (same dimension as input)
- Residual connection: output = adapted + input (prevents large shifts)

**Initialization:**
- Near-identity initialization to avoid large representation shifts at the start of contrastive training
- Weights initialized with gain=0.1 (xavier_uniform)
- Biases initialized to zero
- Residual connection ensures output ≈ input when weights are small

**Why near-identity initialization:**
- Prevents the model from starting with a large representation shift
- Allows the adapters to learn useful transformations gradually
- Maintains stability when transitioning from λ=0 to λ>0

## Positive Definition

**Positive pairs:** For each valid temporal position (b, t), the positive pair is:
- `adapted_audio[b, t]` (from AudioAdapter, or aligned_audio if λ=0)
- `adapted_visual[b, t]` (from VisualAdapter, or visual_tokens if λ=0)

**Key properties:**
- The positive pair comes from the same (batch, time) position
- This ensures the model learns to match aligned audio and visual at the same temporal moment
- Invalid positions (no audio-video overlap) are excluded via masking
- Uses adapted representations when λ>0, original representations when λ=0

## Negative Definition

**Negative pairs:** All other positions in the flattened [B*T] batch serve as negatives (in-batch negatives). Specifically:

- For a given position (batch_idx, time_idx), the negatives are all other positions in the flattened [B*T] batch.
- This includes:
  - Same clip, different timesteps
  - Different clips, same timestep
  - Different clips, different timesteps

**Rationale for in-batch negatives:**
- Simple and efficient to implement
- Standard practice in InfoNCE-based contrastive learning
- Provides a diverse set of negatives without requiring explicit negative sampling
- Prevents the model from collapsing to a trivial solution (all projections identical)

## Symmetric InfoNCE

**Architecture:** Symmetric InfoNCE is used by default:
- Audio-to-visual: treat each audio query, match with positive visual at same (b,t)
- Visual-to-audio: treat each visual query, match with positive audio at same (b,t)
- Final loss: 0.5 * (L_audio_to_visual + L_visual_to_audio)

**Rationale:**
- Provides balanced learning signal for both modalities
- Encourages symmetry in the learned representations
- Standard practice in multimodal contrastive learning

## Temperature

**Default temperature: 0.07**

The temperature parameter τ controls the sharpness of the softmax in InfoNCE:

- Lower temperature (e.g., 0.01): Sharper softmax, more confident predictions, can lead to mode collapse
- Higher temperature (e.g., 0.5): Softer softmax, more uniform predictions, slower learning
- 0.07 is a commonly used value in contrastive learning (e.g., SimCLR, MoCo)

The temperature is configurable via `contrastive.temperature` in the YAML config.

## Projection Dimensions

**Default projection_dim: 128**

Separate projection heads are used for audio and visual:

**AudioProjectionHead:**
- Input: 256-D (adapted audio tokens)
- Hidden: 256-D (with batch norm, ReLU, dropout)
- Output: 128-D (L2-normalized)

**VisualProjectionHead:**
- Input: 256-D (adapted visual tokens)
- Hidden: 256-D (with batch norm, ReLU, dropout)
- Output: 128-D (L2-normalized)

**Rationale:**
- 128-D is a common embedding dimension for contrastive learning
- The hidden dimension (256) maintains capacity while providing a bottleneck
- Batch normalization stabilizes training
- L2 normalization ensures projections lie on the unit sphere, making cosine similarity meaningful
- Separate heads allow modality-specific projection learning

## Masking

**Masked positions are excluded from contrastive loss:**

- Only valid positions (where audio-video overlap exists) are included in the contrastive loss
- Invalid positions (no overlap) are masked out
- This prevents the model from learning from empty or corrupted temporal positions

**Implementation:**
- The mask from Phase 11 (based on `bucket_counts > 0`) is reused
- Only valid positions are included in the flattened batch for InfoNCE computation
- If all positions in a batch are invalid, the contrastive loss is zero

## Gradient Flow and Frozen Encoders

**Gradient flow for lambda>0:**
- Gradients from contrastive loss flow through projection heads AND adapters
- Adapters transform the representations before cross-attention
- Cross-attention receives adapted representations, so contrastive gradients can influence the fused representation
- Frozen encoders (Phase 5 audio, Phase 8 visual) do NOT receive gradients (requires_grad=False)
- Adapters, projection heads, cross-attention, and sync head are trainable

**Gradient flow for lambda=0:**
- Adapters are bypassed (use_adapters=False)
- Cross-attention receives original representations (aligned_audio, visual_tokens)
- Only sync loss is computed (no contrastive loss)
- Exact Phase 11 baseline behavior

**Implementation:**
- Encoders are frozen in `SyncTrainer._freeze_encoders()`
- Adapters and projection heads are initialized with requires_grad=True
- `use_adapters` flag controls whether adapters are applied (based on lambda_contrastive > 0)
- For lambda=0, `use_adapters=False` ensures exact Phase 11 path

**Why this design:**
- Keeps encoders frozen as per Phase 11 baseline
- Allows contrastive learning to improve the representations used by cross-attention (via adapters)
- Preserves exact Phase 11 baseline when lambda=0
- Prevents catastrophic forgetting of encoder weights
- Maintains consistency with Phase 11 design

## Lambda Configuration

**Default lambda_contrastive: 0.0**

The λ parameter controls the relative weight of the contrastive loss and whether adapters are used:

- λ = 0: Phase 11 baseline (sync loss only, adapters bypassed)
- λ > 0: Combined loss: L = L_sync + λ * L_contrastive (adapters active)

**Recommended ablation values:**
- λ = 0: Sync-only baseline (Phase 11)
- λ = 0.1: Weak contrastive regularization
- λ = 0.5: Balanced combination
- λ = 1.0: Equal weighting

The λ parameter is configurable via `training.lambda_contrastive` in the YAML config.

## Configuration Fields

**Added to configs/av_align.yaml:**

```yaml
contrastive:
  enabled: false  # Enable/disable contrastive learning
  projection_dim: 128  # Output projection dimension
  hidden_dim: 256  # Hidden layer dimension for projection heads
  adapter_hidden_dim: 256  # Hidden layer dimension for adapters
  dropout: 0.1  # Dropout rate
  use_bn: true  # Use batch normalization
  temperature: 0.07  # InfoNCE temperature
  reduction: mean  # Loss reduction (mean/sum/none)
  symmetric: true  # Use symmetric InfoNCE (audio->visual + visual->audio)

training:
  lambda_contrastive: 0.0  # Weight for contrastive loss (0 = sync-only, adapters bypassed)
```

## Limitations

1. **No independent validation of contrastive learning:** The effect of contrastive learning can only be determined by training with λ=0 and λ>0 and comparing the results. This requires running actual training experiments.

2. **In-batch negatives may not be optimal:** While in-batch negatives are simple and effective, some contrastive learning methods use memory banks or larger negative sets for better performance. This could be explored in future work.

3. **Frozen encoders:** The encoders remain frozen, so contrastive learning can only affect the adapters, projection heads, and cross-attention, not the encoder representations themselves. This is consistent with Phase 11 but may limit the potential benefit of contrastive learning.

4. **Adapter capacity:** The adapters have a bottleneck architecture (256 -> 256 -> 256) which may limit their representational capacity. More complex adapter architectures could be explored in future work.

5. **Temperature sensitivity:** The performance of InfoNCE can be sensitive to the temperature parameter. The default value (0.07) is a reasonable starting point, but optimal values may vary depending on the dataset and task.

6. **Adapter initialization:** While near-identity initialization helps with stability, it may slow down initial learning if the optimal transformation is far from identity. Alternative initialization strategies could be explored.

## Scientific Constraints

**Do not claim that contrastive learning improves performance until experiments are run.**

The λ=0 vs. λ>0 comparison requires actual training runs. This implementation provides the capability to run those experiments, but does not make any claims about which configuration is better.
