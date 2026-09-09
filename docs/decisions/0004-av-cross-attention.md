# 4. Audio-visual bidirectional cross-attention — track decisions (Phase 10)

- Status: **Accepted (module implemented and unit-tested).** Not trained; no sync
  head / sync loss / paired-AV data yet.
- Date: 2026-09-08
- Scope: the bidirectional cross-attention / fusion representation only.
- Related: `0003-av-temporal-alignment.md` (Phase 9 — the aligned inputs this
  consumes), `0002-video-deepfake-detection.md` (D13 — visual branch is
  auxiliary).

---

## D1 — Purpose

Given the Phase-9 outputs `audio_aligned [B, T, 256]` and
`visual_tokens [B, T, 256]` (already time-aligned, 1:1, same length `T`), model
the **relationship** between the two modalities and emit a single per-timestep
fused AV representation for the Phase-11 sync head.

Phase 10 is *only* the cross-attention/fusion module. It does **not** pool,
classify, produce a sync score, add shifted-audio negatives, or apply a
contrastive / InfoNCE loss — those are Phases 11/12. It is fully trainable;
Phase 9's frozen encoders and deterministic alignment are unchanged.

## D2 — Frozen architectural constraints (from the approved proposal)

- Consume Phase 9's `audio_aligned` and `visual_tokens` directly. Both are
  256-D → **no projection layers** are added.
- Genuinely bidirectional:
  - **audio queries visual**: `Q = audio_aligned`, `K,V = visual_tokens`
    → `audio_query_visual` (`a2v`).
  - **visual queries audio**: `Q = visual_tokens`, `K,V = audio_aligned`
    → `visual_query_audio` (`v2a`).
- Output sequence length stays `T` (temporal correspondence preserved).
- `torch.nn.MultiheadAttention(batch_first=True)` — no custom attention kernel.
- The module never classifies and never produces a learned scalar sync score.
- Gradients flow through the module and into its inputs; the module is trainable.

## D3 — Architecture

```
audio_aligned [B,T,256]        visual_tokens [B,T,256]
       │  (Q)      ┌────────────┤  (K,V)
       ▼           ▼            │
  CrossAttentionBlock  "audio_to_visual"   ──►  a2v [B,T,256]
       │  (K,V)     ▲            │  (Q)
       └────────────┘            ▼
                       CrossAttentionBlock "visual_to_audio"  ──►  v2a [B,T,256]

  fused = LayerNorm( Linear(512→256)( concat([a2v, v2a], dim=-1) ) )   [B,T,256]
```

Each directional stack (`src/models/common/cross_attention.py::CrossAttentionBlock`)
is `n_layers` pre-norm cross-attention layers followed by a trailing
`LayerNorm`. One layer, with a **query-stream residual** on both sub-layers:

```
kv  = LN_kv(context)
q   = q + Dropout( MHA( LN_q(q), kv, kv, key_padding_mask=…, need_weights=False ) )
q   = q + Dropout( FFN( LN_ff(q) ) )       FFN: Linear(256→1024) → GELU → Dropout → Linear(1024→256)
```

Defaults: `dim=256`, `num_heads=4` (`d/head = 64`), `n_layers=1` per direction,
`ff_dim=1024`, `dropout=0.1`. `dim % num_heads == 0` is asserted.

**No additional positional encoding** by default — the inputs already carry each
encoder's sinusoidal PE + 3 transformer layers, and the aligner preserves order.
`add_positional_encoding: true` (config knob) prepends a shared
`SinusoidalPositionalEncoding` as a future ablation.

`batch_first=True` everywhere — matches the whole codebase; Phase 9 outputs are
`[B, T, D]`; no transposes.

Optional `audio_key_padding_mask` / `visual_key_padding_mask` (`[B, T]` bool,
default `None`). A caller *may* pass `bucket_counts == 0` from
`AVEncoderOutput` to down-weight Phase-9 fallback buckets; the module does not
derive it (avoids coupling to Phase 9 internals and all-masked-row NaNs). If
used, the caller must leave ≥1 unmasked key per query row.

## D4 — Number of layers and heads

- **`n_layers = 1` per direction (default).** The two per-modality encoders
  already do 3 layers of within-modality temporal modelling; Phase 10's job is
  one clean bidirectional context exchange, and a shallow stack keeps `fused`
  close to the aligned inputs (what a per-timestep sync head wants). `n_layers=2`
  (each layer re-attends the *original* other modality) is a cheap, configurable
  ablation.
- **`num_heads = 4`**, consistent with every transformer in the project.

## D5 — LayerNorm / residual / FFN

Pre-norm (`LN` before MHA and before FFN), query-stream residual on both
sub-layers, no residual on the K/V context stream, trailing `LayerNorm` on each
directional output, and a `LayerNorm` after the fusion projection. GELU FFN
`256 → 1024 → 256`. Dropout `0.1` on the attention output and inside/after the
FFN. Mirrors `TemporalTransformerEncoder`.

## D6 — Fusion mechanism

**`concat_proj` (production default):** `fused = LayerNorm( Linear(512→256)(
concat([a2v, v2a], -1) ) )`.

| option | params (n_layers=1) | rationale |
|---|---|---|
| **concat_proj** | +131 K | **Chosen.** Lowest-assumption; **keeps both directional views available** so the projection (and Phase 11) can form the audio↔visual *comparison* a sync score needs. Output stays 256-D → Phase 11 stays lightweight. |
| `gated` (`g·a2v + (1−g)·v2a`, `g = σ(Linear(512→256)(concat))`) | +131 K | Configurable ablation. Per-dim soft selection collapses the two views into one → less comparison info reaches Phase 11. |
| `add` (`a2v + v2a`) | +0 | Configurable ablation. `a2v` ("visual seen through audio") and `v2a` ("audio seen through visual") have different semantics; addition discards which-is-which — exactly the agreement signal Phase 11 must read. Not recommended for production. |

`fusion` is a config value; `add` / `gated` exist only as ablation / test
variants.

## D7 — Trainability / gradients

`BidirectionalCrossAttention` parameters all `requires_grad=True`. Phase 9's
`AVEncoder` produces `audio_aligned` / `visual_tokens` under `torch.no_grad()`,
so they arrive as **non-grad** tensors; Phase 10 still trains normally (its
params get gradients; the graph simply stops at the frozen-encoder outputs).
Tested with `requires_grad_(True)` inputs (gradients reach both inputs and every
parameter) and with real `AVEncoderOutput` tensors (module params receive
gradients on frozen-encoder features).

## D8 — Output contract for Phase 11

`BidirectionalCrossAttentionOutput`:

| field | shape | meaning |
|---|---|---|
| `fused` | `[B, T, 256]` | per-timestep fused AV representation → Phase 11 sync head |
| `audio_query_visual` | `[B, T, 256]` | `a2v` (returned unfused for inspection / ablation) |
| `visual_query_audio` | `[B, T, 256]` | `v2a` |

Temporal structure is retained so Phase 11 can produce a per-window / per-timestep
sync score. Phase 10 introduces **no** learned scalar sync score (that is
Phase 11).

## D9 — Parameter count (measured)

| config | params |
|---|---|
| `n_layers=1`, `concat_proj` (**default**) | **1,713,408** |
| `n_layers=1`, `gated` | 1,713,408 |
| `n_layers=1`, `add` | 1,582,080 |
| `n_layers=2`, `concat_proj` | 3,293,952 |

Comparable to the audio (~2.6 M) and visual (~2.6 M) encoders; small in absolute
terms.

## D10 — GPU memory (T=32, D=256, RTX 3050 6 GB)

Attention is O(T²) = 1024 — negligible. Largest activation is the FFN hidden
`[B, 32, 1024]` (512 KB/sample fp32). Forward ≈ 2–3 MB/sample; with autograd
≈ 6–10 MB/sample → ~0.2–0.3 GB of Phase-10 activations at batch 32, well under
1 GB at batch 64. The 6 GB card is not a constraint.

## D11 — Compatibility with Phase 9

Directly compatible: same dtype (float32), device, length `T`, and `D = 256`;
`256 % 4 == 0`. `nn.MultiheadAttention(embed_dim=256, batch_first=True)` consumes
the tensors as-is — **no projection layers** (constraint 3). `bucket_counts` is
not required; it is available as an optional mask source.

## D12 — Risks / open issues

- **No locality bias.** Cross-attention can match content at any timestep; the
  encoders' absolute PE gives only a weak positional signal, and Phase 10 adds no
  relative-position bias or local window. Left out of Phase 10 by design; a
  relative bias / ±k-frame mask / frame-offset feature is the natural Phase-11
  follow-up.
- **Fusion could collapse to one branch.** `concat_proj` mitigates the semantic
  mismatch vs `add`, but the projection could still learn to ignore a branch.
  Tests assert both branches influence `fused`; Phase 11 can monitor.
- **`audio_aligned` fallback regions have duplicated tokens** (`bucket_counts ==
  0`), so the "visual queries audio" direction sees repeated keys there. Benign
  for attention; Phase 11 may down-weight sync scores where `bucket_counts == 0`.
- **Capacity vs "not another classifier".** At `n_layers=2` (~3.3 M params) the
  module has enough capacity to behave like an independent classifier rather than
  a relationship model. The module architecture cannot enforce relationship-only
  behaviour — that inductive bias comes from the Phase-11+ training objective.
  The default stays minimal (`n_layers=1`).
- **CUDA non-determinism** in `nn.MultiheadAttention` backward on some paths;
  reproducibility tests run on CPU.
- **`no_grad` on Phase-9 encoder outputs** blocks any future end-to-end
  fine-tuning through Phase 10 — that would require a Phase-9 change (out of
  scope).
- **Config placement.** `cross_attention:` lives under `av_align:` (one
  pipeline, one file); `AVAlignConfig` ignores it, so Phase 9 config behaviour is
  unchanged.

## D13 — Data limitation (unchanged from Phase 9)

Celeb-DF-v2 is video-only; there is no paired audio+video corpus. Phase 10's
module is implemented and unit-tested with synthetic tokens and a tiny
`AVEncoder`, but the cross-attention has **not** been, and cannot yet be,
exercised or validated on real paired AV data. No claim is made about real-data
AV performance. A paired-AV dataset (FakeAVCeleb / DFDC / LRS3 / VoxCeleb2 /
AVSpeech) remains a prerequisite for running Phases 9–12 end to end.

---

## Files (Phase 10)

- `src/models/common/cross_attention.py` — `CrossAttentionBlock` (modality-agnostic directional stack)
- `src/models/fusion/cross_attention.py` — `BidirectionalCrossAttention`, `BidirectionalCrossAttentionOutput`, `CrossAttentionConfig`
- `src/models/fusion/__init__.py` — exports (additive)
- `configs/av_align.yaml` — `av_align.cross_attention` sub-block (additive; does not affect `AVAlignConfig` / Phase 9)
- `tests/test_fusion_cross_attention.py`

Not touched: `src/models/fusion/temporal_align.py`, `src/models/fusion/av_encoder.py`,
the audio/visual encoders, `src/models/common/temporal_transformer.py`,
`src/models/heads/*`, `src/losses/*`, `Trainer`, manifests, exported checkpoints.
