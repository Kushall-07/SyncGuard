# 3. Audio-visual temporal alignment — track decisions (Phase 9)

- Status: **Accepted (operator + wrapper implemented and unit-tested).**
  Not validated on real paired audio+video — no such data exists yet (see D8).
- Date: 2026-09-08
- Scope: deterministic time-aware alignment of the shared audio encoder tokens
  onto the visual encoder's frame grid. No fusion model, no sync objective.
- Related: `0001-audio-spoof-detection.md` (Phase 5 shared audio encoder, D6),
  `0002-video-deepfake-detection.md` (Phase 8 visual encoder, D7/D13)

---

## D1 — Purpose

Phase 9 provides the **temporal correspondence layer** that later phases build on:
given continuous audio tokens and continuous visual tokens for the same clip
window, produce an audio representation that is index-aligned 1:1 with the visual
tokens, using a fixed, deterministic, parameter-free rule.

It deliberately stops there:

- **Phase 10** — cross-attention over the aligned streams.
- **Phase 11** — the sync head.
- **Phase 12** — the contrastive / sync objective.

None of those are implemented here.

## D2 — Frozen architectural constraints (from the approved review)

- The audio branch uses the **Phase-5 shared CNN+Transformer *representation*
  encoder** (`src.models.audio.encoder.AudioEncoder`, exported checkpoint,
  `variant == "cnn_transformer"`). It is **not** the Phase-4/5 score-level
  CNN+Transformer *ensemble*. `AVEncoder` raises if the checkpoint's variant is
  anything else.
- Audio encoder output stays continuous temporal tokens `[B, T_a, D_a]`.
- Visual encoder output stays continuous temporal tokens `[B, T_v, D_v]`.
- Audio and visual representations are **deterministically temporally aligned**.
- The alignment is **average-pooling audio tokens into video-rate buckets** — no
  learned alignment module, no parameters.
- Both encoders are **frozen** for Phase 9: `eval()` + `requires_grad_(False)`,
  and kept in `eval()` even when the parent `AVEncoder` is put in `train()`.
  Their architectures and checkpoint contents are not modified.

## D3 — Time-aware temporal-correspondence alignment (definition)

`src/models/fusion/temporal_align.py::align_audio_to_video`.

Let `dt_a = audio_token_seconds`, `fps` the per-clip video frame rate, `T_v` the
number of video tokens, `window_seconds` the wall-clock length of the aligned
window (default `T_v / fps`). All times are measured from the start of the
window and clipped to `[0, window_seconds)`.

- **Audio token `j`** occupies `[j·dt_a, (j+1)·dt_a)` seconds. For the Phase-5
  encoder `dt_a = hop_length · time_downsample / sample_rate = 160·8/16000 =
  0.080 s` (12.5 Hz), derived at load time from the checkpoint payload — never
  hard-coded.
- **Video bucket `k`** is `[k/fps, (k+1)/fps)` seconds.
- **Bucket value** = the mean of every audio token whose interval overlaps the
  bucket (half-open overlap: `a_hi > b_lo` and `a_lo < b_hi`).
- **Empty bucket** (no overlapping audio token — happens when the audio runs out
  before the window ends, or under `audio_valid_len` truncation): filled by the
  **nearest audio token by interval centre** (`empty_bucket="nearest"`, ties →
  lowest index) or by **zeros** (`empty_bucket="zero"`).
- **`bucket_counts[b, k]`** = number of audio tokens averaged into bucket `k`;
  `0` marks a fallback-filled bucket, so downstream code can mask or down-weight
  it.
- No NaNs are ever produced (an all-zero weight row yields a zero vector, not a
  divide-by-zero).

Internally the operator is the fixed linear map `aligned = W @ audio_tokens`,
where `W ∈ [B, T_v, T_a]` is built entirely from the timing scalars.

## D4 — Why `F.adaptive_avg_pool1d` (and interpolation) is rejected

`adaptive_avg_pool1d(audio, T_v)` maps by **index proportion**: output bucket `k`
is the mean of input indices `[floor(k·T_a/T_v), ceil((k+1)·T_a/T_v))`. It

- ignores the real 80 ms extent of each audio token and the `T_v/fps` extent of
  each video bucket;
- ignores the per-clip `fps` (28.76–30.0 Hz across Celeb-DF), so the mapping is
  wrong by a clip-dependent amount;
- ignores the `center=True` mel + CNN `ceil`-pool slack, where the last audio
  token covers a partial hop;
- when audio is shorter than the window, blends distant indices into the trailing
  buckets instead of holding the last real token.

A regression test asserts the two outputs differ for a realistic
`14 → 32, fps 29.97` case, and that our trailing (fallback) buckets equal an
actual audio token while `adaptive_avg_pool1d`'s do not.

This is **time-aware temporal correspondence alignment**, not interpolation and
not adaptive pooling.

## D5 — Tensor shapes / API

```
align_audio_to_video(
    audio_tokens: Tensor,                 # [B, T_a, D]
    *, n_video_tokens: int,               # T_v
    audio_token_seconds: float,           # e.g. 0.08
    video_fps: float | Tensor,            # scalar or [B]
    window_seconds: float | Tensor | None = None,   # default T_v / fps
    empty_bucket: str = "nearest",        # "nearest" | "zero"
    audio_valid_len: Tensor | None = None,          # [B]  (right-padded batches)
) -> (aligned_audio: [B, T_v, D],  bucket_counts: [B, T_v] int64)

AudioToVideoAligner(nn.Module)            # fixed audio_token_seconds / empty_bucket; no parameters
    .forward(audio_tokens, *, n_video_tokens, video_fps, window_seconds=None, audio_valid_len=None)
```

```
AVEncoder(audio_encoder_pt, visual_encoder_pt, *, empty_bucket="nearest", map_location="cpu")
    .forward(mel_window: [B, n_mels, T_mel],   # the encoder's own AudioConfig
             landmarks:  [B, T_v, N, C],
             video_fps:  float | [B],
             *, window_seconds=None, audio_valid_len=None) -> AVEncoderOutput

AVEncoderOutput(
    audio_tokens:  [B, T_a, 256],
    visual_tokens: [B, T_v, 256],
    audio_aligned: [B, T_v, 256],   # index-aligned to visual_tokens
    bucket_counts: [B, T_v] int64,
    video_fps:     [B] float32,
)
```

Reference numbers (Phase-5 audio encoder, dense Phase-8 video window):
`D_a = D_v = 256`; audio token 80 ms / 12.5 Hz; `T_a = ceil((1 + N//160)/8)` ≈
13–14 for a ~1.07 s window; `T_v = 32`; per-clip `window_seconds ≈ 32/fps ≈
1.067–1.113 s`.

## D6 — Differentiability

`W` depends only on `dt_a`, `fps`, `T_a`, `window_seconds` — constants w.r.t. the
token values — so `aligned = W @ audio_tokens` is differentiable w.r.t.
`audio_tokens` (mean, indexing, `one_hot` scatter; no `.detach()`, no
value-dependent branching). A test backpropagates through both `empty_bucket`
modes. Phase 12's objective can therefore flow gradients through the aligner,
even though the Phase-5/8 encoders themselves stay frozen.

Inside `AVEncoder` the two encoders are run under `torch.no_grad()` (they are
frozen; there is no reason to build their graph). A downstream Phase-10+ module
consuming `audio_aligned` / `visual_tokens` still trains normally — the gradient
simply stops at those tensors.

## D7 — Frozen-encoder policy

`AVEncoder.__init__` calls `load_audio_encoder` / `load_visual_encoder`, then for
each: `enc.eval(); enc.requires_grad_(False)`. `AVEncoder.train(mode)` re-asserts
`eval()` on both sub-encoders. `audio_token_seconds`, `n_mels`, `audio_dim`,
`visual_dim` and the `cnn_transformer` check are read from the export payloads.
Nothing is written back to the checkpoints.

## D8 — Known data limitation: Celeb-DF-v2 is video-only

Every Celeb-DF-v2 `.mp4` in the project (Celeb-real, Celeb-synthesis,
YouTube-real — verified across all three folders) has **no audio stream**. The
public Celeb-DF-v2 release ships video only. The Phase-5 audio encoder was
trained on ASVspoof (speech, no video). **There is no paired audio+video dataset
in the project.**

Consequences:

- The Phase 9 **operator and `AVEncoder` are fully implemented and unit-tested**
  against a brute-force interval-overlap reference, with synthetic tokens / tiny
  exported encoders. Their behaviour is specified and verified.
- The alignment has **not** been, and cannot currently be, **exercised on real
  paired AV data**. No claim is made about real-data AV performance or about
  empirical validation of the alignment on paired AV clips.
- `configs/av_align.yaml`'s `visual_encoder_pt` path is a placeholder — a Phase-8
  run must be re-run with `--export-encoder` to produce `visual_encoder.pt`
  (Phase 8's 3-seed runs did not export one).

## D9 — Future paired-AV dataset requirement

Before Phase 9 alignment (and Phases 10–12) can be run end-to-end, a corpus with
**synchronised audio and video of talking faces** must be added, e.g.
FakeAVCeleb, DFDC (has audio), LRS2 / LRS3, VoxCeleb2, or AVSpeech. That work
includes: an audio-extraction + windowed-log-mel path that crops the audio to the
exact video window `[frame_idx0/fps, (frame_idx0 + T_v)/fps)` using the encoder's
`AudioConfig`, and a paired dataset yielding `(mel_window, landmarks, fps,
label)`. `CelebDFLandmarkDataset.timing(index)` already exposes the per-clip
`fps` / `frame_idx0` / `window_seconds` needed for the crop.

---

## Files (Phase 9)

- `src/models/fusion/temporal_align.py` — `align_audio_to_video`, `AudioToVideoAligner`
- `src/models/fusion/av_encoder.py` — `AVEncoder`, `AVEncoderOutput`, `AVAlignConfig`, `load_av_align_config`
- `src/models/fusion/__init__.py` — exports
- `src/data/celebdf_dataset.py` — `CelebDFLandmarkDataset.timing(index)` (additive; `__getitem__` unchanged)
- `configs/av_align.yaml` — encoder paths + alignment / empty-bucket / target-grid policy (no training settings)
- `tests/test_fusion_temporal_align.py`, `tests/test_fusion_av_encoder.py`

Not touched: audio/visual encoder architectures and checkpoints, `Trainer`,
manifests, and anything belonging to Phases 10–12.
