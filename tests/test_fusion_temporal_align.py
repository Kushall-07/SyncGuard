"""Phase 9: time-aware temporal-correspondence alignment operator.

Every behavioural test compares :func:`align_audio_to_video` against a
hand-written brute-force interval-overlap reference (``_ref_align`` below). The
operator and the reference both run their interval arithmetic in float64 for the
comparison tests so overlap membership matches bit-for-bit at bucket boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.fusion.temporal_align import AudioToVideoAligner, align_audio_to_video


# --------------------------------------------------------------- brute-force ref

def _ref_align(audio: torch.Tensor, *, t_v: int, dt_a: float, fps, window_seconds=None,
               empty_bucket: str = "nearest", valid_len=None):
    A = audio.detach().cpu().double().numpy()
    B, T_a, D = A.shape
    fps_arr = np.asarray(fps, dtype=np.float64)
    fps_arr = np.full(B, float(fps_arr)) if fps_arr.ndim == 0 else fps_arr.astype(np.float64)
    if window_seconds is None:
        win = t_v / fps_arr
    else:
        ws = np.asarray(window_seconds, dtype=np.float64)
        win = np.full(B, float(ws)) if ws.ndim == 0 else ws.astype(np.float64)
    vlen = np.full(B, T_a) if valid_len is None else np.asarray(valid_len).reshape(-1)

    out = np.zeros((B, t_v, D), dtype=np.float64)
    counts = np.zeros((B, t_v), dtype=np.int64)
    for b in range(B):
        w, f, n = win[b], fps_arr[b], int(vlen[b])
        a_iv = [(min(j * dt_a, w), min((j + 1) * dt_a, w)) for j in range(T_a)]
        a_ctr = [(j + 0.5) * dt_a for j in range(T_a)]
        for k in range(t_v):
            klo, khi = min(k / f, w), min((k + 1) / f, w)
            members = [j for j in range(n) if a_iv[j][1] > klo and a_iv[j][0] < khi]
            if members:
                out[b, k] = A[b, members].mean(axis=0)
                counts[b, k] = len(members)
            elif empty_bucket == "zero" or n == 0:
                pass  # zeros
            else:
                kctr = (k + 0.5) / f
                j_near = min(range(n), key=lambda j: (abs(a_ctr[j] - kctr), j))
                out[b, k] = A[b, j_near]
    return out, counts


def _check(audio, *, t_v, dt_a, fps, atol=1e-9, **kw):
    ref_out, ref_counts = _ref_align(audio, t_v=t_v, dt_a=dt_a, fps=fps, **kw)
    got_out, got_counts = align_audio_to_video(
        audio.double(), n_video_tokens=t_v, audio_token_seconds=dt_a, video_fps=fps, **kw
    )
    assert not torch.isnan(got_out).any()
    assert got_counts.dtype == torch.int64
    np.testing.assert_allclose(got_out.numpy(), ref_out, atol=atol)
    np.testing.assert_array_equal(got_counts.numpy(), ref_counts)
    return got_out, got_counts


def _rand(b, t_a, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t_a, d, generator=g, dtype=torch.float64)


# --------------------------------------------------------------------- the cases

def test_audio_coarser_than_video():
    # 12.5 Hz audio (0.08 s tokens) vs 30 fps video (0.033 s buckets): each audio
    # token interval is wider than a bucket, so it feeds several buckets and the
    # window is fully covered -> genuine averaging (1-2 tokens/bucket), no fallback.
    audio = _rand(2, 14, seed=1)                                # spans [0, 1.12) s
    out, counts = _check(audio, t_v=32, dt_a=0.08, fps=30.0)    # window 32/30 = 1.067 s
    assert (counts >= 1).all()
    assert counts.max() >= 2                                    # buckets straddling two audio tokens
    assert out.shape == (2, 32, 4)


def test_empty_buckets_use_nearest_audio_token_fallback():
    # audio runs out before the window ends -> trailing buckets have no overlap
    audio = _rand(2, 8, seed=20)                                # spans [0, 0.64) s
    out, counts = _check(audio, t_v=32, dt_a=0.08, fps=30.0)    # window 1.067 s
    empty = counts == 0
    assert empty.any() and empty[:, -1].all()                  # tail buckets are fallback
    assert not torch.isnan(out).any()
    # every fallback bucket equals the nearest real audio token (here token 7)
    for b in range(2):
        for k in torch.nonzero(empty[b]).flatten().tolist():
            assert torch.allclose(out[b, k], audio[b, 7].double(), atol=1e-12)


def test_audio_finer_than_video():
    # 100 Hz audio (0.01 s) vs 30 fps -> every bucket holds >= 1 audio token
    audio = _rand(2, 64, seed=2)
    _, counts = _check(audio, t_v=16, dt_a=0.01, fps=30.0)
    assert (counts >= 1).all()


def test_exact_rate_1to1_mapping():
    # binary-exact rates: dt_a = 1/8, fps = 8  -> token j and bucket j coincide
    audio = _rand(3, 20, seed=3)
    out, counts = align_audio_to_video(audio, n_video_tokens=20, audio_token_seconds=0.125,
                                       video_fps=8.0)
    assert torch.allclose(out, audio, atol=1e-6)
    assert torch.equal(counts, torch.ones(3, 20, dtype=torch.int64))


def test_t_audio_equals_t_video_non_exact_rate():
    audio = _rand(2, 32, seed=4)
    out, _ = _check(audio, t_v=32, dt_a=0.08, fps=30.0)
    assert out.shape == (2, 32, 4)


def test_t_audio_equals_one():
    audio = _rand(2, 1, seed=5)
    out, counts = _check(audio, t_v=10, dt_a=0.08, fps=30.0)
    # only one token exists -> every bucket resolves to it (overlap or nearest)
    assert torch.allclose(out, audio.expand(2, 10, 4).double(), atol=1e-9)
    assert counts.max() <= 1


def test_non_integer_ratio():
    audio = _rand(2, 14, seed=6)
    _check(audio, t_v=32, dt_a=0.08, fps=29.97)


def test_per_item_fps_variation_and_batched_equals_loop():
    audio = _rand(3, 14, seed=7)
    fps = torch.tensor([30.0, 28.757, 25.0], dtype=torch.float64)
    batched, bc = _check(audio, t_v=32, dt_a=0.08, fps=fps)
    for b in range(3):
        one, _ = align_audio_to_video(audio[b:b + 1].double(), n_video_tokens=32,
                                      audio_token_seconds=0.08, video_fps=float(fps[b]))
        assert torch.allclose(batched[b:b + 1], one, atol=1e-9)


def test_explicit_window_seconds_differs_from_token_span():
    audio = _rand(2, 14, seed=8)                     # token span 14*0.08 = 1.12 s
    for ws in (1.5, 0.8, 1.0666667):                 # longer / shorter / ~T_v/fps
        out, _ = _check(audio, t_v=32, dt_a=0.08, fps=30.0, window_seconds=ws)
        assert not torch.isnan(out).any()


def test_deterministic_repeated_calls():
    audio = _rand(2, 14, seed=9)
    a = align_audio_to_video(audio, n_video_tokens=32, audio_token_seconds=0.08, video_fps=29.97)
    b = align_audio_to_video(audio, n_video_tokens=32, audio_token_seconds=0.08, video_fps=29.97)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_right_padded_audio_with_valid_len():
    real = _rand(2, 14, seed=10)
    pad = torch.full((2, 6, 4), 1e6, dtype=torch.float64)          # garbage padding
    padded = torch.cat([real, pad], dim=1)                          # [2, 20, 4]
    vlen = torch.tensor([14, 10])

    got, gc = align_audio_to_video(padded, n_video_tokens=32, audio_token_seconds=0.08,
                                   video_fps=30.0, audio_valid_len=vlen)
    assert not torch.isnan(got).any() and got.abs().max() < 1e5   # padding never leaked
    # each item must match aligning only its valid prefix
    for b, n in enumerate(vlen.tolist()):
        ref, rc = align_audio_to_video(real[b:b + 1, :n].double(), n_video_tokens=32,
                                       audio_token_seconds=0.08, video_fps=30.0)
        assert torch.allclose(got[b:b + 1], ref, atol=1e-9)
        assert torch.equal(gc[b:b + 1], rc)


def test_empty_bucket_zero_mode():
    audio = _rand(2, 8, seed=11)                                   # short -> trailing empty buckets
    out, counts = _check(audio, t_v=32, dt_a=0.08, fps=30.0, empty_bucket="zero")
    empty = counts == 0
    assert empty.any()
    assert (out[empty].abs() < 1e-12).all()                        # exact zeros where no overlap
    assert (counts[~empty] >= 1).all()


def test_gradient_propagation_through_audio_tokens():
    audio = torch.randn(1, 14, 4, dtype=torch.float64, requires_grad=True)
    for mode in ("nearest", "zero"):
        out, _ = align_audio_to_video(audio, n_video_tokens=32, audio_token_seconds=0.08,
                                      video_fps=29.97, empty_bucket=mode)
        g, = torch.autograd.grad(out.sum(), audio, retain_graph=True)
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().sum() > 0                                    # some token feeds the output


def test_diverges_from_adaptive_avg_pool1d():
    # realistic Phase-9 shape: 14 audio tokens -> 32 video buckets at 29.97 fps
    audio = _rand(1, 14, seed=12)
    aligned, counts = align_audio_to_video(audio, n_video_tokens=32, audio_token_seconds=0.08,
                                           video_fps=29.97)
    aap = F.adaptive_avg_pool1d(audio.transpose(1, 2), 32).transpose(1, 2)   # index-proportional
    assert not torch.allclose(aligned, aap, atol=1e-3)

    # and again with audio shorter than the window, where the two differ hardest:
    # our trailing buckets equal a real audio token; adaptive_avg_pool1d blends indices
    short = _rand(1, 8, seed=21)
    al2, c2 = align_audio_to_video(short, n_video_tokens=32, audio_token_seconds=0.08,
                                   video_fps=29.97)
    aap2 = F.adaptive_avg_pool1d(short.transpose(1, 2), 32).transpose(1, 2)
    empty = c2[0] == 0
    assert empty.any()
    for k in torch.nonzero(empty).flatten().tolist():
        assert (al2[0, k] - short[0]).abs().sum(-1).min() < 1e-9      # == some real token
    assert not torch.allclose(al2, aap2, atol=1e-3)


# ------------------------------------------------------------- wrapper + guards

def test_module_wrapper_matches_function_and_has_no_params():
    audio = _rand(2, 14, seed=13)
    mod = AudioToVideoAligner(audio_token_seconds=0.08, empty_bucket="nearest")
    assert list(mod.parameters()) == []
    m_out, m_c = mod(audio, n_video_tokens=32, video_fps=29.97)
    f_out, f_c = align_audio_to_video(audio, n_video_tokens=32, audio_token_seconds=0.08,
                                      video_fps=29.97)
    assert torch.equal(m_out, f_out) and torch.equal(m_c, f_c)


def test_input_validation():
    audio = _rand(1, 4, seed=14)
    with pytest.raises(ValueError):
        align_audio_to_video(audio, n_video_tokens=8, audio_token_seconds=0.08,
                             video_fps=30.0, empty_bucket="lerp")
    with pytest.raises(ValueError):
        align_audio_to_video(audio, n_video_tokens=8, audio_token_seconds=0.0, video_fps=30.0)
    with pytest.raises(ValueError):
        align_audio_to_video(audio, n_video_tokens=8, audio_token_seconds=0.08, video_fps=-1.0)
    with pytest.raises(ValueError):
        align_audio_to_video(audio, n_video_tokens=0, audio_token_seconds=0.08, video_fps=30.0)


def test_float32_path_shapes_and_no_nan():
    audio = torch.randn(2, 14, 8)                                  # float32
    out, counts = align_audio_to_video(audio, n_video_tokens=32, audio_token_seconds=0.08,
                                       video_fps=29.97)
    assert out.shape == (2, 32, 8) and out.dtype == torch.float32
    assert counts.shape == (2, 32) and not torch.isnan(out).any()
