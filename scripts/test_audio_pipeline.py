"""Phase 2A smoke test: synthetic WAV -> preprocess -> log-mel -> CUDA -> PNG.

Run from the repository root:

    python scripts/test_audio_pipeline.py

It generates a handful of synthetic signals, writes them as real WAV files under
``outputs/audio/wav/``, runs the full preprocessing + mel-spectrogram pipeline on
each (asserting shapes / dtype / sample rate / normalization), repeats one signal
end-to-end on CUDA when available, and saves per-signal and overview PNGs under
``outputs/audio/``. Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_audio_config
from src.features.mel_spectrogram import compute_log_mel
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.synthetic import SYNTHETIC_SIGNALS, write_wav

CONFIG_PATH = REPO_ROOT / "configs" / "audio.yaml"
OUT_DIR = REPO_ROOT / "outputs" / "audio"
WAV_DIR = OUT_DIR / "wav"

TOL = 1e-4


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plot_signal(name: str, waveform: torch.Tensor, mel: torch.Tensor, sr: int, path: Path) -> None:
    wav = waveform.squeeze().cpu().numpy()
    mel_np = mel.cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axes[0].plot((torch.arange(wav.shape[0]) / sr).numpy(), wav, linewidth=0.6)
    axes[0].set(title=f"{name} - waveform", xlabel="time (s)", ylabel="amplitude")
    axes[0].set_ylim(-1.05, 1.05)
    im = axes[1].imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    axes[1].set(title=f"{name} - log-mel ({mel_np.shape[0]} mels x {mel_np.shape[1]} frames)",
                xlabel="frame", ylabel="mel bin")
    fig.colorbar(im, ax=axes[1], label="dB")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_overview(mels: dict[str, torch.Tensor], path: Path) -> None:
    n = len(mels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (name, mel) in zip(axes, mels.items()):
        ax.imshow(mel.cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set(title=name, xlabel="frame")
        ax.set_yticks([])
    axes[0].set_ylabel("mel bin")
    fig.suptitle("SyncGuard Phase 2A - synthetic log-mel spectrograms")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    cfg = load_audio_config(CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_n = cfg.sample_rate  # 1.0 s of audio at the target rate
    expected_t = 1 + expected_n // cfg.mel.hop_length

    rows: list[tuple[str, int, str, str, str]] = []
    mels: dict[str, torch.Tensor] = {}

    for name, generator in SYNTHETIC_SIGNALS.items():
        samples, native_sr = generator()
        wav_path = write_wav(WAV_DIR / f"{name}.wav", samples, native_sr)

        waveform = preprocess_audio(wav_path, cfg)
        _check(waveform.dtype == torch.float32, f"{name}: dtype {waveform.dtype} != float32")
        _check(tuple(waveform.shape) == (1, expected_n),
               f"{name}: waveform shape {tuple(waveform.shape)} != (1, {expected_n})")

        peak = waveform.abs().max().item()
        if name == "silence":
            _check(peak == 0.0, f"{name}: expected all-zero, peak={peak}")
        elif cfg.normalize == "peak":
            _check(abs(peak - cfg.norm_target) < TOL,
                   f"{name}: peak {peak:.6f} != norm_target {cfg.norm_target}")

        mel = compute_log_mel(waveform, cfg)
        _check(tuple(mel.shape) == (cfg.mel.n_mels, expected_t),
               f"{name}: mel shape {tuple(mel.shape)} != ({cfg.mel.n_mels}, {expected_t})")
        _check(torch.isfinite(mel).all().item(), f"{name}: mel contains non-finite values")

        mels[name] = mel
        _plot_signal(name, waveform, mel, cfg.sample_rate, OUT_DIR / f"{name}.png")
        rows.append((name, native_sr, str(tuple(waveform.shape)), str(tuple(mel.shape)), "cpu"))

    _plot_overview(mels, OUT_DIR / "mel_overview.png")

    # End-to-end CUDA pass on one signal.
    cuda_line = "CUDA not available - skipped GPU pass"
    if torch.cuda.is_available():
        name = "chirp_80_4000hz"
        samples, native_sr = SYNTHETIC_SIGNALS[name]()
        waveform = preprocess_audio(
            (WAV_DIR / f"{name}.wav"), cfg, device="cuda"
        )
        _check(waveform.is_cuda, "CUDA pass: preprocessed waveform is not on CUDA")
        mel = compute_log_mel(waveform, cfg, device="cuda")
        _check(mel.is_cuda, "CUDA pass: mel spectrogram is not on CUDA")
        _check(tuple(mel.shape) == (cfg.mel.n_mels, expected_t), "CUDA pass: wrong mel shape")
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        cuda_line = (
            f"CUDA pass OK on {torch.cuda.get_device_name(0)} "
            f"| mel {tuple(mel.shape)} device={mel.device} "
            f"| memory_allocated={allocated_mb:.2f} MB"
        )
        rows.append((name + " (cuda)", native_sr, str(tuple(waveform.shape)),
                     str(tuple(mel.shape)), "cuda"))

    header = f"{'signal':<24}{'in_sr':>8}  {'waveform':>12}  {'log_mel':>14}  {'device':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name, in_sr, wav_shape, mel_shape, dev in rows:
        print(f"{name:<24}{in_sr:>8}  {wav_shape:>12}  {mel_shape:>14}  {dev:>7}")
    print()
    print(cuda_line)
    print(f"\nArtifacts written to: {OUT_DIR}")
    print("Phase 2A smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 2A smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
