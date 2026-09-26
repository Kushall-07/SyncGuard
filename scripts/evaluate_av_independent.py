#!/usr/bin/env python3
"""Independent audio-visual evaluation framework (Phase 13B research validation).

This script is deliberately NOT another controlled-shift experiment. It runs the
existing, frozen SyncGuard audio-visual pipeline (frozen audio/visual encoders +
trained cross-attention + trained Sync Head) over a set of clips at their NATIVE
audio-video timing (no artificial temporal shift is introduced). Its purpose is to
let a genuinely independent evaluation set be "plugged in" without silently
recomputing or reinterpreting what the model's score means.

WHY THIS SCRIPT EXISTS
-----------------------
The rest of the project's AV-synchronization evidence (see `evaluate_sync.py` /
`docs/experiments.md`) comes from a *controlled-shift* experiment: the same clip's
own audio is either left aligned (shift = 0, treated as the positive class) or
shifted by a fixed offset (treated as the negative class). That experiment shows
the Sync Head is sensitive to artificial temporal misalignment. It does NOT show
that the model has been evaluated against real, naturally occurring desynchronization,
because no shift is a real deepfake artifact.

This script investigates the other axis: for datasets that ship *content*
manipulation labels (real / audio-only fake / video-only fake / audio+video fake,
e.g. LAV-DF), what does the Sync Head's native-timing score distribution look like
per category? This is useful and honest, but it is explicitly NOT a synchronization
accuracy number, because manipulation labels are not synchronization ground truth.
See "LABEL SEMANTICS" below.

LABEL SEMANTICS (do not blur these)
------------------------------------
A. True synchronization ground truth: a clip is *actually* known to be temporally
   misaligned (e.g. by dataset construction or manual annotation). No dataset used
   here provides this. Native (as-shipped) audio-video pairs are assumed aligned
   only because they were captured/authored together, not because any dataset
   documents an explicit "is_synchronized" label.
B. Manipulation labels: whether content was altered (real / audio-only /
   video-only / audio+video), independent of whether the result is temporally
   misaligned. A manipulated clip can still be perfectly synchronized.
C. Controlled temporal shifts: an artificial offset introduced by this codebase
   for the experiment in `evaluate_sync.py`. Not present in this script.

This script only ever produces (B)-grouped descriptive statistics of the model's
native-timing sync score. It never converts a manipulation label into a
synchronization target, and it never reports AUC/EER of "sync" against a
manipulation label — that would silently redefine "fake = desynchronized", which
is exactly the conflation this project's decision records (see
`docs/decisions/0005-lavdf-dataset.md`, `test_no_fake_periods_as_sync_labels` in
`tests/test_evaluate_sync.py`) forbid.

If you plug in a dataset that DOES carry genuine synchronization ground truth
(e.g. a manually annotated "desync_start/desync_end" or "is_synchronized" field),
extend `LAVDFSyncSample`/`LAVDFSyncDataset` (`src/data/lavdf_dataset.py`) with that
field and add a metrics branch here that computes AUC/EER against it. Until then,
Section "Metrics produced" below stays descriptive-only by design, not by omission.

DATASET SCOPE OF THIS SCRIPT (current run)
-------------------------------------------
This script is dataset-agnostic in *format* (any manifest following the LAV-DF
schema documented in `docs/datasets.md` can be pointed at it via --manifest /
--video-dir / --audio-dir / --landmarks-dir / --split), but no genuinely different
data source (i.e. not LAV-DF) was available locally or was downloaded to exercise
it in this pass. See `docs/datasets.md` for the datasets that were investigated as
candidates (FakeAVCeleb, DFDC, KoDF) and why they were not downloaded automatically.

Metrics produced
-----------------
- Per-sample native-timing aggregate sync score (mean over valid windows).
- Per-category (REAL / AUDIO-ONLY / VIDEO-ONLY / AUDIO+VIDEO) descriptive
  statistics of that score: n, mean, median, std, min, max.
- NO AUC/EER against manipulation category (see LABEL SEMANTICS).
- A bar chart of mean native-timing sync score by category, if matplotlib is
  available (best-effort; the run does not fail if plotting is unavailable).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_dataset import build_lavdf_datasets
from src.data.sync_pairs import SyncPairConfig, build_sync_pair
from scripts.evaluate_sync import load_trained_model

CATEGORIES = ("REAL", "AUDIO-ONLY", "VIDEO-ONLY", "AUDIO+VIDEO")


def categorize(label_name: str, modify_audio: bool, modify_video: bool) -> str:
    """Map manifest manipulation flags to a category label.

    This is a manipulation-content category ONLY. It is never treated as a
    synchronization target.
    """
    if label_name == "real":
        return "REAL"
    if modify_audio and not modify_video:
        return "AUDIO-ONLY"
    if not modify_audio and modify_video:
        return "VIDEO-ONLY"
    if modify_audio and modify_video:
        return "AUDIO+VIDEO"
    return "UNKNOWN"


def native_timing_scores(
    model,
    dataset,
    device: torch.device,
    audio_token_seconds: float,
    batch_size: int = 32,
) -> list[dict]:
    """Run the frozen AV pipeline at native (shift=0) timing for every sample.

    Returns one record per sample: sample_id, category, aggregate_sync_score,
    n_valid_windows. No artificial shift is applied — this reuses the same
    shift=0 pair construction as the controlled-shift experiment's positive
    class, but is reported on its own here rather than mixed with shifted
    negatives.
    """
    model.eval()

    def collate_fn(batch):
        max_mel_len = max(item["mel_window"].shape[1] for item in batch)
        mel_windows = []
        for item in batch:
            mel = item["mel_window"]
            if mel.shape[1] < max_mel_len:
                mel = torch.nn.functional.pad(mel, (0, max_mel_len - mel.shape[1]))
            mel_windows.append(mel)
        return {
            "mel_window": torch.stack(mel_windows),
            "landmarks": torch.stack([item["landmarks"] for item in batch]),
            "fps": [item["fps"] for item in batch],
            "window_seconds": [item["window_seconds"] for item in batch],
            "sample_id": [item["sample_id"] for item in batch],
            "label_name": [item["label_name"] for item in batch],
            "modify_audio": [item["modify_audio"] for item in batch],
            "modify_video": [item["modify_video"] for item in batch],
        }

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    records: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel_window"].to(device)
            landmarks = batch["landmarks"].to(device)

            audio_tokens = model.audio_encoder(mel).tokens
            visual_tokens = model.visual_encoder(landmarks).tokens

            for b in range(mel.shape[0]):
                pair = build_sync_pair(
                    audio_tokens[b],
                    visual_tokens[b],
                    video_fps=batch["fps"][b],
                    shift_seconds=0.0,
                    audio_token_seconds=audio_token_seconds,
                    window_seconds=batch["window_seconds"][b],
                )
                fused = model.cross_attention(
                    pair.audio_aligned.unsqueeze(0), visual_tokens[b : b + 1]
                ).fused
                logits = model.sync_head(fused).squeeze(0)
                probs = torch.sigmoid(logits)
                valid = probs[pair.mask]

                records.append(
                    {
                        "sample_id": batch["sample_id"][b],
                        "category": categorize(
                            batch["label_name"][b],
                            batch["modify_audio"][b],
                            batch["modify_video"][b],
                        ),
                        "aggregate_sync_score": valid.mean().item() if valid.numel() > 0 else None,
                        "n_valid_windows": int(valid.numel()),
                        "n_total_windows": int(probs.numel()),
                    }
                )
    return records


def summarize_by_category(records: list[dict]) -> dict:
    """Descriptive statistics of native-timing sync score, grouped by manipulation
    category. This is NOT a classification metric — see module docstring.
    """
    summary: dict[str, dict] = {}
    for category in CATEGORIES:
        scores = [
            r["aggregate_sync_score"]
            for r in records
            if r["category"] == category and r["aggregate_sync_score"] is not None
        ]
        if not scores:
            summary[category] = {"n": 0, "note": "no samples with valid windows"}
            continue
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        summary[category] = {
            "n": len(scores),
            "mean_sync_score": mean,
            "median_sync_score": sorted(scores)[len(scores) // 2],
            "std_sync_score": variance**0.5,
            "min_sync_score": min(scores),
            "max_sync_score": max(scores),
        }
    return summary


def maybe_plot(summary: dict, output_dir: Path) -> str | None:
    """Best-effort bar chart of mean native-timing sync score by category."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    categories = [c for c in CATEGORIES if summary.get(c, {}).get("n", 0) > 0]
    if not categories:
        return None
    means = [summary[c]["mean_sync_score"] for c in categories]
    ns = [summary[c]["n"] for c in categories]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, means, color="#7c9478")
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean native-timing sync score")
    ax.set_title("Sync Head score by manipulation category (native timing, no shift)")
    fig.tight_layout()

    out_path = output_dir / "sync_score_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent AV evaluation: native-timing Sync Head scores by manipulation category."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained sync model checkpoint (best.pt)")
    parser.add_argument("--audio-encoder", required=True, help="Path to frozen audio encoder export")
    parser.add_argument("--visual-encoder", required=True, help="Path to frozen visual encoder export")
    parser.add_argument("--config", required=True, help="Path to sync training config YAML")
    parser.add_argument("--manifest", required=True, help="LAV-DF-schema manifest CSV")
    parser.add_argument("--video-dir", required=True, help="Directory containing extracted video files")
    parser.add_argument("--audio-dir", required=True, help="Directory containing extracted/processed audio")
    parser.add_argument("--landmarks-dir", required=True, help="Directory containing precomputed landmark .npz files")
    parser.add_argument(
        "--split",
        default="dev",
        help=(
            "Manifest split value to evaluate (e.g. 'dev'). Passing 'test' evaluates "
            "the held-out LAV-DF test split; this project's own decision record "
            "(docs/decisions/0005-lavdf-dataset.md) reserves that split, so only pass "
            "'test' deliberately, once, for a final reported result."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write results into")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model, audio_token_seconds = load_trained_model(
        args.checkpoint, args.audio_encoder, args.visual_encoder, args.config, device
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    video_dict = cfg.get("av_align", {}).get("video", {})
    audio_cfg = AudioConfig.from_dict(cfg.get("audio", {}))
    video_cfg = VideoDataConfig(
        num_frames=video_dict.get("num_frames", 32), regions=video_dict.get("regions", "face_mouth")
    )
    sp_cfg = SyncPairConfig(**cfg.get("av_align", {}).get("sync_pairs", {}))
    from dataclasses import replace

    sp_cfg = replace(sp_cfg, audio_token_seconds=audio_token_seconds)

    dataset = build_lavdf_datasets(
        manifest_path=args.manifest,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        landmarks_dir=args.landmarks_dir,
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=(args.split,),
        n_video_tokens=video_dict.get("num_frames", 32),
        sync_pair_config=sp_cfg,
        use_negative_pairs=False,
        random_sample=False,
        seed=42,
    )[args.split]

    print(f"Loaded {len(dataset)} samples from split={args.split!r}")

    records = native_timing_scores(model, dataset, device, audio_token_seconds, args.batch_size)
    summary = summarize_by_category(records)

    print("\n=== Native-timing sync score by manipulation category (descriptive only) ===")
    for category, stats in summary.items():
        if stats.get("n", 0) > 0:
            print(
                f"{category}: n={stats['n']} mean={stats['mean_sync_score']:.4f} "
                f"std={stats['std_sync_score']:.4f} min={stats['min_sync_score']:.4f} "
                f"max={stats['max_sync_score']:.4f}"
            )
        else:
            print(f"{category}: n=0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "evaluation_type": "independent_av_native_timing",
        "split": args.split,
        "manifest": str(args.manifest),
        "n_samples": len(records),
        "category_summary": summary,
        "warning": (
            "category_summary groups by MANIPULATION label, not synchronization "
            "ground truth. No dataset used here provides verified desynchronization "
            "labels; do not interpret these numbers as sync-detection accuracy."
        ),
        "per_sample": records,
    }
    results_path = output_dir / "independent_av_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "independent_av_per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "category", "aggregate_sync_score", "n_valid_windows", "n_total_windows"]
        )
        writer.writeheader()
        writer.writerows(records)

    plot_path = maybe_plot(summary, output_dir)

    print(f"\nResults saved to {results_path}")
    print(f"Per-sample CSV saved to {csv_path}")
    if plot_path:
        print(f"Chart saved to {plot_path}")
    else:
        print("Chart skipped (matplotlib not available or no data)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
