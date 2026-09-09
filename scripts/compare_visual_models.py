"""Phase 7: head-to-head of the non-temporal landmark baseline vs the temporal
Transformer for video deepfake detection.

Trains ``model.visual_encoder = "mlp_baseline"`` and ``"transformer"`` on the
*same* manifest, split, seed and augmentation, then writes a comparison table.
The EER/AUC gap is what temporal modelling contributes (the visual analogue of
the Phase 5 CNN-vs-Transformer experiment).

    python scripts/compare_visual_models.py --synthetic --epochs 5

    # Phase 8 dense: 32 contiguous frames from the dense cache
    python scripts/compare_visual_models.py \
        --manifest data/celebdf/manifest_dense.csv \
        --landmarks-dir data/celebdf/landmarks_dense --num-frames 32

``--landmarks-dir`` / ``--num-frames`` override ``video.*`` from the config;
without them the run reads the 16-frame sparse cache and re-subsamples, so the
dense contiguous window never reaches the model.

Output: ``outputs/runs/compare-visual-models-<ts>/comparison.json`` plus a
per-variant run directory each.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.audio_dataset import build_dataloader  # noqa: E402
from src.data.celebdf_dataset import (  # noqa: E402
    build_celebdf_datasets,
    build_celebdf_manifest,
    synthetic_landmark_manifest,
)
from src.data.manifests import Manifest  # noqa: E402
from src.evaluation.metrics import binary_classification_report  # noqa: E402
from src.models.video import DeepfakeClassifier  # noqa: E402
from src.preprocessing.video_augment import FrameAugment  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.deepfake_trainer import DeepfakeTrainer  # noqa: E402
from src.training.utils import RunDirectory, count_parameters, get_device, set_seed  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "outputs" / "data" / "compare_visual_synthetic_landmarks"
VARIANTS = ("mlp_baseline", "transformer")


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    logits = torch.cat([model(x.to(device)).float().cpu() for x, _ in loader])
    targets = torch.cat([y for _, y in loader])
    probs = torch.softmax(logits, dim=1)[:, 1]
    return binary_classification_report(targets, logits.argmax(dim=1), probs)


def _run_variant(variant, cfg, datasets, device, epochs) -> dict:
    set_seed(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)
    model_cfg = dataclasses.replace(cfg.model, visual_encoder=variant)
    model = DeepfakeClassifier(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    run_dir = RunDirectory(cfg.experiment.output_root, f"compare-{variant}")
    run_dir.save_config(dataclasses.replace(cfg, model=model_cfg))
    trainer = DeepfakeTrainer(
        model, opt, config=cfg.training, run_dir=run_dir, device=device,
        class_weights=datasets["train"].label_weights(),
    )
    summary = trainer.fit(
        build_dataloader(datasets["train"], cfg.training, shuffle=True),
        build_dataloader(datasets["dev"], cfg.training, shuffle=False),
        epochs=epochs,
    )
    best = run_dir.checkpoint_path("best")
    if best.is_file():
        load_checkpoint(best, model=model, map_location=device)
    eval_split = "eval" if "eval" in datasets else "dev"
    report = _evaluate(model, build_dataloader(datasets[eval_split], cfg.training, shuffle=False),
                       device)
    return {
        "variant": variant,
        "params": count_parameters(model),
        "epochs_run": summary["epochs_run"],
        "best_monitor": summary["monitor"],
        "best_value": summary["best_metric"],
        "eval_split": eval_split,
        "eval": {k: report[k] for k in ("accuracy", "f1", "roc_auc", "eer")},
        "run_dir": run_dir.path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "deepfake_transformer.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--celebdf-root", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, help="override training.epochs for both runs")
    parser.add_argument("--landmarks-dir", type=Path,
                        help="override video.landmarks_dir (Phase 8: data/celebdf/landmarks_dense)")
    parser.add_argument("--num-frames", type=int,
                        help="override video.num_frames (Phase 8: 32, the dense window length; "
                             "num_frames == window keeps the window contiguous)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Phase 8 dense-cache plumbing: without these overrides the run reads
    # data/celebdf/landmarks (16-frame sparse) and re-subsamples to num_frames,
    # so the dense contiguous window never reaches the model.
    video = cfg.video
    if args.landmarks_dir:
        video = replace(video, landmarks_dir=str(args.landmarks_dir))
    if args.num_frames:
        video = replace(video, num_frames=args.num_frames)
    cfg = replace(cfg, video=video)

    device = get_device()
    epochs = args.epochs or cfg.training.epochs

    if args.manifest:
        manifest, synthetic, lm_dir = Manifest.read_csv(args.manifest), False, Path(cfg.video.landmarks_dir)
    elif args.celebdf_root:
        manifest = build_celebdf_manifest(args.celebdf_root,
                                          dev_identity_frac=cfg.video.dev_identity_frac,
                                          seed=cfg.experiment.seed,
                                          landmarks_dir=cfg.video.landmarks_dir,
                                          min_valid_frames=cfg.video.min_valid_frames)
        synthetic, lm_dir = False, Path(cfg.video.landmarks_dir)
    else:
        manifest = synthetic_landmark_manifest(
            SYNTHETIC_DIR, n_per_split={"train": 96, "dev": 32, "eval": 48},
            n_ids_per_split=8, num_frames=max(8, cfg.video.num_frames), seed=cfg.experiment.seed,
        )
        synthetic, lm_dir = True, SYNTHETIC_DIR

    splits = tuple(s for s in ("train", "dev", "eval") if s in manifest.split_sizes())
    datasets = build_celebdf_datasets(manifest, cfg.video, splits=splits,
                                      coords=cfg.model.landmark_coords, seed=cfg.experiment.seed,
                                      landmarks_dir=lm_dir)
    if cfg.video_augment is not None and cfg.video_augment.enabled:
        datasets["train"].frame_transform = FrameAugment(cfg.video_augment, seed=cfg.experiment.seed)

    results = [_run_variant(v, cfg, datasets, device, epochs) for v in VARIANTS]

    compare_dir = RunDirectory(cfg.experiment.output_root, "compare-visual-models")
    payload = {"synthetic": synthetic, "epochs": epochs, "seed": cfg.experiment.seed,
               "regions": cfg.model.visual_regions, "num_frames": cfg.video.num_frames,
               "eval_split": results[0]["eval_split"], "results": results}
    compare_dir.write_json("comparison.json", payload)

    print(f"\n{'variant':<16}{'params':>12}{'acc':>8}{'f1':>8}{'auc':>8}{'eer':>8}")
    print("-" * 60)
    for r in results:
        e = r["eval"]
        print(f"{r['variant']:<16}{r['params']:>12,}{e['accuracy']:>8.3f}{e['f1']:>8.3f}"
              f"{e['roc_auc']:>8.3f}{e['eer']:>8.3f}")
    delta = results[0]["eval"]["eer"] - results[1]["eval"]["eer"]
    print(f"\ntemporal contribution (baseline EER - transformer EER): {delta:+.4f}")
    print(f"comparison.json -> {compare_dir.path}")
    if synthetic:
        print("(synthetic landmarks - compares plumbing + capacity, not real skill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
