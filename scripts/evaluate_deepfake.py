"""Phase 7: evaluate a trained deepfake classifier and render report + figures.

    # evaluate the best checkpoint of a training run on its eval split
    python scripts/evaluate_deepfake.py --run outputs/runs/deepfake-transformer-baseline-XXXX

    # explicit config + checkpoint on a specific manifest / split
    python scripts/evaluate_deepfake.py --config configs/deepfake_transformer.yaml \
        --checkpoint outputs/runs/.../checkpoints/best.pt \
        --manifest data/celebdf/manifest.csv --split eval

Writes ``report.json`` / ``report.txt`` (accuracy / P / R / F1 / ROC-AUC / EER /
confusion + per-method EER) and PNG figures (confusion, ROC, DET with EER,
``P(real)`` score distributions) under ``<out>`` (default ``<run>/eval``).
``--plot-training`` also renders loss/metric curves from ``metrics.csv``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config  # noqa: E402
from src.data.audio_dataset import build_dataloader  # noqa: E402
from src.data.celebdf_dataset import build_celebdf_datasets, synthetic_landmark_manifest  # noqa: E402
from src.data.manifests import Manifest  # noqa: E402
from src.evaluation.deepfake_eval import evaluate_deepfake_model  # noqa: E402
from src.evaluation.visualization import plot_training_curves, save_spoof_evaluation_figures  # noqa: E402
from src.models.video import DeepfakeClassifier  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.utils import get_device, set_seed  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "outputs" / "data" / "deepfake_eval_synthetic_landmarks"
_CLASS_NAMES = ("fake", "real")


def _resolve_run(args) -> tuple[Config, Path, Path | None]:
    if args.run:
        run = Path(args.run)
        cfg = load_config(run / "config.yaml")
        ckpt = Path(args.checkpoint) if args.checkpoint else run / "checkpoints" / "best.pt"
        return cfg, ckpt, run
    if not (args.config and args.checkpoint):
        raise SystemExit("provide --run, or both --config and --checkpoint")
    return load_config(args.config), Path(args.checkpoint), None


def _resolve_manifest(args, cfg: Config) -> tuple[Manifest, Path, bool]:
    if args.synthetic or not Path(cfg.video.manifest_csv).is_file():
        m = synthetic_landmark_manifest(
            SYNTHETIC_DIR, n_per_split={"train": 32, "dev": 24, "eval": 64},
            n_ids_per_split=8, num_frames=max(8, cfg.video.num_frames),
            seed=cfg.experiment.seed,
        )
        return m, SYNTHETIC_DIR, True
    if args.manifest:
        return Manifest.read_csv(args.manifest), Path(cfg.video.landmarks_dir), False
    return Manifest.read_csv(cfg.video.manifest_csv), Path(cfg.video.landmarks_dir), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="training run dir (config.yaml + best.pt)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--out", type=Path, help="output dir (default <run>/eval)")
    parser.add_argument("--plot-training", action="store_true")
    parser.add_argument("--landmarks-dir", type=Path, help="override video.landmarks_dir")
    parser.add_argument("--num-frames", type=int, help="override video.num_frames")
    args = parser.parse_args()

    cfg, ckpt_path, run = _resolve_run(args)
    if args.landmarks_dir or args.num_frames:
        video = cfg.video
        if args.landmarks_dir:
            video = dataclasses.replace(video, landmarks_dir=str(args.landmarks_dir))
        if args.num_frames:
            video = dataclasses.replace(video, num_frames=args.num_frames)
        cfg = dataclasses.replace(cfg, video=video)
    set_seed(cfg.experiment.seed)
    device = get_device()

    manifest, landmarks_dir, is_synthetic = _resolve_manifest(args, cfg)
    split = args.split if args.split in manifest.split_sizes() else "dev"
    datasets = build_celebdf_datasets(
        manifest, cfg.video, splits=(split,),
        coords=cfg.model.landmark_coords, seed=cfg.experiment.seed,
        landmarks_dir=landmarks_dir,
    )
    loader = build_dataloader(datasets[split], cfg.training, shuffle=False)

    model = DeepfakeClassifier(cfg.model).to(device)
    ckpt = load_checkpoint(ckpt_path, model=model, map_location=device)

    result = evaluate_deepfake_model(model, loader, device)

    out_dir = args.out or (run / "eval" if run else REPO_ROOT / "outputs" / "eval" / cfg.experiment.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": split,
        "visual_encoder": cfg.model.visual_encoder,
        "regions": cfg.model.visual_regions,
        "num_frames": cfg.video.num_frames,
        "synthetic": is_synthetic,
        **result.report,
        "per_method_eer": result.per_attack,
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    figures = save_spoof_evaluation_figures(
        out_dir, y_true=result.y_true, y_score=result.y_score,
        confusion=result.report["confusion_matrix"], prefix=split,
        class_names=_CLASS_NAMES, pos_name="real",
    )
    if args.plot_training and run and (run / "metrics.csv").is_file():
        figures["training_curves"] = plot_training_curves(run / "metrics.csv",
                                                          out_dir / "training_curves.png")

    _write_text_report(out_dir / "report.txt", payload, result)

    print(f"\n{cfg.experiment.name} | {split} split | {result.summary_line()}")
    if result.per_attack:
        worst = max(result.per_attack.items(), key=lambda kv: kv[1]["eer"])
        print(f"per-method EER: {len(result.per_attack)} method(s), "
              f"worst {worst[0]} = {worst[1]['eer'] * 100:.2f}%")
    print(f"figures + report -> {out_dir}")
    if is_synthetic:
        print("(synthetic landmarks - plumbing only, not a real evaluation)")
    return 0


def _write_text_report(path: Path, payload: dict, result) -> None:
    lines = [
        f"checkpoint : {payload['checkpoint']} (epoch {payload['checkpoint_epoch']})",
        f"split      : {payload['split']}   encoder: {payload['visual_encoder']}   "
        f"regions: {payload['regions']}   frames: {payload['num_frames']}"
        + ("   [SYNTHETIC]" if payload["synthetic"] else ""),
        "",
        f"accuracy   : {payload['accuracy']:.4f}",
        f"precision  : {payload['precision']:.4f}",
        f"recall     : {payload['recall']:.4f}",
        f"f1         : {payload['f1']:.4f}",
        f"roc_auc    : {payload.get('roc_auc', float('nan')):.4f}",
        f"eer        : {payload.get('eer', float('nan')) * 100:.2f}%  "
        f"(thr {payload.get('eer_threshold', float('nan')):.4f})",
        f"confusion  : {payload['confusion_matrix']}   [[TN, FP], [FN, TP]]",
    ]
    if result.per_attack:
        lines += ["", "per-method EER (synthesis method vs pooled real):"]
        for name, d in sorted(result.per_attack.items(), key=lambda kv: kv[1]["eer"], reverse=True):
            lines.append(f"  {name:<12} n={d['n_spoof']:<6} EER={d['eer'] * 100:6.2f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
