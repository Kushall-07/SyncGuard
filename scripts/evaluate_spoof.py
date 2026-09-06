"""Phase 6: evaluate a trained spoof classifier and render the report + figures.

    # evaluate the best checkpoint of a training run on its eval split
    python scripts/evaluate_spoof.py --run outputs/runs/spoof-cnn-baseline-XXXX

    # explicit config + checkpoint, on a specific manifest / split
    python scripts/evaluate_spoof.py --config configs/spoof_cnn.yaml \
        --checkpoint outputs/runs/.../checkpoints/best.pt \
        --manifest data/asvspoof/manifest.csv --split eval

Writes ``report.json`` (accuracy / precision / recall / F1 / ROC-AUC / EER /
confusion matrix + per-attack EER), ``report.txt``, and PNG figures (confusion
matrix, ROC, DET with EER, score distributions) under ``<out>`` (default
``<run>/eval``). ``--plot-training`` also renders loss/metric curves from the
run's ``metrics.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config
from src.data.asvspoof_dataset import build_asvspoof_manifest, make_asvspoof_datasets
from src.data.audio_dataset import build_dataloader
from src.data.manifests import Manifest, synthetic_manifest
from src.evaluation.spoof_eval import evaluate_spoof_model
from src.evaluation.visualization import plot_training_curves, save_spoof_evaluation_figures
from src.models.audio import SpoofClassifier
from src.training.checkpoint import load_checkpoint
from src.training.utils import get_device, set_seed

SYNTHETIC_DIR = REPO_ROOT / "outputs" / "data" / "evaluate_spoof_synthetic"


def _resolve_run(args) -> tuple[Config, Path, Path | None]:
    if args.run:
        run = Path(args.run)
        cfg = load_config(run / "config.yaml")
        ckpt = Path(args.checkpoint) if args.checkpoint else run / "checkpoints" / "best.pt"
        return cfg, ckpt, run
    if not (args.config and args.checkpoint):
        raise SystemExit("provide --run, or both --config and --checkpoint")
    return load_config(args.config), Path(args.checkpoint), None


def _resolve_manifest(args, cfg: Config):
    if args.manifest:
        return Manifest.read_csv(args.manifest), False
    if args.asvspoof_root:
        return build_asvspoof_manifest(args.asvspoof_root), False
    if args.synthetic or not Path(cfg.data.manifest_csv).is_file():
        m = synthetic_manifest(
            SYNTHETIC_DIR, n_per_split={"train": 64, "dev": 48, "eval": 96},
            n_speakers_per_split=8, seed=cfg.experiment.seed,
            sample_rate=cfg.audio.sample_rate, duration_s=cfg.data.fixed_seconds,
        )
        return m, True
    return Manifest.read_csv(cfg.data.manifest_csv), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="training run dir (reads config.yaml + best.pt)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asvspoof-root", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--out", type=Path, help="output dir (default <run>/eval)")
    parser.add_argument("--plot-training", action="store_true")
    args = parser.parse_args()

    cfg, ckpt_path, run = _resolve_run(args)
    set_seed(cfg.experiment.seed)
    device = get_device()

    manifest, is_synthetic = _resolve_manifest(args, cfg)
    split = args.split if args.split in manifest.split_sizes() else "dev"
    datasets = make_asvspoof_datasets(
        manifest, cfg.audio, feature=cfg.data.feature,
        fixed_seconds=cfg.data.fixed_seconds, splits=(split,), seed=cfg.experiment.seed,
    )
    loader = build_dataloader(datasets[split], cfg.training, shuffle=False)

    model = SpoofClassifier(cfg.model, n_mels=cfg.audio.mel.n_mels).to(device)
    ckpt = load_checkpoint(ckpt_path, model=model, map_location=device)

    result = evaluate_spoof_model(model, loader, device)

    out_dir = args.out or (run / "eval" if run else REPO_ROOT / "outputs" / "eval" / cfg.experiment.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": split,
        "audio_encoder": cfg.model.audio_encoder,
        "synthetic": is_synthetic,
        **result.report,
        "per_attack_eer": result.per_attack,
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    figures = save_spoof_evaluation_figures(
        out_dir, y_true=result.y_true, y_score=result.y_score,
        confusion=result.report["confusion_matrix"], prefix=split,
    )
    if args.plot_training and run and (run / "metrics.csv").is_file():
        figures["training_curves"] = plot_training_curves(run / "metrics.csv",
                                                          out_dir / "training_curves.png")

    _write_text_report(out_dir / "report.txt", payload, result)

    print(f"\n{cfg.experiment.name} | {split} split | {result.summary_line()}")
    if result.per_attack:
        worst = max(result.per_attack.items(), key=lambda kv: kv[1]["eer"])
        print(f"per-attack EER: {len(result.per_attack)} systems, "
              f"worst {worst[0]} = {worst[1]['eer'] * 100:.2f}%")
    print(f"figures + report -> {out_dir}")
    if is_synthetic:
        print("(synthetic audio - plumbing only, not a real evaluation)")
    return 0


def _write_text_report(path: Path, payload: dict, result) -> None:
    lines = [
        f"checkpoint : {payload['checkpoint']} (epoch {payload['checkpoint_epoch']})",
        f"split      : {payload['split']}   encoder: {payload['audio_encoder']}"
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
        lines += ["", "per-attack EER (spoof system vs pooled bonafide):"]
        for name, d in sorted(result.per_attack.items(), key=lambda kv: kv[1]["eer"], reverse=True):
            lines.append(f"  {name:<6} n={d['n_spoof']:<6} EER={d['eer'] * 100:6.2f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
