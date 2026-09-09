"""Phase 8 diagnostic: can the neural visual models memorise a tiny balanced set?

This is an OVERFIT test, not a generalisation experiment. It trains
``LandmarkMLPBaseline`` and the Visual Transformer (existing architecture,
unchanged) on **200 clips from the TRAIN split only** (100 real + 100 fake,
deterministic, seed 1337) with:

  * the real dense cache (``data/celebdf/landmarks_dense``), full 32-frame
    contiguous window (``num_frames == 32``, no random temporal subsampling)
  * ALL augmentation disabled (frame_transform = None)
  * ordinary cross-entropy, NO class weights
  * weight_decay = 0 (an overfit test removes regularisation pressure; dropout
    stays at the architecture default)
  * no early stopping, no dev / eval - the same 200 samples are used for every
    LR / model variant and for the per-epoch "train" metrics (eval-mode).

Runs a 3-point LR sweep (1e-4 / 3e-4 / 5e-4) per model = 6 canonical runs, plus
3 supplementary Transformer runs with a short per-epoch linear LR warmup
(cleanly supported by ``Trainer(scheduler=...)`` - no trainer redesign).

Outputs under ``outputs/analysis/visual_overfit/``.

    python scripts/visual_overfit_probe.py
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.audio_dataset import build_dataloader  # noqa: E402
from src.data.celebdf_dataset import CelebDFLandmarkDataset  # noqa: E402
from src.data.manifests import Manifest, ManifestRow  # noqa: E402
from src.models.video import DeepfakeClassifier  # noqa: E402
from src.preprocessing.landmarks import region_point_count  # noqa: E402
from src.training.deepfake_trainer import DeepfakeTrainer  # noqa: E402
from src.training.utils import RunDirectory, count_parameters, get_device, set_seed  # noqa: E402

OUT = REPO_ROOT / "outputs" / "analysis" / "visual_overfit"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "celebdf" / "manifest_dense.csv"
DEFAULT_LANDMARKS = REPO_ROOT / "data" / "celebdf" / "landmarks_dense"


# --------------------------------------------------------------- subset selection

def select_overfit_subset(manifest: Manifest, *, n_per_class: int = 100, seed: int = 1337
                          ) -> list[ManifestRow]:
    """Deterministic balanced subset from the TRAIN split only.

    Rule: take the ``train`` rows, split by label, sort each group by
    ``sample_id`` (stable), shuffle each with ``random.Random(seed)``, take the
    first ``n`` of each (``n = min(n_per_class, len(reals), len(fakes))``), and
    concatenate reals + fakes. dev / eval rows are never read.
    """

    import random

    train = list(manifest.split("train"))
    reals = sorted((r for r in train if r.label == 1), key=lambda r: r.sample_id)
    fakes = sorted((r for r in train if r.label == 0), key=lambda r: r.sample_id)
    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    n = min(n_per_class, len(reals), len(fakes))
    return reals[:n] + fakes[:n]


def build_overfit_dataset(subset: list[ManifestRow], landmarks_dir: Path,
                          *, coords: int, num_frames: int, regions: str,
                          video_cfg) -> CelebDFLandmarkDataset:
    vc = dataclasses.replace(video_cfg, landmarks_dir=str(landmarks_dir),
                             num_frames=num_frames, regions=regions)
    return CelebDFLandmarkDataset(
        Manifest(subset), vc, landmarks_dir=landmarks_dir, coords=coords,
        num_frames=num_frames, random_sample=False, frame_transform=None,  # no aug
    )


class _CachedDataset(torch.utils.data.Dataset):
    """Memoise the (deterministic, aug-free) items so every epoch / run reuses
    the same in-RAM tensors instead of re-reading + re-normalising the .npz."""

    def __init__(self, base: CelebDFLandmarkDataset) -> None:
        self._items = [base[i] for i in range(len(base))]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        return self._items[i]


# ------------------------------------------------------------------------ one run

def _run(variant: str, lr: float, warmup_epochs: int, epochs: int, dataset,
         train_cfg, model_cfg, device, out: Path) -> dict:
    set_seed(1337, deterministic=False)
    mcfg = dataclasses.replace(model_cfg, visual_encoder=variant)
    model = DeepfakeClassifier(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    scheduler = None
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: min(1.0, (e + 1) / warmup_epochs))

    label = f"{variant}_lr{lr:.0e}" + (f"_warmup{warmup_epochs}" if warmup_epochs else "")
    run_dir = RunDirectory(out / "runs", label)
    tcfg = dataclasses.replace(train_cfg, learning_rate=lr, epochs=epochs)
    trainer = DeepfakeTrainer(model, opt, config=tcfg, run_dir=run_dir, device=device,
                              scheduler=scheduler, class_weights=None)  # ordinary CE

    train_loader = build_dataloader(dataset, tcfg, shuffle=True, drop_last=False)
    metric_loader = build_dataloader(dataset, tcfg, shuffle=False, drop_last=False)  # same 200

    t0 = time.time()
    trainer.fit(train_loader, metric_loader, epochs=epochs)   # "val_*" == train-set, eval-mode
    dur = time.time() - t0

    rows = list(csv.DictReader(run_dir.metrics_path.open(encoding="utf-8")))
    hist_path = (out / "history" / f"{label}.csv").resolve()
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir.metrics_path, hist_path)
    shutil.rmtree(run_dir.checkpoints, ignore_errors=True)   # keep metrics.csv, drop weights
    try:
        hist_ref = str(hist_path.relative_to(REPO_ROOT))
    except ValueError:
        hist_ref = str(hist_path)

    def col(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

    train_loss = col("train_loss")            # dropout ON
    ev_loss = col("val_loss")                 # dropout OFF, same 200
    ev_auc = col("val_roc_auc")
    ev_acc = col("val_accuracy")
    ev_f1 = col("val_f1")
    return {
        "label": label, "variant": variant, "lr": lr, "warmup_epochs": warmup_epochs,
        "params": count_parameters(model), "epochs": epochs, "seconds": round(dur, 1),
        "final_train_loss_dropout_on": round(train_loss[-1], 4),
        "final_loss": round(ev_loss[-1], 4),
        "min_loss": round(min(ev_loss), 4),
        "final_train_auc": round(ev_auc[-1], 4),
        "max_train_auc": round(max(ev_auc), 4),
        "final_train_acc": round(ev_acc[-1], 4),
        "final_train_f1": round(ev_f1[-1], 4),
        "history_csv": hist_ref,
        "loss_curve": [round(x, 4) for x in ev_loss],
        "auc_curve": [round(x, 4) for x in ev_auc],
    }


def _verdict(r: dict) -> str:
    if r["min_loss"] <= 0.05 and r["max_train_auc"] >= 0.99:
        return "STRONG (memorised)"
    if r["max_train_auc"] >= 0.9 and r["min_loss"] < 0.5:
        return "PARTIAL"
    return "FAILURE (cannot memorise)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "deepfake_transformer.yaml")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=10,
                   help="for the 3 supplementary Transformer runs (0 disables them)")
    p.add_argument("--variants", nargs="+", choices=["mlp_baseline", "transformer"],
                   help="restrict which models run (default: both)")
    p.add_argument("--lrs", nargs="+", type=float,
                   help="restrict the LR sweep (default: 1e-4 3e-4 5e-4)")
    p.add_argument("--out-dir", type=Path, default=OUT,
                   help="output dir (use a separate one to not clobber a prior full run)")
    p.add_argument("--smoke", action="store_true", help="1 tiny run to check plumbing")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    regions, coords = "face_mouth", cfg.model.landmark_coords
    num_frames = 32
    out = args.out_dir.resolve()
    epochs = 3 if args.smoke else args.epochs
    lrs = args.lrs or [1e-4, 3e-4, 5e-4]
    variants = args.variants or ["mlp_baseline", "transformer"]
    warm = 0 if args.smoke else args.warmup_epochs

    combos = [(v, lr, 0) for v in variants for lr in lrs]
    if warm > 0 and "transformer" in variants:
        combos += [("transformer", lr, warm) for lr in lrs]
    if args.smoke:
        combos = combos[:1]

    manifest = Manifest.read_csv(args.manifest)
    subset = select_overfit_subset(manifest, n_per_class=args.n_per_class, seed=1337)
    n_real = sum(1 for r in subset if r.label == 1)
    ids = [r.sample_id for r in subset]
    # structural no-leak check: every picked row is a train row
    train_ids = {r.sample_id for r in manifest.split("train")}
    dev_eval_ids = {r.sample_id for r in manifest if r.split in ("dev", "eval")}
    assert set(ids) <= train_ids, "subset escaped the train split"
    assert set(ids).isdisjoint(dev_eval_ids), "subset leaked into dev/eval"

    dataset = build_overfit_dataset(subset, args.landmarks_dir, coords=coords,
                                    num_frames=num_frames, regions=regions,
                                    video_cfg=cfg.video)

    # ---- verify the tensor the model will actually receive --------------
    x0, y0 = dataset[0]
    n_pts = region_point_count(regions)
    assert x0.shape == (num_frames, n_pts, coords), x0.shape
    tcfg = dataclasses.replace(cfg.training, batch_size=32, num_workers=0, amp=False,
                               weight_decay=0.0, grad_accum_steps=1,
                               monitor="train_loss", monitor_mode="min",
                               early_stopping_patience=0)
    probe_loader = build_dataloader(dataset, tcfg, shuffle=False, drop_last=False)
    xb, yb = next(iter(probe_loader))
    verified_shape = list(xb.shape)
    assert verified_shape[1:] == [num_frames, n_pts, coords], verified_shape
    assert dataset.frame_transform is None                     # aug disabled
    assert not dataset.random_sample                           # no random temporal subsampling

    dataset = _CachedDataset(dataset)                          # freeze the 200 tensors in RAM

    out.mkdir(parents=True, exist_ok=True)
    config = {
        "manifest": str(args.manifest), "landmarks_dir": str(args.landmarks_dir),
        "regions": regions, "n_landmarks": n_pts, "coords": coords,
        "num_frames": num_frames, "random_temporal_subsampling": False,
        "augmentation": "disabled (frame_transform=None)",
        "class_weighting": "disabled (ordinary cross-entropy)",
        "weight_decay": 0.0, "dropout": cfg.model.dropout,
        "visual_tf_dropout": cfg.model.visual_tf_dropout,
        "amp": False, "epochs": epochs, "seed": 1337, "smoke": args.smoke,
        "lrs": lrs, "variants": variants, "warmup_epochs_supplementary": warm,
        "combos": [{"variant": v, "lr": lr, "warmup_epochs": w} for v, lr, w in combos],
        "subset_rule": "train split only; per label sort by sample_id, "
                       "random.Random(1337).shuffle, take first n=min(100,#real,#fake), "
                       "concat reals+fakes",
        "n_subset": len(subset), "n_real": n_real, "n_fake": len(subset) - n_real,
        "verified_input_shape_batch": verified_shape,
        "sample_ids": ids,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"device={device}  out={out}  epochs={epochs}  combos={len(combos)}  "
          f"subset: {len(subset)} clips ({n_real} real / {len(subset)-n_real} fake)  "
          f"input batch shape {verified_shape}")
    results = []
    for variant, lr, w in combos:
        r = _run(variant, lr, w, epochs, dataset, tcfg, cfg.model, device, out)
        r["verdict"] = _verdict(r)
        results.append(r)
        print(f"  {r['label']:<32} epochs_recorded={len(r['loss_curve'])}  "
              f"min_loss={r['min_loss']:.3f}  max_train_auc={r['max_train_auc']:.3f}  "
              f"final_acc={r['final_train_acc']:.3f}  -> {r['verdict']}  ({r['seconds']}s)")

    (out / "summary.json").write_text(json.dumps({"config": config, "runs": results}, indent=2),
                                      encoding="utf-8")
    _write_md(out / "summary.md", results, n_real, len(subset) - n_real, verified_shape, epochs)
    print(f"\n-> {out}")
    return 0


def _write_md(path: Path, runs: list[dict], n_real: int, n_fake: int, shape, epochs: int) -> None:
    lines = [
        "# Phase 8 tiny-subset overfit diagnostic", "",
        f"- Subset: **{n_real} real + {n_fake} fake** from the `train` split of "
        "`manifest_dense.csv` (seed 1337).",
        f"- Cache: `data/celebdf/landmarks_dense`  |  input tensor per batch: "
        f"`{shape}` = `[B, 32, 142, 3]`.",
        "- Augmentation: **off**.  Class weights: **off** (plain CE).  weight_decay: **0**.  "
        f"AMP: off.  Epochs: {epochs}.  No dev/eval, no early stopping.", "",
        "| run | params | min loss (eval-mode) | max train AUC | final train acc | final train F1 | verdict |",
        "|---|--:|--:|--:|--:|--:|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['label']} | {r['params']:,} | {r['min_loss']:.3f} | {r['max_train_auc']:.3f} "
            f"| {r['final_train_acc']:.3f} | {r['final_train_f1']:.3f} | {r['verdict']} |")
    lines += ["", "Per-epoch curves: `history/<run>.csv` (columns include `train_loss`, "
              "`val_loss`, `val_roc_auc`, `val_accuracy`, `val_f1`; the `val_*` columns are "
              "computed on the same 200 training clips in eval mode)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
