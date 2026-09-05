"""Phase 5 smoke test: CNN + Transformer audio encoder end-to-end.

    python scripts/test_audio_transformer.py

Checks that the Transformer encoder preserves token shape, that the shared
``AudioEncoder`` builds both variants, that ``SpoofClassifier`` with
``audio_encoder="cnn_transformer"`` trains (loss decreases) through
``SpoofTrainer``, and that an exported encoder reloads to bit-identical tokens.
Exit code is non-zero on any failure.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.asvspoof_dataset import make_asvspoof_datasets
from src.data.audio_dataset import build_dataloader
from src.data.manifests import synthetic_manifest
from src.models.audio import (
    AudioEncoder,
    AudioTransformerEncoder,
    SinusoidalPositionalEncoding,
    SpoofClassifier,
    export_audio_encoder,
    load_audio_encoder,
)
from src.preprocessing.augment import SpecAugment
from src.training.spoof_trainer import SpoofTrainer
from src.training.utils import RunDirectory, count_parameters, get_device, set_seed

DATA_DIR = REPO_ROOT / "outputs" / "data" / "audio_tf_smoke"
FIXED_SECONDS = 1.0


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "spoof_transformer.yaml")
    cfg = replace(
        cfg,
        model=replace(cfg.model, audio_cnn_channels=(16, 32), audio_embedding_dim=64,
                      audio_tf_layers=2, audio_tf_ff_dim=128, spoof_head_hidden=32),
        data=replace(cfg.data, fixed_seconds=FIXED_SECONDS),
        training=replace(cfg.training, batch_size=8, num_workers=0, epochs=4,
                         early_stopping_patience=0),
    )
    set_seed(cfg.experiment.seed)
    device = get_device()
    n_mels = cfg.audio.mel.n_mels

    # --- positional encoding + transformer shape ------------------------
    pe = SinusoidalPositionalEncoding(64, max_len=32)
    _check(pe(torch.zeros(2, 10, 64)).shape == (2, 10, 64), "pos-encoding changed shape")
    try:
        pe(torch.zeros(1, 40, 64))
    except ValueError:
        pass
    else:
        raise AssertionError("pos-encoding should reject seq_len > max_len")

    tf = AudioTransformerEncoder(cfg.model)
    tokens = torch.randn(3, 13, 64)
    _check(tf(tokens).shape == tokens.shape, "transformer must preserve [B, T, D]")

    # --- shared encoder: both variants ---------------------------------
    enc_cnn = AudioEncoder(replace(cfg.model, audio_encoder="cnn"), n_mels)
    enc_tf = AudioEncoder(cfg.model, n_mels)
    _check(enc_cnn.transformer is None and enc_tf.transformer is not None, "variant wiring wrong")
    mel = torch.randn(2, n_mels, 101)
    _check(enc_cnn(mel).tokens.shape == enc_tf(mel).tokens.shape, "variant token shapes differ")
    _check(count_parameters(enc_tf) > count_parameters(enc_cnn), "transformer should add params")

    # --- train the transformer classifier ----------------------------
    manifest = synthetic_manifest(
        DATA_DIR, n_per_split={"train": 48, "dev": 24}, n_speakers_per_split=6,
        seed=cfg.experiment.seed, sample_rate=cfg.audio.sample_rate, duration_s=FIXED_SECONDS,
    )
    datasets = make_asvspoof_datasets(manifest, cfg.audio, feature="logmel",
                                      fixed_seconds=FIXED_SECONDS, splits=("train", "dev"))
    run_dir = RunDirectory(cfg.experiment.output_root, "phase5-smoke")
    run_dir.save_config(cfg)
    model = SpoofClassifier(cfg.model, n_mels=n_mels)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SpoofTrainer(model, opt, config=cfg.training, run_dir=run_dir, device=device,
                           spec_augment=SpecAugment(cfg.augment.specaugment),
                           class_weights=datasets["train"].label_weights())
    summary = trainer.fit(
        build_dataloader(datasets["train"], cfg.training, shuffle=True),
        build_dataloader(datasets["dev"], cfg.training, shuffle=False),
        epochs=cfg.training.epochs,
    )
    losses = [r["train_loss"] for r in trainer.history]
    _check(losses[-1] < losses[0], f"train loss did not decrease: {losses}")

    # --- export / reload round-trip --------------------------------
    export_path = export_audio_encoder(
        run_dir.checkpoint_path("audio_encoder"),
        encoder=model.encoder, model_cfg=cfg.model, audio_cfg=cfg.audio, n_mels=n_mels,
    )
    reloaded, payload = load_audio_encoder(export_path)              # on CPU
    source_encoder = model.encoder.to("cpu").eval()
    reloaded.eval()
    probe = torch.randn(2, n_mels, 101)
    with torch.no_grad():
        a = source_encoder(probe).tokens
        b = reloaded(probe).tokens
    _check(torch.equal(a, b), "reloaded encoder produced different tokens")
    _check(payload["variant"] == "cnn_transformer", "export lost the variant tag")

    print(f"\ncnn params        : {count_parameters(enc_cnn):,}")
    print(f"cnn+tf params     : {count_parameters(enc_tf):,}")
    print(f"classifier params : {count_parameters(model):,}")
    print(f"train loss        : {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"best {summary['monitor']:<9}: {summary['best_metric']:.4f}")
    print(f"encoder export    : {export_path.name} (variant={payload['variant']})")
    print("\nPhase 5 smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 5 smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
