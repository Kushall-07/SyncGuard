"""Phase 2B unit tests: config, reproducibility, run dirs, checkpoints, metrics, Trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config, ModelConfig, TrainingConfig, load_config
from src.evaluation.metrics import (
    accuracy,
    binary_classification_report,
    equal_error_rate,
    precision_recall_f1,
    roc_auc,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import Trainer
from src.training.utils import RunDirectory, config_to_dict, count_parameters, set_seed

CONFIG_PATH = "configs/default.yaml"


# --------------------------------------------------------------------------- config


def test_load_config_resolves_audio_from_path() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.mel.n_mels == 80
    assert cfg.experiment.seed == 1337


def test_config_rejects_unknown_root_key() -> None:
    with pytest.raises(ValueError):
        Config.from_dict({"experiment": {}, "bogus_section": {}})


def test_model_config_requires_head_divisibility() -> None:
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"audio_embedding_dim": 100, "num_heads": 3})


def test_training_config_rejects_bad_monitor_mode() -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({"monitor_mode": "sideways"})


def test_config_to_dict_is_json_friendly() -> None:
    cfg = load_config(CONFIG_PATH)
    d = config_to_dict(cfg)
    assert d["audio"]["mel"]["hop_length"] == 160
    assert d["training"]["monitor"] == "val_loss"


# ------------------------------------------------------------------- reproducibility


def test_set_seed_makes_torch_rng_deterministic() -> None:
    set_seed(123)
    a = torch.randn(64)
    set_seed(123)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_count_parameters_counts_trainable() -> None:
    model = nn.Linear(10, 4)  # 10*4 weights + 4 bias
    assert count_parameters(model) == 44
    model.bias.requires_grad_(False)
    assert count_parameters(model) == 40
    assert count_parameters(model, trainable_only=False) == 44


# ----------------------------------------------------------------------- run dir


def test_run_directory_layout_and_logging(tmp_path) -> None:
    run = RunDirectory(tmp_path, "unit", timestamp="fixed")
    assert run.path == tmp_path / "unit-fixed"
    assert run.checkpoints.is_dir()

    run.save_config(load_config(CONFIG_PATH))
    assert run.config_path.is_file()

    logger = run.get_logger("syncguard.test")
    logger.info("hello world")
    assert "hello world" in run.log_path.read_text(encoding="utf-8")
    # calling twice must not double-attach handlers
    n = len(logger.handlers)
    run.get_logger("syncguard.test")
    assert len(logger.handlers) == n


# --------------------------------------------------------------------- checkpoint


def test_checkpoint_round_trip(tmp_path) -> None:
    model = nn.Linear(8, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "ck" / "best.pt", model=model, optimizer=opt, epoch=7, best_metric=0.5,
    )
    assert path.is_file()

    fresh = nn.Linear(8, 3)
    assert not torch.equal(fresh.weight, model.weight)
    ckpt = load_checkpoint(path, model=fresh, map_location="cpu")
    assert torch.equal(fresh.weight, model.weight)
    assert ckpt["epoch"] == 7 and ckpt["best_metric"] == 0.5


def test_load_checkpoint_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


# ------------------------------------------------------------------------ metrics


def test_accuracy_and_prf() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(0.6)
    prf = precision_recall_f1(y_true, y_pred)
    assert prf["precision"] == pytest.approx(2 / 3)
    assert prf["recall"] == pytest.approx(2 / 3)


def test_roc_auc_perfect_separation() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y_true, y_score) == pytest.approx(1.0)


def test_equal_error_rate_separable_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.concatenate([rng.normal(-2, 0.5, 100), rng.normal(2, 0.5, 100)])
    eer, threshold = equal_error_rate(y_true, y_score)
    assert 0.0 <= eer < 0.05
    assert np.isfinite(threshold)


def test_equal_error_rate_single_class_is_nan() -> None:
    eer, threshold = equal_error_rate([1, 1, 1], [0.2, 0.3, 0.4])
    assert np.isnan(eer) and np.isnan(threshold)


def test_binary_report_has_expected_keys() -> None:
    report = binary_classification_report([0, 1, 0, 1], [0, 1, 1, 1], [0.2, 0.9, 0.6, 0.8])
    assert {"accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc",
            "eer", "eer_threshold"} <= set(report)
    assert np.array(report["confusion_matrix"]).shape == (2, 2)


# ------------------------------------------------------------------------ trainer


def _tiny_loaders(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(96, 12, generator=g)
    y = (x.sum(dim=1) > 0).long()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True), DataLoader(ds, batch_size=16)


class _MLPTrainer(Trainer):
    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        return {"loss": nn.functional.cross_entropy(logits, y),
                "logits": logits.detach(), "targets": y.detach()}

    def compute_metrics(self, step_outputs):
        logits = torch.cat([o["logits"] for o in step_outputs])
        targets = torch.cat([o["targets"] for o in step_outputs])
        return {"accuracy": accuracy(targets, logits.argmax(dim=1))}


def _make_trainer(tmp_path, **train_overrides):
    set_seed(0)
    model = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 2))
    lr = train_overrides.pop("learning_rate", 1e-2)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    cfg = TrainingConfig(batch_size=16, epochs=6, amp=False, **train_overrides)
    run = RunDirectory(tmp_path, "trainer", timestamp="t")
    return _MLPTrainer(model, opt, config=cfg, run_dir=run, device="cpu"), run


def test_trainer_writes_metrics_and_checkpoints(tmp_path) -> None:
    trainer, run = _make_trainer(tmp_path)
    train_loader, val_loader = _tiny_loaders()
    summary = trainer.fit(train_loader, val_loader, epochs=3)

    assert summary["epochs_run"] == 3
    assert run.checkpoint_path("best").is_file()
    assert run.checkpoint_path("last").is_file()
    lines = run.metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # header + 3 epochs
    assert "val_accuracy" in lines[0]
    assert (run.path / "summary.json").is_file()


def test_trainer_early_stops_when_metric_stalls(tmp_path) -> None:
    # lr=0 -> val_loss never improves after epoch 1; patience=1 -> stop at epoch 2
    trainer, _ = _make_trainer(tmp_path, learning_rate=0.0, early_stopping_patience=1)
    train_loader, val_loader = _tiny_loaders()
    summary = trainer.fit(train_loader, val_loader, epochs=10)
    assert summary["epochs_run"] == 2
    assert summary["best_epoch"] == 1


def test_trainer_raises_on_non_finite_loss(tmp_path) -> None:
    trainer, _ = _make_trainer(tmp_path)
    train_loader, _ = _tiny_loaders()

    def boom(batch):
        return {"loss": torch.tensor(float("nan"), requires_grad=True)}

    trainer.compute_loss = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.fit(train_loader, epochs=1)
