"""Phase 8 final-experiment plumbing: TrainingConfig scheduler / warmup /
class_weight fields, build_scheduler, dev-AUC checkpoint selection, and the
configs/deepfake_transformer_final.yaml values. Existing configs unchanged."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TrainingConfig, load_config
from src.training.trainer import Trainer
from src.training.utils import RunDirectory, build_scheduler


# ---------------------------------------------------------------- TrainingConfig

def test_new_training_fields_have_backward_compatible_defaults() -> None:
    t = TrainingConfig()
    assert t.scheduler == "none"
    assert t.warmup_epochs == 0
    assert t.class_weight == "balanced"          # matches the old hard-coded behaviour


def test_existing_configs_are_unchanged() -> None:
    for name in ("configs/deepfake_transformer.yaml",
                 "configs/deepfake_landmark_baseline.yaml",
                 "configs/spoof_cnn.yaml",
                 "configs/default.yaml"):
        t = load_config(name).training
        assert t.scheduler == "none"
        assert t.warmup_epochs == 0
        assert t.class_weight == "balanced"
    # the video configs still monitor EER with early stopping
    assert load_config("configs/deepfake_transformer.yaml").training.monitor == "val_eer"


@pytest.mark.parametrize("bad", [
    {"scheduler": "linear"},
    {"class_weight": "focal"},
    {"warmup_epochs": -1},
    {"warmup_epochs": 20, "epochs": 20},        # warmup must be < epochs
])
def test_training_config_validates_new_fields(bad) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({"epochs": 40, **bad})


# ---------------------------------------------------------------- build_scheduler

def test_build_scheduler_none_returns_none() -> None:
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    assert build_scheduler(opt, TrainingConfig(epochs=10, scheduler="none")) is None


def test_build_scheduler_cosine_warmup_then_decay() -> None:
    base = 3e-4
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=base)
    cfg = TrainingConfig(epochs=100, warmup_epochs=10, scheduler="cosine",
                         batch_size=32, learning_rate=base)
    sch = build_scheduler(opt, cfg)
    assert isinstance(sch, torch.optim.lr_scheduler.LambdaLR)

    lrs = []
    for _ in range(100):
        w.sum().backward(retain_graph=True)
        opt.step()                     # optimizer.step() BEFORE scheduler.step(): the Trainer order
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()
    lrs = np.array(lrs)

    assert np.all(np.diff(lrs[:10]) > 0)                 # linear ramp over warmup
    assert lrs[9] == pytest.approx(base, rel=1e-9)       # peak == base at end of warmup
    assert np.all(np.diff(lrs[10:]) <= 1e-15)            # cosine monotone decrease
    assert lrs[-1] < base / 50                           # decays to ~0


# ------------------------------------------------- dev-AUC checkpoint selection

class _LinTrainer(Trainer):
    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x.to(self.device))
        loss = nn.functional.cross_entropy(logits, y.to(self.device))
        return {"loss": loss, "logits": logits.detach(), "targets": y}

    def compute_metrics(self, outs):
        from src.evaluation.metrics import binary_classification_report
        logits = torch.cat([o["logits"] for o in outs])
        y = torch.cat([o["targets"] for o in outs])
        r = binary_classification_report(y, logits.argmax(1), torch.softmax(logits, 1)[:, 1])
        return {"roc_auc": r["roc_auc"], "eer": r["eer"]}


def test_trainer_selects_best_on_max_dev_auc_and_logs_lr(tmp_path) -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x[:, 0] + 0.3 * torch.randn(64) > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    cfg = TrainingConfig(batch_size=16, epochs=6, amp=False, scheduler="cosine",
                         warmup_epochs=2, monitor="val_roc_auc", monitor_mode="max",
                         early_stopping_patience=0, learning_rate=1e-2)
    run = RunDirectory(tmp_path / "r", "sel")
    sch = build_scheduler(opt, cfg)
    tr = _LinTrainer(model, opt, config=cfg, run_dir=run, device="cpu", scheduler=sch)
    summary = tr.fit(loader, loader, epochs=6)

    assert summary["monitor"] == "val_roc_auc"
    hist = summary["history"]
    aucs = [row["val_roc_auc"] for row in hist]
    assert summary["best_metric"] == pytest.approx(max(aucs))     # picked the MAX AUC epoch
    assert "lr" in hist[0] and hist[0]["lr"] != hist[-1]["lr"]    # scheduler stepped, lr logged
    assert len(hist) == 6                                         # patience=0 -> no early stop


# ------------------------------------------------- the final experiment config

def test_final_config_has_the_controlled_values() -> None:
    c = load_config("configs/deepfake_transformer_final.yaml")
    t, v, a, m = c.training, c.video, c.video_augment, c.model

    assert t.class_weight == "none"                       # plain cross-entropy
    assert t.learning_rate == pytest.approx(3e-4)
    assert t.scheduler == "cosine" and t.warmup_epochs == 8
    assert t.monitor == "val_roc_auc" and t.monitor_mode == "max"
    assert t.early_stopping_patience == 0                 # no early stopping

    assert v.landmarks_dir.endswith("landmarks_dense")
    assert v.num_frames == 32
    assert v.manifest_csv.endswith("manifest_dense.csv")
    assert v.regions == "face_mouth" and v.normalize == "interocular" and v.align_rotation

    assert a.coord_jitter_std == 0.0 and a.coord_jitter_prob == 0.0
    assert a.time_mask_frames == 0 and a.time_mask_prob == 0.0
    assert a.hflip_prob == 0.5 and a.scale_jitter > 0     # temporally-consistent aug kept

    assert m.visual_encoder == "transformer" and m.landmark_coords == 3
    # architecture dims unchanged from configs/deepfake_transformer.yaml
    base = load_config("configs/deepfake_transformer.yaml").model
    assert (m.visual_tf_layers, m.num_heads, m.visual_embedding_dim, m.visual_tf_ff_dim) == \
           (base.visual_tf_layers, base.num_heads, base.visual_embedding_dim, base.visual_tf_ff_dim)
