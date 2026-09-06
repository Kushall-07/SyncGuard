"""Phase 6 unit tests: curve metrics, per-attack EER, figures, model evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.asvspoof_dataset import make_asvspoof_datasets
from src.data.audio_dataset import build_dataloader
from src.data.manifests import synthetic_manifest
from src.evaluation.metrics import (
    det_points,
    equal_error_rate,
    evaluate_logits,
    per_attack_eer,
    roc_points,
)
from src.evaluation.spoof_eval import evaluate_spoof_model, loader_preserves_order
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_det_curve,
    plot_roc_curve,
    plot_score_distributions,
    plot_training_curves,
    save_spoof_evaluation_figures,
)
from src.models.audio import SpoofClassifier

CFG = load_config("configs/spoof_cnn.yaml")


@pytest.fixture(scope="module")
def separable_scores():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 150 + [1] * 150)
    y_score = np.concatenate([rng.normal(0.3, 0.1, 150), rng.normal(0.7, 0.1, 150)]).clip(0, 1)
    return y_true, y_score


# --------------------------------------------------------------------- curve data


def test_det_points_relates_to_roc(separable_scores) -> None:
    y_true, y_score = separable_scores
    fpr_r, tpr_r, _ = roc_points(y_true, y_score)
    fpr_d, fnr_d, _ = det_points(y_true, y_score)
    assert np.allclose(fpr_r, fpr_d)
    assert np.allclose(fnr_d, 1.0 - tpr_r)


def test_eer_lies_on_det_curve(separable_scores) -> None:
    y_true, y_score = separable_scores
    eer, _ = equal_error_rate(y_true, y_score)
    fpr, fnr, _ = det_points(y_true, y_score)
    gap = np.abs(fpr - fnr)
    assert abs(eer - (fpr[gap.argmin()] + fnr[gap.argmin()]) / 2) < 1e-9
    assert 0.0 <= eer < 0.2


# ------------------------------------------------------------------- per-attack


def test_per_attack_eer_ranks_systems() -> None:
    rng = np.random.default_rng(1)
    bona = rng.normal(0.8, 0.05, 100)
    easy = rng.normal(0.2, 0.05, 50)     # A_easy: well separated -> low EER
    hard = rng.normal(0.78, 0.05, 50)    # A_hard: overlaps bonafide -> high EER
    y_score = np.concatenate([bona, easy, hard])
    y_true = np.array([1] * 100 + [0] * 100)
    attacks = ["-"] * 100 + ["A_easy"] * 50 + ["A_hard"] * 50

    out = per_attack_eer(attacks, y_true, y_score)
    assert set(out) == {"A_easy", "A_hard"}
    assert out["A_easy"]["n_spoof"] == 50
    assert out["A_easy"]["eer"] < out["A_hard"]["eer"]


def test_per_attack_eer_ignores_bonafide_tag() -> None:
    out = per_attack_eer(["-", "-", "A01", "A01"], [1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
    assert list(out) == ["A01"]


# --------------------------------------------------------------------- from logits


def test_evaluate_logits_matches_manual_softmax() -> None:
    # argmax preds: [0, 1, 1, 0]  (row 3 is a tie -> index 0)
    logits = torch.tensor([[2.0, -1.0], [-0.5, 0.5], [0.0, 3.0], [1.0, 1.0]])
    y_true = [0, 1, 1, 1]  # last one wrong -> 3/4 correct
    report = evaluate_logits(logits, y_true)
    assert {"accuracy", "f1", "roc_auc", "eer", "confusion_matrix"} <= set(report)
    assert report["accuracy"] == pytest.approx(0.75)
    # true=[0,1,1,1], pred=[0,1,1,0] -> [[TN, FP], [FN, TP]]
    assert report["confusion_matrix"] == [[1, 0], [1, 2]]


def test_evaluate_logits_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        evaluate_logits(torch.randn(4, 3), [0, 1, 0, 1])


# ------------------------------------------------------------------------ figures


def test_individual_plots_write_files(tmp_path, separable_scores) -> None:
    y_true, y_score = separable_scores
    paths = [
        plot_confusion_matrix([[80, 20], [10, 90]], tmp_path / "cm.png"),
        plot_roc_curve(y_true, y_score, tmp_path / "roc.png"),
        plot_det_curve(y_true, y_score, tmp_path / "det.png"),
        plot_score_distributions(y_true, y_score, tmp_path / "dist.png"),
    ]
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0


def test_save_spoof_evaluation_figures_returns_four(tmp_path, separable_scores) -> None:
    y_true, y_score = separable_scores
    figs = save_spoof_evaluation_figures(
        tmp_path, y_true=y_true, y_score=y_score,
        confusion=[[70, 30], [25, 75]], prefix="eval",
    )
    assert set(figs) == {"confusion_matrix", "roc_curve", "det_curve", "score_distributions"}
    assert all(p.is_file() for p in figs.values())
    assert (tmp_path / "eval_roc_curve.png").is_file()


def test_plot_training_curves_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "epoch,train_loss,val_loss,val_eer,val_roc_auc\n"
        "1,0.9,1.0,0.4,0.6\n2,0.6,0.7,0.3,0.75\n3,0.4,0.55,0.2,0.85\n",
        encoding="utf-8",
    )
    out = plot_training_curves(csv_path, tmp_path / "curves.png")
    assert out.is_file() and out.stat().st_size > 0


# ------------------------------------------------------------------ model eval


@pytest.fixture(scope="module")
def tiny_eval_setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval_audio")
    manifest = synthetic_manifest(
        out, n_per_split={"eval": 32}, n_speakers_per_split=4,
        seed=3, sample_rate=CFG.audio.sample_rate, duration_s=1.0,
    )
    datasets = make_asvspoof_datasets(manifest, CFG.audio, feature="logmel",
                                      fixed_seconds=1.0, splits=("eval",))
    model = SpoofClassifier(CFG.model, n_mels=CFG.audio.mel.n_mels)
    return model, datasets["eval"]


def test_evaluate_spoof_model_shapes_and_scores(tiny_eval_setup) -> None:
    model, ds = tiny_eval_setup
    loader = build_dataloader(ds, CFG.training, shuffle=False)
    result = evaluate_spoof_model(model, loader, "cpu")
    assert result.n_samples == len(ds) == 32
    assert result.y_score.shape == result.y_true.shape == (32,)
    assert ((0.0 <= result.y_score) & (result.y_score <= 1.0)).all()
    assert {"accuracy", "eer", "confusion_matrix"} <= set(result.report)
    # synthetic manifest tags spoof rows SYN00.. -> per-attack breakdown present
    assert result.per_attack and all("eer" in v for v in result.per_attack.values())


def test_loader_preserves_order_flag(tiny_eval_setup) -> None:
    _, ds = tiny_eval_setup
    assert loader_preserves_order(build_dataloader(ds, CFG.training, shuffle=False))
    assert not loader_preserves_order(build_dataloader(ds, CFG.training, shuffle=True))
