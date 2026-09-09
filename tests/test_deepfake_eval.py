"""Phase 7 unit tests: the deepfake evaluation assembler + figure output."""

from __future__ import annotations

import torch

from src.config import ModelConfig, TrainingConfig, VideoDataConfig
from src.data.audio_dataset import build_dataloader
from src.data.celebdf_dataset import build_celebdf_datasets, synthetic_landmark_manifest
from src.evaluation.deepfake_eval import evaluate_deepfake_model
from src.evaluation.visualization import save_spoof_evaluation_figures
from src.models.video import DeepfakeClassifier


def _model() -> DeepfakeClassifier:
    return DeepfakeClassifier(ModelConfig(
        visual_embedding_dim=32, num_heads=4, visual_embed_hidden=48,
        visual_tf_ff_dim=64, visual_tf_layers=2, visual_tf_dropout=0.0,
        spoof_head_hidden=32, dropout=0.0, visual_encoder="transformer",
    ))


def test_evaluate_deepfake_model_report_and_per_method(tmp_path) -> None:
    manifest = synthetic_landmark_manifest(
        tmp_path / "lm", n_per_split={"eval": 32}, n_ids_per_split=4, num_frames=8, seed=3,
    )
    video_cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "lm"), num_frames=8)
    ds = build_celebdf_datasets(manifest, video_cfg, splits=("eval",),
                                landmarks_dir=tmp_path / "lm")["eval"]
    train_cfg = TrainingConfig(batch_size=8, epochs=1, amp=False, num_workers=0)
    loader = build_dataloader(ds, train_cfg, shuffle=False)

    result = evaluate_deepfake_model(_model(), loader, "cpu")
    assert result.n_samples == 32
    assert result.y_score.shape == (32,)
    for key in ("accuracy", "f1", "roc_auc", "eer", "confusion_matrix"):
        assert key in result.report
    # synthetic fakes are all tagged "celebdf-fs" -> exactly one per-method row
    assert set(result.per_attack) == {"celebdf-fs"}


def test_save_figures_with_video_labels(tmp_path) -> None:
    y_true = [0, 0, 1, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.3, 0.8, 0.7, 0.4, 0.9, 0.6, 0.1]
    cm = [[3, 1], [1, 3]]
    figs = save_spoof_evaluation_figures(
        tmp_path, y_true=y_true, y_score=y_score, confusion=cm, prefix="eval",
        class_names=("fake", "real"), pos_name="real",
    )
    assert set(figs) == {"confusion_matrix", "roc_curve", "det_curve", "score_distributions"}
    for path in figs.values():
        assert path.is_file() and path.stat().st_size > 0
