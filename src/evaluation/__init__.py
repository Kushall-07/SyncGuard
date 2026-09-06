"""Evaluation metrics, figures and the spoof-model evaluation orchestrator."""

from src.evaluation.metrics import (
    accuracy,
    binary_classification_report,
    confusion_matrix,
    det_points,
    equal_error_rate,
    evaluate_logits,
    per_attack_eer,
    precision_recall_f1,
    roc_auc,
    roc_points,
)
from src.evaluation.spoof_eval import SpoofEvalResult, evaluate_spoof_model
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_det_curve,
    plot_roc_curve,
    plot_score_distributions,
    plot_training_curves,
    save_spoof_evaluation_figures,
)

__all__ = [
    "accuracy",
    "precision_recall_f1",
    "roc_auc",
    "equal_error_rate",
    "confusion_matrix",
    "roc_points",
    "det_points",
    "per_attack_eer",
    "evaluate_logits",
    "binary_classification_report",
    "SpoofEvalResult",
    "evaluate_spoof_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_det_curve",
    "plot_score_distributions",
    "plot_training_curves",
    "save_spoof_evaluation_figures",
]
