"""Tests for the independent AV evaluation framework (Phase 13B).

Covers only the pure category/summary logic. Model-loading and dataset I/O are
exercised manually against LAV-DF (see docs/experiments.md) rather than in the
fast unit test suite.
"""

from __future__ import annotations

from scripts.evaluate_av_independent import CATEGORIES, categorize, summarize_by_category


def test_categorize_real() -> None:
    assert categorize("real", False, False) == "REAL"


def test_categorize_audio_only() -> None:
    assert categorize("manipulated", True, False) == "AUDIO-ONLY"


def test_categorize_video_only() -> None:
    assert categorize("manipulated", False, True) == "VIDEO-ONLY"


def test_categorize_audio_video() -> None:
    assert categorize("manipulated", True, True) == "AUDIO+VIDEO"


def test_categorize_never_uses_manipulation_as_sync_label() -> None:
    """Category is a manipulation label, not a synchronization target.

    Regression guard: this function must never return anything resembling a
    "sync"/"desync" value, since that would silently redefine manipulation
    labels as synchronization ground truth.
    """
    for label_name, modify_audio, modify_video in [
        ("real", False, False),
        ("manipulated", True, False),
        ("manipulated", False, True),
        ("manipulated", True, True),
    ]:
        category = categorize(label_name, modify_audio, modify_video)
        assert category in CATEGORIES
        assert "sync" not in category.lower()


def test_summarize_by_category_descriptive_stats() -> None:
    records = [
        {"sample_id": "a", "category": "REAL", "aggregate_sync_score": 0.9, "n_valid_windows": 10, "n_total_windows": 10},
        {"sample_id": "b", "category": "REAL", "aggregate_sync_score": 0.8, "n_valid_windows": 10, "n_total_windows": 10},
        {"sample_id": "c", "category": "AUDIO-ONLY", "aggregate_sync_score": 0.7, "n_valid_windows": 10, "n_total_windows": 10},
    ]
    summary = summarize_by_category(records)

    assert summary["REAL"]["n"] == 2
    assert abs(summary["REAL"]["mean_sync_score"] - 0.85) < 1e-9
    assert summary["AUDIO-ONLY"]["n"] == 1
    assert summary["VIDEO-ONLY"]["n"] == 0
    assert summary["AUDIO+VIDEO"]["n"] == 0


def test_summarize_by_category_skips_none_scores() -> None:
    records = [
        {"sample_id": "a", "category": "REAL", "aggregate_sync_score": None, "n_valid_windows": 0, "n_total_windows": 10},
        {"sample_id": "b", "category": "REAL", "aggregate_sync_score": 0.5, "n_valid_windows": 10, "n_total_windows": 10},
    ]
    summary = summarize_by_category(records)

    assert summary["REAL"]["n"] == 1
    assert summary["REAL"]["mean_sync_score"] == 0.5


def test_summary_never_reports_auc_or_eer_keys() -> None:
    """The report is descriptive-only: no classification metric field names."""
    records = [
        {"sample_id": "a", "category": "REAL", "aggregate_sync_score": 0.9, "n_valid_windows": 10, "n_total_windows": 10},
    ]
    summary = summarize_by_category(records)
    for stats in summary.values():
        assert "auc" not in stats
        assert "eer" not in stats
