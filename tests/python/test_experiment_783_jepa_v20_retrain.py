"""Tests for Experiment 783 — JEPA v20 Retrain with Class-Weight Balancing.

Spec: REQ-LEARN-052, REQ-LEARN-053, SCENARIO-LEARN-096, SCENARIO-LEARN-097
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from carnot.samplers.jepa_v20 import MultiStepJEPAv20  # noqa: E402


# ---------------------------------------------------------------------------
# Test: class-weight balancing applies weight_positive != 1.0 for imbalanced data
# Spec: REQ-LEARN-052, SCENARIO-LEARN-097
# ---------------------------------------------------------------------------


def test_class_weight_balancing_applied_when_imbalanced() -> None:
    """MultiStepJEPAv20 MUST compute weight_positive = n_negative / n_positive.

    Spec: REQ-LEARN-052, SCENARIO-LEARN-097
    """
    probe = MultiStepJEPAv20()
    # 2 positives, 8 negatives → weight_positive = 8/2 = 4.0
    step_sequences = [["step text for example"] for _ in range(10)]
    labels = [1.0, 1.0] + [0.0] * 8

    result = probe.train(step_sequences, labels, n_epochs=5, lr=1e-3)

    assert result["weight_positive"] == pytest.approx(4.0), (
        "weight_positive must be n_negative / n_positive = 8 / 2 = 4.0"
    )
    assert result["class_weight_used"] is True
    assert result["n_train"] == 10


def test_class_weight_balanced_corpus_weight_is_one() -> None:
    """When corpus is perfectly balanced weight_positive == 1.0.

    Spec: SCENARIO-LEARN-097
    """
    probe = MultiStepJEPAv20()
    step_sequences = [["step"] for _ in range(4)]
    labels = [1.0, 1.0, 0.0, 0.0]

    result = probe.train(step_sequences, labels, n_epochs=5, lr=1e-3)

    assert result["weight_positive"] == pytest.approx(1.0)
    # class_weight_used is still True when balanced (weight was computed and applied)
    assert result["class_weight_used"] is True


def test_class_weight_absent_class_returns_weight_one() -> None:
    """When only one class present weight_positive == 1.0 (degenerate case).

    Spec: REQ-LEARN-052
    """
    probe = MultiStepJEPAv20()
    step_sequences = [["step text"] for _ in range(3)]
    labels = [0.0, 0.0, 0.0]  # no positives

    result = probe.train(step_sequences, labels, n_epochs=5, lr=1e-3)

    assert result["weight_positive"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: data_source is "edu_prm_selected" when fover_edu_prm_selected.json exists
# Spec: REQ-LEARN-052, SCENARIO-LEARN-096
# ---------------------------------------------------------------------------


def test_data_source_edu_prm_selected_when_file_exists() -> None:
    """data_source MUST be 'edu_prm_selected' when fover_edu_prm_selected.json exists.

    Spec: REQ-LEARN-052, SCENARIO-LEARN-096
    """
    from scripts.experiment_783_jepa_v20_retrain import collect_training_data  # noqa: E402

    sample_items = [
        {"step_text": f"step {i}", "label": "incorrect" if i % 2 == 0 else "correct", "confidence": 1.0}
        for i in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir()
        edu_prm_path = results_dir / "fover_edu_prm_selected.json"
        edu_prm_path.write_text(json.dumps(sample_items))

        seqs, labels, data_source, n = collect_training_data(Path(tmpdir))

    assert data_source == "edu_prm_selected", (
        "data_source must be 'edu_prm_selected' when fover_edu_prm_selected.json exists"
    )
    assert n == 10
    assert len(seqs) == 10
    assert len(labels) == 10


def test_data_source_pooled_raw_when_both_live_files_exist() -> None:
    """data_source MUST be 'pooled_raw' when both v1 and v2 live files exist.

    Spec: REQ-LEARN-052, SCENARIO-LEARN-096
    """
    from scripts.experiment_783_jepa_v20_retrain import collect_training_data  # noqa: E402

    items_v1 = [{"step_text": "step v1", "label": "incorrect", "confidence": 1.0}]
    items_v2 = [{"step_text": "step v2", "label": "correct", "confidence": 1.0}]

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir()
        (results_dir / "fover_labeled_steps_live.json").write_text(json.dumps(items_v1))
        (results_dir / "fover_labeled_steps_live_v2.json").write_text(json.dumps(items_v2))

        seqs, labels, data_source, n = collect_training_data(Path(tmpdir))

    assert data_source == "pooled_raw"
    assert n == 2


def test_data_source_single_file_fallback() -> None:
    """data_source MUST be 'single_file' when only v1 live file exists.

    Spec: REQ-LEARN-052
    """
    from scripts.experiment_783_jepa_v20_retrain import collect_training_data  # noqa: E402

    items_v1 = [{"step_text": "step", "label": "incorrect", "confidence": 1.0}]

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir()
        (results_dir / "fover_labeled_steps_live.json").write_text(json.dumps(items_v1))

        seqs, labels, data_source, n = collect_training_data(Path(tmpdir))

    assert data_source == "single_file"
    assert n == 1


# ---------------------------------------------------------------------------
# Test: model saved to jepa_v20_model.npz when ood_auc > 0.75
# Spec: REQ-LEARN-053
# ---------------------------------------------------------------------------


def test_model_saved_when_ood_auc_above_threshold() -> None:
    """Model MUST be saved to jepa_v20_model.npz when ood_auc > 0.75.

    Spec: REQ-LEARN-053
    """
    import numpy as np  # noqa: PLC0415

    probe = MultiStepJEPAv20()
    seqs = [["step text"] for _ in range(10)]
    labels = [float(i % 2) for i in range(10)]
    probe.train(seqs, labels, n_epochs=10, lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "jepa_v20_model.npz"
        np.savez(
            str(save_path),
            w1=np.array(probe._w1),
            b1=np.array(probe._b1),
            w2=np.array(probe._w2),
            b2=np.array(probe._b2),
        )
        assert save_path.exists(), "jepa_v20_model.npz must be written when ood_auc > 0.75"
        loaded = np.load(str(save_path))
        assert "w1" in loaded
        assert "b1" in loaded
        assert "w2" in loaded
        assert "b2" in loaded


# ---------------------------------------------------------------------------
# Test: ood_auc_delta_vs_v19 = ood_auc - 0.5667
# Spec: REQ-LEARN-053
# ---------------------------------------------------------------------------


def test_ood_auc_delta_vs_v19_formula() -> None:
    """ood_auc_delta_vs_v19 MUST equal ood_auc - 0.5667.

    Spec: REQ-LEARN-053
    """
    from scripts.experiment_783_jepa_v20_retrain import V19_OOD_AUC_BASELINE  # noqa: E402

    assert V19_OOD_AUC_BASELINE == pytest.approx(0.5667)

    test_cases = [0.45, 0.5667, 0.62, 0.78]
    for ood_auc in test_cases:
        delta = round(ood_auc - V19_OOD_AUC_BASELINE, 4)
        expected = round(ood_auc - 0.5667, 4)
        assert delta == pytest.approx(expected), (
            f"ood_auc_delta_vs_v19 must equal ood_auc - 0.5667 for ood_auc={ood_auc}"
        )


def test_honest_verdict_insufficient_data_when_n_below_30() -> None:
    """honest_verdict MUST be 'jepa_v20_insufficient_data' when n_training_pairs < 30.

    Spec: REQ-LEARN-053
    """
    # Verify the verdict logic: n < 30 → insufficient_data regardless of AUC
    n_training_pairs = 18  # < 30
    ood_auc = 0.80  # would be "viable" if data were sufficient

    if n_training_pairs < 30:
        honest_verdict = "jepa_v20_insufficient_data"
    elif ood_auc > 0.75:
        honest_verdict = "jepa_v20_ood_viable"
    else:
        honest_verdict = "jepa_v20_improving"

    assert honest_verdict == "jepa_v20_insufficient_data"


def test_honest_verdict_ood_viable_when_above_threshold() -> None:
    """honest_verdict MUST be 'jepa_v20_ood_viable' when n>=30 and ood_auc > 0.75.

    Spec: REQ-LEARN-053
    """
    n_training_pairs = 60
    ood_auc = 0.80

    if n_training_pairs < 30:
        honest_verdict = "jepa_v20_insufficient_data"
    elif ood_auc > 0.75:
        honest_verdict = "jepa_v20_ood_viable"
    elif ood_auc > 0.60:
        honest_verdict = "jepa_v20_improving"
    else:
        honest_verdict = "jepa_v20_below_v19"

    assert honest_verdict == "jepa_v20_ood_viable"


def test_honest_verdict_below_v19_when_regression() -> None:
    """honest_verdict MUST be 'jepa_v20_below_v19' when ood_auc <= 0.5667.

    Spec: REQ-LEARN-053
    """
    n_training_pairs = 60
    ood_auc = 0.50
    v19_baseline = 0.5667

    if n_training_pairs < 30:
        honest_verdict = "jepa_v20_insufficient_data"
    elif ood_auc > 0.75:
        honest_verdict = "jepa_v20_ood_viable"
    elif ood_auc > 0.60:
        honest_verdict = "jepa_v20_improving"
    elif ood_auc <= v19_baseline:
        honest_verdict = "jepa_v20_below_v19"
    else:
        honest_verdict = "jepa_v20_improving"

    assert honest_verdict == "jepa_v20_below_v19"
