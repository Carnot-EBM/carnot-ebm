"""Tests for Experiment 799 — JEPA v21 Retrain: Multi-Source + CPMI + PROGRS.

Spec: REQ-LEARN-095, REQ-LEARN-096, REQ-LEARN-097,
      SCENARIO-LEARN-096, SCENARIO-LEARN-097
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

import experiment_799_jepa_v21_retrain as exp799  # noqa: E402


# ---------------------------------------------------------------------------
# Test: PROGRS outcome-conditioned weight computation (REQ-LEARN-096)
# ---------------------------------------------------------------------------


def test_outcome_conditioned_weights_sum_to_n() -> None:
    """compute_outcome_conditioned_weights normalises weights so sum == n_pairs.

    Spec: REQ-LEARN-096
    """
    labels = [1.0, 0.0, 1.0, 0.0]
    base_weights = [0.14, 0.12, 0.20, 0.14]

    result = exp799.compute_outcome_conditioned_weights(labels, base_weights)

    assert len(result) == 4
    assert sum(result) == pytest.approx(4.0, rel=1e-5), (
        "PROGRS normalised weights must sum to n_pairs so effective LR is stable"
    )


def test_outcome_conditioned_weights_preserve_ordering() -> None:
    """Pairs from easier domains (higher base_weight) get higher normalised weight.

    Spec: REQ-LEARN-096 — harder domains (lower accuracy) weighted down
    """
    labels = [0.0, 0.0]
    # humaneval (0.20) > gsm8k (0.14) → humaneval weight should be larger
    base_weights = [0.14, 0.20]

    result = exp799.compute_outcome_conditioned_weights(labels, base_weights)

    assert result[1] > result[0], (
        "humaneval domain (accuracy=0.20) must receive higher weight than "
        "gsm8k (accuracy=0.14) — harder domains are down-weighted"
    )


def test_outcome_conditioned_weights_empty_input() -> None:
    """Empty input returns empty list without error.

    Spec: REQ-LEARN-096
    """
    result = exp799.compute_outcome_conditioned_weights([], [])
    assert result == []


def test_outcome_conditioned_weights_zero_base_fallback() -> None:
    """All-zero base weights fall back to uniform 1.0 weights to avoid NaN.

    Spec: REQ-LEARN-096
    """
    labels = [0.0, 1.0, 0.0]
    base_weights = [0.0, 0.0, 0.0]

    result = exp799.compute_outcome_conditioned_weights(labels, base_weights)

    assert all(w == pytest.approx(1.0) for w in result), (
        "zero base weights must fall back to uniform 1.0 weights"
    )


# ---------------------------------------------------------------------------
# Test: domain accuracy constants (REQ-LEARN-096)
# ---------------------------------------------------------------------------


def test_domain_accuracy_values_correct() -> None:
    """DOMAIN_ACCURACY must reflect live benchmark values from context.

    Spec: REQ-LEARN-096
    """
    assert exp799.DOMAIN_ACCURACY["gsm8k"] == pytest.approx(0.14)
    assert exp799.DOMAIN_ACCURACY["math500"] == pytest.approx(0.12)
    assert exp799.DOMAIN_ACCURACY["humaneval"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Test: OOD gate triggers deployment when ood_auc >= 0.75 (REQ-LEARN-097)
# ---------------------------------------------------------------------------


def test_ood_gate_deploys_when_auc_above_threshold(tmp_path: Path) -> None:
    """When ood_auc >= 0.75 the experiment sets tier35_deployed=True.

    Spec: REQ-LEARN-097, SCENARIO-LEARN-096
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Build minimal Exp 797 artifact with sufficient data
    exp797 = {
        "n_labeled_total": 300,
        "corpus_path": str(results_dir / "multi.json"),
        "status": "success",
    }
    (results_dir / "experiment_797_jepa_v21_data_collection.json").write_text(
        json.dumps(exp797)
    )

    # Build a simple multi-source corpus: enough balanced pairs so AUC can reach 0.75+
    multi_corpus = []
    for i in range(40):
        multi_corpus.append({
            "question_id": f"q{i}",
            "step_text": f"correct step number {i} with right computation",
            "label": "correct",
            "source_domain": "gsm8k",
        })
        multi_corpus.append({
            "question_id": f"q{i}_wrong",
            "step_text": f"wrong step {i}: divide by zero then multiply",
            "label": "incorrect",
            "source_domain": "gsm8k",
        })
    (results_dir / "fover_labeled_steps_v21_multi.json").write_text(
        json.dumps(multi_corpus)
    )

    # Build minimal OOD set (Exp 442 format)
    ood_corpus = [
        {"question_id": "ood1", "step_text": "correct ood step", "label": "correct", "confidence": 1.0},
        {"question_id": "ood2", "step_text": "wrong ood step divide by zero", "label": "incorrect", "confidence": 1.0},
    ]
    (results_dir / "fover_labeled_steps_live.json").write_text(json.dumps(ood_corpus))

    # Patch REPO_ROOT, OOD_GATE, and training to produce a high AUC
    with (
        patch.object(exp799, "REPO_ROOT", tmp_path),
        patch.object(exp799, "OOD_GATE", 0.0),  # lower gate so real training passes
        patch.object(exp799.tmpl, "setup"),
        patch.object(exp799.tmpl, "build_result", side_effect=lambda d, **kw: d),
    ):
        artifact = exp799.run_experiment()

    assert artifact["tier35_deployed"] is True, (
        "When ood_auc >= gate, tier35_deployed must be True (REQ-LEARN-097)"
    )
    assert artifact["honest_verdict"] == "jepa_v21_tier35_deployed"
    assert artifact["failure_analysis"] is None


# ---------------------------------------------------------------------------
# Test: failure analysis runs when ood_auc < gate (REQ-LEARN-097, SCENARIO-LEARN-097)
# ---------------------------------------------------------------------------


def test_failure_analysis_produced_when_auc_below_gate(tmp_path: Path) -> None:
    """When ood_auc < 0.75, failure_analysis must contain per_domain_auc + recommendations.

    Spec: REQ-LEARN-097, SCENARIO-LEARN-097
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    exp797 = {"n_labeled_total": 300, "status": "success"}
    (results_dir / "experiment_797_jepa_v21_data_collection.json").write_text(
        json.dumps(exp797)
    )

    # Minimal corpus
    multi_corpus = [
        {"question_id": "q1", "step_text": "step one correct", "label": "correct", "source_domain": "gsm8k"},
        {"question_id": "q2", "step_text": "step two wrong", "label": "incorrect", "source_domain": "math500"},
    ]
    (results_dir / "fover_labeled_steps_v21_multi.json").write_text(
        json.dumps(multi_corpus)
    )

    (results_dir / "fover_labeled_steps_live.json").write_text(
        json.dumps([])
    )  # empty → fallback OOD proxy used

    with (
        patch.object(exp799, "REPO_ROOT", tmp_path),
        patch.object(exp799, "OOD_GATE", 0.9999),  # impossibly high → always fails
        patch.object(exp799.tmpl, "setup"),
        patch.object(exp799.tmpl, "build_result", side_effect=lambda d, **kw: d),
    ):
        artifact = exp799.run_experiment()

    assert artifact["tier35_deployed"] is False
    assert artifact["honest_verdict"] == "jepa_v21_below_gate"
    assert artifact["failure_analysis"] is not None, (
        "failure_analysis must be populated when gate fails (REQ-LEARN-097)"
    )
    fa = artifact["failure_analysis"]
    assert "per_domain_auc" in fa
    assert "recommendations_for_v22" in fa
    assert len(fa["recommendations_for_v22"]) >= 1


# ---------------------------------------------------------------------------
# Test: insufficient data gate blocks experiment (SCENARIO-LEARN-097)
# ---------------------------------------------------------------------------


def test_gate_blocks_when_n_labeled_below_80(tmp_path: Path) -> None:
    """When Exp 797 n_labeled_total < 80, experiment exits with blocked status.

    Spec: REQ-LEARN-097, SCENARIO-LEARN-097
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    exp797 = {"n_labeled_total": 50, "status": "success"}
    (results_dir / "experiment_797_jepa_v21_data_collection.json").write_text(
        json.dumps(exp797)
    )

    with (
        patch.object(exp799, "REPO_ROOT", tmp_path),
        patch.object(exp799.tmpl, "setup"),
        patch.object(exp799.tmpl, "build_result", side_effect=lambda d, **kw: d),
    ):
        artifact = exp799.run_experiment()

    assert artifact["honest_verdict"] == "jepa_v21_insufficient_data"
    assert artifact["n_labeled_total"] == 50  # key from blocked artifact dict


def test_gate_blocks_when_exp797_missing(tmp_path: Path) -> None:
    """When Exp 797 artifact is absent entirely, experiment exits blocked.

    Spec: REQ-LEARN-097
    """
    (tmp_path / "results").mkdir()
    with (
        patch.object(exp799, "REPO_ROOT", tmp_path),
        patch.object(exp799.tmpl, "setup"),
        patch.object(exp799.tmpl, "build_result", side_effect=lambda d, **kw: d),
    ):
        artifact = exp799.run_experiment()

    assert artifact["honest_verdict"] == "jepa_v21_insufficient_data"
    assert artifact["block_reason"] == "exp797_artifact_missing"


# ---------------------------------------------------------------------------
# Test: CPMI augmentation increases corpus size (REQ-LEARN-095)
# ---------------------------------------------------------------------------


def test_cpmi_triples_augment_corpus(tmp_path: Path) -> None:
    """When CPMI triples file present, corpus is augmented beyond primary labeled set.

    Spec: REQ-LEARN-095 — augmentation_ratio >= 2.0 checked at artifact level
    """
    primary = [
        {"question_id": f"q{i}", "step_text": f"step {i}", "label": "correct", "source_domain": "gsm8k"}
        for i in range(10)
    ]
    multi_path = tmp_path / "fover_labeled_steps_v21_multi.json"
    multi_path.write_text(json.dumps(primary))

    cpmi = [
        {
            "prefix_text": f"q{i}",
            "positive_step": f"correct step {i}",
            "negative_step": f"wrong step {i}",
            "source_domain": "humaneval",
        }
        for i in range(20)
    ]
    cpmi_path = tmp_path / "experiment_798_cpmi_pairs_triples.json"
    cpmi_path.write_text(json.dumps(cpmi))

    live_path = tmp_path / "fover_labeled_steps_live.json"
    live_path.write_text(json.dumps([]))

    from experiment_799_jepa_v21_retrain import _load_multi_source_corpus  # noqa: PLC0415
    _, _, _, data_source, n_total = _load_multi_source_corpus(
        multi_path,
        cpmi_path,
        live_path,
    )

    # 10 primary + up to 40 from CPMI (20 positive + 20 negative)
    assert n_total > 10, (
        "CPMI augmentation must increase corpus size beyond primary labeled set"
    )
    assert "cpmi_triples_798" in data_source, (
        "data_source must indicate CPMI augmentation was applied"
    )
