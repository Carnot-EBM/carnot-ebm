"""Tests for Exp 1603 EBCN structural violation scorer.

Spec: REQ-VERIFY-1603, SCENARIO-VERIFY-1603.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import ebcn_scorer as exp


def test_req_verify_1603_dual_head_attention_scores_hidden_states() -> None:
    """REQ-VERIFY-1603: hidden-state scoring uses two normalized attention heads."""

    scorer = exp.EBCNScorer()
    case = exp.synthetic_logical_cases()[0]
    score = scorer.score_hidden_states(case.hidden_states, metadata=case.metadata)

    assert score.energy >= 0.0
    assert score.support_energy >= 0.0
    assert score.contradiction_energy >= 0.0
    assert np.isclose(sum(score.support_attention), 1.0)
    assert np.isclose(sum(score.contradiction_attention), 1.0)
    assert score.head_count == 2
    assert score.dual_head_attention_used is True
    assert score.autoregressive_generation_used is False


def test_scenario_verify_1603_synthetic_contradictions_have_higher_energy() -> None:
    """SCENARIO-VERIFY-1603: direct contradictions score above consistent traces."""

    metrics = exp.evaluate_synthetic_logical_contradictions()

    assert metrics["synthetic_cases_total"] >= 4
    assert metrics["contradiction_cases"] >= 2
    assert metrics["consistent_cases"] >= 2
    assert metrics["contradiction_mean_energy"] > metrics["consistent_mean_energy"]
    assert metrics["energy_gap"] == pytest.approx(
        metrics["contradiction_mean_energy"] - metrics["consistent_mean_energy"]
    )
    assert metrics["dual_head_attention_used"] is True
    assert metrics["autoregressive_generation_used"] is False
    assert metrics["false_accept_rate"] == pytest.approx(0.0)


def test_req_verify_1603_metadata_detects_contradictory_claim_pairs() -> None:
    """REQ-VERIFY-1603: proposition metadata raises scalar violation energy."""

    scorer = exp.EBCNScorer()
    consistent = exp.logical_trace_to_hidden_states(
        [
            ("alpha", True),
            ("beta", False),
            ("alpha", True),
        ]
    )
    contradictory = exp.logical_trace_to_hidden_states(
        [
            ("alpha", True),
            ("beta", False),
            ("alpha", False),
        ]
    )

    consistent_score = scorer.score_hidden_states(
        consistent.hidden_states,
        metadata=consistent.metadata,
    )
    contradictory_score = scorer.score_hidden_states(
        contradictory.hidden_states,
        metadata=contradictory.metadata,
    )

    assert contradictory_score.contradiction_pairs == [("alpha", 0, 2)]
    assert contradictory_score.energy > consistent_score.energy
    assert contradictory_score.contradiction_energy > consistent_score.contradiction_energy


def test_req_verify_1603_rejects_invalid_hidden_state_shape() -> None:
    """REQ-VERIFY-1603: scorer accepts only non-empty two-dimensional hidden states."""

    scorer = exp.EBCNScorer()

    with pytest.raises(ValueError, match="2D"):
        scorer.score_hidden_states(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    with pytest.raises(ValueError, match="non-empty"):
        scorer.score_hidden_states(np.empty((0, 4), dtype=np.float32))


def test_scenario_verify_1603_runner_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1603: runner writes required artifact fields and metrics."""

    output_path = tmp_path / "experiment_1603_ebcn.json"
    artifact = exp.run_experiment_1603_ebcn(
        output_path=output_path,
        run_date="20260509",
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1603
    assert artifact["ebcn_scorer_ready"] is True
    assert artifact["dual_head_attention_used"] is True
    assert artifact["autoregressive_generation_used"] is False
    assert artifact["hidden_state_source"] == "deterministic_synthetic_logical_trace_encoder"
    assert artifact["energy_gap"] > 0.0
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1603_bootstrap_and_cli_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1603: bootstrap artifact, empty metrics, and CLI stay deterministic."""

    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap = exp.write_in_progress_artifact(bootstrap_path, run_date="20260509")
    assert json.loads(bootstrap_path.read_text(encoding="utf-8")) == bootstrap
    assert bootstrap["status"] == "in_progress"
    assert bootstrap["ebcn_scorer_ready"] is False

    empty_metrics = exp.aggregate_case_scores([])
    assert empty_metrics["synthetic_cases_total"] == 0
    assert empty_metrics["ebcn_scorer_ready"] is False
    assert empty_metrics["energy_gap"] == 0.0

    cli_output = tmp_path / "cli_experiment_1603.json"
    rc = exp.main(["--output", str(cli_output), "--run-date", "20260509"])
    assert rc == 0
    assert "ready=True" in capsys.readouterr().out
