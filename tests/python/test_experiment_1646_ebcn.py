"""Tests for Exp 1646 EBCN reasoning-trace coherence prototype.

Spec: REQ-VERIFY-1646, SCENARIO-VERIFY-1646.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1646_ebcn as exp


def test_req_verify_1646_dual_head_state_space_scores_reasoning_traces() -> None:
    """REQ-VERIFY-1646: EBCN scores reasoning traces with dual heads and rollout."""

    scorer = exp.ReasoningTraceEBCN()
    cases = exp.default_reasoning_trace_cases()
    consistent = next(case for case in cases if case.expected_inconsistent is False)
    inconsistent = next(case for case in cases if case.expected_inconsistent is True)

    consistent_score = scorer.score_case(consistent)
    inconsistent_score = scorer.score_case(inconsistent)

    assert consistent_score.energy >= 0.0
    assert inconsistent_score.energy > consistent_score.energy
    assert 0.0 <= consistent_score.coherence_score <= 1.0
    assert 0.0 <= inconsistent_score.coherence_score <= 1.0
    assert inconsistent_score.coherence_score < consistent_score.coherence_score
    assert inconsistent_score.dual_head_attention_used is True
    assert inconsistent_score.state_space_transition_used is True
    assert inconsistent_score.autoregressive_generation_used is False
    assert inconsistent_score.contradiction_pairs


def test_scenario_verify_1646_coherence_accuracy_separates_inconsistencies() -> None:
    """SCENARIO-VERIFY-1646: contradictions have lower coherence scores."""

    artifact = exp.build_artifact(tests_run=["focused pytest"])

    assert artifact["status"] == "complete"
    assert artifact["ebcn_prototype_ready"] is True
    assert artifact["dual_head_attention_used"] is True
    assert artifact["state_space_transition_used"] is True
    assert artifact["autoregressive_generation_used"] is False
    assert artifact["inconsistent_mean_energy"] > artifact["consistent_mean_energy"]
    assert artifact["energy_gap"] == pytest.approx(
        artifact["inconsistent_mean_energy"] - artifact["consistent_mean_energy"]
    )
    assert artifact["coherence_score_accuracy"] >= 0.8
    assert artifact["spec_traces"] == ["REQ-VERIFY-1646", "SCENARIO-VERIFY-1646"]
    assert artifact["tests_run"] == ["focused pytest"]


def test_req_verify_1646_runner_writes_required_json(tmp_path: Path) -> None:
    """REQ-VERIFY-1646: runner writes the required terminal artifact."""

    output_path = tmp_path / "experiment_1646_ebcn.json"
    artifact = exp.run_experiment(
        output_path=output_path,
        run_date="20260509",
        tests_run=["test_req_verify_1646_runner_writes_required_json"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["experiment_id"] == 1646
    assert artifact["hidden_state_source"] == "deterministic_reasoning_trace_state_space"
    assert artifact["coherence_score_accuracy"] == pytest.approx(1.0)
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1646_validation_rejects_missing_and_inconsistent_fields() -> None:
    """REQ-VERIFY-1646: artifact validation catches schema and gate drift."""

    artifact = exp.build_artifact()
    missing_field = dict(artifact)
    del missing_field["coherence_score_accuracy"]
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    gate_drift = dict(artifact)
    gate_drift["coherence_score_accuracy"] = 0.0
    with pytest.raises(AssertionError, match="accuracy"):
        exp.validate_artifact(gate_drift)


def test_req_verify_1646_empty_cases_invalid_steps_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1646: empty/invalid inputs fail closed and CLI writes output."""

    partial = exp.build_artifact(cases=[])
    assert partial["status"] == "blocked"
    assert partial["reasoning_trace_cases_total"] == 0
    assert partial["coherence_score_accuracy"] == 0.0

    scorer = exp.ReasoningTraceEBCN()
    invalid_case = exp.ReasoningTraceCase(
        case_id="empty",
        expected_inconsistent=False,
        steps=(),
    )
    with pytest.raises(ValueError, match="at least one step"):
        scorer.score_case(invalid_case)

    output_path = tmp_path / "cli_experiment_1646.json"
    rc = exp.main(["--output", str(output_path), "--run-date", "20260509"])
    assert rc == 0
    assert "coherence_score_accuracy=" in capsys.readouterr().out
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "complete"
