"""Tests for the Exp 1656 EBRM extracted logical trace scorer.

Spec: REQ-VERIFY-1656, SCENARIO-VERIFY-1656.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.models.ebrm_scorer import (
    EBRMTraceScorer,
    EBRMTraceScorerConfig,
    LogicalTrace,
    LogicalTraceStep,
    REQUIRED_ARTIFACT_FIELDS,
    SPEC_TRACES,
    build_artifact,
    default_synthetic_traces,
    validate_artifact,
    write_artifact,
)


def _coherent_trace() -> LogicalTrace:
    return LogicalTrace(
        trace_id="coherent",
        expected_inconsistent=False,
        steps=(
            LogicalTraceStep(
                step_id="s1",
                proposition="inventory_total_is_five",
                truth_value=True,
                confidence=0.94,
                constraint_ids=("counting",),
            ),
            LogicalTraceStep(
                step_id="s2",
                proposition="answer_uses_inventory_total",
                truth_value=True,
                confidence=0.91,
                supports=("s1",),
                constraint_ids=("counting", "answer_grounding"),
            ),
        ),
    )


def _contradictory_trace() -> LogicalTrace:
    return LogicalTrace(
        trace_id="contradictory",
        expected_inconsistent=True,
        steps=(
            LogicalTraceStep(
                step_id="s1",
                proposition="inventory_total_is_five",
                truth_value=True,
                confidence=0.94,
                constraint_ids=("counting",),
            ),
            LogicalTraceStep(
                step_id="s2",
                proposition="inventory_total_is_five",
                truth_value=False,
                confidence=0.93,
                contradicts=("s1",),
                constraint_ids=("counting",),
            ),
        ),
    )


def test_req_verify_1656_scores_are_continuous_and_explain_components() -> None:
    """REQ-VERIFY-1656: coherent traces score lower than direct contradictions."""

    scorer = EBRMTraceScorer()

    coherent = scorer.score_trace(_coherent_trace())
    contradictory = scorer.score_trace(_contradictory_trace())

    assert coherent.energy >= 0.0
    assert contradictory.energy > coherent.energy
    assert 0.0 <= coherent.coherence_score <= 1.0
    assert 0.0 <= contradictory.coherence_score <= 1.0
    assert contradictory.coherence_score < coherent.coherence_score
    assert contradictory.contradiction_energy > coherent.contradiction_energy
    assert contradictory.component_energies["contradiction_energy"] == pytest.approx(
        contradictory.contradiction_energy
    )
    assert contradictory.continuous_energy_used is True
    assert contradictory.to_dict()["component_energies"]["coverage_energy"] == pytest.approx(
        contradictory.coverage_energy
    )


def test_scenario_verify_1656_batch_scoring_preserves_order_and_dict_inputs() -> None:
    """SCENARIO-VERIFY-1656: extracted dict traces are normalized and scored in order."""

    scorer = EBRMTraceScorer()
    dict_trace = {
        "trace_id": "dict-unsupported",
        "expected_inconsistent": True,
        "steps": [
            {
                "step_id": "s1",
                "claim": "route_uses_east_bridge",
                "polarity": True,
                "confidence": 0.9,
                "constraint_ids": ["route"],
            },
            {
                "step_id": "s2",
                "claim": "route_is_valid",
                "polarity": True,
                "confidence": 0.55,
                "supports": ["missing-step"],
                "constraint_ids": [],
            },
        ],
    }

    rows = scorer.score_traces([_coherent_trace(), dict_trace, _contradictory_trace()])

    assert [row.trace_id for row in rows] == ["coherent", "dict-unsupported", "contradictory"]
    assert rows[1].unsupported_energy > 0.0
    assert rows[1].confidence_energy > 0.0
    assert rows[1].coverage_energy > 0.0
    assert rows[2].energy > rows[0].energy


def test_req_verify_1656_configuration_changes_continuous_energy() -> None:
    """REQ-VERIFY-1656: penalty weights make the continuous score configurable."""

    from carnot.models import EBRMTraceScorer as PackageScorer

    assert PackageScorer is EBRMTraceScorer

    trace = _contradictory_trace()
    base = EBRMTraceScorer().score_trace(trace)
    weighted = EBRMTraceScorer(EBRMTraceScorerConfig(contradiction_weight=5.0)).score_trace(trace)

    assert weighted.energy > base.energy
    assert weighted.contradiction_energy > base.contradiction_energy


def test_req_verify_1656_invalid_inputs_fail_closed() -> None:
    """REQ-VERIFY-1656: malformed extracted traces are rejected before scoring."""

    scorer = EBRMTraceScorer()

    with pytest.raises(ValueError, match="at least one step"):
        scorer.score_trace(LogicalTrace(trace_id="empty", steps=()))

    duplicate = LogicalTrace(
        trace_id="duplicate",
        steps=(
            LogicalTraceStep(step_id="s1", proposition="alpha", truth_value=True),
            LogicalTraceStep(step_id="s1", proposition="beta", truth_value=True),
        ),
    )
    with pytest.raises(ValueError, match="duplicate step_id"):
        scorer.score_trace(duplicate)

    with pytest.raises(ValueError, match="steps"):
        scorer.score_trace({"trace_id": "missing-steps"})

    with pytest.raises(ValueError, match="mapping"):
        scorer.score_trace(["not", "a", "trace"])  # type: ignore[arg-type]


def test_scenario_verify_1656_artifact_schema_and_writer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1656: artifact helpers write required schema fields."""

    output_path = tmp_path / "experiment_1656_ebrm_trace_scorer.json"
    artifact = write_artifact(
        output_path,
        run_date="20260509",
        tests_run=["tests/python/test_ebrm_scorer.py"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1656
    assert artifact["ebrm_trace_scorer_ready"] is True
    assert artifact["continuous_energy_used"] is True
    assert artifact["inconsistent_mean_energy"] > artifact["consistent_mean_energy"]
    assert artifact["energy_gap"] == pytest.approx(
        artifact["inconsistent_mean_energy"] - artifact["consistent_mean_energy"]
    )
    assert artifact["score_accuracy"] >= 0.8
    assert artifact["spec_traces"] == list(SPEC_TRACES)
    assert artifact["tests_run"] == ["tests/python/test_ebrm_scorer.py"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1656_artifact_validation_catches_schema_and_gate_drift() -> None:
    """REQ-VERIFY-1656: artifact validation catches missing fields and failed gates."""

    artifact = build_artifact(cases=default_synthetic_traces())
    validate_artifact(artifact)

    missing = dict(artifact)
    del missing["score_accuracy"]
    with pytest.raises(AssertionError, match="missing required fields"):
        validate_artifact(missing)

    bad_accuracy = dict(artifact, status="complete", score_accuracy=0.0)
    with pytest.raises(AssertionError, match="accuracy"):
        validate_artifact(bad_accuracy)

    bad_spec = dict(artifact, spec_traces=[])
    with pytest.raises(AssertionError, match="spec_traces"):
        validate_artifact(bad_spec)

    blocked = build_artifact(cases=[])
    assert blocked["status"] == "blocked"
    assert blocked["score_accuracy"] == 0.0
    validate_artifact(blocked)
