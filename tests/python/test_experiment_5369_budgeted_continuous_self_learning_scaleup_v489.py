"""Tests for Exp5369 budgeted continuous self-learning scale-up.

Spec refs: REQ-LEARN-5369, SCENARIO-LEARN-5369-GATE,
SCENARIO-LEARN-5369-SCALE, SCENARIO-LEARN-5369-SAFETY-COST.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5369_budgeted_continuous_self_learning_scaleup_v489 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5369_spec_declares_budgeted_scaleup_contract() -> None:
    """REQ-LEARN-5369: OpenSpec anchors fields, gate, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5369") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5369",
        "SCENARIO-LEARN-5369-GATE",
        "SCENARIO-LEARN-5369-SCALE",
        "SCENARIO-LEARN-5369-SAFETY-COST",
        str(exp.RESULT_RELATIVE_PATH),
        "at least 12 multi-session traces",
        "at least 30 checked events",
        "always-full-context baseline, and a no-memory baseline",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_learn_5369_gate_copies_exp5368_readiness() -> None:
    """SCENARIO-LEARN-5369-GATE: Exp5368 readiness opens this scale-up."""

    gate = exp.confirm_source_gate(root=REPO)

    assert gate["all_passed"] is True
    assert gate["budget_curated_memory_ready"] is True
    assert gate["exp5368_status"] == "budget_curated_memory_ready"
    assert gate["source_unsafe_false_accepts_zero"] is True
    assert gate["source_no_weight_mutation"] is True
    assert str(exp.EXP5368_RELATIVE_PATH) in gate["source_artifacts"]


def test_scenario_learn_5369_scaled_traces_have_checked_events_and_safety() -> None:
    """SCENARIO-LEARN-5369-SCALE: replay has >=12 traces and >=30 checks."""

    traces = exp.build_budgeted_multi_session_traces()
    replay = exp.evaluate_budgeted_loop(traces=traces, root=REPO)

    assert len(traces) >= exp.MIN_MULTI_SESSION_TRACES
    assert replay["checked_event_count"] >= exp.MIN_CHECKED_EVENTS
    assert replay["hash_chain"]["valid"] is True
    assert replay["trace_provenance"]["dependency_attribution_rate"] > 0.0
    assert replay["trace_provenance"]["drift_detection_rate"] > 0.0
    assert replay["trace_provenance"]["rollback_recovery_rate"] == 1.0
    assert replay["budget_curation"]["stale_memory_deflection_rate"] == 1.0
    assert replay["budget_curation"]["poison_memory_deflection_rate"] == 1.0
    assert replay["budget_curation"]["retained_bytes_delta"] > 0

    events = [event for trace in traces for event in trace["events"]]
    assert all(event["context_object_version"]["integrity_hash"] for event in events)
    assert all(event["dependency_graph"]["edges"] for event in events)
    assert all(event["verifier_tool_decision"]["selected_verifier"] for event in events)
    assert all("drift_type" in event["drift_injection"] for event in events)
    assert all(event["execution_feedback"]["feedback_id"] for event in events)
    assert all(event["event_hash"].startswith("sha256:") for event in events)


def test_scenario_learn_5369_preserves_quality_reduces_cost_and_deflects_bad_memory() -> None:
    """SCENARIO-LEARN-5369-SAFETY-COST: combined policy is safe and cheaper."""

    replay = exp.evaluate_budgeted_loop(root=REPO)
    comparison = replay["policy_comparison"]
    combined = comparison["combined_metrics"]
    always = comparison["always_full_metrics"]
    no_memory = comparison["no_memory_metrics"]
    bad_rows = [
        row
        for row in replay["budget_curation"]["decision_rows"]
        if row["memory_variant"] in {"stale", "poisoned"}
    ]

    assert comparison["same_event_ids"] is True
    assert comparison["baselines_compared"] == {
        "always_full_context": True,
        "no_memory": True,
    }
    assert comparison["quality_delta_vs_always_full"] >= 0.0
    assert comparison["context_efficiency_delta"] > 0.0
    assert comparison["verifier_cost_delta"] > 0.0
    assert combined["final_quality"] >= always["final_quality"]
    assert combined["final_quality"] >= no_memory["final_quality"]
    assert replay["unsafe_false_accepts"] == 0
    assert all(
        row["trust_decision"] == "UNTRUST" or row["keep_decision"] != "KEEP"
        for row in bad_rows
    )


def test_req_learn_5369_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5369-5: run() writes the required terminal artifact."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5369_budgeted_continuous_"
                "self_learning_scaleup_v489.py -q --no-cov"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/python -m coverage run --source="
                "python/carnot/experiment_5369_budgeted_continuous_"
                "self_learning_scaleup_v489.py -m pytest "
                "tests/python/test_experiment_5369_budgeted_continuous_"
                "self_learning_scaleup_v489.py -q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["budget_curated_memory_ready"] is True
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["continuous_self_learning_budget_scaleup_ready"] is True
    assert artifact["multi_session_trace_count"] >= exp.MIN_MULTI_SESSION_TRACES
    assert artifact["checked_event_count"] >= exp.MIN_CHECKED_EVENTS
    assert artifact["quality_delta_vs_always_full"] >= 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["dependency_attribution_rate"] > 0.0
    assert artifact["drift_detection_rate"] > 0.0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["stale_memory_deflection_rate"] == 1.0
    assert artifact["poison_memory_deflection_rate"] == 1.0
    assert artifact["retained_bytes_delta"] > 0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["no_weight_mutation"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert artifact["readiness_gate"]["all_passed"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5369_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5369: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["status"]["value"] == "complete"
    assert result["budget_curated_memory_ready"] is True
    assert result["continuous_self_learning_budget_scaleup_ready"] is True
    assert result["unsafe_false_accepts"] == 0
    assert result["no_weight_mutation"] is True
    exp.validate_artifact(result)


def test_req_learn_5369_blocks_when_exp5368_gate_is_false(monkeypatch) -> None:
    """REQ-LEARN-5369-1: false Exp5368 readiness blocks the scale-up."""

    failed_gate = {
        "all_passed": False,
        "budget_curated_memory_ready": False,
        "exp5368_status": "blocked_budget_curated_memory_gate",
        "source_unsafe_false_accepts_zero": True,
        "source_no_weight_mutation": True,
        "source_artifacts": [str(exp.EXP5368_RELATIVE_PATH)],
        "failed_gates": ["budget_curated_memory_ready"],
    }
    monkeypatch.setattr(exp, "confirm_source_gate", lambda root=REPO: failed_gate)
    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5369", "outcome": "passed"}],
    )

    assert artifact["status"]["value"] == "blocked_budgeted_continuous_scaleup_gate"
    assert artifact["budget_curated_memory_ready"] is False
    assert artifact["continuous_self_learning_budget_scaleup_ready"] is False
    assert artifact["multi_session_trace_count"] == 0
    assert artifact["checked_event_count"] == 0
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_budgeted_continuous_self_learning_not_ready:"
    )
    exp.validate_artifact(artifact)


def test_req_learn_5369_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5369-5: validation rejects malformed terminal claims."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5369", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_status = deepcopy(artifact)
    bad_status["status"]["value"] = "complete"
    bad_status["continuous_self_learning_budget_scaleup_ready"] = False
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_memory"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_ready = deepcopy(artifact)
    bad_ready["continuous_self_learning_budget_scaleup_ready"] = "yes"
    with pytest.raises(ValueError, match="continuous_self_learning_budget_scaleup_ready"):
        exp.validate_artifact(bad_ready)

    bad_count = deepcopy(artifact)
    bad_count["checked_event_count"] = True
    with pytest.raises(ValueError, match="checked_event_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["context_efficiency_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="context_efficiency_delta"):
        exp.validate_artifact(bad_numeric)

    bad_quality = deepcopy(artifact)
    bad_quality["quality_delta_vs_always_full"] = -0.1
    with pytest.raises(ValueError, match="quality_delta_vs_always_full"):
        exp.validate_artifact(bad_quality)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_ready_status = deepcopy(artifact)
    bad_ready_status["status"]["value"] = "blocked_budgeted_continuous_scaleup_gate"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready_status)

    bad_budget_gate = deepcopy(artifact)
    bad_budget_gate["budget_curated_memory_ready"] = False
    with pytest.raises(ValueError, match="budget_curated_memory_ready"):
        exp.validate_artifact(bad_budget_gate)

    bad_trace_count = deepcopy(artifact)
    bad_trace_count["multi_session_trace_count"] = exp.MIN_MULTI_SESSION_TRACES - 1
    with pytest.raises(ValueError, match="multi_session_trace_count"):
        exp.validate_artifact(bad_trace_count)

    bad_event_count = deepcopy(artifact)
    bad_event_count["checked_event_count"] = exp.MIN_CHECKED_EVENTS - 1
    with pytest.raises(ValueError, match="checked_event_count"):
        exp.validate_artifact(bad_event_count)

    bad_context_delta = deepcopy(artifact)
    bad_context_delta["context_efficiency_delta"] = 0.0
    with pytest.raises(ValueError, match="context_efficiency_delta"):
        exp.validate_artifact(bad_context_delta)

    bad_verifier_delta = deepcopy(artifact)
    bad_verifier_delta["verifier_cost_delta"] = 0.0
    with pytest.raises(ValueError, match="verifier_cost_delta"):
        exp.validate_artifact(bad_verifier_delta)

    bad_retained_delta = deepcopy(artifact)
    bad_retained_delta["retained_bytes_delta"] = 0
    with pytest.raises(ValueError, match="retained_bytes_delta"):
        exp.validate_artifact(bad_retained_delta)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_field = deepcopy(artifact)
    bad_field["status"] = {"value": "complete"}
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_field)

    assert exp._wrapped_value("plain") == "plain"
    assert exp._json_ready(Path("helper")) == "helper"
    assert exp._rate(1, 0) == 0.0
