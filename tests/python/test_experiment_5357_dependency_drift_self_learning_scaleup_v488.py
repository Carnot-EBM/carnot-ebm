"""Tests for Exp5357 dependency-drift self-learning scale-up.

Spec refs: REQ-LEARN-5357, SCENARIO-LEARN-5357-TRACE,
SCENARIO-LEARN-5357-POLICY, SCENARIO-LEARN-5357-ANTI-TAUTOLOGY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5357_dependency_drift_self_learning_scaleup_v488 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5357_spec_declares_dependency_drift_contract() -> None:
    """REQ-LEARN-5357: OpenSpec anchors fields, policies, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5357") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5357",
        "SCENARIO-LEARN-5357-TRACE",
        "SCENARIO-LEARN-5357-POLICY",
        "SCENARIO-LEARN-5357-ANTI-TAUTOLOGY",
        str(exp.RESULT_RELATIVE_PATH),
        "always-full, utility-only, compressor-only, dependency-only",
        "drift-guarded, and combined certificate-gated",
        "Exp5342 v487 scale-up only as quarantined context",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5357_source_gate_checks_dependency_and_drift_artifacts() -> None:
    """REQ-LEARN-5357-1: source gates require Exp5355 and Exp5356 readiness."""

    gate = exp.confirm_source_gate(root=REPO)

    assert gate["all_passed"] is True
    assert gate["dependency_provenance_ready"] is True
    assert gate["memory_tool_drift_ready"] is True
    assert gate["certificate_gate_ready"] is True
    assert gate["source_unsafe_false_accepts_zero"] is True
    assert gate["source_metric_duplicates_clear"] is True
    assert gate["rollback_recovery_ready"] is True
    assert gate["prior_v487_scaleup_quarantined"] is True
    assert gate["prior_v487_metric_duplication_detected"] is True
    assert str(exp.EXP5342_QUARANTINED_RELATIVE_PATH) in gate["excluded_artifacts"]


def test_scenario_learn_5357_traces_have_hashes_dependency_drift_and_rollback() -> None:
    """SCENARIO-LEARN-5357-TRACE: traces carry replayable provenance."""

    traces = exp.build_multi_session_traces()
    hash_chain = exp.validate_hash_chains(traces)
    audit = exp.evaluate_trace_provenance(traces)

    assert 4 <= len(traces) <= 6
    assert hash_chain["valid"] is True
    assert hash_chain["failure_count"] == 0
    assert audit["dependency_attribution_rate"] > 0.0
    assert audit["drift_detection_rate"] > 0.0
    assert audit["rollback_recovery_rate"] == 1.0

    events = [event for trace in traces for event in trace["events"]]
    assert all(event["context_object_version"]["integrity_hash"] for event in events)
    assert all(event["dependency_graph"]["edges"] for event in events)
    assert all(event["verifier_tool_decision"]["selected_verifier"] for event in events)
    assert all("drift_type" in event["drift_injection"] for event in events)
    assert all(event["execution_feedback"]["feedback_id"] for event in events)
    assert all(event["previous_event_hash"] for event in events)
    assert all(event["event_hash"].startswith("sha256:") for event in events)

    rollback_events = [
        event for event in events if event["rollback_event"]["triggered"]
    ]
    assert rollback_events
    assert all(event["rollback_event"]["recovered"] for event in rollback_events)


def test_scenario_learn_5357_policy_comparison_preserves_quality_and_process() -> None:
    """SCENARIO-LEARN-5357-POLICY: combined policy improves process safely."""

    comparison = exp.evaluate_policy_comparison(exp.build_multi_session_traces())
    metrics = comparison["policy_metrics"]
    combined = metrics[exp.COMBINED_POLICY]
    always = metrics[exp.ALWAYS_FULL_POLICY]

    assert set(metrics) == set(exp.POLICY_ARMS)
    assert comparison["same_event_ids"] is True
    assert comparison["all_policies_run"] is True
    assert any(
        metrics[policy]["unsafe_false_accepts"] > 0
        for policy in exp.POLICY_ARMS
        if policy != exp.COMBINED_POLICY
    )
    assert combined["unsafe_false_accepts"] == 0
    assert comparison["quality_delta_vs_always_full"] >= 0.0
    assert combined["final_quality"] >= always["final_quality"]
    assert comparison["process_metric_improved"] is True
    assert comparison["memory_hygiene_delta"] != comparison["context_efficiency_delta"]
    assert comparison["verifier_cost_delta"] > 0.0


def test_scenario_learn_5357_anti_tautology_keeps_aggregate_metrics_distinct() -> None:
    """SCENARIO-LEARN-5357-ANTI-TAUTOLOGY: aggregate metrics are separate."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit dependency drift scale-up", "outcome": "passed"}],
    )
    metric_values = {
        field: artifact[field] for field in exp.AGGREGATE_METRIC_FIELDS
    }

    assert artifact["dependency_attribution_rate"] != artifact["drift_detection_rate"]
    assert artifact["quality_delta_vs_always_full"] == 0.0
    assert artifact["memory_hygiene_delta"] > 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert len(set(metric_values.values())) == len(metric_values)
    assert artifact["duplicated_metric_pairs"] == []
    assert artifact["self_learning_scaleup_ready"] is True
    assert artifact["metric_duplication_check"]["duplicated_metric_pairs"] == []


def test_req_learn_5357_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5357-7: run() writes the required terminal artifact."""

    tests_run = [
        {"command": "unit deterministic dependency drift", "outcome": "passed"},
        {"command": "unit rollback anti-tautology", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "self_learning_scaleup_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert 4 <= artifact["multi_session_trace_count"] <= 6
    assert artifact["context_hash_chain_valid"] is True
    assert artifact["dependency_attribution_rate"] > 0.0
    assert artifact["drift_detection_rate"] > 0.0
    assert artifact["quality_delta_vs_always_full"] >= 0.0
    assert artifact["duplicated_metric_pairs"] == []
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["self_learning_scaleup_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5357_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5357: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["self_learning_scaleup_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_false_accepts"] == 0
    assert result["duplicated_metric_pairs"] == []
    exp.validate_artifact(result)


def test_req_learn_5357_blocked_when_tests_not_recorded() -> None:
    """REQ-LEARN-5357-7: ready gate stays false without test records."""

    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["status"]["value"] == "blocked_dependency_drift_scaleup_gate"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_dependency_drift_scaleup_not_ready:"
    )
    assert "tests_not_recorded" in artifact["honest_verdict"]["value"]
    assert artifact["self_learning_scaleup_ready"] is False
    assert artifact["unsafe_false_accepts"] == 0
    exp.validate_artifact(artifact)


def test_req_learn_5357_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5357-7: artifact validation rejects scalar and gate drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit dependency drift", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_memory"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_learning = deepcopy(artifact)
    bad_learning["continuous_self_learning_target"] = {"value": True}
    with pytest.raises(ValueError, match="continuous_self_learning_target"):
        exp.validate_artifact(bad_learning)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_count = deepcopy(artifact)
    bad_count["multi_session_trace_count"] = True
    with pytest.raises(ValueError, match="multi_session_trace_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["dependency_attribution_rate"] = {"value": 1.0}
    with pytest.raises(ValueError, match="dependency_attribution_rate"):
        exp.validate_artifact(bad_numeric)

    bad_quality = deepcopy(artifact)
    bad_quality["quality_delta_vs_always_full"] = -0.1
    with pytest.raises(ValueError, match="quality_delta_vs_always_full"):
        exp.validate_artifact(bad_quality)

    bad_duplicates = deepcopy(artifact)
    bad_duplicates["duplicated_metric_pairs"] = [
        {
            "left": "memory_hygiene_delta",
            "right": "context_efficiency_delta",
            "value": artifact["memory_hygiene_delta"],
        }
    ]
    with pytest.raises(ValueError, match="duplicated_metric_pairs"):
        exp.validate_artifact(bad_duplicates)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["self_learning_scaleup_ready"] = "yes"
    with pytest.raises(ValueError, match="self_learning_scaleup_ready"):
        exp.validate_artifact(bad_ready)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_hash_ready = deepcopy(artifact)
    bad_hash_ready["context_hash_chain_valid"] = False
    with pytest.raises(ValueError, match="context_hash_chain_valid"):
        exp.validate_artifact(bad_hash_ready)

    bad_rollback_ready = deepcopy(artifact)
    bad_rollback_ready["rollback_recovery_rate"] = 0.5
    with pytest.raises(ValueError, match="rollback_recovery_rate"):
        exp.validate_artifact(bad_rollback_ready)

    bad_process_ready = deepcopy(artifact)
    bad_process_ready["policy_comparison"]["process_metric_improved"] = False
    with pytest.raises(ValueError, match="process_metric_improved"):
        exp.validate_artifact(bad_process_ready)

    bad_wrapped = deepcopy(artifact)
    bad_wrapped["experiment_id"] = {"value": exp.EXPERIMENT_ID}
    with pytest.raises(ValueError, match="experiment_id"):
        exp.validate_artifact(bad_wrapped)


def test_req_learn_5357_error_branches_and_blocked_artifact(monkeypatch) -> None:
    """REQ-LEARN-5357: branch checks fail closed instead of hiding drift."""

    traces = exp.build_multi_session_traces()
    bad_previous = deepcopy(traces)
    bad_previous[0]["events"][0]["previous_event_hash"] = "sha256:wrong"
    assert exp.validate_hash_chains(bad_previous)["failures"][0]["reason"] == (
        "previous_event_hash_mismatch"
    )

    bad_event_hash = deepcopy(traces)
    bad_event_hash[0]["events"][0]["context_object_version"]["payload"] = "changed"
    hash_audit = exp.validate_hash_chains(bad_event_hash)
    assert any(
        failure["reason"] == "event_hash_mismatch"
        for failure in hash_audit["failures"]
    )

    event = deepcopy(traces[0]["events"][0])
    event["utility_memory"]["utility_score"] = 0.1
    utility_row = exp._policy_route(exp.UTILITY_ONLY_POLICY, event)
    assert utility_row["rejection_reasons"] == ["utility_score_too_low"]

    compressor_row = exp._policy_route(exp.COMPRESSOR_ONLY_POLICY, event)
    assert compressor_row["rejection_reasons"] == ["compressor_rejected_context"]

    unknown_row = exp._policy_route("unknown_policy", event)
    assert unknown_row["rejection_reasons"] == ["unknown_policy"]

    assert exp._json_ready(Path("x")) == "x"
    assert exp._wrapped_value("plain") == "plain"
    assert exp._rate(1, 0) == 0.0

    failed_gate = {
        "all_passed": False,
        "no_weight_mutation": True,
        "failed_gates": ["dependency_provenance_ready"],
        "prior_v487_duplicated_metric_pairs": [],
    }
    monkeypatch.setattr(exp, "confirm_source_gate", lambda root=REPO: failed_gate)
    monkeypatch.setattr(exp, "find_duplicated_metric_pairs", lambda metrics: [])
    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["self_learning_scaleup_ready"] is False
    assert artifact["multi_session_trace_count"] == 0
    assert artifact["trace_provenance"]["event_count"] == 0
    assert artifact["policy_comparison"]["all_policies_run"] is False
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_dependency_drift_scaleup_not_ready:"
    )
    exp.validate_artifact(artifact)
