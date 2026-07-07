"""Tests for Exp5342 provenance-bound self-learning scale-up.

Spec refs: REQ-LEARN-5342, SCENARIO-LEARN-5342-PROVENANCE,
SCENARIO-LEARN-5342-ATTACK, SCENARIO-LEARN-5342-POLICY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5342_provenance_bound_self_learning_scaleup_v487 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _events_by_id(traces: exp.JsonList) -> dict[str, exp.JsonDict]:
    return {
        event["event_id"]: event
        for trace in traces
        for event in trace["events"]
    }


def test_req_learn_5342_spec_declares_scaleup_contract() -> None:
    """REQ-LEARN-5342: OpenSpec anchors fields, policies, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5342") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5342",
        "SCENARIO-LEARN-5342-PROVENANCE",
        "SCENARIO-LEARN-5342-ATTACK",
        "SCENARIO-LEARN-5342-POLICY",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5340 utility-weighted context memory fixture",
        "Exp5341 bounded compressor drift monitor fixture",
        "always-full",
        "utility-only",
        "bounded-compressor-only",
        "combined certificate-gated",
        "point-in-time reconstruction",
        "cross-event suspicion telemetry",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5342_loads_upstream_readiness_gates() -> None:
    """REQ-LEARN-5342-1: Exp5340 and Exp5341 gates are verified first."""

    gate = exp.confirm_upstream_readiness(root=REPO)

    assert gate["all_passed"] is True
    assert gate["utility_memory_ready"] is True
    assert gate["compressor_drift_fixture_ready"] is True
    assert gate["utility_certificate_gate_ready"] is True
    assert gate["compressor_certificate_gate_ready"] is True
    assert gate["no_weight_mutation"] is True
    assert gate["failed_gates"] == []


def test_scenario_learn_5342_provenance_hashes_and_reconstruction() -> None:
    """SCENARIO-LEARN-5342-PROVENANCE: hashes and PIT replay are stable."""

    traces = exp.build_multi_session_traces()
    events = _events_by_id(traces)
    chain = exp.validate_hash_chains(traces)
    reconstruction = exp.point_in_time_reconstruction(traces)

    assert 3 <= len(traces) <= 5
    assert chain["valid"] is True
    assert chain["checked_event_count"] == sum(len(trace["events"]) for trace in traces)
    assert reconstruction["point_in_time_reconstruction_rate"] == 1.0
    assert reconstruction["mismatch_count"] == 0
    assert all(trace["trace_id"] and trace["session_id"] for trace in traces)
    assert all(
        event["event_hash"].startswith("sha256:")
        and event["previous_event_hash"].startswith("sha256:")
        and event["context_object_version"]["integrity_hash"].startswith("sha256:")
        and isinstance(event["context_object_version"]["approved"], bool)
        and isinstance(event["context_object_version"]["current"], bool)
        and isinstance(event["supporting_events"], list)
        and isinstance(event["audit_trace"], dict)
        for event in events.values()
    )
    assert events["s2-e3-rollback-patch"]["sidecar"]["recoverable"] is True
    assert events["s2-e3-rollback-patch"]["audit_trace"]["decision"] == "rollback"


def test_scenario_learn_5342_attack_needs_cross_event_telemetry() -> None:
    """SCENARIO-LEARN-5342-ATTACK: aggregate telemetry catches slow poison."""

    traces = exp.build_multi_session_traces()
    telemetry = exp.evaluate_cross_event_telemetry(traces)
    attack_events = [
        event
        for trace in traces
        for event in trace["events"]
        if event["attack_id"] == "poison-runtime-currentness"
    ]

    assert len(attack_events) == 3
    assert all(
        event["per_diff_suspicion"] < exp.PER_DIFF_REJECTION_THRESHOLD
        for event in attack_events
    )
    assert telemetry["per_diff_attack_detection_rate"] == 0.0
    assert telemetry["cross_event_attack_detection_rate"] == 1.0
    assert telemetry["detected_attack_ids"] == ["poison-runtime-currentness"]
    assert telemetry["attack_rollbacks"] == 1

    per_diff_visible = deepcopy(traces)
    per_diff_visible[2]["events"][0]["per_diff_suspicion"] = (
        exp.PER_DIFF_REJECTION_THRESHOLD
    )
    visible_telemetry = exp.evaluate_cross_event_telemetry(per_diff_visible)
    assert visible_telemetry["per_diff_attack_detection_rate"] == 1.0


def test_scenario_learn_5342_policy_comparison_combines_gates() -> None:
    """SCENARIO-LEARN-5342-POLICY: combined policy is safe and cheaper."""

    traces = exp.build_multi_session_traces()
    comparison = exp.evaluate_policy_comparison(traces)
    metrics = comparison["policy_metrics"]
    combined = metrics[exp.COMBINED_POLICY]
    always = metrics[exp.ALWAYS_FULL_POLICY]
    utility = metrics[exp.UTILITY_ONLY_POLICY]
    bounded = metrics[exp.BOUNDED_ONLY_POLICY]

    assert set(metrics) == set(exp.POLICY_ARMS)
    assert exp._same_trace_ids(comparison["policy_rows"]) is True
    assert combined["final_quality"] == always["final_quality"]
    assert combined["final_quality"] == 1.0
    assert combined["unsafe_false_accepts"] == 0
    assert combined["cross_event_attack_detection_rate"] == 1.0
    assert combined["rollback_events"] >= 2
    assert comparison["memory_hygiene_delta"] > 0.0
    assert comparison["context_efficiency_delta"] > 0.0
    assert comparison["verifier_cost_delta"] > 0.0
    assert comparison["process_metric_improved"] is True
    assert utility["unsafe_false_accepts"] > 0
    assert bounded["unsafe_false_accepts"] > 0


def test_req_learn_5342_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5342-7: run() writes the requested terminal artifact."""

    tests_run = [{"command": "unit provenance bound scaleup", "outcome": "passed"}]
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
    assert artifact["multi_session_trace_count"] == 4
    assert artifact["context_hash_chain_valid"] is True
    assert artifact["point_in_time_reconstruction_rate"] == 1.0
    assert artifact["memory_hygiene_delta"] > 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["cross_event_attack_detection_rate"] == 1.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["rollback_events"] >= 2
    assert artifact["self_learning_scaleup_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5342_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5342: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["self_learning_scaleup_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_learn_5342_blocked_artifact_when_upstream_gates_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5342-1/7: failed upstream gates produce blocked artifacts."""

    def blocked_upstream(*, root: Path | str = exp.REPO_ROOT) -> dict[str, object]:
        assert root == REPO
        return {
            "utility_memory_ready": False,
            "compressor_drift_fixture_ready": True,
            "utility_certificate_gate_ready": False,
            "compressor_certificate_gate_ready": True,
            "no_weight_mutation": True,
            "failed_gates": ["utility_memory_ready", "utility_certificate_gate_ready"],
            "all_passed": False,
            "utility_source_honest_verdict": "blocked_utility",
            "compressor_source_honest_verdict": "complete: compressor",
        }

    monkeypatch.setattr(exp, "confirm_upstream_readiness", blocked_upstream)
    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "blocked scaleup unit", "outcome": "passed"}],
    )

    assert artifact["status"]["value"] == "blocked_upstream_or_scaleup_gate"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_self_learning_scaleup_not_ready:"
    )
    assert artifact["self_learning_scaleup_ready"] is False
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["multi_session_trace_count"] == 0
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5342_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5342-7: artifact validation rejects gate and scalar drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit provenance scaleup", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_utility_learning"
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

    bad_bool = deepcopy(artifact)
    bad_bool["context_hash_chain_valid"] = "yes"
    with pytest.raises(ValueError, match="context_hash_chain_valid"):
        exp.validate_artifact(bad_bool)

    bad_numeric = deepcopy(artifact)
    bad_numeric["verifier_cost_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="verifier_cost_delta"):
        exp.validate_artifact(bad_numeric)

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

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5342_blocked_artifact_when_tests_not_recorded() -> None:
    """REQ-LEARN-5342-7: ready gate stays false without test records."""

    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["status"]["value"] == "blocked_upstream_or_scaleup_gate"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_self_learning_scaleup_not_ready:"
    )
    assert "tests_not_recorded" in artifact["honest_verdict"]["value"]
    assert artifact["self_learning_scaleup_ready"] is False
    assert artifact["unsafe_false_accepts"] == 0
    exp.validate_artifact(artifact)


def test_req_learn_5342_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5342: helper paths stay deterministic and explicit."""

    traces = exp.build_multi_session_traces()
    blocked = exp._blocked_evaluation()
    first_event = traces[0]["events"][0]
    altered_traces = deepcopy(traces)
    altered_traces[0]["events"][0]["event_hash"] = "sha256:bad"
    altered_state = deepcopy(traces)
    altered_state[0]["events"][0]["state_after_sha256"] = "sha256:bad"

    assert exp._rate(1, 0) == 0.0
    assert exp._delta(0.3333333333333, 0.1111111111111) == 0.222222
    assert exp._json_ready((Path("x"), {"y": Path("z")})) == ["x", {"y": "z"}]
    assert exp._is_numeric(0.0) is True
    assert exp._is_numeric(True) is False
    assert exp._wrapped_value({"value": "x"}) == "x"
    assert exp._wrapped_value("x") == "x"
    assert exp._event_hash(first_event) == first_event["event_hash"]
    assert exp.validate_hash_chains(altered_traces)["valid"] is False
    assert exp.point_in_time_reconstruction(altered_state)["mismatch_count"] == 1
    assert blocked["self_learning_scaleup_ready"] is False
    assert blocked["policy_metrics"] == {}
    assert exp._policy_route(exp.COMBINED_POLICY, first_event, {})["policy"] == exp.COMBINED_POLICY
    assert exp._policy_route(exp.UTILITY_ONLY_POLICY, first_event, {})["verifier_call"] is False
    assert exp._policy_route(exp.ALWAYS_FULL_POLICY, first_event, {})["verifier_call"] is True

    unknown_policy_route = exp._policy_route("unknown_policy", first_event, {})
    assert unknown_policy_route["accepted"] is False
    assert "unknown_policy" in unknown_policy_route["rejection_reasons"]
