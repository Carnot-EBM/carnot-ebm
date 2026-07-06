"""Tests for Exp 5303 memory stress, conflict, forgetting, and rollback.

Spec refs: REQ-LEARN-5303, SCENARIO-LEARN-5303.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5303_memory_stress_conflict_forgetting_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5303_spec_declares_memory_stress_contract() -> None:
    """REQ-LEARN-5303: OpenSpec anchors the stress panel and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5303") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5303",
        "SCENARIO-LEARN-5303",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.EXP5302_RELATIVE_PATH),
        "memory_policy_candidate_ready",
        "incremental multi-turn memory updates",
        "delayed queries",
        "conflicting updates",
        "stale facts",
        "multi-hop conflicts",
        "selective forgetting",
        "harmful-memory injection",
        "rollback rows",
        "always-full verifier",
        "fixed governed memory",
        "Exp5302 adaptive memory policy candidate",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized_section
    assert "`competency_metrics`" in section
    assert " ".join(mod.FIELD_PRINCIPLES["competency_metrics"].split()) in normalized_section


def test_req_learn_5303_gate_blocks_when_exp5302_candidate_is_not_ready() -> None:
    """REQ-LEARN-5303-1: a false Exp5302 gate produces a blocked artifact."""

    blocked_candidate = {
        "memory_policy_candidate_ready": False,
        "honest_verdict": {"value": "null: candidate not ready"},
        "policy_version": "adaptive-memory-policy-v5302-selection-threshold-v1",
    }

    artifact = mod.build_result_artifact(
        root=REPO,
        exp5302_candidate=blocked_candidate,
        tests_run=[{"command": "gate unit", "outcome": "passed"}],
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_precondition:")
    assert artifact["memory_stress_passed"]["value"] is False
    assert artifact["precondition_checks"]["exp5302_memory_policy_candidate_ready"] is False
    assert artifact["competency_metrics"]["accurate_retrieval"]["adaptive_quality_rate"] == 0.0
    assert artifact["query_rows"] == []
    assert artifact["tests_run"] == [{"command": "gate unit", "outcome": "passed"}]
    mod.validate_artifact(artifact)


def test_req_learn_5303_stress_panel_covers_all_required_competencies() -> None:
    """REQ-LEARN-5303-2: the deterministic panel covers all mandated cases."""

    panel = mod.build_stress_panel()
    case_ids = [row.case_id for row in panel]
    query_competencies = {row.competency for row in panel if row.event_type == "query"}
    control_kinds = {row.control_kind for row in panel}

    assert len(case_ids) == len(set(case_ids))
    assert [row.turn for row in panel] == sorted(row.turn for row in panel)
    assert query_competencies >= set(mod.COMPETENCIES)
    assert {row.event_type for row in panel} >= {
        "update",
        "query",
        "forget",
        "harmful_injection",
    }
    assert control_kinds >= {
        "direct_conflict",
        "multi_hop_conflict",
        "stale_evidence",
        "selective_forgetting",
        "harmful_memory",
    }
    assert any(row.delayed_turns >= 2 for row in panel if row.event_type == "query")
    assert any(row.dependencies for row in panel)


def test_scenario_learn_5303_adaptive_policy_matches_quality_and_avoids_calls() -> None:
    """SCENARIO-LEARN-5303: adaptive stress quality matches full verification."""

    candidate = mod.load_exp5302_candidate(root=REPO)
    evaluation = mod.evaluate_stress_panel(mod.build_stress_panel(), candidate)

    assert evaluation["memory_stress_passed"] is True
    assert evaluation["policy_metrics"]["always_full"]["quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["fixed_governed_memory"]["quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["adaptive_memory_policy"]["quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["always_full"]["full_verifier_calls"] == 8
    assert evaluation["policy_metrics"]["fixed_governed_memory"]["full_verifier_calls"] == 8
    assert evaluation["policy_metrics"]["adaptive_memory_policy"]["full_verifier_calls"] == 3
    assert evaluation["calls_avoided"]["vs_always_full"] == 5
    assert evaluation["calls_avoided"]["additional_vs_fixed_governed_memory"] == 5
    assert evaluation["calls_avoided"]["rate_vs_always_full"] == pytest.approx(0.625)

    for competency in mod.COMPETENCIES:
        metrics = evaluation["competency_metrics"][competency]
        assert metrics["adaptive_quality_rate"] == 1.0
        assert metrics["always_full_quality_rate"] == 1.0
        assert metrics["fixed_governed_memory_quality_rate"] == 1.0

    rows = {row["case_id"]: row for row in evaluation["query_rows"]}
    assert rows["ar-query-runtime"]["adaptive_route"] == mod.ROUTE_MEMORY_CHECK
    assert rows["ttl-query-sensor"]["adaptive_route"] == mod.ROUTE_MEMORY_CHECK
    assert rows["lru-query-chain"]["adaptive_route"] == mod.ROUTE_MEMORY_CHECK
    assert rows["conflict-query-newest"]["adaptive_route"] == mod.ROUTE_MEMORY_CHECK
    assert rows["multi-hop-conflict-query"]["adaptive_route"] == mod.ROUTE_FULL
    assert rows["stale-query-outdated"]["adaptive_route"] == mod.ROUTE_FULL
    assert rows["forget-query-deprecated"]["adaptive_route"] == mod.ROUTE_FULL


def test_req_learn_5303_harmful_memory_and_rollback_are_safe() -> None:
    """REQ-LEARN-5303-4: harmful memory is blocked and rollback restores safety."""

    candidate = mod.load_exp5302_candidate(root=REPO)
    evaluation = mod.evaluate_stress_panel(mod.build_stress_panel(), candidate)
    trace = {row["case_id"]: row for row in evaluation["memory_trace"]}
    rows = {row["case_id"]: row for row in evaluation["query_rows"]}

    assert trace["harmful-injection-autopatch"]["event_type"] == "harmful_injection"
    assert trace["harmful-injection-autopatch"]["status"] == "rolled_back"
    assert trace["harmful-injection-autopatch"]["harmful_active_after_event"] is False
    assert trace["harmful-injection-autopatch"]["restored_safe_value"] == "build_rubric_before_patch"
    assert evaluation["unsafe_false_accepts"]["count"] == 0
    assert evaluation["unsafe_false_accepts"]["unsafe_case_ids_checked"] == [
        "rollback-query-safe-restored"
    ]
    assert evaluation["rollback_success_rate"]["successful"] == 1
    assert evaluation["rollback_success_rate"]["total"] == 1
    assert evaluation["rollback_success_rate"]["rate"] == 1.0

    rollback_row = rows["rollback-query-safe-restored"]
    assert rollback_row["unsafe"] is True
    assert rollback_row["adaptive_route"] == mod.ROUTE_MEMORY_CHECK
    assert rollback_row["adaptive_selected_decision"] == "reject"
    assert rollback_row["adaptive_unsafe_false_accept"] is False


def test_req_learn_5303_stale_conflict_and_failure_attribution_are_reported() -> None:
    """REQ-LEARN-5303-3/5: conflicts, stale facts, and attribution are explicit."""

    candidate = mod.load_exp5302_candidate(root=REPO)
    evaluation = mod.evaluate_stress_panel(mod.build_stress_panel(), candidate)

    stale = evaluation["stale_conflict_handling"]
    assert stale["rate"] == 1.0
    assert stale["resolved_or_escalated"] == 3
    assert stale["case_ids"] == [
        "conflict-query-newest",
        "multi-hop-conflict-query",
        "stale-query-outdated",
    ]
    assert stale["multi_hop_conflict_case_ids"] == ["multi-hop-conflict-query"]
    assert stale["stale_evidence_case_ids"] == ["stale-query-outdated"]

    forgetting = evaluation["selective_forgetting_correctness"]
    assert forgetting["correct"] == 2
    assert forgetting["total"] == 2
    assert forgetting["rate"] == 1.0

    attribution = evaluation["policy_failure_attribution"]
    assert attribution["adaptive_quality_failures"] == []
    assert attribution["fixed_control_limitations"]["full_calls_not_avoided"] == 8
    assert attribution["adaptive_escalation_reasons"] == {
        "multi_hop_conflict": 1,
        "selective_forgetting": 1,
        "stale_evidence": 1,
    }


def test_req_learn_5303_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5303-5: run() writes the required stress artifact schema."""

    tests_run = [{"command": "unit memory stress", "outcome": "passed"}]
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "memory stress passed" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["memory_stress_passed"]["value"] is True
    assert artifact["competency_metrics"]["principle"] == mod.FIELD_PRINCIPLES[
        "competency_metrics"
    ]
    assert artifact["unsafe_false_accepts"]["value"]["count"] == 0
    assert artifact["rollback_success_rate"]["value"]["rate"] == 1.0
    assert artifact["stale_conflict_handling"]["value"]["rate"] == 1.0
    assert artifact["policy_failure_attribution"]["value"]["adaptive_quality_failures"] == []
    assert artifact["calls_avoided"]["value"]["vs_always_full"] == 5
    assert artifact["tests_run"] == tests_run
    assert artifact["source_artifact_checksums"]["exp5302"].startswith("sha256:")

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_learn_5303_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5303: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["memory_stress_passed"]["value"] is True
    assert result["unsafe_false_accepts"]["value"]["count"] == 0
    assert result["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    mod.validate_artifact(result)


def test_req_learn_5303_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5303: artifact validation rejects required field drift."""

    artifact = mod.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit memory stress", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_wrapper = deepcopy(artifact)
    bad_wrapper["unsafe_false_accepts"] = {"value": {"count": 0}}
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        mod.validate_artifact(bad_wrapper)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_competency = deepcopy(artifact)
    del bad_competency["competency_metrics"]["accurate_retrieval"]
    with pytest.raises(ValueError, match="competency_metrics"):
        mod.validate_artifact(bad_competency)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = "passed"
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(bad_tests)
