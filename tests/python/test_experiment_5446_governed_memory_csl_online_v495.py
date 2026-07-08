"""Tests for Exp5446 governed-memory CSL online lifecycle.

Spec refs: REQ-LEARN-5446,
SCENARIO-LEARN-5446-GATES,
SCENARIO-LEARN-5446-CONTROLS,
SCENARIO-LEARN-5446-ROLLBACK,
SCENARIO-LEARN-5446-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5446_governed_memory_csl_online_v495 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5446_governed_memory_csl_online_v495.py "
    "-q --no-cov -n 0"
)


def test_req_learn_5446_spec_declares_governed_online_lifecycle() -> None:
    """REQ-LEARN-5446: OpenSpec anchors the governed online memory lifecycle."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5446") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5446",
        "SCENARIO-LEARN-5446-GATES",
        "SCENARIO-LEARN-5446-CONTROLS",
        "SCENARIO-LEARN-5446-ROLLBACK",
        "SCENARIO-LEARN-5446-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        "raw trace, case memory, procedural skill memory, and declarative rule memory",
        "evidence support, execution dependency, replay success, temporal decay",
        "always-full-context, no-memory, and ungated-memory controls",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5446_fixture_covers_sessions_and_promotion_levels() -> None:
    """REQ-LEARN-5446-1/2: trace stream covers sessions and promotion levels."""

    evaluation = exp.evaluate_governed_memory_loop(root=REPO)
    rows = evaluation["trace_rows"]
    families = {row["trace_family"] for row in rows}
    promoted_levels = {row["promotion_level"] for row in evaluation["promoted_memories"]}

    assert evaluation["multi_session_trace_count"] == len(rows)
    assert evaluation["multi_session_trace_count"] >= 8
    assert families >= exp.REQUIRED_TRACE_FAMILIES
    assert {row["session_id"] for row in rows} >= {"session-a", "session-b", "session-c"}
    assert all(row["raw_trace_receipt"]["checksum"].startswith("sha256:") for row in rows)
    assert evaluation["promotion_level_counts"]["raw_trace"] == len(rows)
    assert promoted_levels == {"case", "skill", "rule"}
    assert evaluation["promotion_level_counts"]["case"] >= 1
    assert evaluation["promotion_level_counts"]["skill"] >= 1
    assert evaluation["promotion_level_counts"]["rule"] >= 1
    assert evaluation["evidence_support_edges"] > 0
    assert evaluation["execution_dependency_edges"] > 0


def test_scenario_learn_5446_replay_and_provenance_gates_precede_routing() -> None:
    """SCENARIO-LEARN-5446-GATES: missing replay or provenance cannot route."""

    evaluation = exp.evaluate_governed_memory_loop(root=REPO)
    promoted = evaluation["promoted_memories"]
    inactive = evaluation["rejected_memories"] + evaluation["abstained_memories"]

    assert promoted
    assert inactive
    for row in promoted:
        assert all(row["gate_results"].values())
        assert row["active_for_routing"] is True
        assert row["routing_influence"] > 0
        assert row["promotion_decision"]["reasons"] == ["all_governance_gates_passed"]

    for row in inactive:
        assert row["active_for_routing"] is False
        assert row["routing_influence"] == 0
        assert row["audit_retained"] is True
        assert not all(row["gate_results"].values())

    base = deepcopy(promoted[0])
    no_replay = exp.apply_governance_gates(base | {"replay_success": False})
    no_evidence = exp.apply_governance_gates(base | {"evidence_support_valid": False})
    no_dependency = exp.apply_governance_gates(
        base | {"execution_dependency_valid": False}
    )
    no_access = exp.apply_governance_gates(base | {"access_control_valid": False})

    assert no_replay["promotion_status"] == "rejected"
    assert "replay_success_failed" in no_replay["promotion_decision"]["reasons"]
    assert no_replay["routing_influence"] == 0
    assert no_evidence["promotion_status"] == "rejected"
    assert "evidence_support_missing" in no_evidence["promotion_decision"]["reasons"]
    assert no_evidence["routing_influence"] == 0
    assert no_dependency["promotion_status"] == "rejected"
    assert "execution_dependency_missing" in no_dependency["promotion_decision"]["reasons"]
    assert no_dependency["routing_influence"] == 0
    assert no_access["promotion_status"] == "abstained"
    assert "access_control_denied" in no_access["promotion_decision"]["reasons"]
    assert no_access["routing_influence"] == 0


def test_scenario_learn_5446_controls_measure_cost_and_safety() -> None:
    """SCENARIO-LEARN-5446-CONTROLS: governed memory preserves quality safely."""

    evaluation = exp.evaluate_governed_memory_loop(root=REPO)
    controls = evaluation["control_metrics"]
    trace_id_sets = evaluation["control_trace_id_sets"]

    assert len({tuple(ids) for ids in trace_id_sets.values()}) == 1
    assert set(controls) == {"always_full_context", "no_memory", "ungated_memory", "governed_memory"}
    assert controls["governed_memory"]["quality_score"] == pytest.approx(
        controls["always_full_context"]["quality_score"]
    )
    assert controls["governed_memory"]["context_cost"] < controls["always_full_context"][
        "context_cost"
    ]
    assert controls["governed_memory"]["verifier_cost"] < controls["always_full_context"][
        "verifier_cost"
    ]
    assert controls["ungated_memory"]["unsafe_false_accepts"] > 0
    assert controls["governed_memory"]["unsafe_false_accepts"] == 0
    assert evaluation["quality_delta_vs_always_full"] == pytest.approx(0.0)
    assert evaluation["context_efficiency_delta"] > 0.0
    assert evaluation["verifier_cost_delta"] > 0.0
    assert evaluation["negative_transfer_deflection_rate"] == 1.0
    assert evaluation["unsafe_false_accepts"] == 0


def test_scenario_learn_5446_rollback_removes_memories_from_future_decisions() -> None:
    """SCENARIO-LEARN-5446-ROLLBACK: promoted memories are reversible."""

    evaluation = exp.evaluate_governed_memory_loop(root=REPO)
    rollback = evaluation["rollback_audit"]
    rolled_back = set(rollback["rolled_back_memory_ids"])

    assert evaluation["rollback_recovery_rate"] == 1.0
    assert rollback["rollback_success"] is True
    assert set(rollback["removed_from_case_sidecar"]) <= rolled_back
    assert set(rollback["removed_from_skill_sidecar"]) <= rolled_back
    assert set(rollback["removed_from_rule_sidecar"]) <= rolled_back
    assert rollback["prior_case_sidecar_restored"] is True
    assert rollback["prior_skill_sidecar_restored"] is True
    assert rollback["prior_rule_sidecar_restored"] is True
    assert evaluation["post_rollback_decisions"]
    assert all(
        rolled_back.isdisjoint(decision["cited_memory_ids"])
        for decision in evaluation["post_rollback_decisions"]
    )


def test_scenario_learn_5446_no_weight_mutation_boundary() -> None:
    """SCENARIO-LEARN-5446-NO-WEIGHT-MUTATION: online learning is sidecar-only."""

    evaluation = exp.evaluate_governed_memory_loop(root=REPO)

    assert evaluation["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "governed_trace_case_skill_rule_sidecars_only",
    }


def test_req_learn_5446_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5446-7: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=[TEST_COMMAND])
    mapping_artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "mapped", "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert mapping_artifact["tests_run"] == [{"command": "mapped", "outcome": "passed"}]
    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["multi_session_trace_count"] == len(artifact["trace_rows"])
    assert artifact["promotion_level_counts"]["raw_trace"] == len(artifact["trace_rows"])
    assert artifact["evidence_support_edges"] > 0
    assert artifact["execution_dependency_edges"] > 0
    assert artifact["replay_success_rate"] == 1.0
    assert artifact["temporal_decay_policy"] == exp.TEMPORAL_DECAY_POLICY
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["quality_delta_vs_always_full"] >= 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["no_weight_mutation"] is True
    assert artifact["governed_csl_loop_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["research_conductor_modified"] is False
    exp.validate_artifact(artifact)


def test_req_learn_5446_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5446-7: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["governed_csl_loop_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5446_blocked_artifact_reports_missing_tests() -> None:
    """REQ-LEARN-5446-7: missing test evidence keeps readiness blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["governed_csl_loop_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5446_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5446-7: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    bad_missing = deepcopy(artifact)
    bad_missing.pop("multi_session_trace_count")
    with pytest.raises(ValueError, match="multi_session_trace_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["continuous_self_learning_task"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_ready = deepcopy(artifact)
    bad_ready["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_ready)

    bad_bool = deepcopy(artifact)
    bad_bool["no_weight_mutation"] = "true"
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["multi_session_trace_count"] = True
    with pytest.raises(ValueError, match="multi_session_trace_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["context_efficiency_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="context_efficiency_delta"):
        exp.validate_artifact(bad_numeric)

    bad_rate = deepcopy(artifact)
    bad_rate["replay_success_rate"] = 0.5
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_rate)

    bad_rate_invalid = deepcopy(artifact)
    bad_rate_invalid["rollback_recovery_rate"] = 1.5
    with pytest.raises(ValueError, match="rollback_recovery_rate"):
        exp.validate_artifact(bad_rate_invalid)

    bad_counts_keys = deepcopy(artifact)
    bad_counts_keys["promotion_level_counts"] = {"raw_trace": len(artifact["trace_rows"])}
    with pytest.raises(ValueError, match="promotion_level_counts"):
        exp.validate_artifact(bad_counts_keys)

    bad_counts_value = deepcopy(artifact)
    bad_counts_value["promotion_level_counts"]["case"] = True
    with pytest.raises(ValueError, match="promotion_level_counts"):
        exp.validate_artifact(bad_counts_value)

    bad_complete_not_ready = deepcopy(artifact)
    bad_complete_not_ready["governed_csl_loop_ready"] = False
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_complete_not_ready)

    bad_blocked_ready = deepcopy(artifact)
    bad_blocked_ready["status"] = "blocked"
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_blocked_ready)

    bad_readiness = deepcopy(artifact)
    bad_readiness["readiness_checks"]["all_passed"] = False
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_readiness)

    bad_csl_task = deepcopy(artifact)
    bad_csl_task["continuous_self_learning_task"] = False
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        exp.validate_artifact(bad_csl_task)

    bad_quality_delta = deepcopy(artifact)
    bad_quality_delta["quality_delta_vs_always_full"] = -0.1
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_quality_delta)

    bad_verifier_delta = deepcopy(artifact)
    bad_verifier_delta["verifier_cost_delta"] = -0.1
    with pytest.raises(ValueError, match="governed_csl_loop_ready"):
        exp.validate_artifact(bad_verifier_delta)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor_modified"):
        exp.validate_artifact(bad_conductor)


def test_req_learn_5446_default_tests_and_defensive_helpers() -> None:
    """REQ-LEARN-5446-4/7: helper branches remain deterministic."""

    tests_run = exp.default_tests_run()
    artifact = exp.build_artifact(root=REPO, tests_run=tests_run)
    no_promoted = exp.verify_rollback_removes_promoted_memories([])

    assert tests_run[0]["command"] == TEST_COMMAND
    assert tests_run[1]["command"].startswith(".venv/bin/coverage run")
    assert tests_run[2]["command"] == ".venv/bin/pytest tests/python -q"
    assert artifact["tests_run"] == tests_run
    assert no_promoted["rollback_success"] is True
    assert no_promoted["rolled_back_memory_ids"] == []
