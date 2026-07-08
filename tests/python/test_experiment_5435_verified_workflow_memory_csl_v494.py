"""Tests for Exp5435 verified workflow-memory CSL fixture.

Spec refs: REQ-LEARN-5435,
SCENARIO-LEARN-5435-CASE-SKILL-SEPARATION,
SCENARIO-LEARN-5435-VERIFY-BEFORE-STORE,
SCENARIO-LEARN-5435-TRAP-DEFLECTION,
SCENARIO-LEARN-5435-ROLLBACK,
SCENARIO-LEARN-5435-RAW-RETENTION-NO-WEIGHT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5435_verified_workflow_memory_csl_v494 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5435_verified_workflow_memory_csl_v494.py -q "
    "--no-cov -n 0"
)


def test_req_learn_5435_spec_declares_workflow_memory_contract() -> None:
    """REQ-LEARN-5435: OpenSpec anchors verified workflow memory."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5435") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5435",
        "SCENARIO-LEARN-5435-CASE-SKILL-SEPARATION",
        "SCENARIO-LEARN-5435-VERIFY-BEFORE-STORE",
        "SCENARIO-LEARN-5435-TRAP-DEFLECTION",
        "SCENARIO-LEARN-5435-ROLLBACK",
        "SCENARIO-LEARN-5435-RAW-RETENTION-NO-WEIGHT",
        str(exp.RESULT_RELATIVE_PATH),
        "positive examples, stale examples, poisoned examples",
        "semantically similar but infeasible retrieval traps",
        "Case memories and skill memories SHALL remain separate typed records",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5435_fixture_covers_required_workflow_families() -> None:
    """REQ-LEARN-5435-1: finite workflow covers positive and unsafe controls."""

    evaluation = exp.evaluate_verified_workflow_memory(root=REPO)
    episodes = evaluation["workflow_episodes"]
    families = {row["episode_family"] for row in episodes}

    assert evaluation["workflow_episode_count"] == len(episodes)
    assert families >= exp.REQUIRED_EPISODE_FAMILIES
    assert evaluation["workflow_episode_count"] >= 7
    assert all(row["workflow_steps"] for row in episodes)
    assert all(row["expected_evidence"] for row in episodes)
    assert all(row["raw_episode_receipt"]["checksum"].startswith("sha256:") for row in episodes)
    assert {row["memory_kind"] for row in episodes} == {"case", "skill"}
    assert evaluation["raw_episodes_retained"] is True


def test_scenario_learn_5435_case_skill_memories_route_separately() -> None:
    """SCENARIO-LEARN-5435-CASE-SKILL-SEPARATION: sidecars stay typed."""

    evaluation = exp.evaluate_verified_workflow_memory(root=REPO)
    routing = evaluation["routing_report"]
    promoted = evaluation["promoted_memories"]

    case_ids = {row["memory_id"] for row in promoted if row["memory_kind"] == "case"}
    skill_ids = {row["memory_id"] for row in promoted if row["memory_kind"] == "skill"}

    assert case_ids
    assert skill_ids
    assert case_ids == set(routing["active_case_memory_ids"])
    assert skill_ids == set(routing["active_skill_memory_ids"])
    assert case_ids.isdisjoint(skill_ids)
    assert evaluation["case_memory_count"] == len(case_ids)
    assert evaluation["skill_memory_count"] == len(skill_ids)
    assert all(row["routing_influence"] > 0 for row in promoted)
    assert all(row["promotion_status"] == "promoted" for row in promoted)


def test_scenario_learn_5435_verify_before_store_gates_all_routing() -> None:
    """SCENARIO-LEARN-5435-VERIFY-BEFORE-STORE: failed gates cannot route."""

    evaluation = exp.evaluate_verified_workflow_memory(root=REPO)

    assert evaluation["verify_before_store_pass_rate"] == pytest.approx(
        len(evaluation["promoted_memories"]) / evaluation["workflow_episode_count"]
    )
    assert evaluation["ontology_kernel_validation_rate"] == 1.0

    for row in evaluation["promoted_memories"]:
        assert all(row["gate_results"].values())
        assert row["active_for_routing"] is True
        assert row["promotion_decision"]["reasons"] == ["all_verify_before_store_gates_passed"]

    inactive = evaluation["rejected_memories"] + evaluation["abstained_memories"]
    assert inactive
    for row in inactive:
        assert not all(row["gate_results"].values())
        assert row["audit_retained"] is True
        assert row["active_for_routing"] is False
        assert row["routing_influence"] == 0
        assert row["raw_episode_receipt"]["checksum"].startswith("sha256:")

    stale_poisoned = [
        row for row in inactive if row["episode_family"] in {"stale", "poisoned"}
    ]
    scarce = [row for row in inactive if row["episode_family"] == "scarce_evidence"]
    assert stale_poisoned
    assert scarce
    assert all(row["promotion_status"] == "rejected" for row in stale_poisoned)
    assert all(row["promotion_status"] == "abstained" for row in scarce)

    promoted = deepcopy(evaluation["promoted_memories"][0])
    no_kernel = promoted | {"kernel_valid": False, "planner_valid": False}
    no_evidence = promoted | {"evidence_reliance_valid": False}
    no_rollback = promoted | {"rollback_pointer": None}
    no_resource = promoted | {"resource_savings": -1.0}

    assert exp.verify_before_store(no_kernel)["promotion_status"] == "rejected"
    assert "kernel_planner_validation_failed" in exp.verify_before_store(no_kernel)[
        "promotion_decision"
    ]["reasons"]
    assert exp.verify_before_store(no_evidence)["promotion_status"] == "abstained"
    assert "evidence_reliance_failed" in exp.verify_before_store(no_evidence)[
        "promotion_decision"
    ]["reasons"]
    assert exp.verify_before_store(no_rollback)["promotion_status"] == "rejected"
    assert "rollback_pointer_missing" in exp.verify_before_store(no_rollback)[
        "promotion_decision"
    ]["reasons"]
    assert exp.verify_before_store(no_resource)["promotion_status"] == "rejected"
    assert "resource_accounting_failed" in exp.verify_before_store(no_resource)[
        "promotion_decision"
    ]["reasons"]


def test_scenario_learn_5435_retrieval_traps_are_deflected_by_kernel_checks() -> None:
    """SCENARIO-LEARN-5435-TRAP-DEFLECTION: similar infeasible plans fail closed."""

    evaluation = exp.evaluate_verified_workflow_memory(root=REPO)
    traps = [
        row
        for row in evaluation["workflow_episodes"]
        if row["episode_family"] == "retrieval_trap"
    ]
    routing = evaluation["routing_report"]

    assert traps
    assert evaluation["retrieval_trap_deflection_rate"] == 1.0
    assert {row["memory_id"] for row in traps} == set(routing["deflected_trap_memory_ids"])
    for row in traps:
        assert row["semantic_similarity_to_positive"] >= 0.9
        assert row["promotion_status"] == "rejected"
        assert row["active_for_routing"] is False
        assert any(
            reason in row["promotion_decision"]["reasons"]
            for reason in ("kernel_planner_validation_failed", "ontology_validation_failed")
        )


def test_scenario_learn_5435_raw_retention_rollback_and_weight_boundary() -> None:
    """SCENARIO-LEARN-5435-ROLLBACK/RAW-RETENTION-NO-WEIGHT: audit state holds."""

    evaluation = exp.evaluate_verified_workflow_memory(root=REPO)

    assert evaluation["rollback_audit"] == {
        "bad_memory_id": "mem5435-poisoned-rollback-probe",
        "injected_into_case_sidecar": True,
        "injected_into_skill_sidecar": True,
        "rollback_removed_from_case_sidecar": True,
        "rollback_removed_from_skill_sidecar": True,
        "prior_case_sidecar_restored": True,
        "prior_skill_sidecar_restored": True,
        "retained_audit_record_after_rollback": True,
        "rollback_success": True,
    }
    assert evaluation["rollback_verified"] is True
    assert evaluation["raw_episodes_retained"] is True
    assert evaluation["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "verified_workflow_memory_sidecars_only",
    }


def test_req_learn_5435_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5435-7: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=[TEST_COMMAND])
    mapping_artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "mapped", "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert mapping_artifact["tests_run"] == [{"command": "mapped", "outcome": "passed"}]
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["workflow_episode_count"] == len(artifact["workflow_episodes"])
    assert artifact["raw_episodes_retained"] is True
    assert artifact["case_memory_count"] == len(artifact["routing_report"]["active_case_memory_ids"])
    assert artifact["skill_memory_count"] == len(artifact["routing_report"]["active_skill_memory_ids"])
    assert artifact["verify_before_store_pass_rate"] > 0.0
    assert artifact["ontology_kernel_validation_rate"] == 1.0
    assert artifact["retrieval_trap_deflection_rate"] == 1.0
    assert artifact["quality_preserved"] is True
    assert artifact["resource_delta"] > 0.0
    assert artifact["rollback_verified"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["verified_workflow_memory_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["research_conductor_modified"] is False
    exp.validate_artifact(artifact)


def test_req_learn_5435_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5435-7: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["verified_workflow_memory_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5435_blocked_artifact_reports_missing_tests() -> None:
    """REQ-LEARN-5435-7: missing test evidence keeps readiness blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["verified_workflow_memory_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5435_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5435-7: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    bad_missing = deepcopy(artifact)
    bad_missing.pop("workflow_episode_count")
    with pytest.raises(ValueError, match="workflow_episode_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["workflow_episode_count"] = "changed"
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
    bad_ready["verified_workflow_memory_ready"] = True
    bad_ready["retrieval_trap_deflection_rate"] = 0.0
    with pytest.raises(ValueError, match="verified_workflow_memory_ready"):
        exp.validate_artifact(bad_ready)

    bad_bool = deepcopy(artifact)
    bad_bool["raw_episodes_retained"] = "true"
    with pytest.raises(ValueError, match="raw_episodes_retained"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["workflow_episode_count"] = True
    with pytest.raises(ValueError, match="workflow_episode_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["resource_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="resource_delta"):
        exp.validate_artifact(bad_numeric)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor_modified"):
        exp.validate_artifact(bad_conductor)


def test_req_learn_5435_defensive_validation_and_gate_branches() -> None:
    """REQ-LEARN-5435-2/7: defensive gate and validation branches are covered."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])
    default_runs = exp.default_tests_run()

    assert default_runs[0]["command"] == TEST_COMMAND
    assert default_runs[1]["command"].startswith(".venv/bin/coverage run")
    assert exp._promoted_reliance_drift([]) == 0.0

    bad_kind = deepcopy(artifact["promoted_memories"][0])
    bad_kind["memory_kind"] = "world"
    bad_empty_steps = deepcopy(artifact["promoted_memories"][0])
    bad_empty_steps["workflow_steps"] = []
    bad_reversed = deepcopy(artifact["promoted_memories"][0])
    bad_reversed["workflow_steps"] = ["step:pocket_cut", "step:set_stock"]

    for row in (bad_kind, bad_empty_steps):
        rescored = exp.verify_before_store(row)
        assert rescored["promotion_status"] == "rejected"
        assert "ontology_validation_failed" in rescored["promotion_decision"]["reasons"]

    reversed_rescored = exp.verify_before_store(bad_reversed)
    assert reversed_rescored["promotion_status"] == "rejected"
    assert "kernel_planner_validation_failed" in reversed_rescored["promotion_decision"]["reasons"]

    bad_case_count = deepcopy(artifact)
    bad_case_count["case_memory_count"] += 1
    with pytest.raises(ValueError, match="case_memory_count"):
        exp.validate_artifact(bad_case_count)

    bad_skill_count = deepcopy(artifact)
    bad_skill_count["skill_memory_count"] += 1
    with pytest.raises(ValueError, match="skill_memory_count"):
        exp.validate_artifact(bad_skill_count)

    bad_complete_not_ready = deepcopy(artifact)
    bad_complete_not_ready["verified_workflow_memory_ready"] = False
    with pytest.raises(ValueError, match="verified_workflow_memory_ready"):
        exp.validate_artifact(bad_complete_not_ready)

    bad_blocked_ready = deepcopy(artifact)
    bad_blocked_ready["status"] = "blocked"
    with pytest.raises(ValueError, match="verified_workflow_memory_ready"):
        exp.validate_artifact(bad_blocked_ready)

    ready_error_mutations = [
        ("case_memory_count", 0),
        ("skill_memory_count", 0),
        ("verify_before_store_pass_rate", 0.0),
        ("ontology_kernel_validation_rate", 0.5),
        ("quality_preserved", False),
        ("rollback_verified", False),
        ("no_weight_mutation", False),
    ]
    for field, value in ready_error_mutations:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match="verified_workflow_memory_ready"):
            exp.validate_artifact(bad)
