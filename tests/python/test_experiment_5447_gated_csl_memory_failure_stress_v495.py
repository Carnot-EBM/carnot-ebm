"""Tests for Exp5447 gated CSL memory failure stress.

Spec refs: REQ-LEARN-5447,
SCENARIO-LEARN-5447-ATTRIBUTION,
SCENARIO-LEARN-5447-CONTROLS,
SCENARIO-LEARN-5447-ROLLBACK,
SCENARIO-LEARN-5447-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5447_gated_csl_memory_failure_stress_v495 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5447_gated_csl_memory_failure_stress_v495.py "
    "-q --no-cov -n 0"
)


def test_req_learn_5447_spec_declares_failure_stress_contract() -> None:
    """REQ-LEARN-5447: OpenSpec anchors the memory failure stress task."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5447") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5447",
        "SCENARIO-LEARN-5447-ATTRIBUTION",
        "SCENARIO-LEARN-5447-CONTROLS",
        "SCENARIO-LEARN-5447-ROLLBACK",
        "SCENARIO-LEARN-5447-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        "governed_csl_loop_ready=true",
        "summarization loss, storage collision, retrieval collision, stale rule reuse",
        "summarization, storage, retrieval, replay, decay, access-control",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5447_precondition_records_exact_upstream_policy() -> None:
    """REQ-LEARN-5447-1: Exp5446 gate and checksum provenance are explicit."""

    evaluation = exp.evaluate_memory_failure_stress(root=REPO)
    upstream = evaluation["upstream_governance_policy"]
    source = json.loads((REPO / exp.EXP5446_RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert evaluation["gated_upstream_ready"] is True
    assert upstream["governed_csl_loop_ready"] is True
    assert upstream["upstream_reproducibility_checksum"] == source["reproducibility_checksum"]
    assert upstream["temporal_decay_policy"] == source["temporal_decay_policy"]
    assert upstream["source_file_checksums"] == source["source_file_checksums"]
    assert upstream["governance_gates"] == list(exp.GOVERNANCE_GATES)

    blocked = exp.evaluate_memory_failure_stress(
        root=REPO,
        upstream_artifact={"governed_csl_loop_ready": False},
    )
    assert blocked["gated_upstream_ready"] is False
    assert all(row["governed_status"] == "blocked_precondition" for row in blocked["failure_cases"])
    assert blocked["memory_failure_case_count"] == len(blocked["failure_cases"])


def test_scenario_learn_5447_operation_attribution_counts_match_cases() -> None:
    """SCENARIO-LEARN-5447-ATTRIBUTION: failures map to one memory operation."""

    evaluation = exp.evaluate_memory_failure_stress(root=REPO)
    rows = evaluation["failure_cases"]
    families = {row["case_family"] for row in rows}
    operations = {row["failure_operation"] for row in rows}
    expected_counts = Counter(row["failure_operation"] for row in rows)

    assert evaluation["memory_failure_case_count"] == len(rows)
    assert evaluation["memory_failure_case_count"] >= 7
    assert families >= exp.REQUIRED_CASE_FAMILIES
    assert operations == set(exp.FAILURE_OPERATIONS)
    assert evaluation["failure_operation_counts"] == dict(sorted(expected_counts.items()))

    for row in rows:
        assert row["failure_operation"] in exp.FAILURE_OPERATIONS
        assert row["governed_status"] in {"rejected", "verification_routed"}
        assert row["active_for_routing"] is False
        assert row["routing_influence"] == 0
        assert row["rejection_decision"]["operation"] == row["failure_operation"]
        assert row["raw_memory_receipt"]["checksum"].startswith("sha256:")


def test_scenario_learn_5447_controls_and_deflection_rates() -> None:
    """SCENARIO-LEARN-5447-CONTROLS: governed memory deflects unsafe controls."""

    evaluation = exp.evaluate_memory_failure_stress(root=REPO)
    controls = evaluation["control_metrics"]
    case_id_sets = evaluation["control_case_id_sets"]

    assert len({tuple(ids) for ids in case_id_sets.values()}) == 1
    assert set(controls) == {"always_full_context", "no_memory", "ungated_memory", "governed_memory"}
    assert controls["governed_memory"]["quality_score"] == pytest.approx(
        controls["always_full_context"]["quality_score"]
    )
    assert controls["ungated_memory"]["unsafe_false_accepts"] > 0
    assert controls["governed_memory"]["unsafe_false_accepts"] == 0
    assert evaluation["stale_memory_deflection_rate"] == 1.0
    assert evaluation["poisoned_memory_deflection_rate"] == 1.0
    assert evaluation["retrieval_collision_deflection_rate"] == 1.0
    assert evaluation["negative_transfer_deflection_rate"] == 1.0
    assert evaluation["quality_delta_vs_always_full"] == pytest.approx(0.0)
    assert evaluation["unsafe_false_accepts"] == 0


def test_scenario_learn_5447_rollback_blocks_rejected_memory_influence() -> None:
    """SCENARIO-LEARN-5447-ROLLBACK: rollback removes bad-memory influence."""

    evaluation = exp.evaluate_memory_failure_stress(root=REPO)
    rollback = evaluation["rollback_audit"]
    rejected = set(evaluation["rejected_memory_ids"])
    rolled_back = set(rollback["rolled_back_memory_ids"])

    assert evaluation["rollback_recovery_rate"] == 1.0
    assert rollback["rollback_success"] is True
    assert rollback["prior_active_sidecar_restored"] is True
    assert rollback["rollback_removed_from_active_sidecar"] is True
    assert rollback["injected_bad_memory_id"] in rolled_back
    assert rejected
    assert rolled_back <= rejected
    assert evaluation["post_rollback_decisions"]
    assert all(
        rejected.isdisjoint(decision["cited_memory_ids"])
        for decision in evaluation["post_rollback_decisions"]
    )


def test_scenario_learn_5447_no_weight_mutation_boundary() -> None:
    """SCENARIO-LEARN-5447-NO-WEIGHT-MUTATION: stress learning is sidecar-only."""

    evaluation = exp.evaluate_memory_failure_stress(root=REPO)

    assert evaluation["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "gated_memory_failure_stress_sidecars_only",
    }


def test_req_learn_5447_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5447-7: run() writes the terminal stress artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=[TEST_COMMAND])
    mapping_artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "mapped", "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert mapping_artifact["tests_run"] == [{"command": "mapped", "outcome": "passed"}]
    assert artifact["status"] == "complete"
    assert artifact["gated_upstream_ready"] is True
    assert artifact["memory_failure_case_count"] == len(artifact["failure_cases"])
    assert artifact["failure_operation_counts"] == dict(
        sorted(Counter(row["failure_operation"] for row in artifact["failure_cases"]).items())
    )
    assert artifact["stale_memory_deflection_rate"] == 1.0
    assert artifact["poisoned_memory_deflection_rate"] == 1.0
    assert artifact["retrieval_collision_deflection_rate"] == 1.0
    assert artifact["negative_transfer_deflection_rate"] == 1.0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["quality_delta_vs_always_full"] >= 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["no_weight_mutation"] is True
    assert artifact["csl_memory_stress_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["research_conductor_modified"] is False
    exp.validate_artifact(artifact)


def test_req_learn_5447_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5447-7: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["csl_memory_stress_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5447_blocked_artifacts_fail_closed() -> None:
    """REQ-LEARN-5447-1/7: missing tests or upstream gate block readiness."""

    no_tests = exp.build_artifact(root=REPO, tests_run=[])
    bad_upstream = exp.build_artifact(
        root=REPO,
        tests_run=[TEST_COMMAND],
        upstream_artifact={"governed_csl_loop_ready": False},
    )

    assert no_tests["status"] == "blocked"
    assert no_tests["csl_memory_stress_ready"] is False
    assert "tests_recorded" in no_tests["readiness_checks"]["failed_checks"]
    assert bad_upstream["status"] == "blocked"
    assert bad_upstream["gated_upstream_ready"] is False
    assert "gated_upstream_ready" in bad_upstream["readiness_checks"]["failed_checks"]
    exp.validate_artifact(no_tests)
    exp.validate_artifact(bad_upstream)


def test_req_learn_5447_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5447-7: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    bad_missing = deepcopy(artifact)
    bad_missing.pop("memory_failure_case_count")
    with pytest.raises(ValueError, match="memory_failure_case_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["gated_upstream_ready"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_model"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_bool = deepcopy(artifact)
    bad_bool["no_weight_mutation"] = "true"
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["memory_failure_case_count"] = True
    with pytest.raises(ValueError, match="memory_failure_case_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["quality_delta_vs_always_full"] = {"value": 1.0}
    with pytest.raises(ValueError, match="quality_delta_vs_always_full"):
        exp.validate_artifact(bad_numeric)

    bad_rate = deepcopy(artifact)
    bad_rate["stale_memory_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="csl_memory_stress_ready"):
        exp.validate_artifact(bad_rate)

    bad_rate_invalid = deepcopy(artifact)
    bad_rate_invalid["poisoned_memory_deflection_rate"] = 1.5
    with pytest.raises(ValueError, match="poisoned_memory_deflection_rate"):
        exp.validate_artifact(bad_rate_invalid)

    bad_counts = deepcopy(artifact)
    bad_counts["failure_operation_counts"]["retrieval"] += 1
    with pytest.raises(ValueError, match="failure_operation_counts"):
        exp.validate_artifact(bad_counts)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor_modified"):
        exp.validate_artifact(bad_conductor)

    bad_readiness = deepcopy(artifact)
    bad_readiness["readiness_checks"]["all_passed"] = False
    with pytest.raises(ValueError, match="csl_memory_stress_ready"):
        exp.validate_artifact(bad_readiness)


def test_req_learn_5447_defensive_readiness_and_helper_branches() -> None:
    """REQ-LEARN-5447-3/7: defensive readiness branches fail closed."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])
    default_runs = exp.default_tests_run()

    assert default_runs[0]["command"] == TEST_COMMAND
    assert default_runs[1]["command"].startswith(".venv/bin/coverage run")
    assert default_runs[2]["command"] == ".venv/bin/pytest tests/python -q"
    assert exp._deflection_rate([], "stale") == 0.0

    ready_error_mutations = [
        ("gated_upstream_ready", False),
        ("stale_memory_deflection_rate", 0.0),
        ("poisoned_memory_deflection_rate", 0.0),
        ("retrieval_collision_deflection_rate", 0.0),
        ("negative_transfer_deflection_rate", 0.0),
        ("rollback_recovery_rate", 0.0),
        ("quality_delta_vs_always_full", -0.1),
        ("unsafe_false_accepts", 1),
        ("no_weight_mutation", False),
    ]
    for field, value in ready_error_mutations:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match="csl_memory_stress_ready"):
            exp.validate_artifact(bad)

    bad_case_count = deepcopy(artifact)
    bad_case_count["memory_failure_case_count"] += 1
    with pytest.raises(ValueError, match="memory_failure_case_count"):
        exp.validate_artifact(bad_case_count)

    bad_complete_not_ready = deepcopy(artifact)
    bad_complete_not_ready["csl_memory_stress_ready"] = False
    with pytest.raises(ValueError, match="csl_memory_stress_ready"):
        exp.validate_artifact(bad_complete_not_ready)

    bad_blocked_ready = deepcopy(artifact)
    bad_blocked_ready["status"] = "blocked"
    with pytest.raises(ValueError, match="csl_memory_stress_ready"):
        exp.validate_artifact(bad_blocked_ready)
