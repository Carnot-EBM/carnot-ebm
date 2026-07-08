"""Tests for Exp5436 CSL memory transfer stress.

Spec refs: REQ-LEARN-5436,
SCENARIO-LEARN-5436-GATE,
SCENARIO-LEARN-5436-NEGATIVE-TRANSFER,
SCENARIO-LEARN-5436-DRIFT,
SCENARIO-LEARN-5436-ROLLBACK,
SCENARIO-LEARN-5436-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5436_csl_memory_transfer_stress_v494 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5436_csl_memory_transfer_stress_v494.py "
    "-q --no-cov -n 0"
)


def test_req_learn_5436_spec_declares_transfer_stress_contract() -> None:
    """REQ-LEARN-5436: OpenSpec anchors transfer stress before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5436") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5436",
        "SCENARIO-LEARN-5436-GATE",
        "SCENARIO-LEARN-5436-NEGATIVE-TRANSFER",
        "SCENARIO-LEARN-5436-DRIFT",
        "SCENARIO-LEARN-5436-ROLLBACK",
        "SCENARIO-LEARN-5436-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        "verified_workflow_memory_ready=true",
        "in-domain, near-domain, out-of-domain, stale, and adversarial",
        "Unsupported or ambiguous transfers SHALL abstain or route to verification",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5436_precondition_and_fixture_coverage() -> None:
    """REQ-LEARN-5436-1/2: Exp5435 gate and transfer families are covered."""

    evaluation = exp.evaluate_transfer_stress(root=REPO)
    rows = evaluation["transfer_rows"]
    families = {row["transfer_family"] for row in rows}
    kinds = {row["source_memory_kind"] for row in rows}

    assert evaluation["source_readiness"] == {
        "exp5435_verified_workflow_memory_ready": True
    }
    assert evaluation["transfer_fixture_count"] == len(rows)
    assert evaluation["transfer_fixture_count"] >= 8
    assert families >= exp.REQUIRED_TRANSFER_FAMILIES
    assert kinds == {"case", "skill"}
    assert all(row["source_memory_id"] for row in rows)
    assert all(row["target_fixture_id"] for row in rows)
    assert all(row["raw_transfer_receipt"]["checksum"].startswith("sha256:") for row in rows)

    blocked = exp.evaluate_transfer_stress(
        root=REPO,
        source_artifact={"verified_workflow_memory_ready": False, "promoted_memories": []},
    )
    assert blocked["source_readiness"]["exp5435_verified_workflow_memory_ready"] is False
    assert blocked["promoted_transfer_count"] == 0
    assert all(row["routing_influence"] == 0 for row in blocked["transfer_rows"])


def test_scenario_learn_5436_ontology_kernel_gates_precede_routing() -> None:
    """SCENARIO-LEARN-5436-GATE: failed gates cannot influence routing."""

    evaluation = exp.evaluate_transfer_stress(root=REPO)
    promoted = evaluation["promoted_transfers"]
    inactive = evaluation["quarantined_transfers"] + evaluation["verification_transfers"]

    assert promoted
    assert inactive
    assert evaluation["promoted_transfer_count"] == len(promoted)
    for row in promoted:
        assert all(row["gate_results"].values())
        assert row["transfer_status"] == "promoted"
        assert row["routing_influence"] > 0
        assert row["guarded_quality_delta"] >= 0.0

    for row in inactive:
        assert row["routing_influence"] == 0
        assert row["active_for_routing"] is False
        assert not all(row["gate_results"].values())
        assert row["transfer_status"] in {"quarantined", "verification_routed", "abstained"}

    base = deepcopy(promoted[0])
    no_source = exp.score_transfer_row(base, source_ready=False)
    no_ontology = exp.score_transfer_row(base | {"ontology_valid": False})
    no_kernel = exp.score_transfer_row(base | {"kernel_valid": False})

    assert no_source["transfer_status"] == "blocked_precondition"
    assert "source_readiness_failed" in no_source["transfer_decision"]["reasons"]
    assert no_source["routing_influence"] == 0
    assert no_ontology["transfer_status"] == "verification_routed"
    assert "ontology_check_failed" in no_ontology["transfer_decision"]["reasons"]
    assert no_kernel["transfer_status"] == "verification_routed"
    assert "kernel_check_failed" in no_kernel["transfer_decision"]["reasons"]


def test_scenario_learn_5436_negative_transfer_is_quarantined_or_deflected() -> None:
    """SCENARIO-LEARN-5436-NEGATIVE-TRANSFER: unsafe transfer is deflected."""

    evaluation = exp.evaluate_transfer_stress(root=REPO)
    rows = evaluation["transfer_rows"]
    negative = [row for row in rows if row["ungated_quality_delta"] < 0.0]
    stale_adversarial = [
        row for row in rows if row["transfer_family"] in {"stale", "adversarial"}
    ]
    unsupported_or_ambiguous = [
        row for row in rows if row["transfer_family"] in {"out_of_domain", "ambiguous"}
    ]

    assert negative
    assert stale_adversarial
    assert unsupported_or_ambiguous
    assert evaluation["negative_transfer_deflection_rate"] == 1.0
    assert evaluation["quarantined_transfer_count"] == len(evaluation["quarantined_transfers"])
    assert evaluation["quarantined_transfer_count"] >= 2
    assert all(row["negative_transfer_deflected"] is True for row in negative)
    assert all(row["routing_influence"] == 0 for row in negative)
    assert all(row["transfer_status"] == "quarantined" for row in stale_adversarial)
    assert all(
        row["transfer_status"] in {"verification_routed", "abstained"}
        for row in unsupported_or_ambiguous
    )


def test_scenario_learn_5436_reliance_drift_is_measured_and_bounded() -> None:
    """SCENARIO-LEARN-5436-DRIFT: high drift is visible only when deflected."""

    evaluation = exp.evaluate_transfer_stress(root=REPO)
    rows = evaluation["transfer_rows"]
    promoted = evaluation["promoted_transfers"]
    high_drift = [
        row for row in rows if row["reliance_drift"] >= exp.MAX_PROMOTED_RELIANCE_DRIFT
    ]

    assert high_drift
    assert evaluation["reliance_drift_metric"] == max(row["reliance_drift"] for row in rows)
    assert all(row["transfer_status"] != "promoted" for row in high_drift)
    assert all(row["reliance_drift"] < exp.MAX_PROMOTED_RELIANCE_DRIFT for row in promoted)
    assert evaluation["in_domain_quality_delta"] > 0.0
    assert evaluation["out_of_domain_quality_delta"] == pytest.approx(0.0)
    assert evaluation["resource_delta"] > 0.0


def test_scenario_learn_5436_rollback_and_weight_boundary() -> None:
    """SCENARIO-LEARN-5436-ROLLBACK/NO-WEIGHT-MUTATION: recovery holds."""

    evaluation = exp.evaluate_transfer_stress(root=REPO)

    assert evaluation["rollback_audit"] == {
        "bad_transfer_id": "transfer5436-bad-promotion-probe",
        "injected_into_active_transfer_sidecar": True,
        "rollback_removed_from_active_transfer_sidecar": True,
        "prior_transfer_sidecar_restored": True,
        "retained_audit_record_after_rollback": True,
        "rollback_success": True,
    }
    assert evaluation["rollback_verified"] is True
    assert evaluation["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "csl_transfer_sidecars_only",
    }


def test_req_learn_5436_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5436-7: run() writes the terminal transfer-stress artifact."""

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
    assert artifact["transfer_fixture_count"] == len(artifact["transfer_rows"])
    assert artifact["in_domain_quality_delta"] > 0.0
    assert artifact["out_of_domain_quality_delta"] == 0.0
    assert artifact["resource_delta"] > 0.0
    assert artifact["negative_transfer_deflection_rate"] == 1.0
    assert artifact["reliance_drift_metric"] >= exp.MAX_PROMOTED_RELIANCE_DRIFT
    assert artifact["promoted_transfer_count"] == len(artifact["promoted_transfers"])
    assert artifact["quarantined_transfer_count"] == len(artifact["quarantined_transfers"])
    assert artifact["rollback_verified"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["csl_transfer_stress_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["research_conductor_modified"] is False
    exp.validate_artifact(artifact)


def test_req_learn_5436_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5436-7: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["csl_transfer_stress_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5436_blocked_artifact_reports_missing_tests() -> None:
    """REQ-LEARN-5436-7: missing test evidence keeps readiness blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["csl_transfer_stress_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5436_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5436-7: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    bad_missing = deepcopy(artifact)
    bad_missing.pop("transfer_fixture_count")
    with pytest.raises(ValueError, match="transfer_fixture_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["transfer_fixture_count"] = "changed"
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
    bad_int["transfer_fixture_count"] = True
    with pytest.raises(ValueError, match="transfer_fixture_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["resource_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="resource_delta"):
        exp.validate_artifact(bad_numeric)

    bad_rate = deepcopy(artifact)
    bad_rate["negative_transfer_deflection_rate"] = 0.0
    with pytest.raises(ValueError, match="csl_transfer_stress_ready"):
        exp.validate_artifact(bad_rate)

    bad_rate_invalid = deepcopy(artifact)
    bad_rate_invalid["negative_transfer_deflection_rate"] = 1.5
    with pytest.raises(ValueError, match="negative_transfer_deflection_rate"):
        exp.validate_artifact(bad_rate_invalid)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor_modified"):
        exp.validate_artifact(bad_conductor)


def test_req_learn_5436_defensive_readiness_and_helper_branches() -> None:
    """REQ-LEARN-5436-3/7: defensive readiness branches fail closed."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])
    default_runs = exp.default_tests_run()

    assert default_runs[0]["command"] == TEST_COMMAND
    assert default_runs[1]["command"].startswith(".venv/bin/coverage run")
    assert exp._mean_delta([], "guarded_quality_delta") == 0.0
    assert exp._deflection_rate([]) == 0.0

    ready_error_mutations = [
        ("promoted_transfer_count", 0),
        ("quarantined_transfer_count", 0),
        ("negative_transfer_deflection_rate", 0.5),
        ("rollback_verified", False),
        ("no_weight_mutation", False),
    ]
    for field, value in ready_error_mutations:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match="csl_transfer_stress_ready"):
            exp.validate_artifact(bad)

    bad_fixture_count = deepcopy(artifact)
    bad_fixture_count["transfer_fixture_count"] += 1
    with pytest.raises(ValueError, match="transfer_fixture_count"):
        exp.validate_artifact(bad_fixture_count)

    bad_promoted_count = deepcopy(artifact)
    bad_promoted_count["promoted_transfer_count"] += 1
    with pytest.raises(ValueError, match="promoted_transfer_count"):
        exp.validate_artifact(bad_promoted_count)

    bad_quarantined_count = deepcopy(artifact)
    bad_quarantined_count["quarantined_transfer_count"] += 1
    with pytest.raises(ValueError, match="quarantined_transfer_count"):
        exp.validate_artifact(bad_quarantined_count)

    bad_complete_not_ready = deepcopy(artifact)
    bad_complete_not_ready["csl_transfer_stress_ready"] = False
    with pytest.raises(ValueError, match="csl_transfer_stress_ready"):
        exp.validate_artifact(bad_complete_not_ready)

    bad_blocked_ready = deepcopy(artifact)
    bad_blocked_ready["status"] = "blocked"
    with pytest.raises(ValueError, match="csl_transfer_stress_ready"):
        exp.validate_artifact(bad_blocked_ready)
