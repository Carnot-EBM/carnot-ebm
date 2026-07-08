"""Tests for Exp5432 ontology-backed constraint-memory fixture.

Spec refs: REQ-STORE-5432, SCENARIO-STORE-5432.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5432_ontology_softlogic_constraint_memory_v494 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/constraint-store/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5432_ontology_softlogic_constraint_memory_v494.py -q "
    "--no-cov -n 0"
)


def test_req_store_5432_spec_declares_deterministic_ontology_contract() -> None:
    """REQ-STORE-5432: OpenSpec anchors the ontology verifier contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-STORE-5432") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-STORE-5432",
        "SCENARIO-STORE-5432",
        str(mod.RESULT_RELATIVE_PATH),
        "RDF-like triples",
        "SHACL-style validation",
        "deterministic planner/solver checks",
        "semantically plausible but infeasible retrievals",
        "`deterministic_ontology_verifier`",
        "`soft_logic_overrode_solver=false`",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_store_5432_fixture_covers_required_row_families() -> None:
    """SCENARIO-STORE-5432: rows cover valid, false, stale, unsupported, retrieval."""

    evaluation = mod.evaluate_fixture()
    rows = evaluation["evaluated_rows"]
    families = {row["fixture_family"] for row in rows}

    assert families >= mod.REQUIRED_FIXTURE_FAMILIES
    assert evaluation["ontology_fixture_count"] == len(rows)
    assert evaluation["ontology_fixture_count"] >= 10
    assert evaluation["triple_count"] == len(evaluation["final_triples"])
    assert all(row["soft_logic"]["exact_verification_routed"] for row in rows)
    assert all(row["soft_logic"]["advisory_only"] is True for row in rows)
    assert evaluation["soft_logic_overrode_solver"] is False
    assert evaluation["deterministic_solver_authority"] is True


def test_req_store_5432_false_triples_are_rejected() -> None:
    """REQ-STORE-5432: false triples and stale updates fail closed."""

    evaluation = mod.evaluate_fixture()
    false_rows = [
        row for row in evaluation["evaluated_rows"] if row["expected_truth"] == "false"
    ]

    assert false_rows
    assert evaluation["false_triple_rejection_rate"] == 1.0
    assert all(row["final_decision"] == "rejected" for row in false_rows)
    assert {
        row["fixture_family"] for row in false_rows
    } >= {"false_triple_update", "stale_relation_update", "infeasible_retrieval"}
    assert any("solver:" in reason for row in false_rows for reason in row["decision_reasons"])


def test_req_store_5432_valid_updates_are_preserved_in_memory_graph() -> None:
    """REQ-STORE-5432: valid triple updates remain in the final graph."""

    evaluation = mod.evaluate_fixture()
    valid_rows = [
        row for row in evaluation["evaluated_rows"] if row["expected_truth"] == "valid"
    ]
    final_triples = {tuple(triple) for triple in evaluation["final_triples"]}

    assert valid_rows
    assert evaluation["valid_update_preservation_rate"] == 1.0
    assert all(row["final_decision"] == "accepted" for row in valid_rows)
    for row in valid_rows:
        for triple in row["proposed_triples"]:
            assert tuple(triple) in final_triples


def test_req_store_5432_unsupported_writes_abstain() -> None:
    """REQ-STORE-5432: unsupported updates abstain instead of becoming facts."""

    evaluation = mod.evaluate_fixture()
    unsupported_rows = [
        row
        for row in evaluation["evaluated_rows"]
        if row["fixture_family"] == "unsupported_memory_write"
    ]
    final_triples = {tuple(triple) for triple in evaluation["final_triples"]}

    assert unsupported_rows
    assert evaluation["unsupported_update_abstention_rate"] == 1.0
    assert all(row["final_decision"] == "abstained" for row in unsupported_rows)
    assert all(
        tuple(triple) not in final_triples
        for row in unsupported_rows
        for triple in row["proposed_triples"]
    )


def test_scenario_store_5432_soft_residual_is_advisory_only() -> None:
    """SCENARIO-STORE-5432: soft residuals cannot override exact authority."""

    rows = mod.build_fixture_rows()
    false_row = next(row for row in rows if row["row_id"] == "row:false:range")
    valid_row = next(row for row in rows if row["row_id"] == "row:valid:part")
    missing_step_row = deepcopy(valid_row)
    missing_step_row.update(
        {
            "row_id": "row:retrieval:missing-step",
            "row_type": "retrieval",
            "fixture_family": "infeasible_retrieval",
            "expected_truth": "false",
            "proposed_triples": [],
            "retrieved_plan": ["step:inspect"],
        }
    )
    unknown_step_row = deepcopy(missing_step_row)
    unknown_step_row["retrieved_plan"] = ["step:unknown"]
    missing_valve_row = deepcopy(
        next(row for row in rows if row["row_id"] == "row:valid:drained")
    )
    missing_valve_row["tool_output_evidence"] = []

    false_eval = mod.evaluate_row(
        false_row,
        graph=mod.seed_graph(),
        soft_residual_override={"total": 0.0, "exact_verification_routed": False},
    )
    valid_eval = mod.evaluate_row(
        valid_row,
        graph=mod.seed_graph(),
        soft_residual_override={"total": 9.0, "exact_verification_routed": True},
    )

    assert false_eval["soft_logic"]["total"] == 0.0
    assert false_eval["soft_logic"]["exact_verification_routed"] is False
    assert false_eval["final_decision"] == "rejected"
    assert false_eval["deterministic_decision"] == "rejected"
    assert false_eval["soft_logic_overrode_solver"] is False

    assert valid_eval["soft_logic"]["total"] == 9.0
    assert valid_eval["final_decision"] == "accepted"
    assert valid_eval["deterministic_decision"] == "accepted"
    assert valid_eval["soft_logic_overrode_solver"] is False

    missing_step_eval = mod.evaluate_row(missing_step_row, graph=mod.seed_graph())
    unknown_step_eval = mod.evaluate_row(unknown_step_row, graph=mod.seed_graph())
    missing_valve_eval = mod.evaluate_row(missing_valve_row, graph=mod.seed_graph())

    assert missing_step_eval["final_decision"] == "rejected"
    assert any("solver:missing_step" in reason for reason in missing_step_eval["decision_reasons"])
    assert unknown_step_eval["final_decision"] == "rejected"
    assert unknown_step_eval["shacl"]["issues"] == ["unknown_step:step:unknown"]
    assert missing_valve_eval["final_decision"] == "rejected"
    assert "solver:valve_closed_evidence_missing" in missing_valve_eval[
        "decision_reasons"
    ]


def test_req_store_5432_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-STORE-5432: run() writes the required terminal artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=[TEST_COMMAND])
    mapping_artifact = mod.build_artifact(
        tests_run=[{"command": "mapped", "outcome": "passed"}]
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert mapping_artifact["tests_run"] == [{"command": "mapped", "outcome": "passed"}]
    assert artifact["status"] == "complete"
    assert artifact["ontology_constraint_memory_ready"] is True
    assert artifact["ontology_fixture_count"] == len(artifact["evaluated_rows"])
    assert artifact["triple_count"] == len(artifact["final_triples"])
    assert artifact["deterministic_solver_authority"] is True
    assert artifact["soft_logic_residuals_recorded"] is True
    assert artifact["soft_logic_overrode_solver"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["tests_run"][0]["command"] == TEST_COMMAND
    assert mod.default_tests_run()[0]["command"].endswith("--no-cov -n 0")


def test_req_store_5432_repository_artifact_matches_replay() -> None:
    """REQ-STORE-5432: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(tests_run=result["tests_run"])

    assert result == replay
    assert result["ontology_constraint_memory_ready"] is True
    assert result["inference_substrate"] == "deterministic_ontology_verifier"


def test_req_store_5432_blocked_artifact_reports_missing_tests() -> None:
    """REQ-STORE-5432: missing test evidence keeps readiness blocked."""

    artifact = mod.build_artifact(tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["ontology_constraint_memory_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    mod.validate_artifact(artifact)


def test_req_store_5432_validation_rejects_claim_drift() -> None:
    """REQ-STORE-5432: validation rejects malformed ready claims."""

    artifact = mod.build_artifact(tests_run=[TEST_COMMAND])

    bad_missing = deepcopy(artifact)
    bad_missing.pop("ontology_fixture_count")
    with pytest.raises(ValueError, match="ontology_fixture_count"):
        mod.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["ontology_fixture_count"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_override = deepcopy(artifact)
    bad_override["soft_logic_overrode_solver"] = True
    with pytest.raises(ValueError, match="soft_logic_overrode_solver"):
        mod.validate_artifact(bad_override)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete"
    bad_status["ontology_constraint_memory_ready"] = False
    with pytest.raises(ValueError, match="ontology_constraint_memory_ready"):
        mod.validate_artifact(bad_status)

    bad_ready_status = deepcopy(artifact)
    bad_ready_status["status"] = "blocked"
    with pytest.raises(ValueError, match="ontology_constraint_memory_ready"):
        mod.validate_artifact(bad_ready_status)

    bad_ready = deepcopy(artifact)
    bad_ready["ontology_constraint_memory_ready"] = True
    bad_ready["valid_update_preservation_rate"] = 0.5
    with pytest.raises(ValueError, match="ontology_constraint_memory_ready"):
        mod.validate_artifact(bad_ready)

    bad_authority = deepcopy(artifact)
    bad_authority["deterministic_solver_authority"] = False
    with pytest.raises(ValueError, match="ontology_constraint_memory_ready"):
        mod.validate_artifact(bad_authority)

    bad_count = deepcopy(artifact)
    bad_count["ontology_fixture_count"] = 999
    with pytest.raises(ValueError, match="ontology_fixture_count"):
        mod.validate_artifact(bad_count)

    bad_triple_count = deepcopy(artifact)
    bad_triple_count["triple_count"] = 999
    with pytest.raises(ValueError, match="triple_count"):
        mod.validate_artifact(bad_triple_count)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(bad_tests)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor_modified"):
        mod.validate_artifact(bad_conductor)
