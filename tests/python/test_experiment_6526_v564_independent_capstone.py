"""Tests for Exp6526 V564 independent evidence capstone.

Spec refs: REQ-CAPSTONE-6526, SCENARIO-CAPSTONE-6526-INVENTORY,
SCENARIO-CAPSTONE-6526-ROW-RECONSTRUCTION,
SCENARIO-CAPSTONE-6526-MISSING-TASK-CLOSURE,
SCENARIO-CAPSTONE-6526-GATE-SPELLING,
SCENARIO-CAPSTONE-6526-VERDICT-NEXT-STATE,
SCENARIO-CAPSTONE-6526-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6526_v564_independent_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-CAPSTONE-6526: build the capstone from checked-in V564 evidence."""

    root = tmp_path_factory.mktemp("exp6526")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_capstone_6526_spec_declares_contract() -> None:
    """REQ-CAPSTONE-6526: OpenSpec owns the Exp6526 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6526") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CAPSTONE-6526-INVENTORY",
        "SCENARIO-CAPSTONE-6526-ROW-RECONSTRUCTION",
        "SCENARIO-CAPSTONE-6526-MISSING-TASK-CLOSURE",
        "SCENARIO-CAPSTONE-6526-GATE-SPELLING",
        "SCENARIO-CAPSTONE-6526-VERDICT-NEXT-STATE",
        "SCENARIO-CAPSTONE-6526-SCHEMA",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=false`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_6526_inventory_schema_and_checksum(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6526-INVENTORY/SCHEMA: inventory is complete."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_partial_v564_independent_capstone"
    assert artifact["honest_verdict"].startswith("complete_partial_")
    assert artifact["verdict_class"] == "partial"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    rows = artifact["task_inventory_rows"]
    assert len(rows) == len(mod.EXPECTED_TASKS)
    assert {row["task_id"] for row in rows} == set(mod.EXPECTED_TASKS)
    assert all(row["exists"] is True for row in rows)
    assert all(str(row["sha256"]).startswith("sha256:") for row in rows)
    assert all(row["row_support"]["row_count"] >= 0 for row in rows)
    assert all(row["authority"] in mod.AUTHORITY_CLASSES for row in rows)
    assert {row["task_id"]: row["observed_value"] for row in rows}["6517"] == 1.0
    assert {row["task_id"]: row["required_field"] for row in rows}["6524"] == (
        "arc_generalization_slot_complete_score"
    )
    assert {row["task_id"]: row["verdict_class"] for row in rows}["6524"] == "blocked"


def test_scenario_capstone_6526_gate_spelling_retired_and_transaction(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6526-GATE-SPELLING: gate contracts are exact."""

    gate_rows = artifact["gate_contract_rows"]
    assert {row["gate_id"] for row in gate_rows} >= {
        "6516_requires_6514_atomic_artifact_contract_ready_score",
        "6516_requires_6515_v564_method_contract_ready_score",
        "6520_requires_6519_certified_structural_headroom_score",
        "6523_requires_6522_csl_execution_complete_score",
    }
    assert all(row["field_spelled_in_upstream"] is True for row in gate_rows)
    assert all(row["field_spelled_in_roadmap"] is True for row in gate_rows)
    assert all(row["gate_passed"] is True for row in gate_rows)

    retired = artifact["retired_scope_audit"]
    assert retired["retired_structured_dependency_violation_count"] == 0
    assert retired["no_structured_dependency_names_retired_task"] is True
    assert set(retired["retired_task_ids"]) == set(mod.RETIRED_TASK_IDS)

    transaction = artifact["transaction_audit"]
    assert transaction["exp6516_used_exp6514_transaction"] is True
    assert transaction["final_transaction_verified"] is True
    assert transaction["shards_complete_and_resumable"] is True
    assert transaction["direct_immutable_inputs"]["exp6504"]["structured_dependency_used"] is False
    assert transaction["direct_immutable_inputs"]["exp6510"]["structured_dependency_used"] is False


def test_scenario_capstone_6526_row_reconstruction_and_authority(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6526-ROW-RECONSTRUCTION: claims reduce from rows."""

    claims = {row["claim_id"]: row for row in artifact["comparative_claim_rows"]}
    assert claims["structural_headroom"]["eligibility"] == "eligible_positive"
    assert claims["structural_headroom"]["observed_value"] == 1.0
    assert claims["structural_headroom"]["row_support"]["row_count"] == 136
    assert claims["learned_router"]["eligibility"] == "eligible_positive"
    assert claims["learned_router"]["observed_value"] == 1.0
    assert claims["learned_router"]["row_support"]["held_benefit_beyond_best_structural_units"] == 28
    assert claims["continuous_self_learning"]["eligibility"] == "eligible_positive"
    assert claims["continuous_self_learning"]["observed_value"] == 1.0
    assert claims["adaptive_validation"]["eligibility"] == "eligible_positive"
    assert claims["arc_generalization"]["eligibility"] == "blocked_missing_evidence"
    assert claims["hardware_continuity"]["eligibility"] == "preserve_blocked_no_command"

    assert artifact["learned_router_claim_eligible_score"] == 1.0
    assert artifact["continuous_self_learning_claim_eligible_score"] == 1.0
    assert artifact["structural_headroom_decision"]["next_state"] == "expand_after_positive"
    assert artifact["adaptive_validation_decision"]["score"] == 1.0
    assert artifact["arc_generalization_decision"]["solve_claim_made"] is False
    assert artifact["hardware_continuity_decision"]["hardware_command_count"] == 0
    assert artifact["hardware_continuity_decision"]["hardware_speedup_claim"] is False

    exact = artifact["exact_authority_audit"]
    assert exact["exact_solver_is_release_authority"] is True
    assert exact["candidate_preservation_passed"] is True
    assert exact["exception_table_held_contamination_free"] is True
    assert exact["zero_unsafe_writes"] is True
    assert exact["zero_unsafe_uses"] is True
    assert exact["final_full_audit_complete"] is True


def test_scenario_capstone_6526_missing_closure_next_states_and_rows(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6526-MISSING-TASK-CLOSURE: blocks stay blocked."""

    next_states = {row["lineage_id"]: row["next_state"] for row in artifact["next_state_rows"]}
    assert next_states["learned_router"] == "expand_after_positive"
    assert next_states["continuous_self_learning"] == "expand_after_positive"
    assert next_states["arc_generalization"] == "preserve_watch"
    assert next_states["hardware_continuity"] == "preserve_watch"
    assert set(next_states.values()) <= set(mod.NEXT_STATES)

    discrepancies = {row["discrepancy_id"]: row for row in artifact["discrepancy_rows"]}
    assert discrepancies["arc_missing_outcome_bearing_receipts"]["severity"] == "lineage_blocked"
    assert discrepancies["gatemate_missing_new_physical_receipt"]["severity"] == "lineage_blocked"
    assert all(row["promoted_to_success"] is False for row in discrepancies.values())

    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["verdict_class_from_rows"] == "partial"
    assert aggregate["blocked_lineage_count"] == 2
    assert aggregate["learned_router_claim_eligible_score_from_rows"] == 1.0
    assert aggregate["continuous_self_learning_claim_eligible_score_from_rows"] == 1.0
    assert aggregate["per_unit_row_type_counts"]["task_inventory"] == len(mod.EXPECTED_TASKS)
    assert artifact["gate_check_summary"]["all_capstone_checks_passed"] is True

    per_rows = artifact["per_unit_rows"]
    assert len(per_rows) == aggregate["per_unit_row_count"]
    assert {row["row_type"] for row in per_rows} >= {
        "task_inventory",
        "comparative_claim",
        "gate_contract",
        "discrepancy",
        "protected_file",
    }


def test_scenario_capstone_6526_validation_rejects_bad_artifacts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6526-SCHEMA: malformed capstones fail validation."""

    missing = deepcopy(artifact)
    missing.pop("task_inventory_rows")
    assert "missing required field: task_inventory_rows" in mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["learned_router_claim_eligible_score"] = 0.0
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    bad_next = deepcopy(artifact)
    bad_next["next_state_rows"][0]["next_state"] = "rerun_big"
    assert "next_state_rows[0].next_state invalid: rerun_big" in mod.validate_artifact(bad_next)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in mod.validate_artifact(bad_oracle)


def test_scenario_capstone_6526_cli_roundtrip(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6526-SCHEMA: CLI writes and validates JSON."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert mod.validate_artifact(payload) == []
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
