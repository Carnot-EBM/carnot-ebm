"""Tests for Exp6560 V567 independent capstone.

Spec refs: REQ-CAPSTONE-6560,
SCENARIO-CAPSTONE-6560-INVENTORY,
SCENARIO-CAPSTONE-6560-RECOMPUTE,
SCENARIO-CAPSTONE-6560-CLOSED-CLASSES,
SCENARIO-CAPSTONE-6560-ADOPTION,
SCENARIO-CAPSTONE-6560-PUBLICATION-HANDOFF,
SCENARIO-CAPSTONE-6560-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6560_v567_independent_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6560", "exit_code": 0}]


def _tool_receipts() -> dict[str, Any]:
    return {
        "adversarial_verify_upstream": {
            "command": "stub adversarial",
            "exit_code": 0,
            "json": {
                "reports": [
                    {
                        "artifact": "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json",
                        "flag_count": 1,
                        "max_severity": 1,
                        "flags": [{"kind": "SUBSTRATE_HAS_NO_DURATION_FLOOR", "severity": "warn"}],
                    }
                ],
                "flagged_count": 1,
            },
        },
        "row_consistency_upstream": {
            "command": "stub row lint",
            "exit_code": 0,
            "stdout": "verdict-row-consistency: checked 8, skipped 4",
        },
        "exclusion_manifest_lint": {
            "command": "stub exclusion lint",
            "exit_code": 1,
            "stdout": "SCOPE_MATCHED_PRIOR_FAILURE exp6557",
        },
        "arc_orphan_solver_lint": {"command": "stub orphan", "exit_code": 0, "stdout": "ok"},
        "arc_levelup_guarantee_lint": {"command": "stub arc floor", "exit_code": 0, "stdout": "ok"},
        "publication_gate": {
            "command": "stub publication",
            "exit_code": 0,
            "json": {
                "paper_ready": False,
                "gates": {
                    "G1": {"pass": True, "detail": "headline measured"},
                    "G2": {"pass": False, "detail": "external reproducer still open"},
                    "G3": {"pass": True, "detail": "narrowing clean"},
                    "G4": {"pass": True, "detail": "numbers trace"},
                },
                "unmet_gates": ["G2"],
            },
        },
    }


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "experiment_6560.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        tool_receipts_override=_tool_receipts(),
    )


def test_req_capstone_6560_spec_declares_contract() -> None:
    """REQ-CAPSTONE-6560: OpenSpec owns the Exp6560 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6560") :]
    for marker in (
        "SCENARIO-CAPSTONE-6560-INVENTORY",
        "SCENARIO-CAPSTONE-6560-RECOMPUTE",
        "SCENARIO-CAPSTONE-6560-CLOSED-CLASSES",
        "SCENARIO-CAPSTONE-6560-ADOPTION",
        "SCENARIO-CAPSTONE-6560-PUBLICATION-HANDOFF",
        "SCENARIO-CAPSTONE-6560-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_6560_inventory_preserves_closed_classes(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6560-INVENTORY/CLOSED-CLASSES: blocks stay blocks."""

    artifact = _artifact(tmp_path)
    inventory = {
        row["experiment_id"]: row for row in artifact["expected_and_observed_task_inventory"]
    }
    classes = {row["experiment_id"]: row for row in artifact["closed_verdict_class_rows"]}
    eligibility = {row["experiment_id"]: row for row in artifact["artifact_eligibility_rows"]}

    assert mod.validate_artifact(artifact) == []
    assert len(inventory) == 12
    assert all(row["artifact_exists"] for row in inventory.values())
    assert classes["exp6549"]["closed_verdict_class"] == "positive"
    assert classes["exp6550"]["closed_verdict_class"] == "positive"
    assert classes["exp6551"]["closed_verdict_class"] == "null"
    assert classes["exp6553"]["closed_verdict_class"] == "blocked"
    assert classes["exp6554"]["closed_verdict_class"] == "blocked"
    assert classes["exp6557"]["closed_verdict_class"] == "blocked"
    assert classes["exp6559"]["closed_verdict_class"] == "blocked"
    assert classes["exp6555"]["closed_verdict_class"] == "null"
    assert eligibility["exp6555"]["live_verifier_max_severity"] == 1
    assert eligibility["exp6555"]["quarantined"] is False
    assert eligibility["exp6557"]["blocked_gate_artifact"] is True
    assert eligibility["exp6557"]["evidence_eligible"] is False


def test_scenario_capstone_6560_recomputes_rows_and_adoption(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6560-RECOMPUTE/ADOPTION: adoption derives from rows."""

    artifact = _artifact(tmp_path)
    production = {
        row["check_id"]: row for row in artifact["independent_production_integration_rows"]
    }
    csl = {row["check_id"]: row for row in artifact["independent_csl_rows"]}
    saturation = {
        row["check_id"]: row for row in artifact["independent_constraint_saturation_rows"]
    }
    adoption = {row["mechanism"]: row for row in artifact["claim_and_adoption_matrix"]}

    assert production["disabled_adapter_identity"]["observed_value"] is True
    assert production["python_rust_parity"]["observed_value"] is True
    assert production["fallback_reachability"]["observed_value"] is True
    assert production["exact_output_equality"]["observed_value"] is True
    assert production["rollback"]["observed_value"] is True
    assert csl["current_value_positive"]["observed_value"] is False
    assert csl["retained_family_noninferior"]["observed_value"] is False
    assert csl["missing_live_evidence_block"]["observed_value"] is True
    assert saturation["phase_curve_established"]["observed_value"] is True
    assert saturation["benefit_beyond_longer_flat"]["observed_value"] is True
    assert saturation["independent_audit_blocked"]["observed_value"] is True
    assert adoption["production_adapter"]["state"] == "default-off"
    assert adoption["rust_pyo3_state"]["state"] == "default-off"
    assert adoption["reversible_memory_controller"]["state"] == "experiment-only"
    assert adoption["csl_policy"]["state"] == "blocked"
    assert adoption["constraint_saturation_policy"]["state"] == "experiment-only"
    assert adoption["arc_supervisor_selection"]["state"] == "experiment-only"
    assert adoption["gatemate_hardware"]["state"] == "blocked"
    assert all(row["state"] in mod.ADOPTION_STATES for row in adoption.values())


def test_scenario_capstone_6560_publication_handoff_and_atomic(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6560-PUBLICATION-HANDOFF/ATOMIC: output is bounded."""

    artifact = _artifact(tmp_path)

    assert artifact["status"] == "complete_v567_independent_capstone_null"
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["v567_capstone_ready_score"] == 1.0
    assert artifact["publication_gate_g1_g4"]["unmet_gates"] == ["G2"]
    assert (
        artifact["publication_gate_g1_g4"][
            "v567_integration_closes_independent_reproducer_requirement"
        ]
        is False
    )
    assert artifact["unmet_gates"] == ["G2"]
    assert len(artifact["v568_handoff"]["largest_remaining_prd_gaps"]) == 3
    assert artifact["document_reconciliation_receipts"]["operator_stop_rule_deferred_files"] == [
        "_bmad/traceability.md",
        "ops/changelog.md",
        "ops/status.md",
    ]
    assert artifact["protected_files_unchanged"]["research_roadmap_yaml_unchanged"] is True
    assert artifact["protected_files_unchanged"]["research_conductor_py_unchanged"] is True
    assert artifact["aggregate_row_recomputation"]["clean_positive_count"] == 3
    assert artifact["aggregate_row_recomputation"]["blocked_count"] == 4
    assert artifact["gate_check_summary"]["required_adjudication_inputs_present"] is True
    assert "exclusion_manifest_lint" in artifact["gate_check_summary"]["failed_or_nonzero_checks"]
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    written = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "written.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        tool_receipts_override=_tool_receipts(),
    )
    assert (tmp_path / "written.json").is_file()
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)


def test_scenario_capstone_6560_validation_edges(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6560-ATOMIC: schema validation fails closed."""

    artifact = _artifact(tmp_path)
    assert mod.sha256_file(None) == "missing"
    assert mod.sha256_file(tmp_path / "missing.txt") == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._row_lint_status({"row_consistency_upstream": []}) == {
        "status": "not_run",
        "exit_code": None,
    }
    assert mod._artifact_path_for_task([], "exp9999") == Path("")
    assert mod._list_rows({"rows": "not-list"}, "rows") == []
    assert mod._publication_gate({"publication_gate": {"json": []}})["gates"] == {}
    default_tests = mod._tests_run_receipts(None, _tool_receipts())
    assert any(row["source"] == "required_run_command" for row in default_tests)

    closed_cases = [
        ({}, {}, "blocked"),
        ({"status": "complete_positive"}, {"max_severity": 2}, "disqualified"),
        ({"status": "complete_blocked_gate"}, {}, "blocked"),
        ({"status": "partial_result"}, {}, "partial"),
        ({"status": "disqualified_result"}, {}, "disqualified"),
        ({"status": "complete_positive"}, {}, "positive"),
        ({"status": "unknown"}, {}, "partial"),
        ({"status": "complete_positive", "verifier_is_oracle": True}, {}, "circular_positive"),
        ({"status": "complete_positive", "acceptance_gate_passed": False}, {}, "blocked"),
        (
            {
                "status": "complete_positive",
                "independent_exact_equality_receipt": {
                    "all_exact_outputs_equal": False,
                    "changed_output_count": 0,
                },
            },
            {},
            "disqualified",
        ),
        (
            {
                "status": "complete_positive",
                "independent_exact_equality_receipt": {
                    "all_exact_outputs_equal": True,
                    "changed_output_count": 1,
                },
            },
            {},
            "disqualified",
        ),
        (
            {
                "status": "complete_positive",
                "aggregate_row_recomputation": {"exact_outputs_equal": False},
            },
            {},
            "disqualified",
        ),
    ]
    for payload, report, expected in closed_cases:
        assert mod._closed_verdict_class(payload, report) == expected

    reason_rows = [
        ({"closed_verdict_class": "positive", "quarantined": True}, "live_critical"),
        (
            {"closed_verdict_class": "circular_positive", "blocked_gate_artifact": False},
            "positive_shape",
        ),
        ({"closed_verdict_class": "partial", "blocked_gate_artifact": False}, "usable"),
        (
            {"closed_verdict_class": "disqualified", "blocked_gate_artifact": False},
            "disqualified",
        ),
    ]
    for row, expected_fragment in reason_rows:
        assert expected_fragment in mod._class_reason(row)

    bad = deepcopy(artifact)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert any("missing required fields" in error for error in mod.validate_artifact(bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["verdict_class"] = "positive"
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "capstone verdict_class must be null, partial, blocked, or disqualified" in (
        mod.validate_artifact(bad)
    )

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    bad["artifact_eligibility_rows"] = []
    bad["closed_verdict_class_rows"] = []
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "artifact_eligibility_rows must contain 12 rows" in errors
    assert "closed_verdict_class_rows must contain 12 rows" in errors
    assert "field_principles must cover required fields" in errors
    assert "field_provenance must cover exactly required fields" in errors

    bad = deepcopy(artifact)
    bad["claim_and_adoption_matrix"][0]["state"] = "launched"
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "claim_and_adoption_matrix contains invalid adoption state" in (
        mod.validate_artifact(bad)
    )

    bad = deepcopy(artifact)
    bad["expected_and_observed_task_inventory"] = artifact["expected_and_observed_task_inventory"][
        :-1
    ]
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "expected_and_observed_task_inventory must contain 12 rows" in (
        mod.validate_artifact(bad)
    )

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["research_roadmap_yaml_unchanged"] = False
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "protected files changed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["research_conductor_py_unchanged"] = False
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "scripts/research_conductor.py changed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"] = []
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "protected_files_unchanged must be an object" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["closed_verdict_class_rows"][0]["closed_verdict_class"] = "mystery"
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    assert "closed_verdict_class_rows contains invalid class" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = None
    assert "reproducibility_checksum missing or malformed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:" + "1" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert "unloadable artifact" in mod.validate_artifact(malformed)[0]

    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    assert "artifact top level must be an object" in mod.validate_artifact(not_object)

    json_path = tmp_path / "artifact.json"
    json_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert mod.validate_artifact(json_path) == []

    with pytest.raises(ValueError, match="forced validation error"):
        original = mod.validate_artifact
        try:
            mod.validate_artifact = lambda _value: ["forced validation error"]  # type: ignore[method-assign]
            mod.build_artifact(
                repo_root=REPO,
                result_path=tmp_path / "bad.json",
                write=False,
                duration_s=1.0,
                tests_run=TESTS_RUN,
                tool_receipts_override=_tool_receipts(),
            )
        finally:
            mod.validate_artifact = original  # type: ignore[method-assign]
