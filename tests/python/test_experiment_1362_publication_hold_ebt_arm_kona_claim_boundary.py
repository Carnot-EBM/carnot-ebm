"""Tests for Exp 1362 publication-hold EBT/ARM/Kona claim boundary.

Spec refs: REQ-KONA-009, SCENARIO-KONA-009.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import publication_hold_ebt_arm_kona_claim_boundary as exp1362


def _sources() -> dict[str, dict[str, Any]]:
    return {
        "exp1324": {
            "status": "complete",
            "source_metrics": {
                "exp1312_certificate_parse_rate": 0.71223,
                "exp1312_certificate_truthfulness_rate": 0.69697,
            },
            "honest_verdict": "diagnostic_complete_parse_gate_shortfall_parser_recovery_needed",
        },
        "exp1344": {
            "status": "complete",
            "dvi_ready": True,
            "headline_result_allowed": False,
            "self_learning_delta_overall": 1.596429,
            "honest_verdict": "failure_type_memory_policy_dvi_ready_replay_non_headline",
        },
        "exp1349": {
            "status": "complete",
            "external_dependency_claim_allowed": False,
            "parity_gaps": [{"gap": "native_kona_style_reasoning"}],
            "publication_claim_changes_needed": ["keep publication hold active"],
            "honest_verdict": (
                "external_parity_gap_audit_complete_local_evidence_only_no_kona_or_external_dependency_claim"
            ),
        },
        "exp1353": {
            "status": "complete",
            "certificate_case_count": 4,
            "certificate_parse_rate": 0.0,
            "certificate_truthfulness_rate": 0.0,
            "trigger_token_hit_rate": 0.0,
            "unknown_preservation_rate": 0.0,
            "headline_result_allowed": True,
            "honest_verdict": "sota_triggered_certificate_v7_measured",
        },
        "exp1354": {
            "status": "complete",
            "dominant_skill_gap": "symbolization",
            "symbolization_pass_rate": 0.0,
            "skill_split_claim_allowed": True,
        },
        "exp1355": {
            "status": "blocked",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: exp1353.certificate_parse_rate "
                "(actual=0.0 >= expected=0.75)"
            ),
        },
        "exp1313": {
            "status": "blocked",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: exp1312.certificate_parse_rate "
                "(actual=0.71223 >= expected=0.75)"
            ),
        },
        "exp1316": {
            "status": "blocked",
            "gate_check_summary": (
                "1 of 2 gate(s) failed; first failure: exp1312.certificate_parse_rate "
                "(actual=0.71223 >= expected=0.75)"
            ),
        },
        "exp1358": {
            "status": "complete",
            "fresh_verified_sample_count": 0,
            "update_is_replay_only": True,
            "dvi_ready": True,
            "nonforgetting_certificate_rate": 1.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": -0.846154,
            "honest_verdict": "verifier_selected_memory_replay_only_dvi_ready_non_headline",
        },
        "exp1361": {
            "status": "complete",
            "hardware_claim_allowed": False,
            "kv260_claim_allowed": False,
            "energy_equivalence_error": 0.0,
            "state_expansion_ratio": 4.0,
            "energy_equivalence_proxy": {"hardware_energy_measurement": False},
            "next_hardware_requirements": [{"claim_gate": "required_before_hardware_claim"}],
            "metadata": {
                "hardware_executed": False,
                "external_hardware_executed": False,
                "synthesis_performed": False,
                "board_executed": False,
            },
            "honest_verdict": "cpu_only_pdit_certificate_state_mapping_ready_hardware_not_run",
        },
    }


def _write_sources(project: Path) -> None:
    for key, rel_path in exp1362.SOURCE_PATHS.items():
        path = project / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_sources()[key]), encoding="utf-8")


def test_req_kona_009_builds_complete_publication_hold_boundary() -> None:
    """REQ-KONA-009: Exp 1362 records every required field without overclaiming."""

    artifact = exp1362.build_artifact(
        _sources(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    exp1362.validate_artifact(artifact)
    assert exp1362.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["publication_hold_state"] == "active"
    assert artifact["external_dependency_claim_allowed"] is False
    assert artifact["honest_verdict"] == exp1362.HONEST_VERDICT


def test_req_kona_009_certificate_and_semantic_gates_stay_blocking() -> None:
    """REQ-KONA-009: failed certificate and semantic gates remain publication blockers."""

    artifact = exp1362.build_artifact(_sources())
    certificate = artifact["certificate_evidence_summary"]

    assert certificate["certificate_claim_allowed"] is False
    assert certificate["certificate_parse_rate"] == 0.0
    assert certificate["certificate_truthfulness_rate"] == 0.0
    assert certificate["dominant_skill_gap"] == "symbolization"
    assert certificate["semantic_validator_state"]["exp1355_status"] == "blocked"
    assert "actual=0.0" in certificate["semantic_validator_state"]["exp1355_gate_check_summary"]


def test_scenario_kona_009_self_learning_and_dvi_are_replay_only() -> None:
    """SCENARIO-KONA-009: replay-only self-learning cannot lift the hold."""

    artifact = exp1362.build_artifact(_sources())
    self_learning = artifact["self_learning_evidence_summary"]

    assert self_learning["headline_self_learning_claim_allowed"] is False
    assert self_learning["fresh_verified_sample_count"] == 0
    assert self_learning["update_is_replay_only"] is True
    assert self_learning["dvi_status"]["exp1316_status"] == "blocked"
    assert artifact["publication_hold_rationale"]["fresh_self_learning_available"] is False


def test_req_kona_009_external_hardware_and_ebt_arm_claims_are_disallowed() -> None:
    """REQ-KONA-009: external, hardware, and EBT/ARM equivalence claims remain false."""

    artifact = exp1362.build_artifact(_sources())

    assert artifact["hardware_evidence_summary"]["hardware_execution_claim_allowed"] is False
    assert artifact["hardware_evidence_summary"]["hardware_executed"] is False
    assert artifact["ebt_arm_claim_boundary"]["architectural_inspiration_allowed"] is True
    assert artifact["ebt_arm_claim_boundary"]["local_ebt_arm_equivalence_proven"] is False
    assert "Carnot has achieved EBT" in json.dumps(
        artifact["ebt_arm_claim_boundary"]["disallowed_language"]
    )


def test_req_kona_009_validation_rejects_hold_lift_or_external_claims() -> None:
    """REQ-KONA-009: validation rejects artifacts that exceed local evidence."""

    artifact = exp1362.build_artifact(_sources())

    external = dict(artifact)
    external["external_dependency_claim_allowed"] = True
    with pytest.raises(ValueError, match="external_dependency_claim_allowed"):
        exp1362.validate_artifact(external)

    hold_lifted = dict(artifact)
    hold_lifted["publication_hold_state"] = "lifted"
    with pytest.raises(ValueError, match="publication hold"):
        exp1362.validate_artifact(hold_lifted)


def test_scenario_kona_009_run_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-KONA-009: the runner writes bootstrap then terminal local artifact."""

    _write_sources(tmp_path)
    output = tmp_path / exp1362.DEFAULT_OUT_PATH
    writes: list[str] = []

    artifact = exp1362.run(
        project_root=tmp_path,
        out_path=output,
        run_date="20260505",
        write_observer=lambda _path, payload: writes.append(str(payload["status"])),
    )

    assert writes == ["in_progress", "complete"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["input_resolution"]["exp1349"]["requested_available"] is False
    assert artifact["input_resolution"]["exp1349"]["used_available"] is True
