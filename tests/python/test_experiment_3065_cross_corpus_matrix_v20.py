"""Tests for Exp 3065 cross-corpus matrix v20.

Spec refs: REQ-REPORT-3065, SCENARIO-REPORT-3065.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v20_3065 as mod


REQUIRED_FIELDS = {
    "matrix_v20_ready",
    "rows_total",
    "clean_rows",
    "flagged_rows",
    "bounded_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "projection_only_rows",
    "missing_rows",
    "retired_rows",
    "publication_blocker_count",
    "publication_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
CLASS_FIELDS = (
    "clean_rows",
    "flagged_rows",
    "bounded_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "projection_only_rows",
    "missing_rows",
    "retired_rows",
)
BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str, evidence_class: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": evidence_class,
        "summary": {"status": status},
    }


def _matrix_v19() -> dict[str, Any]:
    rows = [
        _row("v19-clean", "clean", "prior_clean"),
        _row("v19-flagged", "flagged", "prior_flagged"),
        _row("v19-bounded", "bounded", "prior_bounded"),
        _row("v19-blocked", "blocked", "prior_blocked"),
        _row("v19-gated", "gated_skipped", "prior_gated"),
        _row("v19-projection", "projection_only", "prior_projection"),
        _row("v19-missing", "missing", "prior_missing"),
        _row("v19-retired", "retired", "prior_retired"),
    ]
    return {
        "artifact": "experiment_3052_cross_corpus_matrix_v19",
        "matrix_v19_ready": True,
        "rows_total": len(rows),
        "clean_count": 1,
        "flagged_count": 1,
        "bounded_count": 1,
        "blocked_count": 1,
        "gated_skipped_count": 1,
        "projection_only_count": 1,
        "missing_count": 1,
        "retired_count": 1,
        "repair_claim_status": "bounded",
        "fr11_self_learning_status": "controller_only_solver_feedback_and_locality_ready",
        "gatemate_status": "blocked_output_contract",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "rows": rows,
        "honest_verdict": "complete: matrix_v19_ready=true",
    }


def _capstone_v285() -> dict[str, Any]:
    return {
        "artifact": "experiment_3053_capstone_v285",
        "capstone_ready": True,
        "paper_ready": False,
        "repair_claim_status": "bounded",
        "fr11_self_learning_status": "controller_only_solver_feedback_and_locality_ready",
        "gatemate_status": "blocked_output_contract",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _repair_ledger() -> dict[str, Any]:
    return {
        "artifact": "experiment_3055_repair_headline_retirement_and_blocker_ledger_v1",
        "repair_headline_retirement_ready": True,
        "repair_claim_status": "bounded",
        "flagged_adversarial": True,
        "extracted_repair_blockers": [
            {
                "row_id": "exp3028:adversarial_flags",
                "classification": "true_blocker",
                "blocking": True,
                "source_artifact": "results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json",
                "source_field": "corrigendum_pending",
                "rationale": "TAUTOLOGY remains uncleared.",
            }
        ],
        "still_bounded_repair_claims": [
            {
                "row_id": "repair:headline_status",
                "claim_id": "headline_status",
                "source_artifact": "results/experiment_3042_repair_promotion_reconciliation_v3.json",
                "source_field": "repair_claim_status",
                "repair_claim_status": "bounded",
                "allowed_wording": "Bounded repair evidence only.",
            }
        ],
        "retired_repair_claims": [
            {
                "row_id": "exp3029:headline_sota_repair_clean_methodology",
                "claim_id": "headline_sota_repair_clean_methodology",
                "source_artifact": "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
                "source_field": "retired_or_blocked_claims[headline_sota_repair_clean_methodology]",
                "allowed_wording": "Do not use headline SOTA repair wording.",
            }
        ],
        "rerun_prerequisites": [{"gate": "verifier_gain", "required": True}],
        "honest_verdict": "complete: repair_headline_retirement_ready=true",
    }


def _write_sources(root: Path, *, omit: set[Path] | None = None) -> None:
    omit = omit or set()
    payloads: dict[Path, dict[str, Any]] = {
        mod.MATRIX_V19_REL_PATH: _matrix_v19(),
        mod.CAPSTONE_V285_REL_PATH: _capstone_v285(),
        mod.EXP3054_REL_PATH: {
            "artifact": "experiment_3054_archive_v285_activate_v286",
            "archive_v285_activate_v286_ready": True,
            "prior_capstone_ready": True,
            "prior_paper_ready": False,
            "carry_forward_blockers": [{"blocker_id": "repair_bounded"}],
        },
        mod.EXP3055_REL_PATH: _repair_ledger(),
        mod.EXP3056_REL_PATH: {
            "artifact": "experiment_3056_repair_de_tautology_protocol_v1",
            "repair_de_tautology_protocol_ready": True,
            "promotion_disqualifiers": [{"id": "prior_tautology_not_cleared"}],
        },
        mod.EXP3057_REL_PATH: {
            "artifact": "experiment_3057_local_sota_solution_verifier_gain_panel_v1",
            "solution_verifier_calibration_ready": True,
            "verifier_gain_delta": -0.125,
            "false_positive_rate": 0.0,
            "false_negative_rate": 1.0,
            "flagged_adversarial": True,
        },
        mod.EXP3058_REL_PATH: {
            "artifact": "experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1",
            "llm_guided_smt_pilot_ready": True,
            "guided_success_count": 6,
            "solver_only_success_count": 6,
            "guidance_vs_solver_only": {"guided_minus_solver_only_success_count": 0},
            "flagged_adversarial": True,
        },
        mod.EXP3059_ACTUAL_REL_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [{"passed": False, "artifact_field": "verifier_gain_delta"}],
            "honest_verdict": "blocked_gate_check_failed",
        },
        mod.EXP3060_REL_PATH: {
            "artifact": "experiment_3060_fr11_solver_self_model_trace_schema_v1",
            "solver_self_model_trace_ready": True,
            "allowed_edit_targets": [{"name": "controller_weights"}],
        },
        mod.EXP3061_REL_PATH: {
            "artifact": "experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1",
            "fr11_delayed_regression_ready": True,
            "promotion_decision": "controller_only_delayed_regression_ready",
            "edit_targets_used": ["controller_weights", "trace_memory"],
            "family_holdout_delta": 0.4,
            "delayed_regression_delta": 0.4,
            "prior_retention_delta": 0.0,
            "flagged_adversarial": True,
            "inference_substrate": {
                "model_weight_training": False,
                "model_weight_mutation": False,
            },
        },
        mod.EXP3062_REL_PATH: {
            "artifact": "experiment_3062_kan_pwa_locality_verification_audit_v1",
            "kan_pwa_verification_ready": False,
            "claim_promotion_useful": False,
            "exact_controller_anchor_bound_available": True,
            "locality_bound": 0.75,
            "promotion_decision": "controller_locality_evidence_only",
        },
        mod.EXP3063_REL_PATH: {
            "artifact": "experiment_3063_gatemate_no_rerun_operator_action_ledger_v1",
            "gatemate_no_rerun_ledger_ready": True,
            "gatemate_rerun_allowed": False,
            "downstream_tasks_blocked": [{"task_id": "exp3050", "allowed_to_rerun": False}],
            "missing_operator_actions": [{"missing_item": "host_reader_command"}],
        },
        mod.EXP3064_REL_PATH: {
            "artifact": "experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1",
            "ssqa_boundary_ledger_ready": True,
            "ssqa_readback_allowed": False,
            "ssqa_status": "gated_skipped_host_visible_smoke_missing",
            "host_visible_smoke_evidence": {"present": False},
        },
    }
    for rel_path, payload in payloads.items():
        if rel_path not in omit:
            _write_json(root, rel_path, payload)


def _rows_by_id(artifact: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact[field]}


def test_req_report_3065_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3065: OpenSpec declares the matrix v20 contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3065" in spec
    assert "SCENARIO-REPORT-3065" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3065_builds_v20_with_explicit_row_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3065: v20 names every class, blocker, and source citation."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.25)
    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    flagged = _rows_by_id(artifact, "flagged_rows")
    bounded = _rows_by_id(artifact, "bounded_rows")
    blocked = _rows_by_id(artifact, "blocked_rows")
    skipped = _rows_by_id(artifact, "gated_skipped_rows")
    missing = _rows_by_id(artifact, "missing_rows")
    retired = _rows_by_id(artifact, "retired_rows")

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v20_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: matrix_v20_ready=true")
    assert artifact["rows_total"] == sum(len(artifact[field]) for field in CLASS_FIELDS)
    assert all(isinstance(artifact[field], list) for field in CLASS_FIELDS)
    assert artifact["publication_blocker_count"] == len(artifact["publication_blockers"])
    assert {row["status"] for row in artifact["publication_blockers"]} <= BLOCKING_STATUSES

    all_rows = [row for field in CLASS_FIELDS for row in artifact[field]]
    assert all(
        {
            "row_id",
            "status",
            "source_artifact",
            "source_field",
            "evidence_class",
            "blocker_class",
            "claim_scope",
            "summary",
        }
        <= row.keys()
        for row in all_rows
    )

    assert "solver:local_sota_solution_verifier_gain_panel" in flagged
    assert (
        flagged["solver:local_sota_solution_verifier_gain_panel"]["summary"]["verifier_gain_delta"]
        == -0.125
    )
    assert "solver:aquaforte_smt_pilot" in flagged
    assert "fr11:delayed_regression" in flagged
    assert "repair:headline_status" in bounded
    assert "kan:pwa_locality_audit" in bounded
    assert "repair:de_tautology_disqualifiers" in blocked
    assert "gatemate:no_rerun_ledger" in blocked
    assert "repair:gated_sota_rerun" in skipped
    assert "ssqa:host_visible_readback_boundary" in skipped
    assert "source:exp3059_requested_v1_alias" in missing
    assert "repair:headline_sota_repair_clean_methodology" in retired

    assert source_by_path[mod.EXP3059_REQUESTED_REL_PATH.as_posix()]["present"] is False
    assert source_by_path[mod.EXP3059_REQUESTED_REL_PATH.as_posix()]["missing_recorded"] is True
    assert source_by_path[mod.EXP3059_ACTUAL_REL_PATH.as_posix()]["present"] is True
    assert source_by_path[mod.EXP3064_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3064_REL_PATH
    )

    statuses = artifact["status_summaries"]
    assert statuses["repair"]["status"] == "bounded_and_gated_skipped"
    assert statuses["repair"]["citations"][0]["source_field"] == "repair_claim_status"
    assert statuses["solver_grounded_verification"]["status"] == "flagged_solver_grounded_no_gain"
    assert statuses["fr11"]["status"] == "controller_only_delayed_regression_ready_flagged"
    assert statuses["kan_pwa"]["status"] == "bounded_controller_anchor_audit_not_promoted"
    assert statuses["gatemate"]["status"] == "blocked_no_rerun_operator_actions_required"
    assert statuses["ssqa"]["status"] == "gated_skipped_host_visible_smoke_missing"

    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def test_req_report_3065_missing_required_source_blocks_readiness(tmp_path: Path) -> None:
    """REQ-REPORT-3065: missing optional aliases do not block, but required sources do."""

    _write_sources(tmp_path, omit={mod.CAPSTONE_V285_REL_PATH})

    artifact = mod.build_artifact(tmp_path)

    assert artifact["matrix_v20_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_matrix_v20_preconditions")
    assert artifact["required_source_errors"] == [
        {"experiment_id": "exp3053", "reason": "missing_or_malformed_required_artifact"}
    ]
    assert {row["row_id"] for row in artifact["missing_rows"]} >= {
        "source:exp3053",
        "source:exp3059_requested_v1_alias",
    }


def test_req_report_3065_optional_missing_sources_are_machine_readable(tmp_path: Path) -> None:
    """REQ-REPORT-3065: absent optional .286 artifacts become missing rows."""

    _write_json(tmp_path, mod.MATRIX_V19_REL_PATH, _matrix_v19())
    _write_json(tmp_path, mod.CAPSTONE_V285_REL_PATH, _capstone_v285())

    artifact = mod.build_artifact(tmp_path)
    missing_ids = {row["row_id"] for row in artifact["missing_rows"]}

    assert artifact["matrix_v20_ready"] is True
    assert artifact["required_source_errors"] == []
    assert "source:exp3054" in missing_ids
    assert "source:exp3064" in missing_ids
    assert "source:exp3059_requested_v1_alias" in missing_ids


def test_req_report_3065_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3065: writing and malformed input handling stay deterministic."""

    _write_sources(tmp_path)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v20_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.5)
    assert saved["rows_total"] == sum(len(saved[field]) for field in CLASS_FIELDS)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("unknown") == "missing"
    assert mod.blocker_class("clean") == "none"
    assert mod.blocker_class("flagged") == "adversarial_or_methodology_flag"
    assert mod._capstone_v285_rows({"capstone_ready": False})[0]["status"] == "blocked"
    assert (
        mod._fr11_delayed_rows(
            {
                "fr11_delayed_regression_ready": True,
                "inference_substrate": {"model_weight_training": True},
            }
        )[0]["status"]
        == "blocked"
    )
    assert (
        mod._fr11_delayed_rows(
            {
                "fr11_delayed_regression_ready": True,
                "inference_substrate": {"model_weight_training": False},
            }
        )[0]["status"]
        == "bounded"
    )
    assert (
        mod._fr11_delayed_rows(
            {
                "fr11_delayed_regression_ready": False,
                "inference_substrate": {"model_weight_training": False},
            }
        )[0]["status"]
        == "blocked"
    )
    assert mod._status_from_gate_payload({"status": "blocked"}) == "blocked"
    assert mod._honest_verdict(False, 1, [], []) == (
        "blocked_matrix_v20_preconditions: rows_machine_readable=false; rows_total=1"
    )
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._float_or_none(False) is None
    assert mod._float_or_none("bad") is None
