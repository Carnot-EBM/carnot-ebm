"""Tests for Exp 3079 cross-corpus matrix v21.

Spec refs: REQ-REPORT-3079, SCENARIO-REPORT-3079.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v21_3079 as mod


REQUIRED_FIELDS = {
    "matrix_v21_ready",
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
    "rows",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
COUNT_FIELDS = {
    "clean": "clean_rows",
    "flagged": "flagged_rows",
    "bounded": "bounded_rows",
    "blocked": "blocked_rows",
    "gated_skipped": "gated_skipped_rows",
    "projection_only": "projection_only_rows",
    "missing": "missing_rows",
    "retired": "retired_rows",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v20_row(row_id: str, status: str, source_artifact: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": source_artifact,
        "source_field": "status",
        "evidence_class": "v20_fixture",
        "blocker_class": mod.blocker_class(status),
        "claim_scope": "fixture_claim",
        "summary": {"status": status},
    }


def _matrix_v20() -> dict[str, Any]:
    rows_by_status = {
        "clean_rows": [_v20_row("v20-clean", "clean", "results/v20-clean.json")],
        "flagged_rows": [_v20_row("v20-flagged", "flagged", "results/v20-flagged.json")],
        "bounded_rows": [_v20_row("v20-bounded", "bounded", "results/v20-bounded.json")],
        "blocked_rows": [_v20_row("v20-blocked", "blocked", "results/v20-blocked.json")],
        "gated_skipped_rows": [
            _v20_row("v20-gated", "gated_skipped", "results/v20-gated.json")
        ],
        "projection_only_rows": [
            _v20_row("v20-projection", "projection_only", "results/v20-projection.json")
        ],
        "missing_rows": [
            _v20_row(
                "source:exp3059_requested_v1_alias",
                "missing",
                mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
            ),
            _v20_row("v20-true-missing", "missing", "results/missing-truth.json"),
        ],
        "retired_rows": [_v20_row("v20-retired", "retired", "results/v20-retired.json")],
    }
    rows = [row for values in rows_by_status.values() for row in values]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
            "blocker_class": row["blocker_class"],
        }
        for row in rows
        if row["status"] not in {"clean", "retired"}
    ]
    return {
        "artifact": "experiment_3065_cross_corpus_matrix_v20",
        "matrix_v20_ready": True,
        "paper_ready": False,
        "rows_total": len(rows),
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "source_artifacts": [],
        "honest_verdict": "complete: matrix_v20_ready=true",
        **rows_by_status,
    }


def _normalization_ledger() -> dict[str, Any]:
    return {
        "artifact": "experiment_3068_matrix_v20_artifact_alias_blocker_normalization_v1",
        "matrix_v20_normalization_ready": True,
        "artifact_aliases": [
            {
                "alias_id": "exp3059_requested_v1_to_actual_gate_blocked",
                "experiment_id": "exp3059",
                "requested_path": mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
                "actual_path": mod.EXP3059_ACTUAL_REL_PATH.as_posix(),
                "requested_present": False,
                "actual_present": True,
                "claim_effect": "artifact_hygiene_only_research_status_stays_gated_skipped",
            }
        ],
        "blocker_categories": {
            "artifact_hygiene_blockers": [
                {
                    "row_id": "source:exp3059_requested_v1_alias",
                    "status": "missing",
                    "source_artifact": mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
                    "source_field": "source_artifacts.present",
                    "claim_scope": "source_artifact_accounting",
                }
            ]
        },
        "publication_blocker_count_before": 7,
        "normalized_blocker_count_estimate": 6,
        "honest_verdict": "complete: matrix_v20_normalization_ready=true",
    }


def _write_sources(root: Path, *, omit_normalization: bool = False) -> None:
    _write_json(root, mod.MATRIX_V20_REL_PATH, _matrix_v20())
    _write_json(
        root,
        mod.CAPSTONE_V286_REL_PATH,
        {
            "artifact": "experiment_3066_capstone_v286",
            "capstone_ready": True,
            "paper_ready": False,
            "publication_blockers": [{"row_id": str(i)} for i in range(7)],
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
        },
    )
    if not omit_normalization:
        _write_json(root, mod.EXP3068_REL_PATH, _normalization_ledger())
    _write_json(
        root,
        mod.EXP3059_ACTUAL_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3067_REL_PATH,
        {
            "artifact": "experiment_3067_archive_v286_activate_v287",
            "archive_v286_activate_v287_ready": True,
            "prior_paper_ready": False,
            "honest_verdict": "complete: archive_v286_activate_v287_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3069_REL_PATH,
        {
            "artifact": "experiment_3069_solver_verifier_failure_autopsy_protocol_v1",
            "verifier_failure_autopsy_ready": True,
            "honest_verdict": "complete: verifier_failure_autopsy_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3070_REL_PATH,
        {
            "artifact": "experiment_3070_first_token_abstention_sota_panel_v1",
            "first_token_panel_ready": True,
            "abstention_precision": 0.5,
            "rejection_recall": 0.25,
            "verifier_gain_delta_with_abstention": 0.5,
            "false_positive_rate": 0.25,
            "false_negative_rate": 0.25,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            "honest_verdict": "complete: first_token_panel_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3071_REL_PATH,
        {
            "artifact": "experiment_3071_verge_mcs_smt_correction_pilot_v1",
            "mcs_feedback_ready": True,
            "guided_success_count": 5,
            "solver_only_success_count": 5,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "honest_verdict": "complete: mcs_feedback_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3072_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "gate_check_summary": "1 of 3 gate(s) failed",
            "gates_evaluated": [{"passed": False, "artifact_field": "abstention_precision"}],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3073_REL_PATH,
        {
            "artifact": "experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1",
            "ebt_arm_adapter_feasibility_ready": True,
            "ebt_arm_adapter_feasible": True,
            "adapter_implementation_claimed": False,
            "blockers": ["no adapter implementation claimed"],
            "honest_verdict": "complete: future_path_feasible_no_implementation_claim",
        },
    )
    _write_json(
        root,
        mod.EXP3074_REL_PATH,
        {
            "artifact": "experiment_3074_llguidance_aprad_repair_protocol_v1",
            "grammar_constrained_repair_protocol_ready": True,
            "de_tautology_disqualifier_count": 12,
            "honest_verdict": "complete: grammar_constrained_repair_protocol_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3076_REL_PATH,
        {
            "artifact": "experiment_3076_fr11_online_soundness_completeness_budget_v1",
            "soundness_completeness_budget_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "honest_verdict": "complete_fr11_soundness_completeness_budget_ready",
        },
    )
    _write_json(
        root,
        mod.EXP3077_REL_PATH,
        {
            "artifact": "experiment_3077_fr11_soundness_bounded_online_self_learning_pilot_v1",
            "fr11_soundness_bounded_ready": True,
            "soundness_mistakes": 0,
            "completeness_mistakes": 1,
            "mistake_budget_delta": {"all_gates_passed": False},
            "promotion_decision": "controller_only_budget_exceeded",
            "honest_verdict": "complete_fr11_soundness_bounded_budget_exceeded",
        },
    )
    _write_json(
        root,
        mod.EXP3078_REL_PATH,
        {
            "artifact": "experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1",
            "gatemate_ssqa_refresh_ready": True,
            "gatemate_rerun_allowed": False,
            "ssqa_readback_allowed": False,
            "missing_operator_actions": [{"missing_item": "host_reader_command"}],
            "hardware_execution_claim_made": False,
            "speedup_claim_made": False,
            "honest_verdict": "complete: gatemate_ssqa_refresh_ready=true",
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-25 19:21 UTC | Gated grammar-constrained SOTA repair micro-panel | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3072-gated-local-sota-verifier-calibration) |\n",
    )


def _rows_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact["rows"]}


def test_req_report_3079_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3079: OpenSpec declares the v21 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3079" in spec
    assert "SCENARIO-REPORT-3079" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3079_aggregates_aliases_and_dot287_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3079: aliases retire hygiene rows while blockers remain visible."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.5)
    rows = _rows_by_id(artifact)
    counts = Counter(row["status"] for row in artifact["rows"])
    source_paths = {row["path"] for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v21_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete: matrix_v21_ready=true")
    assert artifact["rows_total"] == len(artifact["rows"])
    assert all(isinstance(artifact[field], int) for field in COUNT_FIELDS.values())
    assert sum(artifact[field] for field in COUNT_FIELDS.values()) == artifact["rows_total"]
    assert artifact["publication_blocker_count"] == sum(
        1 for row in artifact["rows"] if row["status"] not in {"clean", "retired"}
    )
    for status, field in COUNT_FIELDS.items():
        assert artifact[field] == counts[status]
    assert {row["source_artifact"] for row in artifact["rows"]} <= source_paths

    alias_row = rows["v20:source:exp3059_requested_v1_alias"]
    assert alias_row["status"] == "retired"
    assert alias_row["source_artifact"] == mod.EXP3059_ACTUAL_REL_PATH.as_posix()
    assert alias_row["summary"]["alias_applied"] is True
    assert rows["v20:v20-true-missing"]["status"] == "missing"

    assert rows["dot287:exp3070_first_token_abstention"]["status"] == "flagged"
    assert rows["dot287:exp3071_verge_mcs_feedback"]["status"] == "flagged"
    assert rows["dot287:exp3072_verifier_calibration_gate"]["status"] == "gated_skipped"
    assert rows["dot287:exp3073_ebt_arm_adapter_feasibility"]["status"] == "projection_only"
    assert rows["dot287:exp3075_repair_micro_panel"]["status"] == "gated_skipped"
    assert rows["dot287:exp3076_fr11_budget"]["status"] == "flagged"
    assert rows["dot287:exp3077_fr11_soundness_bounded_pilot"]["status"] == "flagged"
    assert rows["dot287:exp3078_gatemate_operator_refresh"]["status"] == "blocked"
    assert rows["dot287:exp3078_ssqa_readback_refresh"]["status"] == "gated_skipped"
    assert rows["dot287:exp3075_repair_micro_panel"]["source_artifact"] == (
        mod.CONDUCTOR_LOG_REL_PATH.as_posix()
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts_and_conductor_log",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def test_req_report_3079_missing_required_source_blocks_readiness(tmp_path: Path) -> None:
    """REQ-REPORT-3079: missing required authority artifacts block matrix readiness."""

    _write_sources(tmp_path, omit_normalization=True)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["matrix_v21_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_matrix_v21_preconditions")
    assert "exp3068" in {row["experiment_id"] for row in artifact["required_source_errors"]}
    assert artifact["publication_blocker_count"] == sum(
        1 for row in artifact["rows"] if row["status"] not in {"clean", "retired"}
    )


def test_req_report_3079_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3079: writing and helper edge cases stay deterministic."""

    _write_sources(tmp_path)
    malformed = tmp_path / "malformed.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v21_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_text(tmp_path / "missing.txt") == ""
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.sha256_file(tmp_path / mod.MATRIX_V20_REL_PATH) == _sha256(
        tmp_path / mod.MATRIX_V20_REL_PATH
    )
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("unknown") == "missing"
    assert mod.blocker_class("retired") == "retired_claim"
