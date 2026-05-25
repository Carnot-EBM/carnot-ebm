"""Tests for Exp 3068 matrix-v20 alias and blocker normalization.

Spec refs: REQ-REPORT-3068, SCENARIO-REPORT-3068.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import matrix_v20_artifact_alias_blocker_normalization_3068 as mod


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
REQUIRED_FIELDS = {
    "matrix_v20_normalization_ready",
    "artifact_aliases",
    "missing_artifacts_after_aliasing",
    "blocker_categories",
    "exp3059_alias_status",
    "publication_blocker_count_before",
    "normalized_blocker_count_estimate",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
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


def _row(
    row_id: str,
    status: str,
    source_artifact: str,
    *,
    source_field: str = "status",
    claim_scope: str = "claim_scope",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence_class": claim_scope,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"status": status},
    }


def _matrix_v20_payload() -> dict[str, Any]:
    classes = {field: [] for field in CLASS_FIELDS}
    classes["clean_rows"] = [
        _row("archive:v286_activation", "clean", "results/experiment_3054_archive_v285_activate_v286.json")
    ]
    classes["flagged_rows"] = [
        _row(
            "solver:local_sota_solution_verifier_gain_panel",
            "flagged",
            "results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json",
            source_field="verifier_gain_delta",
            claim_scope="local_sota_solution_verifier_gain",
        )
    ]
    classes["bounded_rows"] = [
        _row(
            "v19:repair:headline_status",
            "bounded",
            "results/experiment_3042_repair_promotion_reconciliation_v3.json",
            source_field="repair_claim_status",
            claim_scope="repair_headline_boundary",
        ),
        _row(
            "repair:headline_status",
            "bounded",
            "results/experiment_3042_repair_promotion_reconciliation_v3.json",
            source_field="repair_claim_status",
            claim_scope="repair_headline_boundary",
        ),
    ]
    classes["blocked_rows"] = [
        _row(
            "repair:de_tautology_disqualifiers",
            "blocked",
            "results/experiment_3056_repair_de_tautology_protocol_v1.json",
            source_field="promotion_disqualifiers",
            claim_scope="repair_headline_boundary",
        ),
        _row(
            "gatemate:no_rerun_ledger",
            "blocked",
            "results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json",
            source_field="gatemate_rerun_allowed",
            claim_scope="hardware_rerun_gate",
        ),
    ]
    classes["gated_skipped_rows"] = [
        _row(
            "repair:gated_sota_rerun",
            "gated_skipped",
            mod.EXP3059_ACTUAL_REL_PATH.as_posix(),
            source_field="gate_check_summary",
            claim_scope="repair_live_rerun",
        ),
        _row(
            "ssqa:host_visible_readback_boundary",
            "gated_skipped",
            "results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json",
            source_field="ssqa_status",
            claim_scope="host_visible_readback_gate",
        ),
    ]
    classes["projection_only_rows"] = [
        _row(
            "v19:v18:exp3026",
            "projection_only",
            "results/experiment_3038_cross_corpus_matrix_v18.json",
            source_field="matrix_rows[exp3026]",
            claim_scope="prior_v18_carry_forward",
        )
    ]
    classes["missing_rows"] = [
        _row(
            "source:exp3059_requested_v1_alias",
            "missing",
            mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
            source_field="source_artifacts.present",
            claim_scope="source_artifact_accounting",
        ),
        _row(
            "v19:gatemate:host_visible_smoke",
            "missing",
            "results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json",
            source_field="gatemate_host_visible_smoke_passed",
            claim_scope="host_visible_hardware_transcript",
        ),
        _row(
            "v19:v18:exp3039",
            "missing",
            "results/experiment_3038_cross_corpus_matrix_v18.json",
            source_field="matrix_rows[exp3039]",
            claim_scope="prior_v18_carry_forward",
        ),
    ]
    classes["retired_rows"] = [
        _row(
            "repair:headline_sota_repair_clean_methodology",
            "retired",
            "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
            source_field="retired_repair_claims",
            claim_scope="retired_repair_headline_wording",
        )
    ]
    all_rows = [row for field in CLASS_FIELDS for row in classes[field]]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "blocker_class": row["blocker_class"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
        }
        for row in all_rows
        if row["status"] not in {"clean", "retired"}
    ]
    return {
        "artifact": "experiment_3065_cross_corpus_matrix_v20",
        "matrix_v20_ready": True,
        "paper_ready": False,
        "rows_total": len(all_rows),
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "source_artifacts": [
            None,
            {
                "experiment_id": "exp3059_requested_v1_alias",
                "path": mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
                "role": "requested_exp3059_alias",
                "required": False,
                "present": False,
            },
            {
                "experiment_id": "exp3059",
                "path": mod.EXP3059_ACTUAL_REL_PATH.as_posix(),
                "role": "gated_sota_repair_rerun_gate_result",
                "required": False,
                "present": True,
            },
        ],
        "honest_verdict": "complete: matrix_v20_ready=true",
        **classes,
    }


def _capstone_payload(blocker_count: int) -> dict[str, Any]:
    return {
        "artifact": "experiment_3066_capstone_v286",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blockers": [{"row_id": str(index)} for index in range(blocker_count)],
        "repair_claim_status": "bounded_and_gated_skipped",
        "solver_grounding_status": "flagged_solver_grounded_no_gain",
        "fr11_self_learning_status": "controller_only_delayed_regression_ready_flagged",
        "kan_pwa_status": "bounded_controller_anchor_audit_not_promoted",
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_sources(root: Path, *, omit_actual_exp3059: bool = False) -> dict[str, Any]:
    matrix = _matrix_v20_payload()
    _write_json(root, mod.MATRIX_V20_REL_PATH, matrix)
    _write_json(root, mod.CAPSTONE_V286_REL_PATH, _capstone_payload(len(matrix["publication_blockers"])))
    if not omit_actual_exp3059:
        _write_json(
            root,
            mod.EXP3059_ACTUAL_REL_PATH,
            {
                "experiment": 3059,
                "schema": "blocked_gate_check_v1",
                "status": "blocked",
                "gate_check_summary": (
                    "1 of 2 gate(s) failed; first failure: "
                    "exp3057-local-sota-solution-verifier-gain-panel.verifier_gain_delta"
                ),
                "blocked_at_layer": "conductor_pre_gate",
                "honest_verdict": "blocked_gate_check_failed",
            },
        )
    _write_json(root, Path("results/experiment_3054_archive_v285_activate_v286.json"), {"artifact": "exp3054"})
    _write_json(root, Path("results/experiment_3038_cross_corpus_matrix_v18.json"), {"artifact": "exp3038"})
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-25 11:00 UTC | Unrelated task | OK | ignored |",
                "| 2026-05-25 11:52 UTC | Archive .285 and activate .286 | OK | 81 passed |",
                "| 2026-05-25 13:04 UTC | Gated SOTA repair de-tautology rerun | GATE_BLOCK | 1 of 2 gate(s) failed |",
                "| 2026-05-25 15:55 UTC | Cross-corpus matrix v20 | OK | 81 passed |",
                "| 2026-05-25 16:09 UTC | Capstone .286 | OK | 81 passed |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.OPS_STATUS_REL_PATH, "paper_ready=false; exp3059 alias pending\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "Cross-corpus matrix v20 blocker_count=33\n")
    return matrix


def test_req_report_3068_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3068: OpenSpec declares the normalization-ledger contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3068" in spec
    assert "SCENARIO-REPORT-3068" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3068_aliases_exp3059_without_cleaning_research(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3068: the filename alias reduces only artifact hygiene."""

    matrix = _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=4.5)
    categories = artifact["blocker_categories"]
    aliases = artifact["artifact_aliases"]
    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v20_normalization_ready"] is True
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["honest_verdict"].startswith("complete: matrix_v20_normalization_ready=true")
    assert artifact["publication_blocker_count_before"] == len(matrix["publication_blockers"])
    assert artifact["normalized_blocker_count_estimate"] == len(matrix["publication_blockers"]) - 1

    assert aliases == [
        {
            "alias_id": "exp3059_requested_v1_to_actual_gate_blocked",
            "experiment_id": "exp3059",
            "requested_path": mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
            "actual_path": mod.EXP3059_ACTUAL_REL_PATH.as_posix(),
            "requested_present": False,
            "actual_present": True,
            "actual_status": "blocked",
            "actual_honest_verdict": "blocked_gate_check_failed",
            "non_destructive": True,
            "claim_effect": "artifact_hygiene_only_research_status_stays_gated_skipped",
        }
    ]
    assert artifact["exp3059_alias_status"] == (
        "actual_gate_blocked_artifact_present_alias_v21_to_actual_without_rewrite"
    )
    assert {row["path"] for row in artifact["missing_artifacts_after_aliasing"]} == {
        "results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json"
    }

    assert {row["row_id"] for row in categories["artifact_hygiene_blockers"]} == {
        "source:exp3059_requested_v1_alias"
    }
    assert {row["row_id"] for row in categories["research_blockers"]} >= {
        "solver:local_sota_solution_verifier_gain_panel",
        "repair:gated_sota_rerun",
        "v19:gatemate:host_visible_smoke",
    }
    assert {row["row_id"] for row in categories["honest_bounded_rows"]} == {
        "v19:repair:headline_status",
        "repair:headline_status",
    }
    assert {row["row_id"] for row in categories["retired_rows"]} == {
        "repair:headline_sota_repair_clean_methodology"
    }
    assert categories["duplicate_rows"] == [
        {
            "duplicate_key": (
                "results/experiment_3042_repair_promotion_reconciliation_v3.json"
                "|repair_claim_status|repair_headline_boundary"
            ),
            "row_ids": ["v19:repair:headline_status", "repair:headline_status"],
            "statuses": ["bounded", "bounded"],
        }
    ]
    assert {row["row_id"] for row in categories["blocked_rows"]} == {
        "repair:de_tautology_disqualifiers",
        "gatemate:no_rerun_ledger",
    }
    assert {row["row_id"] for row in categories["projection_only_rows"]} == {"v19:v18:exp3026"}
    assert {row["row_id"] for row in categories["true_missing_evidence"]} == {
        "v19:gatemate:host_visible_smoke",
        "v19:v18:exp3039",
    }

    assert source_by_path[mod.MATRIX_V20_REL_PATH.as_posix()]["present"] is True
    assert source_by_path[mod.CAPSTONE_V286_REL_PATH.as_posix()]["present"] is True
    assert source_by_path[mod.EXP3059_ACTUAL_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3059_ACTUAL_REL_PATH
    )
    assert artifact["actual_result_filenames"] == [
        "experiment_3038_cross_corpus_matrix_v18.json",
        "experiment_3054_archive_v285_activate_v286.json",
        "experiment_3059_gated_sota_repair_de_tautology_rerun.json",
        "experiment_3065_cross_corpus_matrix_v20.json",
        "experiment_3066_capstone_v286.json",
    ]
    assert len(artifact["conductor_log_entries_3054_3066"]) == 4
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts_and_filenames",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }
    assert artifact["no_historical_artifact_rewrite"] is True
    assert artifact["no_research_claim_cleaned_by_alias"] is True


def test_req_report_3068_missing_actual_exp3059_blocks_alias_readiness(tmp_path: Path) -> None:
    """REQ-REPORT-3068: a missing actual Exp 3059 artifact cannot be papered over."""

    matrix = _write_sources(tmp_path, omit_actual_exp3059=True)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["matrix_v20_normalization_ready"] is False
    assert artifact["artifact_aliases"] == []
    assert artifact["exp3059_alias_status"] == "blocked_actual_gate_artifact_missing"
    assert artifact["publication_blocker_count_before"] == len(matrix["publication_blockers"])
    assert artifact["normalized_blocker_count_estimate"] == len(matrix["publication_blockers"])
    assert artifact["honest_verdict"].startswith("blocked_matrix_v20_normalization_preconditions")
    assert {row["path"] for row in artifact["missing_artifacts_after_aliasing"]} >= {
        mod.EXP3059_REQUESTED_REL_PATH.as_posix(),
        mod.EXP3059_ACTUAL_REL_PATH.as_posix(),
    }


def test_req_report_3068_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3068: writing and malformed input handling remain deterministic."""

    _write_sources(tmp_path)
    malformed = tmp_path / "malformed.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v20_normalization_ready"] is True
    assert saved["duration_s"] == pytest.approx(3.25)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_text(tmp_path / "missing.txt") == ""
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("unknown") == "missing"
    assert mod.blocker_class("retired") == "retired_claim"
    assert mod._exp3059_alias_status(tmp_path, []) == "requested_alias_file_present_no_alias_needed"
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
