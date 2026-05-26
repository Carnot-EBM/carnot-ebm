"""Tests for the Exp 3094 milestone .288 capstone.

Spec refs: REQ-REPORT-3094, SCENARIO-REPORT-3094.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v288_3094 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "verifier_gain_status",
    "repair_claim_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "gatemate_status",
    "ssqa_status",
    "next_milestone_recommendation",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str, claim_scope: str, evidence_class: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"source_status": status},
    }


def _matrix_v22(
    *,
    blockers: bool = True,
    model_gaps: bool = True,
    missing_repair_input: bool = True,
) -> dict[str, Any]:
    if blockers:
        rows = [
            _row("verifier-flag", "flagged", "local_sota_solution_verifier_gain", "panel"),
            _row("verifier-gate", "gated_skipped", "verifier_gain_recovery_gate", "gate"),
            _row("repair-bounded", "bounded", "repair_headline_boundary", "repair"),
            _row("repair-formal", "flagged", "solver_grounded_repair_feedback", "feedback"),
            _row("repair-missing", "missing", "repair_live_rerun", "repair"),
            _row("fr11-budget", "clean", "controller_only_online_learning_budget", "fr11"),
            _row("ebt-sidecar", "projection_only", "future_adapter_context", "ebt_arm"),
            _row("gatemate-blocked", "blocked", "hardware_rerun_gate", "gatemate"),
            _row("ssqa-gated", "gated_skipped", "host_visible_readback_gate", "ssqa"),
            _row("capstone-paper", "bounded", "paper_readiness", "capstone"),
            _row("archive-clean", "clean", "milestone_activation", "archive"),
            _row("old-retired", "retired", "retired_claim", "archive"),
        ]
    else:
        rows = [
            _row("verifier-clean", "clean", "local_sota_solution_verifier_gain", "panel"),
            _row("repair-retired", "retired", "repair_headline_boundary", "repair"),
            _row("fr11-budget", "clean", "controller_only_online_learning_budget", "fr11"),
            _row("ebt-integrated", "clean", "future_adapter_context", "ebt_arm"),
            _row("gatemate-clean", "clean", "hardware_rerun_gate", "gatemate"),
            _row("ssqa-clean", "clean", "host_visible_readback_gate", "ssqa"),
            _row("capstone-paper", "clean", "paper_readiness", "capstone"),
        ]
    blockers_rows = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "blocker_class": row["blocker_class"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
        }
        for row in rows
        if row["status"] in mod.PUBLICATION_BLOCKING_STATUSES
    ]
    input_paths = [
        mod.MATRIX_V22_REL_PATH.as_posix(),
        "results/source_capstone_v287.json",
        "results/source_blocker_ledger.json",
        "results/source_fr11.json",
        "results/source_ebt_arm.json",
        "results/source_gatemate_ssqa.json",
    ]
    if missing_repair_input:
        input_paths.append("results/source_missing_repair_micro_panel.json")
    return {
        "artifact": "experiment_3093_cross_corpus_matrix_v22",
        "matrix_v22_ready": True,
        "rows_total": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers_rows),
        "publication_blockers": blockers_rows,
        "rows": rows,
        "blocker_delta_from_v21": -2 if blockers else -9,
        "blocker_reconciliation_from_ledger": {
            "publication_blocker_count_before": len(blockers_rows) + (2 if blockers else 9),
            "publication_blocker_count_after": len(blockers_rows),
            "blocker_delta_from_v21": -2 if blockers else -9,
            "decreases": [
                {
                    "count": 2,
                    "ledger_category": "fr11_budget",
                    "row_ids": ["fr11-a", "fr11-b"],
                    "reason": "controller-only FR-11 gates passed",
                }
            ],
            "increases": [],
            "neutral_replacements": [],
        },
        "capstone_input_artifacts": input_paths,
        "missing_artifacts": [
            {
                "path": "results/source_missing_repair_micro_panel.json",
                "reason": "expected .288 repair micro-panel artifact is absent",
            }
        ]
        if missing_repair_input
        else [],
        "headline_model_spec_gaps": [
            {
                "row_id": "verifier-flag",
                "source_artifact": "results/source_live_panel.json",
                "missing_model_ids": ["model-b"],
                "present_model_ids": ["model-a"],
                "reason": "mandated model_specs missing for live LLM artifact",
            }
        ]
        if model_gaps
        else [],
        "honest_verdict": "complete: matrix_v22_ready=true",
        "inference_substrate": {
            "kind": "aggregation_from_checked_in_artifacts",
            "executes_models": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
        },
    }


def _write_sources(root: Path, matrix: dict[str, Any]) -> None:
    _write_json(root, mod.MATRIX_V22_REL_PATH, matrix)
    for rel_path in matrix["capstone_input_artifacts"]:
        if rel_path == mod.MATRIX_V22_REL_PATH.as_posix():
            continue
        if rel_path in {item["path"] for item in matrix.get("missing_artifacts", [])}:
            continue
        _write_json(
            root,
            rel_path,
            {
                "artifact": Path(rel_path).stem,
                "ready": True,
                "honest_verdict": f"complete: {Path(rel_path).stem}",
            },
        )


def test_req_report_3094_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3094: OpenSpec declares the .288 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3094" in spec
    assert "SCENARIO-REPORT-3094" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3094_builds_capstone_without_paper_overclaim(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3094: matrix v22 blockers and claim boundaries stay visible."""

    matrix = _matrix_v22()
    _write_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=5.25)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 9
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["verifier_gain_status"] == "flagged_or_gated_verifier_gain_recovery_incomplete"
    assert artifact["repair_claim_status"] == "bounded_flagged_gated_missing_verifier_gated"
    assert artifact["fr11_self_learning_status"] == "clean_controller_only_zero_mistake_budget"
    assert artifact["ebt_arm_status"] == "projection_only_sidecar_schema_no_model_integration"
    assert artifact["gatemate_status"] == "blocked_no_rerun_operator_actions_required"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"

    assert artifact["status_movement_from_v21"]["blocker_delta_from_v21"] == -2
    assert artifact["status_movement_from_v21"]["fr11_controller_only_rows_moved"] == [
        "fr11-a",
        "fr11-b",
    ]
    assert artifact["prd_gap_summary"]["verifier_repair"]["publication_blocker_count"] == 5
    assert artifact["prd_gap_summary"]["fr11"]["publication_blocker_count"] == 0
    assert artifact["prd_gap_summary"]["ebt_arm_bridge"]["statuses_present"] == [
        "projection_only"
    ]
    assert artifact["prd_gap_summary"]["hardware_evidence"]["statuses_present"] == [
        "blocked",
        "gated_skipped",
    ]
    assert artifact["paper_ready_checks"] == [
        {
            "check": "capstone_ready",
            "passed": True,
            "reason": "matrix v22 authority loaded and row/blocker counts reconcile",
        },
        {
            "check": "publication_blocker_count_zero",
            "passed": False,
            "reason": "publication_blocker_count=9",
        },
        {
            "check": "headline_model_spec_gaps_clear",
            "passed": False,
            "reason": "headline_model_spec_gaps=1",
        },
        {
            "check": "headline_missing_inputs_clear",
            "passed": False,
            "reason": "missing_capstone_input_artifacts=1",
        },
    ]
    assert artifact["next_milestone_recommendation"].startswith(
        "2026.05.289: clear verifier/repair first"
    )

    assert sources[mod.MATRIX_V22_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V22_REL_PATH
    )
    assert sources["results/source_missing_repair_micro_panel.json"]["present"] is False
    assert artifact["missing_capstone_input_artifacts"] == [
        {
            "path": "results/source_missing_repair_micro_panel.json",
            "reason": "named by matrix v22 capstone_input_artifacts but not readable",
        }
    ]
    assert artifact["source_artifacts_loaded"] == {
        "named_by_matrix_v22": 7,
        "present": 6,
        "readable_json_object": 6,
        "missing_or_malformed": 1,
    }
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v22_and_named_capstone_input_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }
    assert artifact["no_new_model_execution"] is True
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["status_updates_written"] is False


def test_req_report_3094_allows_paper_ready_only_when_matrix_is_clear(tmp_path: Path) -> None:
    """REQ-REPORT-3094: paper readiness requires zero blockers and exact-grounded evidence."""

    matrix = _matrix_v22(blockers=False, model_gaps=False, missing_repair_input=False)
    _write_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["verifier_gain_status"] == "clean_verifier_gain_exact_grounded"
    assert artifact["repair_claim_status"] == "clean_or_retired"
    assert artifact["ebt_arm_status"] == "clean_adapter_implementation_evidence"
    assert artifact["gatemate_status"] == "clean_host_visible_output_ready"
    assert artifact["ssqa_status"] == "clean_host_visible_readback_ready"
    assert all(check["passed"] is True for check in artifact["paper_ready_checks"])


def test_req_report_3094_blocks_when_matrix_v22_is_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3094: missing matrix v22 blocks capstone readiness."""

    artifact = mod.build_artifact(tmp_path)

    assert artifact["capstone_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked_capstone_v288_preconditions")
    assert artifact["required_source_errors"] == [
        {
            "path": mod.MATRIX_V22_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_required_artifact",
        }
    ]


def test_req_report_3094_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3094: helper edges are deterministic and fail closed."""

    matrix = _matrix_v22(blockers=False, model_gaps=False, missing_repair_input=False)
    _write_sources(tmp_path, matrix)
    malformed = tmp_path / "malformed.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("bad") == "missing"
    assert mod.blocker_class("clean") == "none"
    assert mod.blocker_class("projection_only") == "projection_only"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._source_role(Path("results/prior_matrix.json")) == "matrix_input_context"

    not_ready = dict(matrix)
    not_ready["matrix_v22_ready"] = False
    _write_json(tmp_path, mod.MATRIX_V22_REL_PATH, not_ready)
    blocked = mod.build_artifact(tmp_path)
    assert "matrix v22 authority is not ready" in blocked["invariant_violations"]

    mismatched = dict(matrix)
    mismatched["rows_total"] = 999
    mismatched["status_counts"] = dict(matrix["status_counts"])
    mismatched["status_counts"]["clean"] += 1
    mismatched["publication_blocker_count"] = 1
    _write_json(tmp_path, mod.MATRIX_V22_REL_PATH, mismatched)
    mismatch_artifact = mod.build_artifact(tmp_path)
    assert "matrix v22 rows_total does not match rows" in mismatch_artifact["invariant_violations"]
    assert "matrix v22 status_counts do not match rows" in mismatch_artifact["invariant_violations"]
    assert (
        "publication_blocker_count does not match publication_blockers"
        in mismatch_artifact["invariant_violations"]
    )
