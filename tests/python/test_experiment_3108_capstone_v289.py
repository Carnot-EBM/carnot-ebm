"""Tests for the Exp 3108 milestone .289 capstone.

Spec refs: REQ-REPORT-3108, SCENARIO-REPORT-3108.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v289_3108 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "verifier_gain_status",
    "repair_claim_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "sampler_hardware_status",
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


def _matrix_v23(
    *,
    blockers: bool = True,
    model_gaps: bool = True,
    missing_repair_input: bool = True,
) -> dict[str, Any]:
    if blockers:
        rows = [
            _row("verifier-model-gap", "model_spec_gap", "local_sota_solution_verifier_gain", "live_llm"),
            _row("verifier-gate", "gated_skipped", "verifier_gain_recovery_gate", "gate"),
            _row("formal-blocked", "blocked", "solver_grounded_repair_feedback", "z3_feedback"),
            _row("repair-bounded", "bounded", "repair_headline_boundary", "repair"),
            _row("repair-missing", "missing", "repair_live_rerun", "repair"),
            {
                **_row(
                    "fr11-stress",
                    "clean",
                    "controller_only_stress_boundary_no_promotion",
                    "fr11_resyn_kancl_stress",
                ),
                "summary": {
                    "promotion_decision": "blocked",
                    "soundness_mistakes": 0,
                    "completeness_mistakes": 12,
                },
            },
            _row("ebt-sidecar", "projection_only", "future_adapter_context", "ebt_arm"),
            _row("clut-diagnostic", "diagnostic_only", "cpu_microbench_diagnostic", "clut_sampler"),
            _row("gatemate-blocked", "blocked", "hardware_rerun_gate", "gatemate"),
            _row("ssqa-gated", "gated_skipped", "host_visible_readback_gate", "ssqa"),
            _row("capstone-paper", "bounded", "paper_readiness", "capstone"),
            _row("archive-clean", "clean", "milestone_activation", "archive"),
            _row("old-retired", "retired", "retired_claim", "archive"),
        ]
    else:
        rows = [
            _row("verifier-clean", "clean", "local_sota_solution_verifier_gain", "live_llm"),
            _row("repair-retired", "retired", "repair_headline_boundary", "repair"),
            {
                **_row("fr11-promoted", "clean", "controller_only_self_learning_budget", "fr11"),
                "summary": {
                    "promotion_decision": "controller_only",
                    "soundness_mistakes": 0,
                    "completeness_mistakes": 0,
                },
            },
            _row("ebt-clean", "clean", "future_adapter_context", "ebt_arm"),
            _row("clut-clean", "clean", "hardware_sampler_adjacency", "clut_sampler"),
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
        mod.MATRIX_V23_REL_PATH.as_posix(),
        "results/source_matrix_v22.json",
        "results/source_capstone_v288.json",
        "results/source_blocker_ledger.json",
        "results/source_fr11.json",
        "results/source_ebt_arm.json",
        "results/source_clut_sampler.json",
        "results/source_gatemate_ssqa.json",
    ]
    if missing_repair_input:
        input_paths.append("results/source_missing_repair_micro_panel.json")
    return {
        "artifact": "experiment_3107_cross_corpus_matrix_v23",
        "matrix_v23_ready": True,
        "rows_total": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers_rows),
        "publication_blockers": blockers_rows,
        "rows": rows,
        "blocker_delta_from_v22": 0 if blockers else -9,
        "blocker_reconciliation_from_exp3096": {
            "publication_blocker_count_before": len(blockers_rows),
            "publication_blocker_count_after": len(blockers_rows),
            "blocker_delta_from_v22": 0 if blockers else -9,
            "decreases": [{"count": 8, "row_ids": ["old"], "reason": "old blockers retired"}],
            "increases": [{"count": 8, "row_ids": ["new"], "reason": "new blockers added"}],
            "neutral_replacements": [],
        },
        "capstone_input_artifacts": input_paths,
        "missing_artifacts": [
            {
                "path": "results/source_missing_repair_micro_panel.json",
                "reason": "expected .289 repair micro-panel artifact is absent",
            }
        ]
        if missing_repair_input
        else [],
        "headline_model_spec_gaps": [
            {
                "row_id": "verifier-model-gap",
                "source_artifact": "results/source_live_panel.json",
                "missing_model_ids": [],
                "present_model_ids": ["model-a", "model-b"],
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            }
        ]
        if model_gaps
        else [],
        "honest_verdict": "complete: matrix_v23_ready=true",
        "inference_substrate": {
            "kind": "aggregation_from_checked_in_artifacts",
            "executes_models": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
        },
    }


def _write_sources(root: Path, matrix: dict[str, Any]) -> None:
    _write_json(root, mod.MATRIX_V23_REL_PATH, matrix)
    for rel_path in matrix["capstone_input_artifacts"]:
        if rel_path == mod.MATRIX_V23_REL_PATH.as_posix():
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


def test_req_report_3108_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3108: OpenSpec declares the .289 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3108" in spec
    assert "SCENARIO-REPORT-3108" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3108_builds_capstone_without_paper_overclaim(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3108: matrix v23 statuses and claim boundaries stay visible."""

    matrix = _matrix_v23()
    _write_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=9.5)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 9
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["verifier_gain_status"] == "model_spec_gap_or_gated_verifier_gain_recovery_incomplete"
    assert artifact["repair_claim_status"] == "blocked_gated_missing_verifier_gated_repair_not_promoted"
    assert artifact["fr11_self_learning_status"] == "clean_controller_only_soundness_zero_completeness_promotion_blocked"
    assert artifact["ebt_arm_status"] == "projection_only_sidecar_pipeline_no_model_integration"
    assert artifact["sampler_hardware_status"] == "diagnostic_only_cpu_microbench_no_hardware_speedup"
    assert artifact["gatemate_status"] == "blocked_no_rerun_operator_actions_required_no_speedup_claim"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_readback_missing"

    assert artifact["matrix_v23_summary"]["blocker_delta_from_v22"] == 0
    assert artifact["status_movement_from_v22"]["blocker_delta_from_v22"] == 0
    assert artifact["prd_gap_summary"]["verifier_repair"]["publication_blocker_count"] == 5
    assert artifact["prd_gap_summary"]["fr11_self_learning"]["publication_blocker_count"] == 0
    assert artifact["prd_gap_summary"]["ebt_arm_bridge"]["statuses_present"] == ["projection_only"]
    assert artifact["prd_gap_summary"]["sampler_hardware_adjacency"]["statuses_present"] == [
        "diagnostic_only"
    ]
    assert artifact["prd_gap_summary"]["gatemate_ssqa_evidence"]["statuses_present"] == [
        "blocked",
        "gated_skipped",
    ]
    assert artifact["paper_ready_checks"] == [
        {
            "check": "capstone_ready",
            "passed": True,
            "reason": "matrix v23 authority loaded and row/blocker counts reconcile",
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
        {
            "check": "headline_exact_grounding_clear",
            "passed": False,
            "reason": "headline_blocking_rows=5",
        },
    ]
    assert artifact["next_milestone_recommendation"].startswith(
        "2026.05.290: clear verifier/repair first (5 blocker rows"
    )

    assert sources[mod.MATRIX_V23_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V23_REL_PATH
    )
    assert sources["results/source_missing_repair_micro_panel.json"]["present"] is False
    assert artifact["missing_capstone_input_artifacts"] == [
        {
            "path": "results/source_missing_repair_micro_panel.json",
            "reason": "named by matrix v23 capstone_input_artifacts but not readable",
        }
    ]
    assert artifact["source_artifacts_loaded"] == {
        "named_by_matrix_v23": 9,
        "present": 8,
        "readable_json_object": 8,
        "missing_or_malformed": 1,
    }
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v23_and_named_capstone_input_artifacts",
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


def test_req_report_3108_allows_paper_ready_only_when_matrix_is_clear(tmp_path: Path) -> None:
    """REQ-REPORT-3108: paper readiness requires zero blockers and exact-grounded evidence."""

    matrix = _matrix_v23(blockers=False, model_gaps=False, missing_repair_input=False)
    _write_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["verifier_gain_status"] == "clean_verifier_gain_exact_grounded"
    assert artifact["repair_claim_status"] == "clean_or_retired"
    assert artifact["fr11_self_learning_status"] == "clean_controller_only_zero_mistake_budget"
    assert artifact["ebt_arm_status"] == "clean_adapter_implementation_evidence"
    assert artifact["sampler_hardware_status"] == "clean_sampler_hardware_adjacency_evidence"
    assert artifact["gatemate_status"] == "clean_host_visible_output_ready"
    assert artifact["ssqa_status"] == "clean_host_visible_readback_ready"
    assert all(check["passed"] is True for check in artifact["paper_ready_checks"])


def test_req_report_3108_blocks_when_matrix_v23_is_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3108: missing matrix v23 blocks capstone readiness."""

    artifact = mod.build_artifact(tmp_path)

    assert artifact["capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_capstone_v289_preconditions")
    assert artifact["paper_ready"] is False
    assert artifact["required_source_errors"] == [
        {
            "path": mod.MATRIX_V23_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_required_artifact",
        }
    ]


def test_req_report_3108_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3108: helper behavior is deterministic and fail-closed."""

    matrix = _matrix_v23(missing_repair_input=False)
    _write_sources(tmp_path, matrix)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["missing_capstone_input_artifacts"] == []
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("diagnostic") == "diagnostic_only"
    assert mod.normal_status("bad") == "missing"
    assert mod.blocker_class("clean") == "none"
    assert mod.blocker_class("diagnostic_only") == "diagnostic_only"
    assert mod.blocker_class("model_spec_gap") == "model_spec_gap"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._source_role(Path("results/some_capstone.json")) == "capstone_input_context"
    assert mod._source_role(Path("results/some_matrix.json")) == "matrix_input_context"
    assert mod._experiment_id(Path("results/experiment_9999_x.json"), {}) == "experiment_9999_x"
    assert mod._experiment_id(Path("ops/status.md"), {}) == "source:ops/status.md"
    assert mod._publication_blockers({}, [_row("repair", "missing", "repair_live_rerun", "repair")])[
        0
    ]["status"] == "missing"
    assert mod._publication_blocker_count({}, []) == 0
    assert mod._matrix_status_counts({}) == {status: 0 for status in mod.STATUSES}
    assert mod._fr11_self_learning_status(
        [_row("fr11-flag", "blocked", "controller_only_self_learning_budget", "fr11")]
    ) == "flagged_controller_only_budget_exceeded"
    assert mod._ebt_arm_status([_row("ebt-block", "blocked", "future_adapter_context", "ebt")]) == (
        "bounded_or_blocked_no_model_integration"
    )
    assert mod._sampler_hardware_status([]) == "missing_sampler_hardware_adjacency_evidence"

    violations = mod._invariant_violations(
        {"matrix_v23_ready": False, "rows_total": 2, "status_counts": {"clean": 1}},
        [_row("one", "clean", "scope", "evidence")],
        {status: 0 for status in mod.STATUSES},
        [],
        3,
        [{"path": "missing", "reason": "missing_or_malformed_required_artifact"}],
    )
    assert violations == [
        "required source artifacts missing or malformed",
        "matrix v23 authority is not ready",
        "matrix v23 rows_total does not match rows",
        "matrix v23 status_counts do not match rows",
        "publication_blocker_count does not match publication_blockers",
    ]
