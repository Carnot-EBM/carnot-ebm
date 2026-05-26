"""Tests for Exp 3120 cross-corpus matrix v24.

Spec refs: REQ-REPORT-3120, SCENARIO-REPORT-3120.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v24_3120 as mod


REQUIRED_FIELDS = {
    "matrix_v24_ready",
    "rows_total",
    "status_counts",
    "publication_blocker_count",
    "blocker_delta_from_v23",
    "missing_artifacts",
    "headline_model_spec_gaps",
    "verifier_repair_status",
    "fr11_status",
    "architecture_boundary_status",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_DENSE = "unsloth/gemma-4-31B-it-GGUF"
GEMMA_MIDDLE = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
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
        "row_origin": "matrix_v23_test",
    }


def _matrix_v23() -> dict[str, Any]:
    rows = [
        _row("carry:clean", "clean", "unchanged_clean", "carry"),
        _row("carry:bounded", "bounded", "unchanged_bounded", "carry"),
        _row("carry:flagged", "flagged", "unchanged_flagged", "carry"),
        _row("capstone:v288_paper_readiness", "bounded", "paper_readiness", "capstone"),
        _row(
            "dot289:exp3099_local_sota_confidence_abstention_panel",
            "model_spec_gap",
            "local_sota_solution_verifier_gain",
            "live_llm",
        ),
        _row(
            "dot289:exp3100_z3_oracle_feedback",
            "blocked",
            "solver_grounded_repair_feedback",
            "solver",
        ),
        _row(
            "dot289:exp3101_local_sota_verifier_calibration_gate",
            "gated_skipped",
            "verifier_gain_recovery_gate",
            "gate",
        ),
        _row("dot289:exp3102_structured_repair_micro_panel", "missing", "repair_live_rerun", "repair"),
        _row(
            "dot289:exp3103_fr11_stress_promotion_boundary",
            "clean",
            "controller_only_stress_boundary_no_promotion",
            "fr11",
        ),
        _row(
            "dot289:exp3104_ebt_arm_sidecar_pipeline_boundary",
            "projection_only",
            "future_adapter_context",
            "ebt_arm",
        ),
        _row(
            "dot289:exp3105_clut_random_variate_sampler_microbench",
            "diagnostic_only",
            "cpu_microbench_diagnostic",
            "sampler",
        ),
        _row("dot289:exp3106_gatemate_operator_evidence", "blocked", "hardware_rerun_gate", "fpga"),
        _row(
            "dot289:exp3106_ssqa_readback_evidence",
            "gated_skipped",
            "host_visible_readback_gate",
            "fpga",
        ),
    ]
    blockers = [
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
    return {
        "artifact": "experiment_3107_cross_corpus_matrix_v23",
        "matrix_v23_ready": True,
        "rows_total": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows) for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "rows": rows,
        "headline_model_spec_gaps": [
            {
                "row_id": "dot289:exp3099_local_sota_confidence_abstention_panel",
                "source_artifact": mod.EXP3099_REL_PATH.as_posix(),
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            }
        ],
        "honest_verdict": "complete: matrix_v23_ready=true",
    }


def _capstone_v289(blocker_count: int) -> dict[str, Any]:
    return {
        "artifact": "experiment_3108_capstone_v289",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_sources(root: Path, *, contradiction: bool = False) -> None:
    matrix = _matrix_v23()
    _write_json(root, mod.MATRIX_V23_REL_PATH, matrix)
    _write_json(root, mod.CAPSTONE_V289_REL_PATH, _capstone_v289(matrix["publication_blocker_count"]))
    _write_json(
        root,
        mod.EXP3109_REL_PATH,
        {"archive_v289_activate_v290_ready": True, "honest_verdict": "complete: archive=true"},
    )
    _write_json(
        root,
        mod.EXP3110_REL_PATH,
        {
            "sota_model_manifest_ready": True,
            "mandatory_headline_model_ids": [QWEN, GEMMA_DENSE, GEMMA_MIDDLE],
            "present_model_ids": [GEMMA_MIDDLE],
            "missing_model_ids": [QWEN, GEMMA_DENSE],
            "cached_sota_pair_available": False,
            "selected_headline_model_ids": [GEMMA_MIDDLE],
            "headline_claim_allowed": True,
            "honest_verdict": "complete: sota_model_manifest_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3111_REL_PATH,
        {
            "certified_coherence_feedback_v3_ready": not contradiction,
            "certificate_count": 72,
            "unsat_core_count": 40,
            "exact_ground_truth_count": 72,
            "honest_verdict": "complete: certified_coherence_feedback_v3_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3112_REL_PATH,
        {
            "logic_regularized_verifier_pilot_ready": True,
            "exact_ground_truth_count": 27,
            "verifier_recall_delta": 1.0,
            "false_positive_delta": 0.0,
            "promotion_claim_made": False,
            "honest_verdict": "complete: logic_regularized_verifier_pilot_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3113_REL_PATH,
        {
            "diagnostic_verifier_calibration_v5_ready": True,
            "repair_gate_state": "unblocked",
            "verifier_gain_delta_with_certified_coherence": 0.5,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: diagnostic_verifier_calibration_v5_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3114_REL_PATH,
        {
            "fragment_verification_pilot_ready": True,
            "failing_fragment_count": 4,
            "repair_target_manifest_path": "results/fragment_verification_pilot_3114/repair_target_manifest.jsonl",
            "honest_verdict": "complete: fragment verification pilot ready",
        },
    )
    _write_json(
        root,
        mod.EXP3115_REL_PATH,
        {
            "repair_micro_panel_v4_artifact_ready": True,
            "repair_unblocked": True,
            "repair_run_executed": True,
            "repair_success_delta": 0.0,
            "false_repair_accept_rate": 0.0,
            "intent_preservation_rate": 0.0,
            "honest_verdict": "complete: repair_micro_panel_v4_artifact_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3116_REL_PATH,
        {
            "fr11_unsolvable_curriculum_ready": True,
            "controller_only": True,
            "no_weight_update_claim": True,
            "promotion_decision": "controller_only",
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "honest_verdict": "complete_fr11_unsolvable_curriculum_controller_only_guard_passed",
        },
    )
    _write_json(
        root,
        mod.EXP3117_REL_PATH,
        {
            "sidecar_score_correlation_boundary_v3_ready": True,
            "no_live_model_integration_claim": True,
            "no_weight_update_claim": True,
            "no_speedup_claim": True,
            "honest_verdict": "complete: sidecar_score_correlation_boundary_v3_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3118_REL_PATH,
        {
            "status": "success",
            "clut_backend_integration_boundary_v2_ready": True,
            "default_backend_preserved": True,
            "distribution_checks_passed": True,
            "hardware_claim_made": False,
            "hardware_commands_run": [],
            "honest_verdict": "complete: clut_cpu backend adapter is opt-in",
        },
    )
    _write_json(
        root,
        mod.EXP3119_REL_PATH,
        {
            "operator_evidence_ingestion_v4_ready": True,
            "gatemate_rerun_allowed": False,
            "ssqa_readback_allowed": False,
            "missing_operator_actions": [{"missing_item": "host_visible_smoke_evidence"}],
            "hardware_commands_run": [],
            "speedup_claim_made": False,
            "honest_verdict": "complete: operator_evidence_ingestion_v4_ready=true",
        },
    )


def test_req_report_3120_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3120: OpenSpec declares the v24 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3120" in spec
    assert "SCENARIO-REPORT-3120" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3120_builds_v24_without_promoting_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3120: .290 rows replace old blockers without overclaiming."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    blockers = {row["row_id"] for row in artifact["publication_blockers"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v24_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["rows_total"] == len(artifact["rows"]) == 26
    assert artifact["status_counts"] == {
        "clean": 4,
        "flagged": 1,
        "bounded": 6,
        "blocked": 1,
        "gated_skipped": 1,
        "missing": 0,
        "retired": 10,
        "projection_only": 1,
        "diagnostic_only": 2,
        "model_spec_gap": 0,
    }
    assert artifact["publication_blocker_count"] == 10
    assert artifact["blocker_delta_from_v23"] == 0
    assert artifact["missing_artifacts"] == []
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["dot289:exp3099_local_sota_confidence_abstention_panel"]["status"] == "retired"
    assert rows["dot289:exp3102_structured_repair_micro_panel"]["status"] == "retired"
    assert rows["dot290:exp3110_sota_model_spec_cache_manifest"]["status"] == "bounded"
    assert rows["dot290:exp3111_certified_coherence_feedback"]["status"] == "clean"
    assert rows["dot290:exp3112_logic_regularized_verifier_pilot"]["status"] == "diagnostic_only"
    assert rows["dot290:exp3113_diagnostic_verifier_calibration"]["status"] == "diagnostic_only"
    assert rows["dot290:exp3115_explicit_repair_gate_micro_panel"]["status"] == "bounded"
    assert rows["dot290:exp3116_fr11_curriculum_retention_guard"]["status"] == "bounded"
    assert rows["dot290:exp3117_ebt_arm_sidecar_score_correlation"]["status"] == "projection_only"
    assert rows["dot290:exp3118_clut_sampler_backend_integration"]["status"] == "bounded"
    assert rows["dot290:exp3119_gatemate_operator_evidence"]["status"] == "blocked"
    assert rows["dot290:exp3119_ssqa_readback_evidence"]["status"] == "gated_skipped"

    assert "dot290:exp3112_logic_regularized_verifier_pilot" not in blockers
    assert "dot290:exp3113_diagnostic_verifier_calibration" not in blockers
    assert artifact["headline_model_spec_gaps"] == [
        {
            "row_id": "dot290:exp3110_sota_model_spec_cache_manifest",
            "source_artifact": mod.EXP3110_REL_PATH.as_posix(),
            "missing_model_ids": [QWEN, GEMMA_DENSE],
            "present_model_ids": [GEMMA_MIDDLE],
            "reason": "mandated headline model cache coverage incomplete; cached SOTA pair unavailable",
        }
    ]
    assert artifact["verifier_repair_status"]["repair_claim_status"] == (
        "bounded_no_positive_repair_delta"
    )
    assert artifact["verifier_repair_status"]["repair_run_executed"] is True
    assert artifact["fr11_status"]["status"] == "bounded_controller_only_no_weight_update_claim"
    assert artifact["architecture_boundary_status"] == {
        "ebt_arm_status": "projection_only_no_live_model_integration",
        "clut_status": "bounded_cpu_only_no_hardware_speedup",
        "gatemate_status": "blocked_operator_evidence_incomplete",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
    }
    assert artifact["honest_verdict_status_contradictions"] == []
    assert sources[mod.EXP3110_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3110_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v23_capstone_v289_and_dot290_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "executes_solvers": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3120_missing_sources_and_contradictions_are_explicit(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3120: absent artifacts and verdict/readiness conflicts stay visible."""

    empty = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert empty["matrix_v24_ready"] is False
    assert empty["honest_verdict"].startswith("blocked_matrix_v24_preconditions")
    assert empty["publication_blocker_count"] == 0
    assert [row["path"] for row in empty["required_source_errors"]] == [
        mod.MATRIX_V23_REL_PATH.as_posix(),
        mod.CAPSTONE_V289_REL_PATH.as_posix(),
    ]
    assert mod.EXP3115_REL_PATH.as_posix() in {
        row["path"] for row in empty["missing_artifacts"]
    }

    _write_sources(tmp_path, contradiction=True)
    artifact = mod.build_artifact(tmp_path)
    contradictions = artifact["honest_verdict_status_contradictions"]

    assert artifact["matrix_v24_ready"] is True
    assert contradictions == [
        {
            "experiment_id": "exp3111",
            "path": mod.EXP3111_REL_PATH.as_posix(),
            "ready_field": "certified_coherence_feedback_v3_ready",
            "ready_value": False,
            "honest_verdict": "complete: certified_coherence_feedback_v3_ready=true",
            "reason": "success verdict contradicts false readiness field",
        }
    ]
    assert {
        row["row_id"]: row["status"] for row in artifact["rows"]
    }["dot290:exp3111_certified_coherence_feedback"] == "blocked"


def test_req_report_3120_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3120: helper behavior is deterministic and fail-closed."""

    _write_sources(tmp_path)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=2.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v24_ready"] is True
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
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("bad") is None
    assert mod._has_success_verdict("success_ready") is True
    assert mod._has_success_verdict("passed: ok") is True
    assert mod._has_success_verdict("shipped_ready") is True
    assert mod._has_blocked_verdict("blocked_precondition") is True
    assert mod._has_blocked_verdict("failed: nope") is True
    assert mod._status_for_ready(False, {}, "ready") == "missing"
    assert mod._status_for_ready(True, {"ready": True}, "ready") == "clean"
    assert mod._status_for_ready(True, {"ready": False}, "ready") == "blocked"
    assert mod._status_for_ready(True, {"status": "success"}, "ready") == "clean"
    assert mod._claim_entry({"row_id": "x", "status": "bad"})["status"] == "missing"
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean", "scope", "evidence")]})[
        0
    ]["row_id"] == "carry"
    assert mod._capstone_v289_row({"capstone_ready": False})["status"] == "blocked"
    assert mod._replacement_row_id("not-replaced") == ""

    assert mod._model_manifest_row({})["status"] == "missing"
    assert mod._model_manifest_row({"sota_model_manifest_ready": False})["status"] == "blocked"
    assert (
        mod._model_manifest_row(
            {"sota_model_manifest_ready": True, "headline_claim_allowed": False}
        )["status"]
        == "model_spec_gap"
    )
    assert (
        mod._model_manifest_row(
            {
                "sota_model_manifest_ready": True,
                "headline_claim_allowed": True,
                "selected_headline_model_ids": [GEMMA_MIDDLE],
                "cached_sota_pair_available": True,
            }
        )["status"]
        == "clean"
    )

    assert mod._logic_pilot_row({})["status"] == "missing"
    assert mod._logic_pilot_row({"logic_regularized_verifier_pilot_ready": False})[
        "status"
    ] == "blocked"
    assert mod._logic_pilot_row(
        {"logic_regularized_verifier_pilot_ready": True, "promotion_claim_made": True}
    )["status"] == "bounded"

    assert mod._diagnostic_calibration_row({})["status"] == "missing"
    assert mod._diagnostic_calibration_row({"diagnostic_verifier_calibration_v5_ready": False})[
        "status"
    ] == "blocked"

    assert mod._repair_micro_panel_row({})["status"] == "missing"
    assert mod._repair_micro_panel_row({"repair_micro_panel_v4_artifact_ready": False})[
        "status"
    ] == "blocked"
    assert (
        mod._repair_micro_panel_row(
            {
                "repair_micro_panel_v4_artifact_ready": True,
                "repair_unblocked": True,
                "repair_run_executed": True,
                "repair_success_delta": 0.5,
                "intent_preservation_rate": 1.0,
            }
        )["status"]
        == "clean"
    )

    assert mod._fr11_curriculum_row({})["status"] == "missing"
    assert mod._fr11_curriculum_row({"fr11_unsolvable_curriculum_ready": False})[
        "status"
    ] == "blocked"
    assert mod._fr11_curriculum_row(
        {"fr11_unsolvable_curriculum_ready": True, "promotion_decision": "promoted"}
    )["status"] == "clean"

    blocked_verdict_conflict = mod._verdict_status_contradictions(
        [
            {
                "experiment_id": "expX",
                "path": "results/x.json",
                "ready_field": "ready",
                "payload": {"ready": True, "honest_verdict": "blocked_precondition"},
            }
        ]
    )
    assert blocked_verdict_conflict[0]["reason"] == (
        "blocked verdict contradicts true readiness field"
    )

    assert mod._verifier_repair_status(
        {"exp3115": {"repair_success_delta": 1.0}},
        [_row("dot290:exp3115_explicit_repair_gate_micro_panel", "clean", "repair_live_rerun", "repair")],
    )["repair_claim_status"] == "clean_repair_claim"
    assert mod._verifier_repair_status({"exp3115": {}}, [])["repair_claim_status"] == (
        "missing_repair_artifact"
    )
    assert mod._verifier_repair_status(
        {"exp3115": {}},
        [_row("dot290:exp3115_explicit_repair_gate_micro_panel", "blocked", "repair_live_rerun", "repair")],
    )["repair_claim_status"] == "blocked_repair_claim"

    assert mod._fr11_status({"fr11_unsolvable_curriculum_ready": False})[
        "status"
    ] == "blocked_fr11_precondition"
    assert mod._fr11_status({"fr11_unsolvable_curriculum_ready": True})[
        "status"
    ] == "clean_weight_update_claim_supported"

    violations = mod._invariant_violations(
        {"matrix_v23_ready": False},
        {"capstone_ready": False},
        [_row("flagged", "flagged", "scope", "evidence")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v23 authority is not ready",
        "capstone .289 authority is not ready",
        "status_counts keys do not match required v24 statuses",
        "status_counts do not sum to rows_total",
        "publication_blocker_count does not match row statuses",
    ]
