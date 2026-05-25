"""Tests for Exp 3039 milestone .284 capstone.

Spec refs: REQ-REPORT-3039, SCENARIO-REPORT-3039.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v284_3039 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "repair_claim_status",
    "fr11_self_learning_status",
    "gatemate_status",
    "ssqa_status",
    "matrix_v18_summary",
    "blockers_remaining",
    "next_milestone_focus",
    "recommended_next_actions",
    "inference_substrate",
    "honest_verdict",
}

FORBIDDEN_TOP_LEVEL = {
    "model_specs",
    "target_model",
    "cuda",
    "CUDA",
    "gguf",
    "GGUF",
    "gpu_inventory",
    "headline_models_used",
    "live_model_metadata",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_row(
    experiment_id: str,
    status: str,
    task_class: str,
    *,
    repair_claim_status: str = "not_applicable",
    fr11_self_learning_promotable: bool | None = None,
    gatemate_output_contract_ready: bool | None = None,
    host_visible_output_observed: bool | None = None,
    ssqa_gate_status: str | None = None,
    upstream_flags: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "milestone": "2026.05.284",
        "status": status,
        "task_class": task_class,
        "planned_path": f"results/{experiment_id}.json",
        "actual_path": f"results/{experiment_id}.json",
        "planned_path_present": True,
        "actual_path_present": status != "missing",
        "source_honest_verdict": f"complete: {experiment_id} {status}",
        "repair_claim_status": repair_claim_status,
        "fr11_self_learning_promotable": fr11_self_learning_promotable,
        "gatemate_output_contract_ready": gatemate_output_contract_ready,
        "host_visible_output_observed": host_visible_output_observed,
        "ssqa_gate_status": ssqa_gate_status,
        "upstream_flags": upstream_flags or [],
        "summary": summary or {},
    }


def _matrix_v18() -> dict[str, Any]:
    rows = [
        _matrix_row("exp3026", "projection_only", "archive_activation"),
        _matrix_row(
            "exp3027",
            "flagged",
            "methodology_corrigendum",
            repair_claim_status="rerun_required",
            upstream_flags=["DURATION_TOO_SHORT:critical"],
        ),
        _matrix_row(
            "exp3028",
            "flagged",
            "repair_rerun",
            repair_claim_status="clean_candidate_flagged",
            upstream_flags=["TAUTOLOGY:critical", "METHODOLOGY_MISSING:warn"],
            summary={
                "clean_repair_rerun_ready": True,
                "n_tasks": 24,
                "n_live_transcripts": 24,
                "pass_at_1_delta": 0.375,
                "false_accept_delta": 0.0,
                "tautology_gate_clean": True,
            },
        ),
        _matrix_row(
            "exp3029",
            "flagged",
            "repair_boundary_audit",
            repair_claim_status="bounded",
            upstream_flags=["METHODOLOGY_MISSING:warn"],
            summary={"repair_claim_status": "bounded", "bounded_claim_count": 1},
        ),
        _matrix_row(
            "exp3030",
            "clean",
            "validator_frontier_corrigendum",
            summary={
                "verified_region_count": 40,
                "unresolved_region_count": 2,
                "fallback_only_count": 1,
                "missing_authority_count": 0,
            },
        ),
        _matrix_row(
            "exp3031",
            "flagged",
            "dccd_structured_repair_panel",
            repair_claim_status="pilot_panel_flagged",
            upstream_flags=["DURATION_TOO_SHORT:critical"],
            summary={"n_cases": 3, "false_accept_delta": 0.0, "intent_drift_delta": 0.0},
        ),
        _matrix_row(
            "exp3032",
            "clean",
            "fr11_heldout_replay",
            fr11_self_learning_promotable=False,
            summary={
                "continuous_self_learning_tested": True,
                "heldout_trace_count": 8,
                "feasible_infeasible_auc_delta": 0.5,
                "shuffled_feedback_delta": -0.5,
                "tautology_risk_cleared": True,
            },
        ),
        _matrix_row(
            "exp3033",
            "clean",
            "fr11_nonforgetting_stress",
            fr11_self_learning_promotable=True,
            summary={
                "fr11_self_learning_promotable": True,
                "promotion_decision": "controller_only_promotable",
                "heldout_delta_after_update": 0.875,
                "shuffled_control_delta": -2.125,
                "drift_failure_count": 0,
            },
        ),
        _matrix_row(
            "exp3034",
            "blocked",
            "gatemate_output_contract",
            gatemate_output_contract_ready=False,
            host_visible_output_observed=False,
            summary={"operator_action_count": 3},
        ),
        _matrix_row(
            "exp3035",
            "gated_skipped",
            "gatemate_output_shim",
            gatemate_output_contract_ready=False,
            host_visible_output_observed=False,
        ),
        _matrix_row(
            "exp3036",
            "gated_skipped",
            "gatemate_host_visible_flash_smoke",
            gatemate_output_contract_ready=False,
            host_visible_output_observed=False,
        ),
        _matrix_row(
            "exp3037",
            "gated_skipped",
            "ssqa_boundary",
            host_visible_output_observed=False,
            ssqa_gate_status="gate_skipped",
            summary={"ssqa_boundary_ready": True, "resource_report_count": 0},
        ),
        _matrix_row("exp3038", "clean", "cross_corpus_matrix"),
        _matrix_row("exp3039", "missing", "capstone"),
    ]
    return {
        "schema": "carnot.cross_corpus_matrix.v18_284_task_coverage.v1",
        "artifact": "experiment_3038_cross_corpus_matrix_v18",
        "run_date": "20260525",
        "milestone": "2026.05.284",
        "matrix_v18_ready": True,
        "rows_total": 14,
        "clean": 4,
        "flagged": 4,
        "blocked": 1,
        "gated_skipped": 3,
        "projection_only": 1,
        "pilot_only": 0,
        "missing": 1,
        "retired": 0,
        "matrix_rows": rows,
        "baseline_v17_summary": {
            "matrix_v17_ready": True,
            "clean": 40,
            "flagged": 29,
            "blocked": 10,
            "gated_skipped": 3,
            "projection_only": 10,
            "pilot_only": 4,
            "missing": 1,
        },
        "recommended_next_actions": [
            "Keep repair wording bounded until matrix and capstone repair blockers clear.",
            "Carry FR-11 forward as controller-only self-learning.",
            "Resolve the GateMate host-visible output pinout before rerunning shim.",
            "Keep SSQA as gate-skipped until GateMate host-visible output is observed.",
        ],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "executes_models": False,
            "source": "checked_in_artifacts",
        },
        "honest_verdict": (
            "complete: matrix_v18_ready=true; rows_total=14; clean=4; flagged=4; "
            "blocked=1; gated_skipped=3; projection_only=1; pilot_only=0; missing=1; "
            "retired=0"
        ),
    }


def _write_ready_sources(root: Path) -> None:
    _write_json(root, mod.MATRIX_V18_REL_PATH, _matrix_v18())
    _write_json(
        root,
        mod.EXP3028_REL_PATH,
        {
            "clean_repair_rerun_ready": True,
            "repair_controller_clean": True,
            "clean_repair_claim_promotable_candidate": True,
            "n_tasks": 24,
            "n_live_transcripts": 24,
            "pass_at_1_delta": 0.375,
            "pass_at_k_delta": 0.375,
            "false_accept_delta": 0.0,
            "tautology_gate_clean": True,
            "intent_drift_count": 0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "model_specs": [{"hf_id": "fixture/gemma-4-26b"}],
            "headline_models_used": ["fixture/gemma-4-26b"],
            "honest_verdict": "complete: clean_repair_rerun_ready=true; n_tasks=24",
        },
    )
    _write_json(
        root,
        mod.EXP3029_REL_PATH,
        {
            "repair_promotion_boundary_ready": True,
            "repair_claim_status": "bounded",
            "promotable_claims": [],
            "bounded_claims": [{"claim_id": "exp3028_clean_repair_candidate"}],
            "retired_or_blocked_claims": [
                {"claim_id": "headline_sota_repair_clean_methodology"}
            ],
            "honest_verdict": "complete: repair_claim_status=bounded",
        },
    )
    _write_json(
        root,
        mod.EXP3030_REL_PATH,
        {
            "validator_frontier_corrigendum_ready": True,
            "verified_region_count": 40,
            "irrelevant_region_count": 2,
            "unresolved_region_count": 2,
            "fallback_only_count": 1,
            "missing_authority_count": 0,
            "honest_verdict": "complete: validator_frontier_corrigendum_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3032_REL_PATH,
        {
            "fr11_heldout_replay_ready": True,
            "continuous_self_learning_tested": True,
            "heldout_trace_count": 8,
            "feasible_infeasible_auc_delta": 0.5,
            "shuffled_feedback_delta": -0.5,
            "tautology_risk_cleared": True,
            "information_asymmetry_enforced": True,
            "invariant_violations": [],
            "honest_verdict": "complete_fr11_heldout_replay_ready",
        },
    )
    _write_json(
        root,
        mod.EXP3033_REL_PATH,
        {
            "fr11_nonforgetting_stress_ready": True,
            "fr11_self_learning_promotable": True,
            "promotion_decision": "controller_only_promotable",
            "prior_retention_delta": 0.0,
            "heldout_delta_after_update": 0.875,
            "shuffled_control_delta": -2.125,
            "drift_failures": [],
            "honest_verdict": "complete_controller_only_promotable",
        },
    )
    _write_json(
        root,
        mod.EXP3034_REL_PATH,
        {
            "gatemate_output_contract_ready": False,
            "host_visible_io_plan_ready": False,
            "selected_output_path": "explicit_no_ready_contract",
            "exact_operator_action_required": [
                "Provide an authoritative GateMate output pinout.",
                "Choose and commit the matching host reader command.",
                "Keep downstream flash smoke gated.",
            ],
            "speedup_claim_made": False,
            "sampler_claim_made": False,
            "thermodynamic_claim_made": False,
            "honest_verdict": "complete: blocked_gatemate_output_contract_pinout_missing",
        },
    )
    _write_json(
        root,
        mod.EXP3035_GATE_CHECK_REL_PATH,
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3037_REL_PATH,
        {
            "ssqa_boundary_ready": True,
            "ssqa_gate_status": "gate_skipped",
            "ssqa_performance_claim_allowed": False,
            "resource_report_paths": [],
            "exact_blocker_or_next_action": [
                "Exp 3036 artifact missing.",
                "Provide an authoritative GateMate output pinout.",
            ],
            "inference_substrate": {
                "kind": "ssqa_hardware_boundary_artifact",
                "host_visible_output_observed": False,
                "speedup_claim": False,
                "latency_claim": False,
                "board_performance_claim": False,
            },
            "honest_verdict": "complete: ssqa_gate_skipped_exp3036_missing",
        },
    )


def test_req_report_3039_spec_entry_present() -> None:
    """REQ-REPORT-3039: the research-reporting spec anchors Exp 3039."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3039" in spec
    assert "SCENARIO-REPORT-3039" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3039_builds_bounded_capstone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3039: .284 closes without paper or hardware overclaim."""
    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.25)
    blockers = {row["area"]: row for row in artifact["blockers_remaining"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["repair_claim_status"] == "bounded"
    assert artifact["fr11_self_learning_status"] == "controller_only_promotable"
    assert artifact["gatemate_status"] == "blocked_pinout_missing_bounded"
    assert artifact["ssqa_status"] == "gate_skipped_bounded_no_performance_claim"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=false")

    assert artifact["matrix_v18_summary"]["rows_total"] == 14
    assert artifact["matrix_v18_summary"]["counts_match_rows"] is True
    assert artifact["matrix_v18_summary"]["status_by_experiment"]["exp3039"] == "missing"
    assert artifact["matrix_v18_summary"]["flagged"] == 4
    assert artifact["matrix_v18_summary"]["gated_skipped"] == 3

    assert blockers["repair"]["status"] == "bounded"
    assert blockers["matrix_nonclean"]["counts"]["flagged"] == 4
    assert blockers["gatemate"]["status"] == "blocked_pinout_missing_bounded"
    assert blockers["ssqa"]["status"] == "gate_skipped_bounded_no_performance_claim"
    assert blockers["exp3036"]["status"] == "missing_or_gated_skipped"

    assert artifact["what_284_proved"]["validator_frontier"]["verified_region_count"] == 40
    assert artifact["what_284_proved"]["fr11_self_learning"]["heldout_trace_count"] == 8
    assert len(artifact["recommended_next_actions"]) == 5
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["status_updates_written"] is False


def test_req_report_3039_keeps_live_metadata_inside_citations(tmp_path: Path) -> None:
    """REQ-REPORT-3039: aggregation output has no top-level live-model metadata."""
    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert FORBIDDEN_TOP_LEVEL.isdisjoint(artifact.keys())
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "source": "checked_in_artifacts",
    }
    serialized_citations = json.dumps(artifact["cited_upstream_artifacts"], sort_keys=True)
    assert "fixture/gemma-4-26b" in serialized_citations
    artifact_without_citations = dict(artifact)
    artifact_without_citations.pop("cited_upstream_artifacts")
    assert "fixture/gemma-4-26b" not in json.dumps(artifact_without_citations, sort_keys=True)


def test_req_report_3039_blocks_without_matrix_v18(tmp_path: Path) -> None:
    """REQ-REPORT-3039: matrix v18 is required for capstone readiness."""
    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V18_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["capstone_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_matrix_v18_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp3038",
            "path": mod.MATRIX_V18_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_3039_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3039: write_artifact emits the deliverable JSON."""
    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["paper_ready"] is False
    assert saved["duration_s"] == pytest.approx(0.125)
    assert saved["source_checksums"][mod.EXP3033_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3033_REL_PATH
    )


def test_req_report_3039_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3039: helper edges preserve malformed and unusual states."""
    malformed = tmp_path / "bad.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._status_counts([]) == {
        "clean": 0,
        "flagged": 0,
        "blocked": 0,
        "gated_skipped": 0,
        "projection_only": 0,
        "pilot_only": 0,
        "missing": 0,
        "retired": 0,
    }
    assert mod._status_counts([{"status": "unexpected"}])["missing"] == 1
    assert mod._rows_by_exp([{"experiment_id": "expX"}]) == {"expX": {"experiment_id": "expX"}}
    assert mod._safe_bool(True) is True
    assert mod._safe_bool("true") is None
    assert mod._mapping({"x": 1}) == {"x": 1}
    assert mod._mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []
    assert (
        mod._repair_claim_status({"exp3029": {}}, {"exp3029": {"repair_claim_status": "bounded"}})
        == "bounded"
    )
    assert mod._repair_claim_status({"exp3029": {}}, {}) == "unknown"
    assert mod._fr11_self_learning_status({"exp3032": {}, "exp3033": {}}, {}) == (
        "blocked_or_unproven"
    )
    assert (
        mod._gatemate_status(
            {
                "exp3034": {"gatemate_output_contract_ready": True},
                "exp3036": {"host_visible_output_observed": True},
            },
            {},
        )
        == "host_visible_output_observed"
    )
    assert mod._gatemate_status({"exp3034": {"gatemate_output_contract_ready": False}}, {}) == (
        "blocked_or_unbounded"
    )
    assert (
        mod._ssqa_status(
            {"exp3037": {"ssqa_gate_status": "run", "resource_report_paths": ["pnr.rpt"]}},
            {},
        )
        == "bounded_resource_evidence"
    )
    assert mod._ssqa_status({"exp3037": {"ssqa_gate_status": "blocked"}}, {}) == (
        "blocked_or_unbounded"
    )
    assert mod._missing_artifacts(
        [
            {
                "planned_path_present": True,
                "present": False,
                "planned_path": "planned.json",
                "actual_path": "actual.json",
            }
        ]
    ) == ["actual.json"]
    assert mod._int_or(True, 7) == 7
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("not-float") is None
