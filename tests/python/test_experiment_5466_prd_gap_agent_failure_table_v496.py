"""Tests for Exp5466 V496 PRD gap and agent-failure table.

Spec refs: REQ-REPORT-5466, SCENARIO-REPORT-5466,
SCENARIO-REPORT-5466-MISSING-OR-SKIPPED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import prd_gap_agent_failure_table_v496_5466 as exp5466


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / exp5466.OUTPUT_REL_PATH


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(root: Path, rel_path: str, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(f"{text}\n", encoding="utf-8")


def _lane_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def _failure(report: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(report["agent_failure_taxonomy"][key])


def _minimal_upstreams() -> dict[str, dict[str, Any]]:
    return {
        "results/experiment_5454_transition_v496.json": {
            "honest_verdict": "complete: transition receipt",
            "status": "complete",
            "milestone": "2026.07.496",
            "blocked_lanes": [{"lane": "token_internal_access"}],
            "closed_lanes": [{"lane": "prior_closed"}],
            "partial_lanes": [{"lane": "prior_partial"}],
            "honest_null_lanes": [{"lane": "hardware_speedup_claim"}],
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "results/experiment_5455_source_delta_v496.json": {
            "honest_verdict": "complete: source delta appended",
            "status": "complete",
            "milestone": "2026.07.496",
            "new_actionable_findings_count": 4,
            "retired_scopes_reopened": False,
            "research_references_updated": True,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json": {
            "honest_verdict": (
                "complete: guided-decoding corrigendum clean; Exp5444 headline "
                "readiness blocked"
            ),
            "status": "complete",
            "milestone": "2026.07.496",
            "guided_decoding_corrigendum_clean": True,
            "prior_flagged_adversarial": True,
            "invalid_tautological_fields": ["abstention_rate"],
            "rerun_gate_reason": "prior_headline_blocked",
            "inference_substrate": "posthoc_row_metric_audit_no_llm",
        },
        "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json": {
            "honest_verdict": "complete: live panel ran; readiness false",
            "status": "complete",
            "milestone": "2026.07.496",
            "verifier_guided_decoding_ready": False,
            "lcd_bias_check_passed": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "precondition_details": {"all_passed": True, "blocked_preconditions": []},
            "gpu_offload_verified": True,
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "runtime_receipt": {"offload_evidence": True},
            "model_specs": [{"local_path_available": True, "gpu_offload_verified": True}],
            "inference_substrate": "live_llm_inference",
        },
        "results/experiment_5458_minimal_core_claim_repair_v496.json": {
            "honest_verdict": "complete: deterministic minimal-core repairs accepted",
            "milestone": "2026.07.496",
            "minimal_core_repair_ready": True,
            "exact_final_authority": True,
            "repaired_accept_rate_after_exact_recheck": 1.0,
            "unrepaired_reject_rate": 1.0,
            "inference_substrate": "deterministic_solver_core_repair_no_llm",
        },
        "results/experiment_5459_constraint_distortion_guard_v496.json": {
            "honest_verdict": "complete: deterministic constraint-distortion guard ready",
            "milestone": "2026.07.496",
            "distortion_guard_ready": True,
            "truth_preserving_compliance_rate": 1.0,
            "unsupported_fabrication_rate": 0.0,
            "exact_final_authority": True,
            "inference_substrate": "deterministic_distortion_guard_no_llm",
        },
        "results/experiment_5460_csl_policy_bandit_v496.json": {
            "honest_verdict": "complete: frozen-model governed CSL policy updated",
            "status": "complete",
            "milestone": "2026.07.496",
            "continuous_self_learning_task": True,
            "csl_policy_ready": True,
            "no_weight_mutation": True,
            "rollback_recovery_rate": 1.0,
            "cumulative_constraint_violations": 0,
            "quality_delta_vs_naive_icl": 0.073,
            "context_efficiency_delta": 0.52,
            "inference_substrate": "deterministic_frozen_model_policy_no_weight_update",
        },
        "results/experiment_5461_gated_sota_csl_memory_routing_v496.json": {
            "honest_verdict": "complete: live SOTA GGUF governed CSL memory routing",
            "status": "complete",
            "milestone": "2026.07.496",
            "csl_sota_memory_routing_ready": True,
            "gpu_offload_verified": True,
            "no_weight_mutation": True,
            "negative_transfer_deflection_rate": 1.0,
            "quality_delta_vs_no_memory": 0.75,
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "runtime_receipt": {"offload_evidence": True},
            "precondition_details": {"all_passed": True, "blocked_preconditions": []},
            "inference_substrate": "live_llm_inference_with_frozen_policy_state",
        },
        "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json": {
            "honest_verdict": "complete: assumptions stayed advisory",
            "status": "complete",
            "milestone": "2026.07.496",
            "minimal_core_pbit_bridge_ready": True,
            "solver_authoritative": True,
            "fallback_completeness_rate": 1.0,
            "hardware_speedup_claim": False,
            "readiness_blockers": [],
            "claim_limits": ["advisory assumptions only"],
            "inference_substrate": "deterministic_solver_pbit_pdit_fixture",
        },
        "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json": {
            "honest_verdict": "complete: timing receipts; hardware_speedup_claim=false",
            "milestone": "2026.07.496",
            "hardware_receipts_ready": True,
            "gated_upstream_ready": True,
            "hashes_match_before_timing_compare": True,
            "hardware_speedup_claim": False,
            "board_reachability": {
                "kv260": {"reachable": False, "blocked_reason": "blocked_kv260_ssh"},
                "polarfire": {"reachable": True, "blocked_reason": None},
            },
            "timing_repeat_counts": {"cpu": 10, "kv260": 0, "polarfire": 10},
            "timing_comparison": {"comparison_performed": True, "hardware_speedup_claim": False},
            "preconditions_checked": True,
            "inference_substrate": "cpu_and_reachable_board_timing_receipts",
        },
        "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json": {
            "honest_verdict": "complete: ARC precheck ready; no solve claimed",
            "status": "complete",
            "milestone": "2026.07.496",
            "arc_metric_integrity_ready": True,
            "registry_precheck_performed": True,
            "duplicate_solve_rejected": True,
            "off_path_solve_rejected": True,
            "target_shortlist": [{"game": "bp35", "target_level": 3}],
            "inference_substrate": "live_path_precheck_no_solve_claim",
        },
        "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json": {
            "honest_verdict": "honest_null: bp35 L3 bounded_budget_no_levelup",
            "status": "honest_null",
            "milestone": "2026.07.496",
            "new_level_banked": False,
            "offline_reproduced": False,
            "registry_precheck_performed": True,
            "failure_mode": "bounded_budget_no_levelup",
            "live_attempt_count": 12,
            "solve_provenance": "live_agent_self_discovery",
            "source_reading_used": False,
            "inference_substrate": "arc_live_agent_self_discovery",
        },
    }


def _populate_upstreams(root: Path, payloads: dict[str, dict[str, Any]]) -> None:
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)
    _write_json(
        root,
        "results/experiment_5456_guided_decoding_tautology_corrigendum_v496_metric_dependency_graph.json",
        {"schema": "fixture", "invalid_tautological_fields": ["abstention_rate"]},
    )
    _write_jsonl(
        root,
        "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496_rows.jsonl",
        [{"row_id": "5457-row", "gpu_offload_evidence": True}],
    )
    _write_jsonl(
        root,
        "results/experiment_5461_gated_sota_csl_memory_routing_v496_rows.jsonl",
        [{"row_id": "5461-row", "gpu_offload_evidence": True}],
    )
    _write_json(
        root,
        "results/experiment_5464_arc_perception_feature_receipts_v496.json",
        {"connected_component_rows": [{"id": "component"}]},
    )


def test_req_report_5466_spec_declares_required_artifact_fields() -> None:
    """REQ-REPORT-5466: OpenSpec anchors the Exp5466 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5466") :]

    assert "SCENARIO-REPORT-5466" in section
    assert "SCENARIO-REPORT-5466-MISSING-OR-SKIPPED" in section
    assert str(exp5466.OUTPUT_REL_PATH) in section
    assert "FR-11" in section
    assert "FR-12" in section
    for field in exp5466.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for key in exp5466.FAILURE_MODE_KEYS:
        assert key.replace("_", "/") in section or key.replace("_", " ") in section


def test_scenario_report_5466_builds_complete_table_from_actual_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5466: complete upstreams produce separated PRD lanes."""

    _populate_upstreams(tmp_path, _minimal_upstreams())

    report = exp5466.build_report(tmp_path, tests_run=["unit 5466"])

    assert report["milestone"] == "2026.07.496"
    assert report["artifact_paths_read"] == sorted(report["artifact_paths_read"])
    assert report["missing_artifacts"] == []
    assert report["skipped_gated_tasks"] == []
    assert report["docs_updated"] == []
    assert report["inference_substrate"] == exp5466.INFERENCE_SUBSTRATE
    assert report["tests_run"] == ["unit 5466"]
    assert set(exp5466.MAIN_ARTIFACT_PATHS).issubset(set(report["artifact_paths_read"]))
    assert "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496_rows.jsonl" in (
        report["artifact_paths_read"]
    )

    assert _lane_names(report["closed_lanes"]) == {
        "transition_traceability",
        "source_delta_refresh",
        "guided_decoding_corrigendum",
        "minimal_core_claim_repair",
        "constraint_distortion_guard",
        "csl_policy_bandit",
        "sota_csl_memory_routing",
        "arc_metric_integrity_precheck",
    }
    assert _lane_names(report["partial_lanes"]) == {
        "active_constraint_pbit_pdit_bridge",
        "hardware_boundary_exchange_receipts",
    }
    assert _lane_names(report["blocked_lanes"]) == {
        "local_sota_distortion_guarded_decoding",
        "token_internal_hidden_claims",
    }
    assert _lane_names(report["honest_null_lanes"]) == {
        "arc_connected_component_salience_levelup",
        "hardware_speedup_claim",
    }

    prd_map = report["prd_requirement_map"]
    assert prd_map["FR-11"]["classification"] == "closed"
    assert "results/experiment_5460_csl_policy_bandit_v496.json" in prd_map["FR-11"][
        "evidence_artifacts"
    ]
    assert prd_map["FR-12"]["classification"] == "partial"
    assert "local_sota_distortion_guarded_decoding" in prd_map["FR-12"]["blocked_or_partial_lanes"]

    assert _failure(report, "tautology")["observed"] is True
    assert _failure(report, "missing_hardware")["observed"] is True
    assert _failure(report, "no_bank_arc")["observed"] is True
    assert _failure(report, "gguf_offload_gaps")["observed"] is False
    assert _failure(report, "unsupported_hidden_internal_claims")["observed"] is True
    exp5466.validate_artifact(report)


def test_scenario_report_5466_missing_and_skipped_inputs_fail_honest(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5466-MISSING-OR-SKIPPED: absent and skipped work stays visible."""

    payloads = _minimal_upstreams()
    payloads["results/experiment_5458_minimal_core_claim_repair_v496.json"] = {
        "honest_verdict": "skipped_gated: upstream core gate stayed closed",
        "status": "skipped_gated",
        "skipped_reason": "upstream_core_gate_closed",
        "milestone": "2026.07.496",
    }
    payloads.pop("results/experiment_5459_constraint_distortion_guard_v496.json")
    _populate_upstreams(tmp_path, payloads)

    report = exp5466.build_report(tmp_path)

    assert report["missing_artifacts"] == [
        "results/experiment_5459_constraint_distortion_guard_v496.json"
    ]
    assert report["skipped_gated_tasks"] == [
        {
            "artifact_path": "results/experiment_5458_minimal_core_claim_repair_v496.json",
            "honest_verdict": "skipped_gated: upstream core gate stayed closed",
            "reason": "upstream_core_gate_closed",
            "status": "skipped_gated",
        }
    ]
    assert report["missing_lanes"] == [
        {
            "lane": "constraint_distortion_guard",
            "missing_artifacts": [
                "results/experiment_5459_constraint_distortion_guard_v496.json"
            ],
        }
    ]
    assert report["honest_verdict"].startswith("blocked:")
    exp5466.validate_artifact(report)


def test_req_report_5466_write_artifact_uses_required_result_path(tmp_path: Path) -> None:
    """REQ-REPORT-5466: writer emits the required Exp5466 JSON path."""

    _populate_upstreams(tmp_path, _minimal_upstreams())

    output = exp5466.write_artifact(tmp_path, tests_run=["unit 5466"])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / exp5466.OUTPUT_REL_PATH
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["tests_run"] == ["unit 5466"]
    exp5466.validate_artifact(payload)


def test_req_report_5466_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-REPORT-5466: checked-in deliverable is stable under replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp5466.build_report(REPO)

    assert checked_in == replay
    exp5466.validate_artifact(checked_in)


def test_req_report_5466_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5466: validation catches missing fields and unsupported doc drift."""

    _populate_upstreams(tmp_path, _minimal_upstreams())
    report = exp5466.build_report(tmp_path)
    exp5466.validate_artifact(report)

    for key, value, error in (
        ("milestone", "2026.07.495", "milestone"),
        ("docs_updated", ["ops/status.md"], "docs_updated"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("honest_verdict", "done", "honest_verdict"),
    ):
        bad = dict(report)
        bad[key] = value
        with pytest.raises(ValueError, match=error):
            exp5466.validate_artifact(bad)

    missing_field = dict(report)
    missing_field.pop("artifact_paths_read")
    with pytest.raises(ValueError, match="artifact_paths_read"):
        exp5466.validate_artifact(missing_field)

    missing_failure = dict(report)
    missing_failure["agent_failure_taxonomy"] = {
        key: value
        for key, value in report["agent_failure_taxonomy"].items()
        if key != "tautology"
    }
    with pytest.raises(ValueError, match="tautology"):
        exp5466.validate_artifact(missing_failure)

    bad_taxonomy = dict(report)
    bad_taxonomy["agent_failure_taxonomy"] = []
    with pytest.raises(ValueError, match="agent_failure_taxonomy"):
        exp5466.validate_artifact(bad_taxonomy)

    bad_list = dict(report)
    bad_list["closed_lanes"] = {}
    with pytest.raises(ValueError, match="closed_lanes"):
        exp5466.validate_artifact(bad_list)

    bad_missing_verdict = dict(report)
    bad_missing_verdict["missing_artifacts"] = ["results/missing.json"]
    bad_missing_verdict["honest_verdict"] = "complete: contradictory"
    with pytest.raises(ValueError, match="blocked"):
        exp5466.validate_artifact(bad_missing_verdict)


def test_req_report_5466_records_failure_variant_branches(tmp_path: Path) -> None:
    """REQ-REPORT-5466: defensive branches remain explicit and test-covered."""

    payloads = _minimal_upstreams()
    payloads["results/experiment_5454_transition_v496.json"]["blocked_lanes"] = []
    payloads["results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json"][
        "precondition_details"
    ] = {"all_passed": False, "blocked_preconditions": ["model_missing"]}
    payloads["results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json"][
        "runtime_receipt"
    ] = {"offload_evidence": False}
    _populate_upstreams(tmp_path, payloads)

    report = exp5466.build_report(tmp_path)

    assert {
        "lane": "token_internal_hidden_claims",
        "missing_artifacts": ["results/experiment_5454_transition_v496.json"],
    } in report["missing_lanes"]
    assert _failure(report, "duration_precondition_failures")["observed"] is True
    assert _failure(report, "gguf_offload_gaps")["observed"] is True

    payloads["results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json"][
        "gpu_offload_verified"
    ] = False
    for rel_path, payload in payloads.items():
        _write_json(tmp_path, rel_path, payload)
    second = exp5466.build_report(tmp_path)

    assert _failure(second, "gguf_offload_gaps")["observed"] is True


def test_req_report_5466_rejects_non_object_main_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5466: malformed upstream JSON cannot become evidence."""

    path = tmp_path / exp5466.MAIN_ARTIFACT_PATHS[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp5466.build_report(tmp_path)
