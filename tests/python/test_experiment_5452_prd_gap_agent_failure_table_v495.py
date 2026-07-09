"""Tests for the Exp5452 V495 PRD gap and agent-failure table.

Spec refs: REQ-HARNESS-015, SCENARIO-HARNESS-010,
SCENARIO-HARNESS-011.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import prd_gap_agent_failure_table_v495_5452 as exp5452


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"
RESULT_PATH = REPO / exp5452.OUTPUT_REL_PATH


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _lane_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def _pattern(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row["pattern"] == name)


def _minimal_upstreams() -> dict[str, dict[str, Any]]:
    return {
        "results/experiment_5441_transition_v495.json": {
            "honest_verdict": "complete: transition receipt",
            "closed_lanes": [{"lane": "fixture_closed"}],
            "partial_lanes": [{"lane": "fixture_partial"}],
            "blocked_lanes": [{"lane": "token_internal_feature_lane_closed"}],
            "honest_null_lanes": [{"lane": "hardware_speedup_claim"}],
            "next_task_range": "exp5441-exp5453",
        },
        "results/experiment_5442_source_delta_v495.json": {
            "honest_verdict": "complete: 3 new actionable deltas appended",
            "sources_checked": ["arxiv", "openreview"],
            "new_actionable_findings_count": 3,
            "new_references_added": [{"title": "fixture"}],
            "retired_scopes_reopened": False,
            "research_references_updated": True,
        },
        "results/experiment_5443_verifier_potential_prefix_fixture_v495.json": {
            "honest_verdict": "complete: verifier fixture ready",
            "verifier_potential_fixture_ready": True,
            "exact_final_authority": True,
            "prefix_final_disagreement_cases": 6,
            "metric_independence_checks_passed": True,
        },
        "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json": {
            "honest_verdict": "complete: decoding pilot ready",
            "verifier_guided_decoding_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            "guided_validity_delta_vs_grammar_only": -0.5,
            "guided_validity_delta_vs_unconstrained": 0.25,
            "gpu_offload_verified": True,
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "precondition_details": {"all_passed": True, "blocked_preconditions": []},
            "model_specs": [{"local_path_available": True}],
        },
        "results/experiment_5445_static_ast_kb_witness_constraints_v495.json": {
            "honest_verdict": "complete: deterministic witnesses ready",
            "ast_kb_witness_ready": True,
            "ast_parse_success_rate": 1.0,
            "valid_call_accept_rate": 1.0,
            "nonexistent_call_reject_rate": 1.0,
            "unsafe_false_accepts": 0,
        },
        "results/experiment_5446_governed_memory_csl_online_v495.json": {
            "honest_verdict": "complete: governed memory ready",
            "governed_csl_loop_ready": True,
            "replay_success_rate": 1.0,
            "unsafe_false_accepts": 0,
            "no_weight_mutation": True,
            "rollback_recovery_rate": 1.0,
            "negative_transfer_deflection_rate": 1.0,
            "quality_delta_vs_always_full": 0.0,
            "context_efficiency_delta": 0.2,
        },
        "results/experiment_5447_gated_csl_memory_failure_stress_v495.json": {
            "honest_verdict": "complete: memory stress ready",
            "csl_memory_stress_ready": True,
            "gated_upstream_ready": True,
            "unsafe_false_accepts": 0,
            "no_weight_mutation": True,
            "rollback_recovery_rate": 1.0,
            "stale_memory_deflection_rate": 1.0,
            "poisoned_memory_deflection_rate": 1.0,
            "retrieval_collision_deflection_rate": 1.0,
            "negative_transfer_deflection_rate": 1.0,
        },
        "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json": {
            "honest_verdict": "complete: assumptions stayed advisory",
            "pbit_assumption_bridge_ready": True,
            "solver_authoritative": True,
            "fallback_completeness_rate": 1.0,
            "hardware_speedup_claim": False,
            "claim_limits": ["advisory assumptions only"],
        },
        "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json": {
            "honest_verdict": "complete: timing receipts ready",
            "hardware_receipts_ready": True,
            "gated_upstream_ready": True,
            "hashes_match_before_timing_compare": True,
            "hardware_speedup_claim": False,
            "board_reachability": {
                "kv260": {"reachable": False, "blocked_reason": "blocked_kv260_ssh"},
                "gatemate": {
                    "reachable": False,
                    "blocked_reason": "blocked_gatemate_diagnostic_only",
                },
                "polarfire": {"reachable": True, "blocked_reason": None},
            },
            "timing_repeat_counts": {"cpu": 10, "polarfire": 10, "kv260": 0, "gatemate": 0},
            "timing_summary": {"cpu": {"mean_s": 1.0}, "polarfire": {"mean_s": 3.0}},
            "timing_comparison": {"comparison_performed": True, "hardware_speedup_claim": False},
        },
        "results/experiment_5450_arc_measurement_access_live_levelup_v495.json": {
            "honest_verdict": "honest_null: bounded_budget_no_levelup",
            "status": "honest_null",
            "arc_new_level_banked": False,
            "new_levels_banked": 0,
            "new_level_reproduced": False,
            "reproduction_gate": {"reproduced": False},
            "residual_wall": "bounded_budget_no_levelup",
            "live_attempt_count": 39,
        },
        "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json": {
            "honest_verdict": "complete: bounded certificate ready",
            "kan_certificate_ready": True,
            "gated_upstreams_ready": True,
            "claim_count": 4,
            "true_measured_claim_preservation_rate": 1.0,
            "false_property_rejection_rate": 1.0,
            "unsupported_claim_rejection_rate": 1.0,
            "hardware_speedup_claim_rejected": True,
            "token_internal_claim_rejected": True,
            "broad_kan_claim_made": False,
            "claim_limits": ["no broad KAN soundness claim"],
            "claim_records": [
                {
                    "claim_id": "unsupported_hardware_speedup_from_certificate",
                    "claim_kind": "unsupported",
                    "statement": "The certificate proves hardware speedup.",
                    "rejected": True,
                    "missing_evidence": ["board_timing_receipt"],
                },
                {
                    "claim_id": "unsupported_token_level_access_from_certificate",
                    "claim_kind": "unsupported",
                    "statement": "The certificate proves token access.",
                    "rejected": True,
                    "missing_evidence": ["token_logprobs"],
                },
                {
                    "claim_id": "unsupported_internal_state_access_from_certificate",
                    "claim_kind": "unsupported",
                    "statement": "The certificate proves hidden-state access.",
                    "rejected": True,
                    "missing_evidence": ["hidden_state_tensor"],
                },
                {
                    "claim_id": "unsupported_broad_kan_soundness",
                    "claim_kind": "broad_soundness",
                    "statement": "The certificate proves broad KAN soundness.",
                    "rejected": True,
                    "missing_evidence": ["general_kan_soundness_theorem"],
                },
            ],
        },
    }


def test_req_harness_015_spec_declares_exp5452_required_artifact_fields() -> None:
    """REQ-HARNESS-015: OpenSpec names the Exp5452 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    normalized = " ".join(spec.split())

    assert "SCENARIO-HARNESS-011" in spec
    assert "experiment_5452_prd_gap_agent_failure_table_v495.json" in spec
    for field in exp5452.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in spec
    for goal in exp5452.PRD_GOALS:
        assert goal in normalized


def test_scenario_harness_011_classifies_v495_lanes_from_existing_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-HARNESS-011: V495 table preserves closed, bounded, and null lanes."""

    payloads = _minimal_upstreams()
    for rel_path, payload in payloads.items():
        _write_json(tmp_path, rel_path, payload)

    report = exp5452.build_report(tmp_path)

    assert report["milestone"] == "2026.07.495"
    assert report["artifacts_expected"] == list(exp5452.DEFAULT_UPSTREAM_PATHS)
    assert report["artifacts_found"] == list(exp5452.DEFAULT_UPSTREAM_PATHS)
    assert report["closed_count"] == 6
    assert report["partial_count"] == 4
    assert report["blocked_count"] == 1
    assert report["honest_null_count"] == 2
    assert report["missing_count"] == 0
    assert _lane_names(report["prd_gap_table"]) == {
        "transition_traceability",
        "source_delta_traceability",
        "verifier_potential_reasoning",
        "local_sota_decoding_pilot",
        "ast_kb_witness_constraints",
        "governed_online_memory_csl",
        "csl_memory_failure_stress",
        "pbit_assumption_bridge",
        "hardware_timing_receipts",
        "arc_live_progress",
        "kan_measurement_certificate",
        "token_internal_feature_access",
        "hardware_speedup_claim",
    }
    assert report["inference_substrate"] == exp5452.INFERENCE_SUBSTRATE
    assert report["honest_verdict"].startswith("complete:")
    exp5452.validate_artifact(report)

    for row in report["prd_gap_table"]:
        assert row["classification"] in exp5452.CLASSIFICATIONS
        assert row["prd_goals"]
        assert row["evidence_citation"]
        for field in row["supporting_fields"]:
            artifact_payload = payloads[field["artifact_path"]]
            assert field["present"] is True
            assert field["value"] == artifact_payload[field["field_name"]]


def test_scenario_harness_010_missing_artifact_is_classified_missing(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-010: missing expected artifacts are not fabricated."""

    missing_path = "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"
    for rel_path, payload in _minimal_upstreams().items():
        if rel_path != missing_path:
            _write_json(tmp_path, rel_path, payload)

    report = exp5452.build_report(tmp_path)

    assert report["artifacts_missing"] == [missing_path]
    assert report["artifacts_found"] == [
        path for path in exp5452.DEFAULT_UPSTREAM_PATHS if path != missing_path
    ]
    assert report["missing_count"] == 1
    assert report["honest_verdict"].startswith("blocked:")
    missing_rows = [row for row in report["prd_gap_table"] if row["classification"] == "missing"]
    assert _lane_names(missing_rows) == {"local_sota_decoding_pilot"}
    assert missing_rows[0]["missing_artifacts"] == [missing_path]
    exp5452.validate_artifact(report)


def test_req_harness_015_agent_failure_patterns_and_unsupported_claims(
    tmp_path: Path,
) -> None:
    """REQ-HARNESS-015: failure patterns and unsupported claims cite artifacts."""

    for rel_path, payload in _minimal_upstreams().items():
        _write_json(tmp_path, rel_path, payload)

    report = exp5452.build_report(tmp_path)

    patterns = report["agent_failure_table"]
    assert {row["pattern"] for row in patterns} == set(exp5452.AGENT_FAILURE_PATTERNS)
    assert _pattern(patterns, "precondition block")["observed"] is False
    assert _pattern(patterns, "gate block")["observed"] is False
    assert _pattern(patterns, "implementation failure")["observed"] is False
    assert _pattern(patterns, "no-bank")["classification"] == "honest_null"
    assert _pattern(patterns, "measurement unavailable")["observed"] is True
    assert _pattern(patterns, "tautology risk")["observed"] is True
    assert _pattern(patterns, "unsupported claim")["observed"] is True

    unsupported_ids = {row["claim_id"] for row in report["unsupported_claims_detected"]}
    assert unsupported_ids == {
        "unsupported_hardware_speedup_from_certificate",
        "unsupported_token_level_access_from_certificate",
        "unsupported_internal_state_access_from_certificate",
        "unsupported_broad_kan_soundness",
    }
    assert all(row["rejected"] is True for row in report["unsupported_claims_detected"])


def test_req_harness_015_write_artifact_uses_required_result_path(tmp_path: Path) -> None:
    """REQ-HARNESS-015: writer emits the required Exp5452 artifact path."""

    for rel_path, payload in _minimal_upstreams().items():
        _write_json(tmp_path, rel_path, payload)

    output = exp5452.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / exp5452.OUTPUT_REL_PATH
    assert payload["honest_verdict"].startswith("complete:")
    exp5452.validate_artifact(payload)


def test_req_harness_015_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-HARNESS-015: checked-in deliverable is stable under replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp5452.build_report(REPO)

    assert checked_in == replay
    exp5452.validate_artifact(checked_in)


def test_req_harness_015_rejects_non_object_upstream_json(tmp_path: Path) -> None:
    """REQ-HARNESS-015: malformed upstream JSON cannot be cited as evidence."""

    path = tmp_path / exp5452.DEFAULT_UPSTREAM_PATHS[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp5452.build_report(tmp_path)
