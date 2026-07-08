"""Tests for the Exp5439 PRD gap and failure-taxonomy table.

Spec refs: REQ-HARNESS-015, SCENARIO-HARNESS-010.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import prd_gap_agent_failure_table_v494_5439 as exp5439


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-harnesses" / "spec.md"


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _lane_names(lanes: list[dict[str, Any]]) -> set[str]:
    return {str(lane["lane"]) for lane in lanes}


def _minimal_upstreams() -> dict[str, dict[str, Any]]:
    return {
        "results/experiment_5428_transition_v494.json": {
            "honest_verdict": "complete: transition receipt",
            "blocked_lanes": [
                {
                    "lane": "token_internal_feature_lane_closed",
                    "terminal_evidence": {"backend_receipt_present": False},
                }
            ],
        },
        "results/experiment_5429_source_delta_v494.json": {
            "honest_verdict": "complete: source deltas appended",
            "new_actionable_findings_count": 3,
        },
        "results/experiment_5430_structured_tautology_corrigendum_v494.json": {
            "honest_verdict": "complete: corrigendum clean",
            "structured_corrigendum_clean": True,
            "adversarial_verify_clean": True,
            "row_count_recomputed": 21,
        },
        "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json": {
            "honest_verdict": "complete: structured taxonomy ready",
            "structured_taxonomy_replication_ready": True,
            "metric_independence_checks_passed": True,
            "accepted_risk_bound": 0.25884,
            "accepted_risk_bound_threshold": 0.35,
        },
        "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json": {
            "honest_verdict": "complete: ontology verified",
            "ontology_constraint_memory_ready": True,
            "valid_update_preservation_rate": 1.0,
            "false_triple_rejection_rate": 1.0,
            "unsupported_update_abstention_rate": 1.0,
            "soft_logic_overrode_solver": False,
        },
        "results/experiment_5433_active_constraint_diversity_lns_v494.json": {
            "honest_verdict": "complete: LNS ready",
            "active_constraint_diversity_ready": True,
            "solver_validity_preserved": True,
            "work_delta": 138,
            "claim_limits": ["hints are advisory"],
        },
        "results/experiment_5434_pbit_polarfire_timing_variance_v494.json": {
            "honest_verdict": "complete: timing receipts ready",
            "timing_variance_receipts_ready": True,
            "measurement_access_complete": True,
            "same_workload_hash_match": True,
            "same_result_hash_match": True,
            "hardware_speedup_claim": False,
        },
        "results/experiment_5435_verified_workflow_memory_csl_v494.json": {
            "honest_verdict": "complete: workflow memory ready",
            "verified_workflow_memory_ready": True,
            "verify_before_store_pass_rate": 0.5,
            "retrieval_trap_deflection_rate": 1.0,
            "rollback_verified": True,
            "no_weight_mutation": True,
        },
        "results/experiment_5436_csl_memory_transfer_stress_v494.json": {
            "honest_verdict": "complete: transfer stress ready",
            "csl_transfer_stress_ready": True,
            "in_domain_quality_delta": 0.08,
            "out_of_domain_quality_delta": 0.0,
            "negative_transfer_deflection_rate": 1.0,
            "rollback_verified": True,
            "no_weight_mutation": True,
        },
        "results/experiment_5437_arc_live_reinduction_levelup_v494.json": {
            "honest_verdict": "honest_null: bounded_budget_no_levelup",
            "status": "honest_null",
            "arc_new_level_banked": False,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "newly_reached_levels": [],
            "attempt_count": 51,
            "frontier_expansion_count": 22,
            "failure_mode": "bounded_budget_no_levelup",
        },
        "results/experiment_5438_kan_ontology_measurement_certificate_v494.json": {
            "honest_verdict": "complete: certificate ready",
            "kan_ontology_certificate_ready": True,
            "false_property_rejection_rate": 1.0,
            "true_property_preservation_rate": 1.0,
            "missing_evidence_detected": True,
            "broad_kan_verification_claim": False,
            "missing_evidence_controls": [
                {"unsupported_claim_type": "token_access", "rejected": True},
                {"unsupported_claim_type": "internal_state", "rejected": True},
            ],
            "claim_limits": ["token and internal claims require evidence absent here"],
        },
    }


def test_req_harness_015_spec_declares_exp5439_contract() -> None:
    """REQ-HARNESS-015: the OpenSpec contract names the Exp5439 table."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HARNESS-015" in spec
    assert "SCENARIO-HARNESS-010" in spec
    assert "experiment_5439_prd_gap_agent_failure_table_v494.json" in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_req_harness_015_classifies_v494_lanes_from_actual_fields(tmp_path: Path) -> None:
    """REQ-HARNESS-015: lane tables cite existing fields and taxonomy counts."""

    payloads = _minimal_upstreams()
    for rel_path, payload in payloads.items():
        _write_json(tmp_path, rel_path, payload)

    report = exp5439.build_report(tmp_path)

    assert report["upstream_artifacts_missing"] == []
    assert len(report["upstream_artifacts_read"]) == len(exp5439.DEFAULT_UPSTREAM_PATHS)
    assert _lane_names(report["closed_lanes"]) == {
        "structured_verification",
        "continuous_self_learning",
        "ontology_memory",
    }
    assert _lane_names(report["partial_lanes"]) == {
        "solver_guidance",
        "hardware",
        "certificates",
    }
    assert _lane_names(report["blocked_lanes"]) == {"token_internal_feature_access"}
    assert _lane_names(report["honest_null_lanes"]) == {
        "arc_live_progress",
        "hardware_speedup_claim",
    }
    assert report["missing_lanes"] == []
    assert report["prd_gap_table_ready"] is True
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["honest_verdict"].startswith("complete:")

    counts = report["failure_taxonomy_counts"]
    assert counts["structured-corrigendum"] == 1
    assert counts["workflow-memory"] == 2
    assert counts["solver/hardware"] == 3
    assert counts["ARC"] == 1
    assert counts["measurement-access"] == 4
    assert counts["token/internal"] == 1
    assert counts["tool-use"] == 1
    assert counts["planning"] == 3
    assert counts["reasoning"] == 4
    assert counts["calibration"] == 1
    assert counts["memory-transfer"] == 1
    assert counts["live-environment"] == 1
    assert counts["hardware"] == 2

    for lane_group in (
        "closed_lanes",
        "partial_lanes",
        "blocked_lanes",
        "honest_null_lanes",
    ):
        for lane in report[lane_group]:
            for field in lane["supporting_fields"]:
                artifact_payload = payloads[field["artifact_path"]]
                assert field["present"] is True
                assert field["field_name"] in artifact_payload
                assert field["value"] == artifact_payload[field["field_name"]]


def test_scenario_harness_010_missing_upstream_is_not_fabricated(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-010: absent upstreams become missing lanes."""

    payloads = _minimal_upstreams()
    missing_path = "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json"
    for rel_path, payload in payloads.items():
        if rel_path != missing_path:
            _write_json(tmp_path, rel_path, payload)

    report = exp5439.build_report(tmp_path)

    assert report["upstream_artifacts_missing"] == [missing_path]
    assert report["prd_gap_table_ready"] is False
    assert _lane_names(report["missing_lanes"]) == {"structured_verification"}
    assert _lane_names(report["closed_lanes"]) == {
        "continuous_self_learning",
        "ontology_memory",
    }

    missing_lane = report["missing_lanes"][0]
    assert missing_lane["classification"] == "missing"
    assert missing_lane["missing_artifacts"] == [missing_path]
    assert missing_lane["missing_supporting_fields"] == []
    assert all(
        field["artifact_path"] != missing_path
        for field in missing_lane["supporting_fields"]
    )
    assert report["honest_verdict"].startswith("blocked:")


def test_req_harness_015_write_artifact_uses_required_result_path(tmp_path: Path) -> None:
    """REQ-HARNESS-015: the writer emits the required Exp5439 artifact path."""

    for rel_path, payload in _minimal_upstreams().items():
        _write_json(tmp_path, rel_path, payload)

    output = exp5439.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / exp5439.OUTPUT_REL_PATH
    assert payload["prd_gap_table_ready"] is True
    assert payload["honest_verdict"].startswith("complete:")


def test_req_harness_015_rejects_non_object_upstream_json(tmp_path: Path) -> None:
    """REQ-HARNESS-015: malformed upstream JSON cannot be used as evidence."""

    path = tmp_path / exp5439.DEFAULT_UPSTREAM_PATHS[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp5439.build_report(tmp_path)
