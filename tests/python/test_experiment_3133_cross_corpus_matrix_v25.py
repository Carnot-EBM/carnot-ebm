"""Tests for Exp 3133 cross-corpus matrix v25.

Spec refs: REQ-REPORT-3133, SCENARIO-REPORT-3133.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v25_3133 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "matrix_v25_ready",
    "rows_total",
    "prior_publication_blocker_count",
    "publication_blocker_count",
    "blocker_delta_from_v24",
    "missing_artifacts",
    "status_counts",
    "headline_claim_allowance_summary",
    "verifier_repair_summary",
    "fr11_summary",
    "architecture_boundary_summary",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str, claim_scope: str = "carry") -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": "v24_carry",
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"source_status": status},
        "row_origin": "matrix_v24_test",
    }


def _matrix_v24(*, ready: bool = True) -> dict[str, Any]:
    rows = [
        _row("carry:clean", "clean"),
        _row("carry:bounded", "bounded", "baseline_bounded"),
        _row("carry:gated", "gated_skipped", "baseline_gate"),
        _row("carry:diagnostic", "diagnostic_only", "baseline_diagnostic"),
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
        "artifact": "experiment_3120_cross_corpus_matrix_v24",
        "matrix_v24_ready": ready,
        "rows_total": len(rows),
        "rows": rows,
        "status_counts": {
            status: sum(row["status"] == status for row in rows) for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "honest_verdict": "complete: matrix_v24_ready=true",
    }


def _capstone_v290(*, ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3121_capstone_v290",
        "capstone_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 36,
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_required_baseline(
    root: Path, *, matrix_ready: bool = True, capstone_ready: bool = True
) -> None:
    _write_json(root, mod.MATRIX_V24_REL_PATH, _matrix_v24(ready=matrix_ready))
    _write_json(root, mod.CAPSTONE_V290_REL_PATH, _capstone_v290(ready=capstone_ready))


def _write_dot291_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3122_REL_PATH,
        {
            "archive_v290_activate_v291_ready": True,
            "prior_publication_blocker_count": 36,
            "honest_verdict": "complete: archive_v290_activate_v291_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "sota_cache_manifest_v2_ready": True,
            "cached_sota_pair_available": False,
            "headline_claim_allowed": True,
            "present_model_ids": [GEMMA26],
            "missing_model_ids": [QWEN, GEMMA31],
            "selected_headline_model_ids": [GEMMA26],
            "honest_verdict": "complete: cache manifest ready; pair unavailable",
        },
    )
    _write_json(
        root,
        mod.EXP3124_REL_PATH,
        {
            "difficulty_stratified_live_sota_panel_v6_ready": True,
            "headline_claim_allowed": False,
            "repair_gate_state": "blocked_false_accept",
            "verifier_gain_delta": 0.0,
            "false_accept_rate": 0.5,
            "false_reject_rate": 0.0,
            "live_call_count": 6,
            "selected_model_ids": [GEMMA26],
            "honest_verdict": "complete_blocked_false_accept: false_accept_rate=0.5",
        },
    )
    _write_json(
        root,
        mod.EXP3125_REL_PATH,
        {
            "prefix_closed_bound_pilot_ready": True,
            "lower_bound": 0.004,
            "upper_bound": 0.006,
            "bound_width": 0.002,
            "explored_prefix_count": 453,
            "accepted_prefix_count": 2,
            "limitations": ["finite fixture-conditioned token prior"],
            "honest_verdict": "complete: bounded prefix pilot ready",
        },
    )
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "fragment_time_monitor_v1_ready": True,
            "monitor_event_count": 360,
            "monitor_violation_count": 2,
            "contradiction_count": 2,
            "satisfiable_drift_count": 0,
            "ledger_consistency_rate": 0.666667,
            "honest_verdict": "complete: fragment monitor ready",
        },
    )
    _write_json(
        root,
        mod.EXP3127_REL_PATH,
        {
            "experiment": 3127,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "artifact_field": "repair_gate_state",
                    "expected": "unblocked",
                    "actual": "blocked_false_accept",
                    "passed": False,
                }
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3128_REL_PATH,
        {
            "fr11_evoenv_pilot_v1_ready": True,
            "continuous_self_learning_targeted": True,
            "live_model_environment_synthesis": False,
            "admitted_environment_count": 3,
            "candidate_environment_count": 5,
            "no_weight_update_claim": True,
            "soundness_errors": 0,
            "completeness_errors": 0,
            "retention_delta": 0.0,
            "honest_verdict": "complete: solver-only environment admission",
        },
    )
    _write_json(
        root,
        mod.EXP3129_REL_PATH,
        {
            "fr11_constraint_memory_audit_v1_ready": True,
            "admitted_environment_count": 3,
            "no_weight_update_claim": True,
            "promotion_recommendation": (
                "promote_controller_environment_memory_only_block_model_weight_learning"
            ),
            "soundness_errors": 0,
            "completeness_errors": 0,
            "forgetting_regression_count": 0,
            "satisfiable_drift_count": 0,
            "ledger_consistency_rate": 0.666667,
            "honest_verdict": "complete: controller memory only",
        },
    )
    _write_json(
        root,
        mod.EXP3130_REL_PATH,
        {
            "arm_ebt_energy_budget_sidecar_v2_ready": True,
            "live_integration": False,
            "live_call_count": 6,
            "integration_blockers": ["no generation-path sidecar hook"],
            "correlation_metrics": {"sidecar_energy": {"pearson_reject_or_repair": 0.8795236244}},
            "honest_verdict": "complete: sidecar diagnostic, no live integration",
        },
    )
    _write_json(
        root,
        mod.EXP3131_REL_PATH,
        {
            "kan_pwa_milp_audit_v1_ready": True,
            "kan_code_present": True,
            "abstraction_count": 2,
            "milp_property_check_count": 1,
            "milp_property_pass_count": 1,
            "claim_boundary": {
                "proves": "bounded two-unit PWA abstraction accounting",
                "does_not_prove": ["deployed verifier improvement"],
            },
            "implementation_blockers": [],
            "honest_verdict": "complete_kan_pwa_milp_abstraction_audit_v1",
        },
    )
    _write_json(
        root,
        mod.EXP3132_REL_PATH,
        {
            "hardware_evidence_sampler_boundary_v5_ready": True,
            "gatemate_evidence_complete": False,
            "ssqa_readback_ready": False,
            "speedup_claim_allowed": False,
            "hardware_commands_run": [],
            "missing_operator_evidence": [{"missing_item": "ssqa:host_visible_smoke_evidence"}],
            "sampler_boundary_decisions": {
                "clut": "CPU simulation",
                "gatemate": "blocked",
                "ssqa": "blocked",
            },
            "honest_verdict": "complete: hardware boundary ready; blocked evidence remains",
        },
    )


def test_req_report_3133_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3133: OpenSpec declares the v25 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3133" in spec
    assert "SCENARIO-REPORT-3133" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3133_builds_v25_from_dot291_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3133: .291 rows stay bounded or blocked when evidence says so."""

    _write_required_baseline(tmp_path)
    _write_dot291_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.5)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    blockers = {row["row_id"] for row in artifact["publication_blockers"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v25_ready"] is True
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["rows_total"] == len(artifact["rows"]) == 15
    assert artifact["prior_publication_blocker_count"] == 2
    assert artifact["publication_blocker_count"] == 12
    assert artifact["blocker_delta_from_v24"] == 10
    assert artifact["status_counts"] == {
        "clean": 2,
        "flagged": 0,
        "bounded": 7,
        "blocked": 3,
        "gated_skipped": 1,
        "missing": 0,
        "retired": 0,
        "projection_only": 1,
        "diagnostic_only": 1,
        "model_spec_gap": 0,
    }
    assert artifact["missing_artifacts"] == []
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["dot291:exp3122_archive_handoff"]["status"] == "clean"
    assert rows["dot291:exp3123_sota_cache_coverage"]["status"] == "bounded"
    assert rows["dot291:exp3124_live_verifier_lift"]["status"] == "blocked"
    assert rows["dot291:exp3125_prefix_bounds"]["status"] == "bounded"
    assert rows["dot291:exp3126_fragment_time_monitors"]["status"] == "bounded"
    assert rows["dot291:exp3127_repair_ladder"]["status"] == "blocked"
    assert rows["dot291:exp3128_fr11_evoenv"]["status"] == "bounded"
    assert rows["dot291:exp3129_fr11_memory"]["status"] == "bounded"
    assert rows["dot291:exp3130_arm_ebt_energy_budget"]["status"] == "projection_only"
    assert rows["dot291:exp3131_kan_pwa_milp"]["status"] == "bounded"
    assert rows["dot291:exp3132_hardware_sampler_boundary"]["status"] == "blocked"
    assert "carry:diagnostic" not in blockers

    allowance = artifact["headline_claim_allowance_summary"]
    assert allowance["sota_cache_headline_allowed"] is True
    assert allowance["live_verifier_headline_allowed"] is False
    assert allowance["cached_sota_pair_available"] is False
    assert allowance["missing_model_ids"] == [QWEN, GEMMA31]
    assert "live_verifier_lift" in allowance["blocked_headline_claims"]

    verifier = artifact["verifier_repair_summary"]
    assert verifier["live_verifier_status"] == "blocked"
    assert verifier["repair_ladder_status"] == "blocked"
    assert verifier["repair_gate_state"] == "blocked_false_accept"
    assert verifier["false_accept_rate"] == pytest.approx(0.5)
    assert verifier["repair_ladder_blocked_at_layer"] == "conductor_pre_gate"

    fr11 = artifact["fr11_summary"]
    assert fr11["evoenv_status"] == "bounded"
    assert fr11["memory_status"] == "bounded"
    assert fr11["no_weight_update_claim"] is True
    assert fr11["model_weight_learning_allowed"] is False
    assert fr11["ledger_consistency_rate"] == pytest.approx(0.666667)

    architecture = artifact["architecture_boundary_summary"]
    assert architecture["arm_ebt_status"] == "projection_only"
    assert architecture["kan_pwa_milp_status"] == "bounded"
    assert architecture["hardware_sampler_status"] == "blocked"
    assert architecture["speedup_claim_allowed"] is False
    assert architecture["live_integration"] is False
    assert architecture["hardware_commands_run"] == []
    assert set(artifact["architecture_boundary_rows"]) == {
        "dot291:exp3130_arm_ebt_energy_budget",
        "dot291:exp3131_kan_pwa_milp",
        "dot291:exp3132_hardware_sampler_boundary",
    }
    assert artifact["diagnostic_only_rows"] == ["carry:diagnostic"]
    assert {row["row_id"] for row in artifact["gated_skips"]} == {
        "carry:gated",
        "dot291:exp3127_repair_ladder",
    }
    assert sources[mod.EXP3123_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3123_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_dot291_artifacts",
        "source": "matrix_v24_capstone_v290_and_dot291_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3133_missing_optional_artifacts_are_rows_not_successes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3133: absent `.291` evidence stays visible without blocking the matrix."""

    _write_required_baseline(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    dot291_rows = [row for row in artifact["rows"] if row["row_id"].startswith("dot291:")]

    assert artifact["matrix_v25_ready"] is True
    assert len(dot291_rows) == 11
    assert {row["status"] for row in dot291_rows} == {"missing"}
    assert artifact["publication_blocker_count"] == 13
    assert artifact["blocker_delta_from_v24"] == 11
    assert len(artifact["missing_artifacts"]) == 11
    assert all(
        row["reason"] == "missing_or_malformed_dot291_artifact"
        for row in artifact["missing_artifacts"]
    )

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["matrix_v25_ready"] is False
    assert empty["honest_verdict"].startswith("blocked_matrix_v25_preconditions")
    assert [row["path"] for row in empty["required_source_errors"]] == [
        mod.MATRIX_V24_REL_PATH.as_posix(),
        mod.CAPSTONE_V290_REL_PATH.as_posix(),
    ]


def test_req_report_3133_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3133: helper behavior is deterministic and fail-closed."""

    _write_required_baseline(tmp_path)
    _write_dot291_sources(tmp_path)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=2.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v25_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("diagnostic") == "diagnostic_only"
    assert mod.normal_status("pilot_only") == "bounded"
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
    assert mod._ready_status(False, {}, "ready") == "missing"
    assert mod._ready_status(True, {"ready": True}, "ready") == "clean"
    assert mod._ready_status(True, {"ready": False}, "ready") == "blocked"
    assert mod._ready_status(True, {"status": "success"}, "ready") == "clean"
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean")]})[0]["row_id"] == "carry"

    assert mod._sota_cache_row({})["status"] == "missing"
    assert mod._sota_cache_row({"sota_cache_manifest_v2_ready": False})["status"] == "blocked"
    assert (
        mod._sota_cache_row(
            {"sota_cache_manifest_v2_ready": True, "headline_claim_allowed": False}
        )["status"]
        == "model_spec_gap"
    )
    assert (
        mod._sota_cache_row(
            {
                "sota_cache_manifest_v2_ready": True,
                "headline_claim_allowed": True,
                "cached_sota_pair_available": True,
                "selected_headline_model_ids": [GEMMA26],
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._live_verifier_row(
            {
                "difficulty_stratified_live_sota_panel_v6_ready": True,
                "headline_claim_allowed": True,
                "repair_gate_state": "unblocked",
                "false_accept_rate": 0.0,
                "verifier_gain_delta": 0.0,
            }
        )["status"]
        == "bounded"
    )
    assert mod._live_verifier_row({})["status"] == "missing"
    assert (
        mod._live_verifier_row({"difficulty_stratified_live_sota_panel_v6_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._live_verifier_row(
            {
                "difficulty_stratified_live_sota_panel_v6_ready": True,
                "headline_claim_allowed": True,
                "repair_gate_state": "unblocked",
                "false_accept_rate": 0.0,
                "verifier_gain_delta": 0.1,
            }
        )["status"]
        == "clean"
    )
    assert mod._fragment_time_row({"fragment_time_monitor_v1_ready": False})["status"] == "blocked"
    assert (
        mod._fragment_time_row(
            {"fragment_time_monitor_v1_ready": True, "ledger_consistency_rate": 1.0}
        )["status"]
        == "clean"
    )
    assert mod._repair_ladder_row({"status": "success"})["status"] == "clean"
    assert mod._repair_ladder_row({"status": "bounded"})["status"] == "bounded"
    assert mod._fr11_evoenv_row({"fr11_evoenv_pilot_v1_ready": False})["status"] == "blocked"
    assert (
        mod._fr11_evoenv_row({"fr11_evoenv_pilot_v1_ready": True, "retention_delta": 0.25})[
            "status"
        ]
        == "clean"
    )
    assert (
        mod._fr11_memory_row({"fr11_constraint_memory_audit_v1_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._fr11_memory_row(
            {
                "fr11_constraint_memory_audit_v1_ready": True,
                "ledger_consistency_rate": 1.0,
                "no_weight_update_claim": False,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._arm_ebt_row({"arm_ebt_energy_budget_sidecar_v2_ready": False})["status"] == "blocked"
    )
    assert (
        mod._arm_ebt_row(
            {"arm_ebt_energy_budget_sidecar_v2_ready": True, "live_integration": True}
        )["status"]
        == "clean"
    )
    assert mod._kan_row({"kan_pwa_milp_audit_v1_ready": False})["status"] == "blocked"
    assert (
        mod._kan_row({"kan_pwa_milp_audit_v1_ready": True, "implementation_blockers": ["x"]})[
            "status"
        ]
        == "blocked"
    )
    assert mod._kan_row({"kan_pwa_milp_audit_v1_ready": True})["status"] == "clean"
    assert (
        mod._hardware_row({"hardware_evidence_sampler_boundary_v5_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._hardware_row(
            {
                "hardware_evidence_sampler_boundary_v5_ready": True,
                "gatemate_evidence_complete": True,
                "ssqa_readback_ready": True,
                "speedup_claim_allowed": True,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._hardware_row(
            {
                "hardware_evidence_sampler_boundary_v5_ready": True,
                "gatemate_evidence_complete": True,
                "ssqa_readback_ready": True,
                "speedup_claim_allowed": False,
            }
        )["status"]
        == "bounded"
    )

    violations = mod._invariant_violations(
        {"matrix_v24_ready": False},
        {"capstone_ready": False},
        [_row("flagged", "flagged")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v24 authority is not ready",
        "capstone v290 authority is not ready",
        "status_counts keys do not match required v25 statuses",
        "status_counts do not sum to rows_total",
    ]
    full_counts = {status: 0 for status in mod.STATUSES}
    full_counts["flagged"] = 1
    blocker_violation = mod._invariant_violations(
        {"matrix_v24_ready": True},
        {"capstone_ready": True},
        [_row("flagged", "flagged")],
        full_counts,
        [],
        [],
    )
    assert blocker_violation == ["publication_blocker_count does not match row statuses"]
