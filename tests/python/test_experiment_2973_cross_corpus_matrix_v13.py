"""Tests for Exp 2973 cross-corpus matrix v13.

Spec refs: REQ-REPORT-2973, SCENARIO-REPORT-2973.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v13_2973 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "matrix_v13_ready",
    "inference_substrate",
    "upstream_artifacts_read",
    "upstream_checksums",
    "clean_rows",
    "flagged_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "pilot_only_rows",
    "forbidden_claims_absent",
    "repair_replication_summary",
    "solver_frontier_summary",
    "self_learning_summary",
    "kan_memory_summary",
    "hardware_state_summary",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        mod.MATRIX_V12_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v12_ready=true",
            "matrix_v12_ready": True,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "clean_rows": ["corpus:FoVer", "exp2953_threshold_policy"],
            "flagged_rows": ["exp2952_structured_repair_delta", "exp2959_nl_to_z3_execution_repair"],
            "blocked_rows": ["exp2957_gatemate_flash_smoke"],
            "gated_skipped_rows": [],
            "pilot_only_rows": ["corpus:MBPP"],
            "aggregation_only_rows": ["exp2943_matrix_v11_carry_forward"],
            "code_repair_delta_summary": {
                "n_tasks": 4,
                "pass_at_1_delta": 0.25,
                "pass_at_k_delta": 0.25,
                "syntax_failure_rate_delta": -0.0625,
                "false_accept_delta": -0.125,
                "artifact_flagged": True,
            },
            "solver_state_summary": {
                "n_items": 12,
                "parseability_rate": 0.083333,
                "solver_verified_accuracy": 0.0,
                "answer_accuracy": 0.083333,
                "artifact_flagged": True,
            },
            "self_learning_delta_summary": {
                "artifact_ready": True,
                "heldout_utility_after": 0.352886467009,
                "heldout_utility_delta": 0.111859239286,
                "forgetting_guard_passed": True,
                "artifact_flagged": True,
            },
            "hardware_state_summary": {
                "gatemate": {
                    "board_detected": False,
                    "flash_attempted": False,
                    "flash_succeeded": False,
                    "flash_state": "blocked_board_not_detected",
                }
            },
            "forbidden_claims_absent": True,
        },
    )
    _write_json(
        root,
        mod.CAPSTONE_V278_REL_PATH,
        {
            "honest_verdict": "complete: milestone_278_capstone; paper_ready=false",
            "paper_ready": False,
            "forbidden_claims_absent": True,
            "gaps_remaining": [
                "Repair delta remains flagged.",
                "Self-learning evidence needs non-tautological guard.",
                "GateMate flash evidence is absent.",
            ],
            "outcome_summaries": {
                "code_repair": {"pass_at_1_delta": 0.25, "safe_claim": "pilot_only"},
                "solver": {"parseability_rate": 0.083333, "solver_verified_accuracy": 0.0},
            },
        },
    )
    _write_json(
        root,
        mod.EXP2962_REL_PATH,
        {
            "honest_verdict": "complete: archive_ready=true",
            "archive_ready": True,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP2963_REL_PATH,
        {
            "honest_verdict": "complete: DCCD structured-repair protocol ready",
            "dccd_repair_protocol_ready": True,
            "n_tasks_planned_min": 20,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2964_REL_PATH,
        {
            "honest_verdict": "complete: DCCD repair replication did not clear promotion gates",
            "inference_substrate": "live_llm_inference",
            "n_tasks": 20,
            "baseline_pass_at_1": 0.2,
            "taxonomy_repair_pass_at_1": 0.45,
            "dccd_repair_pass_at_1": 0.0,
            "pass_at_1_delta": -0.2,
            "baseline_pass_at_k": 0.3,
            "dccd_repair_pass_at_k": 0.0,
            "pass_at_k_delta": -0.3,
            "syntax_failure_rate_delta": 0.3,
            "schema_failure_rate_delta": 0.95,
            "false_accept_delta": -0.05,
            "dccd_repair_replication_clean": False,
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2965_REL_PATH,
        {
            "honest_verdict": "complete: bounded certificate audit ready",
            "inference_substrate": "deterministic_wiring",
            "beaver_style_certificate_ready": True,
            "available_repair_candidate_count": 32,
            "available_repair_candidate_audited_count": 32,
            "full_beaver_claim": False,
            "validation_fixture_passed": True,
            "validation_fixture_count": 5,
        },
    )
    _write_json(
        root,
        mod.EXP2966_REL_PATH,
        {
            "honest_verdict": "complete: exact skill-labeled logic frontier materialized",
            "inference_substrate": "deterministic_wiring",
            "logic_frontier_materialized": True,
            "n_items": 24,
            "reference_z3_execution_rate": 1.0,
            "reference_solver_accuracy": 1.0,
            "skill_label_counts": {"symbolization": 24, "validity": 10},
            "manifest_sha256": "frontier-sha",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2967_REL_PATH,
        {
            "honest_verdict": "complete: local SOTA DCCD formalizations did not clear gate",
            "inference_substrate": "live_llm_inference",
            "n_items": 24,
            "baseline_parseability_rate": 0.083333,
            "baseline_solver_verified_accuracy": 0.0,
            "parseability_rate": 0.25,
            "solver_verified_accuracy": 0.208333,
            "answer_accuracy": 0.25,
            "z3_execution_rate": 0.208333,
            "formalization_delta_clean": False,
            "failure_categories": {"solver_verified_correct": 5, "unparseable": 18},
            "skill_wise_metrics": {"symbolization": {"parseability_rate": 0.25}},
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2968_REL_PATH,
        {
            "honest_verdict": "complete: deterministic partial monitor harness ready",
            "inference_substrate": "deterministic_wiring",
            "partial_monitor_harness_ready": True,
            "fixture_trace_count": 5,
            "fixture_checks_passed": True,
            "full_streaming_verification_claim": False,
            "latency_estimate_ms": 4.67,
            "coverage_by_event": {"partial_code_block": 2, "solver_query": 1},
        },
    )
    _write_json(
        root,
        mod.EXP2969_REL_PATH,
        {
            "honest_verdict": "complete: non_tautological_self_learning_ready",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "continuous_self_learning_task": True,
            "non_tautological_self_learning_ready": True,
            "leakage_check_passed": True,
            "frozen_heldout_utility": 0.033333333333,
            "random_replay_heldout_utility": 0.142857142857,
            "prior_utility_gated_heldout_utility": 0.329686371841,
            "new_heldout_utility": 0.236013686912,
            "heldout_utility_delta_vs_random": 0.093156544055,
            "negative_control_delta": 0.0,
            "forgetting_guard_passed": True,
            "rollback_triggered": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2970_REL_PATH,
        {
            "honest_verdict": "complete: kan_forgetting_guard_ready",
            "inference_substrate": "deterministic_wiring",
            "kan_forgetting_guard_ready": True,
            "selected_policy": "per_knot_importance_update",
            "forgetting_threshold": 0.05,
            "forgetting_delta_by_policy": {"eager_update": 0.75, "per_knot_importance_update": 0.0},
            "current_domain_utility": {"per_knot_importance_update": 1.0},
            "old_domain_utility": {"per_knot_importance_update": 1.0},
            "high_dimensional_claim_allowed": False,
            "no_synthesis_claim": True,
            "no_analog_claim": True,
        },
    )
    _write_json(
        root,
        mod.EXP2971_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_flash_preconditions_ready",
            "inference_substrate": "hardware_preflight",
            "gatemate_board_detected": True,
            "bitstream_sha256_verified": True,
            "gatemate_flash_preconditions_ready": True,
            "bitstream_sha256": "bitstream-sha",
            "flash_command": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb bitstream.bit",
        },
    )
    _write_json(
        root,
        mod.EXP2972_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_flash_contact_smoke_no_readback",
            "inference_substrate": "hardware_smoke",
            "board_detected": True,
            "bitstream_sha256_verified": True,
            "flash_attempted": True,
            "flash_succeeded": True,
            "smoke_vector_passed": False,
            "observed_output_sha256": "output-sha",
            "timing_observation": {"post_flash_contact_detected": True},
        },
    )


def test_req_report_2973_spec_anchor_exists() -> None:
    """REQ-REPORT-2973: OpenSpec declares the matrix v13 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2973" in spec
    assert "SCENARIO-REPORT-2973" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2973_builds_v13_from_279_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2973: v13 carries .278 facts and adds .279 row buckets."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["matrix_v13_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["forbidden_claims_absent"] is True

    assert "corpus:FoVer" in artifact["clean_rows"]
    assert "exp2965_beaver_style_certificates" in artifact["clean_rows"]
    assert "exp2970_kan_forgetting_guard" in artifact["clean_rows"]
    assert "exp2971_gatemate_flash_preflight" in artifact["clean_rows"]
    assert "exp2972_gatemate_flash_contact_hash" in artifact["clean_rows"]
    assert "exp2952_structured_repair_delta" in artifact["flagged_rows"]
    assert "exp2963_dccd_repair_protocol" in artifact["flagged_rows"]
    assert "exp2964_dccd_repair_replication" in artifact["flagged_rows"]
    assert "exp2966_logic_frontier_materializer" in artifact["flagged_rows"]
    assert "exp2967_solver_frontier_formalization" in artifact["flagged_rows"]
    assert "exp2969_non_tautological_fr11" in artifact["flagged_rows"]
    assert "exp2957_gatemate_flash_smoke" in artifact["blocked_rows"]
    assert "corpus:MBPP" in artifact["pilot_only_rows"]
    assert "exp2968_partial_monitor_harness" in artifact["pilot_only_rows"]
    assert artifact["gated_skipped_rows"] == []
    assert "exp2962_archive_activation" in artifact["aggregation_only_rows"]

    repair = artifact["repair_replication_summary"]
    assert repair["prior_278_pass_at_1_delta"] == pytest.approx(0.25)
    assert repair["dccd_repair_replication_clean"] is False
    assert repair["n_tasks"] == 20
    assert repair["pass_at_1_delta"] == pytest.approx(-0.2)
    assert repair["pass_at_k_delta"] == pytest.approx(-0.3)
    assert repair["syntax_failure_rate_delta"] == pytest.approx(0.3)
    assert repair["false_accept_delta"] == pytest.approx(-0.05)
    assert repair["artifact_flagged"] is True

    solver = artifact["solver_frontier_summary"]
    assert solver["frontier_materialized"] is True
    assert solver["frontier_n_items"] == 24
    assert solver["reference_solver_accuracy"] == pytest.approx(1.0)
    assert solver["formalization_n_items"] == 24
    assert solver["baseline_parseability_rate"] == pytest.approx(0.083333)
    assert solver["parseability_rate"] == pytest.approx(0.25)
    assert solver["parseability_delta_vs_278"] == pytest.approx(0.166667)
    assert solver["solver_verified_accuracy_delta_vs_278"] == pytest.approx(0.208333)
    assert solver["formalization_delta_clean"] is False

    self_learning = artifact["self_learning_summary"]
    assert self_learning["non_tautological_self_learning_ready"] is True
    assert self_learning["leakage_check_passed"] is True
    assert self_learning["heldout_utility_delta_vs_random"] == pytest.approx(0.093156544055)
    assert self_learning["negative_control_delta"] == pytest.approx(0.0)
    assert self_learning["artifact_flagged"] is True

    kan = artifact["kan_memory_summary"]
    assert kan["kan_forgetting_guard_ready"] is True
    assert kan["selected_policy"] == "per_knot_importance_update"
    assert kan["high_dimensional_claim_allowed"] is False
    assert kan["claim_boundary"] == "bounded_fixture_no_synthesis_or_analog_claim"

    hardware = artifact["hardware_state_summary"]
    assert hardware["gatemate"]["prior_278_flash_state"] == "blocked_board_not_detected"
    assert hardware["gatemate"]["board_detected"] is True
    assert hardware["gatemate"]["flash_preconditions_ready"] is True
    assert hardware["gatemate"]["flash_attempted"] is True
    assert hardware["gatemate"]["flash_succeeded"] is True
    assert hardware["gatemate"]["smoke_vector_passed"] is False
    assert hardware["gatemate"]["observed_output_sha256"] == "output-sha"

    expected_paths = {
        mod.MATRIX_V12_REL_PATH.as_posix(),
        mod.CAPSTONE_V278_REL_PATH.as_posix(),
        mod.EXP2962_REL_PATH.as_posix(),
        mod.EXP2963_REL_PATH.as_posix(),
        mod.EXP2964_REL_PATH.as_posix(),
        mod.EXP2965_REL_PATH.as_posix(),
        mod.EXP2966_REL_PATH.as_posix(),
        mod.EXP2967_REL_PATH.as_posix(),
        mod.EXP2968_REL_PATH.as_posix(),
        mod.EXP2969_REL_PATH.as_posix(),
        mod.EXP2970_REL_PATH.as_posix(),
        mod.EXP2971_REL_PATH.as_posix(),
        mod.EXP2972_REL_PATH.as_posix(),
    }
    assert {item["path"] for item in artifact["upstream_artifacts_read"]} == expected_paths
    assert artifact["upstream_checksums"][mod.EXP2964_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2964_REL_PATH
    )


def test_req_report_2973_blocks_when_exp2969_precondition_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-2973: Exp 2969 readiness is required before v13 promotion."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2969_REL_PATH,
        {
            "honest_verdict": "blocked_missing_slice_evidence",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "non_tautological_self_learning_ready": False,
            "leakage_check_passed": False,
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "blocked_non_tautological_self_learning_not_ready"
    assert artifact["matrix_v13_ready"] is False
    assert "exp2969_non_tautological_fr11" in artifact["blocked_rows"]
    assert artifact["self_learning_summary"]["non_tautological_self_learning_ready"] is False
    assert artifact["upstream_checksums"][mod.EXP2969_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2969_REL_PATH
    )


def test_req_report_2973_tolerates_missing_optional_279_branches(tmp_path: Path) -> None:
    """REQ-REPORT-2973: absent optional branches become gated-skipped rows."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.EXP2967_REL_PATH).unlink()
    (tmp_path / mod.EXP2965_REL_PATH).write_text("{not-json}\n", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["matrix_v13_ready"] is True
    assert "exp2965_beaver_style_certificates" in artifact["gated_skipped_rows"]
    assert "exp2967_solver_frontier_formalization" in artifact["gated_skipped_rows"]
    assert artifact["upstream_checksums"][mod.EXP2965_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2965_REL_PATH
    )
    assert artifact["upstream_checksums"][mod.EXP2967_REL_PATH.as_posix()] is None


def test_req_report_2973_blocks_missing_required_sources(tmp_path: Path) -> None:
    """REQ-REPORT-2973: missing required carry-forward sources fail closed."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V12_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.125)

    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["matrix_v13_ready"] is False
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp2960",
            "path": mod.MATRIX_V12_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_2973_write_artifact_persists_compact_json(tmp_path: Path) -> None:
    """REQ-REPORT-2973: write_artifact emits the stable deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=4.125)
    saved = json.loads(output.read_text(encoding="utf-8"))
    rendered = json.dumps(saved, sort_keys=True)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v13_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    assert "KV260 hardware speedup" not in rendered
    assert "Boltzmann" not in rendered
    assert "thermalization" not in rendered
    assert "TSU performance" not in rendered
    assert "Kona performance" not in rendered
    assert "native EBT training" not in rendered


def test_req_report_2973_helper_edges_keep_classification_honest(tmp_path: Path) -> None:
    """REQ-REPORT-2973: helper edges preserve blocked, flagged, and legacy buckets."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._class_from_flags({"honest_verdict": "blocked_board"}, default="clean") == "blocked"
    assert mod._class_from_flags({"flagged_adversarial": True}, default="clean") == "flagged"
    assert mod._class_from_flags({}, default="clean") == "clean"
    assert (
        mod._class_from_ready({}, "some_ready_flag", ready_class="clean")
        == "gated-skipped"
    )
    assert (
        mod._class_from_ready({"some_ready_flag": True}, "some_ready_flag", ready_class="clean")
        == "clean"
    )
    assert (
        mod._class_from_ready(
            {"some_ready_flag": True, "corrigendum_pending": [{"kind": "FLAG"}]},
            "some_ready_flag",
            ready_class="clean",
        )
        == "flagged"
    )
    assert mod._dccd_replication_class({}) == "gated-skipped"
    assert (
        mod._dccd_replication_class({"dccd_repair_replication_clean": True})
        == "clean"
    )
    assert mod._formalization_class({}) == "gated-skipped"
    assert mod._formalization_class({"formalization_delta_clean": True}) == "clean"
    assert mod._partial_monitor_class({}) == "gated-skipped"
    assert (
        mod._partial_monitor_class({"partial_monitor_harness_ready": False})
        == "blocked"
    )
    assert mod._gatemate_flash_class({}) == "gated-skipped"
    assert mod._gatemate_flash_class({"flash_succeeded": False}) == "blocked"
    assert mod._v12_bucket({"rows_clean": ["legacy"]}, "clean") == ["legacy"]
    assert mod._v12_bucket({"clean_rows": ["current"], "rows_clean": ["legacy"]}, "clean") == [
        "current"
    ]
    assert mod._get_path({"a": 1}, "a.b") is None
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
