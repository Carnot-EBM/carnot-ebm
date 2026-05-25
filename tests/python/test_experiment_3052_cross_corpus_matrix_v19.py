"""Tests for Exp 3052 cross-corpus matrix v19.

Spec refs: REQ-REPORT-3052, SCENARIO-REPORT-3052.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v19_3052 as mod


REQUIRED_FIELDS = {
    "matrix_v19_ready",
    "rows_total",
    "clean_count",
    "flagged_count",
    "bounded_count",
    "blocked_count",
    "gated_skipped_count",
    "projection_only_count",
    "missing_count",
    "retired_count",
    "repair_claim_status",
    "fr11_self_learning_status",
    "gatemate_status",
    "ssqa_status",
    "rows",
    "source_artifacts",
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


def _v18_matrix() -> dict[str, Any]:
    return {
        "artifact": "experiment_3038_cross_corpus_matrix_v18",
        "matrix_v18_ready": True,
        "rows_total": 4,
        "matrix_rows": [
            {
                "experiment_id": "exp3026",
                "status": "projection_only",
                "task_class": "archive_activation",
                "source_honest_verdict": "complete: archived",
                "summary": {"next_milestone": "2026.05.284"},
            },
            {
                "experiment_id": "exp3028",
                "status": "flagged",
                "task_class": "repair_rerun",
                "repair_claim_status": "clean_candidate_flagged",
                "upstream_flags": ["TAUTOLOGY:critical"],
                "summary": {"clean_repair_rerun_ready": True},
            },
            {
                "experiment_id": "exp3030",
                "status": "clean",
                "task_class": "validator_frontier_corrigendum",
                "summary": {"verified_region_count": 40},
            },
            {
                "experiment_id": "exp3036",
                "status": "gated_skipped",
                "task_class": "gatemate_host_visible_flash_smoke",
                "host_visible_output_observed": False,
                "summary": {"gate_status": "gated_skipped"},
            },
        ],
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts"},
        "honest_verdict": "complete: matrix_v18_ready=true",
    }


def _capstone() -> dict[str, Any]:
    return {
        "artifact": "experiment_3039_capstone_v284",
        "capstone_ready": True,
        "paper_ready": False,
        "repair_claim_status": "bounded",
        "fr11_self_learning_status": "controller_only_promotable",
        "gatemate_status": "blocked_pinout_missing_bounded",
        "ssqa_status": "gate_skipped_bounded_no_performance_claim",
        "blockers_remaining": [
            {"area": "repair", "status": "bounded"},
            {"area": "gatemate", "status": "blocked_pinout_missing_bounded"},
        ],
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts"},
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _flag_hygiene() -> dict[str, Any]:
    return {
        "artifact": "experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1",
        "flag_hygiene_ready": True,
        "rows_reviewed": 3,
        "true_blocker_rows": [
            {
                "row_id": "exp3028:adversarial_flags",
                "classification": "true_blocker",
                "blocking": True,
                "source_artifact": mod.MATRIX_V18_REL_PATH.as_posix(),
                "source_field": "matrix_rows[exp3028].upstream_flags",
            }
        ],
        "aggregation_false_positive_rows": [],
        "missing_metadata_rows": [],
        "unresolved_bound_rows": [],
        "hardware_blocked_rows": [],
        "gate_skipped_rows": [],
        "honest_verdict": "complete: flag_hygiene_ready=true",
    }


def _repair_reconciliation(*, candidate: bool = False) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    status = "clean_candidate" if candidate else "bounded"
    if not candidate:
        blockers = [
            {
                "row_id": "exp3028:adversarial_flags",
                "classification": "true_blocker",
                "blocking": True,
                "source_artifact": "results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json",
                "source_field": "corrigendum_pending",
                "rationale": "repair flags remain",
            },
            {
                "row_id": "exp3029:headline_sota_repair_clean_methodology",
                "classification": "true_blocker",
                "blocking": True,
                "source_artifact": "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
                "source_field": "retired_or_blocked_claims[headline_sota_repair_clean_methodology]",
                "evidence": {
                    "claim_id": "headline_sota_repair_clean_methodology",
                    "classification": "retired",
                },
            },
            {
                "row_id": "exp3029:unsupported_exp3016_headline_repair_promotion",
                "classification": "true_blocker",
                "blocking": True,
                "source_artifact": "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
                "source_field": "retired_or_blocked_claims[unsupported_exp3016_headline_repair_promotion]",
                "evidence": {
                    "claim_id": "unsupported_exp3016_headline_repair_promotion",
                    "classification": "retired",
                },
            },
        ]
    return {
        "artifact": "experiment_3042_repair_promotion_reconciliation_v3",
        "repair_reconciliation_ready": True,
        "repair_promotion_candidate": candidate,
        "repair_claim_status": status,
        "remaining_blockers": blockers,
        "repair_delta_summary": {"pass_at_1_delta": 0.375, "n_tasks": 24},
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts"},
        "honest_verdict": f"complete: repair_claim_status={status}",
    }


def _fingerprint() -> dict[str, Any]:
    return {
        "artifact": "experiment_3043_verified_speculation_transcript_fingerprint_v1",
        "status": "complete",
        "fingerprint_live_ready": True,
        "deterministic_replay_passed": True,
        "n_prompts": 3,
        "transcript_fingerprints": [{"id": idx} for idx in range(6)],
        "performance_or_repair_promotion_claim": False,
        "legacy_smoke_only_used": False,
        "models_used": ["fixture/headline-gguf"],
        "inference_substrate": {"recorded_before_model_load": True},
        "honest_verdict": "complete: fingerprint_live_ready=true",
    }


def _fr11_solver() -> dict[str, Any]:
    return {
        "artifact": "experiment_3046_fr11_solver_feedback_self_learning_loop_v1",
        "fr11_solver_feedback_ready": True,
        "continuous_self_learning_task": True,
        "promotion_decision": "controller_only_solver_feedback_ready",
        "edit_targets_used": ["controller_weights", "trace_memory"],
        "family_holdout_delta": 0.5,
        "prior_retention_delta": 0.0,
        "no_feedback_delta": 0.0,
        "shuffled_control_delta": -0.5,
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "controller_weight_update": True,
        },
        "honest_verdict": "complete_fr11_solver_feedback_controller_loop_ready",
    }


def _kan_locality() -> dict[str, Any]:
    return {
        "artifact": "experiment_3047_kan_locality_nonforgetting_probe_v2",
        "kan_locality_probe_ready": True,
        "promotion_decision": "controller_locality_evidence_only",
        "locality_metric": 0.75,
        "changed_anchor_count": 2,
        "anchored_prior_count": 2,
        "heldout_delta": 0.5,
        "prior_retention_delta": 0.0,
        "irrelevant_control_delta": 0.0,
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "kan_model_weight_training": False,
        },
        "honest_verdict": "complete_kan_locality_controller_probe_ready",
    }


def _gatemate_contract(*, ready: bool = False) -> dict[str, Any]:
    return {
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": ready,
        "selected_output_signal": "done",
        "ccf_binding": {"pin": "IO_EB_B7"} if ready else {},
        "host_reader_command": "read done" if ready else "",
        "expected_transcript": ["done=1 PASS"] if ready else [],
        "missing_operator_actions": [] if ready else ["Provide pinout", "Commit reader"],
        "hardware_execution_claim_made": False,
        "hardware_execution_performed": False,
        "speedup_claim_made": False,
        "inference_substrate": {"model_inference": False, "hardware_execution": False},
        "honest_verdict": "complete: contract_ready" if ready else "complete: blocked_contract",
    }


def _gatemate_smoke_passed() -> dict[str, Any]:
    return {
        "gatemate_host_visible_smoke_passed": True,
        "observed_transcript": ["done=1 PASS"],
        "expected_transcript": ["done=1 PASS"],
        "transcript_matched": True,
        "hardware_execution_claim_made": True,
        "speedup_claim_made": False,
        "boltzmann_claim_made": False,
        "sampler_claim_made": False,
        "inference_substrate": {"model_inference": False, "hardware_execution": True},
        "honest_verdict": "complete: gatemate_host_visible_smoke_passed=true",
    }


def _ssqa_gate_skipped() -> dict[str, Any]:
    return {
        "experiment": 3051,
        "status": "blocked",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": "1 of 1 gate(s) failed",
        "gates_evaluated": [
            {
                "upstream": "exp3050-gatemate-host-visible-flash-smoke-v5",
                "artifact_field": "gatemate_host_visible_smoke_passed",
                "expected": True,
                "actual": None,
                "passed": False,
                "reason": "upstream artifact not found",
            }
        ],
        "honest_verdict": "blocked_gate_check_failed",
    }


def _ssqa_ready() -> dict[str, Any]:
    return {
        "ssqa_gate_artifact_ready": True,
        "ssqa_status": "eligible_for_micro_panel",
        "consumed_gatemate_smoke": True,
        "observed_transcript_ref": "results/raw/gatemate/transcript.txt",
        "missing_inputs": [],
        "speedup_claim_made": False,
        "sampler_claim_made": False,
        "inference_substrate": {"model_inference": False},
        "honest_verdict": "complete: ssqa_gate_artifact_ready=true",
    }


def _write_sources(
    root: Path,
    *,
    repair_candidate: bool = False,
    contract_ready: bool = False,
    smoke: dict[str, Any] | None = None,
    ssqa: dict[str, Any] | None = None,
) -> None:
    _write_json(root, mod.MATRIX_V18_REL_PATH, _v18_matrix())
    _write_json(root, mod.CAPSTONE_V284_REL_PATH, _capstone())
    _write_json(root, mod.EXP3041_REL_PATH, _flag_hygiene())
    _write_json(root, mod.EXP3042_REL_PATH, _repair_reconciliation(candidate=repair_candidate))
    _write_json(root, mod.EXP3043_REL_PATH, _fingerprint())
    _write_json(root, mod.EXP3046_REL_PATH, _fr11_solver())
    _write_json(root, mod.EXP3047_REL_PATH, _kan_locality())
    _write_json(root, mod.EXP3048_REL_PATH, _gatemate_contract(ready=contract_ready))
    if smoke is not None:
        _write_json(root, mod.EXP3050_REL_PATH, smoke)
    _write_json(root, mod.EXP3051_REL_PATH, ssqa or _ssqa_gate_skipped())


def _rows_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact["rows"]}


def test_req_report_3052_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3052: OpenSpec declares the matrix v19 contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3052" in spec
    assert "SCENARIO-REPORT-3052" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3052_classifies_current_v19_claim_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3052: bounded, blocked, skipped, missing, and retired rows remain visible."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.5)
    rows = _rows_by_id(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert FORBIDDEN_TOP_LEVEL.isdisjoint(artifact)
    assert artifact["matrix_v19_ready"] is True
    assert artifact["honest_verdict"].startswith("complete: matrix_v19_ready=true")
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["rows_total"] == len(artifact["rows"]) == 15
    assert artifact["clean_count"] == 3
    assert artifact["flagged_count"] == 1
    assert artifact["bounded_count"] == 4
    assert artifact["blocked_count"] == 1
    assert artifact["gated_skipped_count"] == 2
    assert artifact["projection_only_count"] == 1
    assert artifact["missing_count"] == 1
    assert artifact["retired_count"] == 2

    assert rows["repair:headline_status"]["status"] == "bounded"
    assert artifact["repair_claim_status"] == "bounded"
    assert rows["fr11:solver_feedback"]["status"] == "bounded"
    assert rows["fr11:kan_locality"]["status"] == "bounded"
    assert artifact["fr11_self_learning_status"] == (
        "controller_only_solver_feedback_and_locality_ready"
    )
    assert rows["gatemate:output_contract"]["status"] == "blocked"
    assert rows["gatemate:host_visible_smoke"]["status"] == "missing"
    assert artifact["gatemate_status"] == "blocked_output_contract"
    assert rows["ssqa:readback_gate"]["status"] == "gated_skipped"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"
    assert rows["exp3029:headline_sota_repair_clean_methodology"]["status"] == "retired"
    assert rows["exp3029:unsupported_exp3016_headline_repair_promotion"]["status"] == "retired"
    assert rows["fingerprint:verified_speculation"]["status"] == "clean"

    assert all(
        {"source_artifact", "status", "evidence_class", "blocker_class"} <= row.keys()
        for row in artifact["rows"]
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }
    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.EXP3050_REL_PATH.as_posix()]["present"] is False
    assert source_by_path[mod.EXP3048_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3048_REL_PATH
    )


def test_req_report_3052_promotes_repair_and_hardware_only_when_gates_pass(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3052: repair, GateMate, and SSQA promotion require their exact gates."""

    _write_sources(
        tmp_path,
        repair_candidate=True,
        contract_ready=True,
        smoke=_gatemate_smoke_passed(),
        ssqa=_ssqa_ready(),
    )

    artifact = mod.build_artifact(tmp_path)
    rows = _rows_by_id(artifact)

    assert rows["repair:headline_status"]["status"] == "clean"
    assert artifact["repair_claim_status"] == "clean_candidate"
    assert rows["gatemate:output_contract"]["status"] == "clean"
    assert rows["gatemate:host_visible_smoke"]["status"] == "clean"
    assert artifact["gatemate_status"] == "host_visible_transcript_ready"
    assert rows["ssqa:readback_gate"]["status"] == "clean"
    assert artifact["ssqa_status"] == "eligible_for_micro_panel"


def test_req_report_3052_blocks_fr11_on_model_weight_scope_violation(tmp_path: Path) -> None:
    """REQ-REPORT-3052: controller-only FR-11 evidence cannot become model-weight learning."""

    _write_sources(tmp_path)
    bad_fr11 = _fr11_solver()
    bad_fr11["inference_substrate"] = {
        "live_llm_inference": False,
        "model_weight_training": True,
        "model_weight_mutation": False,
    }
    _write_json(tmp_path, mod.EXP3046_REL_PATH, bad_fr11)

    artifact = mod.build_artifact(tmp_path)
    rows = _rows_by_id(artifact)

    assert rows["fr11:solver_feedback"]["status"] == "blocked"
    assert rows["fr11:solver_feedback"]["blocker_class"] == "model_weight_scope_violation"
    assert artifact["fr11_self_learning_status"] == "blocked_model_weight_scope_violation"


def test_req_report_3052_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3052: write_artifact emits the deliverable JSON."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=10.0, now_s=11.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v19_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)
    assert saved["rows_total"] == len(saved["rows"])


def test_req_report_3052_helper_edges_preserve_malformed_and_missing_inputs(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3052: malformed, missing, and unknown statuses fail closed."""

    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._normal_status("pilot_only") == "bounded"
    assert mod._normal_status("gate_skipped") == "gated_skipped"
    assert mod._normal_status("gated-skipped") == "gated_skipped"
    assert mod._normal_status("not-a-status") == "missing"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._bool_field({"flag": True}, "flag") is True
    assert mod._bool_field({"flag": False}, "flag") is False
    assert mod._bool_field({"flag": "yes"}, "flag") is False
    assert mod._int_or_none(None) is None
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._float_or_none(None) is None
    assert mod._float_or_none(False) is None
    assert mod._float_or_none("bad") is None

    assert mod._capstone_row({})["status"] == "missing"
    assert mod._capstone_row({"capstone_ready": False})["status"] == "blocked"
    assert mod._capstone_row({"capstone_ready": True, "paper_ready": True})["status"] == "clean"
    assert mod._repair_row({})["status"] == "missing"
    assert mod._repair_row({"repair_reconciliation_ready": False})["status"] == "blocked"
    assert (
        mod._repair_row({"repair_reconciliation_ready": True, "repair_claim_status": "retired"})[
            "status"
        ]
        == "retired"
    )
    assert mod._repair_claim_status({}) == "missing"

    blocked_fr11 = {
        "fr11_solver_feedback_ready": False,
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
        },
    }
    bad_kan = _kan_locality()
    bad_kan["inference_substrate"] = {"kan_model_weight_training": True}
    assert mod._fr11_solver_row({})["status"] == "missing"
    assert mod._fr11_solver_row(blocked_fr11)["status"] == "blocked"
    assert mod._kan_locality_row({})["status"] == "missing"
    assert mod._kan_locality_row(bad_kan)["status"] == "blocked"
    assert (
        mod._kan_locality_row(
            {
                "kan_locality_probe_ready": False,
                "inference_substrate": {"model_weight_training": False},
            }
        )["status"]
        == "blocked"
    )
    assert mod._fr11_self_learning_status(_fr11_solver(), {}) == (
        "controller_only_solver_feedback_ready"
    )
    assert mod._fr11_self_learning_status({}, _kan_locality()) == "controller_only_locality_ready"
    assert mod._fr11_self_learning_status({"fr11_solver_feedback_ready": False}, {}) == "blocked"
    assert mod._fr11_self_learning_status({}, {}) == "missing"
    assert mod._fr11_self_learning_status({}, bad_kan) == "blocked_model_weight_scope_violation"

    assert mod._gatemate_status({}, {}) == "missing_host_visible_transcript"
    assert mod._gatemate_status({}, _gatemate_smoke_passed()) == "blocked_host_visible_transcript"
    assert mod._ssqa_gate_row({}, {})["status"] == "missing"
    assert mod._ssqa_gate_row({"status": "manual_review"}, _gatemate_smoke_passed())[
        "status"
    ] == "blocked"
    assert mod._ssqa_status({}, {}) == "missing"
    assert mod._ssqa_status({"status": "manual_review"}, _gatemate_smoke_passed()) == "blocked"

    assert mod._source_present([], "exp0000") is False
    empty_counts = {status: 0 for status in mod.STATUSES}
    assert mod._honest_verdict(False, empty_counts, 0, []) == (
        "blocked_matrix_v19_rows_unclassified; rows_total=0; clean=0; flagged=0; "
        "bounded=0; blocked=0; gated_skipped=0; projection_only=0; missing=0; retired=0"
    )
    assert mod._honest_verdict(False, empty_counts, 0, ["exp3038"]) == (
        "blocked_required_source_missing: exp3038"
    )
