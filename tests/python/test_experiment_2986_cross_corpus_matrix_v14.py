"""Tests for Exp 2986 cross-corpus matrix v14.

Spec refs: REQ-REPORT-2986, SCENARIO-REPORT-2986.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v14_2986 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "matrix_v14_ready",
    "milestone",
    "clean_count",
    "flagged_count",
    "blocked_count",
    "gated_skipped_count",
    "pilot_only_count",
    "projection_only_count",
    "row_count",
    "rows",
    "repair_claim_status",
    "solver_claim_status",
    "fr11_claim_status",
    "hardware_claim_status",
    "model_compliance_summary",
    "claim_boundary_violations",
    "next_milestone_recommendations",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_matrix_v13() -> dict[str, Any]:
    return {
        "artifact": "experiment_2973_cross_corpus_matrix_v13",
        "honest_verdict": "complete: matrix_v13_ready=true",
        "matrix_v13_ready": True,
        "clean_rows": ["prior_clean"],
        "flagged_rows": ["prior_flagged"],
        "blocked_rows": ["prior_blocked"],
        "gated_skipped_rows": ["prior_gated"],
        "pilot_only_rows": ["prior_pilot"],
        "aggregation_only_rows": ["prior_aggregation"],
        "forbidden_claims_absent": True,
    }


def _base_capstone_v279() -> dict[str, Any]:
    return {
        "artifact": "experiment_2974_capstone_v279",
        "honest_verdict": "complete: milestone_279_capstone; paper_ready=false",
        "milestone": "2026.05.279",
        "paper_ready": False,
        "next_milestone_recommendations": [
            "DCCD .280: repair schema failures.",
            "Solver .280: remove tautology flags.",
        ],
    }


def _write_ready_sources(root: Path) -> None:
    _write_json(root, mod.MATRIX_V13_REL_PATH, _base_matrix_v13())
    _write_json(root, mod.CAPSTONE_V279_REL_PATH, _base_capstone_v279())
    _write_json(
        root,
        mod.EXP2975_REL_PATH,
        {
            "honest_verdict": "complete: archive_ready=true",
            "archive_ready": True,
            "scripts_research_conductor_modified": False,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP2976_REL_PATH,
        {
            "honest_verdict": "complete: intent-preserving protocol ready",
            "intent_preserving_repair_protocol_ready": True,
            "trace_execution_plan_ready": True,
            "prior_failure_addressed": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP2977_REL_PATH,
        {
            "honest_verdict": "blocked_cached_sota_pair_unavailable_cpu_smoke_only",
            "repair_rerun_clean": False,
            "n_tasks": 2,
            "pass_at_1_delta": 0.0,
            "pass_at_k_delta": 0.0,
            "syntax_failure_rate_delta": -1.0,
            "schema_failure_rate_delta": 0.0,
            "false_accept_delta": 0.0,
            "headline_result": False,
            "legacy_model_used_only_for_smoke": True,
            "models_used": ["Qwen/Qwen3.5-0.8B"],
            "mandatory_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        mod.EXP2978_REL_PATH,
        {
            "honest_verdict": "complete: diagnostic telemetry only",
            "telemetry_panel_ready": True,
            "semantic_energy_signal_usable": True,
            "first_step_signal_usable": True,
            "no_headline_verifier_claim": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "mandatory_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "inference_substrate": "mixed_artifact_and_optional_live_llm",
        },
    )
    _write_json(
        root,
        mod.EXP2979_REL_PATH,
        {
            "honest_verdict": "complete: deterministic solver frontier ready",
            "mcs_feedback_schema_ready": True,
            "frontier_upgrade_ready": True,
            "reference_solver_verified_accuracy": 1.0,
            "reference_z3_execution_rate": 1.0,
            "inference_substrate": "deterministic_z3_and_artifact_generation",
        },
    )
    _write_json(
        root,
        mod.EXP2980_REL_PATH,
        {
            "honest_verdict": "complete: feedback formalization cleared gates",
            "formalization_feedback_clean": True,
            "n_items": 6,
            "parseability_rate": 1.0,
            "solver_verified_accuracy": 1.0,
            "answer_accuracy": 1.0,
            "z3_execution_rate": 1.0,
            "tautology_flag_rate": 0.0,
            "headline_result": True,
            "feedback_repair_delta": 0.833333,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "mandatory_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "inference_substrate": "live_llm_inference_plus_z3",
        },
    )
    _write_json(
        root,
        mod.EXP2981_REL_PATH,
        {
            "honest_verdict": "complete: deterministic partial monitor promoted",
            "partial_monitor_promoted": True,
            "fixture_count": 11,
            "live_trace_count": 6,
            "prefix_failure_localization_rate": 1.0,
            "false_alarm_rate": 0.0,
            "full_streaming_verification_claim": False,
            "promotion_gates": {
                "full_streaming_claim_supported": {"passed": False},
                "prefix_failure_localization_rate": {"passed": True},
            },
            "inference_substrate": "deterministic_monitor_harness",
        },
    )
    _write_json(
        root,
        mod.EXP2982_REL_PATH,
        {
            "honest_verdict": "complete: fr11_independent_self_learning_ready",
            "continuous_self_learning_task": True,
            "fr11_independent_metrics_evaluated": True,
            "fr11_independent_self_learning_ready": True,
            "no_identical_metric_flag": True,
            "forgetting_guard_passed": True,
            "heldout_independent_delta_vs_random": {"pass_at_1": 0.17},
            "negative_control_delta": {"pass_at_1": 0.0},
            "inference_substrate": "aggregation_and_deterministic_replay",
        },
    )
    _write_json(
        root,
        mod.EXP2983_REL_PATH,
        {
            "honest_verdict": "complete: trace_to_skill_memory_ready",
            "trace_to_skill_memory_ready": True,
            "continuous_self_learning_task": True,
            "heldout_skill_reuse_delta": 0.22222223,
            "negative_control_delta": 0.0,
            "leakage_flag": False,
            "headline_result": False,
            "fresh_live_llm_inference_used": False,
            "pilot_source": ".280",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "mandatory_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "inference_substrate": "artifact_replay_and_optional_live_llm",
        },
    )
    _write_json(
        root,
        mod.EXP2984_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_no_readback_no_host_smoke_io",
            "board_detected": True,
            "flash_succeeded": True,
            "readback_attempted": False,
            "readback_supported": False,
            "smoke_vector_attempted": False,
            "smoke_vector_passed": False,
            "sampler_claim_allowed": False,
            "speedup_claim_allowed": False,
            "thermodynamic_claim_allowed": False,
            "inference_substrate": "physical_gatemate_board",
        },
    )
    _write_json(
        root,
        mod.EXP2985_REL_PATH,
        {
            "honest_verdict": "complete: register_map_plan_ready_projection_only",
            "register_map_plan_ready": True,
            "projection_only": True,
            "sampler_claim_allowed": False,
            "speedup_claim_allowed": False,
            "thermodynamic_claim_allowed": False,
            "inference_substrate": "architecture_projection_only",
        },
    )


def _row_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact["rows"]}


def test_req_report_2986_spec_anchor_exists() -> None:
    """REQ-REPORT-2986: OpenSpec declares the matrix v14 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2986" in spec
    assert "SCENARIO-REPORT-2986" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2986_builds_v14_from_280_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2986: v14 carries .279 rows and classifies .280 rows."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    rows = _row_by_id(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["matrix_v14_ready"] is True
    assert artifact["milestone"] == "2026.05.280"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["row_count"] == 17
    assert artifact["clean_count"] == 4
    assert artifact["flagged_count"] == 5
    assert artifact["blocked_count"] == 3
    assert artifact["gated_skipped_count"] == 1
    assert artifact["pilot_only_count"] == 1
    assert artifact["projection_only_count"] == 3
    assert artifact["claim_boundary_violations"] == []

    assert rows["carry_forward:prior_clean"]["status"] == "clean"
    assert rows["carry_forward:prior_flagged"]["status"] == "flagged"
    assert rows["carry_forward:prior_blocked"]["status"] == "blocked"
    assert rows["carry_forward:prior_gated"]["status"] == "gated-skipped"
    assert rows["carry_forward:prior_pilot"]["status"] == "pilot-only"
    assert rows["carry_forward:prior_aggregation"]["status"] == "projection-only"

    assert rows["exp2975_archive_activation"]["status"] == "projection-only"
    assert rows["exp2976_repair_protocol"]["status"] == "flagged"
    assert rows["exp2977_intent_preserving_repair"]["status"] == "blocked"
    assert rows["exp2978_semantic_energy_telemetry"]["status"] == "flagged"
    assert rows["exp2979_solver_feedback_frontier"]["status"] == "clean"
    assert rows["exp2980_solver_formalization_feedback"]["status"] == "flagged"
    assert rows["exp2981_partial_monitor_promotion"]["status"] == "clean"
    assert rows["exp2982_fr11_independent_metric_gate"]["status"] == "clean"
    assert rows["exp2983_trace_to_skill_memory_pilot"]["status"] == "flagged"
    assert rows["exp2984_gatemate_readback_smoke"]["status"] == "blocked"
    assert rows["exp2985_ssqa_register_map_plan"]["status"] == "projection-only"

    assert rows["exp2980_solver_formalization_feedback"]["claim_boundary_guard_passed"] is True
    assert rows["exp2980_solver_formalization_feedback"]["model_compliance"]["status"] == (
        "flagged_mandated_model_evidence"
    )
    assert rows["exp2977_intent_preserving_repair"]["model_compliance"]["status"] == (
        "legacy_smoke_only"
    )
    assert rows["exp2984_gatemate_readback_smoke"]["hardware_compliance"]["status"] == (
        "blocked_no_readback_or_smoke_output"
    )
    assert rows["exp2985_ssqa_register_map_plan"]["hardware_compliance"]["status"] == (
        "projection_only"
    )

    assert artifact["repair_claim_status"].startswith("blocked:")
    assert artifact["solver_claim_status"].startswith("flagged:")
    assert artifact["fr11_claim_status"].startswith("clean:")
    assert artifact["hardware_claim_status"].startswith("blocked:")
    assert artifact["model_compliance_summary"]["legacy_smoke_only"] == 1
    assert artifact["model_compliance_summary"]["flagged_mandated_model_evidence"] == 3
    assert artifact["source_checksums"][mod.EXP2982_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2982_REL_PATH
    )
    assert len(artifact["next_milestone_recommendations"]) >= 4


def test_req_report_2986_blocks_when_fr11_precondition_not_evaluated(tmp_path: Path) -> None:
    """REQ-REPORT-2986: Exp 2982 independent metric evaluation is mandatory."""

    _write_ready_sources(tmp_path)
    payload = json.loads((tmp_path / mod.EXP2982_REL_PATH).read_text(encoding="utf-8"))
    payload["fr11_independent_metrics_evaluated"] = False
    payload["fr11_independent_self_learning_ready"] = False
    _write_json(tmp_path, mod.EXP2982_REL_PATH, payload)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    rows = _row_by_id(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "blocked_fr11_independent_metrics_not_evaluated"
    assert artifact["matrix_v14_ready"] is False
    assert rows["exp2982_fr11_independent_metric_gate"]["status"] == "blocked"
    assert artifact["fr11_claim_status"].startswith("blocked:")


def test_req_report_2986_records_missing_optional_rows_as_gated_skipped(tmp_path: Path) -> None:
    """REQ-REPORT-2986: optional missing or malformed .280 artifacts stay explicit."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.EXP2980_REL_PATH).unlink()
    (tmp_path / mod.EXP2977_REL_PATH).write_text("{not-json}\n", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.75)
    rows = _row_by_id(artifact)

    assert artifact["matrix_v14_ready"] is True
    assert rows["exp2980_solver_formalization_feedback"]["status"] == "gated-skipped"
    assert rows["exp2977_intent_preserving_repair"]["status"] == "gated-skipped"
    assert artifact["source_checksums"][mod.EXP2980_REL_PATH.as_posix()] is None
    assert artifact["source_checksums"][mod.EXP2977_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2977_REL_PATH
    )


def test_req_report_2986_blocks_missing_required_sources(tmp_path: Path) -> None:
    """REQ-REPORT-2986: missing required carry-forward sources fail closed."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V13_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.5)

    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["matrix_v14_ready"] is False
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp2973",
            "path": mod.MATRIX_V13_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_2986_claim_boundary_violation_prevents_projection_promotion(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2986: unsupported hardware claims are surfaced as violations."""

    _write_ready_sources(tmp_path)
    payload = json.loads((tmp_path / mod.EXP2985_REL_PATH).read_text(encoding="utf-8"))
    payload["speedup_claim_allowed"] = True
    _write_json(tmp_path, mod.EXP2985_REL_PATH, payload)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.125)
    rows = _row_by_id(artifact)

    assert rows["exp2985_ssqa_register_map_plan"]["status"] == "flagged"
    assert rows["exp2985_ssqa_register_map_plan"]["claim_boundary_guard_passed"] is False
    assert artifact["claim_boundary_violations"] == [
        {
            "row_id": "exp2985_ssqa_register_map_plan",
            "violation": "unsupported_hardware_claim_allowed",
            "fields": ["speedup_claim_allowed"],
        }
    ]


def test_req_report_2986_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-2986: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=5.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v14_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.25)
    assert saved["row_count"] == len(saved["rows"])


def test_req_report_2986_helper_edges_keep_classification_honest(tmp_path: Path) -> None:
    """REQ-REPORT-2986: helpers keep absent, flagged, and blocked inputs honest."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._blocked_verdict("gate_blocked_missing_gpu") is True
    assert mod._blocked_verdict("complete: ok") is False
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [{"kind": "X"}]}) is True
    assert mod._has_flags({}) is False
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._status_with_flags("clean", {"flagged_adversarial": True}, True) == "flagged"
    assert mod._status_with_flags("clean", {}, False) == "flagged"
    assert mod._status_with_flags("clean", {}, True) == "clean"
    assert mod._model_compliance(
        {
            "mandatory_headline_model_ids": ["mandated"],
            "models_used": ["other"],
        }
    )["status"] == "non_compliant_missing_mandated_model"
    assert mod._model_compliance(
        {
            "mandatory_headline_model_ids": ["mandated"],
            "models_used": ["mandated"],
        }
    )["status"] == "compliant"
    assert mod._hardware_compliance(
        {"inference_substrate": "physical_gatemate_board", "smoke_vector_passed": True}
    )["status"] == "compliant_hardware_readback_or_smoke"
    assert mod._monitor_claim_violations(
        "monitor",
        {
            "full_streaming_verification_claim": True,
            "promotion_gates": {"full_streaming_claim_supported": {"passed": True}},
        },
    ) == []
    assert mod._monitor_claim_violations(
        "monitor",
        {
            "full_streaming_verification_claim": True,
            "promotion_gates": {"full_streaming_claim_supported": {"passed": False}},
        },
    ) == [
        {
            "row_id": "monitor",
            "violation": "unsupported_full_streaming_verification_claim",
            "fields": ["full_streaming_verification_claim"],
        }
    ]
    assert mod._get_path({"a": 1}, "a.b") is None
    assert mod._repair_claim_status([{"claim_class": "repair_eval", "status": "clean"}]).startswith(
        "clean:"
    )
    assert mod._repair_claim_status(
        [{"claim_class": "repair_protocol", "status": "projection-only"}]
    ).startswith("projection-only:")
    assert mod._solver_claim_status(
        [{"claim_class": "solver_eval", "status": "blocked"}]
    ).startswith("blocked:")
    assert mod._solver_claim_status(
        [{"claim_class": "solver_eval", "status": "clean"}]
    ).startswith("clean:")
    assert mod._fr11_claim_status(
        [{"row_id": "exp2982_fr11_independent_metric_gate", "status": "flagged"}]
    ).startswith("flagged:")
    assert mod._fr11_claim_status([]).startswith("gated-skipped:")
    assert mod._hardware_claim_status(
        [{"claim_class": "hardware_register_map_plan", "status": "projection-only"}]
    ).startswith("projection-only:")
    assert mod._hardware_claim_status([]).startswith("gated-skipped:")
