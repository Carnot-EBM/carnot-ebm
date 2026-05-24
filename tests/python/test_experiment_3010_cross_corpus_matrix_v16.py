"""Tests for Exp 3010 cross-corpus matrix v16.

Spec refs: REQ-REPORT-3010, SCENARIO-REPORT-3010.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v16_3010 as mod


REQUIRED_FIELDS = {
    "matrix_v16_ready",
    "clean_count",
    "flagged_count",
    "blocked_count",
    "gated_skipped_count",
    "missing_count",
    "projection_only_count",
    "repaired_claims",
    "still_blocked_claims",
    "claim_boundary_violations",
    "recommended_next_actions",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_row(row_id: str, status: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": "exp2998",
        "milestone": "2026.05.281",
        "status": status,
        "claim_class": "prior_v15",
        "evidence_type": "matrix_v15_row",
        "source_honest_verdict": f"prior {status}",
        "claim_boundary": "prior row copied for v16 carry-forward",
        "summary": {"source_status": status},
    }


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        mod.MATRIX_V15_REL_PATH,
        {
            "artifact": "experiment_2998_cross_corpus_matrix_v15",
            "honest_verdict": "complete: matrix_v15_ready=true",
            "matrix_v15_ready": True,
            "rows": [
                _prior_row("prior_clean", "clean"),
                _prior_row("prior_flagged", "flagged"),
                _prior_row("prior_blocked", "blocked"),
                _prior_row("prior_gated", "gated-skipped"),
                _prior_row("prior_pilot", "pilot-only"),
                _prior_row("prior_projection", "projection-only"),
                _prior_row("prior_missing", "missing"),
            ],
        },
    )
    _write_json(
        root,
        mod.CAPSTONE_V281_REL_PATH,
        {
            "artifact": "experiment_2999_capstone_v281",
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
            "capstone_ready": True,
            "paper_ready": False,
            "gaps_remaining": ["repair flagged", "GateMate blocked", "SSQA missing"],
            "next_milestone_recommendations": ["preserve paper-v6 narrowing"],
        },
    )
    _write_json(
        root,
        mod.EXP3000_REL_PATH,
        {
            "honest_verdict": "complete: archive_ready=true",
            "archive_ready": True,
            "activated_milestone": "2026.05.282",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP3001_REL_PATH,
        {
            "artifact": "experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1",
            "honest_verdict": "success: at least one mandated headline SOTA GGUF produced a live transcript",
            "sota_headline_ready": True,
            "preconditions_checked": True,
            "sota_models_available": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
            "live_transcript_paths": ["results/raw/exp3001/gemma.json"],
            "legacy_smoke_only_used": False,
            "model_specs": {
                "headline_models": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                    "unsloth/gemma-4-26B-A4B-it-GGUF",
                ],
                "smoke_only_models": ["Qwen/Qwen3.5-0.8B"],
            },
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        mod.EXP3002_REL_PATH,
        {
            "artifact": "experiment_3002_metamorphic_repair_oracle_audit_v1",
            "honest_verdict": "flagged: metamorphic oracle ready; downstream repair promotion must rerun against it",
            "metamorphic_oracle_ready": True,
            "false_accept_probe_ready": True,
            "tautology_probe_ready": True,
            "n_source_items": 24,
            "n_metamorphic_variants": 59,
            "relation_types": ["alpha_renaming", "input_permutation"],
            "inference_substrate": "deterministic_oracle_audit_no_live_llm",
        },
    )
    _write_json(
        root,
        mod.EXP3003_REL_PATH,
        {
            "artifact": "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1",
            "honest_verdict": "flagged: metamorphic repair rerun did not clear gates",
            "repair_rerun_clean": False,
            "headline_result": True,
            "n_tasks": 24,
            "n_metamorphic_variants": 59,
            "pass_at_1_delta": 0.4166666666666667,
            "pass_at_k_delta": 0.4166666666666667,
            "syntax_failure_rate_delta": 0.5,
            "schema_failure_rate_delta": 0.5,
            "false_accept_delta": 0.0,
            "tautology_gate_clean": True,
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "inference_substrate": "live_llm_inference_with_metamorphic_replay",
        },
    )
    _write_json(
        root,
        mod.EXP3004_REL_PATH,
        {
            "artifact": "experiment_3004_aquaforte_beaver_live_retry_provenance_v2",
            "honest_verdict": "clean: live retry provenance repaired and enumerator fallback separated",
            "substrate_corrigendum_promotable": True,
            "live_retry_provenance_clean": True,
            "enumerator_fallback_separated": True,
            "contamination_detected": False,
            "impossible_duration_flag": False,
            "duration_seconds_live": 10.25,
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "live_transcript_paths": ["results/raw/exp3004/live.json"],
            "model_checksums": {
                "unsloth/gemma-4-26B-A4B-it-GGUF": {"status": "available", "bounded_sha256": "abc"}
            },
        },
    )
    _write_json(
        root,
        mod.EXP3005_REL_PATH,
        {
            "artifact": "experiment_3005_solver_to_validator_tree_expansion_v1",
            "honest_verdict": "ready: expanded deterministic validator-tree corpus exact-checked",
            "validator_tree_expanded": True,
            "all_trees_exact_checked": True,
            "partial_viability_checked": True,
            "llm_judge_used": False,
            "n_validator_trees": 20,
            "inference_substrate": "deterministic_runtime_and_z3_validator_tree_corpus",
        },
    )
    _write_json(
        root,
        mod.EXP3006_REL_PATH,
        {
            "artifact": "experiment_3006_eqr_fixed_point_energy_diagnostic_v1",
            "honest_verdict": "ready: fixed-point diagnostic over cached validator trajectories complete",
            "fixed_point_diagnostic_ready": True,
            "native_eqr_claim_made": False,
            "convergence_rate": 1.0,
            "energy_monotonicity_rate": 1.0,
            "negative_control_rejection_rate": 1.0,
        },
    )
    _write_json(
        root,
        mod.EXP3007_REL_PATH,
        {
            "artifact": "experiment_3007_fr11_attractor_trace_memory_stability_v1",
            "honest_verdict": "ready: trace_memory_stability_ready",
            "trace_memory_stability_ready": True,
            "continuous_self_learning_task": True,
            "independent_self_learning_boundary_preserved": True,
            "convergence_guard_passed": True,
            "drift_guard_passed": True,
            "forgetting_guard_passed": True,
            "negative_control_rejected": True,
            "native_attractor_model_claim_made": False,
            "self_reported_memory_utility_counted": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "heldout_delta": 0.5,
            "heldout_task_count": 4,
            "inference_substrate": "artifact_replay_from_exact_verifier_traces",
        },
    )
    _write_json(
        root,
        mod.EXP3008_REL_PATH,
        {
            "honest_verdict": "blocked_flash_failed",
            "host_visible_io_ready": False,
            "hardware_smoke_boundary_recorded": True,
            "board_detected": True,
            "flash_attempted": True,
            "flash_succeeded": False,
            "readback_attempted": False,
            "readback_supported": False,
            "readback_hash": "",
            "smoke_vector_attempted": False,
            "smoke_vector_passed": False,
            "sampler_claim_allowed": False,
            "sampler_claim_made": False,
            "speedup_claim_allowed": False,
            "speedup_claim_made": False,
            "thermodynamic_claim_made": False,
            "boltzmann_claim_made": False,
            "io_transport_diagnosis": {"missing_interface": "no host-visible output transport"},
            "inference_substrate": "hardware_smoke",
        },
    )


def _row_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact["rows"]}


def test_req_report_3010_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3010: OpenSpec declares the matrix v16 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3010" in spec
    assert "SCENARIO-REPORT-3010" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3010_builds_v16_from_282_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3010: v16 aggregates .282 without requiring all pass."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    rows = _row_by_id(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete: matrix_v16_ready=true")
    assert artifact["matrix_v16_ready"] is True
    assert artifact["milestone"] == "2026.05.282"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["row_count"] == 17
    assert artifact["clean_count"] == 5
    assert artifact["flagged_count"] == 4
    assert artifact["blocked_count"] == 2
    assert artifact["gated_skipped_count"] == 2
    assert artifact["pilot_only_count"] == 1
    assert artifact["projection_only_count"] == 2
    assert artifact["missing_count"] == 1

    assert rows["carry_forward_v15:prior_clean"]["status"] == "clean"
    assert rows["exp3000_archive_activation"]["status"] == "projection-only"
    assert rows["exp3001_sota_cache"]["status"] == "clean"
    assert rows["exp3002_metamorphic_oracle"]["status"] == "flagged"
    assert rows["exp3003_metamorphic_repair"]["status"] == "flagged"
    assert rows["exp3004_aquaforte_beaver_provenance"]["status"] == "clean"
    assert rows["exp3005_validator_tree_expansion"]["status"] == "clean"
    assert rows["exp3006_fixed_point_diagnostic"]["status"] == "clean"
    assert rows["exp3007_fr11_trace_memory_stability"]["status"] == "flagged"
    assert rows["exp3008_gatemate_host_visible_io"]["status"] == "blocked"
    assert rows["exp3009_ssqa_dual_bram_report"]["status"] == "gated-skipped"
    assert rows["exp3009_ssqa_dual_bram_report"]["summary"]["missing_artifact_present"] is False

    assert artifact["claim_rows"]["exp3003_repair"]["status"] == "flagged"
    assert artifact["claim_rows"]["exp3004_substrate_provenance"]["status"] == "clean"
    assert artifact["claim_rows"]["exp3007_fr11_stability"]["status"] == "flagged"
    assert artifact["claim_rows"]["exp3008_gatemate_io"]["status"] == "blocked"
    assert artifact["claim_rows"]["exp3009_ssqa"]["status"] == "gated-skipped"
    assert "exp3004_aquaforte_beaver_substrate_provenance" in artifact["repaired_claims"]
    assert "exp3003_metamorphic_repair_flagged" in artifact["still_blocked_claims"]
    assert "exp3009_ssqa_gate_skipped_until_gatemate_io_ready" in artifact["still_blocked_claims"]
    assert artifact["claim_boundary_violations"] == []
    assert artifact["source_checksums"][mod.EXP3007_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3007_REL_PATH
    )
    assert mod.EXP3009_REL_PATH.as_posix() in artifact["missing_artifacts"]


def test_req_report_3010_blocks_when_required_sources_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3010: required carry-forward sources fail closed."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V15_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["matrix_v16_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp2998",
            "path": mod.MATRIX_V15_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_3010_hardware_claim_violation_flags_gatemate_row(tmp_path: Path) -> None:
    """REQ-REPORT-3010: unsupported hardware claims are boundary violations."""

    _write_ready_sources(tmp_path)
    payload = json.loads((tmp_path / mod.EXP3008_REL_PATH).read_text(encoding="utf-8"))
    payload["speedup_claim_made"] = True
    _write_json(tmp_path, mod.EXP3008_REL_PATH, payload)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    rows = _row_by_id(artifact)

    assert rows["exp3008_gatemate_host_visible_io"]["status"] == "flagged"
    assert rows["exp3008_gatemate_host_visible_io"]["claim_boundary_guard_passed"] is False
    assert artifact["claim_boundary_violations"] == [
        {
            "row_id": "exp3008_gatemate_host_visible_io",
            "violation": "unsupported_hardware_claim",
            "fields": ["speedup_claim_made"],
        }
    ]


def test_req_report_3010_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3010: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v16_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    assert saved["row_count"] == len(saved["rows"])


def test_req_report_3010_helper_edges_keep_classification_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3010: helpers keep absent, malformed, and flagged inputs honest."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._blocked_verdict("blocked_flash_failed") is True
    assert mod._blocked_verdict("complete: ok") is False
    assert mod._gated_verdict("gate_blocked_upstream") is True
    assert mod._gated_verdict("complete: ok") is False
    assert mod._flagged_verdict("flagged: check") is True
    assert mod._flagged_verdict("complete: ok") is False
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [{"kind": "X"}]}) is True
    assert mod._has_flags({}) is False
    assert mod._status_with_guards("clean", {"honest_verdict": "flagged: x"}, []) == "flagged"
    assert mod._status_with_guards("clean", {"honest_verdict": "blocked_x"}, []) == "blocked"
    assert mod._status_with_guards("clean", {"honest_verdict": "gate_blocked_x"}, []) == (
        "gated-skipped"
    )
    assert mod._status_with_guards("clean", {}, [{"violation": "x"}]) == "flagged"
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._string_list("x") == []
    assert mod._mapping([]) == {}
    assert mod._model_boundary({})["status"] == "not_applicable"
    assert mod._model_boundary({"model_specs": {"headline_models": ["mandated"]}})[
        "status"
    ] == "non_compliant_missing_mandated_model"
    assert mod._model_boundary(
        {"model_specs": {"headline_models": ["mandated"]}, "headline_models_used": ["mandated"]}
    )["status"] == "compliant"
    assert mod._model_boundary(
        {
            "model_specs": {"headline_models": ["mandated"]},
            "headline_models_used": ["mandated"],
            "honest_verdict": "flagged: check",
        }
    )["status"] == "flagged_mandated_model_evidence"
    assert mod._hardware_boundary({"projection_only": True})["status"] == "projection_only"
    assert mod._self_learning_boundary({})["status"] == "not_applicable"
    assert mod._self_learning_boundary(
        {
            "continuous_self_learning_task": True,
            "independent_self_learning_boundary_preserved": True,
            "forgetting_guard_passed": True,
            "negative_control_rejected": True,
            "native_attractor_model_claim_made": False,
            "self_reported_memory_utility_counted": False,
        }
    )["status"] == "clean"
    carried = mod._prior_v15_rows({"rows": [1, {"row_id": "bad", "status": "unknown"}]})
    assert carried[0]["row_id"] == "carry_forward_v15:bad"
    assert carried[0]["status"] == "missing"
    assert mod._exp3009_row({}, {"host_visible_io_ready": True})["status"] == "missing"
    assert mod._exp3009_row(
        {"projection_only": True, "honest_verdict": "complete: projection"},
        {"host_visible_io_ready": True},
    )["status"] == "projection-only"
    assert mod._claim_violations("row", {"llm_judge_used": True}) == [
        {
            "row_id": "row",
            "violation": "llm_as_verifier_boundary_violation",
            "fields": ["llm_judge_used"],
        }
    ]
    assert "exp3007_fr11_trace_memory_stability" in mod._repaired_claims(
        [{"row_id": "exp3007_fr11_trace_memory_stability", "status": "clean"}]
    )
