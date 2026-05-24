"""Tests for Exp 2998 cross-corpus matrix v15.

Spec refs: REQ-REPORT-2998, SCENARIO-REPORT-2998.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v15_2998 as mod


REQUIRED_FIELDS = {
    "matrix_v15_ready",
    "n_clean",
    "n_flagged",
    "n_blocked",
    "n_gated_skipped",
    "n_pilot_only",
    "n_projection_only",
    "claim_rows",
    "hardware_claim_boundary",
    "self_learning_claim_boundary",
    "unresolved_blockers",
    "honest_verdict",
}

CLAIM_KEYS = {
    "sota_cache",
    "hard_code_manifest",
    "repair",
    "solver_provenance",
    "aquaforte_beaver_substrate",
    "prompt_validator_protocol",
    "fr11_self_learning",
    "gatemate",
    "ssqa",
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
        "source_experiment_id": "exp2986",
        "milestone": "2026.05.280",
        "status": status,
        "claim_class": "prior_v14",
        "evidence_type": "matrix_v14_row",
        "source_honest_verdict": "complete: prior",
        "summary": {"source_status": status},
    }


def _base_matrix_v14() -> dict[str, Any]:
    rows = [
        _prior_row("prior_clean", "clean"),
        _prior_row("prior_flagged", "flagged"),
        _prior_row("prior_blocked", "blocked"),
        _prior_row("prior_pilot", "pilot-only"),
        _prior_row("prior_projection", "projection-only"),
    ]
    return {
        "artifact": "experiment_2986_cross_corpus_matrix_v14",
        "honest_verdict": "complete: matrix_v14_ready=true",
        "matrix_v14_ready": True,
        "rows": rows,
    }


def _base_capstone_v280() -> dict[str, Any]:
    return {
        "artifact": "experiment_2987_capstone_v280",
        "honest_verdict": "complete: milestone_280_capstone; paper_ready=false",
        "milestone": "2026.05.280",
        "paper_ready": False,
        "paper_ready_blockers": ["repair_not_ready", "hardware_not_ready"],
        "gaps_remaining": ["repair blocked", "hardware blocked"],
        "matrix_v14_ready": True,
    }


def _write_ready_sources(root: Path) -> None:
    _write_json(root, mod.MATRIX_V14_REL_PATH, _base_matrix_v14())
    _write_json(root, mod.CAPSTONE_V280_REL_PATH, _base_capstone_v280())
    _write_json(
        root,
        mod.EXP2988_REL_PATH,
        {
            "honest_verdict": "complete: archive_ready=true",
            "archive_ready": True,
            "activated_milestone": "2026.05.281",
            "status_updates_written": False,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP2989_REL_PATH,
        {
            "artifact": "experiment_2989_sota_gguf_cache_provenance_preflight_v1",
            "honest_verdict": "success: at least one mandated headline SOTA GGUF produced a live transcript",
            "sota_headline_ready": True,
            "preconditions_checked": True,
            "sota_models_available": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
            "sota_models_attempted": [
                {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "cache_status": "missing"},
                {"hf_id": "unsloth/gemma-4-31B-it-GGUF", "cache_status": "missing"},
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "cache_status": "resolved",
                    "generation_status": "generated",
                    "gpu_backed": True,
                    "transcript_path": "results/raw/exp2989/gemma.json",
                },
            ],
            "live_transcript_paths": ["results/raw/exp2989/gemma.json"],
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        mod.EXP2990_REL_PATH,
        {
            "artifact": "experiment_2990_verifier_backed_hard_code_stress_manifest_v1",
            "honest_verdict": "ready: verifier-backed hard-code stress set validated",
            "hard_code_stress_set_ready": True,
            "n_items": 24,
            "all_items_have_tests": True,
            "all_reference_solutions_pass": True,
            "all_baseline_candidates_fail": True,
            "flaky_items": [],
            "rejected_item_ids": [],
            "manifest_path": "datasets/repair_hard/manifest_v1.jsonl",
            "inference_substrate": "deterministic_executable_manifest_generation",
        },
    )
    _write_json(
        root,
        mod.EXP2991_REL_PATH,
        {
            "artifact": "experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1",
            "honest_verdict": "flagged: hard-set repair did not clear promotion gates",
            "repair_rerun_clean": False,
            "headline_result": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "n_tasks": 24,
            "pass_at_1_delta": 0.4166666666666667,
            "pass_at_k_delta": 0.4166666666666667,
            "verifier_false_accept_delta": 0.0,
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        mod.EXP2992_REL_PATH,
        {
            "honest_verdict": "reproduced: solver-feedback formalization gain reproduced",
            "solver_provenance_reproduced": True,
            "formalization_clean": True,
            "n_items": 12,
            "parseability": 1.0,
            "solver_verified_accuracy": 1.0,
            "answer_accuracy": 1.0,
            "z3_execution_rate": 1.0,
            "tautology_rate": 0.0,
            "model_checksums_recorded": True,
            "prompt_hashes_recorded": True,
            "z3_transcript_hashes_recorded": True,
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "headline_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
            "inference_substrate": "live_llm_inference_plus_z3_provenance",
        },
    )
    _write_json(
        root,
        mod.EXP2993_REL_PATH,
        {
            "artifact": "experiment_2993_aquaforte_beaver_substrate_corrigendum_v1",
            "honest_verdict": "complete: live retry measured separately",
            "substrate_corrigendum_complete": True,
            "live_llm_retry_measured": True,
            "enumerator_only_fallback_measured": True,
            "substrate_labels_corrected": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "verifier_results_by_condition": {
                "live_llm_retry": {"measured": True, "pass_rate": 0.0},
                "enumerator_only_fallback": {"measured": True, "pass_rate": 1.0},
            },
            "inference_substrate": "live_llm_inference_plus_exact_verifier_and_enumerator_fallback",
        },
    )
    _write_json(
        root,
        mod.EXP2994_REL_PATH,
        {
            "artifact": "experiment_2994_prompt_validator_dialogue_schema_v1",
            "honest_verdict": "complete: prompt-validator dialogue protocol ready",
            "prompt_validator_protocol_ready": True,
            "exact_verifier_authority_preserved": True,
            "static_transition_representation_designed": True,
            "live_llm_judge_used": False,
            "llm_inference_run": False,
            "no_speed_claim_made": True,
            "n_validator_tree_fixtures": 3,
            "inference_substrate": "deterministic_prompt_validator_harness",
        },
    )
    _write_json(
        root,
        mod.EXP2995_REL_PATH,
        {
            "artifact": "experiment_2995_fr11_verifier_grounded_trace_memory_v2",
            "honest_verdict": "ready: verifier_grounded_trace_memory_ready",
            "continuous_self_learning_task": True,
            "trace_memory_ready": True,
            "independent_self_learning_boundary_preserved": True,
            "no_identical_metric_flag": True,
            "forgetting_guard_passed": True,
            "leakage_flag": False,
            "controls_improve_equally": False,
            "n_trace_memories": 8,
            "heldout_metric_deltas": {"pass_at_1": 1.0},
            "negative_control_deltas": {"disabled_update": {"pass_at_1": 0.0}},
            "inference_substrate": "artifact_replay_from_solver_and_validator_traces",
        },
    )
    _write_json(
        root,
        mod.EXP2996_REL_PATH,
        {
            "honest_verdict": "blocked_flash_failed",
            "hardware_smoke_boundary_recorded": True,
            "preconditions_checked": True,
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
            "missing_interface": "no host-visible transport",
            "inference_substrate": "physical_gatemate_board",
        },
    )


def _row_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact["rows"]}


def test_req_report_2998_spec_anchor_exists() -> None:
    """REQ-REPORT-2998: OpenSpec declares the matrix v15 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2998" in spec
    assert "SCENARIO-REPORT-2998" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2998_builds_v15_from_281_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2998: v15 aggregates .281 without requiring all pass."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    rows = _row_by_id(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete: matrix_v15_ready=true")
    assert artifact["matrix_v15_ready"] is True
    assert artifact["milestone"] == "2026.05.281"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["row_count"] == 15
    assert artifact["n_clean"] == 6
    assert artifact["n_flagged"] == 3
    assert artifact["n_blocked"] == 2
    assert artifact["n_gated_skipped"] == 0
    assert artifact["n_pilot_only"] == 1
    assert artifact["n_projection_only"] == 2
    assert artifact["n_missing"] == 1
    assert CLAIM_KEYS <= artifact["claim_rows"].keys()

    assert rows["carry_forward_v14:prior_clean"]["status"] == "clean"
    assert rows["carry_forward_v14:prior_flagged"]["status"] == "flagged"
    assert rows["exp2988_archive_activation"]["status"] == "projection-only"
    assert rows["exp2989_sota_cache"]["status"] == "clean"
    assert rows["exp2990_hard_code_manifest"]["status"] == "clean"
    assert rows["exp2991_intent_preserving_repair"]["status"] == "flagged"
    assert rows["exp2992_solver_provenance"]["status"] == "clean"
    assert rows["exp2993_aquaforte_beaver_substrate"]["status"] == "flagged"
    assert rows["exp2994_prompt_validator_protocol"]["status"] == "clean"
    assert rows["exp2995_fr11_trace_memory"]["status"] == "clean"
    assert rows["exp2996_gatemate_readback_smoke"]["status"] == "blocked"
    assert rows["exp2997_ssqa_dual_bram_rtl_pnr"]["status"] == "missing"

    assert rows["exp2989_sota_cache"]["summary"]["n_available_sota_models"] == 1
    assert rows["exp2991_intent_preserving_repair"]["paper_claim_eligible"] is False
    assert rows["exp2993_aquaforte_beaver_substrate"]["summary"]["live_llm_retry_measured"] is True
    assert rows["exp2993_aquaforte_beaver_substrate"]["summary"]["enumerator_only_fallback_measured"] is True
    assert rows["exp2994_prompt_validator_protocol"]["summary"]["llm_inference_run"] is False
    assert rows["exp2995_fr11_trace_memory"]["self_learning_boundary"]["status"] == "clean"
    assert rows["exp2996_gatemate_readback_smoke"]["hardware_boundary"]["status"] == "blocked"

    assert artifact["claim_rows"]["repair"]["status"] == "flagged"
    assert artifact["claim_rows"]["ssqa"]["status"] == "missing"
    assert artifact["hardware_claim_boundary"]["gatemate"]["status"] == "blocked"
    assert artifact["hardware_claim_boundary"]["ssqa"]["status"] == "missing"
    assert artifact["hardware_claim_boundary"]["forbidden_claims_absent"] is True
    assert artifact["self_learning_claim_boundary"]["status"] == "clean"
    assert artifact["paper_v6_claim_boundary"]["forbidden_claims_absent"] is True
    assert "exp2997_ssqa_dual_bram_rtl_pnr" in {b["row_id"] for b in artifact["unresolved_blockers"]}
    assert artifact["source_checksums"][mod.EXP2995_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2995_REL_PATH
    )


def test_req_report_2998_blocks_when_required_sources_missing(tmp_path: Path) -> None:
    """REQ-REPORT-2998: required v14/capstone sources fail closed."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V14_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["matrix_v15_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp2986",
            "path": mod.MATRIX_V14_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_2998_records_present_gate_block_as_gated_skipped(tmp_path: Path) -> None:
    """REQ-REPORT-2998: structured gate skips stay visible."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2991_REL_PATH,
        {
            "honest_verdict": "gate_blocked_sota_cache_or_hard_set",
            "inference_substrate": "live_llm_inference",
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.5)
    rows = _row_by_id(artifact)

    assert artifact["matrix_v15_ready"] is True
    assert rows["exp2991_intent_preserving_repair"]["status"] == "gated-skipped"
    assert artifact["n_gated_skipped"] == 1


def test_req_report_2998_hardware_claim_violation_flags_row(tmp_path: Path) -> None:
    """REQ-REPORT-2998: unsupported hardware claims are boundary violations."""

    _write_ready_sources(tmp_path)
    payload = json.loads((tmp_path / mod.EXP2996_REL_PATH).read_text(encoding="utf-8"))
    payload["speedup_claim_made"] = True
    _write_json(tmp_path, mod.EXP2996_REL_PATH, payload)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.125)
    rows = _row_by_id(artifact)

    assert rows["exp2996_gatemate_readback_smoke"]["status"] == "flagged"
    assert rows["exp2996_gatemate_readback_smoke"]["claim_boundary_guard_passed"] is False
    assert artifact["hardware_claim_boundary"]["forbidden_claims_absent"] is False
    assert artifact["claim_boundary_violations"] == [
        {
            "row_id": "exp2996_gatemate_readback_smoke",
            "violation": "unsupported_hardware_claim",
            "fields": ["speedup_claim_made"],
        }
    ]


def test_req_report_2998_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-2998: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=4.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v15_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.25)
    assert saved["row_count"] == len(saved["rows"])


def test_req_report_2998_helper_edges_keep_classification_honest(tmp_path: Path) -> None:
    """REQ-REPORT-2998: helpers keep absent, malformed, and flagged inputs honest."""

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
    assert mod._gated_verdict("SKIP: upstream gate false") is True
    assert mod._gated_verdict("complete: ok") is False
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [{"kind": "X"}]}) is True
    assert mod._has_flags({}) is False
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._status_with_guards("clean", {"flagged_adversarial": True}, []) == "flagged"
    assert mod._status_with_guards("clean", {}, [{"violation": "x"}]) == "flagged"
    assert mod._status_with_guards("clean", {}, []) == "clean"
    assert mod._status_with_guards("clean", {"honest_verdict": "blocked_x"}, []) == "blocked"
    assert mod._status_with_guards("clean", {"honest_verdict": "gate_blocked_x"}, []) == (
        "gated-skipped"
    )
    assert mod._model_boundary({})["status"] == "not_applicable"
    assert mod._model_boundary({"models_used": ["other"], "headline_model_ids": ["mandated"]})[
        "status"
    ] == "non_compliant_missing_mandated_model"
    assert mod._model_boundary(
        {"models_used": ["mandated"], "headline_model_ids": ["mandated"]}
    )["status"] == "compliant"
    assert mod._model_boundary(
        {"models_used": ["mandated"], "headline_model_ids": ["mandated"], "flagged_adversarial": True}
    )["status"] == "flagged_mandated_model_evidence"
    assert mod._hardware_boundary({})["status"] == "not_applicable"
    assert mod._hardware_boundary({"projection_only": True})["status"] == "projection_only"
    assert mod._hardware_boundary(
        {"inference_substrate": "physical_gatemate_board", "smoke_vector_passed": True}
    )["status"] == "clean"
    assert mod._self_learning_boundary({})["status"] == "not_applicable"
    assert mod._string_list("x") == []
    assert mod._mapping({"a": 1}) == {"a": 1}
    assert mod._mapping([]) == {}
    carried = mod._prior_v14_rows({"rows": [1, {"row_id": "bad_status", "status": "unknown"}]})
    assert carried == [
        {
            "row_id": "carry_forward_v14:bad_status",
            "source_experiment_id": "exp2986",
            "milestone": "2026.05.281",
            "status": "missing",
            "claim_class": "prior_v14_carry_forward",
            "evidence_type": "matrix_v14_row",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "headline_eligible": False,
            "paper_claim_eligible": False,
            "claim_boundary": "Matrix v14 row carried forward without metric recomputation or claim promotion.",
            "claim_boundary_guard_passed": True,
            "claim_boundary_violations": [],
            "source_honest_verdict": "",
            "upstream_flags": [],
            "model_boundary": {"status": "not_applicable"},
            "hardware_boundary": {"status": "not_applicable"},
            "self_learning_boundary": {"status": "not_applicable"},
            "summary": {
                "source_matrix": "v14",
                "source_row_id": "bad_status",
                "source_status": "missing",
                "source_claim_class": "",
            },
        }
    ]
