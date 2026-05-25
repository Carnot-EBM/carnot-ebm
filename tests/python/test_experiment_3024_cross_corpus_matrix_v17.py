"""Tests for Exp 3024 cross-corpus matrix v17.

Spec refs: REQ-REPORT-3024, SCENARIO-REPORT-3024.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v17_3024 as mod


REQUIRED_FIELDS = {
    "matrix_v17_ready",
    "clean_count",
    "flagged_count",
    "blocked_count",
    "gated_skipped_count",
    "missing_count",
    "projection_only_count",
    "repaired_claims",
    "still_blocked_claims",
    "claim_boundary_violations",
    "cited_upstream_artifacts",
    "inference_substrate",
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
        "source_experiment_id": "exp3010",
        "milestone": "2026.05.282",
        "status": status,
        "claim_class": "prior_v16",
        "evidence_type": "matrix_v16_row",
        "source_honest_verdict": f"prior {status}",
        "claim_boundary": "prior row copied for v17 carry-forward",
        "summary": {"source_status": status},
    }


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        mod.MATRIX_V16_REL_PATH,
        {
            "artifact": "experiment_3010_cross_corpus_matrix_v16",
            "honest_verdict": "complete: matrix_v16_ready=true",
            "matrix_v16_ready": True,
            "rows": [
                _prior_row("prior_clean", "clean"),
                _prior_row("prior_flagged", "flagged"),
                _prior_row("prior_blocked", "blocked"),
                _prior_row("prior_gated", "gated-skipped"),
                _prior_row("prior_pilot", "pilot-only"),
                _prior_row("prior_projection", "projection-only"),
                _prior_row("prior_missing", "missing"),
            ],
            "repaired_claims": ["exp3004_aquaforte_beaver_substrate_provenance"],
            "still_blocked_claims": ["exp3008_gatemate_host_visible_io_blocked"],
        },
    )
    _write_json(
        root,
        mod.CAPSTONE_V282_REL_PATH,
        {
            "artifact": "experiment_3011_capstone_v282",
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
            "capstone_ready": True,
            "paper_ready": False,
            "matrix_v16_ready": True,
        },
    )
    _write_json(
        root,
        mod.EXP3012_REL_PATH,
        {
            "archive_ready": True,
            "archived_milestone": "2026.05.282",
            "activated_milestone": "2026.05.283",
            "research_complete_updated": True,
            "status_updates_written": False,
            "n_tasks_archived": 12,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "honest_verdict": "complete: archive_ready=true; activated_milestone=2026.05.283",
        },
    )
    _write_json(
        root,
        mod.EXP3013_REL_PATH,
        {
            "honest_verdict": "complete: headline SOTA transcript and top-k logprob telemetry ready",
            "sota_headline_ready": True,
            "sota_logprob_ready": True,
            "preconditions_checked": True,
            "headline_models_available": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
            "live_transcript_paths": ["results/raw/exp3013/gemma.json"],
            "legacy_smoke_only_used": False,
            "model_specs": {
                "headline_models": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                    "unsloth/gemma-4-26B-A4B-it-GGUF",
                ],
                "smoke_only_models": ["Qwen/Qwen3.5-0.8B"],
            },
            "model_checksums": {
                "unsloth/gemma-4-26B-A4B-it-GGUF": {"status": "available"}
            },
            "inference_substrate": "llama_cpp_gpu",
        },
    )
    _write_json(
        root,
        mod.EXP3014_REL_PATH,
        {
            "honest_verdict": "complete: repair failure taxonomy ready",
            "repair_failure_taxonomy_ready": True,
            "n_cached_candidates_audited": 24,
            "syntax_failure_count": 12,
            "schema_failure_count": 12,
            "false_accept_count": 0,
            "tautology_failure_count": 1,
            "intent_drift_count": 1,
            "halluguard_ntk_claim_made": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP3015_REL_PATH,
        {
            "honest_verdict": "complete: offline repair acceptance controller ready",
            "acceptance_controller_ready": True,
            "n_candidates_evaluated": 24,
            "false_accept_delta_offline": 0.0,
            "syntax_failure_delta_offline": -0.5,
            "schema_failure_delta_offline": -0.5,
            "pass_at_1_delta_offline": 0.58,
            "llm_judge_used": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP3016_REL_PATH,
        {
            "honest_verdict": "complete: acceptance-controlled SOTA repair rerun gates passed",
            "repair_controller_clean": True,
            "headline_result": True,
            "preconditions_checked": True,
            "n_tasks": 24,
            "n_metamorphic_variants": 59,
            "model_specs": {
                "headline_models": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                    "unsloth/gemma-4-26B-A4B-it-GGUF",
                ],
                "smoke_only_models": ["Qwen/Qwen3.5-0.8B"],
            },
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "model_checksums": {
                "unsloth/gemma-4-26B-A4B-it-GGUF": {"status": "available"}
            },
            "pass_at_1_delta": 0.375,
            "pass_at_k_delta": 0.375,
            "false_accept_delta": 0.0,
            "tautology_gate_clean": True,
            "syntax_failure_rate_delta": 0.0,
            "schema_failure_rate_delta": 0.0,
            "live_transcript_paths": ["results/raw/exp3016/repair.json"],
            "verifier_log_paths": ["results/verifier_transcripts/exp3016/repair.json"],
            "inference_substrate": "live_sota_gguf_repair_with_acceptance_controller",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP3017_REL_PATH,
        {
            "honest_verdict": "complete: NSVIF instruction validator-tree corpus exact-checked",
            "instruction_validator_tree_ready": True,
            "n_instruction_items": 20,
            "n_validator_trees": 20,
            "exact_check_coverage": 0.97561,
            "all_authoritative_nodes_exact_checked": True,
            "z3_transcript_paths": ["results/exp3017/z3.json"],
            "runtime_transcript_paths": ["results/exp3017/runtime.json"],
            "rejected_items": [],
            "llm_judge_used": False,
        },
    )
    _write_json(
        root,
        mod.EXP3018_REL_PATH,
        {
            "honest_verdict": "complete: validator frontier certificate ready",
            "frontier_certificate_ready": True,
            "n_frontier_items": 44,
            "n_prefix_closed_items": 8,
            "certified_safe_count": 20,
            "certified_violating_count": 20,
            "unresolved_count": 2,
            "enumerator_fallback_separated": True,
            "live_llm_evidence_used": False,
            "transcript_paths": ["results/exp3018/frontier.json"],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP3019_REL_PATH,
        {
            "honest_verdict": "complete_flagged_tautology_risk_feasibility_channel_diagnostic",
            "feasibility_channel_diagnostic_ready": True,
            "n_rows": 47,
            "feasible_infeasible_auc": 1.0,
            "negative_control_rejection_rate": 1.0,
            "heldout_metric_correlation": 0.96,
            "tautology_risk_flag": True,
            "reused_label_as_feature": False,
            "native_dsp_claim_made": False,
        },
    )
    _write_json(
        root,
        mod.EXP3020_REL_PATH,
        {
            "honest_verdict": "complete: verifier_feedback_controller_ready",
            "verifier_feedback_controller_ready": True,
            "continuous_self_learning_task": True,
            "independent_self_learning_boundary_preserved": True,
            "n_replay_items": 68,
            "heldout_delta": 0.5,
            "negative_control_delta": 0.0,
            "forgetting_guard_passed": True,
            "drift_guard_passed": True,
            "tautology_risk_flag": False,
            "native_llm_training_claim_made": False,
        },
    )
    _write_json(
        root,
        mod.EXP3021_REL_PATH,
        {
            "honest_verdict": "complete: blocked_gatemate_transport_pinout_missing",
            "gatemate_transport_rtl_ready": False,
            "host_visible_io_plan_ready": False,
            "preconditions_checked": True,
            "board_detected": True,
            "simulation_or_lint_passed": True,
            "pnr_or_synthesis_attempted": False,
            "io_transport_path": "blocked:gatemate_pinout_missing",
            "rtl_paths": ["hardware/gatemate/ising_n16_gatemate.v"],
            "ccf_paths": ["hardware/gatemate/ising_n16_gatemate.ccf"],
            "transcript_paths": ["logs/exp3021/yosys.txt"],
            "sampler_claim_made": False,
            "speedup_claim_made": False,
            "inference_substrate": "hardware_transport_preflight",
        },
    )
    _write_json(
        root,
        mod.EXP3022_REL_PATH,
        {
            "experiment": 3022,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp3021-gatemate-rtl-ccf-host-visible-transport-shim",
                    "artifact_field": "gatemate_transport_rtl_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3023_REL_PATH,
        {
            "honest_verdict": "complete: ssqa_gate_skipped_exp3022_host_visible_io_not_ready",
            "ssqa_artifact_written": True,
            "ssqa_gate_status": "gate_skipped",
            "ssqa_rtl_pnr_report_ready": False,
            "preconditions_checked": True,
            "upstream_host_visible_io_ready": False,
            "rtl_path": "hardware/gatemate/ising_n16_gatemate.v",
            "pnr_report_path": "",
            "resource_report_path": "",
            "smoke_hook_paths": ["hardware/gatemate/ising_n16_gatemate_test_vector.json"],
            "projection_only": True,
            "sampler_claim_made": False,
            "speedup_claim_made": False,
            "inference_substrate": "hardware_gate_artifact",
        },
    )


def _row_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in artifact["rows"]}


def test_req_report_3024_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3024: OpenSpec declares the matrix v17 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3024" in spec
    assert "SCENARIO-REPORT-3024" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3024_builds_v17_from_283_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3024: v17 aggregates .283 without live metadata leakage."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    rows = _row_by_id(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v17_ready"] is True
    assert artifact["honest_verdict"].startswith("complete: matrix_v17_ready=true")
    assert artifact["milestone"] == "2026.05.283"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["row_count"] == 19
    assert artifact["clean_count"] == 4
    assert artifact["flagged_count"] == 6
    assert artifact["blocked_count"] == 2
    assert artifact["gated_skipped_count"] == 3
    assert artifact["pilot_only_count"] == 1
    assert artifact["projection_only_count"] == 2
    assert artifact["missing_count"] == 1

    assert rows["exp3016_repair_acceptance_controller"]["status"] == "flagged"
    assert rows["exp3020_fr11_verifier_feedback_controller"]["status"] == "clean"
    assert rows["exp3022_gatemate_transport_flash_smoke"]["status"] == "gated-skipped"
    assert rows["exp3023_ssqa_explicit_gate_artifact"]["status"] == "gated-skipped"
    assert rows["exp3023_ssqa_explicit_gate_artifact"]["summary"]["ssqa_artifact_written"] is True

    assert artifact["claim_rows"]["exp3016_repair"]["status"] == "flagged"
    assert artifact["claim_rows"]["exp3020_fr11_self_learning"]["status"] == "clean"
    assert artifact["claim_rows"]["exp3022_gatemate_io"]["status"] == "gated-skipped"
    assert artifact["claim_rows"]["exp3023_ssqa"]["status"] == "gated-skipped"
    assert "exp3020_fr11_verifier_feedback_controller" in artifact["repaired_claims"]
    assert "exp3023_ssqa_artifact_presence_repaired_gate_skipped_not_promotable" in artifact[
        "repaired_claims"
    ]
    assert "exp3016_repair_acceptance_controller_flagged" in artifact["still_blocked_claims"]
    assert "exp3022_gatemate_transport_flash_smoke_gated_skipped" in artifact[
        "still_blocked_claims"
    ]
    assert artifact["claim_boundary_violations"] == []

    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    serialized_citations = json.dumps(artifact["cited_upstream_artifacts"], sort_keys=True)
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in serialized_citations
    artifact_without_citations = dict(artifact)
    artifact_without_citations.pop("cited_upstream_artifacts")
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" not in json.dumps(
        artifact_without_citations,
        sort_keys=True,
    )
    assert artifact["source_checksums"][mod.EXP3020_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3020_REL_PATH
    )


def test_req_report_3024_blocks_when_required_sources_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3024: required carry-forward sources fail closed."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V16_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["matrix_v17_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp3010",
            "path": mod.MATRIX_V16_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_3024_hardware_claim_violation_overrides_gate_skip(tmp_path: Path) -> None:
    """REQ-REPORT-3024: unsupported hardware claims are boundary violations."""

    _write_ready_sources(tmp_path)
    payload = json.loads((tmp_path / mod.EXP3023_REL_PATH).read_text(encoding="utf-8"))
    payload["speedup_claim_made"] = True
    _write_json(tmp_path, mod.EXP3023_REL_PATH, payload)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    rows = _row_by_id(artifact)

    assert rows["exp3023_ssqa_explicit_gate_artifact"]["status"] == "flagged"
    assert rows["exp3023_ssqa_explicit_gate_artifact"]["claim_boundary_guard_passed"] is False
    assert artifact["claim_boundary_violations"] == [
        {
            "row_id": "exp3023_ssqa_explicit_gate_artifact",
            "violation": "unsupported_hardware_claim",
            "fields": ["speedup_claim_made"],
        }
    ]


def test_req_report_3024_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3024: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v17_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    assert saved["row_count"] == len(saved["rows"])


def test_req_report_3024_helper_edges_keep_classification_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3024: helpers keep absent, malformed, and flagged inputs honest."""

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
    assert mod._blocked_verdict("complete: blocked_pinout_missing") is True
    assert mod._blocked_verdict("complete: ok") is False
    assert mod._gated_verdict("gate_blocked_upstream") is True
    assert mod._gated_verdict("blocked_gate_check_failed") is True
    assert mod._gated_verdict("complete: ok") is False
    assert mod._flagged_verdict("flagged: check") is True
    assert mod._flagged_verdict("complete_flagged_tautology") is True
    assert mod._flagged_verdict("complete: ok") is False
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [{"kind": "X"}]}) is True
    assert mod._has_flags({}) is False
    assert mod._status_with_guards("clean", {"honest_verdict": "blocked_x"}, []) == "blocked"
    assert mod._status_with_guards("clean", {"honest_verdict": "blocked_gate_check_failed"}, []) == (
        "gated-skipped"
    )
    assert mod._status_with_guards("gated-skipped", {}, [{"violation": "x"}]) == "flagged"
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._string_list("x") == []
    assert mod._mapping([]) == {}
    assert mod._source_model_provenance({}) == {}
    assert mod._source_hardware_provenance({}) == {}
    assert mod._self_learning_boundary({})["status"] == "not_applicable"
    assert mod._self_learning_boundary(
        {
            "continuous_self_learning_task": True,
            "independent_self_learning_boundary_preserved": True,
            "heldout_delta": 1.0,
            "negative_control_delta": 0.0,
            "forgetting_guard_passed": True,
            "drift_guard_passed": True,
            "tautology_risk_flag": False,
            "native_llm_training_claim_made": False,
        }
    )["status"] == "clean"
    carried = mod._prior_v16_rows({"rows": [1, {"row_id": "bad", "status": "unknown"}]})
    assert carried[0]["row_id"] == "carry_forward_v16:bad"
    assert carried[0]["status"] == "missing"
    assert mod._claim_violations("row", {"llm_judge_used": True}) == [
        {
            "row_id": "row",
            "violation": "llm_as_verifier_boundary_violation",
            "fields": ["llm_judge_used"],
        }
    ]
    assert mod._exp3023_row({}, {"status": "blocked"})["status"] == "missing"
