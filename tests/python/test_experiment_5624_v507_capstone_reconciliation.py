"""Tests for Exp5624 V507 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5624, SCENARIO-CAPSTONE-5624,
SCENARIO-CAPSTONE-5624-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5624-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5624_v507_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _payloads() -> dict[Path, Any]:
    return {
        Path("results/experiment_5613_transition_v507.json"): {
            "schema": "carnot.experiment_5613.transition_v507.v1",
            "experiment_id": "exp5613-transition-v507",
            "status": "complete",
            "current_task_range": "exp5613-exp5624",
            "honest_verdict": "complete: transition",
            "promoted_substrates": [{"key": "lossless_response_envelope"}],
            "retired_scopes": [{"key": "unmatched_cdls_crossover"}],
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5614_v507_source_delta_ingestion.json"): {
            "schema": "carnot.experiment_5614.v507_source_delta_ingestion.v1",
            "experiment_id": "exp5614-v507-source-delta-ingestion",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: no new non-duplicate actionable V507 source deltas",
            "planner_marker_found": True,
            "new_references_added": [],
            "duplicates_suppressed": [{"source_id": "retain_or_adapt_2607_05609"}],
            "closed_scopes_reopened": False,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5615_native_llamacpp_cuda_runtime_certificate.json"): {
            "schema": "carnot.experiment_5615.native_llamacpp_cuda_runtime_certificate.v507",
            "experiment_id": "exp5615-native-llamacpp-cuda-runtime-certificate",
            "honest_verdict": "blocked_native_cuda_runtime_certificate_failed_terminal_retirement_evidence",
            "inference_substrate": "live_llm_inference",
            "models_certified_count": 0,
            "models_certified_denominator": 3,
            "runtime_certificate_ready_score": 0.0,
            "lossless_replay_rate": 1.0,
            "stop_control_pass_rate": 1.0,
            "semantic_false_accept_count": 0,
            "orphan_process_count": 0,
            "cuda_build_capability": {"native_cuda_ready": True},
            "offload_layers_by_model": {
                "qwen": {"requested": "all", "observed": 0},
                "gemma31": {"requested": "all", "observed": 0},
                "gemma26": {"requested": "all", "observed": 0},
            },
            "gpu_memory_delta_by_model": {"qwen": 1, "gemma31": 1, "gemma26": 1},
            "no_task_accuracy_computed": True,
            "solve_verify_accuracy_inferred": False,
        },
        Path("results/experiment_5616_exact_nonstationary_constraint_stream.json"): {
            "schema": "carnot.experiment_5616.exact_nonstationary_constraint_stream.v1",
            "experiment_id": "exp5616-exact-nonstationary-constraint-stream",
            "honest_verdict": "complete: fixture ready",
            "fixture_ready_score": 1.0,
            "oracle_label_error_count": 0,
            "exact_oracle_label_count": 107136,
            "dataset_path": "data/research/experiment_5616_exact_nonstationary_constraint_stream.jsonl",
            "dataset_row_count": 17856,
            "task_durations": [1, 2, 4, 8, 16, 32],
            "instances_per_condition": 32,
            "inference_substrate": "deterministic_verifier",
        },
        Path("results/experiment_5617_kan_critical_task_duration_map.json"): {
            "schema": "carnot.experiment_5617.kan_critical_task_duration_map.v1",
            "experiment_id": "experiment_5617_kan_critical_task_duration_map",
            "honest_verdict": "complete: critical_task_duration_d16_estimated",
            "fixture_hash": "sha256:fixture",
            "duration_cells": [1, 2, 4, 8, 16, 32],
            "critical_task_duration": 16,
            "critical_duration_fit_r2": 0.0,
            "nondegenerate_switch_cases": [{}, {}],
            "unsafe_false_accept_count": {"total": 0},
            "lazy_identity_guard_passed": True,
            "llm_invoked": False,
            "llm_weight_training": False,
            "inference_substrate": "exact_constraint_stream_active_spline_kan_no_llm",
        },
        Path("results/experiment_5618_predictive_window_kan_self_learning.json"): {
            "schema": "carnot.experiment_5618.predictive_window_kan_self_learning.v1",
            "experiment_id": "experiment_5618_predictive_window_kan_self_learning",
            "honest_verdict": "complete: predictive_window_active_spline_kan_self_learning_ready",
            "continuous_self_learning_ready": True,
            "controller_gate_receipt": {"adaptive_ale_beats_best_fixed": True},
            "delta_ale_vs_best_fixed": {"mean": 0.150694},
            "forward_transfer_delta": {"mean": 0.345833},
            "backward_retention_delta": {"mean": 0.165972},
            "forgetting_delta": {"mean": -0.165972},
            "unsafe_false_accept_count": {"total": 0},
            "rollback_positive_control": {"passed": True},
            "delayed_regression_passed": True,
            "no_model_weight_mutation": True,
            "kan_spline_state_mutated": True,
            "poison_update_disposition": {"rejected": 1152, "rolled_back": 1},
            "llm_invoked": False,
            "llm_weight_training": False,
            "inference_substrate": "exact_constraint_stream_active_spline_kan_no_llm",
        },
        Path("results/experiment_5619_arc_forward_inverse_transition_cycle.json"): {
            "schema": "carnot.exp5619.arc_forward_inverse_transition_cycle.v1",
            "experiment_id": 5619,
            "honest_verdict": "complete: transition_cycle_verifier_safe_over_abstaining_not_useful_terminal",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "solve_provenance": "development_proxy",
            "cycle_verifier_positive_control_rate": 0.009259,
            "corruption_reject_rate": 1.0,
            "unsafe_transition_accept_count": 0,
            "valid_transition_accept_rate": 0.009259,
            "abstention_rate": 0.75,
            "source_files_read": False,
            "per_game_adapter_used": False,
            "inference_substrate": "offline_arcade_live_agent_runtime_filters_no_new_llm",
        },
        Path("results/experiment_5620_arc_cycle_guarded_live_update_ab.json"): {
            "schema": "blocked_gate_check_v1",
            "experiment": 5620,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [{"passed": False}, {"passed": True}],
        },
        Path("results/experiment_5621_arc_live_self_discovery_levelup_v507.json"): {
            "schema": "arc_live_self_discovery_levelup_attempt.v2",
            "experiment_id": 5621,
            "honest_verdict": "complete: no_new_arc_level_banked_bp35_L9_bounded_live_attempt_v507",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "solve_provenance": "live_agent_self_discovery",
            "live_attempt_executed": True,
            "live_branch_configuration": {"baseline_unchanged": True, "source_status": "blocked"},
            "target_selection_receipt": {"selected_game": "bp35", "selected_level": "L9"},
            "levels_before": 177,
            "levels_after": 177,
            "new_reproducible_levels": [],
            "offline_reproduced": False,
            "registry_updated": False,
            "target_reached_live": False,
            "source_files_read": False,
            "per_game_adapter_used": False,
            "llm_invoked": False,
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        Path("results/experiment_5622_cdls_exact_kernel_audit.json"): {
            "schema": "carnot.experiment_5622.cdls_exact_kernel_audit.v1",
            "experiment_id": "exp5622-cdls-exact-kernel-audit",
            "honest_verdict": "complete: corrected cDLS exact kernel audit ready",
            "correction_applied": True,
            "kernel_audit_ready_score": 1.0,
            "transition_row_sum_error_max": 4e-16,
            "detailed_balance_residual_max": 5e-18,
            "exact_distribution_tv_max": 5e-14,
            "energy_histogram_tv_max": 5e-14,
            "broken_kernel_controls_rejected": True,
            "quality_gate_specified_count": 4,
            "inference_substrate": "deterministic_verifier",
        },
        Path("results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json"): {
            "schema": "carnot.experiment_5623.cdls_multiseed_cpu_cuda_crossover.v1",
            "experiment_id": "exp5623-cdls-multiseed-cpu-cuda-crossover",
            "honest_verdict": "complete: no quality-matched crossover pairs entered speedups",
            "upstream_gate_receipt": {"ready": True, "kernel_audit_ready_score": 1.0},
            "preconditions": {"blocked_reasons": [], "cpu_available": True, "cuda_available": True},
            "seeds": [5623, 5624, 5625, 5626, 5627],
            "instance_sizes": [128, 256, 512, 1024],
            "samples_per_pair": 10000,
            "quality_gate_results_by_pair": [
                {"included_in_speedups": False, "exclusion_reason": "quality_gate_failed"}
            ],
            "successful_matched_pairs": [],
            "crossover_claim_allowed": False,
            "crossover_size": None,
            "board_speedup_claimed": False,
            "timing_intervals_by_size": [],
            "inference_substrate": "matched_cpu_cuda_exact_ising_sampling",
        },
    }


def _make_root(
    root: Path,
    *,
    omit: Path | None = None,
    malformed: Path | None = None,
) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        _write_text(root, rel_path)
    for rel_path, payload in _payloads().items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)
    _write_text(
        root,
        "results/experiment_5615_native_llamacpp_cuda_runtime_certificate.responses.jsonl",
        "{}\n",
    )
    _write_json(
        root,
        "results/experiment_5621_arc_live_self_discovery_levelup_v507_trace.json",
        {"trace": []},
    )
    _write_json(
        root,
        "results/experiment_5623_cdls_multiseed_cpu_cuda_crossover_sufficient_statistics.json",
        {"schema": "stats"},
    )


def test_req_capstone_5624_spec_declares_v507_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5624: OpenSpec declares the V507 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5624") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in (*mod.PRIMARY_ARTIFACT_PATHS, *mod.SIDECAR_ARTIFACT_PATHS):
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5624_live_repo_preserves_terminal_statuses() -> None:
    """SCENARIO-CAPSTONE-5624: live V507 evidence stays narrow and terminal."""

    artifact = mod.run_capstone(
        root=REPO,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    statuses = artifact["terminal_status_by_task"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["expected_task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert artifact["missing_tasks"] == []
    assert artifact["malformed_tasks"] == []
    assert statuses["exp5615-native-llamacpp-cuda-runtime-certificate"]["status"] == "blocked"
    assert statuses["exp5619-arc-forward-inverse-transition-cycle"]["status"] == "flagged"
    assert statuses["exp5620-arc-cycle-guarded-live-update-ab"]["status"] == "gate_skipped"
    assert statuses["exp5621-arc-live-self-discovery-levelup-v507"]["status"] == "flagged"

    assert artifact["native_runtime_verdict"]["certificate_ready"] is False
    assert artifact["native_runtime_verdict"]["models_certified_count"] == 0
    assert artifact["headline_claims"]["exact_drift_fixture"]["claim_allowed"] is True
    assert artifact["headline_claims"]["critical_duration_map"]["claim_allowed"] is True
    assert artifact["headline_claims"]["predictive_kan_self_learning"]["claim_allowed"] is False
    assert artifact["headline_claims"]["arc_transition_cycle"]["claim_allowed"] is False
    assert artifact["headline_claims"]["arc_new_registry_levels"]["claim_allowed"] is False
    assert artifact["headline_claims"]["cdls_exact_kernel"]["claim_allowed"] is True
    assert artifact["headline_claims"]["cdls_cpu_cuda_crossover"]["claim_allowed"] is False
    assert (
        artifact["promotion_decisions"]["predictive_window_kan_self_learning"]["decision"]
        == "do_not_promote_preregistered_gate_failed"
    )
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_registry_delta_evidence"]["selected_game"] == "bp35"
    assert artifact["continuous_self_learning_verdict"]["artifact_reports_ready"] is True
    assert artifact["continuous_self_learning_verdict"]["promotion_allowed"] is False
    assert artifact["hardware_sampling_verdict"]["exact_kernel_ready"] is True
    assert artifact["hardware_sampling_verdict"]["crossover_claim_allowed"] is False
    assert artifact["documents_reconciled"]["protected_files"]["research-roadmap.yaml"] is True
    assert (
        artifact["documents_reconciled"]["protected_files"]["scripts/research_conductor.py"] is True
    )
    assert "ops/status.md" in artifact["documents_reconciled"]["delegated_by_stop_rule"]
    assert artifact["documents_reconciled"]["research_roadmap_next_missing"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5624_missing_and_malformed_inputs_block_claims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5624-MISSING-MALFORMED: bad inputs fail closed."""

    missing = Path("results/experiment_5615_native_llamacpp_cuda_runtime_certificate.json")
    malformed = Path("results/experiment_5618_predictive_window_kan_self_learning.json")
    _make_root(tmp_path, omit=missing, malformed=malformed)

    artifact = mod.run_capstone(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["malformed_artifacts"] == [malformed.as_posix()]
    assert (
        artifact["terminal_status_by_task"]["exp5615-native-llamacpp-cuda-runtime-certificate"][
            "status"
        ]
        == "missing"
    )
    assert (
        artifact["terminal_status_by_task"]["exp5618-predictive-window-kan-self-learning"]["status"]
        == "malformed"
    )
    assert artifact["native_runtime_verdict"]["certificate_ready"] is False
    assert artifact["headline_claims"]["predictive_kan_self_learning"]["claim_allowed"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5624_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5624-FIELD-PRINCIPLES: schema drift is invalid."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "expected_task_ids" in mod.validate_artifact({**artifact, "expected_task_ids": []})
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": 1})
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": "0"})
    assert "native_runtime_verdict" in mod.validate_artifact(
        {**artifact, "native_runtime_verdict": []}
    )
    assert "native_runtime_verdict" in mod.validate_artifact(
        {**artifact, "native_runtime_verdict": {"certificate_ready": "no"}}
    )
    assert "documents_reconciled" in mod.validate_artifact(
        {
            **artifact,
            "documents_reconciled": {
                **artifact["documents_reconciled"],
                "protected_files": {"research-roadmap.yaml": False},
            },
        }
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "artifacts_found" in mod.validate_artifact({**artifact, "artifacts_found": "all"})
    bad_statuses = dict(artifact["terminal_status_by_task"])
    bad_statuses.pop("exp5613-transition-v507")
    assert "terminal_status_by_task" in mod.validate_artifact(
        {**artifact, "terminal_status_by_task": bad_statuses}
    )


def test_scenario_capstone_5624_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5624-FIELD-PRINCIPLES: defensive paths stay explicit."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    payload, meta = mod._read_json_any(list_json)
    assert payload == {}
    assert meta["error"] == "not_json_object"

    missing_sidecar_payload, missing_sidecar_meta = mod._read_jsonl_sidecar(
        tmp_path / "missing.jsonl"
    )
    assert missing_sidecar_payload == {}
    assert missing_sidecar_meta["error"] == "missing"
    jsonl = tmp_path / "responses.jsonl"
    jsonl.write_text("\n{}\n", encoding="utf-8")
    assert mod._read_jsonl_sidecar(jsonl)[1]["line_count"] == 1
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("{bad\n", encoding="utf-8")
    assert mod._read_jsonl_sidecar(bad_jsonl)[1]["error"] == "malformed_jsonl"

    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "-7"}, "value") == -7
    assert mod._int({"value": "bad"}, "value") == 0
    assert mod._float({"value": "1.25"}, "value") == 1.25
    assert mod._float({"value": "bad"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert (
        mod._status_for_payload({"honest_verdict": "unclear"}, {"exists": True, "loadable": True})
        == "unknown"
    )
    assert mod._zero_false_accepts({"total": 0}) is True
    assert mod._zero_false_accepts({"total": 1}) is False
    assert mod._zero_false_accepts({"nested": [{"total": 0}, False, 0]}) is True
    assert mod._zero_false_accepts([{"total": 0}, 1]) is False
    assert mod._zero_false_accepts("missing") is False
    assert mod._count({}) == 0
    assert mod._count({"total": "3"}) == 3
    assert mod._count([{}, {}]) == 2
    assert mod._count(True) == 1
    assert mod._count(2.9) == 2
    assert mod._count("none") == 0

    assert mod._failed_gate_summary({"failed_gate_summary_by_device": {"cpu": {"tv": 1}}}) == {
        "cpu": {"tv": 1}
    }
    assert mod._failed_gate_summary({"quality_gate_results_by_pair": "bad"}) == {}
    assert mod._failed_gate_summary({"quality_gate_results_by_pair": ["bad"]}) == {}
    assert mod._failed_gate_summary(
        {"quality_gate_results_by_pair": [{"device": "cpu", "failed_gates": ["tv", "tv"]}]}
    ) == {"cpu": {"tv": 2}}

    assert mod._load_tests_run(None) == mod.DEFAULT_TESTS_RUN
    tests_run_path = tmp_path / "tests_run.json"
    tests_run_path.write_text(
        json.dumps([{"command": "unit", "exit_code": 0}, "ignored"]) + "\n",
        encoding="utf-8",
    )
    assert mod._load_tests_run(tests_run_path) == [{"command": "unit", "exit_code": 0}]
    bad_tests_run_path = tmp_path / "bad_tests_run.json"
    bad_tests_run_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_tests_run(bad_tests_run_path)


def test_scenario_capstone_5624_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5624: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []
