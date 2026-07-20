"""Tests for the Exp5743 V513 transition receipt.

Spec refs: REQ-REPORT-5743, SCENARIO-REPORT-5743,
SCENARIO-REPORT-5743-DEPENDENCY-MAP,
SCENARIO-REPORT-5743-ARC-GATE-SCHEMA,
SCENARIO-REPORT-5743-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
import pytest

from carnot import experiment_5743_transition_v513 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id in mod.EXPECTED_TASK_IDS:
        row: JsonDict = {"id": task_id, "deliverable": f"results/{task_id}.json"}
        if task_id == "exp5747-sota-exact-proposal-utility-panel":
            row["gated_on"] = [
                {
                    "upstream": "exp5746-exact-proposal-utility-benchmark",
                    "artifact_field": "benchmark_ready_score",
                    "op": ">=",
                    "value": 1.0,
                    "principle": "benchmark readiness before proposal utility",
                }
            ]
        if task_id == "exp5748-selective-exact-feedback-search":
            row["gated_on"] = [
                {
                    "upstream": "exp5747-sota-exact-proposal-utility-panel",
                    "artifact_field": "overall_proposal_utility_positive",
                    "op": "==",
                    "value": True,
                    "principle": "feedback only after one-shot utility",
                }
            ]
        if task_id == "exp5750-dependent-task-continuous-self-learning":
            row["gated_on"] = [
                {
                    "upstream": "exp5749-csl-render-matched-mechanism-audit",
                    "artifact_field": "kan_mechanism_residual",
                    "op": ">",
                    "value": 0.0,
                    "principle": "scale only if KAN residual is positive",
                }
            ]
        if task_id == "exp5752-allocation-free-rust-python-10x-or-retire-benchmark":
            row["gated_on"] = [
                {
                    "upstream": "exp5751-rust-restart-parity-repair",
                    "artifact_field": "restart_parity_ready_score",
                    "op": ">=",
                    "value": 1.0,
                    "principle": "timing after restart parity",
                }
            ]
        if task_id == "exp5753-arc-generic-primitive-live-registry-ab":
            row["gated_on"] = [
                {
                    "upstream": "exp5745-arc-causal-gate-schema-corrigendum",
                    "artifact_field": "counterfactual_receipt_coverage_score",
                    "op": "==",
                    "value": 1.0,
                    "principle": "live A/B after scalar coverage normalization",
                }
            ]
        tasks.append(row)
    return {
        "milestone": mod.CURRENT_MILESTONE,
        "milestone_title": "Decision-Useful Exact Proposals",
        "tasks": tasks,
    }


def _artifact_payloads() -> dict[Path, JsonDict]:
    primitive_rows = [
        {"primitive": "object_displacement", "composite_utility_delta": 0.170709},
        {"primitive": "reversible_or_noop_action", "composite_utility_delta": 0.154099},
        {"primitive": "boundary_or_collision", "composite_utility_delta": 0.217492},
        {"primitive": "inventory_or_state_toggle", "composite_utility_delta": 0.176004},
        {"primitive": "agent_relative_motion", "composite_utility_delta": 0.173253},
        {"primitive": "repeated_action_loop", "composite_utility_delta": 0.183333},
        {"primitive": "delayed_effect", "composite_utility_delta": 0.171722},
    ]
    gate_rows = [
        {
            "upstream": "exp5740-arc-game-blind-primitive-causal-audit",
            "artifact_field": "positive_causal_primitive_count",
            "op": ">=",
            "expected": 1,
            "actual": 7,
            "passed": True,
        },
        {
            "upstream": "exp5740-arc-game-blind-primitive-causal-audit",
            "artifact_field": "source_leak_count",
            "op": "==",
            "expected": 0,
            "actual": 1,
            "passed": False,
        },
        {
            "upstream": "exp5740-arc-game-blind-primitive-causal-audit",
            "artifact_field": "game_identity_leak_count",
            "op": "==",
            "expected": 0,
            "actual": 2,
            "passed": False,
        },
        {
            "upstream": "exp5740-arc-game-blind-primitive-causal-audit",
            "artifact_field": "counterfactual_receipt_coverage",
            "op": "==",
            "expected": 1.0,
            "actual": {
                "candidate_count": 7,
                "meets_minimum_n": True,
                "minimum_positive_candidate_paired_replay_count": 322,
                "paired_replay_count": 20759,
                "trace_step_count": 9975,
            },
            "passed": False,
        },
    ]
    return {
        mod.EXP5731_TRANSITION_PATH: {
            "honest_verdict": "complete: v512 transition archived terminal .511 evidence",
            "current_task_range": "exp5731-exp5742",
        },
        mod.EXP5732_SOURCE_PATH: {
            "honest_verdict": "complete: accepted 3 non-duplicate actionable V512 source deltas; no roadmap ID, gate, benchmark, or hardware claim changed",
            "flagged_adversarial": True,
            "duration_s": 0.049658,
            "inference_substrate": "web_and_bibliographic_search_only",
            "accepted_findings": [{}, {}, {}],
            "benchmark_compute_claimed": False,
        },
        mod.EXP5733_CHANNEL_PATH: {
            "honest_verdict": "complete: sealed_finite_choice_proposal_channel_qualified",
            "proposal_channel_ready_score": 1.0,
            "positive_control_count": 30,
            "negative_control_count": 12,
            "receipt_failure_count": 0,
            "label_collision_count": 0,
            "validator_disagreement_count": 0,
            "model_accuracy": {
                "unsloth/Qwen3.6-35B-A3B-GGUF": 0.47619,
                "unsloth/gemma-4-26B-A4B-it-GGUF": 0.571429,
                "unsloth/gemma-4-31B-it-GGUF": 0.52381,
            },
        },
        mod.EXP5734_STREAM_PATH: {
            "honest_verdict": "complete: sealed_chronological_sota_exact_proposal_stream_ready",
            "sota_proposal_stream_ready_score": 1.0,
            "row_count": 96,
            "proposal_conflict_count": 53,
            "validator_disagreement_count": 0,
        },
        mod.EXP5735_CSL_PATH: {
            "honest_verdict": "complete: zero_gated_residual_spline_kan_csl_ready",
            "function_preserving_insertion_score": 1.0,
            "zero_gate_csl_ready_score": 1.0,
            "unsafe_update_count": 0,
            "max_update_latency_ms": 0.25,
            "arm_metrics": {
                "zero_gated_residual_spline_growth": {"suffix_error": 0.146067},
                "parameter_matched_mlp_residual": {"suffix_error": 0.061798},
            },
        },
        mod.EXP5736_LIFECYCLE_PATH: {
            "honest_verdict": "complete: csl_lifecycle_conflict_rollback_ready",
            "csl_lifecycle_ready_score": 1.0,
            "operation_counts": {"total": 74},
            "rollback_state_hash_matches": True,
            "unsafe_propagation_count": 0,
        },
        mod.EXP5737_SOTA_CSL_PATH: {
            "honest_verdict": "complete: sota_stream_csl_shadow_ingress_ready",
            "sota_csl_ingress_ready_score": 1.0,
        },
        mod.EXP5738_BATCH_PATH: {
            "honest_verdict": "complete: one-axis sample_batch backend is semantically and distributionally ready; no timing, software speedup, hardware, FPGA, or TSU claim",
            "batch_backend_ready_score": 1.0,
            "energy_trace_mismatch_count": 0,
            "checkpoint_mismatch_count": 0,
            "restart_mismatch_count": 0,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5739_10X_PATH: {
            "honest_verdict": "complete: terminal null; matched batched Rust/Python CPU evidence did not prove the strict consecutive larger-size 10x lower-bound rule",
            "quality_matched_pair_count": 728,
            "paired_speedup_intervals": [
                {"size": 48, "quality_matched": True},
                {"size": 96, "quality_matched": False},
                {"size": 192, "quality_matched": False},
            ],
            "excluded_pair_reasons": [{"reason": "restart_match", "count": 352}],
            "rust_batched_10x_ready_score": 0.0,
            "software_speedup_claimed": False,
            "hardware_speedup_claimed": False,
            "timing_claimed": True,
        },
        mod.EXP5740_ARC_PATH: {
            "honest_verdict": "complete: game_blind_primitive_causal_audit_positive_count_7_no_policy_or_registry_credit",
            "positive_causal_primitive_count": 7,
            "source_leak_count": 1,
            "game_identity_leak_count": 2,
            "counterfactual_receipt_coverage": gate_rows[3]["actual"],
            "primitive_candidates": primitive_rows,
            "registry_modified": False,
            "policy_modified": False,
            "solve_provenance": "development_proxy",
        },
        mod.EXP5741_ARC_AB_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "3 of 4 gate(s) failed; first failure: exp5740-arc-game-blind-primitive-causal-audit.source_leak_count (actual=1 == expected=0)",
            "gates_evaluated": gate_rows,
            "duration_s": 0.0,
        },
        mod.EXP5742_CAPSTONE_PATH: {
            "honest_verdict": "complete: v512 reconciled; proposal_channel_ready=true; sota_proposal_stream_ready=true; continuous_self_learning_credited=true; batch_backend_ready=true; rust_batched_10x_ready=false; arc_registry_delta=0; arc_solve_credited=false",
            "proposal_channel_ready": True,
            "sota_proposal_stream_ready": True,
            "continuous_self_learning_credited": True,
            "batch_backend_ready": True,
            "rust_batched_10x_ready": False,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
            "gate_skip_receipts": {
                "exp5741-arc-generic-primitive-live-ab": {
                    "honest_verdict": "blocked_gate_check_failed",
                    "blocked_at_layer": "conductor_pre_gate",
                    "gate_check_summary": "3 of 4 gate(s) failed",
                    "gates_evaluated": gate_rows,
                }
            },
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    for rel_path, payload in _artifact_payloads().items():
        if rel_path == omit:
            continue
        _write_json(root, rel_path, payload)
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            _write_text(root, rel_path, yaml.safe_dump(_roadmap_payload(), sort_keys=False))
        elif rel_path == mod.VNEXT_RELATIVE_PATH:
            _write_text(root, rel_path, "**Task range:** `exp5743`-`exp5754`\n")
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                "\n".join(
                    [
                        "| t | Transition terminal .511 evidence | OK | done |",
                        "| t | Ingest post-V512 2025-2026 source deltas | FLAGGED | DURATION_TOO_SHORT |",
                        "| t | Qualify a sealed finite-choice SOTA proposal channel | OK | done |",
                        "| t | Build exact-attested SOTA proposal stream | OK | done |",
                        "| t | Insert zero-gated KAN residual sidecar | OK | done |",
                        "| t | Exercise CSL lifecycle conflict rollback | OK | done |",
                        "| t | Shadow-ingest the exact SOTA stream | OK | done |",
                        "| t | Expose one-axis Rust sample_batch backend | OK | done |",
                        "| t | Run one-axis batched 10x crossover benchmark | OK | terminal null |",
                        "| t | Audit ARC game-blind primitive causality | OK | done |",
                        "| t | Gated on Exp5740 causal primitive readiness | GATE_BLOCK | 3 failed |",
                        "| t | Reconcile .512 | OK | done |",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.RESEARCH_COMPLETE_RELATIVE_PATH:
            _write_text(root, rel_path, "duplicated historical block\nexp5742-v512-capstone\n")
        elif rel_path == mod.CONDUCTOR_RELATIVE_PATH:
            _write_text(root, rel_path, "# conductor fixture\n")
        else:
            _write_text(root, rel_path)


def test_spec_contains_req_report_5743() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-5743" in text
    assert "SCENARIO-REPORT-5743-ARC-GATE-SCHEMA" in text
    assert "experiment_5743_transition_v513.py" in text


def test_build_report_preserves_v512_terminal_evidence(tmp_path: Path) -> None:
    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        tests_run=[
            {"command": "unit", "exit_code": 0},
            {"command": "coverage", "exit_code": 0},
        ],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["honest_verdict"].startswith("complete:")
    assert report["proposal_channel_ready"] is True
    assert report["sota_proposal_stream_ready"] is True
    assert report["continuous_self_learning_credited"] is True
    assert report["batch_backend_ready"] is True
    assert report["rust_batched_10x_ready"] is False
    assert report["arc_registry_delta"] == 0
    assert report["arc_solve_credited"] is False
    assert report["proposal_conflict_count"] == 53
    assert report["kan_suffix_error"] == 0.146067
    assert report["mlp_suffix_error"] == 0.061798
    assert report["restart_exclusion_count"] == 352
    assert report["current_task_range"] == "exp5743-exp5754"
    assert report["timing_claimed"] is False
    assert report["hardware_speedup_claimed"] is False
    assert report["inference_substrate"] == "artifact_reconciliation_only"

    assert report["source_delta_duration_flag"]["flagged_adversarial"] is True
    assert report["source_delta_duration_flag"]["duration_s"] == 0.049658
    assert report["proposal_channel_evidence"]["control_count"] == 42
    assert report["proposal_channel_evidence"]["model_accuracy_range"] == [0.47619, 0.571429]
    assert report["proposal_channel_evidence"]["utility_claimed"] is False
    assert report["sota_stream_evidence"]["science_row_count"] == 96
    assert report["rust_batched_null_evidence"]["quality_matched_pair_count"] == 728
    assert report["rust_batched_null_evidence"]["qualified_large_size_count"] == 1

    arc_issue = report["arc_gate_schema_issue"]
    assert arc_issue["source_leak_count_field"] == 1
    assert arc_issue["game_identity_leak_count_field"] == 2
    assert arc_issue["coverage_field_type"] == "object"
    assert arc_issue["live_ab_ran"] is False
    assert report["arc_causal_effects_preserved"]["positive_causal_primitive_count"] == 7
    assert report["v512_task_verdicts"]["exp5741-arc-generic-primitive-live-ab"]["status"] == "gate_skipped"
    assert report["v512_conductor_outcomes"]["exp5741-arc-generic-primitive-live-ab"]["outcome"] == "GATE_BLOCK"
    assert report["dependency_chain_retired_id_check"]["valid"] is True
    assert report["protected_files"]["research-roadmap.yaml"]["modified_by_transition"] is False
    assert report["protected_files"]["scripts/research_conductor.py"]["modified_by_transition"] is False
    assert report["completion_ledger_duplicate_blocks_preserved"] is True
    assert report["reproducibility_checksum"]

    assert set(report).issubset(report["field_principles"])
    assert all(report["field_principles"][field] for field in report)


def test_missing_capstone_blocks_transition(tmp_path: Path) -> None:
    _make_root(tmp_path, omit=mod.EXP5742_CAPSTONE_PATH)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["honest_verdict"].startswith("blocked:")
    assert f"missing_artifact:{mod.EXP5742_CAPSTONE_PATH.as_posix()}" in report["failed_preconditions"]
    assert report["source_capstone_hash"] is None


def test_defensive_helpers_cover_schema_edges(tmp_path: Path) -> None:
    assert mod._float_value({"x": True}, "x") == 1.0
    assert mod._float_value({"x": "2.5"}, "x") == 2.5
    assert mod._float_value({"x": "bad"}, "x", 7.0) == 7.0
    assert mod._float_value({}, "x", 3.0) == 3.0
    assert mod._nested({"a": 1}, "a", "b") is None
    assert mod._extract_outcome(None) == "MISSING_LOG_LINE"
    assert mod._extract_outcome("| t | task | WEIRD | detail |") == "LOGGED"
    assert mod._fallback_outcome("unknown-status") == "UNKNOWN"
    assert mod._model_accuracy_range({"model_accuracy": []}) == []

    fallback_channel = mod._proposal_channel_evidence(
        {
            "score_vector_row_count": 126,
            "model_accuracy": {"a": 0.5, "b": 0.75, "c": 0.25},
        }
    )
    assert fallback_channel["control_count"] == 42
    fallback_stream = mod._sota_stream_evidence({"score_vector_hashes": ["a", "b"]})
    assert fallback_stream["science_row_count"] == 2
    assert mod._restart_exclusion_count({"excluded_pair_reasons": "bad"}) == 0
    assert mod._qualified_large_size_count({"paired_speedup_intervals": "bad"}) == 0
    assert mod._coverage_type(1.0) == "number"
    assert mod._coverage_type(None) == "missing"
    assert mod._coverage_type("scalar") == "str"

    retired = mod._dependency_retired_id_check(
        {"exp5744-v513-source-delta-ingestion": {"depends_on": ["exp5739-one-axis-batched-10x-crossover"]}},
        {"exp5745-arc-causal-gate-schema-corrigendum": [{"upstream": "exp9999-missing"}]},
    )
    assert retired["valid"] is False
    assert len(retired["retired_or_unknown_references"]) == 2

    (tmp_path / "results").mkdir()
    _write_text(tmp_path, "results/experiment_5744_prior_collision.json")
    _write_text(tmp_path, "python/carnot/experiment_5743_transition_v513.py")
    _write_text(tmp_path, "python/carnot/__pycache__/experiment_5743_transition_v513.pyc")
    collision = mod._collision_check(tmp_path, mod.EXPECTED_TASK_IDS)
    assert collision["collision_free"] is False
    assert "results/experiment_5744_prior_collision.json" in collision["path_collisions"]
    assert "python/carnot/experiment_5743_transition_v513.py" not in collision["path_collisions"]
    assert not any("__pycache__" in path for path in collision["path_collisions"])


def test_validation_loading_and_emit_report(tmp_path: Path) -> None:
    _make_root(tmp_path)
    assert mod._load_tests_run(None)[0]["status"] == "not_run"

    valid = tmp_path / "valid-tests.json"
    valid.write_text(json.dumps([{"command": "unit", "exit_code": 0}]), encoding="utf-8")
    assert mod._load_tests_run(valid) == [{"command": "unit", "exit_code": 0}]

    invalid = tmp_path / "invalid-tests.json"
    invalid.write_text(json.dumps({"command": "unit"}), encoding="utf-8")
    with pytest.raises(ValueError, match="validation results"):
        mod._load_tests_run(invalid)

    output = tmp_path / "out" / "transition.json"
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["reproducibility_checksum"]
    assert report["status"] == "complete"


def test_failed_precondition_helper_lists_gate_and_protected_failures() -> None:
    failures = mod._failed_preconditions(
        ["missing.json"],
        ["bad.json"],
        {"milestone": "wrong"},
        {"range_matches_roadmap": False, "collision_free": False},
        {"valid": False},
        {"research-roadmap.yaml": {"modified_by_transition": True}},
        {
            "proposal_channel_ready": False,
            "sota_proposal_stream_ready": False,
            "continuous_self_learning_credited": False,
            "batch_backend_ready": False,
            "rust_batched_10x_ready": True,
            "arc_registry_delta": 1,
            "arc_solve_credited": True,
        },
    )
    assert "missing_artifact:missing.json" in failures
    assert "malformed_artifact:bad.json" in failures
    assert "roadmap_milestone_not_2026.07.513" in failures
    assert "roadmap_task_range_mismatch" in failures
    assert "current_task_id_collision" in failures
    assert "retired_or_unknown_dependency_reference" in failures
    assert "protected_file_modified:research-roadmap.yaml" in failures
