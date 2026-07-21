"""Tests for Exp5768 V514 capstone reconciliation.

Spec refs: REQ-REPORT-5768, SCENARIO-REPORT-5768,
SCENARIO-REPORT-5768-MISSING-OR-BLOCKED,
SCENARIO-REPORT-5768-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5768_v514_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: JsonDict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dict(payload)
    if "reproducibility_checksum" in tmp:
        tmp["reproducibility_checksum"] = ""
        tmp["reproducibility_checksum"] = mod.payload_checksum(tmp)
    path.write_text(json.dumps(tmp, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _blocked_gate(reason: str, gates: list[JsonDict]) -> JsonDict:
    return {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": reason,
        "gates_evaluated": gates,
    }


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        mod.EXP5755_PATH: {
            "schema": "carnot.experiment_5755.transition_v514.v1",
            "status": "blocked",
            "honest_verdict": "blocked: exp5755 transition preconditions failed: next_range_collision_count=3",
            "blocked_task_ids": ["exp5747-sota-exact-proposal-utility-panel"],
            "scientific_null_task_ids": ["exp5753-arc-generic-primitive-live-registry-ab"],
            "negative_result_task_ids": ["exp5749-csl-render_matched-mechanism-audit"],
            "reproducibility_checksum": "",
        },
        mod.EXP5756_PATH: {
            "schema": "carnot.experiment_5756.v514_source_delta_ingestion.v1",
            "status": "complete",
            "honest_verdict": "complete: no new non-duplicate actionable V514 source deltas; references left unchanged",
            "accepted_findings": [],
            "references_modified": False,
            "reproducibility_checksum": "",
        },
        mod.EXP5757_PATH: {
            "schema": "carnot.experiment_5757.proposal_benchmark_scalar_bridge.v1",
            "status": "complete",
            "honest_verdict": "complete: exact proposal benchmark scalars bridged for downstream gates",
            "benchmark_bridge_ready_score": 1.0,
            "benchmark_ready_score": 1.0,
            "heldout_partition_disjoint_score": 1.0,
            "adversarial_verification_clean_score": 1.0,
            "row_hash_count": 180,
            "structure_receipt_failure_count": 0,
            "solution_receipt_failure_count": 0,
            "validator_disagreement_count": 0,
            "unsafe_synthesis_count": 0,
            "upstream_modified": False,
            "llm_inference_used": False,
            "gate_replay_receipts": {"passed": True, "summary": "7 gate(s) satisfied"},
            "reproducibility_checksum": "",
        },
        mod.EXP5758_PATH: {
            "schema": "carnot.experiment_5758.rust_parity_scalar_bridge.v1",
            "status": "complete",
            "honest_verdict": "complete: Exp5751 Rust parity receipts bridged to bare scalar gates",
            "rust_benchmark_gate_ready_score": 1.0,
            "restart_parity_ready_score": 1.0,
            "distributional_parity_score": 1.0,
            "fallback_equivalence_score": 1.0,
            "production_backend_reachable_score": 1.0,
            "parity_case_count": 3,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
            "sampler_code_modified": False,
            "upstream_modified": False,
            "gate_replay_receipts": {"passed": True, "summary": "4 gate(s) satisfied"},
            "reproducibility_checksum": "",
        },
        mod.EXP5759_PATH: {
            "schema": "carnot.experiment_5759.sota_exact_proposal_utility_panel.v1",
            "status": "complete",
            "honest_verdict": "complete: sota_exact_proposal_utility_measured_gate_not_ready",
            "models_used": ["qwen", "gemma31", "gemma26"],
            "proposal_utility_delta_overall": 0.003416373531,
            "proposal_utility_lcb": -0.045291796847,
            "proposal_utility_ready_score": 0.0,
            "flagship_nonregression_count": 0,
            "validator_disagreement_count": 0,
            "authority_violation_count": 0,
            "science_row_count": 60,
            "generated_text_scoring_used": False,
            "llm_judge_used": False,
            "token_scores_are_semantic_authority": False,
            "model_weight_mutation": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "",
        },
        mod.EXP5760_PATH: _blocked_gate(
            "2 of 4 gate(s) failed; first failure: proposal_utility_lcb",
            [
                {
                    "upstream": "exp5759-sota-exact-proposal-utility-panel",
                    "artifact_field": "proposal_utility_lcb",
                    "op": ">",
                    "expected": 0.0,
                    "actual": -0.045291796847,
                    "passed": False,
                }
            ],
        ),
        mod.EXP5761_PATH: {
            "schema": "carnot.experiment_5761.exact_constraint_acquisition_benchmark.v1",
            "status": "complete",
            "honest_verdict": "complete: exact_constraint_acquisition_benchmark_ready",
            "ca_benchmark_ready_score": 1.0,
            "train_dev_science_disjoint_score": 1.0,
            "exact_validator_disagreement_count": 0,
            "structure_receipt_failure_count": 0,
            "solution_receipt_failure_count": 0,
            "llm_inference_used": False,
            "instance_count": 120,
            "reproducibility_checksum": "",
        },
        mod.EXP5762_PATH: {
            "schema": "carnot.experiment_5762.query_driven_constraint_lifecycle.v1",
            "status": "complete",
            "honest_verdict": "complete: query_driven_constraint_lifecycle_credited",
            "constraint_recovery_gain_lcb": 0.143812,
            "prefix_retention_pass_score": 1.0,
            "unsafe_update_count": 0,
            "rejected_update_propagation_count": 0,
            "rollback_hash_mismatch_count": 0,
            "restart_equivalence": {"all_passed": True, "rollback_hash_mismatch_count": 0},
            "continuous_self_learning_credited": True,
            "model_weight_mutation": False,
            "production_default_enabled": False,
            "oracle_boundary_violation_count": 0,
            "reproducibility_checksum": "",
        },
        mod.EXP5763_PATH: {
            "schema": "carnot.experiment_5763.dependent_task_constraint_acquisition.v1",
            "status": "complete",
            "honest_verdict": "complete: dependent_task_constraint_acquisition_credited",
            "dependent_task_ca_ready_score": 1.0,
            "continuous_self_learning_credited": True,
            "nonforgetting_certificate": {"all_prefixes_exact": True, "certificate_rate": 1.0},
            "unsafe_update_count": 0,
            "rejected_update_propagation_count": 0,
            "rollback_hash_mismatch_count": 0,
            "restart_equivalence": {"all_passed": True},
            "model_weight_mutation": False,
            "production_default_enabled": False,
            "reproducibility_checksum": "",
        },
        mod.EXP5764_PATH: {
            "status": "complete",
            "honest_verdict": "complete: profiled hot path; no timing or hardware claim",
            "optimized_path_ready_score": 1.0,
            "semantic_parity_score": 1.0,
            "distributional_parity_score": 1.0,
            "production_backend_reachable_score": 1.0,
            "timing_promotion_claimed": False,
            "hardware_speedup_claimed": False,
            "two_axis_exchange_reopened": False,
            "reproducibility_checksum": "",
        },
        mod.EXP5765_PATH: {
            "status": "complete",
            "honest_verdict": "complete: final benchmark did not prove 10x; technique retired",
            "rust_10x_claimed": False,
            "rust_10x_retired": True,
            "consecutive_larger_size_rule_passed": False,
            "matched_quality_gate_passed": True,
            "speedup_lcb_by_size": {"48": 2.36, "96": 1.65, "192": 1.17, "256": 1.05},
            "hardware_speedup_claimed": False,
            "two_axis_exchange_reopened": False,
            "reproducibility_checksum": "",
        },
        mod.EXP5766_PATH: {
            "status": "complete",
            "honest_verdict": "complete: loo_component_interaction_audit_no_heldout_gain_no_causal_interactions",
            "registry_precheck": {"ok": True, "public_game_count": 25, "registry_level_count": 183},
            "loo_generalization_delta": 0.0,
            "loo_generalization_delta_lcb": 0.0,
            "causal_interaction_count": 0,
            "solve_provenance": "development_proxy",
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
            "source_leak_count": 0,
            "game_identity_leak_count": 0,
            "outer_loop_re_used": False,
            "per_game_adapter_used": False,
            "source_read_used": False,
            "production_default_enabled": False,
            "reproducibility_checksum": "",
        },
        mod.EXP5767_PATH: _blocked_gate(
            "2 of 4 gate(s) failed; first failure: loo_generalization_delta_lcb",
            [
                {
                    "upstream": "exp5766-arc-loo-component-interaction-audit",
                    "artifact_field": "loo_generalization_delta_lcb",
                    "op": ">",
                    "expected": 0.0,
                    "actual": 0.0,
                    "passed": False,
                }
            ],
        ),
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    for rel_path, payload in _artifact_payloads().items():
        if rel_path == omit:
            continue
        _write_json(root, rel_path, payload)

    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.CONDUCTOR_LOG_PATH:
            _write_text(
                root,
                rel_path,
                "\n".join(
                    [
                        "| t | Transition terminal .513 evidence | FAIL | precondition collision |",
                        "| t | Ingest post-V514 source deltas | OK | complete |",
                        "| t | Build a lossless bare-scalar bridge for the ready | OK | complete |",
                        "| t | Build a lossless bare-scalar bridge for repaired R | OK | complete |",
                        "| t | Gated on Exp5757 scalar readiness: measure SOTA pr | OK | complete |",
                        "| t | Gated on Exp5759 utility>0: allocate exact conflic | GATE_BLOCK | 2 failed |",
                        "| t | Build an exact MPMMine-shaped benchmark for typed | OK | complete |",
                        "| t | Gated on Exp5761 exact corpus: learn typed constra | OK | complete |",
                        "| t | Gated on Exp5762 recovery>0: scale constraint acqu | OK | complete |",
                        "| t | Gated on Exp5758 scalar parity: profile and optimi | OK | exists |",
                        "| t | Gated on Exp5764 parity: run the final matched Rus | OK | complete |",
                        "| t | Measure ARC leave-one-game-out generalization and | OK | complete |",
                        "| t | Gated on Exp5766 held-out delta>0: harden one game | GATE_BLOCK | 2 failed |",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.ARC_REGISTRY_PATH:
            _write_text(
                root, rel_path, "reproducible_total_games: 25\nreproducible_total_levels: 183\n"
            )
        elif rel_path == mod.CAPABILITIES_PATH:
            _write_text(root, rel_path / "research-reporting" / "spec.md", "REQ-REPORT-5768\n")
        else:
            _write_text(root, rel_path)


def test_spec_contains_req_report_5768_contract() -> None:
    """REQ-REPORT-5768: the OpenSpec contract names the artifact and safeguards."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5768") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "SCENARIO-REPORT-5768-MISSING-OR-BLOCKED" in section
    assert "cached_artifact_reconciliation_no_llm" in section
    assert "experiment_5768_v514_capstone_reconciliation.py" in section


def test_scenario_report_5768_reconciles_v514_branches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5768: executed, blocked, negative, null, and promoted states separate."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0}],
    )

    assert report["honest_verdict"].startswith("complete:")
    assert report["proposal_bridge_ready"] is True
    assert report["rust_bridge_ready"] is True
    assert report["proposal_panel_executed"] is True
    assert report["proposal_utility_ready"] is False
    assert report["selective_feedback_executed"] is False
    assert report["selective_feedback_ready"] is False

    assert report["continuous_self_learning_executed"] is True
    assert report["continuous_self_learning_credited"] is True
    assert report["kan_scaleup_retired"] is True
    assert report["constraint_acquisition_ready"] is True
    assert report["dependent_task_ca_executed"] is True
    assert report["dependent_task_ca_ready"] is True

    assert report["rust_hot_path_ready"] is True
    assert report["rust_benchmark_executed"] is True
    assert report["rust_10x_claimed"] is False
    assert report["rust_10x_retired"] is True
    assert report["arc_loo_audit_executed"] is True
    assert report["arc_loo_generalization_positive"] is False
    assert report["arc_composition_executed"] is False
    assert report["arc_live_generalization_delta"] == 0.0
    assert report["solve_provenance"] == "development_proxy"
    assert report["arc_registry_delta"] == 0
    assert report["arc_solve_credited"] is False

    assert set(report["blocked_task_ids"]) == {
        "exp5755-transition-v514",
        "exp5760-selective-exact-feedback-search",
        "exp5767-arc-game-blind-composition-hardening",
    }
    assert report["negative_result_task_ids"] == ["exp5759-sota-exact-proposal-utility-panel"]
    assert set(report["scientific_null_task_ids"]) == {
        "exp5765-one-axis-final-10x-crossover",
        "exp5766-arc-loo-component-interaction-audit",
    }
    assert "exp5762-query-driven-constraint-lifecycle" in report["promoted_task_ids"]
    assert (
        report["task_outcome_matrix"]["exp5760-selective-exact-feedback-search"]["blocked_gate"]
        is True
    )
    assert (
        report["task_outcome_matrix"]["exp5760-selective-exact-feedback-search"]["negative"]
        is False
    )
    assert (
        report["task_outcome_matrix"]["exp5759-sota-exact-proposal-utility-panel"]["negative"]
        is True
    )
    assert (
        report["artifact_hashes"]["exp5759-sota-exact-proposal-utility-panel"]["checksum_matches"]
        is True
    )

    assert report["hardware_claims"]["speedup_claimed"] is False
    assert report["model_weight_mutation"] is False
    assert report["closed_scopes_reopened"] is False
    assert report["public_docs_modified"] is False
    assert report["publication_performed"] is False
    assert report["inference_substrate"] == "cached_artifact_reconciliation_no_llm"
    assert all(
        row["status"] != "passed" for row in report["e2e_checks"] if row["requires_hardware"]
    )


def test_scenario_report_5768_missing_inputs_remain_blocked_not_negative(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5768-MISSING-OR-BLOCKED: missing artifacts are not science failures."""

    _make_root(tmp_path, omit=mod.EXP5759_PATH)
    report = mod.build_report(tmp_path)

    row = report["task_outcome_matrix"]["exp5759-sota-exact-proposal-utility-panel"]
    assert row["artifact_present"] is False
    assert row["complete"] is False
    assert row["negative"] is False
    assert row["null"] is False
    assert report["proposal_panel_executed"] is False
    assert report["proposal_utility_ready"] is False
    assert "exp5759-sota-exact-proposal-utility-panel" not in report["negative_result_task_ids"]
    assert report["artifact_hashes"]["exp5759-sota-exact-proposal-utility-panel"]["sha256"] is None


def test_scenario_report_5768_emit_and_schema_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-5768-FIELD-PRINCIPLES: emitted artifact is stable and annotated."""

    _make_root(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "unit", "exit_code": 0}],
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["reproducibility_checksum"] == report["reproducibility_checksum"]
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert mod.artifact_schema_errors(written) == []
    assert set(written).issubset(written["field_principles"])
    assert all(written["field_principles"][field] for field in written)

    assert mod._status_for_payload({}, {"exists": False}) == "missing"
    assert mod._status_for_payload({}, {"exists": True, "loadable": False}) == "malformed"
    assert mod._status_for_payload({"schema": "blocked_gate_check_v1"}, {}) == "blocked-gate"
    assert mod._status_for_payload({"status": "blocked"}, {}) == "blocked-gate"
    assert mod._status_for_payload({"honest_verdict": "blocked: x"}, {}) == "blocked-precondition"
    assert mod._status_for_payload({"honest_verdict": "complete: x"}, {}) == "complete"
    assert mod._status_for_payload({"honest_verdict": "success: x"}, {}) == "complete"
    assert mod._status_for_payload({"status": "odd"}, {}) == "odd"
    assert mod._number({"x": True}, "x") == 1.0
    assert mod._number({"x": "bad"}, "x", 2.0) == 2.0
    assert mod._number({}, "missing", 3.0) == 3.0
    assert mod._bool({"x": "true"}, "x") is True
    assert mod._bool({"x": 0}, "x") is False
    assert mod._bool({}, "missing") is False
    assert mod._checksum_matches({"reproducibility_checksum": "not-a-sha"}, "") is False
    assert mod._checksum_matches({"reproducibility_checksum": ""}, "") is False
    assert mod._latest_log_line("a\nneedle first\nneedle second\n", ("needle",)) == (
        "needle second"
    )
    assert mod._outcome_from_line("| t | task | OK | done |") == "OK"
    assert mod._outcome_from_line("| t | task | FAIL | done |") == "FAIL"
    assert mod._outcome_from_line("| t | task | ODD | done |") == "LOGGED"
    assert mod._outcome_from_line(None) == "MISSING_LOG_LINE"

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    payload, meta = mod._read_json_any(scalar)
    assert payload == {"_non_mapping_payload": []}
    assert meta["loadable"] is False
    assert meta["error"] == "json_payload_not_mapping"

    broken = dict(written)
    broken.pop("status")
    broken["field_principles"] = {}
    broken["inference_substrate"] = "wrong"
    broken["kan_scaleup_retired"] = False
    broken["arc_registry_delta"] = 1
    broken["arc_solve_credited"] = True
    broken["model_weight_mutation"] = True
    broken["closed_scopes_reopened"] = True
    broken["public_docs_modified"] = True
    broken["publication_performed"] = True
    broken["honest_verdict"] = "ambiguous"
    broken["reproducibility_checksum"] = "bad"
    errors = mod.artifact_schema_errors(broken)
    assert "missing:status" in errors
    assert "field_principles.schema" in errors
    assert "inference_substrate" in errors
    assert "kan_scaleup_retired" in errors
    assert "arc_registry_delta" in errors
    assert "arc_solve_credited" in errors
    assert "model_weight_mutation" in errors
    assert "closed_scopes_reopened" in errors
    assert "public_docs_modified" in errors
    assert "publication_performed" in errors
    assert "honest_verdict" in errors
    assert "reproducibility_checksum" in errors

    no_principles = dict(written)
    no_principles["field_principles"] = None
    assert "field_principles" in mod.artifact_schema_errors(no_principles)
    stale_checksum = dict(written)
    stale_checksum["status"] = "tampered"
    assert "reproducibility_checksum_mismatch" in mod.artifact_schema_errors(stale_checksum)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="artifact schema errors"):
        mod.emit_report(tmp_path, output_path=tmp_path / "bad.json")

    original = mod.FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(tmp_path)
    finally:
        mod.FIELD_PRINCIPLES["status"] = original
