"""Tests for the Exp5716 V510 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5716, SCENARIO-CAPSTONE-5716,
SCENARIO-CAPSTONE-5716-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5716-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5716_v510_capstone_reconciliation as mod


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


def _payloads(*, include_5710: bool = True) -> dict[Path, JsonDict]:
    payloads: dict[Path, JsonDict] = {
        mod.EXP5647_CAPSTONE_PATH: {
            "schema": "carnot.experiment_5647.v509_capstone_reconciliation.v1",
            "status": "complete",
            "experiment_id": "exp5647-v509-capstone-reconciliation",
            "honest_verdict": (
                "complete: v509 reconciled; fr11_promoted=True; "
                "arc_registry_delta=0; two_axis_quality_promoted=False"
            ),
            "fr11_independent_promotion_status": {"promoted": True},
            "one_axis_replica_exchange_preserved": True,
            "two_axis_quality_status": {"promoted": False},
            "rust_parity_status": {"promoted": False},
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5706_TRANSITION_PATH: {
            "schema": "carnot.experiment_5706.transition_v510.v1",
            "status": "complete",
            "experiment_id": "exp5706-transition-v510",
            "honest_verdict": (
                "complete: v510 transition archived .509 evidence; "
                "fr11_promoted=True; shadow_default_enabled=False"
            ),
            "fr11_promoted": True,
            "fr11_shadow_default_enabled": False,
            "arc_registry_count": 177,
            "arc_registry_delta": 0,
            "one_axis_replica_exchange_promoted": True,
            "two_axis_quality_promoted": False,
            "retirements_applied": [
                {
                    "scope": "arc_counterexample_patched_transition_model_exp5641",
                    "manifest_entry_present": True,
                    "preserves": ["generic_arc_models"],
                },
                {
                    "scope": "two_axis_beta_lambda_tempering_extension_exp5645",
                    "manifest_entry_present": True,
                    "preserves": ["one_axis_temperature_exchange"],
                },
            ],
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
            "protected_file_checks": {
                "research-roadmap.yaml": {"unchanged": True},
                "scripts/research_conductor.py": {"unchanged": True},
            },
        },
        mod.EXP5707_SOURCE_PATH: {
            "schema": "carnot.experiment_5707.v510_source_delta_ingestion.v1",
            "status": "complete",
            "experiment_id": "exp5707-v510-source-delta-ingestion",
            "honest_verdict": "complete: no new non-duplicate actionable V510 source deltas",
            "references_updated": False,
            "roadmap_change_required": False,
        },
        mod.EXP5708_CANARY_PATH: {
            "schema": "carnot.experiment_5708.sota_exact_constraint_canary.v1",
            "status": "blocked",
            "experiment_id": "experiment_5708_sota_exact_constraint_canary",
            "honest_verdict": "blocked: parse_failures",
            "blocked_reasons": ["parse_failures"],
            "sota_canary_ready_score": 0.0,
            "cuda_offload_authenticated": True,
            "cuda_offload_authenticated_score": 1.0,
            "validator_disagreement_count": 0,
            "parse_failure_count": 47,
            "missing_row_count": 0,
            "manifest_row_count": 50,
            "native_json_grammar_used": False,
            "retired_runtime_used": False,
            "external_scorer_used": False,
            "stream_root_commitment": "sha256:stream",
            "shadow_prefix_hash": "sha256:prefix",
            "sealed_suffix_hash": "sha256:suffix",
            "n_gpu_layers_requested": -1,
            "n_gpu_layers_offloaded": 31,
            "model_repo_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gguf_filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "model_hash": "sha256:model",
            "row_manifest_path": "results/experiment_5708_sota_exact_constraint_canary.rows.jsonl",
        },
        mod.EXP5709_SHADOW_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5709,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 3 gate(s) failed; first failure: "
                "exp5708-sota-exact-constraint-canary.sota_canary_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp5708-sota-exact-constraint-canary",
                    "artifact_field": "sota_canary_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                }
            ],
        },
        mod.EXP5711_ARC_QUAL_PATH: {
            "schema": "carnot.experiment_5711.arc_relational_goal_energy_live_qualification.v1",
            "status": "complete",
            "experiment": 5711,
            "honest_verdict": "complete: relational_goal_energy_live_route_qualified_no_solve_claim",
            "relational_goal_energy_ready_score": 1.0,
            "live_path_reachable": True,
            "live_path_reachable_score": 1.0,
            "candidate_guidance_call_count": 1,
            "frontier_goal_bias_call_count": 1,
            "candidate_order_change_count": 1,
            "fallback_order_equivalence": True,
            "zero_variance_fallback_count": 1,
            "negative_control_results": [{"unsafe_route_accepted": False}],
            "corrupted_control_results": [{"unsafe_route_accepted": False}],
            "per_game_leakage_detected": False,
            "per_game_constant_scan": {"per_game_constants_detected": False},
            "outer_loop_bfs_used": False,
            "game_source_read_count": 0,
            "game_adapter_count": 0,
            "solve_provenance": "development_proxy",
            "new_levels_claimed": 0,
        },
        mod.EXP5712_ARC_AB_PATH: {
            "schema": "carnot.experiment_5712.arc_relational_goal_energy_live_ab.v1",
            "status": "complete",
            "experiment": 5712,
            "honest_verdict": "complete: relational_live_route_null_no_promotion",
            "relational_live_ab_ready_score": 0.0,
            "budget_parity_receipt": {"matched": True},
            "successful_pair_count": 6,
            "level_regression_count": 0,
            "unsafe_route_accept_count": 0,
            "new_levels_claimed": 0,
            "registry_updated": False,
            "solve_provenance": "development_proxy",
            "candidate_order_change_count": {"control": 0, "treatment": 0},
            "route_activation_count": {"control": 0, "treatment": 0},
            "paired_intervals": {
                "retained_level_delta": {"ci95_low": 0.0, "ci95_high": 0.0},
                "actions_saved_per_reproduced_level": {
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                },
            },
            "negative_controls_preserved": True,
            "failed_pair_reasons": [],
        },
        mod.EXP5713_ARC_LEVEL_PATH: {
            "schema": "arc_live_self_discovery_levelup_attempt.v510",
            "status": "complete",
            "experiment_id": "exp5713-arc-live-self-discovery-levelup-v510",
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_L9_bounded_live_attempt_v510",
            "solve_provenance": "live_agent_self_discovery",
            "selected_game": "lf52",
            "selected_level": "L9",
            "target_level": 9,
            "independent_reproduction_pass": False,
            "offline_reproduced": False,
            "reproduction_seed_count": 0,
            "reproduced_levels": 0,
            "registry_count_before": 177,
            "registry_count_after": 177,
            "registry_delta": 0,
            "registry_updated": False,
            "critical_flags": [],
            "level_transition_events": [],
            "llm_used": False,
            "hand_solution_used": False,
            "outer_loop_bfs_used": False,
            "game_source_read_count": 0,
            "game_adapter_count": 0,
        },
        mod.EXP5714_RUST_PARITY_PATH: {
            "schema": "carnot.experiment_5714.one_axis_tempering_rust_parity.v1",
            "status": "complete",
            "experiment_id": "exp5714-one-axis-rust-python-exact-parity",
            "honest_verdict": (
                "complete: one-axis corrected-cDLS Rust/Python parity is exact "
                "within frozen tolerances; portability only, no speed claim"
            ),
            "one_axis_rust_parity_ready_score": 1.0,
            "broken_control_rejected": True,
            "broken_control_rejected_score": 1.0,
            "energy_error_max": 1e-16,
            "target_marginal_delta": 1e-16,
            "deterministic_decision_parity": True,
            "cross_language_restart_pass": True,
            "checkpoint_roundtrip_pass": True,
            "python_fallback_equivalence": True,
            "two_axis_code_added": False,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5715_RUST_QUALITY_PATH: {
            "schema": "carnot.experiment_5715.one_axis_tempering_rust_quality_restart.v1",
            "status": "complete",
            "experiment_id": "exp5715-one-axis-rust-quality-restart-parity",
            "honest_verdict": (
                "complete: one-axis Rust/Python hard-instance quality and "
                "cross-language restart parity pass; no timing or hardware claim"
            ),
            "one_axis_rust_quality_ready_score": 1.0,
            "source_quality_receipt": {"eligible": True, "quality_mixing_ready": True},
            "upstream_gate_receipts": {
                "exp5634": {"ready": True, "quality_mixing_ready": True},
                "exp5714": {"ready": True, "one_axis_rust_parity_ready_score": 1.0},
            },
            "successful_seed_count": {"value": 5, "attempted_seed_count": 5},
            "material_regression_count": 0,
            "failed_seed_reasons": [],
            "python_to_rust_restart_pass": True,
            "rust_to_python_restart_pass": True,
            "transition_budget_parity": {
                "matched_cold_target_collection": True,
                "matched_corrected_transition_budget": True,
                "wall_time_compared": False,
            },
            "swap_schedule_parity": {"matched_language_swap_schedule": True},
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
    }
    if include_5710:
        payloads[mod.EXP5710_ISOLATED_PATH] = {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5710,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "pre-emptive skip: upstream exp5709 not promoted",
            "gates_evaluated": [
                {
                    "upstream": "exp5709-fr11-prospective-shadow-stream",
                    "artifact_field": "prospective_shadow_ready_score",
                    "actual": None,
                    "expected": 1.0,
                    "passed": False,
                }
            ],
        }
    return payloads


def _make_root(
    root: Path,
    *,
    include_5710: bool = True,
    omit: Path | None = None,
    malformed: Path | None = None,
) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        _write_text(root, rel_path)
    _write_text(root, mod.EXP5708_ROWS_PATH, '{"row_id": "r0"}\n')
    _write_text(root, mod.EXP5713_TRACE_PATH, '{"trajectory_hash": "sha256:trace"}\n')
    for rel_path, payload in _payloads(include_5710=include_5710).items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)


def test_req_capstone_5716_spec_declares_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5716: OpenSpec declares the V510 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5716") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.PRIMARY_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5716_live_repo_reconciles_observed_evidence() -> None:
    """SCENARIO-CAPSTONE-5716: live V510 evidence promotes only earned claims."""

    artifact = mod.run_capstone(
        root=REPO,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.EXP5710_ISOLATED_PATH.as_posix() in artifact["missing_artifacts"]
    assert len(artifact["upstream_artifacts"]) == len(mod.UPSTREAM_ARTIFACT_PATHS)
    assert artifact["upstream_gate_statuses"][mod.EXP5708_TASK_ID]["status"] == "blocked"
    assert artifact["upstream_gate_statuses"][mod.EXP5709_TASK_ID]["status"] == "gate_skipped"
    assert artifact["upstream_gate_statuses"][mod.EXP5710_TASK_ID]["status"] == "missing"

    assert artifact["v509_fr11_promotion_preserved"] is True
    assert artifact["sota_canary_status"]["promoted"] is False
    assert artifact["sota_canary_status"]["parse_failure_count"] == 47
    assert artifact["cuda_offload_status"]["authenticated"] is True
    assert artifact["cuda_offload_status"]["promotes_canary"] is False
    assert artifact["prospective_shadow_status"]["promoted"] is False
    assert artifact["isolated_canary_status"]["promoted"] is False
    assert artifact["isolated_canary_status"]["artifact_present"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["model_weight_mutation"] is False

    assert artifact["arc_relational_qualification_status"]["qualified"] is True
    assert artifact["arc_relational_qualification_status"]["solve_claimed"] is False
    assert artifact["arc_relational_live_ab_status"]["promoted"] is False
    assert artifact["arc_relational_live_ab_status"]["unsafe_route_accept_count"] == 0
    assert artifact["arc_registry_count_before"] == 177
    assert artifact["arc_registry_count_after"] == 177
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_provenance"]["solve_credited"] is False
    assert artifact["arc_solve_provenance"]["solve_provenance"] == "live_agent_self_discovery"

    assert artifact["one_axis_python_promotion_preserved"] is True
    assert artifact["one_axis_rust_parity_status"]["promoted"] is True
    assert artifact["one_axis_rust_quality_restart_status"]["promoted"] is True
    assert artifact["two_axis_retirement_preserved"] is True
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert all(row["unchanged"] for row in artifact["forbidden_files_unchanged"].values())
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5716_fixture_with_gate_skipped_isolated_canary_completes(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5716: gate-skipped canary is recorded without promotion."""

    _make_root(tmp_path, include_5710=True)
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["missing_artifacts"] == []
    assert artifact["isolated_canary_status"]["artifact_present"] is True
    assert artifact["isolated_canary_status"]["status"] == "gate_skipped"
    assert artifact["isolated_canary_status"]["promoted"] is False
    assert {row["scope"] for row in artifact["retirements_applied"]} >= {
        "arc_counterexample_patched_transition_model_exp5641",
        "two_axis_beta_lambda_tempering_extension_exp5645",
        "fr11_prospective_shadow_stream_exp5709_same_verdict",
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5716_missing_and_malformed_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5716-MISSING-MALFORMED: bad inputs block promotion."""

    _make_root(
        tmp_path,
        include_5710=True,
        omit=mod.EXP5715_RUST_QUALITY_PATH,
        malformed=mod.EXP5714_RUST_PARITY_PATH,
    )
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [mod.EXP5715_RUST_QUALITY_PATH.as_posix()]
    assert artifact["malformed_artifacts"] == [mod.EXP5714_RUST_PARITY_PATH.as_posix()]
    assert artifact["one_axis_rust_parity_status"]["promoted"] is False
    assert artifact["one_axis_rust_quality_restart_status"]["promoted"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5716_validation_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5716-FIELD-PRINCIPLES: validation rejects laundering."""

    _make_root(tmp_path, include_5710=True)
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
    )
    assert mod.validate_artifact(artifact) == []

    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "upstream_artifacts" in mod.validate_artifact({**artifact, "upstream_artifacts": []})
    assert "upstream_gate_statuses" in mod.validate_artifact(
        {**artifact, "upstream_gate_statuses": {}}
    )
    assert "v509_fr11_promotion_preserved" in mod.validate_artifact(
        {**artifact, "v509_fr11_promotion_preserved": False}
    )
    assert "sota_canary_status" in mod.validate_artifact(
        {**artifact, "sota_canary_status": {"promoted": True}}
    )
    assert "cuda_offload_status" in mod.validate_artifact(
        {**artifact, "cuda_offload_status": {"authenticated": False}}
    )
    assert "prospective_shadow_status" in mod.validate_artifact(
        {**artifact, "prospective_shadow_status": {"promoted": True}}
    )
    assert "isolated_canary_status" in mod.validate_artifact(
        {**artifact, "isolated_canary_status": {"promoted": True}}
    )
    assert "production_default_enabled" in mod.validate_artifact(
        {**artifact, "production_default_enabled": True}
    )
    assert "model_weight_mutation" in mod.validate_artifact(
        {**artifact, "model_weight_mutation": True}
    )
    assert "arc_relational_live_ab_status" in mod.validate_artifact(
        {**artifact, "arc_relational_live_ab_status": {"promoted": True}}
    )
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": 1})
    bad_solve = dict(artifact["arc_solve_provenance"])
    bad_solve["solve_credited"] = True
    assert "arc_solve_provenance" in mod.validate_artifact(
        {**artifact, "arc_solve_provenance": bad_solve}
    )
    assert "one_axis_python_promotion_preserved" in mod.validate_artifact(
        {**artifact, "one_axis_python_promotion_preserved": False}
    )
    assert "one_axis_rust_parity_status" in mod.validate_artifact(
        {**artifact, "one_axis_rust_parity_status": {"promoted": False}}
    )
    assert "one_axis_rust_quality_restart_status" in mod.validate_artifact(
        {**artifact, "one_axis_rust_quality_restart_status": {"promoted": False}}
    )
    assert "two_axis_retirement_preserved" in mod.validate_artifact(
        {**artifact, "two_axis_retirement_preserved": False}
    )
    assert "timing_claimed" in mod.validate_artifact({**artifact, "timing_claimed": True})
    assert "hardware_speedup_claimed" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claimed": True}
    )
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "adversarial_verification_summary" in mod.validate_artifact(
        {**artifact, "adversarial_verification_summary": []}
    )
    assert "arc_relational_qualification_status" in mod.validate_artifact(
        {**artifact, "arc_relational_qualification_status": []}
    )
    assert "arc_relational_qualification_status" in mod.validate_artifact(
        {**artifact, "arc_relational_qualification_status": {"qualified": False}}
    )
    assert "arc_registry_count_before" in mod.validate_artifact(
        {**artifact, "arc_registry_count_before": "177"}
    )
    assert "arc_registry_count_after" in mod.validate_artifact(
        {**artifact, "arc_registry_count_after": "177"}
    )
    assert "retirements_applied" in mod.validate_artifact({**artifact, "retirements_applied": []})
    assert "spec_reconciliation" in mod.validate_artifact({**artifact, "spec_reconciliation": []})
    assert "ops_reconciliation" in mod.validate_artifact({**artifact, "ops_reconciliation": []})
    assert "known_issue_reconciliation" in mod.validate_artifact(
        {**artifact, "known_issue_reconciliation": []}
    )
    assert "test_commands" in mod.validate_artifact({**artifact, "test_commands": {}})
    assert "test_exit_codes" in mod.validate_artifact({**artifact, "test_exit_codes": []})
    assert "e2e_check_receipts" in mod.validate_artifact({**artifact, "e2e_check_receipts": {}})
    assert "forbidden_files_unchanged" in mod.validate_artifact(
        {
            **artifact,
            "forbidden_files_unchanged": {
                "research-roadmap.yaml": {"unchanged": False},
                "scripts/research_conductor.py": {"unchanged": True},
            },
        }
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "missing_artifacts" in mod.validate_artifact(
        {**artifact, "missing_artifacts": ["not-in-status.json"]}
    )
    assert "malformed_artifacts" in mod.validate_artifact(
        {**artifact, "malformed_artifacts": ["not-in-status.json"]}
    )
    assert mod._paths_with_status({"upstream_gate_statuses": []}, "missing") == set()


def test_scenario_capstone_5716_writer_cli_and_helpers_cover_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-5716-FIELD-PRINCIPLES: helpers fail closed."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    payload, meta = mod._read_json_any(list_json)
    assert payload == {}
    assert meta["error"] == "not_json_object"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad\n", encoding="utf-8")
    bad_payload, bad_meta = mod._read_json_any(bad_json)
    assert bad_payload == {}
    assert bad_meta["error"] == "malformed_json"
    missing_payload, missing_meta = mod._read_json_any(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"

    assert mod._status_for_payload({}, {"exists": False}) == "missing"
    assert mod._status_for_payload({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._status_for_payload(
            {"schema": "blocked_gate_check_v1"}, {"exists": True, "loadable": True}
        )
        == "gate_skipped"
    )
    assert (
        mod._status_for_payload({"flagged_adversarial": True}, {"exists": True, "loadable": True})
        == "flagged"
    )
    assert (
        mod._status_for_payload(
            {"honest_verdict": "blocked: x"}, {"exists": True, "loadable": True}
        )
        == "blocked"
    )
    assert (
        mod._status_for_payload(
            {"honest_verdict": "complete: x"}, {"exists": True, "loadable": True}
        )
        == "complete"
    )
    assert (
        mod._status_for_payload({"honest_verdict": "unknown"}, {"exists": True, "loadable": True})
        == "unknown"
    )
    assert mod._number({"x": "1.5"}, "x") == 1.5
    assert mod._number({"x": "bad"}, "x") == 0.0
    assert mod._int({"x": True}, "x") == 1
    assert mod._int({"x": "-2"}, "x") == -2
    assert mod._int({"x": "bad"}, "x") == 0
    assert mod._bool({"x": "true"}, "x") is True
    assert mod._bool({"x": "false"}, "x") is False
    assert mod._all_controls_safe([{"unsafe_route_accepted": False}]) is True
    assert mod._all_controls_safe([{"unsafe_route_accepted": True}]) is False
    assert mod._all_controls_safe([]) is False
    assert mod._all_controls_safe(["bad"]) is False
    assert mod._all_controls_safe("bad") is False
    assert mod._bool({"x": 1}, "x") is True
    assert mod._load_validation_results(None) == mod.DEFAULT_VALIDATION_RESULTS

    missing_yaml, missing_yaml_meta = mod._read_yaml_mapping(tmp_path / "missing.yaml")
    assert missing_yaml == {}
    assert missing_yaml_meta["error"] == "missing"
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":\n", encoding="utf-8")
    malformed_yaml, malformed_yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert malformed_yaml == {}
    assert malformed_yaml_meta["error"] == "malformed_yaml"
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    list_mapping, list_mapping_meta = mod._read_yaml_mapping(list_yaml)
    assert list_mapping == {}
    assert list_mapping_meta["error"] == "not_yaml_mapping"
    assert mod._registry_count({"reproducible_total_levels": True}) is None
    assert mod._registry_count({"reproducible_total_levels": "177"}) == 177

    statuses = {
        task_id: {"status": "complete", "flagged_adversarial": False}
        for task_id in mod.UPSTREAM_ARTIFACT_PATHS
    }
    arc_qual = mod._derive_arc_qualification(
        {
            **_payloads()[mod.EXP5711_ARC_QUAL_PATH],
            "per_game_constant_scan": "bad",
        },
        statuses,
    )
    assert arc_qual["qualified"] is True
    assert mod._interval_excludes_zero_positive("bad") is False
    arc_ab = mod._derive_arc_ab(
        {"budget_parity_receipt": "bad", "paired_intervals": {"bad": "row"}},
        statuses,
    )
    assert arc_ab["promoted"] is False
    arc_solve = mod._derive_arc_solve({"critical_flags": "bad"}, statuses)
    assert arc_solve["critical_flags"] == []
    flagged_artifacts = {rel_path.as_posix(): {} for rel_path in mod.PRIMARY_ARTIFACT_PATHS}
    flagged_artifacts[mod.EXP5708_CANARY_PATH.as_posix()] = {
        "flagged_adversarial": True,
        "critical_flags": [{"kind": "critical"}],
        "corrigendum_pending": [{"kind": "pending"}],
    }
    flagged_summary = mod._derive_adversarial_summary(flagged_artifacts, statuses)
    assert flagged_summary["flagged_or_corrigendum_rows"][0]["task_id"] == mod.EXP5708_TASK_ID

    _make_root(tmp_path, include_5710=True)
    validation = [
        {"command": "focused", "exit_code": 0, "status": "passed"},
        {"command": "broad", "exit_code": 2, "status": "pre_existing_failure"},
    ]
    artifact = mod.write_capstone(
        root=tmp_path,
        validation_results=validation,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
    )
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["test_exit_codes"]["broad"] == 2

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    assert mod._load_validation_results(validation_path) == validation
    bad_validation = tmp_path / "bad-validation.json"
    bad_validation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_validation_results(bad_validation)

    output_path = tmp_path / "custom" / "capstone.json"
    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                str(output_path),
                "--validation-results",
                str(validation_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID

    monkeypatch.setattr(mod, "run_capstone", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError):
        mod.write_capstone(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
