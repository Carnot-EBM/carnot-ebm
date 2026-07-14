"""Tests for the Exp5647 V509 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5647, SCENARIO-CAPSTONE-5647,
SCENARIO-CAPSTONE-5647-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5647-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5647_v509_capstone_reconciliation as mod


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


def _payloads() -> dict[Path, JsonDict]:
    return {
        mod.EXP5636_TRANSITION_PATH: {
            "schema": "carnot.experiment_5636.transition_v509.v1",
            "status": "complete",
            "experiment_id": "exp5636-transition-v509",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: v509 transition loaded",
            "promoted_substrates": [
                {"key": "one_axis_temperature_exchange_exact"},
                {"key": "one_axis_temperature_exchange_quality"},
            ],
            "retired_scopes": [{"key": "arc_epistemic_probe_exp5630"}],
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        mod.EXP5637_SOURCE_PATH: {
            "schema": "carnot.experiment_5637.v509_source_delta_ingestion.v1",
            "status": "complete",
            "experiment_id": "exp5637-v509-source-delta-ingestion",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: accepted 1 non-duplicate actionable V509 source delta",
            "new_references_added": ["baba_in_wonderland_2605_16725"],
            "closed_scopes_reopened": False,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        mod.EXP5638_SCHEMA_PATH: {
            "schema": "carnot.experiment_5638.fr11_gate_schema_corrigendum.v1",
            "experiment": 5638,
            "status": "complete",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: hash_bound_scalar_gate_contract_ready",
            "gate_contract_ready_score": 1.0,
            "source_hash_exact": True,
            "unsafe_false_accept_count_total": 0,
            "raw_unsafe_false_accept_count": {"by_arm": {"a": 0, "b": 0}, "total": 0},
            "by_arm_sum": 0,
            "by_arm_reconciliation_pass": True,
            "source_continuous_self_learning_ready": True,
            "scientific_recompute_performed": False,
            "source_artifact_modified": False,
            "inference_substrate": "deterministic_artifact_schema_normalization",
        },
        mod.EXP5639_AUDIT_PATH: {
            "schema": "carnot.experiment_5639.anytime_valid_csl_independent_audit.v1",
            "status": "complete",
            "task_id": "exp5639-anytime-valid-csl-independent-audit",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: anytime_valid_csl_independent_audit_ready",
            "upstream_gate_receipts": {"both_structured_gates_enforced": True},
            "fr11_independent_promotion_ready_score": 1.0,
            "paired_benefit_intervals": {
                "frozen_no_update": {"lower": 0.25, "mean": 0.3, "upper": 0.4},
                "loss_smoothed": {"lower": 0.14, "mean": 0.2, "upper": 0.3},
            },
            "pathwise_risk": {
                "risk_limit": 0.1,
                "risk_intervals": {
                    "worst": {"upper": 0.07, "within_limit": True},
                },
            },
            "worst_group_coverage": {
                "coverage": 0.904762,
                "adequately_powered_groups_only": True,
                "n": 168,
                "floor": 0.9,
            },
            "unsafe_false_accept_count_total": 0,
            "retention_pass": True,
            "poison_rejection_pass": True,
            "checkpoint_replay_pass": True,
            "critical_flag_count": 0,
            "independent_recomputation": {
                "row_level_replay_performed": True,
                "exp5628_aggregate_metrics_used_as_authority": False,
                "checkpoint_receipts_replayed": 900,
            },
            "adversarial_controls": {
                "poison": {"critical": True, "pass": True},
                "replay": {"critical": True, "pass": True},
            },
            "inference_substrate": "independent_anytime_valid_replay_over_exact_labels",
        },
        mod.EXP5640_SHADOW_PATH: {
            "schema": "carnot.experiment_5640.fr11_shadow_pipeline_integration.v1",
            "status": "complete",
            "experiment_id": "exp5640-fr11-shadow-pipeline-integration",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: fr11_shadow_ready_not_default_enabled",
            "fr11_shadow_integration_ready_score": 1.0,
            "default_enabled": False,
            "feature_flag": "CARNOT_FR11_SHADOW_ADAPTER",
            "exact_verifier_authority": True,
            "benefit_evidence_within_exp5639_bound": True,
            "unsafe_update_accept_count": 0,
            "model_weight_mutation": False,
            "shadow_offline_parity": True,
            "default_path_equivalence": True,
            "checkpoint_atomicity_pass": True,
            "restart_replay_pass": True,
            "rollback_pass": True,
            "ledger_lineage_complete": True,
            "inference_substrate": "exact_verifier_gated_conformal_kan_shadow_adapter",
        },
        mod.EXP5641_ARC_MODEL_PATH: {
            "schema": "carnot.exp5641.arc_counterexample_executable_model.v1",
            "status": "blocked",
            "experiment_id": "exp5641-arc-counterexample-executable-model",
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked: counterexample_patched_executable_model_retired_terminal",
            "executable_model_ready_score": 0.0,
            "unsafe_patch_accept_count": 0,
            "all_receipt_replay_pass": True,
            "patched_vs_unpatched_error_reduction_interval": {
                "lower": -0.003893,
                "mean": 0.002778,
                "upper": 0.009449,
            },
            "agent_owned_evidence_only": True,
            "solve_provenance": "development_proxy",
            "source_read": False,
            "game_adapter_used": False,
            "offline_ground_truth_bfs_used": False,
            "inference_substrate": "deterministic_counterexample_patched_executable_model",
        },
        mod.EXP5642_ARC_LIVE_AB_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5642,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "artifact_field": "executable_model_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                },
                {"artifact_field": "unsafe_patch_accept_count", "actual": 0, "passed": True},
            ],
        },
        mod.EXP5643_ARC_LEVEL_PATH: {
            "schema": "arc_live_self_discovery_levelup_attempt.v4",
            "status": "complete",
            "experiment_id": 5643,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_L8_bounded_live_attempt_v509",
            "target_level": 8,
            "level_reached": 0,
            "live_attempt_executed": True,
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "registry_count_before": 177,
            "registry_count_after": 177,
            "registry_delta": 0,
            "new_reproducible_levels": [],
            "reproduced_levels": 0,
            "registry_updated": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"},
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
            ],
            "source_read": False,
            "game_adapter_used": False,
            "outer_loop_re_used": False,
            "offline_bfs_used": False,
            "llm_invoked": False,
            "inference_substrate": "live_agent_environment_interaction",
        },
        mod.EXP5644_TWO_AXIS_EXACT_PATH: {
            "schema": "carnot.experiment_5644.two_axis_parallel_tempering_exact_audit.v1",
            "status": "complete",
            "experiment_id": "exp5644-two-axis-parallel-tempering-exact-audit",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: exact two-axis beta-lambda label-exchange invariant audit ready for downstream quality trials",
            "two_axis_invariant_ready_score": 1.0,
            "exact_joint_target_tv": 1e-16,
            "exact_target_replica_tv": 3e-16,
            "horizontal_detailed_balance_error_max": 8e-19,
            "vertical_detailed_balance_error_max": 1e-18,
            "transition_row_error_max": 2e-16,
            "target_feasibility_marginal_error": 4e-16,
            "deterministic_replay_pass": True,
            "broken_control_rejected": True,
            "hardware_speedup_claimed": False,
            "timing_claimed": False,
            "inference_substrate": "cpu_exact_enumeration_and_corrected_cdls",
        },
        mod.EXP5645_TWO_AXIS_QUALITY_PATH: {
            "schema": "carnot.experiment_5645.two_axis_tempering_hard_constraint_quality.v1",
            "status": "blocked",
            "experiment_id": "exp5645-two-axis-tempering-hard-constraint-quality",
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked: two-axis constraint-penalty exchange did not clear every preregistered quality promotion gate",
            "two_axis_quality_ready_score": 0.0,
            "successful_seed_count": 5,
            "invalid_execution_count": 0,
            "material_quality_regression_count": 2,
            "target_diagnostics": {"within_exactness_bounds": True},
            "hardware_speedup_claimed": False,
            "timing_claimed": False,
            "inference_substrate": "cpu_two_axis_corrected_cdls_replica_exchange",
        },
        mod.EXP5646_RUST_PARITY_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5646,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "artifact_field": "two_axis_quality_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                },
                {
                    "artifact_field": "material_quality_regression_count",
                    "actual": 2,
                    "expected": 0,
                    "passed": False,
                },
            ],
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


def _artifact_map(payloads: dict[Path, JsonDict] | None = None) -> dict[str, JsonDict]:
    return {path.as_posix(): payload for path, payload in (payloads or _payloads()).items()}


def test_req_capstone_5647_spec_declares_v509_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5647: OpenSpec declares the V509 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5647") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.PRIMARY_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5647_live_repo_reconciles_only_supported_claims() -> None:
    """SCENARIO-CAPSTONE-5647: live V509 evidence promotes only audited claims."""

    artifact = mod.run_capstone(
        root=REPO,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["upstream_artifacts"]) == 11
    assert artifact["upstream_gate_statuses"]["exp5642-arc-executable-model-live-ab"][
        "status"
    ] == "gate_skipped"
    assert artifact["upstream_gate_statuses"]["exp5646-two-axis-tempering-rust-parity"][
        "status"
    ] == "gate_skipped"

    assert artifact["fr11_schema_corrigendum_status"]["schema_repair_ready"] is True
    assert artifact["fr11_schema_corrigendum_status"]["promoted_as_science"] is False
    assert artifact["fr11_independent_promotion_status"]["promoted"] is True
    assert artifact["fr11_shadow_integration_status"]["ready"] is True
    assert artifact["fr11_shadow_integration_status"]["default_enabled"] is False
    assert artifact["fr11_shadow_integration_status"]["automatic_production_enablement"] is False

    assert artifact["arc_executable_model_status"]["promoted"] is False
    assert artifact["arc_executable_model_status"]["exp5630_retired_preserved"] is True
    assert artifact["arc_solve_provenance"]["solve_credited"] is False
    assert artifact["arc_solve_provenance"]["critical_flag_blocks_credit"] is True
    assert artifact["arc_registry_count_before"] == 177
    assert artifact["arc_registry_count_after"] == 177

    assert artifact["one_axis_replica_exchange_preserved"] is True
    assert artifact["two_axis_invariant_status"]["promoted"] is True
    assert artifact["two_axis_quality_status"]["promoted"] is False
    assert artifact["rust_parity_status"]["promoted"] is False
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    critical_flags = artifact["adversarial_verification_summary"]["critical_flags"]
    assert any(row["task_id"] == "exp5643-arc-live-self-discovery-levelup-v509" for row in critical_flags)
    assert any(row["scope"] == "two_axis_quality_extension_exp5645" for row in artifact["retirements_applied"])
    assert "ops/status.md" in artifact["ops_reconciliation"]["delegated_by_stop_rule"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5647_missing_and_malformed_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5647-MISSING-MALFORMED: bad inputs block promotion."""

    missing = mod.EXP5639_AUDIT_PATH
    malformed = mod.EXP5644_TWO_AXIS_EXACT_PATH
    _make_root(tmp_path, omit=missing, malformed=malformed)

    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["malformed_artifacts"] == [malformed.as_posix()]
    assert artifact["fr11_independent_promotion_status"]["promoted"] is False
    assert artifact["two_axis_invariant_status"]["promoted"] is False
    assert artifact["two_axis_quality_status"]["promoted"] is False
    assert artifact["rust_parity_status"]["promoted"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5647_validation_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5647-FIELD-PRINCIPLES: overclaims are invalid."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "upstream_artifacts" in mod.validate_artifact({**artifact, "upstream_artifacts": []})
    assert "upstream_gate_statuses" in mod.validate_artifact({**artifact, "upstream_gate_statuses": {}})
    assert "terminal_status_by_task" in mod.validate_artifact(
        {**artifact, "terminal_status_by_task": {"only": {}}}
    )
    assert "fr11_schema_corrigendum_status" in mod.validate_artifact(
        {**artifact, "fr11_schema_corrigendum_status": {"promoted_as_science": True}}
    )
    assert "fr11_independent_promotion_status" in mod.validate_artifact(
        {**artifact, "fr11_independent_promotion_status": {"promoted": False}}
    )
    assert "arc_registry_count_before" in mod.validate_artifact(
        {**artifact, "arc_registry_count_before": "177"}
    )
    assert "arc_registry_count_after" in mod.validate_artifact(
        {**artifact, "arc_registry_count_after": "177"}
    )
    bad_solve = dict(artifact["arc_solve_provenance"])
    bad_solve["registry_delta"] = 1
    assert "arc_solve_provenance" in mod.validate_artifact(
        {**artifact, "arc_solve_provenance": bad_solve}
    )
    bad_solve = dict(artifact["arc_solve_provenance"])
    bad_solve["solve_credited"] = True
    assert "arc_solve_provenance" in mod.validate_artifact(
        {**artifact, "arc_solve_provenance": bad_solve}
    )
    assert "one_axis_replica_exchange_preserved" in mod.validate_artifact(
        {**artifact, "one_axis_replica_exchange_preserved": False}
    )
    assert "fr11_shadow_integration_status" in mod.validate_artifact(
        {
            **artifact,
            "fr11_shadow_integration_status": {
                "default_enabled": True,
                "automatic_production_enablement": True,
            },
        }
    )
    assert "arc_executable_model_status" in mod.validate_artifact(
        {**artifact, "arc_executable_model_status": {"promoted": True}}
    )
    assert "two_axis_invariant_status" in mod.validate_artifact(
        {**artifact, "two_axis_invariant_status": {"promoted": False}}
    )
    assert "two_axis_quality_status" in mod.validate_artifact(
        {**artifact, "two_axis_quality_status": {"promoted": True}}
    )
    assert "rust_parity_status" in mod.validate_artifact(
        {**artifact, "rust_parity_status": {"promoted": True}}
    )
    assert "hardware_speedup_claimed" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claimed": True}
    )
    assert "timing_claimed" in mod.validate_artifact({**artifact, "timing_claimed": True})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "new_inference"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})


def test_scenario_capstone_5647_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5647-FIELD-PRINCIPLES: helpers fail closed."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    payload, meta = mod._read_json_any(list_json)
    assert payload == {}
    assert meta["error"] == "not_json_object"
    missing_payload, missing_meta = mod._read_json_any(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"

    assert mod._severity_rank("critical") > mod._severity_rank("warn")
    assert mod._max_severity([{"severity": "warn"}, {"severity": "critical"}]) == "critical"
    assert mod._max_severity(["bad", {"severity": "info"}]) == "info"
    assert mod._max_severity("bad") == "none"
    assert mod._number({"value": "1.5"}, "value") == 1.5
    assert mod._number({"value": "bad"}, "value") == 0.0
    assert mod._number({"value": []}, "value") == 0.0
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "-2"}, "value") == -2
    assert mod._int({"value": "bad"}, "value") == 0
    assert mod._bool({"value": "true"}, "value") is True
    assert mod._bool({"value": "false"}, "value") is False
    assert mod._bool({"value": 1}, "value") is True
    assert mod._is_gate_skip({"schema": "blocked_gate_check_v1"}) is True
    assert mod._is_blocked({"honest_verdict": "blocked: x"}) is True
    assert mod._is_complete({"honest_verdict": "complete: x"}) is True
    assert mod._status_for_payload({}, {"exists": False}) == "missing"
    assert mod._status_for_payload({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._status_for_payload({"honest_verdict": "unclear"}, {"exists": True, "loadable": True})
        == "unknown"
    )
    assert mod._load_validation_results(None) == mod.DEFAULT_VALIDATION_RESULTS
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps([{"command": "unit", "exit_code": 0}, "ignored"]) + "\n",
        encoding="utf-8",
    )
    assert mod._load_validation_results(validation_path) == [{"command": "unit", "exit_code": 0}]
    bad_validation_path = tmp_path / "bad_validation.json"
    bad_validation_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_validation_results(bad_validation_path)
    benefit_pass, benefit_receipt = mod._benefit_gate({"paired_benefit_intervals": {"bad": 1}})
    assert benefit_pass is False
    assert benefit_receipt["minimum_lower"] is None
    pathwise_pass, pathwise_receipt = mod._pathwise_gate(
        {
            "pathwise_risk_upper_bound": {
                "risk_limit": 0.1,
                "risk_intervals": {"a": {"upper": 0.05, "within_limit": True}},
            }
        }
    )
    assert pathwise_pass is True
    assert pathwise_receipt["max_upper"] == 0.05


def test_scenario_capstone_5647_branch_decisions_are_evidence_based() -> None:
    """SCENARIO-CAPSTONE-5647: gate reasons follow observed evidence."""

    payloads = _payloads()
    payloads[mod.EXP5639_AUDIT_PATH] = {
        **payloads[mod.EXP5639_AUDIT_PATH],
        "paired_benefit_intervals": {"bad": {"lower": 0.0}},
    }
    assert (
        mod._derive_fr11_independent(_artifact_map(payloads))["failed_condition"]
        == "benefit_interval_lower_bound_not_positive"
    )

    payloads = _payloads()
    payloads[mod.EXP5639_AUDIT_PATH]["pathwise_risk"]["risk_intervals"]["worst"][
        "within_limit"
    ] = False
    assert (
        mod._derive_fr11_independent(_artifact_map(payloads))["failed_condition"]
        == "pathwise_risk_gate_failed"
    )

    payloads = _payloads()
    payloads[mod.EXP5639_AUDIT_PATH]["worst_group_coverage"]["coverage"] = 0.8
    assert (
        mod._derive_fr11_independent(_artifact_map(payloads))["failed_condition"]
        == "worst_group_coverage_gate_failed"
    )

    payloads = _payloads()
    payloads[mod.EXP5639_AUDIT_PATH]["critical_flag_count"] = 1
    assert (
        mod._derive_fr11_independent(_artifact_map(payloads))["failed_condition"]
        == "critical_flag_count_nonzero"
    )

    payloads = _payloads()
    payloads[mod.EXP5642_ARC_LIVE_AB_PATH] = {
        "honest_verdict": "complete: live ab ready",
        "live_reachability_pass": True,
        "known_level_utility_pass": True,
        "unsafe_model_accept_count": 0,
    }
    payloads[mod.EXP5641_ARC_MODEL_PATH] = {
        **payloads[mod.EXP5641_ARC_MODEL_PATH],
        "status": "complete",
        "honest_verdict": "complete: executable model ready",
        "executable_model_ready_score": 1.0,
        "patched_vs_unpatched_error_reduction_interval": {"lower": 0.01},
    }
    arc = mod._derive_arc_executable(_artifact_map(payloads))
    assert arc["promoted"] is True

    payloads = _payloads()
    payloads[mod.EXP5641_ARC_MODEL_PATH] = {
        **payloads[mod.EXP5641_ARC_MODEL_PATH],
        "status": "complete",
        "honest_verdict": "complete: ready score but no utility",
        "executable_model_ready_score": 1.0,
        "patched_vs_unpatched_error_reduction_interval": {"lower": 0.0},
    }
    payloads[mod.EXP5642_ARC_LIVE_AB_PATH] = {
        "status": "complete",
        "honest_verdict": "complete: live ab ready",
        "live_reachability_pass": True,
        "known_level_utility_pass": True,
        "unsafe_model_accept_count": 0,
    }
    assert (
        mod._derive_arc_executable(_artifact_map(payloads))["failed_condition"]
        == "known_level_utility_not_positive"
    )

    payloads[mod.EXP5641_ARC_MODEL_PATH]["patched_vs_unpatched_error_reduction_interval"] = {
        "lower": 0.01
    }
    payloads[mod.EXP5641_ARC_MODEL_PATH]["heldout_transition_error_by_arm"] = []
    payloads[mod.EXP5642_ARC_LIVE_AB_PATH]["live_reachability_pass"] = False
    assert (
        mod._derive_arc_executable(_artifact_map(payloads))["failed_condition"]
        == "exp5642_live_reachability_not_passed"
    )

    payloads[mod.EXP5642_ARC_LIVE_AB_PATH]["live_reachability_pass"] = True
    payloads[mod.EXP5642_ARC_LIVE_AB_PATH]["known_level_utility_pass"] = False
    assert (
        mod._derive_arc_executable(_artifact_map(payloads))["failed_condition"]
        == "exp5642_known_level_utility_not_passed"
    )

    payloads[mod.EXP5642_ARC_LIVE_AB_PATH]["known_level_utility_pass"] = True
    payloads[mod.EXP5641_ARC_MODEL_PATH]["all_receipt_replay_pass"] = False
    assert (
        mod._derive_arc_executable(_artifact_map(payloads))["failed_condition"]
        == "exact_replay_or_zero_unsafe_gate_failed"
    )

    payloads[mod.EXP5643_ARC_LEVEL_PATH] = {
        **payloads[mod.EXP5643_ARC_LEVEL_PATH],
        "flagged_adversarial": False,
        "corrigendum_pending": [],
        "offline_reproduced": True,
        "independent_generic_reproduction": True,
        "registry_delta": 1,
        "registry_count_after": 178,
        "registry_updated": True,
    }
    solve = mod._derive_arc_solve(_artifact_map(payloads))
    assert solve["solve_credited"] is True

    payloads[mod.EXP5645_TWO_AXIS_QUALITY_PATH] = {
        **payloads[mod.EXP5645_TWO_AXIS_QUALITY_PATH],
        "status": "complete",
        "honest_verdict": "complete: two-axis quality ready",
        "two_axis_quality_ready_score": 1.0,
        "material_quality_regression_count": 0,
    }
    payloads[mod.EXP5645_TWO_AXIS_QUALITY_PATH]["target_diagnostics"] = []
    two_axis = mod._derive_two_axis(_artifact_map(payloads))
    assert two_axis["two_axis_quality_status"]["promoted"] is False
    payloads[mod.EXP5645_TWO_AXIS_QUALITY_PATH]["target_diagnostics"] = {
        "within_exactness_bounds": True
    }
    payloads[mod.EXP5646_RUST_PARITY_PATH] = {
        "status": "complete",
        "honest_verdict": "complete: rust parity ready",
        "rust_parity_ready_score": 1.0,
        "parity_pass": True,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
    }
    two_axis = mod._derive_two_axis(_artifact_map(payloads))
    assert two_axis["two_axis_quality_status"]["promoted"] is True
    assert two_axis["rust_parity_status"]["promoted"] is True


def test_scenario_capstone_5647_writer_and_cli_emit_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5647: writer and CLI persist the deliverable."""

    _make_root(tmp_path)
    validation = [{"command": "unit", "exit_code": 0, "status": "passed"}]

    artifact = mod.write_capstone(
        root=tmp_path,
        validation_results=validation,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
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


def test_scenario_capstone_5647_validation_and_cli_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-5647-FIELD-PRINCIPLES: invalid artifacts are rejected."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    bad_principles = dict(artifact["field_principles"])
    bad_principles["honest_verdict"] = "wrong"
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": bad_principles}
    )
    assert "adversarial_verification_summary" in mod.validate_artifact(
        {**artifact, "adversarial_verification_summary": []}
    )
    assert "retirements_applied" in mod.validate_artifact(
        {**artifact, "retirements_applied": "none"}
    )
    assert "test_exit_codes" in mod.validate_artifact({**artifact, "test_exit_codes": []})
    assert "reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )

    monkeypatch.setattr(mod, "run_capstone", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError):
        mod.write_capstone(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
