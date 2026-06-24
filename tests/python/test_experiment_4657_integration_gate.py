"""Tests for Exp 4657 submitted A1/A2 integration gate.

Spec refs: REQ-ARC-WMTE-4657, SCENARIO-ARC-WMTE-4657.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _submitted_config(*, value_weight: float = 1e-12) -> dict[str, Any]:
    return {
        "policy": "E3AgentPolicy",
        "target_levels": 3,
        "value_weight": value_weight,
        "value_head_feature_subset": "cross_game_features_v3:v2_plus_frame_delta",
        "verifier_is_oracle": False,
        "qd_generation_enabled": False,
        "qd_generation_mode": "energy_fitness_map_elites_sequence_generator",
        "live_submit_package_path": "results/experiment_4643_submission_package_operator_resubmit.json",
    }


def _a1_artifact(*, value_weight: float = 1e-12) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration.",
        "verifier_is_oracle": False,
        "value_weight_set": value_weight,
        "feature_subset": "cross_game_features_v3:v2_plus_frame_delta",
        "per_node_feature_cost_ms": 0.64,
        "sim_timed_out": False,
        "live_first_win_rate_value_routed": 0.04,
        "live_solve_rate_value_routed": 0.0,
        "live_baseline_value_weight_zero": {"first_win_rate": 0.04, "solve_rate": 0.0},
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
        "chosen_submitted_config": "unchanged",
    }


def _a2_artifact(*, chosen: Any = "unchanged") -> dict[str, Any]:
    return {
        "honest_verdict": "complete: energy_fitness_qd_no_winner_generated_honest_null_gap_sharpened",
        "verifier_is_oracle": False,
        "chosen_submitted_config": chosen,
        "live_solve_rate_qd": 0.0,
        "live_solve_rate_search_baseline": 0.0,
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
    }


def _package_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "success: package_refreshed_live_submittable_57_above_33",
        "live_submittable_level_count": 57,
        "ready_for_operator_submit": True,
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "submitted_agent_import": True,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "package_artifact_present": True,
        "spec_has_req_4657": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4657_spec_declares_integration_contract() -> None:
    """REQ-ARC-WMTE-4657: OpenSpec declares the integration artifact schema."""

    from carnot import experiment_4657_integration_gate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4657" in spec
    assert "SCENARIO-ARC-WMTE-4657" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4657_audits_a1_integrated_a2_unchanged() -> None:
    """REQ-ARC-WMTE-4657: A1 cost-fix ships while null A2 stays disabled."""

    from carnot import experiment_4657_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(),
    )

    assert audit["levers_integrated"] == ["A1_value_routing_cost_fix"]
    assert audit["config_integrated"]["value_weight"] == 1e-12
    assert audit["config_integrated"]["value_head_feature_subset"] == (
        "cross_game_features_v3:v2_plus_frame_delta"
    )
    assert audit["a1"]["reason"] == "submitted_config_matches_a1_cost_fix"
    assert audit["a2"]["integrated"] is False
    assert audit["a2"]["reason"] == "chosen_submitted_config_unchanged"

    unchanged = mod.audit_config_integration(
        a1_artifact=_a1_artifact(value_weight=0.0),
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(value_weight=0.0),
    )
    assert unchanged["levers_integrated"] == []
    assert unchanged["config_integrated"].startswith("unchanged:")
    assert unchanged["a1"]["reason"] == "a1_value_weight_not_positive"

    mismatch = mod.audit_config_integration(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(value_weight=0.0),
    )
    assert mismatch["a1"]["reason"] == "submitted_value_weight_mismatch"

    subset_bad = mod.audit_config_integration(
        a1_artifact={**_a1_artifact(), "feature_subset": "full_v3"},
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    assert subset_bad["a1"]["reason"] == "value_head_feature_subset_mismatch"

    oracle_bad = mod.audit_config_integration(
        a1_artifact={**_a1_artifact(), "verifier_is_oracle": True},
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    assert oracle_bad["a1"]["reason"] == "verifier_oracle_not_false"

    qd_disabled = mod.audit_config_integration(
        a1_artifact=_a1_artifact(value_weight=0.0),
        a2_artifact=_a2_artifact(chosen={"qd_generation_enabled": True}),
        submitted_agent_config=_submitted_config(value_weight=0.0),
    )
    assert qd_disabled["a2"]["reason"] == "submitted_qd_generation_disabled"

    qd_enabled_config = {**_submitted_config(value_weight=0.0), "qd_generation_enabled": True}
    qd_integrated = mod.audit_config_integration(
        a1_artifact=_a1_artifact(value_weight=0.0),
        a2_artifact=_a2_artifact(chosen={"qd_generation_enabled": True}),
        submitted_agent_config=qd_enabled_config,
    )
    assert qd_integrated["levers_integrated"] == ["A2_energy_fitness_qd_generator"]
    assert qd_integrated["a2"]["reason"] == "submitted_config_matches_a2_qd_generator"


def test_scenario_arc_wmte_4657_builds_success_artifact_with_no_regression() -> None:
    """SCENARIO-ARC-WMTE-4657: integrated A1 config is measured against pre-integration."""

    from carnot import experiment_4657_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    metrics = mod.measure_integrated_metrics(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        package_artifact=_package_artifact(),
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == (
        "success: integrated_a1_value_routing_cost_fix_shipped_parity_green"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["live_first_win_rate_integrated"] == 0.04
    assert artifact["live_first_win_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["live_multi_level_solve_rate_integrated"] == 0.0
    assert artifact["live_multi_level_solve_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["live_submittable_level_count_integrated"] == 57
    assert artifact["no_regression_vs_pre_integration"] is True
    assert artifact["parity_test_green"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    assert mod._as_float(True, 7.0) == 7.0
    assert mod._as_int(None, 9) == 9

    bad = dict(artifact)
    bad.pop("schema")
    bad.update(
        {
            "honest_verdict": "not-terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "parity_test_green": False,
            "live_submittable_level_count_integrated": 1,
            "field_principles": {},
            "submitted_to_leaderboard": True,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(bad)
    assert "missing required field schema" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "verifier_is_oracle_false" in errors
    assert "parity_test_green" in errors
    assert "live_submittable_level_count_integrated" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "reproducibility_checksum" in errors

    failed = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics={**metrics, "no_regression_vs_pre_integration": False},
        parity_test={"passed": False, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        duration_s=1.0,
    )
    assert failed["honest_verdict"] == "complete: integration_parity_or_regression_failed"

    qd_only_audit = {
        "levers_integrated": ["A2_energy_fitness_qd_generator"],
        "config_integrated": {"levers_integrated": ["A2_energy_fitness_qd_generator"]},
    }
    qd_only = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=qd_only_audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config={**_submitted_config(), "qd_generation_enabled": True},
        duration_s=1.0,
    )
    assert qd_only["honest_verdict"] == "success: integrated_a2_qd_generator_shipped_parity_green"

    both_audit = {
        "levers_integrated": [
            "A1_value_routing_cost_fix",
            "A2_energy_fitness_qd_generator",
        ],
        "config_integrated": {
            "levers_integrated": [
                "A1_value_routing_cost_fix",
                "A2_energy_fitness_qd_generator",
            ]
        },
    }
    both = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=both_audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config={**_submitted_config(), "qd_generation_enabled": True},
        duration_s=1.0,
    )
    assert both["honest_verdict"] == (
        "success: integrated_a1_value_routing_and_a2_qd_generator_shipped_parity_green"
    )


def test_scenario_arc_wmte_4657_reports_unchanged_when_both_levers_null() -> None:
    """SCENARIO-ARC-WMTE-4657: both-null upstream config remains explicitly unchanged."""

    from carnot import experiment_4657_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_artifact(value_weight=0.0),
        a2_artifact=_a2_artifact(),
        submitted_agent_config=_submitted_config(value_weight=0.0),
    )
    metrics = mod.measure_integrated_metrics(
        a1_artifact=_a1_artifact(value_weight=0.0),
        a2_artifact=_a2_artifact(),
        package_artifact=_package_artifact(),
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(value_weight=0.0),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"
    assert artifact["config_integrated"].startswith("unchanged:")
    assert artifact["parity_test_green"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4657_run_writes_artifact_and_blocks_missing_precondition(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4657: run writes stable JSON and fails closed when blocked."""

    from carnot import experiment_4657_integration_gate as mod

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4657\nSCENARIO-ARC-WMTE-4657\n", encoding="utf-8")
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())
    _write_json(tmp_path / mod.PACKAGE_RELATIVE_PATH, _package_artifact())

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        import_checker=lambda: {"submitted_agent_import": True},
        now=lambda: 10.0,
    )

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    assert artifact["preconditions_checked"]["a1_artifact_present"] is True
    assert artifact["parity_test_green"] is True

    blocked = mod.run(
        root=tmp_path / "missing",
        offline_arcade_checker=lambda: True,
        parity_check=lambda _root: {"passed": True},
        submitted_agent_config=_submitted_config(),
        import_checker=lambda: {"submitted_agent_import": True},
        now=lambda: 20.0,
    )
    assert blocked["honest_verdict"] == "blocked_agents_md_read"
    assert blocked["parity_test_green"] is False
    assert (tmp_path / "missing" / mod.RESULT_RELATIVE_PATH).exists()
