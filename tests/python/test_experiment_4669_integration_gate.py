"""Tests for Exp 4669 ARC sprint submitted A1/A2 integration gate.

Spec refs: REQ-ARC-WMTE-4669, SCENARIO-ARC-WMTE-4669.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _submitted_config() -> dict[str, Any]:
    return {
        "policy": "E3AgentPolicy",
        "target_levels": 3,
        "value_weight": 1e-12,
        "value_head_feature_subset": "cross_game_features_v3:v2_plus_frame_delta",
        "value_head_checkpoint": "models/arc_dagger_value_routing_v3.json",
        "value_head_distribution_corrected": True,
        "verifier_is_oracle": False,
        "qd_generation_enabled": False,
    }


def _previous_gate() -> dict[str, Any]:
    return {
        "honest_verdict": "success: integrated_a1_value_routing_cost_fix_shipped_parity_green",
        "live_first_win_rate_integrated": 0.04,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_submittable_level_count_integrated": 57,
        "config_integrated": {"levers_integrated": ["A1_value_routing_cost_fix"]},
        "reproducibility_checksum": "sha256:prev",
    }


def _a1_null_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: l2_goal_induction_no_deepening_residual_"
            "single_exemplar_goal_insufficient"
        ),
        "verifier_is_oracle": False,
        "chosen_submitted_config": None,
        "win_state_exemplar_injected": False,
        "goal_predicate_satisfiable": {"lp85": False, "sc25": False},
        "l2_plan_reaches_goal": {"lp85": False, "sc25": False},
        "generic_agent_reached_level": {"lp85": 1, "sc25": 0},
        "metric_harness_fixed": {
            "target_levels": 2,
            "break_at_first_win": False,
            "qwen_port_props_verified": True,
        },
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:a1",
    }


def _a2_unchanged_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: dagger_distribution_corrected_no_live_lift_residual_logged.",
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "live_first_win_rate_corrected": 0.04,
        "live_solve_rate_corrected": 0.0,
        "live_baseline_winning_path_trained": {"first_win_rate": 0.04, "solve_rate": 0.0},
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
        "model_checkpoint": "models/arc_dagger_value_routing_v3.json",
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:a2",
    }


def _package_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4667_refresh_submission_package",
        "live_submittable_level_count": 59,
        "ready_for_operator_submit": True,
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "submitted_agent_import": True,
        "previous_gate_artifact_present": True,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "package_artifact_present": True,
        "spec_has_req_4669": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4669_spec_declares_integration_contract() -> None:
    """REQ-ARC-WMTE-4669: OpenSpec declares the 4669 integration artifact schema."""

    from carnot import experiment_4669_integration_gate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4669" in spec
    assert "SCENARIO-ARC-WMTE-4669" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4669_audits_null_a1_and_unchanged_a2() -> None:
    """REQ-ARC-WMTE-4669: upstream null levers keep submitted config unchanged."""

    from carnot import experiment_4669_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )

    assert audit["levers_integrated"] == []
    assert audit["config_integrated"].startswith("unchanged:")
    assert audit["a1"]["integrated"] is False
    assert audit["a1"]["reason"] == "chosen_submitted_config_null_or_no_l2_goal_induction_win"
    assert audit["a1"]["metric_harness_fixed"] is True
    assert audit["a2"]["integrated"] is False
    assert audit["a2"]["reason"] == "chosen_submitted_config_unchanged"
    assert audit["a2"]["submitted_value_head_distribution_corrected"] is True

    a1_chosen = mod.audit_config_integration(
        a1_artifact={
            **_a1_null_artifact(),
            "chosen_submitted_config": {
                "level_up_reinduction_win_state_exemplar": True,
                "goal_predicate_satisfiability_check": True,
                "metric_harness_fixed": True,
            },
            "win_state_exemplar_injected": True,
            "goal_predicate_satisfiable": {"lp85": True},
            "l2_plan_reaches_goal": {"lp85": True},
            "generic_agent_reached_level": {"lp85": 2},
        },
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config={
            **_submitted_config(),
            "level_up_reinduction_win_state_exemplar": True,
            "goal_predicate_satisfiability_check": True,
        },
    )
    assert a1_chosen["levers_integrated"] == ["A1_l2_goal_induction"]
    assert a1_chosen["a1"]["reason"] == "submitted_config_matches_a1_l2_goal_induction"


def test_req_arc_wmte_4669_audits_distribution_corrected_value_config() -> None:
    """REQ-ARC-WMTE-4669: A2 config must match checkpoint, weight, and feature subset."""

    from carnot import experiment_4669_integration_gate as mod

    chosen = {
        "value_weight": 1e-12,
        "value_head_feature_subset": "cross_game_features_v3:v2_plus_frame_delta",
        "value_head_checkpoint": "models/arc_dagger_value_routing_v3.json",
        "value_head_distribution_corrected": True,
    }
    audit = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={**_a2_unchanged_artifact(), "chosen_submitted_config": chosen},
        submitted_agent_config=_submitted_config(),
    )

    assert audit["levers_integrated"] == ["A2_distribution_corrected_value_head"]
    assert audit["config_integrated"]["value_weight"] == 1e-12
    assert audit["a2"]["reason"] == "submitted_config_matches_a2_distribution_corrected_value"

    mismatch = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={**_a2_unchanged_artifact(), "chosen_submitted_config": chosen},
        submitted_agent_config={**_submitted_config(), "value_weight": 0.0},
    )
    assert mismatch["a2"]["reason"] == "submitted_value_weight_mismatch"

    checkpoint_bad = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={**_a2_unchanged_artifact(), "chosen_submitted_config": chosen},
        submitted_agent_config={**_submitted_config(), "value_head_checkpoint": "models/base.json"},
    )
    assert checkpoint_bad["a2"]["reason"] == "submitted_value_head_checkpoint_mismatch"

    oracle_bad = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={**_a2_unchanged_artifact(), "chosen_submitted_config": chosen},
        submitted_agent_config={**_submitted_config(), "verifier_is_oracle": True},
    )
    assert oracle_bad["a2"]["reason"] == "verifier_oracle_not_false"

    invalid = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={**_a2_unchanged_artifact(), "chosen_submitted_config": ["bad"]},
        submitted_agent_config=_submitted_config(),
    )
    assert invalid["a2"]["reason"] == "chosen_submitted_config_invalid"

    nonpositive = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={
            **_a2_unchanged_artifact(),
            "chosen_submitted_config": {**chosen, "value_weight": 0.0},
        },
        submitted_agent_config=_submitted_config(),
    )
    assert nonpositive["a2"]["reason"] == "a2_value_weight_not_positive"

    subset_bad = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={
            **_a2_unchanged_artifact(),
            "chosen_submitted_config": {**chosen, "value_head_feature_subset": "full_v3"},
        },
        submitted_agent_config=_submitted_config(),
    )
    assert subset_bad["a2"]["reason"] == "value_head_feature_subset_mismatch"

    flag_bad = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact={
            **_a2_unchanged_artifact(),
            "chosen_submitted_config": {**chosen, "value_head_distribution_corrected": False},
        },
        submitted_agent_config=_submitted_config(),
    )
    assert flag_bad["a2"]["reason"] == "submitted_distribution_corrected_flag_mismatch"


def test_req_arc_wmte_4669_audits_l2_goal_induction_mismatches() -> None:
    """REQ-ARC-WMTE-4669: A1 rejects oracle, unwon, and mismatched configs."""

    from carnot import experiment_4669_integration_gate as mod

    chosen = {
        "level_up_reinduction_win_state_exemplar": True,
        "goal_predicate_satisfiability_check": True,
        "metric_harness_fixed": True,
    }
    winning_a1 = {
        **_a1_null_artifact(),
        "chosen_submitted_config": chosen,
        "win_state_exemplar_injected": True,
        "goal_predicate_satisfiable": True,
        "l2_plan_reaches_goal": True,
        "generic_agent_reached_level": 2,
    }
    submitted = {
        **_submitted_config(),
        "level_up_reinduction_win_state_exemplar": True,
        "goal_predicate_satisfiability_check": True,
    }

    oracle_bad = mod.audit_config_integration(
        a1_artifact={**winning_a1, "verifier_is_oracle": True},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert oracle_bad["a1"]["reason"] == "verifier_oracle_not_false"

    unwon = mod.audit_config_integration(
        a1_artifact={**winning_a1, "goal_predicate_satisfiable": False},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert unwon["a1"]["reason"] == "a1_upstream_l2_goal_induction_not_winning"

    mismatch = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    assert mismatch["a1"]["reason"] == "submitted_l2_goal_induction_config_mismatch"


def test_scenario_arc_wmte_4669_builds_unchanged_artifact_with_no_regression() -> None:
    """SCENARIO-ARC-WMTE-4669: null A1/A2 still re-measure with parity and no regression."""

    from carnot import experiment_4669_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_null_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    metrics = mod.measure_integrated_metrics(
        previous_gate_artifact=_previous_gate(),
        a1_artifact=_a1_null_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        package_artifact=_package_artifact(),
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        source_artifacts={
            "previous_gate": "results/experiment_4657_integration_gate.json",
            "a1": "results/experiment_4664_l2_goal_predicate_induction_live.json",
            "a2": "results/experiment_4665_dagger_distribution_shift_value_routing.json",
        },
        source_artifact_checksums={"previous_gate": "sha256:prev", "a1": "sha256:a1"},
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["config_integrated"].startswith("unchanged:")
    assert artifact["live_first_win_rate_integrated"] == 0.04
    assert artifact["live_multi_level_solve_rate_integrated"] == 0.0
    assert artifact["live_submittable_level_count_integrated"] == 59
    assert artifact["live_first_win_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["live_multi_level_solve_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["no_regression_vs_pre_integration"] is True
    assert artifact["parity_test_green"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    failed = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit={**audit, "levers_integrated": ["A2_distribution_corrected_value_head"]},
        metrics={**metrics, "no_regression_vs_pre_integration": False},
        parity_test={"passed": False, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert failed["honest_verdict"] == "complete: integration_parity_or_regression_failed"

    bad = dict(artifact)
    bad.pop("schema")
    bad.update(
        {
            "honest_verdict": "not-terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "parity_test_green": False,
            "live_submittable_level_count_integrated": 33,
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

    no_regression_bad = dict(artifact)
    no_regression_bad["no_regression_vs_pre_integration"] = False
    no_regression_bad["reproducibility_checksum"] = mod.payload_checksum(no_regression_bad)
    assert "no_regression_vs_pre_integration" in mod.artifact_schema_errors(no_regression_bad)


def test_scenario_arc_wmte_4669_success_verdict_variants_and_defensive_helpers() -> None:
    """SCENARIO-ARC-WMTE-4669: success verdicts are stable for each integrated lever."""

    from carnot import experiment_4669_integration_gate as mod

    metrics = {
        "live_first_win_rate_integrated": 0.08,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.04,
        "live_multi_level_solve_rate_integrated": 0.2,
        "live_multi_level_solve_rate_pre_integration": 0.0,
        "live_multi_level_solve_rate_delta_vs_pre_integration": 0.2,
        "live_submittable_level_count_integrated": 59,
        "no_regression_vs_pre_integration": True,
    }

    a1_only = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit={
            "levers_integrated": ["A1_l2_goal_induction"],
            "config_integrated": {"levers_integrated": ["A1_l2_goal_induction"]},
        },
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert a1_only["honest_verdict"] == "success: integrated_a1_l2_goal_induction_shipped_parity_green"

    a2_only = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit={
            "levers_integrated": ["A2_distribution_corrected_value_head"],
            "config_integrated": {
                "levers_integrated": ["A2_distribution_corrected_value_head"]
            },
        },
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert a2_only["honest_verdict"] == (
        "success: integrated_a2_distribution_corrected_value_head_shipped_parity_green"
    )

    both = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit={
            "levers_integrated": ["A1_l2_goal_induction", "A2_distribution_corrected_value_head"],
            "config_integrated": {
                "levers_integrated": [
                    "A1_l2_goal_induction",
                    "A2_distribution_corrected_value_head",
                ]
            },
        },
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert both["honest_verdict"] == (
        "success: integrated_a1_l2_goal_induction_and_a2_value_head_shipped_parity_green"
    )

    empty_multi = mod.measure_integrated_metrics(
        previous_gate_artifact={**_previous_gate(), "live_submittable_level_count_integrated": 58},
        a1_artifact={"generic_agent_reached_level": {}},
        a2_artifact={},
        package_artifact={},
    )
    assert empty_multi["live_submittable_level_count_integrated"] == 58

    assert mod._as_float(True, 7.0) == 7.0
    assert mod._as_float("bad", 7.0) == 7.0
    assert mod._as_float(float("nan"), 7.0) == 7.0
    assert mod._as_int(None, 9) == 9
    assert mod._as_int("bad", 9) == 9
    assert mod._truthy_values(False) == [False]
    assert mod._max_int_value("2") == 2


def test_scenario_arc_wmte_4669_run_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4669: run writes the stable result artifact."""

    from carnot import experiment_4669_integration_gate as mod

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.SPEC_RELATIVE_PATH).write_text("REQ-ARC-WMTE-4669\n", encoding="utf-8")
    _write_json(tmp_path / mod.PREVIOUS_GATE_RELATIVE_PATH, _previous_gate())
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_null_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_unchanged_artifact())
    _write_json(tmp_path / mod.PACKAGE_RELATIVE_PATH, _package_artifact())

    artifact = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: True,
        import_checker=lambda: {"submitted_agent_import": True},
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        now=iter([10.0, 10.5]).__next__,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"

    blocked = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        import_checker=lambda: {"submitted_agent_import": True},
        parity_check=lambda _root: {"passed": True},
        submitted_agent_config=_submitted_config(),
        now=iter([20.0, 20.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["preconditions_checked"]["blocked_resource"] == "offline_arcade"
