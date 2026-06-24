"""Tests for Exp 4681 submitted A1/A2 integration gate.

Spec refs: REQ-ARC-WMTE-4681, SCENARIO-ARC-WMTE-4681.
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
        "hierarchical_subgoal_search_enabled": False,
        "hierarchical_subgoal_budget": 3,
        "factored_planner_enabled": False,
        "factored_trust_threshold": 0.75,
        "verifier_is_oracle": False,
    }


def _previous_gate() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: integration_unchanged_both_levers_null",
        "live_first_win_rate_integrated": 0.04,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_submittable_level_count_integrated": 59,
        "config_integrated": "unchanged: fixture",
        "reproducibility_checksum": "sha256:prev",
    }


def _a1_unchanged_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating"
        ),
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "generic_agent_reached_level": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "no_subgoal_ablation_reached_level": 0,
        "random_subgoal_ablation_reached_level": 0,
        "generic_first_win_by_config": {
            "explore_budget_800": {
                "first_win_rate": 0.04,
                "multi_level_rate": 0.0,
            }
        },
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:a1",
    }


def _a2_unchanged_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: poe_world_factored_planner_no_coverage_gain_residual_logged",
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "candidate_generation_coverage_factored": 0.0,
        "candidate_generation_coverage_flat_baseline": 0.0,
        "coverage_delta": 0.0,
        "live_first_win_rate_factored": 0.0,
        "live_solve_rate_factored": 0.0,
        "live_baseline_flat_search": {"first_win_rate": 0.04, "solve_rate": 0.0},
        "first_win_rate_delta": -0.04,
        "solve_rate_delta": 0.0,
        "offline_reproduced": False,
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:a2",
    }


def _package_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4679_refresh_submission_package",
        "live_submittable_level_count": 60,
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
        "spec_has_req_4681": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4681_spec_declares_integration_contract() -> None:
    """REQ-ARC-WMTE-4681: OpenSpec declares the 4681 integration artifact schema."""

    from carnot import experiment_4681_integration_gate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4681" in spec
    assert "SCENARIO-ARC-WMTE-4681" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4681_audits_unchanged_upstream_configs() -> None:
    """REQ-ARC-WMTE-4681: unchanged A1/A2 configs keep submitted config unchanged."""

    from carnot import experiment_4681_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )

    assert audit["config_changed"] is False
    assert audit["levers_integrated"] == []
    assert audit["config_integrated"].startswith("unchanged:")
    assert audit["a1"]["reason"] == "chosen_submitted_config_unchanged"
    assert audit["a1"]["submitted_hierarchical_subgoal_search_enabled"] is False
    assert audit["a2"]["reason"] == "chosen_submitted_config_unchanged"
    assert audit["a2"]["submitted_factored_planner_enabled"] is False


def test_req_arc_wmte_4681_audits_cleared_a1_a2_configs_and_mismatches() -> None:
    """REQ-ARC-WMTE-4681: cleared subgoal/factored configs must match submitted config."""

    from carnot import experiment_4681_integration_gate as mod

    a1_chosen = {
        "hierarchical_subgoal_search_enabled": True,
        "hierarchical_subgoal_budget": 4,
    }
    a2_chosen = {"factored_planner_enabled": True, "factored_trust_threshold": 0.8}
    winning_a1 = {
        **_a1_unchanged_artifact(),
        "honest_verdict": "success: hierarchical_subgoal_generic_agent_new_level_lp85_L2",
        "chosen_submitted_config": a1_chosen,
        "generic_agent_reached_level": 2,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "no_subgoal_ablation_reached_level": 1,
        "random_subgoal_ablation_reached_level": 1,
    }
    winning_a2 = {
        **_a2_unchanged_artifact(),
        "honest_verdict": "success: poe_world_factored_planner_coverage_up_live_firstwin_lift_lp85",
        "chosen_submitted_config": a2_chosen,
        "coverage_delta": 0.2,
        "first_win_rate_delta": 0.1,
        "live_first_win_rate_factored": 0.14,
    }
    submitted = {
        **_submitted_config(),
        "hierarchical_subgoal_search_enabled": True,
        "hierarchical_subgoal_budget": 4,
        "factored_planner_enabled": True,
        "factored_trust_threshold": 0.8,
    }

    audit = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=winning_a2,
        submitted_agent_config=submitted,
    )

    assert audit["config_changed"] is True
    assert audit["levers_integrated"] == [
        "A1_hierarchical_subgoal_search",
        "A2_poe_world_factored_planner",
    ]
    assert audit["config_integrated"]["hierarchical_subgoal_budget"] == 4
    assert audit["config_integrated"]["factored_trust_threshold"] == 0.8

    a1_mismatch = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    assert a1_mismatch["a1"]["reason"] == "submitted_hierarchical_subgoal_config_mismatch"

    a1_invalid = mod.audit_config_integration(
        a1_artifact={**winning_a1, "chosen_submitted_config": {"hierarchical_subgoal_search_enabled": True}},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert a1_invalid["a1"]["reason"] == "a1_chosen_subgoal_config_invalid"

    a1_bad_type = mod.audit_config_integration(
        a1_artifact={**winning_a1, "chosen_submitted_config": ["bad"]},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert a1_bad_type["a1"]["reason"] == "chosen_submitted_config_invalid"

    a1_oracle = mod.audit_config_integration(
        a1_artifact={**winning_a1, "verifier_is_oracle": True},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert a1_oracle["a1"]["reason"] == "verifier_oracle_not_false"

    a1_unwon = mod.audit_config_integration(
        a1_artifact={**winning_a1, "offline_reproduced": False},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert a1_unwon["a1"]["reason"] == "a1_hierarchical_subgoal_gate_not_cleared"

    a2_mismatch = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=winning_a2,
        submitted_agent_config={**submitted, "factored_trust_threshold": 0.75},
    )
    assert a2_mismatch["a2"]["reason"] == "submitted_factored_planner_config_mismatch"

    a2_oracle = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "verifier_is_oracle": True},
        submitted_agent_config=submitted,
    )
    assert a2_oracle["a2"]["reason"] == "verifier_oracle_not_false"

    a2_bad_type = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "chosen_submitted_config": ["bad"]},
        submitted_agent_config=submitted,
    )
    assert a2_bad_type["a2"]["reason"] == "chosen_submitted_config_invalid"

    a2_unwon = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "coverage_delta": 0.0},
        submitted_agent_config=submitted,
    )
    assert a2_unwon["a2"]["reason"] == "a2_factored_planner_gate_not_cleared"

    a2_invalid = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "chosen_submitted_config": {"factored_planner_enabled": True}},
        submitted_agent_config=submitted,
    )
    assert a2_invalid["a2"]["reason"] == "a2_chosen_factored_planner_config_invalid"


def test_scenario_arc_wmte_4681_builds_unchanged_artifact_with_tautology_guard() -> None:
    """SCENARIO-ARC-WMTE-4681: unchanged integration records identical metrics honestly."""

    from carnot import experiment_4681_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    metrics = mod.measure_integrated_metrics(
        previous_gate_artifact=_previous_gate(),
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        package_artifact=_package_artifact(),
        config_changed=audit["config_changed"],
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        source_artifacts={
            "previous_gate": mod.PREVIOUS_GATE_RELATIVE_PATH,
            "a1": mod.A1_RELATIVE_PATH,
            "a2": mod.A2_RELATIVE_PATH,
            "package": mod.PACKAGE_RELATIVE_PATH,
        },
        source_artifact_checksums={"previous_gate": "sha256:prev", "a1": "sha256:a1"},
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"
    assert artifact["config_changed"] is False
    assert artifact["config_integrated"].startswith("unchanged:")
    assert artifact["live_first_win_rate_integrated"] == 0.04
    assert artifact["live_multi_level_solve_rate_integrated"] == 0.0
    assert artifact["live_submittable_level_count_integrated"] == 59
    assert artifact["live_first_win_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["live_multi_level_solve_rate_delta_vs_pre_integration"] == 0.0
    assert "identical BY CONSTRUCTION" in artifact["tautology_guard"]
    assert artifact["parity_test_green"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad.pop("schema")
    bad.update(
        {
            "honest_verdict": "not-terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "config_changed": "false",
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
    assert "config_changed_bool" in errors
    assert "parity_test_green" in errors
    assert "live_submittable_level_count_integrated" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "reproducibility_checksum" in errors


def test_scenario_arc_wmte_4681_success_metrics_and_defensive_helpers() -> None:
    """SCENARIO-ARC-WMTE-4681: changed configs get success verdicts only without regression."""

    from carnot import experiment_4681_integration_gate as mod

    audit = {
        "config_changed": True,
        "levers_integrated": ["A1_hierarchical_subgoal_search"],
        "config_integrated": {"levers_integrated": ["A1_hierarchical_subgoal_search"]},
    }
    metrics = {
        "live_first_win_rate_integrated": 0.08,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.04,
        "live_multi_level_solve_rate_integrated": 0.2,
        "live_multi_level_solve_rate_pre_integration": 0.0,
        "live_multi_level_solve_rate_delta_vs_pre_integration": 0.2,
        "live_submittable_level_count_integrated": 60,
        "no_regression_vs_pre_integration": True,
        "metric_measurement_note": "fixture",
    }

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert artifact["honest_verdict"] == (
        "success: integrated_a1_hierarchical_subgoal_search_shipped_parity_green"
    )

    failed = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics={**metrics, "no_regression_vs_pre_integration": False},
        parity_test={"passed": False},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert failed["honest_verdict"] == "complete: integration_parity_or_regression_failed"
    assert "no_regression_vs_pre_integration" in mod.artifact_schema_errors(failed)

    both = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit={
            **audit,
            "levers_integrated": [
                "A1_hierarchical_subgoal_search",
                "A2_poe_world_factored_planner",
            ],
        },
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert both["honest_verdict"] == (
        "success: integrated_a1_subgoal_search_and_a2_factored_planner_shipped_parity_green"
    )

    a2_only = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit={
            **audit,
            "levers_integrated": ["A2_poe_world_factored_planner"],
        },
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert (
        a2_only["honest_verdict"]
        == "success: integrated_a2_poe_world_factored_planner_shipped_parity_green"
    )

    measured = mod.measure_integrated_metrics(
        previous_gate_artifact=_previous_gate(),
        a1_artifact={
            **_a1_unchanged_artifact(),
            "generic_first_win_by_config": {"arm": {"first_win_rate": 0.08, "multi_level_rate": 0.2}},
        },
        a2_artifact={
            **_a2_unchanged_artifact(),
            "live_first_win_rate_factored": 0.09,
            "live_solve_rate_factored": 0.1,
        },
        package_artifact={},
        config_changed=True,
    )
    assert measured["live_first_win_rate_integrated"] == 0.09
    assert measured["live_multi_level_solve_rate_integrated"] == 0.2
    assert measured["live_submittable_level_count_integrated"] == 59
    assert mod._max_config_metric({"generic_first_win_by_config": []}, "first_win_rate") == 0.0
    assert mod._as_float(True, 7.0) == 7.0
    assert mod._as_float("bad", 7.0) == 7.0
    assert mod._as_float(float("nan"), 7.0) == 7.0
    assert mod._as_int(None, 9) == 9
    assert mod._as_int("bad", 9) == 9


def test_scenario_arc_wmte_4681_run_writes_artifact_and_blocks(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4681: run writes the stable result artifact."""

    from carnot import experiment_4681_integration_gate as mod

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.SPEC_RELATIVE_PATH).write_text("REQ-ARC-WMTE-4681\n", encoding="utf-8")
    _write_json(tmp_path / mod.PREVIOUS_GATE_RELATIVE_PATH, _previous_gate())
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_unchanged_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_unchanged_artifact())
    _write_json(tmp_path / mod.PACKAGE_RELATIVE_PATH, _package_artifact())

    artifact = mod.run(
        tmp_path,
        import_checker=lambda: {"submitted_agent_import": True},
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        now=iter([10.0, 10.5]).__next__,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["config_changed"] is False
    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"

    (tmp_path / mod.A2_RELATIVE_PATH).unlink()
    blocked = mod.run(
        tmp_path,
        import_checker=lambda: {"submitted_agent_import": True},
        parity_check=lambda _root: {"passed": True},
        submitted_agent_config=_submitted_config(),
        now=iter([20.0, 20.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_a2_artifact_present"
    assert blocked["preconditions_checked"]["blocked_resource"] == "a2_artifact_present"

    import_blocked = mod.check_preconditions(
        tmp_path,
        import_checker=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert import_blocked["submitted_agent_import"] is False
    assert "boom" in import_blocked["submitted_agent_import_error"]
