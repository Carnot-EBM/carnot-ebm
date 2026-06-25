"""Tests for Exp 4705 submitted object-centric/amortized integration gate.

Spec refs: REQ-ARC-WMTE-4705, SCENARIO-ARC-WMTE-4705.
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
        "object_centric_proposal_enabled": False,
        "object_centric_proposal_mode": "connected_component_slots_plus_relational_gaps",
        "object_centric_neighborhood_radius": 2,
        "amortized_first_contact_prior_enabled": False,
        "amortized_first_contact_prior_mode": (
            "frequency_prior_from_cross_game_first_contact_traces"
        ),
        "go_explore_archive_enabled": False,
        "go_explore_archive_mode": "return_then_explore_replayable_prefix_archive",
        "verifier_is_oracle": False,
    }


def _a1_unchanged_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: object_centric_perception_no_new_level_residual_"
            "offpath_calibration_insufficient"
        ),
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "generic_agent_reached_level": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "order1_ablation_reached_level": 0,
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:a1",
    }


def _a2_unchanged_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: amortized_prior_go_explore_no_coverage_gain_residual_logged",
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "coverage_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "offline_reproduced": False,
        "no_prior_ablation_failed": False,
        "go_explore_now_live_reachable": True,
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:a2",
    }


def _scored_measurement() -> dict[str, Any]:
    return {
        "first_win_rate": 0.04,
        "multi_level_deepen_rate": 0.0,
        "scored_lane": {
            "integrated_measurement": {
                "first_win_rate": 0.04,
                "variant_attempts_count": 25,
                "variant_attempts": [{"attempted": True, "solved": False}],
            },
            "bare_measurement": {"first_win_rate": 0.04, "variant_attempts_count": 25},
            "deepening_summary": {"multi_level_solve_rate": 0.0},
            "variant_ids": [1],
            "budget": 200,
        },
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "submitted_agent_import": True,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "spec_has_req_4705": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4705_spec_declares_integration_contract() -> None:
    """REQ-ARC-WMTE-4705: OpenSpec declares the 4705 integration artifact schema."""

    from carnot import experiment_4705_integration_gate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4705" in spec
    assert "SCENARIO-ARC-WMTE-4705" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4705_audits_unchanged_upstream_configs() -> None:
    """REQ-ARC-WMTE-4705: unchanged A1/A2 configs keep submitted config unchanged."""

    from carnot import experiment_4705_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )

    assert audit["config_changed"] is False
    assert audit["levers_integrated"] == []
    assert audit["config_integrated"].startswith("unchanged:")
    assert audit["a1"]["reason"] == "chosen_submitted_config_unchanged"
    assert audit["a1"]["submitted_object_centric_proposal_enabled"] is False
    assert audit["a2"]["reason"] == "chosen_submitted_config_unchanged"
    assert audit["a2"]["submitted_amortized_first_contact_prior_enabled"] is False
    assert audit["a2"]["submitted_go_explore_archive_enabled"] is False


def test_req_arc_wmte_4705_audits_cleared_a1_a2_configs_and_mismatches() -> None:
    """REQ-ARC-WMTE-4705: cleared object/prior configs must match submitted config."""

    from carnot import experiment_4705_integration_gate as mod

    a1_chosen = {
        "object_centric_proposal_enabled": True,
        "object_centric_proposal_mode": "connected_component_slots_plus_relational_gaps",
        "object_centric_neighborhood_radius": 2,
    }
    a2_chosen = {
        "amortized_first_contact_prior_enabled": True,
        "amortized_first_contact_prior_mode": (
            "frequency_prior_from_cross_game_first_contact_traces"
        ),
        "go_explore_archive_enabled": True,
        "go_explore_archive_mode": "return_then_explore_replayable_prefix_archive",
    }
    winning_a1 = {
        **_a1_unchanged_artifact(),
        "honest_verdict": "success: object_centric_perception_generic_agent_new_level_r11l_L1",
        "chosen_submitted_config": a1_chosen,
        "generic_agent_reached_level": 1,
        "offline_reproduced": True,
        "reproduced_levels": 1,
    }
    winning_a2 = {
        **_a2_unchanged_artifact(),
        "honest_verdict": (
            "success: amortized_prior_go_explore_coverage_up_heldout_firstwin_lift_bp35"
        ),
        "chosen_submitted_config": a2_chosen,
        "coverage_delta": 0.2,
        "first_win_rate_delta": 0.08,
        "offline_reproduced": True,
        "no_prior_ablation_failed": True,
    }
    submitted = {
        **_submitted_config(),
        **a1_chosen,
        **a2_chosen,
    }

    audit = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=winning_a2,
        submitted_agent_config=submitted,
    )

    assert audit["config_changed"] is True
    assert audit["levers_integrated"] == [
        "A1_object_centric_proposal",
        "A2_amortized_prior_go_explore",
    ]
    assert audit["config_integrated"]["object_centric_neighborhood_radius"] == 2
    assert audit["config_integrated"]["go_explore_archive_enabled"] is True

    a1_without_radius = mod.audit_config_integration(
        a1_artifact={
            **winning_a1,
            "chosen_submitted_config": {
                "object_centric_proposal_enabled": True,
                "object_centric_proposal_mode": ("connected_component_slots_plus_relational_gaps"),
            },
        },
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config={
            "object_centric_proposal_enabled": True,
            "object_centric_proposal_mode": "connected_component_slots_plus_relational_gaps",
        },
    )
    assert a1_without_radius["a1"]["integrated"] is True

    a1_mismatch = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    assert a1_mismatch["a1"]["reason"] == "submitted_object_centric_config_mismatch"

    a1_invalid = mod.audit_config_integration(
        a1_artifact={
            **winning_a1,
            "chosen_submitted_config": {"object_centric_proposal_enabled": True},
        },
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert a1_invalid["a1"]["reason"] == "a1_chosen_object_centric_config_invalid"

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
    assert a1_unwon["a1"]["reason"] == "a1_object_centric_gate_not_cleared"

    a2_unwon = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "coverage_delta": 0.0},
        submitted_agent_config=submitted,
    )
    assert a2_unwon["a2"]["reason"] == "a2_amortized_prior_gate_not_cleared"

    a2_mismatch = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=winning_a2,
        submitted_agent_config={**submitted, "go_explore_archive_enabled": False},
    )
    assert a2_mismatch["a2"]["reason"] == "submitted_amortized_prior_config_mismatch"

    a2_bad_type = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "chosen_submitted_config": ["bad"]},
        submitted_agent_config=submitted,
    )
    assert a2_bad_type["a2"]["reason"] == "chosen_submitted_config_invalid"

    a2_oracle = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "verifier_is_oracle": True},
        submitted_agent_config=submitted,
    )
    assert a2_oracle["a2"]["reason"] == "verifier_oracle_not_false"

    a2_invalid = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact={**winning_a2, "chosen_submitted_config": {"go_explore_archive_enabled": True}},
        submitted_agent_config=submitted,
    )
    assert a2_invalid["a2"]["reason"] == "a2_chosen_amortized_prior_config_invalid"


def test_scenario_arc_wmte_4705_builds_unchanged_artifact_with_null_delta_markers() -> None:
    """SCENARIO-ARC-WMTE-4705: unchanged integration records null deltas honestly."""

    from carnot import experiment_4705_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    metrics = mod.compare_scored_measurement(
        scored_measurement=_scored_measurement(),
        pre_integration_measurement=None,
        config_changed=audit["config_changed"],
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        source_artifacts={"a1": mod.A1_RELATIVE_PATH, "a2": mod.A2_RELATIVE_PATH},
        source_artifact_checksums={"a1": "sha256:a1", "a2": "sha256:a2"},
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"
    assert artifact["config_changed"] is False
    assert artifact["config_integrated"].startswith("unchanged:")
    assert artifact["first_win_rate_integrated"] == 0.04
    assert artifact["multi_level_deepen_rate_integrated"] == 0.0
    assert artifact["first_win_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["multi_level_deepen_rate_delta_vs_pre_integration"] == 0.0
    assert "config_changed=false" in artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
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
            "field_principles": {},
            "submitted_to_leaderboard": True,
            "no_regression_vs_pre_integration": False,
            "null_delta_methodology_note": "",
            "positive_control_passed": False,
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
    assert "no_regression_vs_pre_integration" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "null_delta_methodology_note" in errors
    assert "positive_control_passed" in errors
    assert "reproducibility_checksum" in errors


def test_scenario_arc_wmte_4705_changed_config_verdicts() -> None:
    """SCENARIO-ARC-WMTE-4705: changed configs report success only without regression."""

    from carnot import experiment_4705_integration_gate as mod

    metrics = mod.compare_scored_measurement(
        scored_measurement={"first_win_rate": 0.08, "multi_level_deepen_rate": 0.02},
        pre_integration_measurement={"first_win_rate": 0.04, "multi_level_deepen_rate": 0.0},
        config_changed=True,
    )
    assert metrics["first_win_rate_delta_vs_pre_integration"] == 0.04
    assert metrics["metric_measurement_note"].startswith("config_changed=true")

    fallback_metrics = mod.compare_scored_measurement(
        scored_measurement={"first_win_rate": 0.08, "multi_level_deepen_rate": 0.02},
        pre_integration_measurement=None,
        config_changed=True,
    )
    assert fallback_metrics["first_win_rate_pre_integration"] == 0.08

    a1_audit = {
        "config_changed": True,
        "levers_integrated": ["A1_object_centric_proposal"],
        "config_integrated": {"levers_integrated": ["A1_object_centric_proposal"]},
    }
    a2_audit = {
        **a1_audit,
        "levers_integrated": ["A2_amortized_prior_go_explore"],
    }
    both_audit = {
        **a1_audit,
        "levers_integrated": [
            "A1_object_centric_proposal",
            "A2_amortized_prior_go_explore",
        ],
    }

    a1_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=a1_audit,
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert a1_artifact["honest_verdict"] == (
        "success: integrated_a1_object_centric_proposal_shipped_parity_green"
    )

    a2_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=a2_audit,
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert a2_artifact["honest_verdict"] == (
        "success: integrated_a2_amortized_prior_go_explore_shipped_parity_green"
    )

    both_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=both_audit,
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert both_artifact["honest_verdict"] == (
        "success: integrated_a1_object_centric_and_a2_amortized_prior_shipped_parity_green"
    )

    failed = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=a1_audit,
        metrics={**metrics, "no_regression_vs_pre_integration": False},
        parity_test={"passed": False},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert failed["honest_verdict"] == "complete: integration_parity_or_regression_failed"


def test_scenario_arc_wmte_4705_run_writes_artifact_and_blocks(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4705: run writes the stable result artifact."""

    from carnot import experiment_4705_integration_gate as mod

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.SPEC_RELATIVE_PATH).write_text("REQ-ARC-WMTE-4705\n", encoding="utf-8")
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_unchanged_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_unchanged_artifact())

    artifact = mod.run(
        tmp_path,
        import_checker=lambda: {"submitted_agent_import": True},
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        measure_scored_lane=lambda _root: _scored_measurement(),
        now=iter([10.0, 10.5]).__next__,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["config_changed"] is False
    assert artifact["honest_verdict"] == "complete: integration_unchanged_both_levers_null"
    assert artifact["positive_control_passed"] is True

    (tmp_path / mod.A2_RELATIVE_PATH).unlink()
    blocked = mod.run(
        tmp_path,
        import_checker=lambda: {"submitted_agent_import": True},
        parity_check=lambda _root: {"passed": True},
        submitted_agent_config=_submitted_config(),
        measure_scored_lane=lambda _root: _scored_measurement(),
        now=iter([20.0, 20.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_a2_artifact_present"
    assert blocked["preconditions_checked"]["blocked_resource"] == "a2_artifact_present"
