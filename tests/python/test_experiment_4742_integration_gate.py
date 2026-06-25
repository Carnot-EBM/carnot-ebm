"""Tests for Exp 4742 .436 submitted-agent integration gate.

Spec refs: REQ-ARC-WMTE-4742, SCENARIO-ARC-WMTE-4742-HONEST-NULL-INTEGRATION.
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
        "goal_energy_candidate_guidance_enabled": True,
        "goal_energy_candidate_guidance_alpha": 0.0,
        "goal_energy_candidate_guidance_beta": 1.0,
        "qd_generation_enabled": False,
        "qd_generation_mode": "energy_fitness_map_elites_sequence_generator",
        "verifier_is_oracle": False,
    }


def _previous_gate() -> dict[str, Any]:
    return {
        "experiment": "experiment_4731_integration_gate",
        "honest_verdict": "complete: integration_no_change_all_levers_unchanged",
        "live_first_win_rate_integrated": 0.04,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "no_regression_vs_pre_integration": True,
        "parity_test_green": True,
        "reproducibility_checksum": "sha256:prev",
    }


def _a1_unchanged_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4737_goal_energy_candidate_generation_valid_test",
        "honest_verdict": (
            "complete: goal_energy_generation_no_first_win_lift_residual_"
            "goal_energy_does_not_up_weight_the_winner"
        ),
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "goal_energy_first_win": 0.0,
        "baseline_first_win": 0.0,
        "goal_energy_vs_baseline_delta": 0.0,
        "parity_test_green": True,
        "positive_control_passed": True,
        "reproducibility_checksum": "sha256:a1",
    }


def _a2_unchanged_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4738_energy_fitness_qd_generation_valid_test",
        "honest_verdict": (
            "complete: energy_qd_generation_no_first_win_lift_residual_"
            "winner_not_in_reachable_mutation_neighborhood"
        ),
        "verifier_is_oracle": False,
        "chosen_submitted_config": "unchanged",
        "energy_qd_first_win": 0.0,
        "naive_search_first_win": 0.0,
        "energy_qd_vs_naive_delta": 0.0,
        "parity_test_green": True,
        "positive_control_passed": True,
        "reproducibility_checksum": "sha256:a2",
    }


def _scored_measurement(first_win_rate: float = 0.04) -> dict[str, Any]:
    return {
        "first_win_rate": first_win_rate,
        "scored_lane": {
            "integrated_measurement": {
                "first_win_rate": first_win_rate,
                "variant_attempts_count": 25,
            },
            "variant_ids": [1],
            "budget": 200,
        },
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_importable": True,
        "submitted_agent_import": True,
        "previous_gate_artifact_present": True,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "spec_has_req_4742": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4742_spec_declares_integration_contract() -> None:
    """REQ-ARC-WMTE-4742: OpenSpec declares the 4742 integration artifact schema."""

    from carnot import experiment_4742_integration_gate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4742" in spec
    assert "SCENARIO-ARC-WMTE-4742-HONEST-NULL-INTEGRATION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4742_audits_unchanged_and_strongest_changed_configs() -> None:
    """REQ-ARC-WMTE-4742: only a lifted non-unchanged .436 config can be selected."""

    from carnot import experiment_4742_integration_gate as mod

    unchanged = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )

    assert unchanged["config_changed"] is False
    assert unchanged["integrated_change"] == "none"
    assert unchanged["a1"]["reason"] == "chosen_submitted_config_unchanged"
    assert unchanged["a2"]["reason"] == "chosen_submitted_config_unchanged"

    a1_config = {
        "goal_energy_candidate_guidance_enabled": True,
        "goal_energy_candidate_guidance_alpha": 0.4,
        "goal_energy_candidate_guidance_beta": 0.6,
    }
    a2_config = {
        "qd_generation_enabled": True,
        "qd_generation_mode": "energy_fitness_map_elites_sequence_generator",
    }
    winning_a1 = {
        **_a1_unchanged_artifact(),
        "honest_verdict": "success: goal_energy_generation_first_win_lift_0.03",
        "chosen_submitted_config": a1_config,
        "goal_energy_vs_baseline_delta": 0.03,
    }
    winning_a2 = {
        **_a2_unchanged_artifact(),
        "honest_verdict": "success: energy_qd_generation_first_win_lift_0.07",
        "chosen_submitted_config": a2_config,
        "energy_qd_vs_naive_delta": 0.07,
    }
    submitted = {**_submitted_config(), **a1_config, **a2_config}

    selected = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=winning_a2,
        submitted_agent_config=submitted,
    )

    assert selected["config_changed"] is True
    assert selected["integrated_change"] == "A2_energy_fitness_qd_generation"
    assert selected["selected_submitted_config"] == a2_config
    assert selected["a2"]["integrated"] is True
    assert selected["a1"]["integrated"] is False
    assert selected["a1"]["reason"] == "not_strongest_lift"

    no_lift = mod.audit_config_integration(
        a1_artifact={**winning_a1, "goal_energy_vs_baseline_delta": 0.0},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config={**_submitted_config(), **a1_config},
    )
    assert no_lift["a1"]["reason"] == "held_out_lift_not_positive"

    mismatch = mod.audit_config_integration(
        a1_artifact=winning_a1,
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    assert mismatch["a1"]["reason"] == "submitted_config_mismatch"

    invalid = mod.audit_config_integration(
        a1_artifact={**winning_a1, "chosen_submitted_config": ["bad"]},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert invalid["a1"]["reason"] == "chosen_submitted_config_invalid"

    oracle = mod.audit_config_integration(
        a1_artifact={**winning_a1, "verifier_is_oracle": True},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert oracle["a1"]["reason"] == "verifier_oracle_not_false"

    upstream_null = mod.audit_config_integration(
        a1_artifact={**winning_a1, "honest_verdict": "complete: no lift"},
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=submitted,
    )
    assert upstream_null["a1"]["reason"] == "upstream_not_success"

    assert mod._as_float(True, 7.0) == 7.0
    assert mod._as_float("not-a-number", 3.0) == 3.0
    assert mod.artifact_checksum({"z": 1}).startswith("sha256:")


def test_scenario_arc_wmte_4742_builds_honest_null_artifact_with_markers() -> None:
    """SCENARIO-ARC-WMTE-4742-HONEST-NULL-INTEGRATION: flat no-op emits markers."""

    from carnot import experiment_4742_integration_gate as mod

    audit = mod.audit_config_integration(
        a1_artifact=_a1_unchanged_artifact(),
        a2_artifact=_a2_unchanged_artifact(),
        submitted_agent_config=_submitted_config(),
    )
    metrics = mod.measure_integrated_metrics(
        previous_gate_artifact=_previous_gate(),
        scored_measurement=None,
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

    assert artifact["honest_verdict"] == "complete: integration_no_change_all_levers_unchanged"
    assert artifact["integrated_change"] == "none"
    assert artifact["live_first_win_rate_integrated"] == 0.04
    assert artifact["live_first_win_rate_pre_integration"] == 0.04
    assert artifact["live_first_win_rate_delta_vs_pre_integration"] == 0.0
    assert artifact["no_regression_vs_pre_integration"] is True
    assert "all .436 A1/A2 selected submitted configs" in artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
    assert artifact["parity_test_green"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "tautology_guard" not in artifact
    assert mod.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad.pop("schema")
    bad.update(
        {
            "honest_verdict": "not-terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "parity_test_green": False,
            "field_principles": {},
            "submitted_to_leaderboard": True,
            "no_regression_vs_pre_integration": False,
            "null_delta_methodology_note": "",
            "positive_control_passed": True,
            "reproducibility_checksum": "sha256:bad",
            "tautology_guard": "ignored prose",
        }
    )
    errors = mod.artifact_schema_errors(bad)
    assert "missing required field schema" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "verifier_is_oracle_false" in errors
    assert "parity_test_green" in errors
    assert "no_regression_vs_pre_integration" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "null_delta_methodology_note" in errors
    assert "positive_control_passed" in errors
    assert "tautology_guard_forbidden" in errors
    assert "reproducibility_checksum" in errors


def test_scenario_arc_wmte_4742_changed_config_verdicts() -> None:
    """REQ-ARC-WMTE-4742: changed configs report success only after a non-regressing gate."""

    from carnot import experiment_4742_integration_gate as mod

    metrics = mod.measure_integrated_metrics(
        previous_gate_artifact=_previous_gate(),
        scored_measurement=_scored_measurement(first_win_rate=0.09),
        config_changed=True,
    )
    assert metrics["live_first_win_rate_delta_vs_pre_integration"] == 0.05

    audit = {
        "config_changed": True,
        "integrated_change": "A2_energy_fitness_qd_generation",
        "selected_submitted_config": {"qd_generation_enabled": True},
        "a1": {},
        "a2": {},
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
        "success: integrated_a2_energy_fitness_qd_generation_first_win_0.05"
    )
    assert artifact["null_delta_methodology_note"] == ""

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
    assert failed["positive_control_passed"] is False

    flat_changed = mod.build_artifact(
        preconditions_checked=_preconditions(),
        audit=audit,
        metrics={
            **metrics,
            "live_first_win_rate_integrated": 0.04,
            "live_first_win_rate_delta_vs_pre_integration": 0.0,
        },
        parity_test={"passed": True},
        submitted_agent_config=_submitted_config(),
        source_artifacts={},
        source_artifact_checksums={},
        duration_s=1.0,
    )
    assert "selected .436 integration measured equal" in flat_changed[
        "null_delta_methodology_note"
    ]


def test_scenario_arc_wmte_4742_run_writes_artifact_and_blocks(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4742-HONEST-NULL-INTEGRATION: run writes stable no-op result."""

    from carnot import experiment_4742_integration_gate as mod

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.SPEC_RELATIVE_PATH).write_text("REQ-ARC-WMTE-4742\n", encoding="utf-8")
    _write_json(tmp_path / mod.PREVIOUS_GATE_RELATIVE_PATH, _previous_gate())
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_unchanged_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_unchanged_artifact())

    artifact = mod.run(
        tmp_path,
        precondition_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        submitted_agent_config=_submitted_config(),
        measure_scored_lane=lambda _root: (_ for _ in ()).throw(AssertionError("no-op")),
        now=iter([10.0, 10.5]).__next__,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["integrated_change"] == "none"
    assert artifact["positive_control_passed"] is True

    blocked = mod.run(
        tmp_path,
        precondition_checker=lambda _root: {
            **_preconditions(),
            "ok": False,
            "a2_artifact_present": False,
            "blocked_resource": "a2_artifact_present",
        },
        parity_check=lambda _root: {"passed": True},
        submitted_agent_config=_submitted_config(),
        measure_scored_lane=lambda _root: _scored_measurement(),
        now=iter([20.0, 20.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_a2_artifact_present"
    assert blocked["preconditions_checked"]["blocked_resource"] == "a2_artifact_present"
