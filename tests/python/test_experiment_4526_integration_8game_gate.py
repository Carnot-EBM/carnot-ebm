"""Tests for Exp 4526 submitted deeper-level integration.

Spec refs: REQ-ARC-WMTE-4526, SCENARIO-ARC-WMTE-4526.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4526_integration_8game_gate as exp4526


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions(ok: bool = True) -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": ok,
        "baseline_file_present": True,
        "a1_artifact_present": True,
        "a2_reach_artifact_present": True,
        "spec_has_req_4526": True,
        "ok": ok,
    }


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4526.GATE_GAMES),
        "solved_games": list(exp4526.CORE_GAMES),
        "efficiency_by_game": {
            "lp85": 2.0069,
            "m0r0": 0.0003,
            "sp80": 0.0001,
            "vc33": 0.0001,
        },
        "core_efficiency": exp4526.CORE_EFFICIENCY_BASELINE,
        "median_actions_on_solved": exp4526.BASELINE_MEDIAN_ACTIONS,
    }


def _a1_artifact() -> dict[str, object]:
    return {
        "honest_verdict": "complete: forward_walk_no_reduction_honest_null",
        "flagged_adversarial": False,
        "chosen_submitted_config": "unchanged",
        "nav_diagnostics_before_after": {
            "before": {"reset_replay_steps": 4576, "forward_walk_hit_rate": 0.016},
            "after": {"reset_replay_steps": 4533, "forward_walk_hit_rate": 0.021},
        },
        "median_actions_on_core_control": 7761.5,
        "median_actions_on_core_best": 7761.5,
    }


def _a2_artifact(*, improved: bool = False, lost_core: bool = False) -> dict[str, object]:
    control_levels = {"lp85": 1, "m0r0": 1, "sp80": 1}
    candidate_levels = {
        "lp85": 2 if improved else 1,
        "m0r0": 0 if lost_core else 1,
        "sp80": 1,
    }
    candidate_efficiency = 3.25 if improved else exp4526.CORE_EFFICIENCY_BASELINE
    return {
        "honest_verdict": "success: lp85_reached_L2_core_efficiency_3.2500_above_2.0074"
        if improved
        else "complete: l1_l2_barrier_diagnosed_depth_cap_honest_null",
        "flagged_adversarial": False,
        "offline_reproduced": improved,
        "core_efficiency_baseline": exp4526.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": candidate_efficiency,
        "deepest_level_reached_per_core_game": {
            "control_max_depth_45": control_levels,
            "energy_verifier_frontier_routing": candidate_levels,
        },
        "levers_tried": [
            {
                "lever": "control_max_depth_45",
                "core_efficiency": exp4526.CORE_EFFICIENCY_BASELINE,
                "delta_vs_baseline": 0.0,
                "deepest_level_by_game": control_levels,
            },
            {
                "lever": "energy_verifier_frontier_routing",
                "core_efficiency": candidate_efficiency,
                "delta_vs_baseline": round(
                    candidate_efficiency - exp4526.CORE_EFFICIENCY_BASELINE,
                    4,
                ),
                "deepest_level_by_game": candidate_levels,
            },
        ],
    }


def _custom_a2(
    *,
    offline_reproduced: bool = True,
    row: object | None = None,
    control_from_deepest: bool = True,
) -> dict[str, object]:
    control_levels = {"lp85": 1, "m0r0": 1, "sp80": 1}
    levers = [
        {
            "lever": "control_fallback",
            "core_efficiency": exp4526.CORE_EFFICIENCY_BASELINE,
            "deepest_level_by_game": control_levels,
        },
    ]
    if row is not None:
        levers.append(row)
    artifact: dict[str, object] = {
        "offline_reproduced": offline_reproduced,
        "core_efficiency_best": 3.25,
        "levers_tried": levers,
    }
    if control_from_deepest:
        artifact["deepest_level_reached_per_core_game"] = {"control_max_depth_45": control_levels}
    return artifact


def _stop_artifact() -> dict[str, object]:
    return {
        "honest_verdict": "success: stop_after_levelup_core_actions_2825_below_control",
        "flagged_adversarial": False,
        "median_actions_on_core_best": 2825.5,
        "levels_per_game_preserved": {"lost_level_depth_games": [], "passed": True},
    }


def _gate_result(*, core_efficiency: float = 2.0074, solved: list[str] | None = None) -> dict[str, object]:
    solved_games = solved or list(exp4526.CORE_GAMES)
    rows = [
        {
            "game": game,
            "solved": game in solved_games,
            "actions": 10 if game in solved_games else 20,
            "efficiency": (core_efficiency / len(exp4526.CORE_GAMES)) if game in exp4526.CORE_GAMES else 0.0,
            "timed_out": False,
        }
        for game in exp4526.GATE_GAMES
    ]
    return {
        "pass": core_efficiency >= exp4526.CORE_EFFICIENCY_BASELINE,
        "verdict": "PASS fixture",
        "baseline_guard": {"ok": True},
        "current": {
            "policy": "e3",
            "games": list(exp4526.GATE_GAMES),
            "per_game": rows,
            "solved_games": solved_games,
            "solved_count": len(solved_games),
            "efficiency_by_game": {
                game: row["efficiency"] for game, row in zip(exp4526.GATE_GAMES, rows)
            },
            "core_efficiency": core_efficiency,
            "median_actions_on_core": 10.0,
            "median_actions_on_solved": 10.0,
            "timed_out_count": 0,
        },
    }


def test_req_arc_wmte_4526_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4526: OpenSpec anchors the integration artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4526" in spec
    assert "SCENARIO-ARC-WMTE-4526" in spec
    assert exp4526.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4526.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4526_selects_only_deeper_core_efficiency_winners() -> None:
    """REQ-ARC-WMTE-4526: retired action trimming is rejected, and null A2 stays unwired."""

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_a2_artifact(),
        stop_after_levelup_artifact=_stop_artifact(),
    )

    assert decision["accepted_levers"] == []
    assert decision["rejected_levers"]["A2_reach_deeper_levels:energy_verifier_frontier_routing"]["reason"] == (
        "no_core_efficiency_gain"
    )
    assert decision["rejected_levers"]["A2_stop_after_levelup"]["reason"] == "action_trimming_retired"

    accepted = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_a2_artifact(improved=True),
        stop_after_levelup_artifact=_stop_artifact(),
    )
    assert accepted["accepted_levers"] == ["A2_reach_deeper_levels:energy_verifier_frontier_routing"]

    lost = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_a2_artifact(improved=True, lost_core=True),
        stop_after_levelup_artifact=_stop_artifact(),
    )
    assert lost["accepted_levers"] == []
    assert lost["rejected_levers"]["A2_reach_deeper_levels:energy_verifier_frontier_routing"]["reason"] == (
        "core_level_regression"
    )


def test_scenario_arc_wmte_4526_honest_null_artifact_is_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4526: no deeper-level winner writes an honest null."""

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_a2_artifact(),
        stop_after_levelup_artifact=_stop_artifact(),
    )
    artifact = exp4526.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(),
        random_seed=4526,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: no_lever_raises_core_efficiency_honest_null"
    assert artifact["core_efficiency_integrated"] == exp4526.CORE_EFFICIENCY_BASELINE
    assert artifact["core_solves_preserved"] is True
    assert artifact["levers_integrated"] == []
    assert artifact["heldout_solve_rate"] == 0.0
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["nav_diagnostics"]["reset_replay_steps_integrated"] == 4576
    assert exp4526.artifact_schema_errors(artifact) == []

    out = exp4526.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4526_success_requires_integrated_gain_and_core_preservation() -> None:
    """REQ-ARC-WMTE-4526: submit readiness needs integrated CORE efficiency above baseline."""

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_a2_artifact(improved=True),
        stop_after_levelup_artifact=_stop_artifact(),
    )
    artifact = exp4526.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(core_efficiency=3.25),
        random_seed=4526,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "success: integrated_core_efficiency_3.2500_above_2.0074"
    assert artifact["ready_for_operator_submit"] is True
    assert exp4526.artifact_schema_errors(artifact) == []

    dropped = exp4526.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(core_efficiency=3.25, solved=["lp85", "sp80", "vc33"]),
        random_seed=4526,
        duration_s=1.0,
    )
    assert dropped["honest_verdict"] == "complete: no_lever_raises_core_efficiency_honest_null"
    assert dropped["core_solves_preserved"] is False
    assert dropped["ready_for_operator_submit"] is False


def test_scenario_arc_wmte_4526_run_writes_injected_gate_measurement(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4526: run writes the 8-game per-level gate artifact."""

    artifact = exp4526.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        load_upstream_artifacts=lambda _root: {
            "a1_forward_walk": _a1_artifact(),
            "a2_reach_deeper_levels": _a2_artifact(),
            "a2_stop_after_levelup": _stop_artifact(),
        },
        gate_runner=lambda **_kwargs: _gate_result(),
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["result_path"] == exp4526.RESULT_RELATIVE_PATH
    assert json.loads((tmp_path / exp4526.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4526_schema_and_blocked_paths() -> None:
    """REQ-ARC-WMTE-4526: schema failures and missing resources are explicit."""

    artifact = exp4526.run(
        write=False,
        preconditions_checked=_preconditions(ok=False),
        baseline=_baseline(),
        gate_runner=lambda **_kwargs: _gate_result(core_efficiency=99.0),
        now=lambda: 10.0,
    )
    assert artifact["honest_verdict"] == "blocked_offline_arcade_import"
    assert artifact["levers_integrated"] == []
    assert artifact["ready_for_operator_submit"] is False

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_a2_artifact(),
        stop_after_levelup_artifact=_stop_artifact(),
    )
    valid = exp4526.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(),
        random_seed=4526,
        duration_s=1.0,
    )
    bad = {
        **valid,
        "honest_verdict": "done",
        "field_principles": {},
        "core_efficiency_baseline": 0.0,
        "core_efficiency_integrated": None,
        "core_solves_preserved": "yes",
        "levers_integrated": "bad",
        "additivity_checked": None,
        "heldout_solve_rate": None,
        "nav_diagnostics": {},
        "ready_for_operator_submit": True,
        "false_negative_risk_checked": False,
        "reproducibility_checksum": "bad",
        "preconditions_checked": "bad",
    }
    del bad["random_seed"]

    errors = exp4526.artifact_schema_errors(bad)

    assert "missing required field random_seed" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "field_principles must match REQ-ARC-WMTE-4526" in errors
    assert "core_efficiency_baseline must equal 2.0074" in errors
    assert "core_efficiency_integrated must be numeric" in errors
    assert "core_solves_preserved must be bool" in errors
    assert "levers_integrated must be a list" in errors
    assert "additivity_checked must be a mapping" in errors
    assert "heldout_solve_rate must be numeric" in errors
    assert "nav_diagnostics must include reset_replay_steps_integrated" in errors
    assert "ready_for_operator_submit cannot be true without success" in errors
    assert "false_negative_risk_checked must be true for complete/success artifacts" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "preconditions_checked must be a mapping" in errors


def test_req_arc_wmte_4526_loaders_and_selector_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4526: loader fallbacks and rejection reasons are deterministic."""

    assert exp4526._read_json(tmp_path / "missing.json") == {}
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )
    for name, relative in exp4526.UPSTREAM_ARTIFACTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")

    assert exp4526.load_gate_baseline(tmp_path)["core_efficiency"] == exp4526.CORE_EFFICIENCY_BASELINE
    assert exp4526.load_gate_baseline(tmp_path / "missing")["median_actions_on_solved"] == (
        exp4526.BASELINE_MEDIAN_ACTIONS
    )
    assert sorted(exp4526.load_upstream_artifacts(tmp_path)) == sorted(exp4526.UPSTREAM_ARTIFACTS)
    assert exp4526._levels_from_mapping(None) == {}
    assert exp4526._a2_control_levels({}) == {game: 0 for game in exp4526.CORE_GAMES}

    flagged_a1 = {**_a1_artifact(), "flagged_adversarial": True}
    flagged_candidate = {
        "lever": "flagged",
        "flagged_adversarial": True,
        "core_efficiency": 3.25,
        "deepest_level_by_game": {"lp85": 2, "m0r0": 1, "sp80": 1},
    }
    no_deeper = {
        "lever": "no_deeper",
        "core_efficiency": 3.25,
        "deepest_level_by_game": {"lp85": 1, "m0r0": 1, "sp80": 1},
    }
    no_repro = {
        "lever": "no_repro",
        "core_efficiency": 3.25,
        "deepest_level_by_game": {"lp85": 2, "m0r0": 1, "sp80": 1},
    }

    decision = exp4526.select_integrated_levers(
        a1_artifact=flagged_a1,
        a2_reach_artifact=_custom_a2(row=flagged_candidate),
    )
    assert decision["rejected_levers"]["A1_forward_walk_navigation"]["reason"] == "flagged_adversarial"
    assert decision["rejected_levers"]["A2_reach_deeper_levels:flagged"]["reason"] == "flagged_adversarial"

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_custom_a2(row="not-a-row", control_from_deepest=False),
    )
    assert decision["a2_control_levels"] == {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 0}

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_custom_a2(row=no_deeper),
    )
    assert decision["rejected_levers"]["A2_reach_deeper_levels:no_deeper"]["reason"] == (
        "no_deeper_core_level"
    )

    decision = exp4526.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_reach_artifact=_custom_a2(row=no_repro, offline_reproduced=False),
    )
    assert decision["rejected_levers"]["A2_reach_deeper_levels:no_repro"]["reason"] == (
        "offline_reproduction_missing"
    )


def test_req_arc_wmte_4526_measurement_fallbacks_and_schema_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4526: measurement fallbacks and defensive schema checks are covered."""

    measurement = {
        "per_game": [
            {"game": "lp85", "solved": True, "efficiency": 1.0, "levels": 2},
            {"game": "m0r0", "solved": True, "efficiency": 0.5, "best_level": 1},
            {"game": "sp80", "solved": False, "efficiency": 0.25},
            "not-a-row",
        ]
    }
    gate_result = {"current": measurement, "baseline_guard": {"ok": True}}
    decision = {
        "accepted_levers": [],
        "rejected_levers": {},
        "upstream_summaries": {},
        "a2_best_delta_core_efficiency": 0.0,
        "nav_diagnostics": {"reset_replay_steps_integrated": 7},
    }
    baseline = {"core_efficiency": exp4526.CORE_EFFICIENCY_BASELINE}
    artifact = exp4526.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=baseline,
        upstream_decision=decision,
        gate_result=gate_result,
        random_seed=4526,
        duration_s=1.0,
    )

    assert artifact["core_efficiency_integrated"] == 1.75
    assert artifact["core_solves_preserved"] is False
    assert artifact["per_game_deepest_level_reached"]["lp85"] == 2
    assert artifact["per_game_deepest_level_reached"]["sp80"] == 0
    assert exp4526._current_measurement(measurement) == measurement
    assert exp4526._efficiency_by_game({"efficiency_by_game": {"lp85": 1}}) == {"lp85": 1.0}

    valid = exp4526.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4526.select_integrated_levers(
            a1_artifact=_a1_artifact(),
            a2_reach_artifact=_a2_artifact(improved=True),
            stop_after_levelup_artifact=_stop_artifact(),
        ),
        gate_result=_gate_result(core_efficiency=3.25),
        random_seed=4526,
        duration_s=1.0,
    )

    mutations = [
        (lambda item: item.__setitem__("inference_substrate", "bad"), "inference_substrate"),
        (
            lambda item: item.__setitem__(
                "honest_verdict", "success: integrated_core_efficiency_1.0000_above_2.0074"
            )
            or item.__setitem__("core_efficiency_integrated", 1.0),
            "success requires core_efficiency_integrated above baseline",
        ),
        (
            lambda item: item.__setitem__("honest_verdict", valid["honest_verdict"])
            or item.__setitem__("core_solves_preserved", False),
            "success requires core_solves_preserved=true",
        ),
        (
            lambda item: item.__setitem__("honest_verdict", valid["honest_verdict"])
            or item.__setitem__("levers_integrated", []),
            "success requires an integrated lever",
        ),
        (
            lambda item: item.__setitem__("operator_submission_performed", True),
            "operator_submission_performed must be false",
        ),
        (
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
            "reproducibility_checksum must match artifact content",
        ),
    ]
    for mutate, expected in mutations:
        changed = dict(valid)
        mutate(changed)
        assert any(expected in error for error in exp4526.artifact_schema_errors(changed))

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4526.write_artifact({**valid, "honest_verdict": "bad"}, root=tmp_path)


def test_req_arc_wmte_4526_run_default_loader_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4526: run can use default precondition and baseline loaders."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )

    monkeypatch.setattr(exp4526, "check_preconditions", lambda _root: _preconditions())
    artifact = exp4526.run(
        root=tmp_path,
        write=False,
        baseline=None,
        load_upstream_artifacts=lambda _root: {
            "a1_forward_walk": _a1_artifact(),
            "a2_reach_deeper_levels": _a2_artifact(),
            "a2_stop_after_levelup": _stop_artifact(),
        },
        gate_runner=lambda **_kwargs: _gate_result(),
        now=lambda: 10.0,
    )
    assert artifact["honest_verdict"].startswith("complete:")

    with pytest.raises(ValueError, match="false_negative_risk_checked"):
        exp4526.run(
            root=tmp_path,
            write=False,
            preconditions_checked=_preconditions(),
            baseline={"core_efficiency": 0.0},
            load_upstream_artifacts=lambda _root: {
                "a1_forward_walk": _a1_artifact(),
                "a2_reach_deeper_levels": _a2_artifact(),
            },
            gate_runner=lambda **_kwargs: _gate_result(),
            now=lambda: 10.0,
        )
