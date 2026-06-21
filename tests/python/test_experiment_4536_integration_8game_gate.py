"""Tests for Exp 4536 submitted A1/A2 integration.

Spec refs: REQ-ARC-WMTE-4536, SCENARIO-ARC-WMTE-4536.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4536_integration_8game_gate as exp4536


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions(ok: bool = True) -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": ok,
        "baseline_file_present": True,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "spec_has_req_4536": True,
        "ok": ok,
    }


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4536.GATE_GAMES),
        "solved_games": list(exp4536.CORE_GAMES),
        "efficiency_by_game": {
            "lp85": 2.0069,
            "m0r0": 0.0003,
            "sp80": 0.0001,
            "vc33": 0.0001,
        },
        "core_efficiency": exp4536.CORE_EFFICIENCY_BASELINE,
        "median_actions_on_solved": 7760.0,
    }


def _levels(lp85: int = 1, m0r0: int = 1, sp80: int = 1, vc33: int = 1) -> dict[str, int]:
    return {"lp85": lp85, "m0r0": m0r0, "sp80": sp80, "vc33": vc33}


def _a1_artifact(
    *,
    flagged: bool = True,
    permitted_null: bool = True,
    improved: bool = False,
    lost_core: bool = False,
    offline_reproduced: bool = False,
) -> dict[str, object]:
    control = _levels()
    candidate = _levels(lp85=2 if improved else 1, m0r0=0 if lost_core else 1)
    best = 3.25 if improved else exp4536.CORE_EFFICIENCY_BASELINE
    artifact: dict[str, object] = {
        "honest_verdict": "complete: reinduction_no_deeper_level_barrier_refined_honest_null",
        "flagged_adversarial": flagged,
        "core_efficiency_baseline": exp4536.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "efficiency_delta": round(best - exp4536.CORE_EFFICIENCY_BASELINE, 4),
        "core_solves_preserved": not lost_core,
        "offline_reproduced": offline_reproduced,
        "target_levels_sweep": [
            {
                "target_levels": 1,
                "core_efficiency": exp4536.CORE_EFFICIENCY_BASELINE,
                "core_solves_preserved": True,
                "deepest_level_by_game": control,
            },
            {
                "target_levels": 2,
                "core_efficiency": best,
                "core_solves_preserved": not lost_core,
                "deepest_level_by_game": candidate,
            },
        ],
    }
    if permitted_null:
        artifact["efficiency_delta"] = 0.0
        artifact["core_efficiency_best"] = exp4536.CORE_EFFICIENCY_BASELINE
        artifact["null_delta_methodology_note"] = (
            "baseline==best because no lever reached a deeper offline-reproduced CORE level."
        )
        artifact["target_levels_sweep"][1]["core_efficiency"] = exp4536.CORE_EFFICIENCY_BASELINE  # type: ignore[index]
        artifact["target_levels_sweep"][1]["deepest_level_by_game"] = control  # type: ignore[index]
    return artifact


def _a2_artifact(
    *,
    flagged: bool = True,
    improved: bool = False,
    lost_core: bool = False,
) -> dict[str, object]:
    control = _levels()
    candidate = _levels(sp80=2 if improved else 1, m0r0=0 if lost_core else 1)
    efficiency = 3.5 if improved else 0.0062
    return {
        "honest_verdict": "complete: energy_routing_no_deeper_level_signal_characterized_honest_null",
        "flagged_adversarial": flagged,
        "core_efficiency_baseline": exp4536.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_energy_routed": efficiency,
        "core_solves_preserved": not lost_core,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "no_energy_control": {
            "core_efficiency": exp4536.CORE_EFFICIENCY_BASELINE,
            "core_solves_preserved": True,
            "deepest_level_by_game": control,
        },
        "energy_routed_measurement": {
            "core_efficiency": efficiency,
            "deepest_level_by_game": candidate,
        },
    }


def _gate_result(*, core_efficiency: float = 2.0074, solved: list[str] | None = None) -> dict[str, object]:
    solved_games = solved or list(exp4536.CORE_GAMES)
    rows = [
        {
            "game": game,
            "solved": game in solved_games,
            "actions": 10 if game in solved_games else 20,
            "efficiency": (core_efficiency / len(exp4536.CORE_GAMES)) if game in exp4536.CORE_GAMES else 0.0,
            "timed_out": False,
            "levels": 1 if game in solved_games else 0,
        }
        for game in exp4536.GATE_GAMES
    ]
    return {
        "pass": core_efficiency >= exp4536.CORE_EFFICIENCY_BASELINE,
        "verdict": "PASS fixture",
        "baseline_guard": {"ok": True},
        "current": {
            "policy": "e3",
            "games": list(exp4536.GATE_GAMES),
            "per_game": rows,
            "solved_games": solved_games,
            "solved_count": len(solved_games),
            "efficiency_by_game": {
                game: row["efficiency"] for game, row in zip(exp4536.GATE_GAMES, rows)
            },
            "core_efficiency": core_efficiency,
            "median_actions_on_core": 10.0,
            "median_actions_on_solved": 10.0,
            "timed_out_count": 0,
        },
    }


def test_req_arc_wmte_4536_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4536: OpenSpec anchors the integration artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4536" in spec
    assert "SCENARIO-ARC-WMTE-4536" in spec
    assert exp4536.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4536.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4536_permits_only_null_delta_adversarial_exception() -> None:
    """REQ-ARC-WMTE-4536: flagged null-delta tautologies are valid nulls, not winners."""

    decision = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=True, permitted_null=True),
        a2_artifact=_a2_artifact(flagged=True),
    )

    assert decision["accepted_levers"] == []
    assert decision["upstream_summaries"]["A1_per_level_goal_reinduction"]["flag_status"] == (
        "permitted_flagged_null"
    )
    assert decision["rejected_levers"]["A1_per_level_goal_reinduction:target_levels_2"]["reason"] == (
        "no_core_efficiency_gain"
    )
    assert decision["rejected_levers"]["A2_energy_trust_next_level_routing:energy_routed"]["reason"] == (
        "flagged_adversarial"
    )
    assert decision["isolated_deltas"]["A1_per_level_goal_reinduction"] == 0.0
    assert decision["isolated_deltas"]["A2_energy_trust_next_level_routing"] == -2.0012

    rejected = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=True, permitted_null=False),
        a2_artifact=_a2_artifact(flagged=False),
    )
    assert rejected["rejected_levers"]["A1_per_level_goal_reinduction"]["reason"] == (
        "flagged_adversarial"
    )


def test_req_arc_wmte_4536_selects_only_deeper_core_efficiency_winners() -> None:
    """REQ-ARC-WMTE-4536: accepted levers must raise CORE efficiency and preserve CORE depth."""

    accepted = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(
            flagged=False,
            permitted_null=False,
            improved=True,
            offline_reproduced=True,
        ),
        a2_artifact=_a2_artifact(flagged=False, improved=True),
    )

    assert accepted["accepted_levers"] == [
        "A1_per_level_goal_reinduction:target_levels_2",
        "A2_energy_trust_next_level_routing:energy_routed",
    ]

    lost = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(
            flagged=False,
            permitted_null=False,
            improved=True,
            lost_core=True,
            offline_reproduced=True,
        ),
        a2_artifact=_a2_artifact(flagged=False, improved=True, lost_core=True),
    )
    assert lost["accepted_levers"] == []
    assert lost["rejected_levers"]["A1_per_level_goal_reinduction:target_levels_2"]["reason"] == (
        "core_level_regression"
    )
    assert lost["rejected_levers"]["A2_energy_trust_next_level_routing:energy_routed"]["reason"] == (
        "core_level_regression"
    )

    unreproduced = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, permitted_null=False, improved=True),
        a2_artifact=_a2_artifact(flagged=False),
    )
    assert unreproduced["rejected_levers"]["A1_per_level_goal_reinduction:target_levels_2"]["reason"] == (
        "offline_reproduction_missing"
    )

    no_deeper_a1 = _a1_artifact(flagged=False, permitted_null=False, improved=False)
    no_deeper_a1["core_efficiency_best"] = 3.25
    no_deeper_a1["efficiency_delta"] = round(3.25 - exp4536.CORE_EFFICIENCY_BASELINE, 4)
    no_deeper_a1["target_levels_sweep"][1]["core_efficiency"] = 3.25  # type: ignore[index]
    no_deeper = exp4536.select_integrated_levers(
        a1_artifact=no_deeper_a1,
        a2_artifact=_a2_artifact(flagged=False),
    )
    assert no_deeper["rejected_levers"]["A1_per_level_goal_reinduction:target_levels_2"]["reason"] == (
        "no_deeper_core_level"
    )

    core_not_preserved_a1 = _a1_artifact(flagged=False, permitted_null=False, improved=True)
    core_not_preserved_a1["target_levels_sweep"][1]["core_solves_preserved"] = False  # type: ignore[index]
    core_not_preserved = exp4536.select_integrated_levers(
        a1_artifact=core_not_preserved_a1,
        a2_artifact=_a2_artifact(flagged=False),
    )
    assert core_not_preserved["rejected_levers"]["A1_per_level_goal_reinduction:target_levels_2"]["reason"] == (
        "core_solves_not_preserved"
    )

    a2_core_not_preserved = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, permitted_null=True),
        a2_artifact={**_a2_artifact(flagged=False, improved=True), "core_solves_preserved": False},
    )
    assert a2_core_not_preserved["rejected_levers"]["A2_energy_trust_next_level_routing:energy_routed"]["reason"] == (
        "core_solves_not_preserved"
    )

    a2_no_deeper = _a2_artifact(flagged=False, improved=False)
    a2_no_deeper["energy_routed_measurement"]["core_efficiency"] = 3.5  # type: ignore[index]
    a2_no_deeper_result = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, permitted_null=True),
        a2_artifact=a2_no_deeper,
    )
    assert a2_no_deeper_result["rejected_levers"]["A2_energy_trust_next_level_routing:energy_routed"]["reason"] == (
        "no_deeper_core_level"
    )


def test_scenario_arc_wmte_4536_honest_null_artifact_is_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4536: no winner writes an honest null with measured per-level efficiency."""

    decision = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
    )
    artifact = exp4536.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(),
        random_seed=4536,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: no_lever_raises_core_efficiency_honest_null"
    assert artifact["core_efficiency_integrated"] == exp4536.CORE_EFFICIENCY_BASELINE
    assert artifact["core_solves_preserved"] is True
    assert artifact["levers_integrated"] == []
    assert artifact["heldout_solve_rate"] == 0.0
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["additivity_checked"]["naive_sum_delta"] == -2.0012
    assert exp4536.artifact_schema_errors(artifact) == []

    out = exp4536.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4536_success_requires_integrated_gain_and_core_preservation() -> None:
    """REQ-ARC-WMTE-4536: submit readiness needs integrated CORE efficiency above baseline."""

    decision = exp4536.select_integrated_levers(
        a1_artifact=_a1_artifact(
            flagged=False,
            permitted_null=False,
            improved=True,
            offline_reproduced=True,
        ),
        a2_artifact=_a2_artifact(flagged=False),
    )
    artifact = exp4536.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(core_efficiency=3.25),
        random_seed=4536,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "success: integrated_core_efficiency_3.2500_above_2.0074"
    assert artifact["ready_for_operator_submit"] is True
    assert exp4536.artifact_schema_errors(artifact) == []

    dropped = exp4536.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(core_efficiency=3.25, solved=["lp85", "sp80", "vc33"]),
        random_seed=4536,
        duration_s=1.0,
    )
    assert dropped["honest_verdict"] == "complete: no_lever_raises_core_efficiency_honest_null"
    assert dropped["core_solves_preserved"] is False
    assert dropped["ready_for_operator_submit"] is False


def test_scenario_arc_wmte_4536_run_writes_injected_gate_measurement(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4536: run writes the 8-game per-level gate artifact."""

    artifact = exp4536.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        load_upstream_artifacts=lambda _root: {
            "a1_per_level_goal_reinduction": _a1_artifact(),
            "a2_energy_trust_next_level_routing": _a2_artifact(),
        },
        gate_runner=lambda **_kwargs: _gate_result(),
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["result_path"] == exp4536.RESULT_RELATIVE_PATH
    assert json.loads((tmp_path / exp4536.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4536_schema_and_blocked_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4536: schema failures and missing resources are explicit."""

    blocked = exp4536.run(
        write=False,
        preconditions_checked=_preconditions(ok=False),
        baseline=_baseline(),
        gate_runner=lambda **_kwargs: _gate_result(core_efficiency=99.0),
        now=lambda: 10.0,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade_import"
    assert blocked["levers_integrated"] == []
    assert blocked["ready_for_operator_submit"] is False

    valid = exp4536.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4536.select_integrated_levers(_a1_artifact(), _a2_artifact()),
        gate_result=_gate_result(),
        random_seed=4536,
        duration_s=1.0,
    )
    bad = {
        **valid,
        "honest_verdict": "done",
        "inference_substrate": "bad",
        "field_principles": {},
        "core_efficiency_baseline": 0.0,
        "core_efficiency_integrated": None,
        "core_solves_preserved": "yes",
        "levers_integrated": "bad",
        "additivity_checked": None,
        "heldout_solve_rate": None,
        "ready_for_operator_submit": True,
        "false_negative_risk_checked": False,
        "operator_submission_performed": True,
        "reproducibility_checksum": "bad",
        "preconditions_checked": "bad",
    }
    del bad["random_seed"]

    errors = exp4536.artifact_schema_errors(bad)

    assert "missing required field random_seed" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match" in errors
    assert "field_principles must match REQ-ARC-WMTE-4536" in errors
    assert "core_efficiency_baseline must equal 2.0074" in errors
    assert "core_efficiency_integrated must be numeric" in errors
    assert "core_solves_preserved must be bool" in errors
    assert "levers_integrated must be a list" in errors
    assert "additivity_checked must be a mapping" in errors
    assert "heldout_solve_rate must be numeric" in errors
    assert "ready_for_operator_submit cannot be true without success" in errors
    assert "false_negative_risk_checked must be true for complete/success artifacts" in errors
    assert "operator_submission_performed must be false" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "preconditions_checked must be a mapping" in errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4536.write_artifact({**valid, "honest_verdict": "bad"}, root=tmp_path)

    success = exp4536.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4536.select_integrated_levers(
            _a1_artifact(
                flagged=False,
                permitted_null=False,
                improved=True,
                offline_reproduced=True,
            ),
            _a2_artifact(flagged=False),
        ),
        gate_result=_gate_result(core_efficiency=3.25),
        random_seed=4536,
        duration_s=1.0,
    )
    success_mutations = [
        (
            lambda item: item.__setitem__("core_efficiency_integrated", 1.0),
            "success requires core_efficiency_integrated above baseline",
        ),
        (
            lambda item: item.__setitem__("core_solves_preserved", False),
            "success requires core_solves_preserved=true",
        ),
        (
            lambda item: item.__setitem__("levers_integrated", []),
            "success requires an integrated lever",
        ),
    ]
    for mutate, expected in success_mutations:
        changed = dict(success)
        mutate(changed)
        assert expected in exp4536.artifact_schema_errors(changed)

    with pytest.raises(ValueError, match="false_negative_risk_checked"):
        exp4536.run(
            root=tmp_path,
            write=False,
            preconditions_checked=_preconditions(),
            baseline={"core_efficiency": 0.0},
            load_upstream_artifacts=lambda _root: {
                "a1_per_level_goal_reinduction": _a1_artifact(),
                "a2_energy_trust_next_level_routing": _a2_artifact(),
            },
            gate_runner=lambda **_kwargs: _gate_result(),
            now=lambda: 10.0,
        )


def test_req_arc_wmte_4536_loader_fallbacks(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4536: loader fallbacks stay deterministic."""

    assert exp4536._read_json(tmp_path / "missing.json") == {}
    assert exp4536._levels_from_mapping(None) == {}
    assert exp4536._permitted_flagged_null({"flagged_adversarial": False}) is False
    assert exp4536._a1_delta({}) == 0.0
    assert exp4536._a1_delta({"core_efficiency_best": 2.5}) == 0.4926
    assert exp4536._a2_delta({}) == 0.0
    assert exp4536._a2_delta({"core_efficiency_energy_routed": 2.5}) == 0.4926
    assert exp4536._a1_control_levels({}) == {game: 0 for game in exp4536.CORE_GAMES}
    assert exp4536._a1_control_levels(
        {"deepest_level_reached_per_core_game": {"1": _levels(lp85=2)}}
    )["lp85"] == 2
    assert exp4536._a2_control_levels({}) == {game: 0 for game in exp4536.CORE_GAMES}
    assert exp4536._a2_control_levels(
        {"deepest_level_reached_per_core_game": {"no_energy_control": _levels(sp80=2)}}
    )["sp80"] == 2
    assert exp4536.select_integrated_levers(
        {
            **_a1_artifact(flagged=False, permitted_null=True),
            "target_levels_sweep": ["bad-row", *_a1_artifact()["target_levels_sweep"]],
        },
        _a2_artifact(),
    )["accepted_levers"] == []
    assert exp4536.load_gate_baseline(tmp_path / "missing")["core_efficiency"] == (
        exp4536.CORE_EFFICIENCY_BASELINE
    )

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )
    for name, relative in exp4536.UPSTREAM_ARTIFACTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")

    assert exp4536.load_gate_baseline(tmp_path)["core_efficiency"] == exp4536.CORE_EFFICIENCY_BASELINE
    assert sorted(exp4536.load_upstream_artifacts(tmp_path)) == sorted(exp4536.UPSTREAM_ARTIFACTS)
