"""Tests for Exp 4592 generation-completeness wiring.

Spec refs: REQ-CAPSTONE-4592, SCENARIO-CAPSTONE-4592,
SCENARIO-CAPSTONE-4592-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4592_generation_completeness_wiring as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions(games: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": True,
        "offline_arcade": True,
        "variant_generator_importable": True,
        "goal_distance_importable": True,
        "graph_explore_importable": True,
        "offline_env_public_games": list(games),
        "leaderboard_submission": False,
        "live_llm_inference": False,
    }


def _runner_factory(
    solved_by_mode: Mapping[str, set[str]],
    actions_by_mode: Mapping[str, Mapping[str, int]] | None = None,
    reproduced_by_mode: Mapping[str, set[str]] | None = None,
    mechanic_by_signature: Mapping[str, str] | None = None,
):
    actions_by_mode = actions_by_mode or {}
    reproduced_by_mode = reproduced_by_mode or solved_by_mode
    mechanic_by_signature = mechanic_by_signature or {}

    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            solved = signature in solved_by_mode.get(mode, set())
            reproduced = signature in reproduced_by_mode.get(mode, set())
            reached = 1 if solved else 0
            gate_level = reached if reproduced else 0
            actions = int(actions_by_mode.get(mode, {}).get(signature, 7 if solved else 21))
            mechanic = mechanic_by_signature.get(signature, "keyboard_graph")
            return {
                "game": game,
                "variant_signature": signature,
                "variant": int(spec["variant"]),
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": solved,
                "winner_generated": solved,
                "reached_level": reached,
                "actions": actions,
                "actions_to_first_levelup": actions if solved else None,
                "solution_labels": ["{\"action\":1,\"data\":null}"] if solved else [],
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": gate_level,
                    "reproduced": reproduced,
                },
                "selected_feature_route": {
                    "mechanic_class": mechanic,
                    "approach": "goal_distance_astar"
                    if mechanic in {"avatar_navigation", "click_connect"}
                    else "systematic_bfs",
                },
                "selected_approach": "goal_distance_astar"
                if mechanic in {"avatar_navigation", "click_connect"}
                else "systematic_bfs",
                "executed_approach": "goal_distance_astar"
                if mechanic in {"avatar_navigation", "click_connect"}
                else "graph_explore_solve_v2",
                "approach_variant_wired": mode == "wired",
                "wiring_mode": mode,
            }

        return run

    return _runner


def test_req_capstone_4592_spec_declares_generation_wiring_contract() -> None:
    """REQ-CAPSTONE-4592: OpenSpec declares the generation-wiring artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4592" in spec
    assert "SCENARIO-CAPSTONE-4592" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_capstone_4592_executor_routes_mechanic_classes_to_wired_generators() -> None:
    """REQ-CAPSTONE-4592: mechanic classes dispatch to goal-distance or graph generators."""

    assert (
        mod._executor_for_route({"mechanic_class": "avatar_navigation", "approach": "systematic_bfs"})
        == "goal_distance_astar"
    )
    assert (
        mod._executor_for_route({"mechanic_class": "click_connect", "approach": "systematic_bfs"})
        == "goal_distance_astar"
    )
    assert (
        mod._executor_for_route({"mechanic_class": "keyboard_graph", "approach": "goal_distance_astar"})
        == "graph_explore_solve_v2"
    )
    assert (
        mod._executor_for_route({"mechanic_class": "click_graph", "approach": "diversity_graph_explore"})
        == "graph_explore_solve_v2"
    )
    assert (
        mod._executor_for_route({"mechanic_class": "hidden_carry_state", "approach": "llm_reasoner"})
        == "default_graph_explore"
    )


def test_req_capstone_4592_wired_runner_dispatches_then_uses_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4592: selected generators run on variants before fallback is used."""

    calls: list[str] = []
    spec = {"variant": 1, "kind": "color", "reflect": None, "variant_signature": "g1~color01"}

    monkeypatch.setattr(
        mod,
        "_route_for_variant",
        lambda *_args, **_kwargs: {
            "mechanic_class": "avatar_navigation",
            "approach": "goal_distance_astar",
        },
    )

    def fake_goal(game: str, _spec: Mapping[str, Any], _budget: int, route: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(f"goal:{game}:{route['mechanic_class']}")
        return {
            "game": game,
            "variant_signature": "g1~color01",
            "attempted": True,
            "solved": False,
            "winner_generated": False,
            "selected_feature_route": dict(route),
            "selected_approach": "goal_distance_astar",
            "executed_approach": "goal_distance_astar",
            "approach_variant_wired": True,
        }

    def fake_default(game: str, spec_: Mapping[str, Any], budget: int) -> dict[str, Any]:
        calls.append(f"default:{game}:{budget}")
        return {
            "game": game,
            "variant_signature": spec_["variant_signature"],
            "attempted": True,
            "solved": True,
            "winner_generated": True,
            "actions": 3,
            "actions_to_first_levelup": 3,
            "reproduction_gate": {"reproduced": True, "reached_level": 1, "claimed_level": 1},
        }

    monkeypatch.setattr(mod, "_run_goal_distance_attempt", fake_goal)
    monkeypatch.setattr(mod.exp4550, "default_variant_runner", fake_default)

    attempt = mod.make_variant_runner("wired", policy={"routes": {}})("g1", spec, 12)

    assert calls == ["goal:g1:avatar_navigation", "default:g1:12"]
    assert attempt["solved"] is True
    assert attempt["fallback_used"] is True
    assert attempt["selected_attempt"]["executed_approach"] == "goal_distance_astar"
    assert attempt["executed_approach"] == "default_graph_explore"


def test_scenario_capstone_4592_artifact_reports_success_delta_and_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4592: wiring beats no-wiring on winner generation and transfer."""

    games = ("g1", "g2", "g3", "g4")
    baseline = {"g1~color01"}
    wired = {"g1~color01", "g2~color01", "g3~color01", "g4~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": baseline,
                "wired": wired,
            },
            mechanic_by_signature={
                "g2~color01": "avatar_navigation",
                "g3~color01": "click_connect",
                "g4~color01": "keyboard_graph",
            },
        ),
        n_bootstrap=80,
    )

    assert artifact["honest_verdict"] == (
        "success: generation_completeness_winner_generated_4of4_above_1of25"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["winner_generated_rate_baseline"] == pytest.approx(0.25)
    assert artifact["winner_generated_rate_with_wiring"] == pytest.approx(1.0)
    assert artifact["winner_generated_delta"] == pytest.approx(0.75)
    assert artifact["generic_transfer_rate_with_wiring"] == pytest.approx(1.0)
    assert artifact["transfer_delta"] == pytest.approx(0.75)
    assert artifact["no_wiring_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["solve_rate_preserved"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["chosen_submitted_config"] == "enable_wired_generation_dispatch"
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4592_honest_null_logs_residual_classes(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4592-FIELD-PRINCIPLES: zero winner delta is annotated."""

    games = ("g1", "g2")
    solved = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": solved,
                "wired": solved,
            },
            actions_by_mode={
                "baseline": {"g1~color01": 9},
                "wired": {"g1~color01": 9},
            },
            mechanic_by_signature={"g2~color01": "avatar_navigation"},
        ),
        n_bootstrap=40,
    )

    assert artifact["honest_verdict"] == (
        "complete: generation_completeness_no_value_honest_null_residual_logged"
    )
    assert artifact["winner_generated_delta"] == 0.0
    assert artifact["transfer_delta"] == 0.0
    assert artifact["actions_delta"] == 0.0
    assert artifact["no_wiring_control_passed"] is True
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["residual_unwired_classes"] == [
        "avatar_navigation:goal_distance_astar:winner_generated=0 count=1"
    ]
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4592_precondition_misses_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4592: missing required resources produce terminal blocked artifacts."""

    assert mod._first_precondition_miss({"offline_arcade": False}) == "offline_arcade"
    assert (
        mod._first_precondition_miss(
            {"offline_arcade": True, "variant_generator_importable": False}
        )
        == "variant_generator_import"
    )
    assert (
        mod._first_precondition_miss(
            {
                "offline_arcade": True,
                "variant_generator_importable": True,
                "goal_distance_importable": False,
            }
        )
        == "goal_distance_import"
    )
    assert (
        mod._first_precondition_miss(
            {
                "offline_arcade": True,
                "variant_generator_importable": True,
                "goal_distance_importable": True,
                "graph_explore_importable": False,
            }
        )
        == "graph_explore_import"
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        preconditions_checked={
            "offline_arcade": True,
            "variant_generator_importable": True,
            "goal_distance_importable": True,
            "graph_explore_importable": True,
            "leaderboard_submission": True,
        },
    )

    assert artifact["honest_verdict"] == "complete: blocked_leaderboard_submission"
    assert artifact["false_negative_risk_checked"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4592_validate_artifact_reports_schema_errors(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4592-FIELD-PRINCIPLES: validation rejects bad field shapes."""

    games = ("g1",)
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory({"baseline": set(), "wired": set()}),
        n_bootstrap=0,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not terminal"
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    bad["winner_generated_rate_with_wiring"] = "1"
    bad["generic_transfer_rate_baseline"] = "0"
    bad["transfer_ci"] = [0.0]
    bad["no_wiring_control_passed"] = "yes"
    bad["residual_unwired_classes"] = "none"
    bad["field_principles"] = {}

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "winner_generated_rate_with_wiring must be a bare float" in errors
    assert "generic_transfer_rate_baseline must be a bare float" in errors
    assert "transfer_ci must be [float, float]" in errors
    assert "no_wiring_control_passed must be a bare bool" in errors
    assert "residual_unwired_classes must be a list" in errors
    assert "missing field principle for honest_verdict" in errors


def test_req_capstone_4592_helper_branches_and_io_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4592: deterministic helper and wrapper branches stay covered."""

    assert mod._as_int("bad", default=3) == 3
    assert mod._gate_reproduced(None) is False
    assert (
        mod._executor_for_route({"mechanic_class": "unknown", "approach": "goal_distance_astar"})
        == "goal_distance_astar"
    )
    assert (
        mod._executor_for_route({"mechanic_class": "unknown", "approach": "systematic_bfs"})
        == "graph_explore_solve_v2"
    )
    assert (
        mod._labels_for_reproduction(
            [{"action": "6", "data": {"x": 2, "y": 3}}],
            {"reflect": None},
        )
        == ['{"action":6,"data":{"x":2,"y":3}}']
    )
    failed = mod._failed_attempt(
        game="g1",
        spec={"variant": 1, "kind": "color", "reflect": None, "variant_signature": "g1~color01"},
        route={"approach": "llm_reasoner"},
        executor="default_graph_explore",
        reason="llm_tail_disabled",
    )
    assert failed["blocked_reason"] == "llm_tail_disabled"
    assert failed["approach_variant_wired"] is False
    assert mod._paired_bootstrap_delta_ci([], [], random_seed=1, n_bootstrap=5) == [0.0, 0.0]

    monkeypatch.setattr(
        mod,
        "_probe_variant_signature",
        lambda _game, _spec: {
            "keyboard_effect_count": 1,
            "click_effect_count": 0,
            "avatar_motion_present": True,
            "cell_connect": False,
            "hidden_carry_state": False,
            "config_toggle": False,
        },
    )
    route = mod._route_for_variant(
        "g1",
        {"variant": 1, "kind": "color", "reflect": None, "variant_signature": "g1~color01"},
        policy={
            "routes": {
                "avatar_navigation": {
                    "approach": "goal_distance_astar",
                    "confidence": 1.0,
                    "source": "test",
                }
            },
            "default_approach": "default_graph_explore",
        },
    )
    assert route["mechanic_class"] == "avatar_navigation"
    assert route["approach"] == "goal_distance_astar"

    def fake_default(game: str, spec: Mapping[str, Any], budget: int) -> dict[str, Any]:
        return {
            "game": game,
            "variant_signature": spec["variant_signature"],
            "attempted": True,
            "solved": True,
            "winner_generated": True,
            "actions": budget,
            "actions_to_first_levelup": budget,
            "reproduction_gate": {"reproduced": True, "reached_level": 1, "claimed_level": 1},
        }

    monkeypatch.setattr(mod.exp4550, "default_variant_runner", fake_default)
    baseline_runner = mod.make_variant_runner("baseline", policy={"routes": {}})
    baseline_attempt = baseline_runner(
        "g1",
        {"variant": 1, "kind": "color", "reflect": None, "variant_signature": "g1~color01"},
        4,
    )
    assert baseline_attempt["wiring_mode"] == "baseline"
    assert baseline_attempt["approach_variant_wired"] is False

    monkeypatch.setattr(
        mod,
        "_route_for_variant",
        lambda *_args, **_kwargs: {"mechanic_class": "keyboard_graph", "approach": "systematic_bfs"},
    )

    def fake_graph(game: str, spec: Mapping[str, Any], budget: int, route: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "game": game,
            "variant_signature": spec["variant_signature"],
            "attempted": True,
            "solved": True,
            "winner_generated": True,
            "actions": budget,
            "actions_to_first_levelup": budget,
            "reproduction_gate": {"reproduced": True, "reached_level": 1, "claimed_level": 1},
            "selected_feature_route": dict(route),
            "selected_approach": route["approach"],
            "executed_approach": "graph_explore_solve_v2",
            "approach_variant_wired": True,
        }

    monkeypatch.setattr(mod, "_run_graph_attempt", fake_graph)
    wired_attempt = mod.make_variant_runner("wired", policy={"routes": {}})(
        "g1",
        {"variant": 1, "kind": "color", "reflect": None, "variant_signature": "g1~color01"},
        5,
    )
    assert wired_attempt["fallback_used"] is False
    assert wired_attempt["executed_approach"] == "graph_explore_solve_v2"

    monkeypatch.setattr(
        mod,
        "_route_for_variant",
        lambda *_args, **_kwargs: {"mechanic_class": "hidden_carry_state", "approach": "llm_reasoner"},
    )
    default_wired = mod.make_variant_runner("wired", policy={"routes": {}})(
        "g1",
        {"variant": 1, "kind": "color", "reflect": None, "variant_signature": "g1~color01"},
        6,
    )
    assert default_wired["fallback_used"] is False
    assert default_wired["executed_approach"] == "default_graph_explore"

    games = ("g1",)
    solved = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=8,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory({"baseline": solved, "wired": solved}),
        n_bootstrap=0,
    )
    assert artifact["residual_unwired_classes"] == [
        "unknown:default_graph_explore:winner_generated=0 count=0"
    ]

    missing_note = dict(artifact, null_delta_methodology_note="")
    assert (
        "null_delta_methodology_note required for zero winner_generated_delta"
        in mod.validate_artifact(missing_note)
    )
    missing_principles = dict(artifact, field_principles=[])
    assert "field_principles missing" in mod.validate_artifact(missing_principles)

    path = mod.write_artifact(tmp_path, artifact=artifact)
    assert path.exists()
    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, artifact=dict(artifact, honest_verdict="bad"))

    monkeypatch.setattr(mod, "build_artifact", lambda _root: artifact)
    writes: list[Mapping[str, Any]] = []
    monkeypatch.setattr(mod, "write_artifact", lambda _root, *, artifact: writes.append(artifact))
    assert mod.run(tmp_path, write=False) == artifact
    assert writes == []
    assert mod.run(tmp_path, write=True) == artifact
    assert writes == [artifact]
