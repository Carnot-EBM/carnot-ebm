"""Tests for Exp 4582 early-play feature router transfer.

Spec refs: REQ-CAPSTONE-4582, SCENARIO-CAPSTONE-4582,
SCENARIO-CAPSTONE-4582-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4582_feature_router_transfer as mod
from carnot.agentic import arc_solve_learning as learning


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions(games: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": True,
        "offline_arcade": True,
        "recommend_approach_importable": True,
        "offline_env_public_games": list(games),
        "leaderboard_submission": False,
    }


def _runner_factory(
    solved_by_mode: Mapping[str, set[str]],
    actions_by_mode: Mapping[str, Mapping[str, int]] | None = None,
):
    actions_by_mode = actions_by_mode or {}

    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            solved = signature in solved_by_mode.get(mode, set())
            reached = 1 if solved else 0
            actions = int(actions_by_mode.get(mode, {}).get(signature, 8 if solved else 19))
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
                "solution_labels": ["ACTION1"] if solved else [],
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": reached,
                    "reproduced": solved,
                },
                "blocked_reason": "",
                "feature_router_mode": mode,
            }

        return run

    return _runner


def test_req_capstone_4582_spec_declares_feature_router_contract() -> None:
    """REQ-CAPSTONE-4582: OpenSpec declares the feature-router artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4582" in spec
    assert "SCENARIO-CAPSTONE-4582" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_capstone_4582_extracts_and_classifies_early_play_features() -> None:
    """REQ-CAPSTONE-4582: first-K action effects classify mechanics without LLM calls."""

    nav_transitions = [
        {
            "action_id": 1,
            "before": [[0, 4, 0], [0, 0, 0]],
            "after": [[0, 0, 4], [0, 0, 0]],
        },
        {
            "action_id": 6,
            "data": {"x": 1, "y": 1},
            "before": [[0, 0], [0, 0]],
            "after": [[0, 0], [3, 3]],
        },
    ]
    nav_signature = learning.extract_early_play_signature(nav_transitions)

    assert nav_signature["keyboard_effect_count"] == 1
    assert nav_signature["click_effect_count"] == 1
    assert nav_signature["avatar_motion_present"] is True
    assert nav_signature["cell_connect"] is True
    assert learning.classify_early_play_mechanic(nav_signature) == "avatar_navigation"

    toggle_signature = learning.extract_early_play_signature(
        [
            {
                "action_id": 6,
                "data": {"x": 0, "y": 0},
                "before": [[0, 0], [0, 0]],
                "after": [[5, 0], [0, 0]],
            },
            {
                "action_id": 6,
                "data": {"x": 0, "y": 0},
                "before": [[5, 0], [0, 0]],
                "after": [[0, 0], [0, 0]],
            },
        ]
    )
    assert toggle_signature["config_toggle"] is True
    assert learning.classify_early_play_mechanic(toggle_signature) == "config_toggle"


def test_req_capstone_4582_classifier_defensive_branches() -> None:
    """REQ-CAPSTONE-4582: signature helpers stay total on noisy early-play probes."""

    assert learning._grid_tuple(None) == ()
    assert learning._action_type("") == "unknown"
    assert learning._grid_tuple({"grid": [[1]]}) == ((1,),)
    assert learning._grid_tuple(SimpleNamespace(frame=[[2]])) == ((2,),)
    assert learning._grid_tuple(7) == ()
    assert learning._changed_cells(((1,),), ()) == []
    assert learning._changed_cells(((1, 2),), ((1,),)) == []
    assert learning._translated_visible_object((), ((1,),)) is False
    assert learning._translated_visible_object(((0, 1),), ((0, 1),)) is False
    assert learning._translated_visible_object(((0, 1, 1),), ((0, 1, 0),)) is False
    assert learning._is_click_action("bad", None) is False
    assert learning._is_keyboard_action("bad", None) is False
    assert learning._cell_connect_effect([]) is False
    assert learning._cell_connect_effect([(0, 0, 0, 1), (0, 1, 0, 2)]) is False
    assert learning._cell_connect_effect([(0, 0, 0, 1), (1, 1, 0, 1)]) is False

    hidden_by_flag = learning.extract_early_play_signature(
        {"transitions": [{"action_id": 1, "before": [[0]], "after": [[0]], "hidden_state_changed": True}]}
    )
    assert learning.classify_early_play_mechanic(hidden_by_flag) == "hidden_carry_state"

    hidden_by_repeated_effect = learning.extract_early_play_signature(
        [
            {"action_id": 1, "before": [[1]], "after": [[2]]},
            {"action_id": 1, "before": [[1]], "after": [[3]]},
            {"action_id": 2, "before": [[4]], "after": [[4]]},
        ]
    )
    assert hidden_by_repeated_effect["hidden_carry_state"] is True

    click_connect = {
        "keyboard_effect_count": 0,
        "click_effect_count": 1,
        "avatar_motion_present": False,
        "cell_connect": True,
        "hidden_carry_state": False,
        "config_toggle": False,
    }
    keyboard_graph = dict(click_connect, click_effect_count=0, cell_connect=False, keyboard_effect_count=1)
    click_graph = dict(click_connect, cell_connect=False)
    assert learning.classify_early_play_mechanic(click_connect) == "click_connect"
    assert learning.classify_early_play_mechanic(keyboard_graph) == "keyboard_graph"
    assert learning.classify_early_play_mechanic(click_graph) == "click_graph"


def test_req_capstone_4582_learns_class_to_approach_from_pos_neg_traces() -> None:
    """REQ-CAPSTONE-4582: positive/negative self-play traces select the winning approach."""

    traces = [
        {
            "mechanic_class": "avatar_navigation",
            "approach": "systematic_bfs",
            "solved": False,
        },
        {
            "mechanic_class": "avatar_navigation",
            "approach": "goal_distance_astar",
            "solved": True,
            "actions_to_first_levelup": 6,
        },
        {
            "mechanic_class": "config_toggle",
            "approach": "diversity_graph_explore",
            "solved": True,
            "actions_to_first_levelup": 10,
        },
    ]

    policy = learning.learn_feature_router_policy(traces)
    route = learning.route_feature_approach(
        {
            "keyboard_effect_count": 2,
            "click_effect_count": 0,
            "avatar_motion_present": True,
            "cell_connect": False,
            "hidden_carry_state": False,
            "config_toggle": False,
        },
        policy=policy,
    )

    assert policy["routes"]["avatar_navigation"]["approach"] == "goal_distance_astar"
    assert route["mechanic_class"] == "avatar_navigation"
    assert route["approach"] == "goal_distance_astar"
    assert route["confidence"] > 0.0


def test_req_capstone_4582_trace_policy_parses_solver_and_result_variants(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4582: trace learning tolerates registry, ledger, and result rows."""

    assert learning._coarse_mechanic_class("hidden checkpoint") == "hidden_carry_state"
    assert learning._coarse_mechanic_class("", {"action_type": "keyboard"}) == "avatar_navigation"
    assert learning._coarse_mechanic_class("", {"action_type": "click"}) == "click_connect"
    assert learning._approach_from_trace({"winner": "cell_count"}) == "goal_distance_astar"
    assert learning._approach_from_trace({"winner": "bfs"}) == "systematic_bfs"
    assert learning._approach_from_trace({"solver": "goal_distance_a_star"}) == "goal_distance_astar"
    assert learning._approach_from_trace({"solver": "llm reasoner"}) == "llm_reasoner"
    assert learning._approach_from_trace({"solver": "go-explore diversity"}) == "diversity_graph_explore"
    assert learning._approach_from_trace({"solver": "graph bfs"}) == "systematic_bfs"
    assert learning._trace_solved({"outcome": {"reproduced": True}}) is True
    assert learning._trace_solved({"result": "won"}) is True

    noisy_root = tmp_path / "noisy"
    (noisy_root / "ops").mkdir(parents=True)
    (noisy_root / "results").mkdir()
    (noisy_root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump({"games": ["bad-row"]}),
        encoding="utf-8",
    )
    (noisy_root / "ops" / "arc_router_ledger.json").write_text(
        '{"entries":["bad-row"]}',
        encoding="utf-8",
    )
    (noisy_root / "results" / "arc_loop_solve_bad.json").write_text(
        "{not json",
        encoding="utf-8",
    )
    assert learning._load_feature_router_trace_rows(noisy_root) == []

    broken_root = tmp_path / "broken"
    (broken_root / "ops").mkdir(parents=True)
    (broken_root / "results").mkdir()
    (broken_root / "ops" / "arc_solve_registry.yaml").write_text("[", encoding="utf-8")
    (broken_root / "ops" / "arc_router_ledger.json").write_text("not-json", encoding="utf-8")
    (broken_root / "results" / "arc_loop_solve_list.json").write_text("[]", encoding="utf-8")
    assert learning._load_feature_router_trace_rows(broken_root) == []

    policy = learning.learn_feature_router_policy(
        [
            "bad-row",
            {"mechanic": "unknown", "selected_approach": "default_graph_explore", "result": "lost"},
        ]
    )
    assert policy["routes"]["unknown"]["approach"] == "default_graph_explore"


def test_req_capstone_4582_recommend_approach_exposes_feature_router() -> None:
    """REQ-CAPSTONE-4582: recommend_approach extends routing while preserving fallback."""

    policy = learning.learn_feature_router_policy(
        [
            {
                "mechanic_class": "config_toggle",
                "approach": "diversity_graph_explore",
                "solved": True,
                "actions_to_first_levelup": 7,
            }
        ]
    )

    rec = learning.recommend_approach(
        "zz99_unseen",
        early_play_signature={
            "keyboard_effect_count": 0,
            "click_effect_count": 2,
            "avatar_motion_present": False,
            "cell_connect": False,
            "hidden_carry_state": False,
            "config_toggle": True,
        },
        feature_router_policy=policy,
    )

    assert rec["confident_transfer"] is False
    assert rec["feature_router"]["enabled"] is True
    assert rec["feature_router"]["mechanic_class"] == "config_toggle"
    assert rec["feature_router"]["approach"] == "diversity_graph_explore"
    assert rec["feature_router"]["no_regression_fallback"]


def test_req_capstone_4582_recommend_approach_known_game_and_disabled_router() -> None:
    """REQ-CAPSTONE-4582: known-game recommendations also carry disabled router metadata."""

    disabled = learning._feature_router_payload(
        early_play_signature=None,
        feature_router_policy=None,
        feature_router_traces=None,
        fallback_strategy={"solver": "fallback_solver"},
    )
    assert disabled["enabled"] is False
    assert disabled["no_regression_fallback"] == "fallback_solver"

    rec = learning.recommend_approach("tr87")
    assert rec["target_game"] == "tr87"
    assert rec["feature_router"]["enabled"] is False
    assert "heuristic_policy" in rec


def test_req_capstone_4582_recommend_approach_program_editor_without_registry_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4582: known survey rows without registry rows still route safely."""

    from carnot.agentic import arc_primitive_library
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(
        learning,
        "_survey_features",
        lambda: {
            "fake": {
                "game": "fake",
                "action_type": "click",
                "spatial": False,
                "difficulty": "",
                "win_kw": set(),
            }
        },
    )
    monkeypatch.setattr(learning, "_registry", lambda: {"games": [], "general_gotchas": []})
    monkeypatch.setattr(arc_primitive_library, "retrieve_primitives", lambda *a, **k: [])
    monkeypatch.setattr(arc_solver_kit, "select_primitive_operators", lambda **_kwargs: [])

    assert isinstance(learning._registry(), dict)
    rec = learning.recommend_approach("fake", mechanic="program_editor")

    assert rec["target_game"] == "fake"
    assert rec["heuristic_policy"]["strategy_solver"].startswith(
        "carnot.agentic.arc_program_editor_model"
    )
    assert rec["retrieved_primitives"] == []


def test_scenario_capstone_4582_artifact_reports_success_delta_and_controls(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4582: router beats default and random route on same variants."""

    games = ("g1", "g2", "g3", "g4")
    baseline = {"g1~color01"}
    router = {"g1~color01", "g2~color01", "g3~color01", "g4~color01"}
    random = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": baseline,
                "feature_router": router,
                "random_route": random,
            }
        ),
        n_bootstrap=80,
    )

    assert artifact["honest_verdict"] == (
        "success: feature_router_generic_transfer_1.000_above_0.04"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["generic_transfer_rate_baseline"] == pytest.approx(0.25)
    assert artifact["generic_transfer_rate_with_router"] == pytest.approx(1.0)
    assert artifact["transfer_delta"] == pytest.approx(0.75)
    assert artifact["solve_rate_preserved"] is True
    assert artifact["random_route_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["winner_generated"]["with_router"] is True
    assert artifact["chosen_submitted_config"] == "enable_feature_router"
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4582_honest_null_records_actions_and_gap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4582-FIELD-PRINCIPLES: zero transfer delta is annotated."""

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
                "feature_router": solved,
                "random_route": set(),
            },
            actions_by_mode={
                "baseline": {"g1~color01": 9},
                "feature_router": {"g1~color01": 9},
            },
        ),
        n_bootstrap=40,
    )

    assert artifact["honest_verdict"] == (
        "complete: feature_router_no_value_honest_null_transfer_gap_sharpened"
    )
    assert artifact["transfer_delta"] == 0.0
    assert artifact["actions_delta"] == 0.0
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["random_route_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["missing_verifier_gaps"]
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4582_precondition_misses_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4582: missing resources produce terminal blocked artifacts."""

    assert mod._first_precondition_miss({"offline_arcade": False}) == "offline_arcade"
    assert (
        mod._first_precondition_miss(
            {"offline_arcade": True, "recommend_approach_importable": False}
        )
        == "recommend_approach_import"
    )
    assert (
        mod._first_precondition_miss(
            {
                "offline_arcade": True,
                "recommend_approach_importable": True,
                "leaderboard_submission": True,
            }
        )
        == "leaderboard_submission"
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        preconditions_checked={"offline_arcade": False, "recommend_approach_importable": True},
    )

    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade"
    assert artifact["false_negative_risk_checked"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4582_route_for_variant_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4582: route selection covers baseline, random, and probe failures."""

    spec = {"variant": 1, "variant_signature": "g1~color01", "kind": "color"}
    policy = learning.learn_feature_router_policy([])

    random_route = mod._route_for_variant("random_route", "g1", spec, policy=policy)
    assert random_route["mechanic_class"] == "random_route_control"
    assert random_route["approach"] in learning.FEATURE_ROUTER_APPROACHES

    baseline = mod._route_for_variant("baseline", "g1", spec, policy=policy)
    assert baseline["enabled"] is False
    assert baseline["approach"] == "default_graph_explore"

    def fail_probe(_game: str, _spec: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(mod, "_probe_variant_signature", fail_probe)
    feature = mod._route_for_variant("feature_router", "g1", spec, policy=policy)
    assert feature["enabled"] is True
    assert feature["signature"]["probe_error"] == "RuntimeError: probe unavailable"
    assert feature["approach"] == "default_graph_explore"


def test_req_capstone_4582_bootstrap_degenerate_paths_and_gap_summary() -> None:
    """REQ-CAPSTONE-4582: CI and gap helpers handle empty and deterministic cases."""

    assert (
        mod._paired_bootstrap_delta_ci([], [], random_seed=1, n_bootstrap=10)
        == [0.0, 0.0]
    )
    baseline = [{"variant_signature": "a", "attempted": True, "solved": False}]
    router = [{"variant_signature": "a", "attempted": True, "solved": True}]
    assert (
        mod._paired_bootstrap_delta_ci(baseline, router, random_seed=1, n_bootstrap=0)
        == [1.0, 1.0]
    )
    assert mod._dominant_route_gaps([{"attempted": True, "solved": True}]) == []
    gaps = mod._dominant_route_gaps(
        [
            {
                "attempted": True,
                "solved": False,
                "selected_approach": "goal_distance_astar",
                "approach_variant_wired": False,
            },
            {
                "attempted": True,
                "solved": False,
                "selected_feature_route": {"mechanic_class": "avatar_navigation"},
                "selected_approach": "goal_distance_astar",
                "approach_variant_wired": False,
            },
        ]
    )
    assert gaps[0].startswith("feature_router_residual_generation_gap")


def test_scenario_capstone_4582_action_win_without_transfer_gain(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4582: lower solved-variant actions can be a win if solve rate is preserved."""

    games = ("g1", "g2")
    solved = {"g1~color01", "g2~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": solved,
                "feature_router": solved,
                "random_route": solved,
            },
            actions_by_mode={
                "baseline": {"g1~color01": 12, "g2~color01": 12},
                "feature_router": {"g1~color01": 7, "g2~color01": 7},
                "random_route": {"g1~color01": 10, "g2~color01": 10},
            },
        ),
        n_bootstrap=20,
    )

    assert artifact["honest_verdict"] == (
        "success: feature_router_actions_to_first_levelup_lower_solve_rate_preserved"
    )
    assert artifact["transfer_delta"] == 0.0
    assert artifact["actions_delta"] == 5.0
    assert artifact["random_route_control_passed"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4582_no_unsolved_gap_fallback(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4582: random-control ties leave false-negative risk open."""

    games = ("g1",)
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
                "feature_router": solved,
                "random_route": solved,
            },
            actions_by_mode={
                "baseline": {"g1~color01": 8},
                "feature_router": {"g1~color01": 8},
                "random_route": {"g1~color01": 8},
            },
        ),
        n_bootstrap=10,
    )

    assert artifact["honest_verdict"] == (
        "complete: feature_router_no_value_control_failed_false_negative_risk_open"
    )
    assert artifact["random_route_control_passed"] is False
    assert artifact["false_negative_risk_checked"] is True
    assert "no-value null is not closed" in artifact["null_delta_methodology_note"]
    assert artifact["missing_verifier_gaps"] == [
        "feature_router_no_value_added; no newly generated winning variant"
    ]


def test_scenario_capstone_4582_validate_artifact_reports_schema_errors(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4582-FIELD-PRINCIPLES: validation rejects bad field shapes."""

    games = ("g1",)
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": set(),
                "feature_router": set(),
                "random_route": set(),
            }
        ),
        n_bootstrap=0,
    )
    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "pending",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "generic_transfer_rate_with_router": "0",
            "generic_transfer_rate_baseline": "0",
            "transfer_delta": "0",
            "actions_delta": "0",
            "random_route_control_passed": "false",
            "false_negative_risk_checked": "false",
            "solve_rate_preserved": "false",
            "offline_reproduced": "false",
            "transfer_ci": [0.0],
            "null_delta_methodology_note": "",
            "winner_generated": [],
            "missing_verifier_gaps": {},
            "field_principles": {},
        }
    )

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "generic_transfer_rate_with_router must be a bare float" in errors
    assert "random_route_control_passed must be a bare bool" in errors
    assert "transfer_ci must be [float, float]" in errors
    assert "winner_generated must be a mapping" in errors
    assert "missing_verifier_gaps must be a list" in errors
    assert any(error.startswith("missing field principle") for error in errors)

    note_errors = mod.validate_artifact(dict(artifact, null_delta_methodology_note=""))
    assert "null_delta_methodology_note required for zero transfer_delta" in note_errors
    principle_errors = mod.validate_artifact(dict(artifact, field_principles=[]))
    assert "field_principles missing" in principle_errors


def test_scenario_capstone_4582_write_and_run_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4582: write_artifact and run produce the JSON artifact."""

    games = ("g1",)
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": set(),
                "feature_router": set(),
                "random_route": set(),
            }
        ),
        n_bootstrap=0,
    )
    path = mod.write_artifact(tmp_path, artifact=artifact)
    assert path.exists()
    assert json_load(path)["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, artifact=dict(artifact, honest_verdict="pending"))

    calls: list[tuple[str, Any]] = []

    def fake_build(root: Path | str) -> dict[str, Any]:
        calls.append(("build", root))
        return artifact

    def fake_write(root: Path | str, *, artifact: Mapping[str, Any] | None = None) -> Path:
        calls.append(("write", artifact))
        return tmp_path / "written.json"

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "write_artifact", fake_write)
    assert mod.run(tmp_path, write=True) == artifact
    assert calls[0][0] == "build"
    assert calls[1][0] == "write"
    calls.clear()
    assert mod.run(tmp_path, write=False) == artifact
    assert calls == [("build", tmp_path)]


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
