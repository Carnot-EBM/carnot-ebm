"""Tests for Exp 4035 hierarchical search over the vc33 verified world model.

Spec refs: REQ-PHASE4-037, SCENARIO-PHASE4-037.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import carnot.agentic.arc_vc33_hierarchical_search as mod
from carnot.agentic.arc_heuristic_search_over_verified_wm import SearchResult
from carnot.agentic.arc_vc33_hierarchical_search import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    GridSearchOutcome,
    Subgoal,
    VC33VerifiedWorldModel,
    artifact_schema_errors,
    blocked_artifact,
    build_exp4035_artifact,
    component_landmark_click_actions,
    decompose_goal_predicate,
    grid_state_features,
    hierarchical_best_first_search,
    load_exp4035_preconditions,
    progress_bar_gap,
    run,
    vc33_goal_distance_heuristic,
    write_artifact,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _aligned_grid() -> np.ndarray:
    grid = np.full((16, 16), 3, dtype=np.int16)
    grid[1, :] = 7
    grid[3:5, 4:6] = 11
    grid[10:12, 4:6] = 11
    return grid


def _misaligned_grid() -> np.ndarray:
    grid = _aligned_grid()
    grid[10:12, 4:6] = 3
    grid[10:12, 9:11] = 11
    grid[1, 12:] = 5
    return grid


def _write_preconditions(root: Path, *, precision: float = 1.0, code: str | None = None) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    predicate_code = (
        "def is_goal(state):\n"
        '    return state["target_color_pairs"] > 0 and state["misaligned_target_pairs"] == 0\n'
        if code is None
        else code
    )
    (results / "experiment_4034_vc33_goal_predicate_induction.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: vc33_goal_predicate_induced_heldout_precision_1.000",
                "goal_predicate_heldout_precision": precision,
                "goal_predicate_code": predicate_code,
                "game": "vc33",
                "inference_substrate": "offline_arc_agi3_goal_predicate_induction_no_oracle",
            }
        )
    )
    (results / "arc3_vc33_world_model_program.py").write_text(
        "def predict(grid, action):\n"
        "    g = grid.copy()\n"
        "    if action[0] == 6 and int(action[1]) == 9 and int(action[2]) == 10:\n"
        "        g[10:12, 9:11] = 3\n"
        "        g[10:12, 4:6] = 11\n"
        "    elif action[0] == 6:\n"
        "        g[int(action[2]), int(action[1])] = 3\n"
        "    return g\n"
    )


def test_req_phase4_037_spec_declares_exp4035_contract() -> None:
    """REQ-PHASE4-037: OpenSpec declares Exp 4035 schema and search constraints."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-037" in spec
    assert "SCENARIO-PHASE4-037" in spec
    assert "experiment_4035_hierarchical_search_over_vc33_wm.json" in spec
    assert "blocked_vc33_goal_predicate_or_wm_missing" in spec
    assert "nodes_expanded <= 50000" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_037_features_heuristic_and_landmarks_are_goal_grounded() -> None:
    """SCENARIO-PHASE4-037: heuristic uses predicate components, distance, and progress."""

    misaligned = grid_state_features(_misaligned_grid())
    aligned = grid_state_features(_aligned_grid())
    actions = component_landmark_click_actions(_misaligned_grid(), max_actions=64)
    subgoals = decompose_goal_predicate(misaligned)

    assert misaligned["unsatisfied_targets"] == 1
    assert aligned["unsatisfied_targets"] == 0
    assert misaligned["manhattan_to_target"] > aligned["manhattan_to_target"]
    assert progress_bar_gap(_misaligned_grid()) > 0
    assert progress_bar_gap(np.zeros((1, 2, 3), dtype=np.int16)) == 0
    assert progress_bar_gap(np.array([[7, 7, 5, 5], [3, 3, 3, 3]], dtype=np.int16)) > 0
    assert vc33_goal_distance_heuristic(misaligned) > vc33_goal_distance_heuristic(aligned)
    assert (6, 4, 3) in actions
    assert len(actions) <= 64
    assert [subgoal.name for subgoal in subgoals] == ["reduce_misaligned_target_pairs_to_0", "full_goal_predicate"]


def test_scenario_phase4_037_hierarchical_search_plans_to_ordered_subgoals() -> None:
    """SCENARIO-PHASE4-037: subgoal search composes low-level best-first plans."""

    start = _misaligned_grid()
    aligned = _aligned_grid()
    calls: list[tuple[int, int]] = []

    def predict(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        calls.append(action)
        return aligned if action == (6, 9, 10) else grid.copy()

    model = VC33VerifiedWorldModel(start, predict, max_branching=16)
    subgoals = [
        Subgoal("reduce_misaligned_target_pairs_to_0", lambda state: state["misaligned_target_pairs"] == 0),
        Subgoal("full_goal_predicate", lambda state: state["unsatisfied_targets"] == 0),
    ]
    outcome = hierarchical_best_first_search(
        model,
        subgoals,
        is_goal=lambda state: state["unsatisfied_targets"] == 0,
        max_expansions=50,
    )

    assert outcome.search.solved is True
    assert outcome.search.actions == [(6, 9, 10)]
    assert outcome.nodes_expanded > 0
    assert outcome.subgoals_attempted == 2
    assert outcome.subgoals_reached == 2
    assert calls
    assert model.branching_factor() > 0.0


def test_scenario_phase4_037_search_reports_unreachable_subgoal_and_predict_errors() -> None:
    """SCENARIO-PHASE4-037: failed low-level searches preserve the bottleneck."""

    quiet_model = VC33VerifiedWorldModel(_misaligned_grid(), lambda grid, action: grid.copy(), max_branching=4)
    outcome = hierarchical_best_first_search(
        quiet_model,
        [Subgoal("impossible", lambda state: False)],
        is_goal=lambda state: False,
        max_expansions=3,
    )

    assert outcome.search.solved is False
    assert outcome.nodes_expanded == 1
    assert outcome.search.bottleneck == "frontier_exhausted"
    assert VC33VerifiedWorldModel(_aligned_grid(), lambda grid, action: grid).branching_factor() == 0.0

    def broken_predict(grid: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        raise RuntimeError("predict failed")

    broken_model = VC33VerifiedWorldModel(_misaligned_grid(), broken_predict, max_branching=4)

    assert broken_model.next_states(broken_model.start_state) == []
    assert broken_model.branching_factor() == 0.0


def test_req_phase4_037_artifact_schema_and_no_solve_verdict_are_bare() -> None:
    """REQ-PHASE4-037: Exp 4035 artifacts preserve required bare fields."""

    outcome = GridSearchOutcome(
        search=SearchResult(
            solved=False,
            actions=[],
            nodes_expanded=12,
            final_state={"unsatisfied_targets": 1},
            bottleneck="frontier_exhausted",
            max_expansions=50,
        ),
        nodes_expanded=12,
        branching_factor=5.5,
        subgoals_attempted=1,
        subgoals_reached=0,
    )
    artifact = build_exp4035_artifact(
        outcome,
        real_env_confirmed=False,
        levels_completed_after=0,
        duration_s=0.25,
        goal_predicate_precision=1.0,
        action_plan=[],
    )
    output = write_artifact(artifact, Path("/tmp/exp4035_test_artifact.json"))

    assert artifact["honest_verdict"] == "complete: search_layer_no_solve_vc33_frontier_exhausted"
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact["search_layer_generalizes"] is False
    assert artifact["heuristic_was_non_bespoke"] is True
    assert artifact["nodes_expanded"] == 12
    assert artifact["branching_factor"] == 5.5
    assert artifact["subgoal_decomposition_used"] is True
    assert artifact["real_env_confirmed"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    bad = dict(artifact)
    bad["honest_verdict"] = "done"
    bad["new_levels_solved_this_task"] = True
    bad["search_layer_generalizes"] = 0
    bad["heuristic_was_non_bespoke"] = 1
    bad["nodes_expanded"] = 51
    bad["branching_factor"] = "5.5"
    bad["subgoal_decomposition_used"] = 1
    bad["real_env_confirmed"] = 1
    bad["inference_substrate"] = None
    bad["max_expansions"] = 50

    errors = artifact_schema_errors(bad)
    missing = artifact_schema_errors({})

    assert any("honest_verdict" in err for err in errors)
    assert any("new_levels_solved_this_task" in err for err in errors)
    assert any("search_layer_generalizes" in err for err in errors)
    assert any("heuristic_was_non_bespoke" in err for err in errors)
    assert any("nodes_expanded" in err for err in errors)
    assert any("branching_factor" in err for err in errors)
    assert any("subgoal_decomposition_used" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)
    assert any("nodes_expanded must not exceed" in err for err in errors)
    assert any("missing required field honest_verdict" in err for err in missing)

    unconfirmed = build_exp4035_artifact(
        GridSearchOutcome(
            search=SearchResult(
                solved=True,
                actions=["symbolic_action"],
                nodes_expanded=2,
                final_state={"unsatisfied_targets": 0},
                bottleneck="",
                max_expansions=10,
            ),
            nodes_expanded=2,
            branching_factor=1.0,
            subgoals_attempted=1,
            subgoals_reached=1,
        ),
        real_env_confirmed=False,
        levels_completed_after=0,
        duration_s=0.1,
        goal_predicate_precision=1.0,
        action_plan=["symbolic_action"],
    )

    assert unconfirmed["honest_verdict"] == "complete: search_layer_no_solve_vc33_real_env_confirmation_failed"
    assert unconfirmed["action_plan"] == ["symbolic_action"]


def test_scenario_phase4_037_preconditions_block_without_valid_predicate_or_wm(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-037: invalid Exp 4034 or world model stops honestly."""

    missing = load_exp4035_preconditions(tmp_path)
    artifact = blocked_artifact(0.0, missing.errors)

    assert missing.ok is False
    assert missing.predict is None
    assert artifact["honest_verdict"] == "blocked_vc33_goal_predicate_or_wm_missing"
    assert artifact_schema_errors(artifact) == []

    _write_preconditions(tmp_path, precision=0.49)
    low_precision = load_exp4035_preconditions(tmp_path)

    assert low_precision.ok is False
    assert any("precision" in error for error in low_precision.errors)

    _write_preconditions(tmp_path, precision=1.0, code="")
    empty_code = load_exp4035_preconditions(tmp_path)

    assert empty_code.ok is False
    assert any("goal_predicate_code is empty" in error for error in empty_code.errors)

    list_path = tmp_path / "list.json"
    list_path.write_text("[]")
    bad_predict = tmp_path / "bad_predict.py"
    bad_predict.write_text("VALUE = 1\n")

    try:
        mod._load_json_object(list_path)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("non-object JSON must fail precondition loading")

    try:
        mod._load_predict(bad_predict)
    except ValueError as exc:
        assert "predict" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("world-model code without predict must fail")


def test_scenario_phase4_037_run_writes_blocked_and_mocked_confirmed_artifacts(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-037: runner writes blocked paths and real-confirmed solve paths."""

    blocked = run(repo_root=tmp_path, write=True)
    blocked_output = tmp_path / "results" / "experiment_4035_hierarchical_search_over_vc33_wm.json"

    assert blocked["honest_verdict"] == "blocked_vc33_goal_predicate_or_wm_missing"
    assert blocked_output.exists()

    _write_preconditions(tmp_path)
    monkeypatch.setattr(
        "carnot.agentic.arc_vc33_hierarchical_search._initial_vc33_grid",
        lambda root: _misaligned_grid(),
    )
    monkeypatch.setattr(
        "carnot.agentic.arc_vc33_hierarchical_search._execute_plan_in_real_env",
        lambda root, actions: (True, 1),
    )

    confirmed = run(repo_root=tmp_path, write=True, max_expansions=50)

    assert confirmed["honest_verdict"] == "complete: search_layer_solved_vc33_L1_real_env_confirmed"
    assert confirmed["new_levels_solved_this_task"] == 1
    assert confirmed["search_layer_generalizes"] is True
    assert confirmed["real_env_confirmed"] is True
    assert artifact_schema_errors(confirmed) == []
