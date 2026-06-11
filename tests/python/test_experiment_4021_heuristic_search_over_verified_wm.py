"""Tests for Exp 4021 heuristic search over a verifier-certified world model.

Spec refs: REQ-PHASE4-030, SCENARIO-PHASE4-030.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.agentic.arc_heuristic_search_over_verified_wm import (
    REQUIRED_ARTIFACT_FIELDS,
    SearchResult,
    artifact_schema_errors,
    best_first_search,
    build_search_artifact,
    coded_goal_distance_heuristic,
    write_artifact,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))


def test_req_phase4_030_spec_declares_search_layer_contract() -> None:
    """REQ-PHASE4-030: OpenSpec declares Exp 4021 and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-030" in spec
    assert "SCENARIO-PHASE4-030" in spec
    assert "experiment_4021_heuristic_search_over_verified_wm.json" in spec
    assert "bound node expansions to at most 50000" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_030_best_first_search_uses_coded_heuristic() -> None:
    """SCENARIO-PHASE4-030: bounded best-first search finds a multi-step plan."""

    start = {"name": "start", "unsatisfied_targets": 2, "manhattan_to_target": 5}
    trap = {"name": "trap", "unsatisfied_targets": 2, "manhattan_to_target": 0}
    mid = {"name": "mid", "unsatisfied_targets": 1, "manhattan_to_target": 3}
    goal = {"name": "goal", "unsatisfied_targets": 0, "manhattan_to_target": 0}
    graph = {
        "start": [("trap", trap), ("move_a", mid)],
        "trap": [],
        "mid": [("move_b", goal)],
        "goal": [],
    }

    result = best_first_search(
        start,
        next_states=lambda state: graph[state["name"]],
        is_goal=lambda state: state["unsatisfied_targets"] == 0,
        heuristic=coded_goal_distance_heuristic,
        max_expansions=50,
    )

    assert result.solved is True
    assert result.actions == ["move_a", "move_b"]
    assert result.final_state == goal
    assert 0 < result.nodes_expanded <= 50
    assert result.bottleneck == ""


def test_req_phase4_030_search_handles_start_goal_and_private_key_fallback() -> None:
    """REQ-PHASE4-030: search handles immediate goals and non-dict cyclic states."""

    start_goal = {"unsatisfied_targets": 0, "manhattan_to_target": 0}
    immediate = best_first_search(
        start_goal,
        next_states=lambda state: [],
        is_goal=lambda state: state["unsatisfied_targets"] == 0,
    )
    cyclic: list[object] = []
    cyclic.append(cyclic)
    cyclic_result = best_first_search(
        cyclic,
        next_states=lambda state: [],
        is_goal=lambda state: False,
        max_expansions=1,
    )

    assert coded_goal_distance_heuristic(("not", "a", "dict")) == 0.0
    assert immediate.solved is True
    assert immediate.nodes_expanded == 0
    assert cyclic_result.solved is False
    assert cyclic_result.bottleneck == "frontier_exhausted"


def test_req_phase4_030_search_respects_expansion_bound() -> None:
    """REQ-PHASE4-030: node expansion accounting is capped by the configured bound."""

    result = best_first_search(
        {"name": "root", "unsatisfied_targets": 1, "manhattan_to_target": 10},
        next_states=lambda state: [
            (
                f"{state['name']}.{idx}",
                {
                    "name": f"{state['name']}.{idx}",
                    "unsatisfied_targets": 1,
                    "manhattan_to_target": 10,
                },
            )
            for idx in range(3)
        ],
        is_goal=lambda state: False,
        heuristic=coded_goal_distance_heuristic,
        max_expansions=7,
    )

    assert result.solved is False
    assert result.nodes_expanded == 7
    assert result.bottleneck == "expansion_bound_exhausted"


def test_req_phase4_030_search_prunes_duplicate_successor_states() -> None:
    """REQ-PHASE4-030: visited-state accounting skips duplicate verifier successors."""

    child = {"name": "child", "unsatisfied_targets": 1, "manhattan_to_target": 3}
    result = best_first_search(
        {"name": "root", "unsatisfied_targets": 2, "manhattan_to_target": 5},
        next_states=lambda state: [("first", child), ("duplicate", child)] if state["name"] == "root" else [],
        is_goal=lambda state: False,
        max_expansions=10,
    )

    assert result.solved is False
    assert result.nodes_expanded == 2
    assert result.bottleneck == "frontier_exhausted"


def test_req_phase4_030_artifact_schema_requires_bare_result_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-030: Exp 4021 artifacts keep required search verdict fields bare."""

    artifact = build_search_artifact(
        SearchResult(
            solved=True,
            actions=["move_a", "move_b"],
            nodes_expanded=4,
            final_state={"unsatisfied_targets": 0},
            bottleneck="",
            max_expansions=50000,
        ),
        game="r11l",
        target_level=4,
        prior_level=3,
        real_env_confirmed=True,
        heuristic_used="coded_unmet_targets_plus_manhattan_progress",
        inference_substrate="test_verified_world_model",
        duration_s=0.25,
    )
    output = write_artifact(artifact, tmp_path / "experiment_4021_heuristic_search_over_verified_wm.json")

    assert artifact["honest_verdict"] == "complete: search_layer_solved_r11l_L4_real_env_confirmed"
    assert artifact["new_levels_solved_this_task"] == 1
    assert artifact["wall_was_search_not_representation"] is True
    assert artifact["real_env_confirmed"] is True
    assert artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    missing = artifact_schema_errors({})
    assert any("missing required field honest_verdict" in err for err in missing)

    bad = dict(artifact)
    bad["honest_verdict"] = "done"
    bad["new_levels_solved_this_task"] = "1"
    bad["wall_was_search_not_representation"] = 1
    bad["nodes_expanded"] = "4"
    bad["heuristic_used"] = []
    bad["real_env_confirmed"] = 1
    bad["inference_substrate"] = None
    bad["nodes_expanded"] = 50001
    bad["max_expansions"] = 50000

    errors = artifact_schema_errors(bad)

    assert any("honest_verdict" in err for err in errors)
    assert any("new_levels_solved_this_task" in err for err in errors)
    assert any("wall_was_search_not_representation" in err for err in errors)
    assert any("nodes_expanded" in err for err in errors)
    assert any("heuristic_used" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)
    assert any("nodes_expanded must not exceed" in err for err in errors)


def test_scenario_phase4_030_no_solve_artifact_reports_bottleneck() -> None:
    """SCENARIO-PHASE4-030: no-solve verdicts remain complete and non-fabricated."""

    artifact = build_search_artifact(
        SearchResult(
            solved=False,
            actions=[],
            nodes_expanded=50000,
            final_state={"unsatisfied_targets": 2},
            bottleneck="expansion_bound_exhausted",
            max_expansions=50000,
        ),
        game="r11l",
        target_level=4,
        prior_level=3,
        real_env_confirmed=False,
        heuristic_used="coded_unmet_targets_plus_manhattan_progress",
        inference_substrate="test_verified_world_model",
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "complete: search_layer_no_solve_r11l_L4_expansion_bound_exhausted"
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact["wall_was_search_not_representation"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_030_runner_writes_mocked_real_env_confirmed_solution(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-030: experiment runner requires live levels_completed confirmation."""

    import experiment_4021_heuristic_search_over_verified_wm as exp

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_run_r11l_wall_search", lambda **kwargs: exp.R11LRunOutcome(
        search=SearchResult(
            solved=True,
            actions=[{"macro": "solve_group"}],
            nodes_expanded=12,
            final_state={"unsatisfied_targets": 0},
            bottleneck="",
            max_expansions=kwargs["max_expansions"],
        ),
        real_env_confirmed=True,
        levels_completed_after=4,
        executed_actions=6,
        per_step_replans=1,
        diagnosis="search found a verifier-simulated plan where single-step re-induction stalled",
    ))

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "complete: search_layer_solved_r11l_L4_real_env_confirmed"
    assert artifact["real_env_confirmed"] is True
    assert artifact["nodes_expanded"] == 12
    assert artifact_schema_errors(artifact) == []
    assert (tmp_path / "results" / exp.RESULT_NAME).exists()


def test_scenario_phase4_030_runner_does_not_claim_unconfirmed_solution(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-030: simulated solves without real-env confirmation are no-solve verdicts."""

    import experiment_4021_heuristic_search_over_verified_wm as exp

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_run_r11l_wall_search", lambda **kwargs: exp.R11LRunOutcome(
        search=SearchResult(
            solved=True,
            actions=[{"macro": "solve_group"}],
            nodes_expanded=8,
            final_state={"unsatisfied_targets": 0},
            bottleneck="real_env_confirmation_failed",
            max_expansions=kwargs["max_expansions"],
        ),
        real_env_confirmed=False,
        levels_completed_after=3,
        executed_actions=0,
        per_step_replans=1,
        diagnosis="simulated plan was not confirmed by levels_completed",
    ))

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "complete: search_layer_no_solve_r11l_L4_real_env_confirmation_failed"
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact["wall_was_search_not_representation"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact_schema_errors(artifact) == []
