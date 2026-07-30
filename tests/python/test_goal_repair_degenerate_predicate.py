"""Goal-repair loop: rescue a DEGENERATE induced is_level_complete with the exemplar-derived
nonzero-count fallback so L1->L2 deepening is not blocked by a constant-false goal.

Origin: 2026-06-25 operator directive. The truncation fix makes the LLM EMIT code, but the induced
win-condition is often degenerate (`return False` / unreachable exact-match). This loop substitutes
the existing `_nonzero_count_predicate` fallback when the induced goal fails the harness
satisfiability check, and only when an L1-completion exemplar is available.

Spec refs: REQ-ARC-WMTE-4544 (bounded LLM re-induction loop — this extends its goal robustness).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_llm_reinduction import (
    _repair_degenerate_goal,
    execute_bounded_llm_reinduction,
)

pytestmark = pytest.mark.memory_watchdog_skip


# --------------------------------------------------------------------------- #
# Unit tests for the repair helper                                            #
# --------------------------------------------------------------------------- #
def _set_corner_engine(grid, _action, _data):
    """Engine that, on any action, fills the top-left cell (nonzero count -> 1)."""
    g = np.asarray(grid).copy()
    g[0, 0] = 1
    return g


def test_repair_returns_none_without_exemplar() -> None:
    """No L1-completion exemplar -> nothing to derive a fallback goal from -> give up (None)."""
    assert (
        _repair_degenerate_goal(
            engine=_set_corner_engine,
            previous_level_complete_grid=None,
            root_grid=np.zeros((1, 2), dtype=np.int16),
        )
        is None
    )


def test_repair_returns_satisfiable_fallback_when_reachable() -> None:
    """Exemplar fill level is reachable by the engine -> return the fallback predicate.

    RENAMED SOURCE (2026-07-29): the `source` string was `exemplar_nonzero_count_fallback`, which
    became factually wrong when the fallback was corrected. It is no longer a `>=`-against-nonzero
    bound and it is no longer referenced to the exemplar alone -- it is now the conjunction of
    "strictly more filled than the level ROOT" (which is what makes it False at the root, so the
    root-true rejection no longer kills the repair) and "at least the exemplar's fill level" (which
    is what keeps the give-up case below reachable). Only the label and the threshold's form changed;
    the assertions here are the same properties as before.
    """
    exemplar = np.array([[1, 0]], dtype=np.int16)  # one filled cell
    repaired = _repair_degenerate_goal(
        engine=_set_corner_engine,
        previous_level_complete_grid=exemplar,
        root_grid=np.zeros((1, 2), dtype=np.int16),
    )
    assert repaired is not None
    assert repaired["source"] == "exemplar_strictly_fuller_than_level_root_fallback"
    assert repaired["satisfiability"]["satisfiable"] is True
    # the returned predicate fires once the grid has >= 1 filled cell, false otherwise
    assert repaired["predicate"](np.array([[1, 0]])) is True
    assert repaired["predicate"](np.array([[0, 0]])) is False


def test_repair_returns_none_when_fallback_unreachable() -> None:
    """Exemplar demands more filled cells than the engine can ever reach -> None (genuine give-up)."""
    exemplar = np.array([[1, 1, 1]], dtype=np.int16)  # nonzero count = 3
    # engine can only ever set the top-left cell -> max reachable nonzero count = 1 < 3
    repaired = _repair_degenerate_goal(
        engine=_set_corner_engine,
        previous_level_complete_grid=exemplar,
        root_grid=np.zeros((1, 3), dtype=np.int16),
    )
    assert repaired is None


# --------------------------------------------------------------------------- #
# Integration: the repair unblocks the bounded re-induction loop              #
# --------------------------------------------------------------------------- #
class _Proposer:
    model_specs = "Qwen3.5-9B-MTP GGUF (/m.gguf)"

    def __init__(self) -> None:
        self.induce_calls = 0
        self.refactor_calls = 0

    def induce(self, _game, _transitions, _cell):
        self.induce_calls += 1
        return True, "candidate"

    def refactor(self, _game, _counterexample):
        self.refactor_calls += 1
        return True, "refined"


def _degenerate_goal(_grid):
    """The LLM's induced is_level_complete: constant-false (the degenerate failure mode)."""
    return False


def _plan_to_goal(engine, goal, start):
    """A one-step plan that reaches `goal` after applying `engine` once, else None."""
    nxt = np.asarray(engine(np.asarray(start), 1, None))
    return [{"action": 1, "data": None}] if bool(goal(nxt)) else None


def _run(previous_level_complete_grid):
    transitions = [
        Transition(
            grid=np.array([[0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 0]], dtype=np.int16),
            level_before=1,
            level_after=1,
        )
    ]
    proposer = _Proposer()
    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0, 0]], dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("c", engine, goal)],
        load_engine=lambda _game: (_set_corner_engine, _degenerate_goal),
        plan_in_model=_plan_to_goal,
        max_rounds=3,
        previous_level_complete_grid=previous_level_complete_grid,
    )
    return proposer, result


def test_repair_unblocks_planning_with_exemplar() -> None:
    """With an exemplar, a degenerate induced goal is repaired -> the loop plans on round 1."""
    proposer, result = _run(previous_level_complete_grid=np.array([[1, 0]], dtype=np.int16))
    assert result.planned is True
    assert result.plan == [{"action": 1, "data": None}]
    assert result.goal_predicate_satisfiable is True
    assert result.refinement_rounds_used == 1
    assert proposer.refactor_calls == 0  # repaired in place, no engine-refactor needed
    assert result.rounds[0]["goal_repaired"] == "exemplar_strictly_fuller_than_level_root_fallback"
    # the planned goal is the reachable fallback, not the degenerate constant-false predicate
    assert result.goal_predicate(np.array([[1, 0]])) is True


def test_control_no_exemplar_stays_degenerate() -> None:
    """CONTROL: identical setup WITHOUT an exemplar -> no repair -> degenerate, caps at 3 rounds.

    This proves the repair (not some other change) is what unblocks planning above."""
    proposer, result = _run(previous_level_complete_grid=None)
    assert result.planned is False
    assert result.skipped == "degenerate_goal_predicate"
    assert result.refinement_rounds_used == 3
    assert all("goal_repaired" not in row for row in result.rounds)


def _noop_engine(grid, _action, _data):
    return np.asarray(grid)


def test_repair_fires_on_refactor_round_not_just_round_one() -> None:
    """The repair block runs every round; here round 1's engine cannot reach the fallback (repair
    returns None -> degenerate -> continue), but round 2's refactored engine CAN -> repair fires on
    the REFACTOR round. Guards the round-2+ path the round-1 integration test does not exercise."""
    engines = iter(
        [
            (_noop_engine, _degenerate_goal),  # round 1: stuck -> fallback unreachable
            (_set_corner_engine, _degenerate_goal),  # round 2 (refactor): reachable -> repair
        ]
    )
    transitions = [
        Transition(
            grid=np.array([[0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 0]], dtype=np.int16),
            level_before=1,
            level_after=1,
        )
    ]
    proposer = _Proposer()
    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0, 0]], dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("c", engine, goal)],
        load_engine=lambda _game: next(engines),
        plan_in_model=_plan_to_goal,
        max_rounds=3,
        previous_level_complete_grid=np.array([[1, 0]], dtype=np.int16),
    )
    assert result.planned is True
    assert result.refinement_rounds_used == 2  # repaired on the 2nd (refactor) round
    assert proposer.refactor_calls == 1
    assert "goal_repaired" not in result.rounds[0]  # round 1 could not repair (unreachable)
    assert result.rounds[1]["goal_repaired"] == "exemplar_strictly_fuller_than_level_root_fallback"
