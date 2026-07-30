"""Tests for wiring `score_goal_predicate_consistency` into a live veto inside
`execute_bounded_llm_reinduction` (task 7 follow-on: the check was built additive-only
and never consulted by any live decision until this).

Spec refs: REQ-ARC-WMTE-5593-3, SCENARIO-ARC-WMTE-5593-3-VETO-FIRES,
SCENARIO-ARC-WMTE-5593-3-VETO-OPT-IN, SCENARIO-ARC-WMTE-5593-3-FALSE-NEGATIVE-RISK-GUARD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
pytestmark = pytest.mark.memory_watchdog_skip


class _RepeatProposer:
    """Always returns the SAME candidate every round (matching
    test_req_arc_wmte_4544_bounded_refinement_caps_at_three_rounds's pattern) -- a
    proposer that never improves, so every round fails identically and the loop just
    exhausts `max_rounds`."""

    model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

    def __init__(self) -> None:
        self.induce_calls = 0
        self.refactor_calls = 0

    def induce(self, _game, _transitions, _cell):
        self.induce_calls += 1
        return True, "candidate"

    def refactor(self, _game, _counterexample):
        self.refactor_calls += 1
        return True, "refined"


def _progress_engine(grid, _action, _data):
    return np.asarray(grid) + 1


def _is_complete_at_two(grid) -> bool:
    return bool(np.asarray(grid)[0, 0] >= 2)


def _plan_in_model(engine, is_done, start_grid):
    grid = np.asarray(start_grid)
    plan: list[dict] = []
    for _ in range(4):
        grid = np.asarray(engine(grid, 1, None))
        plan.append({"action": 1, "data": None})
        if bool(is_done(grid)):
            return plan
    return None


def test_req_arc_wmte_5593_3_veto_fires_on_real_mismatch() -> None:
    """SCENARIO-ARC-WMTE-5593-3-VETO-FIRES: a predicate that disagrees with a real
    observed level-up is vetoed before planning, and the failure is fed to refactor()."""

    # A genuine level-up transition (level_before=1 -> level_after=2) whose next_grid is
    # [[1]] -- `_is_complete_at_two` returns False on it (1 < 2), a real false-negative
    # mismatch against the real observed level-up.
    #
    # FIXTURE EXTENDED 2026-07-29: a level-up row is now graded on the engine's COUNTERFACTUAL
    # rather than on `next_grid` (the rendered post-level-up frame is the next level's re-layout,
    # on which even a CORRECT predicate is False -- measured on ka59). That grading is only trusted
    # when the engine has independently earned it, which requires a NON-level-up row on the SAME
    # action whose real observed effect the engine reproduces. Without one the row is ungradeable
    # and the veto is inert -- correct, but it would make this test assert nothing.
    #
    # So the corroborating row below is added: action 1 on [[-2]] really does produce [[-1]], which
    # `_progress_engine` (grid + 1) predicts exactly. Its values stay BELOW the predicate's
    # threshold, so `_is_complete_at_two` is False there and no level-up occurred -- the row is
    # CONSISTENT and contributes no mismatch of its own. (A corroborating row above the threshold,
    # e.g. [[5]] -> [[6]], would itself be a genuine too-loose mismatch and would muddy what this
    # test is asserting.) The veto then fires on the level-up row exactly as before -- and now on
    # the counterfactual [[1]], where the predicate is still False, so the mismatch this test exists
    # to catch is genuinely caught.
    transitions = [
        Transition(
            grid=np.array([[-2]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[-1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
        Transition(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=2,
        ),
    ]
    proposer = _RepeatProposer()

    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: (_progress_engine, _is_complete_at_two),
        plan_in_model=_plan_in_model,
        max_rounds=1,
        min_goal_predicate_consistency=1.0,
    )

    assert result.planned is False
    assert result.rounds[0]["skipped"] == "goal_predicate_consistency_failed"
    # 0.5, not 0.0: the window now has TWO rows and the corroborating one is graded correctly. The
    # load-bearing property is unchanged -- accuracy is below the 1.0 threshold, so the veto fires.
    assert result.rounds[0]["goal_predicate_consistency_accuracy"] == 0.5
    assert result.rounds[0]["goal_predicate_consistency_accuracy"] < 1.0
    assert result.rounds[0]["goal_predicate_consistency_n_real_levelups"] == 1
    assert result.counterexamples[0]["kind"] == "goal_predicate_consistency_failed"
    # Index 1 is the level-up row (index 0 is the corroborating no-op row, which is consistent and
    # therefore contributes no mismatch).
    assert result.counterexamples[0]["mismatches"] == [
        {"i": 1, "real_levelup": True, "claimed": False}
    ]
    # Never reached plan_in_model -- the veto fires before planning is attempted at all.
    assert result.rounds[0].get("plan_length") is None


def test_req_arc_wmte_5593_3_veto_is_opt_in_disabled_by_default() -> None:
    """SCENARIO-ARC-WMTE-5593-3-VETO-OPT-IN: the SAME mismatching predicate is NOT
    vetoed when min_goal_predicate_consistency stays at its default (0.0, off) --
    backward-compatible with every existing caller that doesn't pass the new kwarg."""

    transitions = [
        Transition(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=2,
        )
    ]

    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=_RepeatProposer(),
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: (_progress_engine, _is_complete_at_two),
        plan_in_model=_plan_in_model,
        max_rounds=1,
        # min_goal_predicate_consistency omitted -- defaults to 0.0 (disabled).
    )

    assert "goal_predicate_consistency_accuracy" not in result.rounds[0]
    assert result.rounds[0].get("skipped") != "goal_predicate_consistency_failed"
    # With the veto off, planning is reached and the reachable goal is actually found.
    assert result.planned is True


def test_req_arc_wmte_5593_3_false_negative_risk_guard_no_real_levelup() -> None:
    """SCENARIO-ARC-WMTE-5593-3-FALSE-NEGATIVE-RISK-GUARD: a window with ZERO real
    level-ups cannot fairly judge a goal predicate (CLAUDE.md FALSE_NEGATIVE_RISK) --
    the veto must not fire even at a strict threshold, since any predicate trivially
    scores 1.0 on an all-no-op window and there is no informative signal either way."""

    transitions = [
        Transition(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=1,  # no real level-up
        )
    ]

    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=_RepeatProposer(),
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: (_progress_engine, _is_complete_at_two),
        plan_in_model=_plan_in_model,
        max_rounds=1,
        min_goal_predicate_consistency=1.0,
    )

    assert result.rounds[0]["goal_predicate_consistency_n_real_levelups"] == 0
    assert result.rounds[0].get("skipped") != "goal_predicate_consistency_failed"
    assert result.planned is True


def test_req_arc_wmte_5593_3_spec_declares_live_veto_contract() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5593-3") :]

    for marker in (
        "REQ-ARC-WMTE-5593-3",
        "SCENARIO-ARC-WMTE-5593-3-VETO-FIRES",
        "SCENARIO-ARC-WMTE-5593-3-VETO-OPT-IN",
        "SCENARIO-ARC-WMTE-5593-3-FALSE-NEGATIVE-RISK-GUARD",
        "min_goal_predicate_consistency",
        "goal_predicate_consistency_failed",
    ):
        assert marker in section
