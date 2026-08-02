"""The agent observes its own level-up and then structurally excludes it from every induce prompt.

THE DEFECT. `next_move` appends the level-up transition to `self.transitions` (line ~5527), and
only THEN calls `_observe_level_boundary` (line ~5572). That calls
`_begin_level_goal_episode`, which does:

    self._episode_transition_start = len(self.transitions)

At that moment `self.transitions` ALREADY CONTAINS the level-up row, so the new start index points
ONE PAST it. `_active_transitions()` returns `self.transitions[self._episode_transition_start:]`
-- and every live induction prompt is built from `_active_transitions()`. So the single positive
example the agent ever produces for itself, the transition where the level counter actually
incremented, is excluded from the prompt that asks it what winning looks like.

WHY THIS MATTERS MORE THAN AN OFF-BY-ONE. Measured 2026-08-01 over 138 frozen induced engines from
21 games: 93 are live, and 71 of those fail on the GOAL rather than the dynamics. Reading all 71,
34 are an unconditional `return False` and 3 more never return. Meanwhile the renderer that would
consume a win transition EXISTS and is careful --
`arc_executable_world_model._transitions_block` emits a "WIN TRANSITION" block whose comment
records that `next_grid` is the next level's opening board and that `win.grid` is one action
short, so it deliberately emits the labelled transition EVENT instead. Across every rebuilt live
prompt in that measurement the WIN TRANSITION block appears ZERO times. The consumer is correct
and starved.

62 of the 71 failures are cells whose window DID straddle a real level-up. The agent had the
positive example. The prompt did not show it.

WHAT THIS TEST DOES NOT ASSERT. It does not claim the fix is to move `_episode_transition_start`
back by one. That would pull a CROSS-LEVEL transition into the new level's DYNAMICS window, and
the dynamics of level N+1 are not the dynamics of the action that ended level N. The right shape
is to carry the win transition separately so the GOAL prompt can use it while the dynamics window
stays clean -- but this test pins the OBSERVABLE DEFECT, not a particular remedy, so it stays
valid under any fix that makes the win transition reachable.

SCENARIO-ARC-WMTE-4533-WIN-TRANSITION-REACHABLE
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy


def _policy() -> E3AgentPolicy:
    return E3AgentPolicy("lp85", proposer=object(), target_levels=3, value_head=None)


def _win_row() -> SimpleNamespace:
    """A stand-in for the transition whose action incremented the level counter."""
    return SimpleNamespace(name="THE_WIN_TRANSITION", level_before=0, level_after=1)


class TestTheWinTransitionIsReachable:
    def test_the_level_up_transition_is_excluded_from_active_transitions(self):
        """THE DEFECT, reproduced in the order the live loop produces it.

        `next_move` appends the transition BEFORE `_observe_level_boundary` runs, so this
        sequence is what actually happens on a real level-up -- not a contrived one.
        """
        policy = _policy()
        policy.transitions = [SimpleNamespace(name="ordinary")]
        policy._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=1)

        win = _win_row()
        policy.transitions.append(win)  # next_move appends it...
        policy._observe_level_boundary(  # ...and only then is the boundary observed
            SimpleNamespace(levels_completed=1), frames_seen=2
        )

        active = policy._active_transitions()
        assert win not in active, (
            "if this now passes the defect is fixed by inclusion in the dynamics window -- check "
            "that was intended, since a cross-level transition in the new level's dynamics is its "
            "own bug"
        )

    def test_the_win_transition_is_reachable_by_SOME_documented_route(self):
        """THE PROPERTY THAT MUST HOLD, independent of how it is achieved.

        The agent must be able to reach the transition it just used to complete a level, so the
        goal prompt can show it. This deliberately accepts ANY route -- a dedicated attribute, a
        list, an accessor -- because the fix should not be forced into one shape by its test.
        It fails today because no route exists at all.
        """
        policy = _policy()
        policy.transitions = [SimpleNamespace(name="ordinary")]
        policy._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=1)
        win = _win_row()
        policy.transitions.append(win)
        policy._observe_level_boundary(SimpleNamespace(levels_completed=1), frames_seen=2)

        reachable = []
        for attr in dir(policy):
            if "win" not in attr.lower() and "complete" not in attr.lower():
                continue
            try:
                value = getattr(policy, attr)
            except Exception:
                continue
            if value is win or isinstance(value, (list, tuple)) and win in value:
                reachable.append(attr)
        assert reachable, (
            "the win transition is unreachable from the policy by any attribute naming itself "
            "win/complete -- the agent cannot show the goal prompt the one positive example it "
            "produced for itself"
        )

    def test_the_grid_that_IS_captured_is_the_next_levels_opening_board(self):
        """Guards the thing that must NOT be 'fixed' by mistake.

        `_previous_level_complete_grid` IS captured at the boundary, and a reader may conclude the
        positive example is already available. It is not the win state: it comes from the frame
        AFTER the counter incremented, so it is the NEXT level's opening board. The 2026-07-29
        correction records this as "the most damaging instance of the win-state poison" -- the
        prompt used to assert it WAS a win state, teaching the model that completion looks like a
        freshly laid-out board. Anyone wiring the win transition must not re-introduce that.
        """
        policy = _policy()
        policy._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=1)
        policy.transitions = [_win_row()]
        events = policy._observe_level_boundary(SimpleNamespace(levels_completed=1), frames_seen=2)
        assert events, "a level-up must raise a boundary event"
        # No frame is supplied by this stand-in, so no grid is captured -- the point is that the
        # field carries a GRID at all, never the transition, so it cannot serve as the example.
        assert not isinstance(
            getattr(policy, "_previous_level_complete_grid", None), SimpleNamespace
        )


def test_the_consumer_exists_and_is_starved():
    """The renderer that would use a win transition is already written and already careful.

    This is what makes the defect a wiring failure rather than a missing feature: fixing the
    reachability is enough, because the prompt side is done.
    """
    from carnot.agentic import arc_executable_world_model as e3

    src = e3.__file__.replace(".pyc", ".py")
    with open(src) as fh:
        body = fh.read()
    assert "WIN TRANSITION (this is how the level was completed)" in body
    assert "Do NOT assume the rendered next frame is the win state" in body


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "--no-cov"]))
