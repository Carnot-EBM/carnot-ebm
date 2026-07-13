"""Regression test: E3AgentPolicy.next_move must not crash on a degenerate frame.

Found 2026-07-12 while running exp5587 (a broader/longer HUD-mask cascade check
than any prior E3AgentPolicy test had exercised): `next_move`'s transition-
collection block called `to_logical(grid_of(latest), self.cell)` UNGUARDED, while
the boundary-events block a few lines below already wraps its own `to_logical`
call in `try/except Exception: pass` for exactly this reason. A degenerate/empty
frame (grid_of(latest) returning a 1-D array, e.g. shape (0,) -- the same failure
class diagnosed earlier this session in the g50t apply_g50t_label incident, a
post-terminal empty-frame sentinel) made `h, w = grid.shape` raise
`ValueError: not enough values to unpack`, killing the entire game's remaining
action budget with an unhandled exception on the live scored path. This crashed
two of exp5587's six roster games (su15 and one other) within the first ~400
actions -- not a rare edge case at longer budgets.

Spec refs: REQ-ARC-WMTE-4491, REQ-ARC-WMTE-4494 (E3AgentPolicy's world-model
transition collection, the code path this fix hardens).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_competition_agent import E3AgentPolicy


class _DegenerateFrame:
    """Mimics an empty/degenerate terminal sentinel: grid_of() sees a 1-D array."""

    def __init__(self) -> None:
        self.frame = np.zeros((0,), dtype=np.int16)
        self.state = "NOT_FINISHED"
        self.levels_completed = 0


class _NormalFrame:
    def __init__(self, level: int = 0, value: int = 1) -> None:
        self.frame = np.full((8, 8), value, dtype=np.int16)
        self.state = "NOT_FINISHED"
        self.levels_completed = level


def test_next_move_survives_a_degenerate_frame_after_a_real_one() -> None:
    """A degenerate frame immediately following a real one must not crash
    next_move -- the exact sequence exp5587 hit (a normal frame sets self._prev,
    then a subsequent degenerate frame makes to_logical's grid.shape unpack
    fail)."""

    policy = E3AgentPolicy("cd82", proposer=None, value_head=lambda _f: 0.0)

    first = _NormalFrame(value=1)
    policy.next_move([], first)

    # Manually arm self._prev the way a real action would, then feed a
    # degenerate frame on the NEXT call -- this is exactly the code path that
    # crashed (the transition-collection block at the top of next_move).
    policy._prev = (np.zeros((8, 8), dtype=np.int16), 1, None)

    degenerate = _DegenerateFrame()
    # Must not raise. Prior to the fix this raised ValueError from to_logical.
    kind2, data2 = policy.next_move([first], degenerate)

    # The policy should still return SOME move (or a graceful terminal
    # signal), not propagate the exception.
    assert kind2 is None or isinstance(kind2, (int, str))


def test_next_move_continues_normally_after_recovering_from_a_degenerate_frame() -> None:
    """After skipping a malformed transition, the policy keeps functioning on
    subsequent normal frames -- the fix must not leave the policy in a broken
    state."""

    policy = E3AgentPolicy("cd82", proposer=None, value_head=lambda _f: 0.0)

    first = _NormalFrame(value=1)
    policy.next_move([], first)
    policy._prev = (np.zeros((8, 8), dtype=np.int16), 1, None)

    degenerate = _DegenerateFrame()
    policy.next_move([first], degenerate)

    # A normal frame right after the degenerate one must also not crash.
    policy._prev = (np.zeros((8, 8), dtype=np.int16), 1, None)
    recovered = _NormalFrame(value=2)
    kind3, data3 = policy.next_move([first, degenerate], recovered)
    assert kind3 is None or isinstance(kind3, (int, str))
