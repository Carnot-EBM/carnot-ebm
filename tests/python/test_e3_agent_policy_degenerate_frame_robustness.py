"""Regression test: E3AgentPolicy.next_move must not crash on a degenerate frame.

Found 2026-07-12 while running exp5587 (a broader/longer HUD-mask cascade check
than any prior E3AgentPolicy test had exercised). TWO separate unguarded call
sites in `next_move` broke on a degenerate/empty frame (grid_of(latest)
returning a 1-D array, e.g. shape (0,) -- the same failure class diagnosed
earlier this session in the g50t apply_g50t_label incident, a post-terminal
empty-frame sentinel):

  1. The transition-collection block (`to_logical(grid_of(latest), self.cell)`,
     ~line 3073) -- fixed first by wrapping it in try/except, mirroring the
     boundary-events block a few lines below which already did this.
  2. The tier-1 explore-phase per-action tick (~line 3114-3120,
     `detect_cell(grid_of(latest))` / `to_logical(...)`) -- found on the
     RE-RUN after fix #1, on the MAIN every-action tier-1 path (not just
     tier-3 induction), completely unguarded.

Two crash sites in the same failure class is a signal the underlying
functions themselves were the real gap, not each call site individually --
so the actual fix is in `detect_cell`/`to_logical`
(`arc_executable_world_model.py`): both now return a safe, defined result
(cell=1, or the grid unchanged) for a non-2D grid instead of raising, so
every current AND future call site is covered without having to keep
finding and patching each one. The transition-collection try/except from
fix #1 is kept as defense-in-depth (it also guards other exceptions in that
block, e.g. Transition construction), matching this file's existing pattern
elsewhere.

Spec refs: REQ-ARC-WMTE-4491, REQ-ARC-WMTE-4494 (E3AgentPolicy's world-model
transition collection, the code path this fix hardens).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical


def test_detect_cell_returns_safe_fallback_for_non_2d_grid() -> None:
    degenerate = np.zeros((0,), dtype=np.int16)
    assert detect_cell(degenerate) == 1


def test_to_logical_returns_input_unchanged_for_non_2d_grid() -> None:
    degenerate = np.zeros((0,), dtype=np.int16)
    result = to_logical(degenerate, 8)
    assert result.shape == degenerate.shape


def test_detect_cell_and_to_logical_unaffected_on_normal_grids() -> None:
    """The defensive guard must not change behavior for real, well-formed grids."""

    grid = np.zeros((64, 64), dtype=np.int16)
    grid[::8, ::8] = 1  # a pattern that repeats every 8 cells

    cell = detect_cell(grid)
    assert cell in (1, 2, 4, 8)
    logical = to_logical(grid, cell)
    assert logical.shape == (64 // cell, 64 // cell)


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


def _isolated_proposer():
    """A LocalGGUFProposer that FAST-FAILS its generate() without any network I/O -- so next_move's
    stall-refactor loop (execute_bounded_llm_reinduction -> refactor -> generate) is still EXERCISED but
    returns in ~0.01s instead of making a real ~600s HTTP call to the live generator. Rationale (2026-07-24):
    `proposer=None` is NOT "no proposer" -- E3AgentPolicy._proposer() lazily builds the LIVE default generator
    (Qwen3.5-9B, timeout=600, intentional for the eval which has no internal time limit), so these robustness
    tests were inadvertently making a real HTTP call that waited the full 600s and blew the pytest 120s limit
    (SIGTERM). Pointing at an unresolvable model + a dead port makes `_ensure_server()` return False
    immediately (no server launch, no socket read), so generate() returns (False, ...) at once and next_move
    takes its graceful fallthrough -- preserving the degenerate-frame robustness this file verifies."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(repo_substr="__unit_test_no_model__", model_path=None, port=1, timeout=1)


def test_next_move_survives_a_degenerate_frame_after_a_real_one() -> None:
    """A degenerate frame immediately following a real one must not crash
    next_move -- the exact sequence exp5587 hit (a normal frame sets self._prev,
    then a subsequent degenerate frame makes to_logical's grid.shape unpack
    fail)."""

    policy = E3AgentPolicy("cd82", proposer=_isolated_proposer(), value_head=lambda _f: 0.0)

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

    policy = E3AgentPolicy("cd82", proposer=_isolated_proposer(), value_head=lambda _f: 0.0)

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


def test_next_move_survives_a_degenerate_frame_on_the_tier1_explore_tick() -> None:
    """Crash site #2: the tier-1 explore-phase per-action tick (not the tier-3
    transition-collection block) also called detect_cell/to_logical unguarded
    on grid_of(latest). A sequence of several real frames (enough that the
    policy stays in phase == "explore", not escalating to induction) followed
    by a degenerate one must not crash."""

    policy = E3AgentPolicy("cd82", proposer=_isolated_proposer(), value_head=lambda _f: 0.0)

    frames: list = []
    latest = None
    for i in range(3):
        frame = _NormalFrame(value=(i % 3) + 1)
        kind, data = policy.next_move(frames, latest)
        frames.append(frame)
        latest = frame

    degenerate = _DegenerateFrame()
    # Must not raise -- prior to the source-level fix this raised ValueError
    # from detect_cell's `h, w = grid.shape` at the explore-tick call site.
    kind, data = policy.next_move(frames, degenerate)
    assert kind is None or isinstance(kind, (int, str))
