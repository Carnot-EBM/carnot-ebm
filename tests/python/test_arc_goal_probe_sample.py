"""REQ-ARC-WMTE-6530: the goal-defect probe must sample the TAIL of a window, not only the head.

WHY THIS FILE EXISTS. Windows are built to END at their level flip, so the win frame is always
the last grid. The probe took `grids[:12]` -- the first six transitions, since grids are appended
(grid, next_grid) per transition -- so on any window longer than six transitions the win was
invisible to it. A predicate that is correct (true only on win frames) then reads constant-False
across everything the probe sees and is rejected, while a predicate that fires early on a non-win
frame reads non-constant and is kept. The gate rejected correct predicates and rewarded false
positives, on exactly the long windows where induction is hardest.

Measured over `window_meta.json` at the time of the fix: 7 of 13 games exceed six transitions.

Scenarios: SCENARIO-ARC-WMTE-6530-TAIL (the last grid is always probed),
SCENARIO-ARC-WMTE-6530-BUDGET (the cap that bounds the hang exposure is preserved),
SCENARIO-ARC-WMTE-6530-SHORT (windows at or under the cap are probed exactly as before).
"""

from __future__ import annotations

import pytest

from carnot.agentic import arc_executable_world_model as wm


# SCENARIO-ARC-WMTE-6530-SHORT
@pytest.mark.parametrize("n", [0, 1, 5, 11, 12])
def test_windows_within_the_cap_are_unchanged(n: int) -> None:
    """At or under the cap the sample is the whole list, so short windows behave exactly as they
    did before the fix. This is what makes the change safe to ship: the games the gate already
    handled correctly see no difference at all."""
    grids = list(range(n))
    assert wm._goal_probe_sample(grids) == grids


# SCENARIO-ARC-WMTE-6530-TAIL
@pytest.mark.parametrize("n", [13, 18, 24, 25, 66])
def test_the_last_grid_is_always_sampled(n: int) -> None:
    """The win frame is the last grid by construction, so a probe that can miss it is the bug.
    24 is the real case: a window at the builder's `WINDOW_K = 12` transition cap is 24 grids,
    exactly twice the probe's 12-grid reach, and the old head-only slice could never reach the
    win. 66 is the 33-transition window the caller's docstring names."""
    grids = list(range(n))
    sample = wm._goal_probe_sample(grids)
    assert grids[-1] in sample, "the win frame must be probed at every window length"
    assert grids[0] in sample, "early-frame coverage from the original slice must be kept"


# SCENARIO-ARC-WMTE-6530-BUDGET
@pytest.mark.parametrize("n", [13, 24, 66, 500])
def test_the_probe_budget_is_never_exceeded(n: int) -> None:
    """The cap bounds the hang exposure described in `_goal_defects`' docstring -- a predicate
    with an unbounded loop is executed once per sampled grid. Widening coverage must not widen
    cost, or the fix trades one hazard for another."""
    sample = wm._goal_probe_sample(list(range(n)))
    assert len(sample) <= wm._GOAL_PROBE_MAX_GRIDS


def test_sample_preserves_order_and_does_not_duplicate() -> None:
    """Head and tail must not overlap into a repeated grid, which would silently spend budget
    probing the same frame twice and reduce real coverage below the cap."""
    grids = list(range(24))
    sample = wm._goal_probe_sample(grids)
    assert len(set(sample)) == len(sample), "no grid probed twice"
    assert sample == sorted(sample), "original frame order preserved"
    assert sample == list(range(6)) + list(range(18, 24))


def test_the_regression_this_fixes_in_its_original_shape() -> None:
    """The incident, reproduced. A 12-transition window is 24 grids with the win at index 23.
    The old expression could not see it; the new one must. Written as the two expressions side
    by side so the test states the defect rather than only the fix."""
    grids = list(range(24))
    win_frame = grids[-1]
    old_slice = grids[: wm._GOAL_PROBE_MAX_GRIDS]
    assert win_frame not in old_slice, "precondition: the old slice missed the win"
    assert win_frame in wm._goal_probe_sample(grids)
