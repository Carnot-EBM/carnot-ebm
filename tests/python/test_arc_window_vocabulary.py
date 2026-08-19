"""GAP-6260 layer 1: an induction window whose actions the planner cannot emit.

WHY THIS FILE EXISTS. An induced engine is only useful if the planner can drive it, and the
planner's move set is `_model_candidates` -- actions 1-5 plus a click. A window whose observed
actions fall outside that set teaches the engine transitions the planner will never ask for, so
the engine can score a perfect `cell_recall` on its own window and still be unplannable.

lp85 is the measured case. Every action in its cached window is 0 (RESET), and its induced engine
produced ZERO novel successors from the planning root -- 37 nodes, exactly one candidate sweep of
no-ops, the win predicate never consulted. Six A/B cells across both arms were spent on it before
anyone noticed.

Scenarios: SCENARIO-GAP-6260-L1-DETECT (the lp85 shape is flagged),
SCENARIO-GAP-6260-L1-CLEAN (real playable windows are not),
SCENARIO-GAP-6260-L1-AUTHORITY (the constant tracks `_model_candidates`),
SCENARIO-GAP-6260-L1-NONBLOCKING (the check reports and does not abort).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as wm


class _T:
    """Minimal stand-in for Transition: the check reads `.action` and nothing else."""

    def __init__(self, action):
        self.action = action


# SCENARIO-GAP-6260-L1-DETECT
def test_the_lp85_shape_is_flagged() -> None:
    """The incident, in its measured form: a five-transition window that is all RESET."""
    assert wm.window_actions_outside_planner_vocabulary([_T(0)] * 5) == {0}


# SCENARIO-GAP-6260-L1-CLEAN
@pytest.mark.parametrize(
    "actions",
    [
        [2, 3],  # ar25, as cached
        [4, 5],  # sp80
        [1, 2, 4],  # tr87
        [1, 2, 3, 4],  # tu93
        [6],  # vc33 -- click-only, and the control that refuted "clicks are lost"
        [5, 6],  # sb26 -- mixed keyboard and click
    ],
)
def test_real_playable_windows_are_not_flagged(actions) -> None:
    """Every other cached window in the corpus. A check that fired on these would be worse than
    no check: it would send the agent re-collecting windows that are fine."""
    assert wm.window_actions_outside_planner_vocabulary([_T(a) for a in actions]) == set()


def test_mixed_window_reports_only_the_unusable_actions() -> None:
    """A window can be partly usable. Report exactly what the planner cannot emit, so a caller
    can tell 'unusable' from 'contains some noise'."""
    assert wm.window_actions_outside_planner_vocabulary([_T(0), _T(2), _T(7), _T(6)]) == {0, 7}


@pytest.mark.parametrize("bad", [None, [], [_T(None)], [_T("2")]])
def test_degenerate_input_is_not_an_error(bad) -> None:
    """This runs on the live induction path, so it must never be the reason induction fails.
    Absent, empty, or non-integer actions yield no finding rather than an exception."""
    assert wm.window_actions_outside_planner_vocabulary(bad) == set()


# SCENARIO-GAP-6260-L1-AUTHORITY
def test_the_constant_tracks_model_candidates() -> None:
    """`_model_candidates` is the authority; the constant mirrors its action set so the check can
    run WITHOUT a grid, which is what makes it free enough to run before induction.

    Asserting the agreement is the point: a silently drifted constant would make this check pass
    on windows the planner can no longer drive, which is the exact failure it exists to catch.
    Written as an assertion rather than a comment because tonight's lesson was that a design
    property stated only in prose is one nobody can see break.
    """
    grid = np.zeros((8, 8), dtype=int)
    grid[2:4, 2:4] = 3  # one component, so the click branch contributes
    emitted = {int(c["action"]) for c in wm._model_candidates(grid)}
    assert emitted, "the planner must offer some candidate on a non-empty grid"
    assert emitted <= wm._PLANNER_ACTION_VOCABULARY, (
        f"_model_candidates emits {sorted(emitted - wm._PLANNER_ACTION_VOCABULARY)}, which the "
        "vocabulary constant does not contain -- the constant has drifted from its authority"
    )


# SCENARIO-GAP-6260-L1-NONBLOCKING
def test_the_check_reports_and_does_not_abort(monkeypatch, capsys) -> None:
    """Refusing to induce is a live-path behaviour change and wants its own measurement first.
    So the wiring records the finding and warns, then induces anyway -- verified here at the
    function level, since a violation must produce a finding without raising."""
    found = wm.window_actions_outside_planner_vocabulary([_T(0)] * 3)
    assert found == {0}
    assert isinstance(found, set), "callers sort this; it must be a plain set"
