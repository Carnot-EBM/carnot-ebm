"""Restore a supervisor arm's eligibility when the agent reaches a new level.

Spec: REQ-ARC-WMTE-7040.

WHY THIS IS ITS OWN MODULE. It is three lines of rule, and it lives apart from
`arc_competition_agent` for one reason: importing that module pulls in the whole carnot stack,
about half a gigabyte, and the suite's per-test memory watchdog correctly refuses a teardown that
grows by that much. A test that cannot import the real function ends up testing a hand-written
mirror of it, and a mirror passes every mutation of the code it mirrors. That happened once on
this very fix, so the extraction is the repair for a decorative test rather than a matter of
taste.
"""

from __future__ import annotations

from typing import Any


def restore_arm_eligibility(explorer: Any, level: int, last_level: int | None) -> int | None:
    """Give the new level its arms back. Returns the level now being tracked.

    The trajectory supervisor clears `_arms_used` on every level-up, so its table believes every
    arm is available again. The force-diversity arm is guarded by `not diversity_active`, which
    reads `explorer._hybrid_diversity` -- and nothing ever set that back. So the arm fired once
    per RUN inside a table designed to reset per LEVEL, and every level after the first ran a
    rung short.

    Restores the OPERATOR BASELINE, never a bare False. A run started with
    `CARNOT_ARC_EXPLORE_DIVERSITY=1` asked for diversity throughout, and switching that off at a
    level-up would be a worse bug than the one this fixes.

    Keyed on a level CHANGE, not on every tick: re-running it mid-level would undo an arm the
    supervisor had just fired. `last_level` of None means no level has been observed yet, which
    is not the same as level 0.
    """

    if explorer is None or level == last_level:
        return last_level
    baseline = getattr(explorer, "_hybrid_diversity_baseline", None)
    if baseline is not None:
        explorer._hybrid_diversity = bool(baseline)
    return level
