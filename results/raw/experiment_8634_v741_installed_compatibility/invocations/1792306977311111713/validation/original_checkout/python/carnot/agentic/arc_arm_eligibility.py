"""Report whether the explorer's diversity draw is IN EFFECT, not merely enabled.

Spec: REQ-ARC-WMTE-7040, as corrected 2026-09-05.

WHAT THIS REPLACED, AND WHY. The first version of this module reset
`explorer._hybrid_diversity` on every level-up, to make the force-diversity arm eligible again.
That was the wrong repair and it introduced a regression: the explorer restores its own randomised
draw once `_steps_since_progress` passes `_stall_threshold`, and clearing the flag removed that
self-restoring behaviour.

The real defect is narrower. The randomised draw runs only when BOTH
`_hybrid_diversity` is set AND `_steps_since_progress > _stall_threshold`
(`arc_competition_agent.py`, the draw's own guard). `_steps_since_progress` resets to 0 whenever
the agent reaches a new best level. The stall threshold is 150 and the supervisor window is 120,
so at the first window on a new level the draw is genuinely NOT running and the arm's diagnosis
is correct.

The snapshot reported `diversity_active` from `_hybrid_diversity` alone, which means ENABLED. The
arm's precondition needs IN EFFECT. So the arm was suppressed on every level after the first by a
field that answered a different question. The instrument was wrong, not the search.

Scale, stated honestly so nobody oversells it: the behavioural gain is roughly thirty actions of
earlier randomised drawing per level, and only in applied mode, because the draw self-restores at
the threshold anyway. The reason to fix it is instrument correctness -- it is why every deep level
in the refinement ledger is missing exactly this arm, which corrupted the exhaustion signal.

Kept in its own module because importing `arc_competition_agent` pulls in roughly half a gigabyte
of the carnot stack and the suite's per-test memory watchdog refuses that teardown. A test that
cannot import the real function ends up testing a mirror, and a mirror passes every mutation of
the code it mirrors -- which happened here once already.
"""

from __future__ import annotations

from typing import Any


def diversity_in_effect(explorer: Any) -> bool:
    """True when the randomised diversity draw is actually running right now.

    Mirrors the draw's own guard: the feature must be enabled AND the search must have stalled
    past the threshold. Reporting only the first is what made the force-diversity arm look spent
    on every level after the first.
    """

    if explorer is None:
        return False
    if not getattr(explorer, "_hybrid_diversity", False):
        return False
    steps = getattr(explorer, "_steps_since_progress", 0)
    threshold = getattr(explorer, "_stall_threshold", 150)
    try:
        return int(steps) > int(threshold)
    except (TypeError, ValueError):
        # An explorer with unreadable counters is reported as NOT in effect: the arm then
        # remains eligible, which is the safe direction for a diagnostic.
        return False
