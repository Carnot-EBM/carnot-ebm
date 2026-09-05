"""Spec: REQ-ARC-WMTE-7040, SCENARIO-ARC-WMTE-7040-A

The snapshot reports whether the diversity draw is IN EFFECT, not merely enabled.

INCIDENT 2026-09-05, found by an adversarial reviewer. The force-diversity arm is guarded by
`not s.diversity_active`. That field was built from `explorer._hybrid_diversity`, which means
ENABLED. The arm's precondition needs IN EFFECT.

The randomised draw runs only when the feature is enabled AND `_steps_since_progress` has passed
`_stall_threshold`. The stall counter resets to 0 at every new best level, the threshold is 150,
and the supervisor window is 120 — so at the first window on a new level the draw is genuinely
not running and the arm's diagnosis is correct. Reporting "enabled" suppressed the arm on every
level after the first, which is why every deep level in the refinement ledger is missing exactly
this arm and why the exhaustion signal was corrupted.

A FIRST ATTEMPT AT THIS FIX WAS WRONG and is recorded rather than hidden. It reset
`_hybrid_diversity` on every level-up. That mutated real search state to repair a reporting
defect, and it removed the explorer's own self-restoring draw, which returns once the stall
threshold is passed. The reviewer caught it. Scale, stated so nobody oversells the repair: the
behavioural gain is about thirty actions of earlier drawing per level, in applied mode only. The
reason to fix it is instrument correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_arm_eligibility import diversity_in_effect  # noqa: E402


class _Explorer:
    def __init__(self, enabled: bool, steps: int, threshold: int = 150) -> None:
        self._hybrid_diversity = enabled
        self._steps_since_progress = steps
        self._stall_threshold = threshold


def test_enabled_but_freshly_reset_is_not_in_effect() -> None:
    """The incident. Right after a level-up the counter is 0 and the draw is not running."""
    assert diversity_in_effect(_Explorer(enabled=True, steps=0)) is False


def test_enabled_and_past_the_threshold_is_in_effect() -> None:
    assert diversity_in_effect(_Explorer(enabled=True, steps=151)) is True


def test_exactly_at_the_threshold_is_not_yet_in_effect() -> None:
    """The draw's own guard is strictly greater than. An off-by-one here re-creates the bug."""
    assert diversity_in_effect(_Explorer(enabled=True, steps=150)) is False


def test_disabled_is_never_in_effect_however_long_the_stall() -> None:
    assert diversity_in_effect(_Explorer(enabled=False, steps=10_000)) is False


def test_a_missing_explorer_is_not_in_effect() -> None:
    assert diversity_in_effect(None) is False


def test_unreadable_counters_report_not_in_effect() -> None:
    """Safe direction for a diagnostic: the arm stays eligible rather than being suppressed."""
    bad = _Explorer(enabled=True, steps=0)
    bad._steps_since_progress = "not a number"
    assert diversity_in_effect(bad) is False


def test_the_agent_builds_the_snapshot_from_the_in_effect_helper() -> None:
    """A correct helper nothing calls is decorative; this is the call site."""
    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    assert "diversity_active=diversity_in_effect(explorer)" in src


def test_the_agent_no_longer_mutates_explorer_state_for_this() -> None:
    """The reverted first attempt reset _hybrid_diversity on level-up and removed the
    explorer's self-restoring draw. Its return would be a regression, so it is asserted gone."""
    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    assert "_hybrid_diversity_baseline" not in src
    assert "restore_arm_eligibility" not in src
