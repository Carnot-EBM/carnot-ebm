"""Spec: REQ-ARC-WMTE-7040, SCENARIO-ARC-WMTE-7040-A

The force-diversity arm becomes eligible again on each new level.

INCIDENT 2026-09-05, found by an adversarial reviewer and confirmed in source. The trajectory
supervisor clears `_arms_used` on every level-up, so its arm table believes every arm is
available again. The force-diversity arm is guarded by `not diversity_active`, which reads
`explorer._hybrid_diversity`. That flag was set at init from the environment and set True when
the arm fired, and NO line anywhere set it back.

So the arm fired once per RUN inside a table designed to reset per LEVEL, and every level after
the first ran one rung short. Measured across the five exhaustion cells in the refinement ledger:
every deep level was missing exactly this arm, and one was missing two.

This also explains a downstream confusion. Because deep levels could never fire the full ladder,
they never satisfied "every enabled arm fired", so the exhaustion trigger was rescued by level 0
alone and the pooled form hid the fact.

The reset restores the OPERATOR BASELINE rather than False. Resetting to False would silently
switch off a run started with CARNOT_ARC_EXPLORE_DIVERSITY=1, which is a worse bug than the one
being fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


from carnot.agentic.arc_arm_eligibility import restore_arm_eligibility as _reset  # noqa: E402


class _Explorer:
    """Minimal stand-in carrying only the two attributes the reset touches."""

    def __init__(self, baseline: bool) -> None:
        self._hybrid_diversity = baseline
        self._hybrid_diversity_baseline = baseline
        self.goal_bias = None


def test_an_arm_fired_on_level_0_is_eligible_again_on_level_1() -> None:
    """The incident: the arm fired once per run and every later level ran a rung short."""
    ex = _Explorer(baseline=False)
    ex._hybrid_diversity = True  # the arm fired on level 0
    _reset(ex, 1, 0)
    assert ex._hybrid_diversity is False, "arm still blocked on the new level"


def test_an_operator_configured_run_keeps_its_diversity_across_a_level_up() -> None:
    """Resetting to False instead of the baseline would switch off an operator's own setting."""
    ex = _Explorer(baseline=True)
    _reset(ex, 1, 0)
    assert ex._hybrid_diversity is True


def test_no_reset_happens_without_a_level_change() -> None:
    """The reset is keyed on ADVANCE. Firing it every tick would undo the arm mid-level."""
    ex = _Explorer(baseline=False)
    ex._hybrid_diversity = True
    _reset(ex, 0, 0)
    assert ex._hybrid_diversity is True


def test_the_first_observation_counts_as_a_change() -> None:
    """`None` means no level seen yet, which is not level 0."""
    ex = _Explorer(baseline=False)
    ex._hybrid_diversity = True
    assert _reset(ex, 0, None) == 0
    assert ex._hybrid_diversity is False


def test_an_explorer_without_the_baseline_is_left_alone() -> None:
    """Defensive: never invent a baseline for an explorer that does not carry one."""

    class _Old:
        _hybrid_diversity = True

    old = _Old()
    _reset(old, 1, 0)
    assert old._hybrid_diversity is True


def test_the_real_explorer_records_a_baseline_at_init() -> None:
    """The rule above is worthless if the real class never sets the attribute."""
    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    assert "self._hybrid_diversity_baseline = self._hybrid_diversity" in src


def test_the_agent_resets_before_building_the_snapshot() -> None:
    """A reset the supervisor cannot see this tick is a reset it acts on one window late."""
    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    reset_at = src.index("restore_arm_eligibility(")
    snapshot_at = src.index("diversity_active=bool(getattr(explorer")
    assert reset_at < snapshot_at, "reset must precede the snapshot that reads the flag"


def test_the_agent_stores_the_returned_level() -> None:
    """Discarding the return value silently turns the reset into an every-tick reset.

    `restore_arm_eligibility` reports the level it is now tracking. If the agent throws that
    away, `_last_supervised_level` stays None, `level != last_level` is always true, and the arm
    is undone on every observation instead of once per level. A mutation that assigned the result
    to a throwaway survived the rest of this file, which is what this test exists to stop.
    """
    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    assert "self._last_supervised_level = restore_arm_eligibility(" in src
