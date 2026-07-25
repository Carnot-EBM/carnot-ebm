"""Runtime click-vocabulary gate on the just-explore tier barrier.

REQ-ARC-WMTE-5836 / SCENARIO: tier-barrier-confined-to-click-vocabulary-games

WHY THIS EXISTS (2026-07-25). The full-spec frontier-discipline A/B
(results/experiment_5836_frontier_discipline_generalization.json) found a real capability gain that
was robust to colour permutation AND reflection (+2..+4 games in every condition), but NO arm was
regression-free once measured PER SEED rather than on an any-seed union. Arm B2's only loss was
tu93 -- the single nav-only game among the baseline's wins -- lost on 2 of 3 seeds in EVERY
condition, while all five of its gains were click games.

That asymmetry is mechanical, not luck: the 5-tier predicate ranks CLICK-TARGET salience
(button-like vs status-bar vs large-flat blobs). A nav-only game has no click targets to rank, so
the barrier cannot express anything useful there -- it can only perturb the move ordering of a
search that was already succeeding. Confining the barrier to games whose action vocabulary contains
click is therefore the mechanism's correct DOMAIN OF DEFINITION, not a carve-out to rescue a number.

HIDDEN-GAME LEGALITY is the load-bearing property these tests protect. The signal must come from
``frame.available_actions`` -- which the env reports at runtime for a game never seen before -- and
NOT from the harness's hardcoded CLICK_GAMES list, which would be per-game knowledge and illegal on
the hidden set. The parse reuses the live adapter's ``_available_action_ids`` so enum / "ACTION6" /
bare-int frames all work.
"""

from __future__ import annotations

from typing import Any

from carnot.agentic.arc_competition_agent import (
    SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED,
    StepwiseExplorer,
)


class _Frame:
    """Minimal stand-in for the env frame: only available_actions is read."""

    def __init__(self, actions: Any) -> None:
        self.available_actions = actions


def _explorer(*, barrier: bool, click_only: bool) -> StepwiseExplorer:
    """A bare explorer carrying only the tier-gate state (no env, no game, no LLM)."""
    exp = StepwiseExplorer.__new__(StepwiseExplorer)
    exp.tier_exhaustion_enabled = barrier
    exp.tier_click_vocab_only = click_only
    exp._fd_click_vocab_seen = False
    return exp


def test_default_is_click_vocab_only_enabled():
    """The fix ships ON: it only bites where the barrier is already enabled, and there it helped."""
    assert SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED is True


def test_barrier_inert_on_nav_only_game():
    """THE REGRESSION THIS FIXES: a nav-only vocabulary must leave the barrier inert.

    This is the tu93 case -- the game arm B2 lost on 2 of 3 seeds in every condition.
    """
    exp = _explorer(barrier=True, click_only=True)
    assert exp._tier_active(_Frame([1, 2, 3, 4])) is False


def test_barrier_active_on_click_game():
    exp = _explorer(barrier=True, click_only=True)
    assert exp._tier_active(_Frame([1, 2, 3, 4, 6])) is True


def test_click_vocabulary_latches_and_is_sticky():
    """Once a click is offered the game IS a click game, even on later click-free frames."""
    exp = _explorer(barrier=True, click_only=True)
    assert exp._tier_active(_Frame([1, 2, 3, 4, 6])) is True
    assert exp._tier_active() is True  # no frame in scope (the _pop/_draw call sites)
    assert exp._tier_active(_Frame([1, 2, 3, 4])) is True  # sticky


def test_inactive_before_any_frame_is_observed():
    """Fails OPEN toward today's behaviour: no observation yet -> barrier does not engage."""
    exp = _explorer(barrier=True, click_only=True)
    assert exp._tier_active() is False


def test_gate_disabled_reproduces_pre_fix_behaviour():
    """The escape hatch must genuinely restore the measured-harmful behaviour, for A/B attribution."""
    exp = _explorer(barrier=True, click_only=False)
    assert exp._tier_active(_Frame([1, 2, 3, 4])) is True
    assert exp._tier_active() is True


def test_barrier_off_dominates_the_click_gate():
    """With the barrier off, click availability is irrelevant -- baseline parity."""
    exp = _explorer(barrier=False, click_only=True)
    assert exp._tier_active(_Frame([1, 2, 3, 4, 6])) is False
    assert exp._tier_active() is False


def test_action_id_forms_are_all_parsed():
    """Enum-like, "ACTION6" string, and bare-int frames must all register as click games."""

    class _Enum:
        def __init__(self, value: int) -> None:
            self.value = value

    for actions in ([6], ["ACTION6"], [_Enum(6)], [1, "ACTION6", 4]):
        exp = _explorer(barrier=True, click_only=True)
        assert exp._tier_active(_Frame(actions)) is True, actions


def test_unparseable_frame_does_not_latch_or_crash():
    """A malformed frame must neither engage the barrier nor raise into the search loop."""
    exp = _explorer(barrier=True, click_only=True)
    assert exp._tier_active(_Frame(object())) is False
    assert exp._fd_click_vocab_seen is False
    assert exp._tier_active(_Frame(None)) is False


def test_nav_only_game_reports_no_node_as_tier_deferred():
    """Consistency of the whole state machine, not just the entry gate.

    `_node_is_tier_deferred` feeds NEGATIVE samples to the online discriminative learner. If the
    barrier is inert but this still reported nodes as deferred, the barrier would poison a
    component it is meant not to touch.
    """
    exp = _explorer(barrier=True, click_only=True)
    exp._active_tier = 0
    exp._tier_active(_Frame([1, 2, 3, 4]))  # nav-only: barrier stays inert
    node = {"untested": [{"action": 1}, {"action": 2}]}
    assert exp._node_has_open_tier(node) is True
    assert exp._node_is_tier_deferred(node) is False
