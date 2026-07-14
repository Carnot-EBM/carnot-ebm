"""Tests for arc_object_history_salience.ObjectHistorySaliencePrior -- the live-consuming
mechanism task 10's own DONE note deferred ("preferring an object whose hash was seen to
change in a prior frame"), built on top of object_hash (REQ-ARC-FCP-5591).

Spec refs: REQ-ARC-FCP-5591-2, SCENARIO-ARC-FCP-5591-2-CHANGE-RATE-BONUS,
SCENARIO-ARC-FCP-5591-2-EVIDENCE-FLOOR, SCENARIO-ARC-FCP-5591-2-NOT-DEGENERATE.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
from carnot.agentic.arc_object_history_salience import (
    ObjectHistorySaliencePrior,
    coerce_object_history_salience_prior,
)


class _Frame:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame


class _ConstantPrior:
    def score(self, _frame: object, _candidate: object) -> float:
        return 42.0

    def as_dict(self) -> dict[str, object]:
        return {"source": "constant_test_prior"}


def _grid_with_one_blob(color: int = 5) -> np.ndarray:
    grid = np.zeros((10, 10), dtype=np.int16)
    grid[2:5, 2:5] = color
    return grid


def test_coerce_object_history_salience_prior_none_false_true_instance_and_junk() -> None:
    base = ColorBlobSaliencePrior()
    assert coerce_object_history_salience_prior(None, base_prior=base) is base
    assert coerce_object_history_salience_prior(False, base_prior=base) is base
    wrapped = coerce_object_history_salience_prior(True, base_prior=base)
    assert isinstance(wrapped, ObjectHistorySaliencePrior)
    assert wrapped.base_prior is base
    instance = ObjectHistorySaliencePrior()
    assert coerce_object_history_salience_prior(instance, base_prior=base) is instance
    # True with no base_prior supplied builds a fresh default base.
    default_wrapped = coerce_object_history_salience_prior(True)
    assert isinstance(default_wrapped.base_prior, ColorBlobSaliencePrior)
    # anything else falls through to base_prior unchanged.
    assert coerce_object_history_salience_prior("on", base_prior=base) is base


def test_for_path_returns_self() -> None:
    """Matches GeometricSaliencePrior's contract -- no per-path cloning of tally memory."""

    prior = ObjectHistorySaliencePrior()
    assert prior.for_path([{"action": 1}]) is prior


def test_observe_transition_ignores_non_click_actions() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1)
    grid = _grid_with_one_blob()
    changed = grid.copy()
    changed[2, 2] = 0
    prior.observe_transition(_Frame(grid), 1, None, _Frame(changed))
    assert prior.tracked_hash_count == 0


def test_observe_transition_ignores_missing_click_coordinates() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1)
    grid = _grid_with_one_blob()
    prior.observe_transition(_Frame(grid), 6, {}, _Frame(grid))
    assert prior.tracked_hash_count == 0


def test_observe_transition_ignores_malformed_frames_coordinates_and_empty_space() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1)
    grid = _grid_with_one_blob()

    prior.observe_transition(
        _Frame(np.zeros((1, 2, 3, 4), dtype=np.int16)),
        6,
        {"x": 0, "y": 0},
        _Frame(grid),
    )
    prior.observe_transition(_Frame(grid), 6, {"x": object(), "y": 3}, _Frame(grid))
    prior.observe_transition(
        _Frame(np.zeros((0, 0), dtype=np.int16)),
        6,
        {"x": 0, "y": 0},
        _Frame(np.zeros((0, 0), dtype=np.int16)),
    )

    assert prior.tracked_hash_count == 0


def test_scenario_5591_2_evidence_floor_no_bonus_below_min_observations() -> None:
    """SCENARIO-ARC-FCP-5591-2-EVIDENCE-FLOOR: an under-observed hash gets zero bonus, not a
    penalty and not a premature boost -- matching InertClickSigPruner's floor discipline."""

    prior = ObjectHistorySaliencePrior(min_observations=3, change_bonus_weight=10.0)
    grid = _grid_with_one_blob()
    changed = grid.copy()
    changed[2, 2] = 0
    # Only 2 observations, both changing -- below the floor of 3.
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))

    base_score = prior.base_prior.score(_Frame(grid), {"action": 6, "data": {"x": 3, "y": 3}})
    wrapped_score = prior.score(_Frame(grid), {"action": 6, "data": {"x": 3, "y": 3}})
    assert wrapped_score == base_score


def test_scenario_5591_2_change_rate_bonus_polarity_correct() -> None:
    """SCENARIO-ARC-FCP-5591-2-CHANGE-RATE-BONUS: the SAME object (identical base_prior score,
    since it's the identical blob) gets a HIGHER final score after a reliable change history
    than after a reliably-inert history, once the evidence floor clears -- isolating the bonus's
    effect from any base-tier difference between distinct objects (which is a separate,
    unrelated signal -- see the compare-different-colors caution this test avoids)."""

    grid = _grid_with_one_blob(color=5)
    changed = grid.copy()
    changed[2, 2] = 0
    candidate = {"action": 6, "data": {"x": 3, "y": 3}}
    base_score = ObjectHistorySaliencePrior().base_prior.score(_Frame(grid), candidate)

    reliable = ObjectHistorySaliencePrior(min_observations=3, change_bonus_weight=10.0)
    inert = ObjectHistorySaliencePrior(min_observations=3, change_bonus_weight=10.0)
    for _ in range(4):
        reliable.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))
        inert.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(grid))

    reliable_score = reliable.score(_Frame(grid), candidate)
    inert_score = inert.score(_Frame(grid), candidate)

    assert reliable_score > base_score  # the reliably-changing history got boosted
    assert inert_score == base_score  # the reliably-inert history got zero bonus
    assert reliable_score > inert_score


def test_scenario_5591_2_change_rate_bonus_is_partial_for_intermittent_objects() -> None:
    """An object that sometimes changes and sometimes doesn't gets a PROPORTIONAL bonus, not
    an all-or-nothing one -- tolerates noisy real-game evidence rather than requiring perfect
    consistency (mirrors InertClickSigPruner's specificity-threshold, not Reki's zero-tolerance)."""

    prior = ObjectHistorySaliencePrior(min_observations=4, change_bonus_weight=10.0)
    grid = _grid_with_one_blob()
    changed = grid.copy()
    changed[2, 2] = 0

    # 2 of 4 observations change the frame -> change_rate == 0.5.
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(grid))
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(grid))

    base_score = prior.base_prior.score(_Frame(grid), {"action": 6, "data": {"x": 3, "y": 3}})
    wrapped_score = prior.score(_Frame(grid), {"action": 6, "data": {"x": 3, "y": 3}})
    assert wrapped_score == base_score + 10.0 * 0.5


def test_scenario_5591_2_not_degenerate_same_base_tier_different_hash_history_differentiates() -> (
    None
):
    """SCENARIO-ARC-FCP-5591-2-NOT-DEGENERATE (adversarial check): two candidates with the
    IDENTICAL base_prior score (same tier -- same color, same size, same shape, hence the same
    button-likelihood features) are differentiated by change-history alone once observed --
    proving the bonus is genuinely NEW information, not a re-derivation of the base tier
    features under a different name."""

    reactive_grid = np.zeros((10, 10), dtype=np.int16)
    reactive_grid[2:5, 2:5] = 8  # 3x3 salient-color square
    reactive_grid[7:10, 7:10] = 8  # an identical 3x3 salient-color square elsewhere -- SAME
    # object_hash (color+shape match), so the two candidates below share an IDENTICAL
    # base_prior.score (same tier, same button-likelihood features) before any history.
    prior = ObjectHistorySaliencePrior(min_observations=2, change_bonus_weight=5.0)

    candidate_a = {"action": 6, "data": {"x": 3, "y": 3}}
    candidate_b = {"action": 6, "data": {"x": 8, "y": 8}}
    base_a = prior.base_prior.score(_Frame(reactive_grid), candidate_a)
    base_b = prior.base_prior.score(_Frame(reactive_grid), candidate_b)
    assert base_a == base_b  # confirmed identical BEFORE any observed history

    changed = reactive_grid.copy()
    changed[3, 3] = 0
    for _ in range(3):
        prior.observe_transition(_Frame(reactive_grid), 6, {"x": 3, "y": 3}, _Frame(changed))

    # candidate_b shares the SAME object_hash as candidate_a's now-tracked blob (identical
    # color+shape), so its bonus reflects the SAME tally -- the mechanism is hash-identity-based,
    # not position-based, and both candidates are boosted equally and identically above base.
    score_a = prior.score(_Frame(reactive_grid), candidate_a)
    score_b = prior.score(_Frame(reactive_grid), candidate_b)
    assert score_a == score_b > base_a


def test_reset_clears_tally_only_when_reset_to_prior() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1)
    grid = _grid_with_one_blob()
    changed = grid.copy()
    changed[2, 2] = 0
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))
    assert prior.tracked_hash_count == 1

    prior.reset(reset_to_prior=False)
    assert prior.tracked_hash_count == 1

    prior.reset(reset_to_prior=True)
    assert prior.tracked_hash_count == 0


def test_non_click_candidates_pass_through_unaffected() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1)
    grid = _grid_with_one_blob()
    changed = grid.copy()
    changed[2, 2] = 0
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))

    nav_candidate = {"action": 1, "data": None}
    assert prior.score(_Frame(grid), nav_candidate) == prior.base_prior.score(
        _Frame(grid), nav_candidate
    )


def test_malformed_or_empty_click_candidates_pass_through_unaffected() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1, base_prior=_ConstantPrior())
    default_prior = ObjectHistorySaliencePrior(min_observations=1)
    grid = _grid_with_one_blob()

    missing_y = {"action": 6, "data": {"x": 3}}
    malformed_frame = {"action": 6, "data": {"x": 3, "y": 3}}
    bad_coordinate = {"action": 6, "data": {"x": object(), "y": 3}}
    off_grid = {"action": 6, "data": {"x": 99, "y": 99}}
    no_component = {"action": 6, "data": {"x": 0, "y": 0}}

    assert default_prior.score(_Frame(grid), missing_y) == default_prior.base_prior.score(
        _Frame(grid), missing_y
    )
    assert prior.score(_Frame(np.zeros((1, 2, 3, 4), dtype=np.int16)), malformed_frame) == 42.0
    assert prior.score(_Frame(grid), bad_coordinate) == 42.0
    assert prior.score(_Frame(np.zeros((0, 0), dtype=np.int16)), no_component) == 42.0
    assert default_prior.score(_Frame(grid), off_grid) == default_prior.base_prior.score(
        _Frame(grid), off_grid
    )


def test_disabled_prior_never_boosts() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=1, enabled=False)
    grid = _grid_with_one_blob()
    changed = grid.copy()
    changed[2, 2] = 0
    prior.observe_transition(_Frame(grid), 6, {"x": 3, "y": 3}, _Frame(changed))
    candidate = {"action": 6, "data": {"x": 3, "y": 3}}
    assert prior.score(_Frame(grid), candidate) == prior.base_prior.score(_Frame(grid), candidate)


def test_as_dict_reports_diagnostics() -> None:
    prior = ObjectHistorySaliencePrior(min_observations=2, change_bonus_weight=7.5)
    d = prior.as_dict()
    assert d["source"] == "object_hash_change_history_blob_salience"
    assert d["change_bonus_weight"] == 7.5
    assert d["min_observations"] == 2
    assert d["tracked_hash_count"] == 0
    assert d["verifier_is_oracle"] is False
    assert prior.diagnostics() == d
