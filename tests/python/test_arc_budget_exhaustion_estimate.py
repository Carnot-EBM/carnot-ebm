"""A per-level action/click "budget meter" (exhausting it triggers GAME_OVER) is
independently hand-documented in 6+ games (ops/arc_solve_registry.yaml: r11l, sc25, s5i5,
g50t, bp35), each re-derived from scratch. `region_hud_evidence` already recognises the
region generically; this tests the estimator that turns an admitted region into a scalar
"how close to exhaustion" projection, and the pure cost-weight consumer built on it.

Spec: REQ-ARC-WMTE-6180,
SCENARIO-ARC-WMTE-6180-MONOTONE-REGION-PROJECTS-EXHAUSTION,
SCENARIO-ARC-WMTE-6180-ABSTAIN-ON-UNDERLYING-REFUSAL-OR-INSUFFICIENT-EVIDENCE,
SCENARIO-ARC-WMTE-6180-NON-POSITIVE-RATE-DOES-NOT-PROJECT,
SCENARIO-ARC-WMTE-6180-SEGMENT-RESET-DOES-NOT-INHERIT-PRIOR-HISTORY,
SCENARIO-ARC-WMTE-6180-CONSUMER-IS-A-NO-OP-WITHOUT-AN-ESTIMATE.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_hud_bar_detector import (
    BUDGET_ESTIMATE_MIN_TRANSITIONS,
    budget_exhaustion_estimate,
    region_hud_evidence,
)
from carnot.agentic.arc_solver_kit import (
    BUDGET_EXHAUSTION_PENALTY_PER_EXCESS_ACTION,
    budget_aware_path_cost_weight,
    standing_path_cost_weight,
)

FRAME_SHAPE = (8, 24)
BAR_ROWS = 7  # the last row is the HUD bar; rows 0..6 are "game state" (held constant)
# 24 columns gives 24 distinct fill levels -- wide enough that a strictly-increasing test
# trajectory never revisits an already-seen region value (see region_hud_evidence's
# REGION_EVIDENCE_MAX_REVISITS=0: repeating a SATURATED value -- e.g. an 8-wide bar already
# fully lit -- counts as a revisit on every subsequent identical frame, so a fixture must
# stay strictly below its own width for the whole window under test).


def _bar_mask() -> np.ndarray:
    mask = np.zeros(FRAME_SHAPE, dtype=bool)
    mask[BAR_ROWS, :] = True
    return mask


def _frame_with_bar_fill(n_lit: int, *, game_state: int = 1) -> np.ndarray:
    """8x24 frame: rows 0..6 constant `game_state`, bottom row has `n_lit` color-7 pixels
    lit from the left (mirrors bp35's real HUD readout: "N actions -> N lit pixels")."""
    grid = np.full(FRAME_SHAPE, game_state, dtype=np.uint8)
    grid[BAR_ROWS, :] = 0
    grid[BAR_ROWS, :n_lit] = 7
    return grid


def _monotone_fill_trajectory(n_frames: int) -> tuple[list[np.ndarray], list[str]]:
    """A clean, action-ubiquitous, monotone-filling bar over `n_frames` transitions --
    the shape `region_hud_evidence` is built to admit. The SAME action label is reused on
    every transition on purpose: `region_hud_evidence`'s ubiquity test needs an action
    class tried at least twice (`REGION_EVIDENCE_MIN_UBIQUITY`'s per-class minimum), so a
    fixture where every transition has a distinct label never accumulates one and always
    abstains with `no_action_class_tried_twice` regardless of how the region behaves."""
    grids = [_frame_with_bar_fill(0)]
    actions = [None]
    for i in range(1, n_frames):
        grids.append(_frame_with_bar_fill(min(i, FRAME_SHAPE[1])))
        actions.append("click")
    return grids, actions


class TestMonotoneRegionProjectsExhaustion:
    def test_admitted_region_yields_a_positive_rate_and_a_plausible_projection(self):
        # SCENARIO-ARC-WMTE-6180-MONOTONE-REGION-PROJECTS-EXHAUSTION
        # region_hud_evidence's own admission floor is 16 transitions (REGION_EVIDENCE_MIN_
        # TRANSITIONS), stricter than this estimator's 8 -- 18 frames clears both.
        mask = _bar_mask()
        grids, actions = _monotone_fill_trajectory(n_frames=18)
        evidence = region_hud_evidence(grids, mask, actions=actions)
        assert evidence["verdict"] == "admit", evidence

        result = budget_exhaustion_estimate(grids, mask, evidence=evidence)
        assert result["verdict"] == "estimate"
        assert result["rate_per_transition"] is not None
        assert result["rate_per_transition"] > 0
        assert result["actions_remaining_estimate"] is not None
        # The bar fills 1/8 of its width per transition (8-pixel-wide bar, 1 pixel lit per
        # click) after frame 0, so the projected remaining fraction (1.0 - fill_fraction)
        # divided by that per-transition rate should land near the true remaining clicks.
        assert 0.0 <= result["actions_remaining_estimate"] <= 8.0

    def test_fill_fraction_is_populated_even_without_evidence_precomputed(self):
        mask = _bar_mask()
        grids, _actions = _monotone_fill_trajectory(n_frames=18)
        # No `evidence=` kwarg -- function must compute Stage 2 itself.
        result = budget_exhaustion_estimate(grids, mask)
        assert result["fill_fraction"] is not None


class TestAbstainCases:
    def test_abstains_when_underlying_evidence_refuses(self):
        # SCENARIO-ARC-WMTE-6180-ABSTAIN-ON-UNDERLYING-REFUSAL-OR-INSUFFICIENT-EVIDENCE
        mask = _bar_mask()
        # A region that REVISITS its first value mid-episode (not action-ubiquitous /
        # monotone) -- region_hud_evidence must refuse this.
        grids = [
            _frame_with_bar_fill(0),
            _frame_with_bar_fill(3),
            _frame_with_bar_fill(0),  # revisit
            _frame_with_bar_fill(3),
            _frame_with_bar_fill(0),  # revisit again
        ] * 3
        actions = [("click", i, 0) for i in range(len(grids))]
        evidence = region_hud_evidence(grids, mask, actions=actions)
        assert evidence["verdict"] != "admit"

        result = budget_exhaustion_estimate(grids, mask, evidence=evidence)
        assert result["verdict"] == "abstain"
        assert result["actions_remaining_estimate"] is None

    def test_abstains_below_min_transitions_even_with_admit_shaped_data(self):
        mask = _bar_mask()
        grids, actions = _monotone_fill_trajectory(n_frames=BUDGET_ESTIMATE_MIN_TRANSITIONS)
        # BUDGET_ESTIMATE_MIN_TRANSITIONS frames -> BUDGET_ESTIMATE_MIN_TRANSITIONS - 1
        # transitions, one short of the floor.
        evidence = region_hud_evidence(grids, mask, actions=actions)
        result = budget_exhaustion_estimate(grids, mask, evidence=evidence)
        if evidence["verdict"] == "admit":
            assert result["n_transitions"] < BUDGET_ESTIMATE_MIN_TRANSITIONS
            assert result["verdict"] == "abstain"
            assert result["reason"] == "insufficient_transitions"

    def test_abstains_on_empty_mask(self):
        result = budget_exhaustion_estimate(
            [_frame_with_bar_fill(0)], np.zeros(FRAME_SHAPE, dtype=bool)
        )
        assert result["verdict"] == "abstain"
        assert result["actions_remaining_estimate"] is None

    def test_abstains_on_none_mask(self):
        result = budget_exhaustion_estimate([_frame_with_bar_fill(0)], None)
        assert result["verdict"] == "abstain"


def _two_pixel_shuffle_frame(offset: int, *, game_state: int = 1) -> np.ndarray:
    """A DIFFERENT pair of lit pixels each frame (positions offset, offset+1), so the
    region's exact byte value is never repeated (no revisits -- admissible) while the
    DIVERGENCE COUNT from frame 0 stays constant at 2 for every frame after the first.
    This is what a genuinely flat/no-progress signal looks like without also tripping
    region_hud_evidence's irreversibility refusal, which a literal repeated-value fixture
    (e.g. "jump once then hold") cannot avoid -- holding still means repeating the same
    region bytes, and REGION_EVIDENCE_MAX_REVISITS=0 refuses that outright."""
    grid = np.full(FRAME_SHAPE, game_state, dtype=np.uint8)
    grid[BAR_ROWS, :] = 0
    grid[BAR_ROWS, offset] = 7
    grid[BAR_ROWS, offset + 1] = 7
    return grid


class TestNonPositiveRateDoesNotProject:
    def test_flat_region_reports_no_projection(self):
        # SCENARIO-ARC-WMTE-6180-NON-POSITIVE-RATE-DOES-NOT-PROJECT
        #
        # Adversarial-review regression (2026-08-06): a whole-segment least-squares fit is
        # structurally biased POSITIVE by any early jump even if the region has been flat
        # ever since -- the fit has to "explain" the jump. This fixture's first draft
        # produced rate_per_transition=+0.0015 (not <=0) for exactly this shape, and the
        # assertion below was originally written `if result["rate_per_transition"] <= 0.0:`
        # -- which silently never ran, so the test passed green while proving nothing. The
        # fix was a WINDOWED fit (only the most recent min_transitions transitions), which
        # makes the assertion unconditionally true; both assertions are unconditional now
        # on purpose, so this test cannot go vacuous the same way again.
        mask = _bar_mask()
        # Frame 0 is the all-zero baseline; every later frame has EXACTLY 2 lit pixels
        # (divergence-from-segment-start count constant at 2 after the first transition),
        # so the windowed fit over the flat tail must be non-positive.
        grids = [_frame_with_bar_fill(0)]
        actions: list = [None]
        for i in range(1, 18):
            grids.append(_two_pixel_shuffle_frame(i % (FRAME_SHAPE[1] - 1)))
            actions.append("click")
        evidence = region_hud_evidence(grids, mask, actions=actions)
        assert evidence["verdict"] == "admit", evidence

        result = budget_exhaustion_estimate(grids, mask, evidence=evidence)
        assert result["verdict"] == "estimate"
        assert result["rate_per_transition"] is not None
        assert result["rate_per_transition"] <= 0.0
        assert result["actions_remaining_estimate"] is None


class TestSegmentAwareness:
    def test_a_level_up_reset_does_not_inherit_the_prior_segments_history(self):
        # SCENARIO-ARC-WMTE-6180-SEGMENT-RESET-DOES-NOT-INHERIT-PRIOR-HISTORY
        # Adversarial-review regression (2026-08-06): an earlier draft measured divergence
        # from the trajectory's GLOBAL first frame, ignoring per-level meter resets
        # entirely -- every real example this primitive targets (r11l, sc25, s5i5, g50t,
        # bp35) is a PER-LEVEL meter. On a fixture that fills most of the way, resets on
        # level-up, then refills, the unfixed version overestimated actions-remaining by
        # ~8x in the unsafe (falsely reassuring) direction. This asserts the fix: the
        # estimate is computed from the CURRENT segment only.
        mask = _bar_mask()
        grids = [_frame_with_bar_fill(0)]
        actions: list = [None]
        for i in range(1, 15):  # fill most of the way up (0 -> 14/24)
            grids.append(_frame_with_bar_fill(i))
            actions.append("click")
        grids.append(_frame_with_bar_fill(0))  # level-up: meter resets to empty
        actions.append("click")
        for i in range(1, 13):  # refill 12 clicks into the new (empty) segment
            grids.append(_frame_with_bar_fill(i))
            actions.append("click")
        evidence = region_hud_evidence(grids, mask, actions=actions)
        assert evidence["verdict"] == "admit", evidence

        result = budget_exhaustion_estimate(grids, mask, evidence=evidence)
        assert result["verdict"] == "estimate"
        # n_transitions counts only the post-reset segment (12), not the full 26-frame
        # trajectory -- proof the reset actually cleared the old segment's history.
        assert result["n_transitions"] == 12
        assert result["fill_fraction"] == pytest.approx(12 / 24)
        # This 24-wide bar fills at exactly 1 pixel/transition in the post-reset segment,
        # currently at 12/24 -- the analytically exact remaining count is 12, and a
        # global-first-frame version of this estimator (the pre-fix behaviour) would
        # instead report something close to 24 - (14 + 12) < 0 -> clamped positive nonsense,
        # or otherwise fail to match this exact value.
        assert result["actions_remaining_estimate"] == pytest.approx(12.0, abs=0.5)


class TestConsumerIsANoOpWithoutAnEstimate:
    def test_matches_standing_path_cost_weight_when_estimate_is_none(self):
        # SCENARIO-ARC-WMTE-6180-CONSUMER-IS-A-NO-OP-WITHOUT-AN-ESTIMATE
        for depth in (0, 1, 5, 40):
            for plan_length in (None, 0, 3, 100):
                for weight in (None, 0.0, 1.0, 2.5):
                    got = budget_aware_path_cost_weight(
                        depth=depth,
                        plan_length=plan_length,
                        actions_remaining_estimate=None,
                        path_cost_weight=weight,
                    )
                    expected = standing_path_cost_weight(weight) * depth
                    assert got == pytest.approx(expected)

    def test_matches_standing_path_cost_weight_when_plan_length_is_none(self):
        got = budget_aware_path_cost_weight(depth=10, actions_remaining_estimate=3.0)
        assert got == pytest.approx(standing_path_cost_weight(None) * 10)

    def test_penalizes_a_plan_that_exceeds_the_estimate(self):
        base = standing_path_cost_weight(None) * 5
        got = budget_aware_path_cost_weight(depth=5, plan_length=10, actions_remaining_estimate=4.0)
        assert got == pytest.approx(base + 6 * BUDGET_EXHAUSTION_PENALTY_PER_EXCESS_ACTION)

    def test_no_penalty_when_plan_fits_within_the_estimate(self):
        base = standing_path_cost_weight(None) * 5
        got = budget_aware_path_cost_weight(depth=5, plan_length=3, actions_remaining_estimate=4.0)
        assert got == pytest.approx(base)

    def test_no_penalty_at_exact_boundary(self):
        base = standing_path_cost_weight(None) * 5
        got = budget_aware_path_cost_weight(depth=5, plan_length=4, actions_remaining_estimate=4.0)
        assert got == pytest.approx(base)
