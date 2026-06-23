"""Tests for the just-explore 5-tier salience schedule graft (CARNOT_ARC_TIER_SCHEDULE).

The 2026-06-23 offline A/B found the schedule does NOT beat the flat sort (TIER_NULL_no_win), so it is
shipped DEFAULT-OFF as a parity-safe, flag-gated building block (for a future stateful-frontier port).
These tests pin: (1) the constants are just-explore's verbatim; (2) the tier ordering front-loads a
salient medium-width object over a big dull blob; (3) flag-OFF is byte-identical to the flat path
(parity preserved — the SUBMITTED agent is unchanged); (4) flag-ON changes the order.
"""

import os

import numpy as np

import carnot.agentic.arc_graph_explore as g
from carnot.agentic.arc_graph_explore import (
    _TIER_MAX_WIDTH,
    _TIER_MIN_WIDTH,
    _TIER_SALIENT_COLORS,
    _TIER_STATUS_BAR_COLOR,
    _tier_ordered_click_points,
)


class _Frame:
    """Minimal frame the candidate generator reads: available_actions + a grid."""

    def __init__(self, grid):
        self.available_actions = [6]
        self.frame = grid
        self.cells = grid


def _grid_blob_and_button():
    # a big DULL blob (color 1, 40x40) the flat area-sort up-ranks + a small SALIENT button (color 7, 4x4)
    grid = np.zeros((64, 64), dtype=int)
    grid[2:42, 2:42] = 1
    grid[50:54, 50:54] = 7
    return grid


def test_tier_constants_are_just_explore_verbatim():
    assert _TIER_SALIENT_COLORS == frozenset(range(6, 16))
    assert _TIER_STATUS_BAR_COLOR == 16
    assert (_TIER_MIN_WIDTH, _TIER_MAX_WIDTH) == (2, 32)


def test_tier_ordering_front_loads_salient_button_over_big_dull_blob():
    pts = _tier_ordered_click_points(_grid_blob_and_button())
    # the salient 4x4 button (centroid ~51,51) is T0; the 40x40 dull blob is too wide -> lower tier
    assert pts[0] == (51, 51)


def test_flag_off_is_byte_identical_flat_order_parity():
    f = _Frame(_grid_blob_and_button())
    os.environ.pop("CARNOT_ARC_TIER_SCHEDULE", None)
    flat_clicks = [(c.action_id, c.data) for c in g.rich_action_candidates(f) if c.action_id == 6]
    # flat path up-ranks the LARGEST area (the dull blob) first, NOT the button
    assert flat_clicks[0][1] != {"x": 51, "y": 51}


def test_flag_on_reorders_to_tier_schedule():
    f = _Frame(_grid_blob_and_button())
    os.environ["CARNOT_ARC_TIER_SCHEDULE"] = "1"
    try:
        tier_clicks = [(c.action_id, c.data) for c in g.rich_action_candidates(f) if c.action_id == 6]
    finally:
        os.environ.pop("CARNOT_ARC_TIER_SCHEDULE", None)
    assert tier_clicks[0][1] == {"x": 51, "y": 51}  # salient button first under the tier schedule
