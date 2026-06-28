"""Tests for the relational-mask move-pruner (deepening enumeration-wall lever, 2026-06-28).

Contract: induce_relational_target_region returns the full-grid region whose change reaches the goal;
RelationalMaskMovePruner learns per-action-class which moves touch that region and prunes ONLY action
classes that (a) have been observed >= min_observations times, (b) NEVER touched the region, and (c)
NEVER produced a level-up. It is conservative (no prune when unproven / region unknown) and never prunes a
level-completing class. verifier_is_oracle stays False. Spec: the deepening branching-reduction lever.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_agi3_goal_induction import induce_relational_target_region
from carnot.agentic.arc_relational_mask_pruner import RelationalMaskMovePruner

# win: a CANVAS block (rows1-3,cols0-2) equals a TARGET block (rows1-3,cols4-6) at offset dx=4.
WIN = np.zeros((6, 8), dtype=int)
WIN[1:4, 0:3] = 1
WIN[1:4, 4:7] = 1
NEARWIN = WIN.copy()
NEARWIN[1, 4] = 0  # one target cell not yet matched


def _gof(frame):
    return np.asarray(frame)


def test_region_covers_canvas_and_target_and_is_none_without_translate():
    region = induce_relational_target_region(WIN, [NEARWIN])
    assert region is not None and region.shape == WIN.shape and region.dtype == bool
    assert region.sum() >= 6  # the self-similarity mask (>= min_mask)
    assert bool(region[1, 4])  # the differing target cell is in the region
    # a frame with a single isolated non-bg cell has no translational target -> None
    lone = np.zeros((6, 6), dtype=int)
    lone[2, 2] = 1
    assert induce_relational_target_region(lone, [np.zeros((6, 6), dtype=int)]) is None


def test_prunes_action_class_that_never_touches_region():
    region = induce_relational_target_region(WIN, [NEARWIN])
    p = RelationalMaskMovePruner(_gof, target_region=region, min_observations=2)
    g0 = WIN.copy()
    g1_out = WIN.copy()
    g1_out[5, 7] = 9  # changes a NON-region cell (row 5 is outside rows1-3)
    for _ in range(3):
        p.observe(g0, '{"action": 2}', g1_out, False)
    assert p.should_prune(g0, '{"action": 2}') is True
    assert p.stats()["pruned"] >= 1


def test_does_not_prune_action_class_that_touches_region():
    region = induce_relational_target_region(WIN, [NEARWIN])
    p = RelationalMaskMovePruner(_gof, target_region=region, min_observations=2)
    g0 = NEARWIN.copy()  # has region cell (1,4)=0
    g1_touch = NEARWIN.copy()
    g1_touch[1, 4] = 1  # GENUINELY changes a region cell
    for _ in range(3):
        p.observe(g0, '{"action": 1}', g1_touch, False)
    assert p.should_prune(g0, '{"action": 1}') is False


def test_never_prunes_a_level_up_action_class():
    region = induce_relational_target_region(WIN, [NEARWIN])
    p = RelationalMaskMovePruner(_gof, target_region=region, min_observations=2)
    g0 = WIN.copy()
    g1_out = WIN.copy()
    g1_out[5, 7] = 9  # does NOT touch region...
    p.observe(g0, '{"action": 3}', g1_out, True)  # ...but produced a level-up once -> sacred
    for _ in range(3):
        p.observe(g0, '{"action": 3}', g1_out, False)
    assert p.should_prune(g0, '{"action": 3}') is False


def test_conservative_below_min_observations():
    region = induce_relational_target_region(WIN, [NEARWIN])
    p = RelationalMaskMovePruner(_gof, target_region=region, min_observations=5)
    g0 = WIN.copy()
    g1_out = WIN.copy()
    g1_out[5, 7] = 9
    p.observe(g0, '{"action": 2}', g1_out, False)  # only 1 obs < 5
    assert p.should_prune(g0, '{"action": 2}') is False


def test_no_region_no_prune_and_undecodable_label():
    p = RelationalMaskMovePruner(_gof, target_region=None, min_observations=1)
    g0 = WIN.copy()
    assert p.should_prune(g0, '{"action": 2}') is False  # region unknown -> never prune
    assert p.should_prune(g0, "not-json") is False  # undecodable -> never prune


def test_online_region_induction_on_levelup():
    # no seed region; feed non-win frames then a level-up frame -> region becomes known online
    p = RelationalMaskMovePruner(_gof, target_region=None, min_observations=2)
    assert p.region is None
    p.observe(NEARWIN.copy(), '{"action": 1}', NEARWIN.copy(), False)  # buffer a non-win
    p.observe(NEARWIN.copy(), '{"action": 1}', WIN.copy(), True)  # level-up frame = WIN -> induce
    assert p.region is not None and p.stats()["region_source"] == "online_levelup"


def test_click_classes_are_keyed_by_location():
    region = induce_relational_target_region(WIN, [NEARWIN])
    p = RelationalMaskMovePruner(_gof, target_region=region, min_observations=2, cell=1, click_bucket=1)
    g0 = WIN.copy()
    out = WIN.copy()
    out[5, 7] = 9
    # two clicks at different cells -> distinct classes; both miss the region -> both prunable
    for _ in range(3):
        p.observe(g0, '{"action": 6, "data": {"x": 0, "y": 5}}', out, False)
    assert p.should_prune(g0, '{"action": 6, "data": {"x": 0, "y": 5}}') is True
    # a click at a DIFFERENT cell is a different, unproven class -> not pruned
    assert p.should_prune(g0, '{"action": 6, "data": {"x": 7, "y": 0}}') is False
