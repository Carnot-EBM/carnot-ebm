"""Unit tests for arc_hazard_pruner.HazardMovePruner -- the wired-in hazard move-pruner.

The hazard MODEL itself (HazardAwareNavWorldModel) is covered by test_arc_nav_world_model.py. These tests
cover the PRUNER's orchestration: action-label decoding, the no-op safety guarantee (never prune without a
trusted hazard fit -- the property that makes it safe to enable for ANY game), and that should_prune
delegates to the fitted model and counts. The end-to-end states-expanded reduction is proven by the live
tu93 A/B recorded in results/arc_hazard_prune_ab_tu93.json (a real run, not a mock).
"""

import numpy as np

from carnot.agentic.arc_hazard_pruner import HazardMovePruner, _default_action_of_label


def test_action_label_decoder_handles_nav_and_non_nav():
    assert _default_action_of_label('{"action": 3}') == 3
    assert _default_action_of_label({"action": 4}) == 4
    assert _default_action_of_label(2) == 2
    assert _default_action_of_label('{"x": 1, "y": 2}') is None  # click game -> not a nav action
    assert _default_action_of_label("not json") is None


def test_no_prune_before_a_model_is_fitted():
    # The safety guarantee: with no trusted hazard model, the pruner NEVER prunes (so it cannot break a
    # solve on a game where no hazard exists or none has been learned yet).
    p = HazardMovePruner(lambda f: f)
    assert p.should_prune(np.zeros((5, 5), dtype=int), '{"action": 1}') is False
    assert p.pruned == 0
    assert p.stats()["model_fitted"] is False


def test_noop_when_no_deaths_observed():
    # Feed only safe nav transitions (avatar translates, never removed). No deaths => no hazard model =>
    # the pruner stays a no-op. This is what protects a non-hazard game from spurious pruning.
    p = HazardMovePruner(lambda f: f, refit_every=4, min_deaths=2)
    g = np.zeros((11, 11), dtype=int)
    g[2:5, 1:4] = 9  # 3x3 avatar (ring colour 9)
    g[3, 2] = 4  # avatar centre marker (colour 4)
    for _ in range(8):
        g2 = np.zeros_like(g)
        g2[2:5, 1:4] = 0  # clear old
        # shift avatar right by 1 column (a consistent displacement the nav fitter can learn)
        ar = np.argwhere(g == 9)
        rmin, cmin = ar[:, 0].min(), ar[:, 1].min()
        g2[rmin : rmin + 3, cmin + 1 : cmin + 4] = 9
        g2[rmin + 1, cmin + 2] = 4
        p.observe(g, '{"action": 4}', g2, leveled_up=False)
        g = g2
    assert p.n_deaths == 0
    assert p._model is None
    assert p.should_prune(g, '{"action": 4}') is False


def test_should_prune_delegates_to_model_and_counts():
    # With a fitted model present, should_prune returns the model's is_lethal verdict and counts prunes.
    class _StubModel:
        def is_lethal(self, grid, action):
            return int(action) == 2  # pretend action 2 is the lethal move

    p = HazardMovePruner(lambda f: f)
    p._model = _StubModel()
    grid = np.zeros((6, 6), dtype=int)
    assert p.should_prune(grid, '{"action": 2}') is True
    assert p.should_prune(grid, '{"action": 1}') is False
    assert p.should_prune(grid, '{"action": 2}') is True
    assert p.pruned == 2  # only the two lethal-move queries counted
    assert p.should_prune(grid, '{"x": 1}') is False  # non-nav label -> never prune


def test_stats_has_expected_shape():
    p = HazardMovePruner(lambda f: f)
    s = p.stats()
    for key in (
        "observed",
        "pruned",
        "n_deaths",
        "lethal_mode",
        "trust",
        "specificity",
        "model_fitted",
    ):
        assert key in s
