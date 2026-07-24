"""Regression: InducedNavWorldModel.is_confident_nav gates the live nav inducer off NON-nav games.

The fitter fits a (spurious) model for source-verified non-navigation games (sk48 two-snake sequence-match;
wa30 Sokoban crate-push), where firing installs a plan that cannot win and wastes actions. is_confident_nav
rejects those (avatar captured padding-0, or <3 directions) while keeping the real nav game (tu93).
Spec: REQ-ARC-WMTE-5844.
"""

from __future__ import annotations

from carnot.agentic.arc_nav_world_model import InducedNavWorldModel


def _model(avatar, disp_keys, goal):
    return InducedNavWorldModel(
        displacement={a: (1, 0) for a in disp_keys},
        avatar_colors=frozenset(avatar),
        bg_color=5,
        floor_color=5,
        wall_colors=frozenset(),
        goal_color=goal,
    )


def test_tu93_like_is_confident():
    # avatar {9,4}, all 4 directions, goal 14 -> confident
    assert _model({9, 4}, [1, 2, 3, 4], 14).is_confident_nav() is True


def test_sk48_like_not_confident_padding_and_few_dirs():
    # avatar includes 0 (padding) AND only 2 directions -> not confident
    assert _model({0, 1}, [1, 2], 3).is_confident_nav() is False


def test_wa30_like_not_confident_padding_in_avatar():
    # avatar includes 0 (padding) even with 4 directions -> not confident
    assert _model({0, 14}, [1, 2, 3, 4], 2).is_confident_nav() is False


def test_no_goal_not_confident():
    assert _model({9, 4}, [1, 2, 3, 4], None).is_confident_nav() is False


def test_too_few_directions_not_confident():
    # a clean avatar but only 2 directions -> not confident (default min 3)
    assert _model({9}, [1, 2], 14).is_confident_nav() is False


class TestGoalEnergy:
    """REQ-ARC-WMTE-5845: the nav model's player->goal Manhattan energy for best-first plan_in_model
    (fewer search nodes than plain BFS -> more robust within the node budget on bigger mazes)."""

    def _m(self, avatar=(9,), goal=14):
        return InducedNavWorldModel(
            displacement={1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)},
            avatar_colors=frozenset(avatar), bg_color=5, floor_color=5,
            wall_colors=frozenset(), goal_color=goal,
        )

    def test_manhattan_distance(self):
        import numpy as np
        g = np.full((10, 10), 5, dtype=np.int16)
        g[1, 1] = 9   # avatar at (1,1)
        g[4, 5] = 14  # goal at (4,5) -> |4-1|+|5-1| = 7
        assert abs(self._m().goal_energy(g) - 7.0) < 1e-6

    def test_zero_when_goal_gone(self):
        import numpy as np
        g = np.full((8, 8), 5, dtype=np.int16)
        g[2, 2] = 9  # avatar present, no goal color -> at/past win
        assert self._m().goal_energy(g) == 0.0

    def test_large_when_avatar_missing(self):
        import numpy as np
        g = np.full((8, 8), 5, dtype=np.int16)
        g[3, 3] = 14  # goal present, no avatar
        assert self._m().goal_energy(g) >= 999.0

    def test_none_goal_color_is_large(self):
        import numpy as np
        g = np.full((8, 8), 5, dtype=np.int16)
        g[1, 1] = 9
        assert self._m(goal=None).goal_energy(g) >= 999.0
