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


class TestGoalPresentGuard:
    """REQ-ARC-WMTE-5883: is_confident_nav(grid=...) must ALSO require the goal colour present in the
    plan-start grid. When the goal is absent, is_level_complete/goal_energy read that absence as already-won,
    so plan_in_model returns a bogus ~1-step 'win' the live agent executes for zero progress. The gate closes
    that. grid=None preserves the original behaviour."""

    import numpy as np

    def _m(self):
        return _model({9, 4}, [1, 2, 3, 4], 14)  # otherwise-confident tu93-like model, goal=14

    def test_confident_when_goal_present_in_grid(self):
        import numpy as np
        g = np.full((6, 6), 5, dtype=int)
        g[2, 2] = 9
        g[4, 4] = 14  # goal present
        assert self._m().is_confident_nav(grid=g) is True

    def test_not_confident_when_goal_absent_from_grid(self):
        import numpy as np
        g = np.full((6, 6), 5, dtype=int)
        g[2, 2] = 9  # goal colour 14 absent entirely
        assert self._m().is_confident_nav(grid=g) is False

    def test_grid_none_preserves_original_behaviour(self):
        # no grid supplied -> the goal-present check is skipped (backward compatible)
        assert self._m().is_confident_nav() is True
        assert self._m().is_confident_nav(grid=None) is True


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


class TestEngineWallBlocking:
    """REQ-ARC-WMTE-5879: the induced nav engine must BLOCK a move into a wall regardless of the avatar's
    per-action STEP SIZE. The prior 'mid-gap' blocking heuristic (r0 + dy//2) degenerated to the avatar's
    OWN origin cell for a 1-cell step (dy//2 == 0), so it never inspected the destination and walked the
    avatar straight through 1-cell-adjacent walls. tu93 masked this (its avatar jumps ~6 cells/action, so the
    mid-gap sampled real intermediate cells); a hidden nav game with unit-step movement would plan through
    walls -> plans that fail in the real env. The fix is a swept-footprint check over every cell the avatar
    enters, from the first step through the destination.
    """

    BG = FLOOR = 5
    AV = 9
    GOAL = 14
    WALL = 3

    def _model(self, disp):
        return InducedNavWorldModel(
            displacement=disp, avatar_colors=frozenset({self.AV}), bg_color=self.BG,
            floor_color=self.FLOOR, wall_colors=frozenset({self.WALL}), goal_color=self.GOAL,
        )

    def _avpos(self, g):
        import numpy as np
        w = np.argwhere(np.asarray(g) == self.AV)
        return tuple(int(x) for x in w[0]) if w.size else None

    def test_unit_step_down_into_wall_is_blocked(self):
        import numpy as np
        m = self._model({1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)})
        g = np.full((6, 6), self.BG, dtype=int)
        g[4, 3] = self.WALL
        g[3, 3] = self.AV
        # move DOWN (action 2) into the wall at (4,3) -> must stay at (3,3)
        assert self._avpos(m.engine(g, 2, None)) == (3, 3)

    def test_unit_step_right_into_wall_is_blocked(self):
        import numpy as np
        m = self._model({1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)})
        g = np.full((6, 6), self.BG, dtype=int)
        g[3, 4] = self.WALL
        g[3, 3] = self.AV
        assert self._avpos(m.engine(g, 4, None)) == (3, 3)

    def test_unit_step_into_free_cell_still_moves(self):
        import numpy as np
        m = self._model({4: (0, 1)})
        g = np.full((6, 6), self.BG, dtype=int)
        g[3, 3] = self.AV
        assert self._avpos(m.engine(g, 4, None)) == (3, 4)

    def test_unit_step_onto_goal_still_covers_it(self):
        import numpy as np
        m = self._model({4: (0, 1)})
        g = np.full((6, 6), self.BG, dtype=int)
        g[3, 4] = self.GOAL
        g[3, 3] = self.AV
        out = np.asarray(m.engine(g, 4, None))
        assert self._avpos(out) == (3, 4)
        assert not bool((out == self.GOAL).any())  # goal covered

    def test_multi_step_blocked_by_destination_wall(self):
        # A 6-cell jump whose ONLY wall is at the destination must also block (the old mid-gap check
        # sampled the middle cell, not the destination, so it missed destination-only walls even for
        # multi-step moves).
        import numpy as np
        m = self._model({4: (0, 6)})
        g = np.full((3, 12), self.BG, dtype=int)
        g[1, 0] = self.AV
        g[1, 6] = self.WALL
        assert self._avpos(m.engine(g, 4, None)) == (1, 0)

    def test_multi_step_clear_path_moves(self):
        import numpy as np
        m = self._model({4: (0, 6)})
        g = np.full((3, 12), self.BG, dtype=int)
        g[1, 0] = self.AV
        assert self._avpos(m.engine(g, 4, None)) == (1, 6)
