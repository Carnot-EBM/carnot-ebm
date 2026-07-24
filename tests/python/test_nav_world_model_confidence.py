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
