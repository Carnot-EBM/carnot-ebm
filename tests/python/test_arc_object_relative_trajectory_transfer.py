"""REQ-ARC-WMTE-TRAJ-TRANSFER-1 (lever #2, 2026-08-07): object-relative trajectory transfer.

Origin: operator directive to adopt the FOYSAL leaderboard-technique-watch lever (the
2026-08-01 entry in docs/research-notes/arc-agi3-leaderboard-technique-watch.md) --
on level-up, replay the previous level's successful trace re-anchored by matching
objects and translating ACTION6 clicks by the mean centroid displacement.

These tests exercise `object_relative_trajectory_transfer` in isolation (no live agent,
no GPU) against small synthetic grids where the correct answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_solver_kit import object_relative_trajectory_transfer

BG = 0


def _grid(h: int, w: int, cells: dict[tuple[int, int], int]) -> np.ndarray:
    g = np.full((h, w), BG, dtype=np.int16)
    for (y, x), color in cells.items():
        g[y, x] = color
    return g


def test_uniform_shift_translates_clicks_and_is_confident() -> None:
    # A single 2x2 block of color 3, shifted +2 rows / +3 cols between the two grids.
    old_grid = _grid(10, 10, {(1, 1): 3, (1, 2): 3, (2, 1): 3, (2, 2): 3})
    new_grid = _grid(10, 10, {(3, 4): 3, (3, 5): 3, (4, 4): 3, (4, 5): 3})
    actions = [{"action": 6, "data": {"x": 15, "y": 15}}]
    result = object_relative_trajectory_transfer(old_grid, new_grid, actions, cell=10)
    assert result["matched_pairs"] == 1
    assert result["total_old_components"] == 1
    assert result["matched_fraction"] == 1.0
    assert result["displacement_std"] == pytest.approx(0.0, abs=1e-9)
    assert result["transfer_confident"] is True
    assert result["oob_dropped"] == 0
    # centroid moved +3 cols (x), +2 rows (y) in logical units; cell=10 -> frame pixels +30/+20
    assert result["translated_actions"] == [{"action": 6, "data": {"x": 45, "y": 35}}]


def test_no_shift_is_identity() -> None:
    old_grid = _grid(8, 8, {(1, 1): 5, (1, 2): 5})
    new_grid = _grid(8, 8, {(1, 1): 5, (1, 2): 5})
    actions = [{"action": 6, "data": {"x": 20, "y": 20}}]
    result = object_relative_trajectory_transfer(old_grid, new_grid, actions, cell=1)
    assert result["mean_dx"] == pytest.approx(0.0)
    assert result["mean_dy"] == pytest.approx(0.0)
    assert result["translated_actions"] == [{"action": 6, "data": {"x": 20, "y": 20}}]


def test_non_click_actions_pass_through_unchanged() -> None:
    old_grid = _grid(8, 8, {(1, 1): 5, (1, 2): 5})
    new_grid = _grid(8, 8, {(2, 2): 5, (2, 3): 5})
    actions = [{"action": 1, "data": None}, {"action": 6, "data": {"x": 10, "y": 10}}]
    result = object_relative_trajectory_transfer(old_grid, new_grid, actions, cell=1)
    assert result["translated_actions"][0] == {"action": 1, "data": None}
    # the click DID move (mean_dx/mean_dy nonzero) -- only the non-click action is untouched
    assert result["translated_actions"][1] != {"action": 6, "data": {"x": 10, "y": 10}}


def test_no_matchable_objects_is_not_confident() -> None:
    # Old grid has an object of a color absent from the new grid entirely.
    old_grid = _grid(8, 8, {(1, 1): 7})
    new_grid = _grid(8, 8, {(1, 1): 9})
    actions = [{"action": 6, "data": {"x": 10, "y": 10}}]
    result = object_relative_trajectory_transfer(old_grid, new_grid, actions)
    assert result["matched_pairs"] == 0
    assert result["matched_fraction"] == 0.0
    assert result["transfer_confident"] is False
    assert result["displacement_std"] == float("inf")


def test_empty_old_grid_is_not_confident_and_does_not_crash() -> None:
    old_grid = np.full((6, 6), BG, dtype=np.int16)
    new_grid = _grid(6, 6, {(1, 1): 4})
    result = object_relative_trajectory_transfer(
        old_grid, new_grid, [{"action": 6, "data": {"x": 1, "y": 1}}]
    )
    assert result["total_old_components"] == 0
    assert result["transfer_confident"] is False


def test_inconsistent_displacement_fails_the_std_gate() -> None:
    # Two same-color objects that move by DIFFERENT amounts -- a non-uniform transform the
    # single mean displacement cannot honestly describe.
    old_grid = _grid(20, 20, {(1, 1): 2, (10, 10): 2})
    new_grid = _grid(20, 20, {(1, 1): 2, (18, 18): 2})  # first stayed put, second moved far
    result = object_relative_trajectory_transfer(
        old_grid, new_grid, [{"action": 6, "data": {"x": 5, "y": 5}}], max_displacement_std=1.0
    )
    assert result["matched_pairs"] == 2
    assert result["matched_fraction"] == 1.0
    assert result["displacement_std"] > 1.0
    assert result["transfer_confident"] is False


def test_low_matched_fraction_fails_the_fraction_gate() -> None:
    # 3 old objects, only 1 has a same-color counterpart in the new grid.
    old_grid = _grid(20, 20, {(1, 1): 2, (5, 5): 3, (10, 10): 4})
    new_grid = _grid(20, 20, {(1, 1): 2})
    result = object_relative_trajectory_transfer(
        old_grid, new_grid, [{"action": 6, "data": {"x": 5, "y": 5}}], min_matched_fraction=0.5
    )
    assert result["matched_pairs"] == 1
    assert result["matched_fraction"] == pytest.approx(1.0 / 3.0)
    assert result["transfer_confident"] is False


def test_out_of_bounds_click_is_dropped_not_clamped() -> None:
    old_grid = _grid(70, 70, {(1, 1): 3})
    new_grid = _grid(70, 70, {(1, 61): 3})  # shifted far right in logical units
    actions = [{"action": 6, "data": {"x": 20, "y": 20}}]
    result = object_relative_trajectory_transfer(old_grid, new_grid, actions, cell=1)
    # mean_dx ~= 60 logical units at cell=1 -> translated x ~= 80, out of the live [0,63] bound
    assert result["oob_dropped"] == 1
    assert result["translated_actions"] == []


def test_scores_by_area_first_then_centroid_distance() -> None:
    # One old component (area=2, color=6, near the top-left). The new grid offers TWO color-6
    # candidates: an area-3 one placed CLOSE to the old position, and an exact area-2 one placed
    # FAR away. Area difference must be compared before centroid distance, so the far exact match
    # wins over the near mismatched one.
    old_grid = _grid(30, 30, {(1, 1): 6, (1, 2): 6})  # area-2 object near (1, 1.5)
    new_grid = _grid(
        30,
        30,
        {
            (2, 1): 6,
            (2, 2): 6,
            (2, 3): 6,  # area-3, CLOSE to the old position
            (25, 25): 6,
            (25, 26): 6,  # area-2 (exact match), FAR from the old position
        },
    )
    result = object_relative_trajectory_transfer(old_grid, new_grid, [])
    assert result["matched_pairs"] == 1
    # a wrong (near, area-3) match would give a small displacement; the correct (far, area-2)
    # match gives a large one -- assert the large one, proving area was compared first.
    assert abs(result["mean_dx"]) > 10 or abs(result["mean_dy"]) > 10


def test_action_missing_xy_data_passes_through() -> None:
    old_grid = _grid(8, 8, {(1, 1): 5})
    new_grid = _grid(8, 8, {(2, 2): 5})
    actions = [{"action": 6, "data": {}}]  # malformed ACTION6, no x/y
    result = object_relative_trajectory_transfer(old_grid, new_grid, actions)
    assert result["translated_actions"] == [{"action": 6, "data": {}}]
