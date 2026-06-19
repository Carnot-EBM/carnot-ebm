"""Tests for the ARC variant generator (manufactured held-out layout variants, 2026-06-19).
Every test asserts (Tests-Must-Run-and-Assert). The core property: the transforms PRESERVE the
mechanic (structure + counts), so a variant is still solvable but is a layout the solver never saw.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic import arc_variant_generator as vg


def test_color_permutation_is_a_bijection_with_fixed_background() -> None:
    cmap = vg.color_permutation("ka59", 1)
    assert cmap.shape == (16,)
    assert cmap[0] == 0  # background fixed
    assert sorted(cmap.tolist()) == list(range(16))  # a permutation (bijection)


def test_color_permutation_is_deterministic_and_variant_dependent() -> None:
    assert np.array_equal(vg.color_permutation("ka59", 1), vg.color_permutation("ka59", 1))
    assert not np.array_equal(vg.color_permutation("ka59", 1), vg.color_permutation("ka59", 2))
    assert not np.array_equal(vg.color_permutation("ka59", 1), vg.color_permutation("dc22", 1))


def test_recolor_preserves_structure_and_counts() -> None:
    grid = np.array([[0, 4, 4], [3, 0, 4], [4, 3, 0]], dtype=np.uint8)
    cmap = vg.color_permutation("ka59", 7)
    out = vg.apply_color_permutation(grid, cmap)
    # structure (which cells are non-empty) is invariant -> mechanic-relevant geometry preserved
    assert np.array_equal(grid != 0, out != 0)
    # a count win-rule is preserved: count of color c == count of cmap[c] in the variant
    for c in (3, 4):
        assert int((grid == c).sum()) == int((out == vg.win_rule_preserved_under_recolor(c, cmap)).sum())


def test_reflection_and_click_remap_are_inverse_consistent() -> None:
    grid = np.arange(12, dtype=np.uint8).reshape(3, 4)  # h=3, w=4
    refl = vg.reflect_grid(grid, axis=1)
    # a cell seen at (x,y) in the reflected view maps back to the real cell with the same value
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            rx, ry = vg.remap_click_for_reflection(x, y, w, h, axis=1)
            assert refl[y, x] == grid[ry, rx]


def test_transform_frame_grid_composes_reflection_then_recolor() -> None:
    grid = np.array([[0, 4], [4, 3]], dtype=np.uint8)
    out = vg.transform_frame_grid(grid, "tr87", 3, reflect=1)
    assert out.shape == grid.shape
    # un-recolor (compare non-bg structure) then un-reflect must recover the original structure
    un_reflected_structure = vg.reflect_grid(out != 0, axis=1)
    assert np.array_equal(un_reflected_structure, grid != 0)
    # and the composite actually changed the pixels (reflect+recolor is not the identity here)
    assert not np.array_equal(out, grid)


def test_variant_signature_is_stable_and_not_a_real_game() -> None:
    sig = vg.variant_signature("ka59", 2)
    assert sig == "ka59~color02"
    assert "~" in sig  # the ~ marks a MANUFACTURED variant, never a real eval game id
