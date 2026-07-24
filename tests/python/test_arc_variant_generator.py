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


# ---------------------------------------------------------------------------
# VariantEnv regression tests (added 2026-07-24).
#
# The transform helpers above were tested from the start, but `VariantEnv` -- the
# class that actually plugs those transforms into a play loop, and therefore the
# only part of this module the A/B harness executes -- had NO coverage at all.
# A terminal-frame crash consequently survived undetected: on `GameState.GAME_OVER`
# the offline env returns a frame whose `.frame` is `[]`, and `_wrap` fed that empty
# stack to `np.stack`, raising `ValueError: need at least one array to stack`. The
# variant harness died on exactly the games that have a death mechanic. These tests
# reproduce that exact incident plus the pass-through invariants it must not break.
# ---------------------------------------------------------------------------


class _FakeFrame:
    """Minimal stand-in for the SDK's FrameDataRaw.

    Carries the `.frame` grid stack plus the non-grid fields the harness reads
    (`levels_completed`, `state`), and supports the `.copy()` path `_wrap` uses.
    """

    def __init__(self, frame, levels_completed=0, state="NOT_FINISHED"):
        self.frame = frame
        self.levels_completed = levels_completed
        self.state = state

    def copy(self):
        return _FakeFrame(self.frame, self.levels_completed, self.state)


class _FakeEnv:
    """Replays a scripted list of frames, recording the action data it was handed."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.seen_data = []

    def reset(self):
        return self._frames[0]

    def step(self, action, data=None):
        self.seen_data.append(data)
        return self._frames[min(len(self.seen_data), len(self._frames) - 1)]


def test_variant_env_recolors_a_normal_frame_and_preserves_structure() -> None:
    grid = np.array([[0, 1, 2], [3, 0, 4]], dtype=np.uint8)
    env = _FakeEnv([_FakeFrame([grid.tolist()])])
    wrapped = vg.VariantEnv(env, "tu93", 1)

    out = np.array(wrapped.reset().frame)

    assert out.shape == (1, 2, 3)
    cmap = vg.color_permutation("tu93", 1)
    assert np.array_equal(out[0], cmap[grid.astype(int)])
    # emptiness (and therefore object structure) is invariant under the recolor
    assert np.array_equal(out[0] != 0, grid != 0)


def test_variant_env_passes_through_terminal_gridless_frame_without_crashing() -> None:
    """The exact 2026-07-24 incident: a GAME_OVER frame carries `.frame == []`."""
    over = _FakeFrame([], levels_completed=2, state="GAME_OVER")
    env = _FakeEnv([over])
    wrapped = vg.VariantEnv(env, "tu93", 1)

    got = wrapped.reset()

    assert got.frame == []
    # the REAL game's level/state must survive untouched so `_level_of` stays honest
    assert got.levels_completed == 2
    assert got.state == "GAME_OVER"


def test_variant_env_keeps_playing_after_a_terminal_frame() -> None:
    """A death mid-run must not abort the episode -- the crash killed whole runs."""
    grid = np.array([[0, 5], [6, 0]], dtype=np.uint8)
    env = _FakeEnv([_FakeFrame([grid.tolist()]), _FakeFrame([], state="GAME_OVER")])
    wrapped = vg.VariantEnv(env, "cd82", 3)

    first = wrapped.reset()
    after_death = wrapped.step("ACTION1", data=None)

    assert np.array(first.frame).shape == (1, 2, 2)
    assert after_death.frame == []
    assert after_death.state == "GAME_OVER"


def test_variant_env_color_permutation_does_not_remap_click_coordinates() -> None:
    """Recolor leaves positions alone, so a click must reach the env unmodified."""
    grid = np.zeros((4, 4), dtype=np.uint8)
    env = _FakeEnv([_FakeFrame([grid.tolist()])])
    wrapped = vg.VariantEnv(env, "su15", 2)

    wrapped.reset()
    wrapped.step("ACTION6", data={"x": 7, "y": 11})

    assert env.seen_data == [{"x": 7, "y": 11}]


def test_variant_env_reflection_inverse_remaps_click_coordinates() -> None:
    """With reflection the view moves, so the click must be mapped back to the real cell."""
    grid = np.zeros((4, 4), dtype=np.uint8)
    env = _FakeEnv([_FakeFrame([grid.tolist()])])
    wrapped = vg.VariantEnv(env, "su15", 2, reflect=1)

    wrapped.reset()
    wrapped.step("ACTION6", data={"x": 7, "y": 11})

    assert env.seen_data == [{"x": 64 - 1 - 7, "y": 11}]
