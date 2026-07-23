"""Tests for the opt-in grid-fallback click-candidate generation
(GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS, 2026-07-23 follow-up to REQ-ARC-FCP-5757).

Root cause: `object_centric_digest` (python/carnot/agentic/arc_solver_kit.py) excludes the single
most-common color wholesale as "background" -- but the most-common color is not always true
background (bp35's win condition needs clicking individual same-row "blocker" cells that all share
that color), so those cells never become click candidates via `rich_action_candidates`, regardless
of search depth or LLM judgment downstream. Fixed by an opt-in (env-var-gated,
`CARNOT_ARC_GRID_FALLBACK_CANDIDATES=1`, default OFF) coarse-tile fallback over the excluded
background mask. Measured directly on bp35's real reproduction-gated winning trajectory: 21/57
generation misses -> 15/57 with the fix (a genuine, disclosed PARTIAL improvement, not a complete
fix -- the remaining misses share candidate-generation budget with real, higher-salience objects).
lf52 and re86 (which had zero generation misses either way) are unaffected on and off.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic.arc_solver_kit import object_centric_digest
from carnot.agentic import arc_graph_explore


def _frame(grid: np.ndarray, available_actions=(1, 6)) -> SimpleNamespace:
    return SimpleNamespace(frame=grid, available_actions=list(available_actions))


class TestObjectCentricDigestGridFallback:
    def test_default_off_matches_prior_behavior_exactly(self):
        grid = np.zeros((16, 16), dtype=int)
        grid[4:6, 4:6] = 3  # a small real object
        d_default = object_centric_digest(grid)
        d_explicit_off = object_centric_digest(grid, emit_grid_fallback_for_background=False)
        assert d_default == d_explicit_off
        assert all(not c.get("is_grid_fallback") for c in d_default["components"])

    def test_fallback_tiles_emitted_for_large_background_color(self):
        grid = np.full((16, 16), 5, dtype=int)  # color 5 is the ENTIRE grid -> "background"
        grid[2, 2] = 3  # one real object so background != only color present
        d = object_centric_digest(
            grid, emit_grid_fallback_for_background=True, grid_fallback_tile_px=4
        )
        assert d["background_color"] == 5
        fallback = [c for c in d["components"] if c.get("is_grid_fallback")]
        assert len(fallback) > 0
        assert all(c["color"] == 5 for c in fallback)

    def test_fallback_tile_covers_a_specific_target_cell(self):
        # mirrors the real bp35 shape: one huge same-color region, a target cell buried inside it
        grid = np.full((32, 32), 5, dtype=int)
        grid[0, 0] = 9  # tiny real object elsewhere
        d = object_centric_digest(
            grid, emit_grid_fallback_for_background=True, grid_fallback_tile_px=8
        )
        fallback = [c for c in d["components"] if c.get("is_grid_fallback")]
        target_y, target_x = 20, 14
        covering = [
            c
            for c in fallback
            if c["bbox"][0] <= target_y <= c["bbox"][2] and c["bbox"][1] <= target_x <= c["bbox"][3]
        ]
        assert covering, "no fallback tile covers the target cell -- tiling gap"

    def test_fails_closed_when_tile_count_exceeds_cap(self):
        grid = np.full((64, 64), 5, dtype=int)
        grid[0, 0] = 9
        # tile_px=1 on a 64x64 background -> ~4095 tiles, way over any reasonable cap
        d = object_centric_digest(
            grid,
            emit_grid_fallback_for_background=True,
            grid_fallback_tile_px=1,
            grid_fallback_max_tiles=64,
        )
        fallback = [c for c in d["components"] if c.get("is_grid_fallback")]
        assert fallback == []  # omitted entirely, not an arbitrary truncated subset

    def test_real_components_unaffected_by_fallback_flag(self):
        grid = np.zeros((16, 16), dtype=int)
        grid[4:6, 4:6] = 3
        d_off = object_centric_digest(grid)
        d_on = object_centric_digest(grid, emit_grid_fallback_for_background=True)
        real_off = [c for c in d_off["components"] if not c.get("is_grid_fallback")]
        real_on = [c for c in d_on["components"] if not c.get("is_grid_fallback")]
        assert real_off == real_on


class TestRichActionCandidatesGridFallbackToggle:
    def setup_method(self):
        os.environ.pop("CARNOT_ARC_GRID_FALLBACK_CANDIDATES", None)

    def teardown_method(self):
        os.environ.pop("CARNOT_ARC_GRID_FALLBACK_CANDIDATES", None)

    def test_flag_off_by_default_no_fallback_candidates(self):
        grid = np.full((16, 16), 5, dtype=int)
        grid[2, 2] = 3
        frame = _frame(grid)
        candidates = arc_graph_explore.rich_action_candidates(frame)
        # only the one real object should produce a click candidate; the 5-colored background
        # (255/256 cells) must NOT flood the output when the flag is off
        click_candidates = [c for c in candidates if c.action_id == 6]
        assert len(click_candidates) <= 2

    def test_flag_on_adds_background_fallback_candidates(self):
        grid = np.full((32, 32), 5, dtype=int)
        grid[0, 0] = 3
        frame = _frame(grid)
        off = arc_graph_explore.rich_action_candidates(frame)
        os.environ["CARNOT_ARC_GRID_FALLBACK_CANDIDATES"] = "1"
        on = arc_graph_explore.rich_action_candidates(frame)
        click_off = [c for c in off if c.action_id == 6]
        click_on = [c for c in on if c.action_id == 6]
        assert len(click_on) > len(click_off)

    def test_flag_on_only_adds_candidates_never_removes_or_changes_real_ones(self):
        # ON can ADD fallback candidates but must never remove or alter a real one -- this is the
        # invariant that matters (byte-identical equality doesn't hold in general: even a
        # maximally-diverse grid still has ONE most-common color by definition, so turning the
        # flag on can always add at least a tiny fallback tile for it).
        grid = np.arange(16).reshape(4, 4)
        frame = _frame(grid)
        off = arc_graph_explore.rich_action_candidates(frame)
        os.environ["CARNOT_ARC_GRID_FALLBACK_CANDIDATES"] = "1"
        on = arc_graph_explore.rich_action_candidates(frame)
        off_keys = {(c.action_id, tuple(sorted((c.data or {}).items()))) for c in off}
        on_keys = {(c.action_id, tuple(sorted((c.data or {}).items()))) for c in on}
        assert off_keys <= on_keys
        assert len(on) >= len(off)
