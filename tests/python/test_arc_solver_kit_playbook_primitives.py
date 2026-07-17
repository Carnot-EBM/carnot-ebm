"""Unit tests for the exploration-playbook primitives in
python/carnot/agentic/arc_solver_kit.py (REQ-ARC-WMTE-5716).

These cover the game-AGNOSTIC exploration moves distilled from the solve corpus
(docs/research-notes/arc-exploration-playbook-20260717.md): empirical action-
semantics probing, absolute-trajectory reading over the multi-layer animation
array, unexplained-glyph detection, bounded reachability with proven-vs-capped
honesty, and death-prefix bisection. Every branch of each new function is
exercised.

Spec: REQ-ARC-WMTE-5716, SCENARIO-ARC-WMTE-5716.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic import arc_solver_kit as kit


# --------------------------------------------------------------------------
# _frame_layers / settled_grid
# --------------------------------------------------------------------------
class _FrameWithFrameAttr:
    def __init__(self, arr):
        self.frame = np.asarray(arr)
        self.levels_completed = 0


class _FrameWithUnderscoreOnly:
    """A frame object exposing only ._frame (no .frame) -> exercises the second
    attribute branch of _frame_layers."""

    def __init__(self, arr):
        self._frame = np.asarray(arr)
        self.levels_completed = 0


def test_frame_layers_from_frame_attr_2d():
    grid = [[1, 2], [3, 4]]
    layers = kit._frame_layers(_FrameWithFrameAttr(grid))
    assert len(layers) == 1
    assert np.array_equal(layers[0], np.asarray(grid))


def test_frame_layers_from_underscore_attr():
    stack = np.zeros((2, 2, 2), dtype=int)
    layers = kit._frame_layers(_FrameWithUnderscoreOnly(stack))
    assert len(layers) == 2


def test_frame_layers_bare_3d_array():
    stack = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    layers = kit._frame_layers(stack)
    assert len(layers) == 2
    assert np.array_equal(layers[-1], stack[-1])


def test_frame_layers_list_of_grids():
    layers = kit._frame_layers([[[1, 1], [1, 1]], [[2, 2], [2, 2]]])
    assert len(layers) == 2


def test_frame_layers_rejects_1d():
    with pytest.raises(ValueError):
        kit._frame_layers(np.asarray([1, 2, 3]))


def test_settled_grid_returns_last_layer():
    stack = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    assert np.array_equal(kit.settled_grid(stack), np.ones((2, 2)))


def test_grid_background_explicit_and_inferred_and_empty():
    assert kit._grid_background(np.zeros((2, 2)), 7) == 7
    grid = np.array([[0, 0, 1], [0, 2, 0]])
    assert kit._grid_background(grid, None) == 0  # 0 is most common
    assert kit._grid_background(np.empty((1, 0), dtype=int), None) == 0  # empty -> 0


# --------------------------------------------------------------------------
# probe_action_semantics
# --------------------------------------------------------------------------
class _ProbeFrame:
    def __init__(self, grid, level):
        self.frame = np.asarray(grid)
        self.levels_completed = level


class _ProbeEnv:
    """Toy: a color-5 player pixel slides right; 'noop' is inert; 'win' levels up;
    'die' drops the level; 'reshape' returns a different-shaped grid."""

    def __init__(self):
        self.pos = 0
        self.level = 0

    def reset(self):
        self.pos = 0
        self.level = 0
        return _ProbeFrame(self._grid(), self.level)

    def _grid(self):
        g = np.zeros((3, 3), dtype=int)
        g[0, min(self.pos, 2)] = 5
        return g

    def step(self, label):
        if label == "R":
            self.pos += 1
        elif label == "win":
            self.level += 1
        elif label == "die":
            self.level -= 1
        elif label == "reshape":
            return _ProbeFrame(np.zeros((1, 1), dtype=int), self.level)
        return _ProbeFrame(self._grid(), self.level)


def _probe_apply(env, label, frame):
    return env.step(label)


def test_probe_action_semantics_no_warmup_no_prefix():
    result = kit.probe_action_semantics(_ProbeEnv, _probe_apply, ["R", "noop", "win"])
    assert result["inert_labels"] == ["noop"]
    assert result["levelup_labels"] == ["win"]
    assert "R" in result["effective_labels"]
    assert result["lethal_labels"] == []
    r_row = next(r for r in result["rows"] if r["label"] == "R")
    assert r_row["changed_cells"] == 2  # player pixel moved: old cell off, new cell on
    assert result["verifier_is_oracle"] is False


def test_probe_action_semantics_warmup_prefix_death_and_reshape():
    # prefix raises the level to 1, so 'die' (level 1->0) counts as a death, and
    # the warmup branch is exercised.
    result = kit.probe_action_semantics(
        _ProbeEnv,
        _probe_apply,
        ["die", "reshape"],
        warmup_label="noop",
        prefix=["win"],
    )
    assert result["lethal_labels"] == ["die"]
    reshape_row = next(r for r in result["rows"] if r["label"] == "reshape")
    assert reshape_row["changed_cells"] is None  # shape changed -> unknown
    assert reshape_row["inert"] is False


# --------------------------------------------------------------------------
# read_absolute_trajectory + _dominant_direction
# --------------------------------------------------------------------------
def _sprite_stack(positions, *, color=5, shape=(5, 5)):
    """A (N,H,W) stack with a single `color` pixel at each (row,col) in positions;
    None marks a layer where the sprite is absent."""
    layers = []
    for pos in positions:
        g = np.zeros(shape, dtype=int)
        if pos is not None:
            g[pos] = color
        layers.append(g)
    return np.stack(layers)


def test_trajectory_single_layer_is_motionless():
    result = kit.read_absolute_trajectory(_sprite_stack([(2, 2)]))
    assert result["direction"] == "none"
    assert result["net_dy"] == 0.0 and result["net_dx"] == 0.0
    assert result["observed_count"] == 1


def test_trajectory_downward_with_absent_layer():
    # sprite at row0, absent, then row4 -> absolute downward motion recovered,
    # the absent middle layer skipped in the delta chain.
    stack = _sprite_stack([(0, 2), None, (4, 2)])
    result = kit.read_absolute_trajectory(stack, color=5)
    assert result["direction"] == "down"
    assert result["net_dy"] == 4.0
    assert result["observed_count"] == 2
    assert len(result["step_deltas"]) == 1  # only one chained delta (0 -> 4)


def test_trajectory_upward_and_horizontal_directions():
    up = kit.read_absolute_trajectory(_sprite_stack([(4, 2), (0, 2)]), color=5)
    assert up["direction"] == "up"
    right = kit.read_absolute_trajectory(_sprite_stack([(2, 0), (2, 4)]), color=5)
    assert right["direction"] == "right"
    left = kit.read_absolute_trajectory(_sprite_stack([(2, 4), (2, 0)]), color=5)
    assert left["direction"] == "left"


def test_trajectory_foreground_mode_when_color_none():
    # color=None -> track any non-background foreground pixel.
    result = kit.read_absolute_trajectory(_sprite_stack([(0, 0), (3, 0)]))
    assert result["direction"] == "down"
    assert result["net_dy"] == 3.0


def test_dominant_direction_all_axes():
    assert kit._dominant_direction(0.0, 0.0) == "none"
    assert kit._dominant_direction(2.0, 1.0) == "down"
    assert kit._dominant_direction(-2.0, 1.0) == "up"
    assert kit._dominant_direction(1.0, 2.0) == "right"
    assert kit._dominant_direction(1.0, -2.0) == "left"


# --------------------------------------------------------------------------
# find_unexplained_glyphs
# --------------------------------------------------------------------------
def _multi_glyph_grid():
    g = np.zeros((5, 5), dtype=int)
    g[0, 0] = 5  # known 2-cell component
    g[0, 1] = 5
    g[2, 2] = 7  # unknown 2-cell component
    g[2, 3] = 7
    g[4, 4] = 9  # unknown single pixel (area 1)
    return g


def test_find_unexplained_glyphs_excludes_known_and_small():
    grid = _multi_glyph_grid()
    result = kit.find_unexplained_glyphs(grid, known_colors=[5], min_area=2)
    assert result["unexplained_colors"] == [7]  # 5 known, 9 too small
    assert result["unexplained_count"] == 1
    comp = result["components"][0]
    assert comp["color"] == 7
    assert len(comp["centroid"]) == 2  # (x, y) click coords
    assert result["verifier_is_oracle"] is False


def test_find_unexplained_glyphs_includes_small_when_min_area_1():
    grid = _multi_glyph_grid()
    result = kit.find_unexplained_glyphs(grid, known_colors=[5], min_area=1)
    assert result["unexplained_colors"] == [7, 9]  # sorted, both unknown
    # sorted by area desc: the 2-cell color-7 component before the 1-cell color-9
    assert [c["color"] for c in result["components"]] == [7, 9]


# --------------------------------------------------------------------------
# bounded_reachability_search
# --------------------------------------------------------------------------
def _line_neighbors(state):
    if state < 3:
        yield ("inc", state + 1)


def test_reachability_bfs_finds_goal():
    result = kit.bounded_reachability_search(0, _line_neighbors, lambda s: s == 3)
    assert result["reached"] is True
    assert result["path"] == ["inc", "inc", "inc"]
    assert result["status"] == "goal"
    assert result["proven_unreachable"] is False


def test_reachability_best_first_with_priority():
    result = kit.bounded_reachability_search(
        0, _line_neighbors, lambda s: s == 3, priority=lambda s, d: -float(s)
    )
    assert result["reached"] is True
    assert result["status"] == "goal"


def test_reachability_start_is_goal():
    result = kit.bounded_reachability_search(0, _line_neighbors, lambda s: s == 0)
    assert result["reached"] is True
    assert result["path"] == []


def test_reachability_exhausted_is_proven():
    result = kit.bounded_reachability_search(0, _line_neighbors, lambda s: s == 99)
    assert result["reached"] is False
    assert result["status"] == "exhausted"
    assert result["proven_unreachable"] is True
    assert result["frontier_remaining"] == 0


def test_reachability_capped_nodes_not_proven():
    result = kit.bounded_reachability_search(0, _line_neighbors, lambda s: s == 99, max_nodes=1)
    assert result["status"] == "capped_nodes"
    assert result["proven_unreachable"] is False


def test_reachability_capped_depth_not_proven():
    result = kit.bounded_reachability_search(0, _line_neighbors, lambda s: s == 3, max_depth=1)
    assert result["status"] == "capped_depth"
    assert result["proven_unreachable"] is False


def test_reachability_state_hash_dedups_cycle():
    # 0 <-> 1 cycle; identity-hash dedup means the frontier empties -> exhausted,
    # exercising the seen-skip branch.
    def cyc(state):
        yield ("flip", 1 - state)

    result = kit.bounded_reachability_search(0, cyc, lambda s: s == 9)
    assert result["status"] == "exhausted"
    assert result["nodes_expanded"] == 2  # states 0 and 1, then both re-seen


def test_reachability_custom_state_hash_projection():
    # A projection hash collapsing parity: from 0, neighbor 2 hashes same as 0 and
    # is skipped, so only one extra state is expanded.
    def step(state):
        yield ("plus2", state + 2)

    result = kit.bounded_reachability_search(
        0, step, lambda s: s == 100, state_hash=lambda s: s % 2, max_nodes=50
    )
    assert result["reached"] is False
    # 0 expands -> 2 (hash 0, already seen) so frontier empties immediately.
    assert result["status"] == "exhausted"


# --------------------------------------------------------------------------
# bisect_death_prefix
# --------------------------------------------------------------------------
def test_bisect_finds_minimal_fatal_prefix():
    actions = ["a", "b", "c", "d", "e"]
    result = kit.bisect_death_prefix(actions, lambda k: k >= 3)
    assert result["fatal_prefix_len"] == 3
    assert result["fatal_action_index"] == 2
    assert result["fatal_action"] == "c"
    assert result["evaluations"] >= 1


def test_bisect_no_death_returns_none():
    result = kit.bisect_death_prefix(["a", "b"], lambda k: False)
    assert result["fatal_prefix_len"] is None
    assert result["fatal_action_index"] is None
    assert result["fatal_action"] is None


def test_bisect_death_before_any_action():
    result = kit.bisect_death_prefix(["a", "b"], lambda k: True)
    assert result["fatal_prefix_len"] == 0
    assert result["fatal_action_index"] is None  # death precedes any action
    assert result["fatal_action"] is None
