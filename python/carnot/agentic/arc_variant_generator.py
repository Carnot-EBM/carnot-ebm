"""ARC-AGI-3 game VARIANT generator — mechanic-preserving observation transforms that manufacture
held-out-from-our-solver layout variants of the 25 public games WITHOUT authoring new game files.

Why (operator 2026-06-19): the ~110 eval games are held out by design (we can't and shouldn't get
them), and the 25 we have ARE all the public games. The one legitimate way to a bigger corpus +
benchmark is to MANUFACTURE training/validation games. The games' levels are hardcoded `Level(...)`
objects (not seeded), so mutating the 700-41k-line game files is fragile. Instead we transform the
OBSERVATION the agent sees, with transforms that PRESERVE the mechanic so the variant is still solvable
but is a layout the solver has never seen -- forcing genuine re-induction (the closest dev-side proxy
to the OOD eval, and a far bigger transfer benchmark than today's 2/7 leave-one-out on 25 games).

Two mechanic-preserving transforms:
  * COLOR PERMUTATION (default; safest): a bijection on the non-background palette. Object structure
    and ALL positions are unchanged, so NO action remap is needed -- a click at (x,y) is the same cell.
    A count/structure/glyph win-rule (e.g. count_4==32) holds under a consistent recolor as count_pi(4)
    ==32, so the variant is solvable and the agent must re-induce the rule in the new palette.
  * REFLECTION (optional): flip the grid on an axis; positions move, so click actions are inverse-
    remapped. Preserves count/structure mechanics; skip it for DIRECTION-sensitive games.

The transform is applied to each observed frame's grid (`arc_agi3_world_model.grid_of`). Integration
into the live play loop is a thin wrapper over the env's reset/action (TODO: wire into the LOO bench);
this module is the reusable, tested transform core.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

# ARC palette is 0..15; 0 is background/empty and is kept fixed so "emptiness" is invariant.
_N_COLORS = 16
_BG = 0


def _seed_int(game_id: str, variant: int) -> int:
    h = hashlib.sha256(f"{game_id}:{variant}".encode()).hexdigest()
    return int(h[:8], 16)


def color_permutation(game_id: str, variant: int) -> np.ndarray:
    """A deterministic bijective recolor map of length _N_COLORS. Index = original color, value = new
    color. Background (0) is fixed; colors 1.._N_COLORS-1 are permuted. Deterministic in
    (game_id, variant) so a variant is reproducible (ARC Solve Reproducibility)."""
    rng = np.random.default_rng(_seed_int(game_id, variant))
    perm = np.arange(_N_COLORS, dtype=np.uint8)
    rest = np.arange(1, _N_COLORS)
    rng.shuffle(rest)
    perm[1:] = rest
    return perm


def apply_color_permutation(grid: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """Recolor a grid by the bijection `cmap` (vectorized LUT). Shape + positions unchanged."""
    g = np.asarray(grid)
    clipped = np.clip(g, 0, _N_COLORS - 1).astype(np.intp)
    return cmap[clipped].astype(g.dtype)


def reflect_grid(grid: np.ndarray, axis: int = 1) -> np.ndarray:
    """Flip the grid (axis 1 = horizontal, axis 0 = vertical). Positions move -> remap click actions."""
    return np.flip(np.asarray(grid), axis=axis).copy()


def remap_click_for_reflection(x: int, y: int, w: int, h: int, axis: int = 1) -> tuple[int, int]:
    """Inverse-map a click in the REFLECTED view back to the real-env coordinate (so the agent's
    click on what it sees lands on the right real cell). 0-indexed, grid w x h."""
    if axis == 1:
        return (w - 1 - int(x), int(y))
    return (int(x), h - 1 - int(y))


def win_rule_preserved_under_recolor(target_color: int, cmap: np.ndarray) -> int:
    """A count/equality win-rule on `target_color` becomes the same rule on cmap[target_color] under
    the recolor -- the agent re-induces THIS. Returned for callers that ground a rule against a variant."""
    return int(cmap[int(target_color)])


def variant_signature(game_id: str, variant: int, kind: str = "color") -> str:
    """Stable id for a manufactured variant, for the registry / benchmark (never a live eval game)."""
    return f"{game_id}~{kind}{variant:02d}"


def transform_frame_grid(
    grid: np.ndarray,
    game_id: str,
    variant: int,
    *,
    reflect: Optional[int] = None,
) -> np.ndarray:
    """The full observation transform for a variant: optional reflection then color-permutation.
    Reflection (if used) also requires the caller to remap click actions via
    remap_click_for_reflection; color-permutation needs no action remap."""
    g = np.asarray(grid)
    if reflect is not None:
        g = reflect_grid(g, axis=reflect)
    return apply_color_permutation(g, color_permutation(game_id, variant))


class VariantEnv:
    """Wrap an offline Arcade game so the AGENT observes a mechanic-preserving VARIANT, while the REAL
    env keeps the true win/level logic. On reset()/step() the observed frame's grid is transformed
    (color-permutation by default -> positions unchanged -> NO click remap; optional reflection ->
    click data inverse-remapped). `levels_completed` and every other field pass through unchanged, so
    `_level_of`/win-detection read the REAL game -- a solve on the variant is a real solve. This is how
    the manufactured variants plug into the offline eval / LOO benchmark (a held-out-from-our-solver
    layout the agent must re-induce). All other methods/attrs delegate to the wrapped env."""

    def __init__(self, env: object, game_id: str, variant: int, *, reflect: Optional[int] = None):
        self._env = env
        self._game_id = game_id
        self._variant = variant
        self._reflect = reflect

    def __getattr__(self, name: str):  # delegate open_scorecard / etc. to the real env
        return getattr(self._env, name)

    def _wrap(self, frame):
        if frame is None:
            return None
        stack = np.array(frame.frame)  # (n, H, W) pixel-grid stack
        if stack.ndim == 2:
            stack = stack[None, ...]
        # TERMINAL / GRIDLESS FRAMES PASS THROUGH UNCHANGED (fixed 2026-07-24).
        # When a game ends the offline env returns a frame whose `.frame` is an EMPTY
        # list (verified on tu93: `GameState.GAME_OVER` -> `.frame == []`), which
        # `np.array` turns into a shape-(0,) 1-D array. The old code then evaluated
        # `range(stack.shape[0])` as `range(0)` and handed `np.stack` an empty list,
        # raising `ValueError: need at least one array to stack` and killing the whole
        # run. That made the variant harness crash on exactly the games that have a
        # death mechanic (tu93 / sp80 / cd82 / lf52 all died here; su15 / vc33 / lp85,
        # which cannot die, silently worked) -- i.e. the generalization benchmark was
        # unusable on 4 of the 7 games the live explorer can actually win.
        # There is nothing to recolor on a gridless frame, and every non-grid field
        # (`levels_completed`, `state`) must pass through untouched so `_level_of` still
        # reads the REAL game, so returning the frame as-is is both the safe and the
        # correct behaviour.
        if stack.ndim != 3 or stack.shape[0] == 0:
            return frame
        out = np.stack(
            [transform_frame_grid(stack[i], self._game_id, self._variant, reflect=self._reflect)
             for i in range(stack.shape[0])]
        )
        f2 = frame.model_copy() if hasattr(frame, "model_copy") else frame.copy()
        object.__setattr__(f2, "frame", out.tolist())
        return f2

    def _remap(self, data):
        # only reflection moves positions; color-permutation needs no remap
        if self._reflect is not None and isinstance(data, dict) and "x" in data and "y" in data:
            x, y = remap_click_for_reflection(int(data["x"]), int(data["y"]), 64, 64, self._reflect)
            return {**data, "x": x, "y": y}
        return data

    def reset(self, *a, **k):
        return self._wrap(self._env.reset(*a, **k))

    def step(self, action, data=None, *a, **k):
        return self._wrap(self._env.step(action, data=self._remap(data), *a, **k))
