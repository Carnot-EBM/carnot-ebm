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
