"""Goal-biased exploration heuristic for first-contact ARC-AGI-3 solving (2026-06-21).

THE PROBLEM (measured, results/arc_compete_sim.json): the undirected explorer reaches a first level-up on
only 1/11 unseen games -- on the rest it burns ~2000 actions and never triggers the sparse terminal
level-completed bit. That exploration-to-first-win wall IS the 0.08 hidden-game score wall. The explorer
treats all reachable states as equal; it has no dense signal pointing at the latent win-mechanic.

THE METHOD (pure PROCESS, no trained weights -- per the operator's value-is-process directive): turn the
sparse terminal bit into a DENSE frontier heuristic by scoring each state with a PRIOR over what ARC win-
states tend to look like -- the same hypothesis classes the offline arc_agi3_goal_induction.py sketches
(object-count reduction, color-count reduction, color disappearance, coverage/completion). A plausibly-
winning state is more ORDERED (fewer distinct objects, fewer colors, higher single-color coverage). We
emit this as an estimated DISTANCE-TO-WIN (lower = more ordered = closer), which plugs straight into the
explorer's A* frontier (priority = depth + value_weight * value, a min-heap: lower value -> expanded
first), exactly where the learned cross-game value head plugs in (load_cross_game_value_head).

This is a SEED prior, applied to games never seen, with NO win example -- the zero-shot first-contact
bias. The online-confirmation refinement (prune/up-weight whichever hypothesis the env's levels_completed
signal actually rewards, once a level-up is observed) is the next layer; this module is the seed it starts
from. It transfers to the hidden eval by construction because it assumes nothing game-specific.

HONEST LIMITATION: the order prior assumes winning = consolidation/completion. Games where winning means
BUILDING (increasing structure) are mis-pointed by it; for those the online-confirmation layer must flip
the sign once a real level-up reveals the true direction. The value of THIS prototype is the measured
lift over the 1/11 undirected baseline -- if a fixed order-prior already lifts first-win count, the
hypothesis holds and the online layer is the amplifier; if not, we learn the prior needs the env-confirmed
direction from step one.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _grid(frame: Any) -> np.ndarray | None:
    """Best-effort logical grid from an arcengine frame (never raise into the A* seam)."""
    try:
        from carnot.agentic.arc_agi3_world_model import grid_of
        g = np.asarray(grid_of(frame))
        return g if g.ndim == 2 else None
    except Exception:
        return None


def disorder(grid: np.ndarray) -> float:
    """Estimated distance-to-win = a multi-hypothesis DISORDER score (lower = more ordered = closer to a
    plausible win). Combines the arc_agi3_goal_induction hypothesis classes into one scalar:

      - object-count: number of 4-connected non-background components (fewer = more consolidated)
      - color-count: number of distinct non-background colors (fewer = simpler)
      - coverage: fraction of cells that are NOT the dominant color (lower = more complete/uniform)

    All three are minimized at a clean/complete/uniform state, the shape most ARC win-states take. Scales
    are normalized so no single term dominates and the combined value sits on a small bounded range that a
    moderate value_weight can trade against search depth."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        return 0.0
    h, w = g.shape
    # color-count: distinct non-zero colors
    colors = np.unique(g)
    n_colors = int((colors != 0).sum())
    # coverage: fraction of cells NOT equal to the most common color (0 => fully uniform)
    vals, counts = np.unique(g, return_counts=True)
    dominant = int(counts.max())
    not_dominant_frac = 1.0 - dominant / float(g.size)
    # object-count: 4-connected non-background components (capped so a noisy frame can't dominate)
    n_objects = _component_count(g)
    n_objects_norm = min(n_objects, 64) / 8.0
    return float(n_objects_norm + n_colors + 4.0 * not_dominant_frac)


def _component_count(g: np.ndarray) -> int:
    """4-connected non-background component count (scipy if present, else a small BFS fallback)."""
    mask = g != 0
    if not mask.any():
        return 0
    try:
        from scipy import ndimage
        _, n = ndimage.label(mask)
        return int(n)
    except Exception:
        seen = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        n = 0
        from collections import deque
        for i in range(h):
            for j in range(w):
                if mask[i, j] and not seen[i, j]:
                    n += 1
                    q = deque([(i, j)])
                    seen[i, j] = True
                    while q:
                        y, x = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
        return n


class GoalBiasValueHead:
    """Frame -> estimated distance-to-win, for the explorer's A* frontier (priority = depth +
    value_weight*value; min-heap, so lower value is expanded first). A zero-shot ORDER prior over the
    arc_agi3_goal_induction hypothesis classes -- no trained weights, no win example, no game-specific
    state. Matches the load_cross_game_value_head callable contract (``value_head(frame) -> float``) so it
    drops into CarnotAgentPolicy(value_head=..., value_weight>0, search_mode="best_first") unchanged."""

    def __init__(self) -> None:
        self._evals = 0

    def __call__(self, frame: Any) -> float:
        self._evals += 1
        g = _grid(frame)
        if g is None:
            return 0.0
        return disorder(g)
