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
    drops into CarnotAgentPolicy(value_head=..., value_weight>0, search_mode="best_first") unchanged.

    REFUTED 2026-06-21 (results/arc_compete_sim.json, explorer_goalbias): the FIXED order direction
    (always bias toward consolidation) lifted first-win 0/10 and CRASHED the one working game (lp85
    20->437 actions) because lp85's win is NOT a consolidation. A fixed directional prior misroutes on
    games whose win-direction differs. Use ConfirmingGoalBiasValueHead instead -- it is direction-AGNOSTIC
    until the env confirms a direction. Kept as the refuted seed."""

    def __init__(self) -> None:
        self._evals = 0

    def __call__(self, frame: Any) -> float:
        self._evals += 1
        g = _grid(frame)
        if g is None:
            return 0.0
        return disorder(g)


def _features(grid: np.ndarray) -> tuple[float, float, float]:
    """The three direction-AGNOSTIC hypothesis axes, as comparable scalars (no assumed win-direction):
    (n_objects, n_colors, dominant-color coverage in [0,1]). A win-state is EXTREMAL on at least one of
    these -- but we do NOT assume which end (clear vs build, uniform vs varied). The explorer rewards
    moving toward an extreme on ANY axis pre-confirmation; once the env's levels_completed reveals a real
    level-up, the axis+direction that moved most becomes the confirmed bias for deepening."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        return (0.0, 0.0, 0.0)
    colors = np.unique(g)
    n_colors = float((colors != 0).sum())
    _, counts = np.unique(g, return_counts=True)
    dom_cov = float(counts.max()) / float(g.size)
    n_objects = float(min(_component_count(g), 64))
    return (n_objects, n_colors, dom_cov)


# per-axis normalizers so |delta| on each axis is comparable (objects 0-64, colors 0-15, coverage 0-1)
_AXIS_SCALE = (64.0, 15.0, 1.0)


class ConfirmingGoalBiasValueHead:
    """Direction-AGNOSTIC, online-CONFIRMING goal-bias value head (the v2 lever, 2026-06-21).

    Fixes the v1 (GoalBiasValueHead) refutation: v1 committed to ONE direction (consolidation) and
    misrouted games whose win lies the other way. v2 assumes only that a win-state is EXTREMAL on SOME
    hypothesis axis -- not which axis, not which end -- and CONFIRMS the true axis+direction online from
    the env's own level signal:

    PRE-CONFIRMATION (no level-up observed yet): value = 1 - max_axis(|feature - start_feature| / scale).
    Reward being far from the START along ANY axis (clear OR build, uniform OR varied). This is
    direction-agnostic, so it cannot misroute the way a fixed prior does; at worst it degrades to
    near-undirected (a weak, permissive signal) rather than actively steering away from the win.

    ONLINE CONFIRMATION: the value head reads ``frame.levels_completed`` on every frame it scores (the
    frame carries it; arc_agi3_live_adapter._levels_completed). The moment it sees a frame whose level
    exceeds the start level, it INDUCES the rewarded hypothesis = the (axis, direction) that moved most
    from start to that win-frame, and switches to BIASING toward extending that confirmed direction (to
    reach the NEXT level efficiently). This is the 'confirm against levels_completed' step, done with no
    extra harness hook -- the level signal rides on the frames the A* already scores.

    No trained weights, no win example, no game-specific state -- pure reusable PROCESS; transfers to the
    hidden eval by construction. Matches value_head(frame)->float (lower = expand first)."""

    def __init__(self) -> None:
        self._evals = 0
        self._start: tuple[float, float, float] | None = None
        self._start_level: int | None = None
        self._confirmed: tuple[int, float] | None = None   # (axis_index, sign in {+1,-1}) once a level-up is seen
        self._max_level_seen = 0

    @staticmethod
    def _level_of(frame: Any) -> int:
        try:
            from carnot.agentic.arc_agi3_live_adapter import _levels_completed
            return int(_levels_completed(frame))
        except Exception:
            return int(getattr(frame, "levels_completed", 0) or 0)

    def _confirm_from_win(self, win_feats: tuple[float, float, float]) -> None:
        """Induce the rewarded (axis, direction): the axis whose normalized move from start was largest."""
        if self._start is None:
            return
        best_axis, best_mag, best_sign = 0, -1.0, 1.0
        for ax in range(3):
            delta = (win_feats[ax] - self._start[ax]) / _AXIS_SCALE[ax]
            if abs(delta) > best_mag:
                best_axis, best_mag, best_sign = ax, abs(delta), (1.0 if delta >= 0 else -1.0)
        if best_mag > 0:
            self._confirmed = (best_axis, best_sign)

    def __call__(self, frame: Any) -> float:
        self._evals += 1
        g = _grid(frame)
        if g is None:
            return 0.0
        feats = _features(g)
        lvl = self._level_of(frame)
        if self._start is None:
            self._start = feats
            self._start_level = lvl
            self._max_level_seen = lvl
        # online confirmation: a frame past the start level reveals the rewarded axis+direction
        if self._confirmed is None and self._start_level is not None and lvl > self._start_level:
            self._confirm_from_win(feats)
        self._max_level_seen = max(self._max_level_seen, lvl)
        start = self._start
        if self._confirmed is not None:
            ax, sign = self._confirmed
            # bias toward EXTENDING the confirmed direction: lower value = further along it
            prog = sign * (feats[ax] - start[ax]) / _AXIS_SCALE[ax]   # >0 = moved the rewarded way
            return float(max(0.0, 1.0 - prog))
        # pre-confirmation: reward extremality on ANY axis (direction-agnostic) -> lower value = more extremal
        extremality = max(abs(feats[ax] - start[ax]) / _AXIS_SCALE[ax] for ax in range(3))
        return float(max(0.0, 1.0 - min(1.0, extremality)))
