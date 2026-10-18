"""Online relational-mask move-pruner -- attacks the within-game DEEPENING enumeration wall by branching-
factor reduction (operator-directed 2026-06-28: "the next most likely way of improving our multi-level
live agent score").

WHY THIS EXISTS. The deepening wall (bank Lk -> reach Lk+1) is the SAME trajectory-enumeration wall as
first-contact: at Lk the agent can REACH the prefix and DETECT the next goal (GAP-4891 relational goal-
energy SEPARATES win from near-win), but the best-first search cannot ENUMERATE the multi-step trajectory
to Lk+1 -- branching x depth defeats it even with a correct goal-energy ordering the frontier (GAP-4891
Stage-2 / .452 A1 WALL_DEEPER). The GAP-4891 Stage-2 was explicitly UNDERPOWERED: it ordered the frontier
by goal-energy but did NOT prune the branches. This module adds the missing half: prune the candidate
actions to those that actually affect the relational TARGET REGION, shrinking the branching factor so a
deeper search becomes tractable.

THE MECHANISM (mirrors HazardMovePruner -- the live-path move-pruner precedent). The relational target
region (induce_relational_target_region) is the level-invariant canvas/target self-similarity area whose
change reaches the goal. The pruner learns, from the search's OWN observed transitions, which action
CLASSES never change any cell in that region, and prunes them -- a learned change-LOCATION prior (which is
learnable, cell_recall 0.727 per .450, unlike the representation-invariant change-VALUE). It is
CONSERVATIVE by construction:

  * it NEVER prunes an action class that has EVER produced a level-up (those are sacred);
  * it only prunes a class after >= min_observations observations that ALL missed the region (a
    specificity gate -- a false prune that broke a solve would surface as a failed reproduction gate);
  * when the region is unknown or a class is unproven, it does not prune (the search runs as before).

The region is supplied at construction (the deepening A/B replays to Lk and induces it) OR induced ONLINE
on the first observed level-up (the live path, no seed needed -- so it is safe to enable for any game and
no-ops on games with no relational structure). verifier_is_oracle: False -- a learned change-location
predictor fit from observed transitions, never the executable oracle that defines correctness.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import numpy as np


def _default_action_of_label(label: Any) -> Optional[dict]:
    """Decode a label to {'action': int, 'data': dict|None}. Returns None for undecodable labels so the
    pruner cleanly no-ops on them (it never prunes what it cannot key)."""
    d: Any = label
    if isinstance(label, str):
        try:
            d = json.loads(label)
        except (ValueError, TypeError):
            return None
    if isinstance(d, dict) and "action" in d:
        try:
            return {"action": int(d["action"]), "data": d.get("data")}
        except (ValueError, TypeError):
            return None
    try:
        return {"action": int(d), "data": None}
    except (ValueError, TypeError):
        return None


class RelationalMaskMovePruner:
    """Prunes search edges whose action CLASS never touches the relational target region. Consumer
    (OfflineSolver) lifecycle per expanded edge: ``should_prune(frame, label)`` BEFORE applying, then
    ``observe(frame_before, label, frame_after, leveled_up)`` after."""

    verifier_is_oracle = False

    def __init__(
        self,
        grid_of: Callable[[Any], np.ndarray],
        *,
        target_region: Optional[np.ndarray] = None,
        action_of_label: Optional[Callable[[Any], Optional[dict]]] = None,
        cell: int = 1,
        min_observations: int = 4,
        click_bucket: int = 1,
    ) -> None:
        self._grid_of = grid_of
        self._action_of = action_of_label or _default_action_of_label
        self.region = None if target_region is None else np.asarray(target_region, dtype=bool)
        self.seeded = target_region is not None
        self.cell = max(1, int(cell))
        self.min_observations = max(1, int(min_observations))
        self.click_bucket = max(1, int(click_bucket))
        # per-action-class tallies: key -> {"obs": n, "touched": n_touched_region, "leveled": n_levelups}
        self._tally: dict[Any, dict[str, int]] = {}
        self._nonwin_buffer: list[np.ndarray] = []  # recent pre-levelup grids, for online region induction
        self.pruned = 0
        self.observed = 0
        self.region_source = "seed" if self.seeded else "unknown"

    def _g2d(self, frame: Any) -> Optional[np.ndarray]:
        try:
            g = np.asarray(self._grid_of(frame))
        except Exception:
            return None
        if g.ndim == 3 and g.shape[0] == 1:
            g = g[0]
        return g if g.ndim == 2 else None

    def _key(self, decoded: dict) -> Any:
        """Action class key: keyboard actions key on the action id; clicks (action 6 + x,y) additionally
        key on the coarse cell the click lands in, so 'click here' and 'click there' are distinct classes."""
        action = int(decoded["action"])
        data = decoded.get("data")
        if isinstance(data, dict) and "x" in data and "y" in data:
            try:
                bx = (int(data["x"]) // self.cell) // self.click_bucket
                by = (int(data["y"]) // self.cell) // self.click_bucket
                return (action, by, bx)
            except (ValueError, TypeError):
                return (action, None, None)
        return (action, None, None)

    def _touches_region(self, g0: np.ndarray, g1: np.ndarray) -> bool:
        if self.region is None or g0.shape != g1.shape or g0.shape != self.region.shape:
            return True  # unknown / shape mismatch -> treat as 'touches' (do not learn-to-prune from it)
        changed = g0 != g1
        return bool((changed & self.region).any())

    def observe(
        self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False
    ) -> None:
        decoded = self._action_of(label)
        if decoded is None:
            return
        g0, g1 = self._g2d(frame_before), self._g2d(frame_after)
        if g0 is None or g1 is None:
            return
        # online region induction: on the FIRST level-up (no seed), induce the relational region from the
        # level-completion frame vs the recent pre-levelup frames.
        if leveled_up and self.region is None:
            self._induce_region_online(g1)
        if not leveled_up and len(self._nonwin_buffer) < 64:
            self._nonwin_buffer.append(g0)
        key = self._key(decoded)
        t = self._tally.setdefault(key, {"obs": 0, "touched": 0, "leveled": 0})
        t["obs"] += 1
        if leveled_up:
            t["leveled"] += 1
        elif self._touches_region(g0, g1):
            t["touched"] += 1
        self.observed += 1

    def _induce_region_online(self, win_grid: np.ndarray) -> None:
        from carnot.agentic.arc_agi3_goal_induction import induce_relational_target_region

        negs = [g for g in self._nonwin_buffer if g.shape == win_grid.shape]
        if not negs:
            return
        region = induce_relational_target_region(win_grid, negs)
        if region is not None:
            self.region = region
            self.region_source = "online_levelup"

    def should_prune(self, frame: Any, label: Any) -> bool:
        if self.region is None:
            return False
        decoded = self._action_of(label)
        if decoded is None:
            return False
        t = self._tally.get(self._key(decoded))
        if t is None or t["obs"] < self.min_observations:
            return False  # unproven class -> never prune
        if t["leveled"] > 0:
            return False  # this class has completed a level before -> SACRED, never prune
        if t["touched"] == 0:  # observed >= min_observations times, NEVER touched the target region
            self.pruned += 1
            return True
        return False

    def stats(self) -> dict:
        return {
            "observed": self.observed,
            "pruned": self.pruned,
            "region_known": self.region is not None,
            "region_source": self.region_source,
            "region_cells": int(self.region.sum()) if self.region is not None else 0,
            "action_classes": len(self._tally),
            "pruned_classes": sum(
                1
                for t in self._tally.values()
                if t["obs"] >= self.min_observations and t["leveled"] == 0 and t["touched"] == 0
            ),
            "verifier_is_oracle": False,
        }


class CompositeMovePruner:
    """Compose move-pruners (e.g. hazard + relational-mask): prune if ANY child prunes; observe ALL.
    Lets the live OfflineSolver carry several independent, conservative pruners at once."""

    verifier_is_oracle = False

    def __init__(self, *pruners: Any) -> None:
        self.pruners = [p for p in pruners if p is not None]

    def should_prune(self, frame: Any, label: Any) -> bool:
        return any(p.should_prune(frame, label) for p in self.pruners)

    def observe(self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False) -> None:
        for p in self.pruners:
            p.observe(frame_before, label, frame_after, leveled_up)

    def stats(self) -> dict:
        return {"composite": [p.stats() for p in self.pruners], "verifier_is_oracle": False}
