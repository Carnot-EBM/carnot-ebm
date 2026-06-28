"""Affordance-priored Go-Explore archive explorer for the ARC-AGI-3 hard tail (2026-06-21).

The hard-tail games (ls20/wa30/su15/tu93/cn04/m0r0/sk48) are STRUCTURE-bound: level-1 needs a deep (13-33
action), specifically-ordered sequence with NO intermediate reward (the level ticks only on the final
action), and some decisive actions are NON-SALIENT/STATEFUL (wa30's ACTION5 pick/place). Neither the
depth-first ride (over-commits one branch) nor random diversity (no systematic deep coverage) finds them.

Go-Explore (Ecoffet et al. 1901.10995 / "first return, then explore") attacks exactly this: maintain an
ARCHIVE of reached states (cells), repeatedly RETURN to an under-explored promising cell (replay its
trajectory in the deterministic offline sim), then EXPLORE from it -- decoupling "reach a deep frontier
state" from "explore it", so deep specific regions get systematic coverage that depth-first riding never
gives. AFFORDANCE-PRIORED: cells are selected by a score that prefers under-visited + recently-discovered
(frontier) cells, and exploration tries the FULL action set (all action types 1-6 + salient clicks, not
just the top-salient subset) so non-salient decisive actions are not skipped. No world model (gated-out,
moot); no LLM. Pure reusable PROCESS over the offline arcade.

Honest scope: Go-Explore gives the deep systematic coverage the tail needs, but without a true goal
gradient it is still searching a large space -- expected to crack the shallower-deep tail games, not all 7
(wa30 at depth 33 with stateful pick/place is the hardest). The LLM-as-reasoner gradient (propose sub-goals)
is the SOTA-flagged next layer on top of this archive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import numpy as np


def _coarse_cell(grid: np.ndarray, level: int, bins: int = 6) -> tuple:
    """Go-Explore cell key: a downsampled (bins x bins) signature of the grid + the level. Groups
    structurally-similar states so the archive stays manageable while distinguishing meaningful layouts."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        return (level, 0)
    h, w = g.shape
    ys = np.linspace(0, h, bins + 1).astype(int)
    xs = np.linspace(0, w, bins + 1).astype(int)
    sig = []
    for i in range(bins):
        for j in range(bins):
            block = g[ys[i] : max(ys[i] + 1, ys[i + 1]), xs[j] : max(xs[j] + 1, xs[j + 1])]
            # dominant colour of the block (mode) -- robust coarse signature
            vals, counts = np.unique(block, return_counts=True)
            sig.append(int(vals[counts.argmax()]) if vals.size else 0)
    return (level, tuple(sig))


def _frame_grid(frame: Any) -> np.ndarray:
    """Extract the 2-D (H,W) grid from a frame.

    BUG FIX (2026-06-25): the live env frame is a ``FrameDataRaw`` whose ``.frame`` attribute is a
    (1,H,W) 3-D array. The old code grabbed that raw 3-D array via ``hasattr(frame, "frame")``, so
    ``observe()``'s ``grid.ndim != 2`` guard early-returned on EVERY live frame and the archive was
    silently dead (observations stayed 0). ``grid_of`` already returns the correct 2-D (H,W) grid for
    a FrameDataRaw, so prefer it; keep the ndarray / ``.frame`` paths as fallbacks and squeeze a
    leading singleton channel so a (1,H,W) raw still collapses to (H,W).
    """
    if isinstance(frame, np.ndarray):
        arr = frame
    else:
        try:
            from carnot.agentic.arc_agi3_world_model import grid_of

            arr = np.asarray(grid_of(frame))
        except Exception:
            arr = np.asarray(getattr(frame, "frame")) if hasattr(frame, "frame") else np.asarray([])
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:  # FrameDataRaw.frame is (1,H,W) -> (H,W)
        arr = arr[0]
    return arr


def _frame_level(frame: Any) -> int:
    try:
        from carnot.agentic.arc_agi3_live_adapter import _levels_completed

        return int(_levels_completed(frame))
    except Exception:
        return int(getattr(frame, "levels_completed", 0) or 0)


def _normalise_prefix(path: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        {"action": int(step["action"]), "data": step.get("data")}
        for step in list(path or [])
        if isinstance(step, Mapping) and step.get("action") is not None
    ]


class GoExploreReplayArchive:
    """Live replay-prefix archive for first-return-then-explore scheduling.

    The live environment cannot teleport to archived states, so every selected
    cell is represented by a reset-replayable action prefix.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        bins: int = 6,
        max_cells: int = 256,
        selector: Optional[Callable[[Sequence[Mapping[str, Any]]], Optional[int]]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.bins = max(1, int(bins))
        self.max_cells = max(1, int(max_cells))
        self._cells: dict[tuple, dict[str, Any]] = {}
        self._observations = 0
        self._selected_prefixes = 0
        # IGE hook (2026-06-28): when a `selector` callable is supplied, the cell CHOICE among eligible
        # cells is delegated to it (e.g. an LLM promisingness judge); the heuristic stays as the fallback
        # when the selector is absent, returns None, or returns an out-of-range index. The plumbing
        # (observe/return/replay) is identical regardless of who picks the cell.
        self.selector = selector
        self._selector_calls = 0
        self._selector_used = 0
        self._selector_fallbacks = 0

    def observe(self, frame: Any, path: Sequence[Mapping[str, Any]] | None) -> None:
        if not self.enabled:
            return
        grid = _frame_grid(frame)
        if grid.ndim != 2 or grid.size == 0:
            return
        prefix = _normalise_prefix(path)
        key = _coarse_cell(grid, _frame_level(frame), bins=self.bins)
        self._observations += 1
        existing = self._cells.get(key)
        if existing is not None and len(existing["prefix"]) <= len(prefix):
            existing["seen"] = int(existing.get("seen", 0)) + 1
            return
        if len(self._cells) >= self.max_cells and key not in self._cells:
            worst = max(
                self._cells,
                key=lambda cell: (
                    int(self._cells[cell].get("visits", 0)),
                    -int(self._cells[cell].get("depth", 0)),
                ),
            )
            del self._cells[worst]
        self._cells[key] = {
            "prefix": prefix,
            "visits": 0,
            "depth": len(prefix),
            "seen": 1,
        }

    def select_prefix(
        self,
        *,
        current_path: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled or not self._cells:
            return []
        current = _normalise_prefix(current_path)
        eligible_items = [
            (key, entry)
            for key, entry in self._cells.items()
            if entry.get("prefix") and entry.get("prefix") != current
        ]
        if not eligible_items:
            return []
        selected = self._select_via_selector(eligible_items)
        if (
            selected is None
        ):  # no selector, selector declined, or out-of-range -> heuristic fallback
            selected = min(
                (entry for _key, entry in eligible_items),
                key=lambda entry: (
                    int(entry.get("visits", 0)),
                    -int(entry.get("depth", 0)),
                    tuple((step["action"], repr(step.get("data"))) for step in entry["prefix"]),
                ),
            )
        selected["visits"] = int(selected.get("visits", 0)) + 1
        self._selected_prefixes += 1
        return [dict(step) for step in selected["prefix"]]

    def _select_via_selector(
        self, eligible_items: list[tuple[tuple, dict[str, Any]]]
    ) -> Optional[dict[str, Any]]:
        """Delegate cell choice to the IGE selector if present. Returns the chosen cell entry, or None to
        signal the caller to use the heuristic (selector absent / >=2 cells needed / declined / bad index)."""
        if self.selector is None or len(eligible_items) < 2:
            return None
        descriptors = []
        for index, (key, entry) in enumerate(eligible_items):
            level = int(key[0]) if isinstance(key, tuple) and key else 0
            sig = (
                key[1]
                if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], tuple)
                else None
            )
            descriptors.append(
                {
                    "index": index,
                    "level": level,
                    "signature": sig,
                    "depth": int(entry.get("depth", 0)),
                    "visits": int(entry.get("visits", 0)),
                    "seen": int(entry.get("seen", 0)),
                }
            )
        self._selector_calls += 1
        try:
            choice = self.selector(descriptors)
        except Exception:
            choice = None
        if choice is None or not isinstance(choice, int) or not (0 <= choice < len(eligible_items)):
            self._selector_fallbacks += 1
            return None
        self._selector_used += 1
        return eligible_items[choice][1]

    def diagnostics(self) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "enabled": bool(self.enabled),
            "stored_cells": int(len(self._cells)),
            "observations": int(self._observations),
            "selected_prefixes": int(self._selected_prefixes),
            "max_depth": max(
                (int(entry.get("depth", 0)) for entry in self._cells.values()), default=0
            ),
            "verifier_is_oracle": False,
            "selector_enabled": bool(self.selector is not None),
            "selector_calls": int(self._selector_calls),
            "selector_used": int(self._selector_used),
            "selector_fallbacks": int(self._selector_fallbacks),
        }
        sel_diag = getattr(self.selector, "diagnostics", None)
        if callable(sel_diag):
            try:
                diag["selector_diagnostics"] = sel_diag()
            except Exception:
                pass
        return diag


def coerce_go_explore_archive(value: Any) -> GoExploreReplayArchive | None:
    if value is None or value is False:
        return None
    if isinstance(value, GoExploreReplayArchive):
        return value
    if isinstance(value, Mapping):
        return GoExploreReplayArchive(
            enabled=bool(value.get("enabled", True)),
            bins=int(value.get("bins", 6)),
            max_cells=int(value.get("max_cells", 256)),
            selector=_coerce_archive_selector(value.get("selector")),
        )
    if value is True:
        return GoExploreReplayArchive()
    return None


def _coerce_archive_selector(value: Any) -> Optional[Callable[..., Optional[int]]]:
    """Normalize the archive's ``selector`` config into a callable | None. A callable is used directly;
    the string ``"ige"`` (or a Mapping with ``{"kind": "ige", ...}``) lazily builds the IGE LLM selector
    (lazy import avoids loading the heavy world-model/LLM module unless IGE is actually requested)."""
    if value is None or value is False:
        return None
    if callable(value):
        return value
    if isinstance(value, str) and value.lower().startswith("ige"):
        from carnot.agentic.arc_ige_cell_selector import coerce_ige_cell_selector

        return coerce_ige_cell_selector(value)
    if isinstance(value, Mapping) and str(value.get("kind", "")).lower().startswith("ige"):
        from carnot.agentic.arc_ige_cell_selector import coerce_ige_cell_selector

        return coerce_ige_cell_selector(dict(value))
    return None


def go_explore_solve(
    game: str,
    *,
    budget: int = 2000,
    explore_steps: int = 25,
    full_action_set: bool = True,
    seed: int = 0,
    warmup: bool = False,
) -> dict:
    """Affordance-priored Go-Explore on the OFFLINE arcade. Returns levels reached + diagnostics."""
    import random
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

    from carnot.agentic.arc_agi3_world_model import frame_hash

    rng = random.Random(seed)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f0 = _warm(env, warmup)

    def ok(frame) -> bool:
        try:
            return np.asarray(grid_of(frame)).ndim == 2
        except Exception:
            return False

    if not ok(f0):
        return {"game": game, "levels_reached": 0, "error": "start frame degenerate"}
    cell0_g = grid_of(f0)
    cell = detect_cell(cell0_g)
    start_level = _levels_completed(f0)
    best_level = start_level

    def cellify(frame) -> tuple:
        # EXACT-grid cell (frame_hash + level): the 6x6 coarse signature collapsed sub-block-change games
        # (sp80) to a single cell so the archive never grew. Exact cells guarantee the archive grows with
        # each distinct state -- the Go-Explore frontier IS the explored state set; the visits+depth weighting
        # focuses return-and-explore on the deep, under-visited frontier. (_coarse_cell kept for reference.)
        return (frame_hash(grid_of(frame)), _levels_completed(frame))

    def candidate_actions(frame) -> list:
        """FULL affordance-ranked candidate set: salient clicks (object-centric) + ALL keyboard action
        types, so non-salient decisive actions (e.g. wa30 ACTION5 pick/place) are never skipped."""
        cands = list(rich_action_candidates(frame))
        have_types = {int(c.action_id) for c in cands}
        if full_action_set:
            avail = list(getattr(frame, "available_actions", []) or range(1, 7))
            for aid in avail:
                if (
                    int(aid) != 6 and int(aid) not in have_types
                ):  # 6 = click (already object-centric)
                    cands.append(type(cands[0])(action_id=int(aid), data=None) if cands else None)
        return [c for c in cands if c is not None]

    # archive: cell -> {traj: [(action_id,data)...], visits, depth}
    archive: dict[tuple, dict] = {cellify(f0): {"traj": [], "visits": 0, "depth": 0}}
    actions = 0
    iters = 0
    first_levelup: Optional[int] = None

    def replay(traj) -> Any:
        nonlocal actions
        f = _warm(env, warmup)
        for aid, data in traj:
            if actions >= budget:
                return f
            f = env.step(_game_action(GameAction, int(aid)), data=data)
            actions += 1
            if f is None or not ok(f):
                return f
        return f

    while actions < budget:
        iters += 1
        # SELECT: weight under-visited + deeper (frontier) cells higher (affordance for progress)
        keys = list(archive.keys())
        weights = [
            1.0 / (1 + archive[k]["visits"]) * (1.0 + 0.15 * archive[k]["depth"]) for k in keys
        ]
        tot = sum(weights) or 1.0
        r = rng.random() * tot
        acc = 0.0
        chosen = keys[-1]
        for k, w in zip(keys, weights):
            acc += w
            if acc >= r:
                chosen = k
                break
        entry = archive[chosen]
        entry["visits"] += 1
        # RETURN: replay to the chosen cell
        f = replay(entry["traj"])
        if f is None or not ok(f) or actions >= budget:
            continue
        # EXPLORE: take explore_steps affordance-ranked-with-randomization actions from the cell
        traj = list(entry["traj"])
        for _ in range(explore_steps):
            if actions >= budget:
                break
            if not ok(f):
                break
            cands = candidate_actions(f)
            if not cands:
                break
            # affordance-priored: bias to the salient head but keep full-set diversity (epsilon-random)
            c = cands[0] if rng.random() < 0.5 else cands[rng.randrange(len(cands))]
            nf = env.step(_game_action(GameAction, int(c.action_id)), data=c.data)
            actions += 1
            if nf is None:
                break
            lvl = _levels_completed(nf) if ok(nf) else best_level
            traj = traj + [(int(c.action_id), c.data)]
            if lvl > best_level:
                best_level = lvl
                if first_levelup is None:
                    first_levelup = actions
                # record the win cell + keep going (deeper levels)
            if ok(nf):
                ck = cellify(nf)
                if ck not in archive:
                    archive[ck] = {"traj": list(traj), "visits": 0, "depth": len(traj)}
            f = nf

    return {
        "game": game,
        "levels_reached": int(best_level - start_level),
        "first_levelup_actions": first_levelup,
        "actions": actions,
        "archive_cells": len(archive),
        "iterations": iters,
        "max_depth": max((e["depth"] for e in archive.values()), default=0),
        "executor": "affordance_go_explore",
    }
