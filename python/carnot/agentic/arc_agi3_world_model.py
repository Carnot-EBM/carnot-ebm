"""ARC-AGI-3 world-model substrate: the persistent per-game GameGraph (M0 of the agent plan).

WHY (docs/research-notes/arc-agi3-agent-research-plan.md): every approach that beats 0 on ARC-AGI-3
replaces "LLM picks the next action per step" with a PERSISTENT STRUCTURE accumulated across the
horizon. The SOTA-3rd training-free method (arXiv:2512.24156) is exactly a directed state-graph
explorer; the SOTA executable-world-model (arXiv:2605.05138) keeps an append-only transition store
it VERIFIES against. This module is the shared structure BOTH families read/write:

  - nodes        : frame-hash -> {levels_completed, available_actions, grid_shape, visits}
  - edges        : (frame_hash, action_key) -> {to, n_changed, level_delta, game_over, count}
  - transition_store : append-only (s, action_key, s', delta) for the Family-B synthesizer (later)
  - deadly       : (frame_hash, action_key) that caused GAME_OVER -> never re-selected
  - persistence  : JSON per game_id so episode RESETS become cheap re-attempts (cross-episode memory)

Perception (compute_grid_delta, objects) is DETERMINISTIC numpy — never an LLM (adopting the
perception-bottleneck mitigation: deltas are computed, not described). An action_key is a small
hashable: (a,) for keyboard actions, (6, x, y) for a click. No rule induction lives here — this is
the substrate the explorer (M1, Family-A) and later the DSL synthesizer (M1b/M2, Family-B) operate on.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np


def grid_of(frame: Any) -> np.ndarray:
    """Extract the 2D color grid (uint8) from a FrameData/raw frame; last frame of any stack."""
    arr = np.array(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr.astype(np.int16)


def frame_hash(grid: np.ndarray) -> str:
    """Stable hash of a grid's contents — the node identity."""
    return hashlib.sha1(np.ascontiguousarray(grid.astype(np.uint8)).tobytes()
                        + bytes(grid.shape)).hexdigest()[:16]


def compute_grid_delta(prev: np.ndarray, nxt: np.ndarray) -> dict:
    """Deterministic changed-cell set between two grids (the load-bearing perception primitive).
    Returns n_changed (0 = no-effect action), the changed-cell coordinate list, and the
    (old->new) color transitions, so the explorer can score effect and the synthesizer can induce."""
    if prev.shape != nxt.shape:
        return {"n_changed": -1, "shape_changed": True, "cells": [],
                "old_shape": list(prev.shape), "new_shape": list(nxt.shape)}
    diff = prev != nxt
    ys, xs = np.where(diff)
    cells = [(int(y), int(x)) for y, x in zip(ys, xs)]
    transitions = [(int(prev[y, x]), int(nxt[y, x])) for y, x in zip(ys, xs)]
    return {"n_changed": int(diff.sum()), "shape_changed": False, "cells": cells,
            "transitions": transitions}


def objects(grid: np.ndarray) -> list[tuple[int, int]]:
    """Connected non-background components -> click-target centroids (the action-pruner's candidates).
    4-neighbour flood fill, pure numpy/python (no scipy). Background = most common color."""
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[counts.argmax()])
    mask = grid != bg
    if not mask.any():
        return []
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    targets = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                cy = sum(c[0] for c in cells) // len(cells)
                cx = sum(c[1] for c in cells) // len(cells)
                targets.append((cy, cx))
    return targets


def action_key(a_int: int, data: Optional[dict]) -> tuple:
    """Small hashable action identity: (a,) for keyboard, (6, x, y) for a click."""
    if a_int == 6 and data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(a_int),)


class GameGraph:
    """Persistent per-game directed state-action graph + transition store + deadly-action set."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}          # key "fh|akey" -> edge dict
        self.deadly: set[str] = set()             # "fh|akey"
        self.transition_store: list[dict] = []    # append-only (for Family-B synthesizer)
        self.max_levels = 0

    @staticmethod
    def _ek(fh: str, akey: tuple) -> str:
        return f"{fh}|{','.join(map(str, akey))}"

    def see_node(self, fh: str, frame: Any) -> None:
        n = self.nodes.get(fh)
        if n is None:
            self.nodes[fh] = {"levels": int(getattr(frame, "levels_completed", 0) or 0),
                              "available_actions": list(getattr(frame, "available_actions", []) or []),
                              "visits": 1}
        else:
            n["visits"] += 1

    def record(self, s_fh: str, akey: tuple, s2_fh: str, delta: dict,
               level_delta: int, game_over: bool) -> None:
        ek = self._ek(s_fh, akey)
        e = self.edges.get(ek)
        if e is None:
            self.edges[ek] = {"from": s_fh, "akey": list(akey), "to": s2_fh,
                              "n_changed": delta.get("n_changed", 0), "level_delta": level_delta,
                              "game_over": game_over, "count": 1}
        else:
            e["count"] += 1
            e["to"] = s2_fh  # last observed (Family-B will detect non-determinism via the store)
        if game_over:
            self.deadly.add(ek)
        self.transition_store.append({"s": s_fh, "akey": list(akey), "s2": s2_fh,
                                      "n_changed": delta.get("n_changed", 0),
                                      "cells": delta.get("cells", [])[:64],
                                      "level_delta": level_delta, "game_over": game_over})

    def tried(self, fh: str, akey: tuple) -> bool:
        return self._ek(fh, akey) in self.edges

    def is_deadly(self, fh: str, akey: tuple) -> bool:
        return self._ek(fh, akey) in self.deadly

    def untested(self, fh: str, candidate_akeys: list[tuple]) -> list[tuple]:
        """Candidate actions from fh that have NOT been tried and are not known-deadly."""
        return [k for k in candidate_akeys if not self.tried(fh, k) and not self.is_deadly(fh, k)]

    def frontier_states(self, candidate_fn) -> set[str]:
        """Known states that still have at least one untested, non-deadly candidate action."""
        out = set()
        for fh, n in self.nodes.items():
            if self.untested(fh, candidate_fn(fh, n)):
                out.add(fh)
        return out

    def shortest_path_action(self, start_fh: str, goals: set[str]) -> Optional[tuple]:
        """BFS over recorded non-deadly edges from start_fh to the nearest goal state; return the
        FIRST action_key on that path (so the explorer can navigate back to a frontier). None if
        unreachable. Edges are the observed transitions (deterministic enough for navigation)."""
        if start_fh in goals:
            return None
        adj: dict[str, list[tuple[str, tuple]]] = {}
        for ek, e in self.edges.items():
            if ek in self.deadly or e.get("game_over"):
                continue
            adj.setdefault(e["from"], []).append((e["to"], tuple(e["akey"])))
        q = deque([(start_fh, None)])
        seen = {start_fh}
        while q:
            cur, first = q.popleft()
            for nxt, akey in adj.get(cur, []):
                if nxt in seen:
                    continue
                f = first if first is not None else akey
                if nxt in goals:
                    return f
                seen.add(nxt)
                q.append((nxt, f))
        return None

    def to_json(self) -> dict:
        return {"game_id": self.game_id, "nodes": self.nodes, "edges": self.edges,
                "deadly": sorted(self.deadly), "n_transitions": len(self.transition_store),
                "max_levels": self.max_levels}

    def persist(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n", "utf-8")

    @classmethod
    def load(cls, path: Path) -> "GameGraph":
        d = json.loads(path.read_text())
        g = cls(d["game_id"])
        g.nodes, g.edges, g.deadly = d["nodes"], d["edges"], set(d["deadly"])
        g.max_levels = d.get("max_levels", 0)
        return g
