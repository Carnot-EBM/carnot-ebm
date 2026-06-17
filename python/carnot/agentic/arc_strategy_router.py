"""STRATEGY-class router for ARC-AGI-3 — the layer ABOVE the goal-distance heuristic router.

Why this exists
---------------
`arc_router` learns WHICH goal-distance heuristic (bfs / cell_count / region_count) wins a game,
but that whole question only makes sense for the GRAPH-EXPLORE / path-route class, where a click or
key moves an avatar toward a goal. The tn36 climb surfaced a different mechanic CLASS — the
PROGRAM-EDITOR (a multi-slot move-program edited via a bit-button palette, executed by a run
trigger). Running the goal-distance portfolio on a program-editor game is a category error: blind
clicks never drive the 5-attribute alignment, so every heuristic NO-ADVANCEs (confirmed: tn36 runs
graph_explore_solve_v2 clean but never wins).

So routing needs a FIRST decision — the STRATEGY CLASS — before the heuristic decision. This module
is that decision. It maps a detected mechanic class to its solving STRATEGY (the solver entrypoint +
search engine + applicability features), and SHORT-CIRCUITS the goal-distance heuristic for classes
where it does not apply. The program-editor model (`scripts/arc3_frame_induction.py`: frame-only
`induce` → `induce_editor_layout` → `find_run_button` → `frame_only_winner_search`) is wired in here
as the `program_editor` strategy.

How the mechanic is detected (frame-only, the live requirement)
---------------------------------------------------------------
- KNOWN game: read the structured `mechanic_class` recorded in `ops/arc_solve_registry.yaml`.
- UNSEEN live game: the caller runs the frame-only classifier `arc3_frame_induction.induce(probe(...))`
  (uses ONLY `grid_of(frame)` + `_levels_completed(frame)`, zero internal state) and injects the
  result via `mechanic=`. The package stays free of the arcade/script dependency; the routing brain
  lives here, the frame probe lives at the I/O edge.

Honest scope (per the design note arc-live-generalization-gap-2026-06-17.md): `program_editor` and
`graph_explore` are WIRED (a real solver exists). `checkpoint_multirun` and `timed_trap_aware` are
DECLARED from the discovered taxonomy but NOT yet frame-only-wired — `route_strategy` returns them
with `wired: False` so a caller knows the class is recognised but its frame-only model is still a
build. The program-editor strategy itself is BLIND-SEARCH-limited frame-only (no graded feedback —
GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK); the durable unlock is an offline-trained transition
verifier, surfaced here as the strategy's `needs` note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"

# The default class: clicks/keys steer an avatar toward a goal (the graph-explore / path-route games
# the goal-distance heuristic router was built for). Anything not positively detected as another
# class routes here.
DEFAULT_MECHANIC = "graph_explore"

# Strategy classes, in PRIORITY order (first applicable wins). Each declares the observable feature
# that selects it, the solver entrypoint, the search engine it runs, and whether a real solver is
# wired yet. `uses_goal_distance_heuristic` says whether the arc_router heuristic decision applies
# (True only for graph_explore — for the others it is a category error and is short-circuited).
STRATEGY_CLASSES: list[dict] = [
    {
        "name": "program_editor",
        "mechanic": "program_editor",
        "wired": True,
        "uses_goal_distance_heuristic": False,
        "applies_features": (
            "a dense contiguous palette of small local-toggle glyph buttons + a "
            "separate run trigger (arc3_frame_induction.induce → 'program_editor')"
        ),
        "solver": (
            "scripts/arc3_frame_induction.py — induce_editor_layout + find_run_button + "
            "frame_only_winner_search (frame-only; zero internal state)"
        ),
        "search_engine": (
            "blind program-space search via the BINARY win signal — uniform-code prior "
            "+ bounded product fallback"
        ),
        "needs": (
            "offline-trained transition verifier ('editor code → object transform') to replace "
            "blind search; frame-only gives NO graded feedback "
            "(GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK)"
        ),
    },
    {
        "name": "checkpoint_multirun",
        "mechanic": "checkpoint_multirun",
        "wired": False,  # internal-state solver exists; frame-only port pending
        "uses_goal_distance_heuristic": False,
        "applies_features": "a region where ending a run advances the object's base (progress carries across runs)",
        "solver": "arc3_tn36_offline_solver._multirun_plan (internal-state; frame-only port pending)",
        "search_engine": "waypoint BFS across checkpoints with a box-collision model",
        "needs": "frame-only checkpoint detection (a cell that, when the run ends on it, persists base advance)",
    },
    {
        "name": "timed_trap_aware",
        "mechanic": "timed_trap_aware",
        "wired": False,  # tn36 L7 timed path-planner not yet built
        "uses_goal_distance_heuristic": False,
        "applies_features": "obstacle cells that toggle visibility on a fixed move cadence (lethal on contact)",
        "solver": "PENDING — timed path-planner (tn36 L7 not yet solved; no fabrication)",
        "search_engine": "timed-state planner over (position, run-move-index, hazard-visibility-parity)",
        "needs": "frame-only periodic-visibility detection + a timed-state search",
    },
    {
        "name": "graph_explore",
        "mechanic": "graph_explore",
        "wired": True,
        "uses_goal_distance_heuristic": True,  # the arc_router heuristic decision applies HERE only
        "applies_features": "clicks/keys move an avatar (large board change) toward a goal; no editor palette",
        "solver": "arc_graph_explore.graph_explore_solve_v2 + arc_heuristic_select portfolio",
        "search_engine": (
            "BFS / best-first over the induced frame graph; goal-distance heuristic from "
            "arc_router.route(features, arc_router.train())"
        ),
        "needs": None,
    },
]

_BY_MECHANIC = {s["mechanic"]: s for s in STRATEGY_CLASSES}


def known_mechanics() -> list[str]:
    """The recognised mechanic classes (the discovered ARC-AGI-3 taxonomy)."""
    return [s["mechanic"] for s in STRATEGY_CLASSES]


def _load_registry(reg: Optional[dict] = None) -> dict:
    if reg is not None:
        return reg
    return yaml.safe_load(REGISTRY.read_text()) if REGISTRY.exists() else {"games": []}


def detect_mechanic(
    game: str, *, mechanic: Optional[str] = None, reg: Optional[dict] = None
) -> str:
    """Return the mechanic CLASS for `game`. Precedence:
      1. an INJECTED `mechanic` (the frame-only classifier's verdict, for an unseen live game),
      2. the structured `mechanic_class` recorded in the registry (a known game),
      3. DEFAULT_MECHANIC (graph_explore) — the goal-distance class the portfolio handles.
    An injected/recorded class that is not in the taxonomy falls back to the default (honest: we
    only route to strategies we recognise)."""
    if mechanic:
        return mechanic if mechanic in _BY_MECHANIC else DEFAULT_MECHANIC
    reg = _load_registry(reg)
    for g in reg.get("games", []):
        if g.get("game") == game:
            recorded = g.get("mechanic_class")
            return recorded if recorded in _BY_MECHANIC else DEFAULT_MECHANIC
    return DEFAULT_MECHANIC


def route_strategy(mechanic: str) -> dict:
    """Map a mechanic class to its solving STRATEGY. Returns a copy of the strategy descriptor plus
    `mechanic` echoed and a `reason`. Unknown mechanic → the default graph_explore strategy."""
    strat = _BY_MECHANIC.get(mechanic, _BY_MECHANIC[DEFAULT_MECHANIC])
    out = dict(strat)
    out["routed_mechanic"] = mechanic
    if mechanic not in _BY_MECHANIC:
        out["reason"] = (
            f"mechanic {mechanic!r} not in the taxonomy → default {DEFAULT_MECHANIC} "
            f"(goal-distance portfolio)"
        )
    elif not strat["wired"]:
        out["reason"] = (
            f"{mechanic} recognised but its frame-only solver is NOT yet wired "
            f"(needs: {strat['needs']})"
        )
    else:
        out["reason"] = f"{mechanic} → {strat['solver'].split(' —')[0].split(' +')[0]}"
    return out


def route_for_game(
    game: str, *, mechanic: Optional[str] = None, reg: Optional[dict] = None
) -> dict:
    """Convenience: detect the mechanic for `game` (registry or injected frame-only verdict) and route
    it to its strategy. This is the single entrypoint the solve loop calls to pick a STRATEGY before
    any heuristic decision."""
    m = detect_mechanic(game, mechanic=mechanic, reg=reg)
    out = route_strategy(m)
    out["game"] = game
    return out
