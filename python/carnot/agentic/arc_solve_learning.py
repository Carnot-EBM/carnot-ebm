"""ARC-AGI-3 solve LEARNING loop — turn the static reuse-substrate (arc_solver_kit
+ ops/arc_solve_registry.yaml) into something that actively SPEEDS UP the next
game by routing it to the closest SOLVED game's recipe, and that surfaces prior
DEAD-ENDS so we don't repeat them.

Why this exists
---------------
2026-06-16: the kit + registry capture what we learned, but a new game's solver
still started from zero (sc25 took ~10 reverse-engineering layers). The operator
asked: "are we learning from our successes and failures as part of our harness so
as to speed up progress?" This module is the success/failure feedback loop:
`recommend_approach(game)` reads the survey features + the registry, ranks the
solved games by similarity, and hands back the most-applicable proven recipe
(solver module, win-condition, action-model, reusable gotchas) PLUS the relevant
dead-ends to avoid. The agent/planner calls it BEFORE reverse-engineering a new
game, so each solve compounds onto the last instead of restarting.

This is the routing layer; the deeper search-acceleration (a verifier/value head
that prunes the BFS — the north-star verifier-routed-efficiency, cf. exp4071) is
the next loop on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
SURVEY = REPO / "results" / "arc3_win_condition_survey.json"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"

_WIN_KEYWORDS = (
    "align",
    "goal",
    "+1",
    "reflect",
    "pattern",
    "template",
    "rotate",
    "exit",
    "spell",
    "cast",
    "drag",
    "click",
    "move",
    "position",
)


def _action_type(s: str) -> str:
    s = (s or "").lower()
    has_click = "click" in s or "[6]" in s or "6]" in s
    has_kbd = "keyboard" in s or "action1" in s or "1-4" in s or "1-5" in s or "1-6" in s
    if has_click and has_kbd:
        return "mixed"
    if has_click:
        return "click"
    if has_kbd:
        return "keyboard"
    return "unknown"


def _features(entry: dict) -> dict:
    wc = str(entry.get("win_condition_summary", "")).lower()
    return {
        "game": entry.get("game", ""),
        "action_type": _action_type(entry.get("available_actions", "")),
        "spatial": bool(entry.get("is_spatial_planning")),
        "difficulty": str(entry.get("win_difficulty", "")),
        "win_kw": {k for k in _WIN_KEYWORDS if k in wc},
    }


def _survey_features() -> dict[str, dict]:
    d = json.load(open(SURVEY))
    pgs = d["per_game_surveys"]
    entries = list(pgs.values()) if isinstance(pgs, dict) else pgs
    return {e.get("game", ""): _features(e) for e in entries}


def _registry() -> dict:
    return yaml.safe_load(open(REGISTRY))


def _solved_games(reg: dict) -> list[dict]:
    """Games with a usable recipe (reproduced, or provisional-with-mechanics)."""
    return [
        g
        for g in reg.get("games", [])
        if g.get("reproducibility") in ("reproduced", "provisional") and g.get("solver")
    ]


def _similarity(a: dict, b: dict) -> float:
    score = 0.0
    if a["action_type"] == b["action_type"] and a["action_type"] != "unknown":
        score += 3.0
    elif "mixed" in (a["action_type"], b["action_type"]):
        score += 1.0  # mixed partially overlaps either
    if a["spatial"] == b["spatial"]:
        score += 1.5
    if a["difficulty"] and a["difficulty"] == b["difficulty"]:
        score += 0.5
    score += 1.0 * len(a["win_kw"] & b["win_kw"])  # shared win-condition vocabulary
    return score


def recommend_approach(target_game: str, *, mechanic: Optional[str] = None) -> dict:
    """Route a NEW game to the closest proven recipe. Returns the ranked solved
    games with their registry recipe (solver, win-condition, action-model,
    reusable gotchas) + the general gotchas + the matched games' dead-ends.

    The FIRST routing decision is the STRATEGY CLASS (arc_strategy_router): a
    program-editor game routes to the frame-only program-editor model and SKIPS
    the goal-distance heuristic portfolio (which only applies to graph-explore);
    a graph-explore game gets the heuristic policy as before. For an unseen live
    game, pass the frame-only-detected class via `mechanic=` (else it is read
    from the registry's structured `mechanic_class`, defaulting to graph_explore).

    Call this BEFORE reverse-engineering a new game (CLAUDE.md ARC Solve
    Reproducibility + Solver-Reuse Discipline)."""
    from . import arc_strategy_router as strat

    feats = _survey_features()
    reg = _registry()
    strategy = strat.route_for_game(target_game, mechanic=mechanic, reg=reg)
    if target_game not in feats:
        from . import arc_solver_kit as kit

        return {
            "error": f"{target_game} not in survey",
            "strategy": strategy,
            "selected_generic_operators": [
                op.as_dict()
                for op in kit.select_primitive_operators(
                    mechanic_class=strategy.get("routed_mechanic", ""), game=target_game
                )
            ],
            "general_gotchas": reg.get("general_gotchas", []),
        }
    tf = feats[target_game]
    from . import arc_solver_kit as kit

    selected_generic_operators = [
        op.as_dict()
        for op in kit.select_primitive_operators(
            mechanic_class=strategy.get("routed_mechanic", ""),
            action_model=str(tf.get("action_type", "")),
            game=target_game,
        )
    ]
    by_game = {g["game"]: g for g in reg.get("games", [])}
    ranked = []
    for solved in _solved_games(reg):
        gid = solved["game"]
        if gid == target_game or gid not in feats:
            continue
        sim = _similarity(tf, feats[gid])
        ranked.append(
            {
                "game": gid,
                "similarity": round(sim, 2),
                "reproducibility": solved.get("reproducibility"),
                "solver": solved.get("solver"),
                "win_condition": solved.get("win_condition"),
                "action_model": solved.get("action_model"),
                "reusable_gotchas": solved.get("gotchas", []),
            }
        )
    ranked.sort(key=lambda r: r["similarity"], reverse=True)
    # The goal-distance heuristic portfolio only applies to the graph-explore class. For a
    # program-editor (or other non-graph-explore) game it is a category error — surface the strategy's
    # own solver instead, so the agent does not waste a portfolio run that can never win.
    if strategy.get("uses_goal_distance_heuristic"):
        policy = _heuristic_policy()
    else:
        policy = {
            "not_applicable": (
                f"goal-distance heuristics do not apply to the "
                f"{strategy['routed_mechanic']} class; use the strategy solver"
            ),
            "strategy_solver": strategy.get("solver"),
            "search_engine": strategy.get("search_engine"),
            "needs": strategy.get("needs"),
        }
    return {
        "target_game": target_game,
        "target_features": {**tf, "win_kw": sorted(tf["win_kw"])},
        "strategy": strategy,
        "selected_generic_operators": selected_generic_operators,
        "recommended": ranked[:3],
        "heuristic_policy": policy,
        "general_gotchas": reg.get("general_gotchas", []),
        "guidance": (
            "FIRST follow the routed STRATEGY (strategy.solver); for graph_explore, start "
            "from the top-ranked solved game's solver + reuse its action-model and gotchas, "
            "only reverse-engineering the DELTA. Import arc_solver_kit; run the reproduction "
            "gate; append new mechanics/dead-ends to the registry."
        ),
    }


def _heuristic_policy() -> dict:
    """The learned WHEN-to-use-which-heuristic policy (arc_heuristic_select). The choice is
    DATA-DRIVEN per game and cannot be decided from survey features alone (it needs the per-
    action cell-impact, measured from a few transitions) — so the policy is: run the portfolio
    selector ONCE a win-state is available; for a first-ever solve (no target) use pure BFS."""
    return {
        "trained_router": "arc_router.route(features, arc_router.train()) — learned decision tree "
        "(thresholds from ops/arc_router_ledger.json; 8/8 leave-one-out); "
        "predicts approach + explore/exploit by novelty",
        "selector": "arc_heuristic_select.select_and_learn(game, win, transitions, mask_hud=...) "
        "— runs the portfolio, banks the winner, records to the router ledger (online)",
        "feature": "per-action cell impact + bfs_expansions headroom probe",
        "rule": {
            "no_win_state_yet (first solve)": "pure BFS — a goal-distance heuristic needs a target",
            "low search headroom (BFS solves cheaply)": "BFS — no room for a heuristic to help",
            "high cell-impact (>= learned ~36 cells/action)": "misplaced_region_distance (8-conn)",
            "low cell-impact (< learned threshold)": "cell_count_distance (Hamming)",
        },
        "default_if_unmeasured": (
            "region_count — it NEVER regressed across the 8-game validation "
            "and wins the high-impact games; cell_count only when low-impact"
        ),
        "captured": "reuse gap_fills/<game>_goal_distance.py first (no recompute) when present",
    }


if __name__ == "__main__":  # pragma: no cover - manual probe
    import sys

    print(json.dumps(recommend_approach(sys.argv[1] if len(sys.argv) > 1 else "vc33"), indent=2))
