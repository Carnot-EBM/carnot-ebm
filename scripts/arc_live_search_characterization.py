#!/usr/bin/env python3
"""Live-search-on-the-scored-env characterization (REQ-ARC-WMTE-5840).

The plan-then-replay policy (REQ-ARC-WMTE-5839) SEARCHES on a separate offline arcade env in __init__, then
replays -- which a HIDDEN scored game does not provide. The live step is to SEARCH on the given (scored) env
itself. The existing live-search engine is `StepwiseExplorer` driven by `run_game` (CarnotAgentPolicy,
force_explore) -- it searches the ONE env the harness gives it via RESET+replay, consuming budget. This
characterizes it with the perception verifier as the routing value_head.

The subtlety this measures: the offline solve of tu93 relied on `branch_mode='fresh_env'` (a BRAND-NEW env
per search node) because tu93's `env.reset()` is NON-IDEMPOTENT (a parity toggle). A LIVE agent on the scored
env has only ONE env and can only RESET it -- it cannot get a fresh env per node. So live single-env
replay-from-reset search on tu93 should be blocked regardless of budget. An IDEMPOTENT-reset ('replay')
navigation game would NOT have this block -- but the public set has none that are clean player->goal nav
games (probed 2026-07-23: lp85/s5i5/vc33 derive no nav pair; only g50t derives (9,8), and its win is not pure
navigation). So this script measures the honest boundary: does more budget rescue live tu93 (no, if the block
is fundamental), and does g50t (the one idempotent candidate) move at all under live verifier-routed search.

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm. verifier_is_oracle: False.
solve_provenance: development_proxy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# (game, budget) -- tu93 gets a LARGE budget to test whether the block is fundamental vs budget-limited.
CASES = [("tu93", 2000), ("g50t", 800)]


def _load_run_game():
    spec = importlib.util.spec_from_file_location(
        "arc_leaderboard_eval", str(ROOT / "scripts" / "arc_leaderboard_eval.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m.run_game


def _make_value_head(player_color: int, target_row: float, target_col: float):
    import numpy as np

    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_entity_hud_perception import _bounded_color_centroid
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

    pc, tr, tc = int(player_color), float(target_row), float(target_col)

    def value(frame, previous_frame=None):  # noqa: ARG001
        try:
            raw = grid_of(frame)
            g = to_logical(raw, detect_cell(raw))
        except Exception:
            return 999.0
        c = _bounded_color_centroid(g, pc, max_area_fraction=0.15)
        if c is None:
            return float(g.shape[0] + g.shape[1])
        return abs(float(c[0]) - tr) + abs(float(c[1]) - tc)

    return value


def _target_from_recon(game: str):
    """Derive (player, goal) + the goal's centroid (logical) for the value_head."""
    import numpy as np

    from carnot.agentic.arc_entity_hud_perception import _as_grid, _bounded_color_centroid, derive_navigation_pair
    from carnot.agentic.arc_perception_navigation import recon

    trans = recon(game, cycles=3)
    pair = derive_navigation_pair(trans)
    if pair is None:
        return None
    player, goal = pair
    g = _as_grid(trans[-1].after)
    c = _bounded_color_centroid(g, goal, max_area_fraction=0.15)
    if c is None:
        return None
    return {"player": int(player), "goal": int(goal), "target_row": float(c[0]), "target_col": float(c[1])}


def main() -> int:
    run_game = _load_run_game()
    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy

    t0 = time.time()
    per_game = []
    for game, budget in CASES:
        row = {"game": game, "budget": budget}
        tgt = _target_from_recon(game)
        row["target"] = tgt
        # baseline default explorer
        b = run_game(game, CarnotAgentPolicy(game, {}, force_explore=True), budget=budget)
        row["baseline"] = {"levels": b.get("levels"), "reached": b.get("reached"), "actions": b.get("actions")}
        # LIVE verifier-routed search on the given env (StepwiseExplorer + perception value_head)
        if tgt is not None:
            vh = _make_value_head(tgt["player"], tgt["target_row"], tgt["target_col"])
            pol = CarnotAgentPolicy(game, {}, force_explore=True, value_head=vh, value_weight=1.0,
                                    search_mode="best_first", navigation_cost_tiebreak=False, lazy_value_top_k=64)
            s = run_game(game, pol, budget=budget)
            row["live_search"] = {"levels": s.get("levels"), "reached": s.get("reached"), "actions": s.get("actions")}
        else:
            row["live_search"] = {"skipped": "no nav pair derived"}
        per_game.append(row)
        print(f"[{game}] budget={budget} target={tgt} | baseline reached {row['baseline'].get('reached')} | "
              f"live_search {row['live_search']}")

    tu = next((r for r in per_game if r["game"] == "tu93"), {})
    art = {
        "experiment": "outer_loop_arc_live_search_characterization",
        "experiment_id": "REQ-ARC-WMTE-5840",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_live_search_characterization.v1",
        "title": "Live verifier-routed search on the SCORED env (StepwiseExplorer + perception verifier, no separate arcade): budget-independence of the tu93 non-idempotent-reset block + g50t.",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5840,
        "methodology_note": "CarnotAgentPolicy(force_explore) searches the ONE env run_game gives it via RESET+replay (no separate offline arcade) -- the true live-search engine. Perception value_head = player->goal Manhattan (derived from a recon on the same game). tu93 gets budget 2000 to test whether its block is fundamental (non-idempotent reset -> no live fresh_env) or budget-limited. Public-game frames for offline dev.",
        "context": "Plan-then-replay (5839) searched a SEPARATE offline arcade twin (unavailable for a hidden game). This tests search ON the given env. Probe 2026-07-23: no clean idempotent-reset nav game in the public set (lp85/s5i5/vc33 derive no pair; only g50t=(9,8), win not pure nav); tu93 is the clean nav game but fresh_env/non-idempotent.",
        "per_game": per_game,
        "tu93_live_reached": tu.get("live_search", {}).get("reached"),
        "duration_s": round(time.time() - t0, 1),
    }
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_live_search_characterization_20260723.json"
    out.write_text(json.dumps(art, indent=2, default=str))
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
