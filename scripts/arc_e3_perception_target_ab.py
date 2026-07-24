#!/usr/bin/env python3
"""E3 search A/B: does a PERCEPTION-DERIVED TARGET routed into the value-head-A* search convert to discovery?
(REQ-ARC-WMTE-5835)

The discovery null (REQ-ARC-WMTE-5834) showed the leaderboard-winner architecture (greedy-direct, no search)
cannot EXECUTE even a correct goal -- 0 levels, both arms. The earned redirect: feed the perception fix into
Carnot's OWN verifier-routed search (E3), which CAN navigate to a named target. This is that experiment.

Both arms use the LLM-FREE pure explorer (`CarnotAgentPolicy(game, {}, force_explore=True)` -- the tier-1
graph_explore search make_carnot_agent runs on unseen games), scored by the real `run_game`:
  - baseline  : default explorer (depth_first_ride, value_weight 0 -> pure BFS).
  - perception: value_head = player->TARGET Manhattan distance, where the target is the nearest non-HUD
    object to the detected player from `perceive_entities` on a short recon of the game's own frames; routed
    best_first with navigation_cost_tiebreak=False + value_weight=1 so the value head actually ROUTES the
    search toward the target (per the E3-injection map). This is the A* routing that "unlocked cn04" before,
    but pointed at a PERCEPTION-DETECTED specific target rather than a generic order-prior (which was REFUTED
    2026-06-21 for fixed-direction misrouting).

Everything is in LOGICAL grid space (recon detectors AND the value-head both to_logical the frame), so the
target coords and the per-frame player centroid live in the SAME space. hud_mask is intentionally NOT
injected here (its raw-vs-logical shape guard is a separate coord risk; E3's auto_hud_mask handles dedup) --
this isolates the value-head routing lever.

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm (pure env-stepping + search;
no GGUF load, no CUDA). Public-game frames stepped for offline dev validation (authorized), never in the
hidden submission. verifier_is_oracle: False (the value head is a perception-derived heuristic, not the win
predicate).
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
sys.path.insert(0, str(ROOT))

GAMES = ["bp35", "lf52", "sc25", "ls20", "tu93", "cn04", "r11l", "ft09"]


def _load_run_game():
    spec = importlib.util.spec_from_file_location(
        "arc_leaderboard_eval", str(ROOT / "scripts" / "arc_leaderboard_eval.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m.run_game


def _make_value_head(player_color: int, target_row: float, target_col: float):
    """A player->target Manhattan-distance heuristic in LOGICAL grid space (lower = closer). Recomputes the
    player centroid from EACH frame (the player moves; the target is fixed for this level). Returns a large
    cost when the player is not visible (merged / off-frame)."""
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_entity_hud_perception import _bounded_color_centroid
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

    pc, tr, tc = int(player_color), float(target_row), float(target_col)

    def value(frame, previous_frame=None):  # noqa: ARG001 -- E3 may pass previous_frame
        try:
            raw = grid_of(frame)
            g = to_logical(raw, detect_cell(raw))
        except Exception:
            return 999.0
        c = _bounded_color_centroid(g, pc, max_area_fraction=0.15)
        if c is None:
            return float(g.shape[0] + g.shape[1])  # player not visible -> max cost
        return abs(float(c[0]) - tr) + abs(float(c[1]) - tc)

    return value


def _recon_target(game: str):
    """Short scripted recon of the game's own frames -> (player_color, target_row, target_col) or None."""
    from scripts.arc_perception_detectors_realframe_gate import _gather

    from carnot.agentic.arc_entity_hud_perception import _candidate_targets, perceive_entities

    trans, _shape, _lvl = _gather(game)
    if not trans:
        return None
    percept = perceive_entities(trans[-1].after, trans)
    targets, player = _candidate_targets(percept)
    player_color = (player or {}).get("color") if player else (percept.mover.color if percept.mover else None)
    if player_color is None or not targets:
        return None
    t = targets[0]
    return {"player_color": int(player_color), "target_row": int(t["row"]), "target_col": int(t["col"]),
            "n_recon_transitions": len(trans)}


def main() -> int:
    run_game = _load_run_game()
    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy

    budget = 250
    t0 = time.time()
    per_game = []
    for game in GAMES:
        row = {"game": game}
        # baseline: default explorer
        try:
            b = run_game(game, CarnotAgentPolicy(game, {}, force_explore=True), budget=budget)
            row["baseline"] = {"levels": b.get("levels"), "reached": b.get("reached"), "actions": b.get("actions")}
        except Exception as exc:  # noqa: BLE001
            row["baseline"] = {"error": repr(exc)[:200]}
        # perception-routed
        tgt = None
        try:
            tgt = _recon_target(game)
        except Exception as exc:  # noqa: BLE001
            row["recon_error"] = repr(exc)[:160]
        if tgt is None:
            row["perception"] = {"skipped": "no player+target detected in recon"}
        else:
            row["target"] = tgt
            try:
                vh = _make_value_head(tgt["player_color"], tgt["target_row"], tgt["target_col"])
                policy = CarnotAgentPolicy(
                    game, {}, force_explore=True, value_head=vh, value_weight=1.0,
                    search_mode="best_first", navigation_cost_tiebreak=False, lazy_value_top_k=64,
                )
                p = run_game(game, policy, budget=budget)
                row["perception"] = {"levels": p.get("levels"), "reached": p.get("reached"), "actions": p.get("actions")}
            except Exception as exc:  # noqa: BLE001
                row["perception"] = {"error": repr(exc)[:200]}
        per_game.append(row)
        print(f"[{game}] baseline={row['baseline'].get('levels', row['baseline'].get('error'))} "
              f"perception={row['perception'].get('levels', row['perception'].get('skipped', row['perception'].get('error')))} "
              f"target={row.get('target')}")

    def _tot(k: str) -> int:
        return sum(int((r.get(k, {}) or {}).get("levels", 0) or 0) for r in per_game)

    art = {
        "experiment": "outer_loop_arc_e3_perception_target_ab",
        "experiment_id": "REQ-ARC-WMTE-5835",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_e3_perception_target_ab.v1",
        "title": "Does a perception-derived target routed into Carnot's value-head-A* explorer convert to discovery? A/B vs the default explorer, 8 games, no LLM.",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5835,
        "reproducibility_checksum": "",
        "action_budget": budget,
        "methodology_note": "Both arms = CarnotAgentPolicy(force_explore=True), the LLM-free tier-1 explorer, scored by arc_leaderboard_eval.run_game. Perception arm sets value_head = player->target Manhattan distance (target = nearest non-HUD object to the detected player from a short perceive_entities recon), best_first + navigation_cost_tiebreak=False + value_weight=1 so the value head routes. All in logical grid space. Public-game frames stepped for offline dev validation (authorized), never in the hidden submission.",
        "baseline_context": "The greedy-direct discovery A/B (REQ-ARC-WMTE-5834) was 0/8 both arms -- execution was the wall. This tests whether search-based routing toward the perception target breaks that.",
        "totals": {"baseline_levels": _tot("baseline"), "perception_levels": _tot("perception")},
        "per_game": per_game,
        "duration_s": round(time.time() - t0, 1),
    }
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_e3_perception_target_ab_20260723.json"
    out.write_text(json.dumps(art, indent=2))
    print(f"TOTALS: baseline={art['totals']['baseline_levels']} perception={art['totals']['perception_levels']}")
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
