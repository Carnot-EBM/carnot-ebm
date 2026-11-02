#!/usr/bin/env python3
"""Generic AUTONOMOUS navigation solve: no per-game hand-RE at all (REQ-ARC-WMTE-5838).

REQ-ARC-WMTE-5837 proved the perception-derived verifier reproduces tu93's solve, but it REUSED the tu93
hand-built GameAdapter's machinery (action_labels/apply/state_key/branch_mode). This closes the last gap:
solve tu93 with a fully GENERIC navigation adapter -- NO per-game code:

  - action_labels : the directional moves available this frame (1-4) -- generic.
  - apply         : env.step(action) -- generic.
  - state_key     : the full logical-grid bytes -- generic.
  - verifier      : the PERCEPTION-derived player->goal Manhattan (derive_navigation_pair on a recon) --
                    autonomous, never hardcoded, never read from source.
  - branch_mode   : tried both 'replay' and 'fresh_env' (auto-selecting reset-idempotency is the noted
                    generalization; here we report which mode solves).
  - depth_caps    : a generic default schedule.

If a game solves under this generic adapter, it means Carnot can self-discover a navigation game's solve from
its OWN frames with zero per-game hand-RE -- the live self-discovery capability the whole thread was after.

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm. verifier_is_oracle: False.
solve_provenance: development_proxy (offline arcade; method validation on an already-registered game, not a
new solve claim). The (player,goal) colors are DERIVED FROM MOTION, never read from source.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

TARGET_LEVEL = 3
GENERIC_DEPTH_CAPS = {1: 40, 2: 60, 3: 80, 4: 90, 5: 90}


def _generic_nav_adapter(perception_verifier, branch_mode: str):
    """A fully game-agnostic navigation GameAdapter: directional moves, env.step, full-grid state key, and a
    PERCEPTION-derived verifier. No per-game code."""
    import json as _json

    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_game_adapters import GameAdapter

    def action_labels(env, frame=None, path=None):
        av = list(getattr(frame, "available_actions", []) or []) if frame is not None else []
        moves = [a for a in av if a in (1, 2, 3, 4)] or [1, 2, 3, 4]
        return [_json.dumps({"action": int(a)}) for a in moves]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, _json.loads(label)["action"]))

    def state_key(game, frame=None):
        if frame is None:
            return None
        from carnot.agentic.arc_agi3_world_model import grid_of

        g = grid_of(frame)
        if getattr(g, "ndim", 2) == 1:
            s = int(round(g.size**0.5))
            if s * s == g.size:
                g = g.reshape(s, s)
        return g.tobytes()

    return GameAdapter(
        game="__generic_nav__",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=perception_verifier,
        warmup_label=None,
        depth_caps=dict(GENERIC_DEPTH_CAPS),
        branch_mode=branch_mode,
    )


def _make_perception_verifier(player_color: int, goal_color: int):
    import numpy as np

    from carnot.agentic.arc_agi3_world_model import grid_of

    def _grid2d(frame):
        g = grid_of(frame)
        if getattr(g, "ndim", 2) == 1:
            s = int(round(g.size**0.5))
            if s * s == g.size:
                g = g.reshape(s, s)
        return g

    def _cent(g, c):
        ys, xs = np.where(g == c)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    def verifier(game, frame=None):  # noqa: ARG001
        if frame is None:
            return 1000.0
        g = _grid2d(frame)
        p, t = _cent(g, int(player_color)), _cent(g, int(goal_color))
        if p is None or t is None:
            return 1000.0
        return abs(p[0] - t[0]) + abs(p[1] - t[1])

    return verifier


def _recon_pair(game: str):
    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import _available_action_ids, _game_action, _game_over
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_entity_hud_perception import Transition, derive_navigation_pair
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    cell = detect_cell(grid_of(frame))
    trans = []
    for a in [4, 2, 3, 1, 4, 2, 3, 1, 4, 4, 2, 2, 1, 1, 3, 3] * 3:
        if _game_over(frame):
            break
        av = set(_available_action_ids(frame) or [1, 2, 3, 4])
        if a not in av:
            a = next(iter(av), 1)
        before = to_logical(grid_of(frame), cell)
        frame = env.step(_game_action(GameAction, a))
        after = to_logical(grid_of(frame), cell)
        trans.append(Transition(before=before, action=a, after=after))
    return derive_navigation_pair(trans), len(trans)


def _solve(game: str, ad, target_level: int = TARGET_LEVEL):
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        game, ad.action_labels, ad.apply, ad.state_key,
        warmup_label=ad.warmup_label, verifier=ad.hand_verifier,
        branch_mode=ad.branch_mode,
    )
    f = solver._replay(env, [])
    cur = kit.frame_level(f)
    full: list = []
    for lvl in range(cur + 1, target_level + 1):
        path, _nodes = solver.solve_level(env, cur, full, ad.depth_caps.get(lvl, 90))
        if path is None:
            break
        search_reached = kit.frame_level(solver.last_frame)
        f = solver._replay(env, full + path)
        cur = max(kit.frame_level(f), search_reached)
        full += path
    gate = kit.reproduce(game, full, ad.apply, warmup_label=ad.warmup_label, claimed_level=cur)
    reproduced = bool(gate.get("reproduced")) if isinstance(gate, dict) else False
    reached = int(gate.get("reached_level", cur)) if isinstance(gate, dict) else int(cur)
    return {"search_reached": int(cur), "gate_reproduced": reproduced,
            "reproduced_level": reached if reproduced else 0, "path_len": len(full)}


def main() -> int:
    game = "tu93"
    t0 = time.time()
    pair, n_recon = _recon_pair(game)
    result = {"game": game, "perception_pair": None if pair is None else list(pair), "n_recon": n_recon,
              "hand_re_pair": [9, 14], "no_hand_adapter": True}
    if pair is None:
        result["error"] = "no navigation pair derived"
    else:
        vh = _make_perception_verifier(pair[0], pair[1])
        for mode in ("fresh_env", "replay"):
            ad = _generic_nav_adapter(vh, mode)
            result[f"generic_{mode}"] = _solve(game, ad)
            r = result[f"generic_{mode}"]
            print(f"[{game}] GENERIC nav ({mode}) pair={pair}: reached L{r['search_reached']} "
                  f"reproduced={r['gate_reproduced']} (L{r['reproduced_level']}) path={r['path_len']}")

    best = max((result.get(f"generic_{m}", {}).get("reproduced_level", 0) for m in ("fresh_env", "replay")),
               default=0)
    art = {
        "experiment": "outer_loop_arc_generic_navigation_solve",
        "experiment_id": "REQ-ARC-WMTE-5838",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_generic_navigation_solve.v1",
        "title": "Solve tu93 with a FULLY GENERIC navigation adapter (no per-game hand-RE) + a perception-derived verifier: autonomous self-discovery of a navigation solve.",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "purpose": "method_validation_autonomous_generic_navigation",
        "not_a_new_solve_claim": "tu93 L3 is already in the registry; this validates that a GENERIC adapter (no per-game code) + a perception-derived verifier reproduces it. No registry change.",
        "random_seed": 5838,
        "methodology_note": "GameAdapter built from GENERIC functions only (directional moves, env.step, full-grid state key) + a perception-derived player->goal verifier whose colors come from derive_navigation_pair on a recon (motion only, never hardcoded, never source). branch_mode tried both. Public-game frames stepped for offline dev.",
        "result": result,
        "best_reproduced_level": best,
        "offline_reproduced": best > 0,
        "reproduced_levels": best,
        "honest_verdict": (f"complete_success_generic_navigation_adapter_reproduces_tu93_L{best}_no_hand_re"
                           if best > 0 else "complete_generic_adapter_did_not_reproduce_investigate"),
        "duration_s": round(time.time() - t0, 1),
    }
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_generic_navigation_solve_20260723.json"
    out.write_text(json.dumps(art, indent=2, default=str))
    print(f"BEST reproduced level (generic, no hand-RE): L{best}")
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
