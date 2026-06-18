"""Does a RICHER cross-game value head ROUTE the search to the reachable floor wins?

The diagnostic showed the 6 gap-1 floor games are ROUTING-limited (the winning actions ARE in the
explorer's candidate set, 91-100%), not action-generation-limited -- so a value head that orders the
search well CAN help. v1 (5 scalar features) was inert. This trains v2 (5 scalars + a 6x6 spatial
occupancy map, frame-only) and A/B-routes graph_explore_solve_v2 (its heuristic slot = A*: depth +
weight*heuristic) on each floor game: BFS (weight 0) vs v1-routed vs v2-routed. Honest question: does
ANY config reach L1 where BFS cannot, within the same expansion budget? Zero quota.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import (
    LearnedVerifier, cross_game_features, cross_game_features_v2,
)
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

FLOOR = ["ar25", "cn04", "ka59", "sc25", "sk48", "wa30"]
MAX_EXP = 8000
WEIGHT = 5.0


def main() -> int:
    from arc_cross_game_verifier_train import collect_pooled
    print("== value-routing v2: does a richer value head route the reachable floor wins? ==", flush=True)
    # train v1 and v2 heads on the same pooled corpus
    heads = {}
    for name, feat in [("v1", cross_game_features), ("v2", cross_game_features_v2)]:
        X, y, per_game = collect_pooled(feat)
        heads[name] = LearnedVerifier(feat).fit(X, y)
        print(f"  trained {name}: {len(X)} states, {len(per_game)} games, {len(X[0])} features", flush=True)
    v2_ckpt = REPO / "models" / "arc_verifier_cross_game_v2.json"
    heads["v2"].save(v2_ckpt, meta={"trained_games": list(per_game),
                                    "feature_names": "cross_game_features_v2",
                                    "provenance": "BRIDGE v2: 5 scalars + 6x6 spatial occupancy"})

    arc = kit.offline_arcade()
    rows = []
    for game in FLOOR:
        rec = {"game": game}
        for label, head in [("bfs", None), ("v1_routed", heads["v1"]), ("v2_routed", heads["v2"])]:
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            heur = (lambda fr, h=head: float(h(fr))) if head is not None else None
            t0 = time.time()
            traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=MAX_EXP, max_depth=50,
                                               heuristic=heur, heuristic_weight=WEIGHT)
            rec[label] = {"reached_level": lvl, "solved": bool(traj), "actions": len(traj or []),
                          "secs": round(time.time() - t0)}
            print(f"  {game:5} {label:10} -> L{lvl} {'SOLVED' if traj else ''} [{rec[label]['secs']}s]", flush=True)
        rows.append(rec)

    bfs_solved = {r["game"] for r in rows if r["bfs"]["solved"]}
    v2_solved = {r["game"] for r in rows if r["v2_routed"]["solved"]}
    v1_solved = {r["game"] for r in rows if r["v1_routed"]["solved"]}
    v2_unlocked = sorted(v2_solved - bfs_solved)
    verdict = (f"complete_value_routing_v2_unlocked_{len(v2_unlocked)}_floor_games_{v2_unlocked}"
               if v2_unlocked else
               "complete_value_routing_v2_no_unlock_routing_lever_insufficient_with_cross_game_value")
    out = {
        "experiment": "arc3_value_routing_v2", "honest_verdict": verdict,
        "max_expansions": MAX_EXP, "heuristic_weight": WEIGHT,
        "bfs_solved": sorted(bfs_solved), "v1_routed_solved": sorted(v1_solved),
        "v2_routed_solved": sorted(v2_solved), "v2_unlocked_over_bfs": v2_unlocked,
        "per_game": rows, "inference_substrate": "offline_sim_no_quota_frame_only",
    }
    (REPO / "results" / "arc3_value_routing_v2.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> bfs={sorted(bfs_solved)} v1={sorted(v1_solved)} v2={sorted(v2_solved)} "
          f"| v2 unlocked over bfs: {v2_unlocked}", flush=True)
    print(f"   {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
