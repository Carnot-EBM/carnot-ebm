#!/usr/bin/env python3
"""GAP-4891 STAGE 2: does the relational goal-energy GUIDE the search to self-discover the next level?
(operator-directed 2026-06-28 "build the relational-goal-energy stage-2 lever").

Stage 1 showed induce_goal_energy_relational SEPARATES (energy 0 on the win, >0 on every non-win) on
cd82/sk48/sp80. Separation is necessary but NOT sufficient: it orders the frontier, but the SEARCH must
actually reach the next level. Stage 2 wires the relational energy into graph_explore_solve_v2's
goal_energy hook and tests whether it banks a NEW reproduction-gated level via live self-discovery --
vs a BFS-only ablation (the energy only COUNTS if it reaches the level where blind BFS does not, or with
fewer expansions; the goal-induction doctrine's mandatory control).

Design (from the wf_b77c3b40-f11 map, all reused parts):
- prefix = the game's banked L1 self-discovery seed (arc_explore_trajectory_<game>.json, graph_explore
  format), rooting the search at the level it reaches (incremental-progress lever).
- goal_energy = induce_goal_energy_relational(win, non_wins) on grids captured by replaying the prefix
  (win = the prefix's level-completion frame; the relational offset is LEVEL-INVARIANT -- the canvas->
  target screen layout is constant -- so an energy induced at level k detects level k+1's goal too).
- graph_explore_solve_v2(env, start_level=prefix_level, prefix=seed, goal_energy=<wrapped grid_of>) vs
  the same call with goal_energy=None (BFS ablation). reproduction-gate both with kit.reproduce.
adapter-FREE, offline, reproduction-gated -> solve_provenance=live_agent_self_discovery.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_goal_induction import induce_goal_energy_relational  # noqa: E402
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels  # noqa: E402

GAME = sys.argv[1] if len(sys.argv) > 1 else "cd82"
MAX_EXPANSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
SEED = 20260628


def _apply(env, label, frame):
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, step["action"]), data=step.get("data"))


def _g(frame):
    try:
        return np.asarray(grid_of(frame))
    except Exception:
        return None


def main() -> int:
    started = time.time()
    np.random.seed(SEED)
    seed_path = REPO / "results" / f"arc_explore_trajectory_{GAME}.json"
    if not seed_path.exists():
        print(f"BLOCKED: no L1 seed for {GAME}"); return 0
    seed = json.loads(seed_path.read_text())
    prefix = seed.get("trajectory") or []
    prefix_level = int(seed.get("reached_level", 0))
    if not prefix or prefix_level < 1:
        print(f"BLOCKED: seed for {GAME} reaches level {prefix_level}"); return 0

    # --- capture win/non-win grids by replaying the prefix; induce the relational energy ---
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    labels = [json.dumps(s) for s in prefix]
    win_grids, non_win_grids, prev = [], [], 0
    f0 = env.reset() if hasattr(env, "reset") else None
    if f0 is not None:
        g0 = _g(f0)
        if g0 is not None:
            non_win_grids.append(g0)
    # replay prefix step-by-step, capturing the level-completion frame as the win exemplar
    cur = arc.make(GAME, scorecard_id=arc.open_scorecard())
    cur.reset()
    for k, lab in enumerate(labels):
        f = _apply(cur, lab, None)
        lv = kit.frame_level(f)
        g = _g(f)
        if g is None:
            continue
        if lv > prev:
            win_grids.append(g); prev = lv
        else:
            non_win_grids.append(g)

    def _uniq(gs):
        seen, out = set(), []
        for g in gs:
            h = g.tobytes()
            if h not in seen:
                seen.add(h); out.append(g)
        return out
    win_grids, non_win_grids = _uniq(win_grids), _uniq(non_win_grids)

    energy_grid = induce_goal_energy_relational(win_grids[0] if win_grids else None, non_win_grids)
    induce_fired = energy_grid is not None
    # wrap grid-based energy for graph_explore (which passes a FRAME)
    def goal_energy(frame, _e=energy_grid):
        g = _g(frame)
        return float(_e(g)) if (g is not None and _e is not None) else 0.0

    print(f"[{GAME}] prefix_level={prefix_level} win={len(win_grids)} neg={len(non_win_grids)} "
          f"induce_fired={induce_fired}", flush=True)

    def _run(use_energy: bool):
        e = arc.make(GAME, scorecard_id=arc.open_scorecard())
        stats: dict = {}
        traj, lvl = graph_explore_solve_v2(
            e, prefix_level, max_expansions=MAX_EXPANSIONS, prefix=list(prefix),
            goal_energy=(goal_energy if use_energy else None), stats=stats,
        )
        reproduced, reached = False, lvl
        if traj is not None and lvl > prefix_level:
            gate = kit.reproduce(GAME, trajectory_labels(traj), _apply, claimed_level=lvl)
            reproduced = bool(gate.get("reproduced"))
            reached = int(gate.get("reached_level", lvl))
        return {"reached_level": int(lvl), "reproduced": reproduced, "gate_reached": reached,
                "states_expanded": int(stats.get("states_expanded", stats.get("expansions", 0) or 0)),
                "traj_len": (len(traj) if traj else 0)}

    treat = _run(True) if induce_fired else {"reached_level": prefix_level, "reproduced": False,
                                             "states_expanded": 0, "traj_len": 0, "skipped": "no_energy"}
    print(f"  [energy] {treat}", flush=True)
    ctrl = _run(False)
    print(f"  [bfs]    {ctrl}", flush=True)

    # the energy COUNTS iff it banks a NEW reproduced level where BFS does not, OR same level fewer expansions
    new_level_energy = bool(treat["reproduced"] and treat["gate_reached"] > prefix_level)
    new_level_bfs = bool(ctrl["reproduced"] and ctrl["gate_reached"] > prefix_level)
    energy_unlocks = bool(new_level_energy and not new_level_bfs)
    energy_more_efficient = bool(
        new_level_energy and new_level_bfs
        and treat["states_expanded"] > 0 and ctrl["states_expanded"] > 0
        and treat["states_expanded"] < ctrl["states_expanded"]
    )
    if not induce_fired:
        verdict = f"complete_stage2_no_relational_energy_for_{GAME}"
    elif energy_unlocks:
        verdict = (f"success_stage2_relational_energy_UNLOCKS_new_level_{GAME}_L{treat['gate_reached']}"
                   f"_bfs_stuck_at_L{ctrl['gate_reached']}")
    elif energy_more_efficient:
        verdict = (f"success_stage2_relational_energy_more_efficient_{GAME}_states_{treat['states_expanded']}"
                   f"_vs_bfs_{ctrl['states_expanded']}")
    elif new_level_energy and new_level_bfs:
        verdict = (f"complete_stage2_both_reach_new_level_no_efficiency_edge_{GAME}"
                   f"_energy_{treat['states_expanded']}_bfs_{ctrl['states_expanded']}")
    else:
        verdict = (f"complete_stage2_neither_banks_new_level_{GAME}_energy_L{treat['gate_reached']}"
                   f"_bfs_L{ctrl['gate_reached']}_search_wall_not_goal_energy")
    art = {
        "experiment": "arc_relational_goal_energy_stage2",
        "schema": "carnot.arc_relational_goal_energy_stage2.v1",
        "honest_verdict": verdict,
        "question": ("does the GAP-4891 relational goal-energy GUIDE graph_explore to self-discover the "
                     "next level (reproduction-gated), beating a BFS ablation?"),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "game": GAME, "prefix_level": prefix_level, "max_expansions": MAX_EXPANSIONS,
        "induce_fired": induce_fired,
        "energy_arm": treat, "bfs_arm": ctrl,
        "energy_unlocks_new_level": energy_unlocks,
        "energy_more_efficient": energy_more_efficient,
        "solve_provenance": "live_agent_self_discovery",
        "used_env_source": False, "read_game_source": False,
        "offline_ground_truth_bfs": False, "hand_calibrated_per_game": False,
        "interpretation": (
            "energy_unlocks=True -> the relational goal-energy ENABLES live self-discovery of the next "
            "level where blind BFS cannot (the GAP-4891 deepening payoff). more_efficient=True -> both "
            "reach it but the energy is cheaper (an efficiency win). neither -> the energy separates "
            "(Stage 1) but does not guide the search past it: the deepening wall is the SEARCH/enumeration, "
            "not goal-detection (consistent with the .452 env-grounded WALL_DEEPER finding)."
        ),
        "cites_upstream": ["arc_within_game_l3_self_induction_*_stage1.json", "wf_b77c3b40-f11"],
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    payload = dict(art); payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    (REPO / "results" / f"arc_relational_goal_energy_stage2_{GAME}.json").write_text(json.dumps(art, indent=2) + "\n")
    print("\n=== VERDICT:", verdict)
    print(f"-> results/arc_relational_goal_energy_stage2_{GAME}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
