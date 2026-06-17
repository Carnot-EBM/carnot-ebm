"""STANDING ARC learning-loop entrypoint — the conductor's ARC north-star task runs
this each milestone instead of bespoke one-off code. It executes the full loop on
the offline sim (zero quota), self-improving across runs:

  1. pick the target game (--game X, or --auto via the registry + transfer routing)
  2. if the game is ADAPTERED (arc_game_adapters): verifier-routed best-first SOLVE
     to +1 level, warm-started by a saved LEARNED verifier if present else the hand
     verifier (arc_solver_kit.OfflineSolver)
  3. REPRODUCTION GATE the solution offline (arc_solver_kit.reproduce)
  4. TRAIN + CHECKPOINT the learned verifier on the accumulated solve traces
     (arc_value_learner) — mirror-ready weights (Rule 3; public release operator-only)
  5. emit a milestone artifact (offline_reproduced / reproduced_levels / states)
  6. if the game is NOT adaptered: emit the transfer-routing recommendation + recipe
     + general gotchas — the agent's per-game RE starting point (the irreducible delta)

Usage (what the conductor task calls):
  .venv/bin/python scripts/arc_loop_solve.py --game lp85 --target-level 3
  .venv/bin/python scripts/arc_loop_solve.py --auto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_game_adapters as adapters
from carnot.agentic import arc_solve_learning as learning
from carnot.agentic.arc_value_learner import LearnedVerifier, collect_trajectory_data
from carnot.agentic.arc_graph_explore import graph_explore_solve, trajectory_labels

CKPT_DIR = REPO / "models"
RESULTS = REPO / "results"


def _ckpt_path(game: str) -> Path:
    return CKPT_DIR / f"arc_verifier_{game}.json"


def solve_adaptered(game: str, target_level: int) -> dict:
    ad = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())

    # warm-start the search with a saved LEARNED verifier if present, else hand verifier
    ckpt = _ckpt_path(game)
    if ckpt.exists() and ad.featurize is not None:
        verifier = LearnedVerifier.load(ckpt, ad.featurize)
        verifier_src = "learned_checkpoint"
    else:
        verifier = ad.hand_verifier
        verifier_src = "hand_verifier_cold_start"

    solver = kit.OfflineSolver(game, ad.action_labels, ad.apply, ad.state_key,
                               warmup_label=ad.warmup_label, verifier=verifier)
    f = solver._replay(env, [])
    cur = kit.frame_level(f)
    full, total_states, X, y = [], 0, [], []
    for lvl in range(cur + 1, target_level + 1):
        path, nodes = solver.solve_level(env, cur, full, ad.depth_caps.get(lvl, 90))
        total_states += nodes
        if path is None:
            break
        if ad.featurize is not None:
            Xi, yi = collect_trajectory_data(env, solver, full, path, ad.featurize)
            X += Xi; y += yi
        f = solver._replay(env, full + path)
        cur = kit.frame_level(f)
        full += path

    # REPRODUCTION GATE
    gate = kit.reproduce(game, full, ad.apply, warmup_label=ad.warmup_label, claimed_level=cur)

    # TRAIN + CHECKPOINT the learned verifier (self-improvement for next run)
    ckpt_written = None
    if X and ad.featurize is not None:
        lv = LearnedVerifier(ad.featurize).fit(X, y)
        lv.save(ckpt, meta={"trained_games": [game], "feature_names": "adapter_featurize",
                            "provenance": f"arc_loop_solve {game}->L{cur}"})
        ckpt_written = str(ckpt.relative_to(REPO))

    return {
        "game": game, "reached_level": cur, "moves": len(full),
        "states_expanded": total_states, "verifier_src": verifier_src,
        "offline_reproduced": bool(gate["reproduced"]), "reproduced_levels": cur,
        "learned_verifier_checkpoint": ckpt_written,
        "reproduction_gate": gate,
        "mode": "standing_arc_loop_offline_no_quota",
    }


def solve_via_explore(game: str, max_actions: int = 140, restarts: int = 60,
                      warmup: bool = False) -> Optional[dict]:
    """ADAPTER-FREE first contact: graph-explore the game; if it advances a level,
    CAPTURE the trajectory, reproduction-gate it, train a verifier from it, and
    persist the trajectory as the adapter SEED — so the next solve is the efficient
    verifier-routed loop, not blind exploration. Returns None if no advance (caller
    falls back to the routing recommendation for per-game RE)."""
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    traj, lvl = graph_explore_solve(env, 0, max_actions=max_actions, restarts=restarts, warmup=warmup)
    if traj is None:
        return None

    def apply(env, label, frame):
        step = json.loads(label)
        return env.step(_game_action(GameAction, step["action"]), data=step.get("data"))

    labels = trajectory_labels(traj)
    gate = kit.reproduce(game, labels, apply, claimed_level=lvl)

    # SEED a verifier from the first solve (generic grid features -> steps-to-go),
    # so the captured solve trains the verifier that will route future searches.
    def featurize_frame(frame):
        g = grid_of(frame)
        nz = int((g != 0).sum())
        return [float(nz), float(len(set(g.flatten().tolist()))), float(g.shape[0] * g.shape[1])]
    X, y = [], []
    f = env.reset()
    if warmup and traj:
        f = apply(env, labels[0], f)
    for i, lab in enumerate(labels):
        X.append(featurize_frame(f)); y.append(float(len(labels) - i))
        f = apply(env, lab, f)
    ckpt = None
    if X:
        lv = LearnedVerifier(featurize_frame).fit(X, y)
        ckpt = CKPT_DIR / f"arc_verifier_{game}.json"
        lv.save(ckpt, meta={"trained_games": [game], "feature_names": ["nonzero", "colors", "cells"],
                            "provenance": f"graph_explore first-solve {game}->L{lvl}"})

    # persist the captured trajectory as the adapter SEED
    seed = RESULTS / f"arc_explore_trajectory_{game}.json"
    seed.write_text(json.dumps({"game": game, "reached_level": lvl, "trajectory": traj}, indent=2))
    return {
        "game": game, "method": "graph_explore_adapter_free", "reached_level": lvl,
        "moves": len(traj), "offline_reproduced": bool(gate["reproduced"]),
        "reproduced_levels": lvl, "trajectory_seed": str(seed.relative_to(REPO)),
        "verifier_seed_checkpoint": (str(ckpt.relative_to(REPO)) if ckpt else None),
        "next": "register a GameAdapter from the seed for verifier-routed re-solving",
        "mode": "standing_arc_loop_graph_explore_no_quota",
    }


def needs_re(game: str) -> dict:
    rec = learning.recommend_approach(game)
    return {
        "game": game, "status": "needs_per_game_RE",
        "transfer_recommendation": rec.get("recommended"),
        "general_gotchas": rec.get("general_gotchas"),
        "guidance": rec.get("guidance"),
        "instruction": ("Reverse-engineer this game's win/action/state DELTA reusing the routed "
                        "recipe, register a GameAdapter in arc_game_adapters.py, then re-run "
                        "this loop. Per CLAUDE.md ARC Solve Reproducibility + Solver-Reuse Discipline."),
        "mode": "standing_arc_loop_routing_only",
    }


def pick_target() -> str:
    # next target: an adaptered game first (extend its levels), else a routed unsolved game
    ad = adapters.adaptered_games()
    return ad[0] if ad else "tn36"


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--target-level", type=int, default=3)
    args = ap.parse_args(argv)
    game = args.game or (pick_target() if args.auto else None)
    if not game:
        ap.error("specify --game X or --auto")

    print(f"== standing ARC loop: game={game} ==")
    if adapters.get_adapter(game):
        out = solve_adaptered(game, args.target_level)        # verifier-routed (efficient)
    else:
        out = solve_via_explore(game) or needs_re(game)       # adapter-free first contact, else route to RE
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"arc_loop_solve_{game}.json").write_text(json.dumps(out, indent=2))
    for k in ("status", "reached_level", "offline_reproduced", "reproduced_levels",
              "states_expanded", "verifier_src", "learned_verifier_checkpoint"):
        if k in out:
            print(f"  {k}: {out[k]}")
    print(f"  wrote results/arc_loop_solve_{game}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
