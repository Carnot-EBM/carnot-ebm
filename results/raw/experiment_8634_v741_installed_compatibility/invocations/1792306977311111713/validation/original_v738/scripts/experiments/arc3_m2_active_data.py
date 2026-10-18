"""M2-v4a: does ACTIVE, coverage-driven data collection lower the induced-model consistency energy
vs passive random exploration? (Operator-chosen lever after M2-v3: attack the DATA bottleneck.)

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). M2-v3 showed the bottleneck moved off
the verifier and off inducer-expressiveness onto DATA quality: ~750 passive random transitions
under-determine the transition rule, because the inducer sees each action's effect from too few
distinct contexts to learn position-INVARIANCE (it memorizes positions instead). Active collection
deliberately samples each action from DIVERSE states (new object configurations, every action balanced,
every clickable color), so the inducer sees the rule's invariances.

Fair test: a COMMON held-out test set (large passive random sample = the agent's true state
distribution) grades two models — one trained on a passive-random budget-N set, one on an active
budget-N set (same N). Active wins if its model's grid-grounded consistency energy on the common test
is lower. Run with the DSL inducer (fast, no codex) on all 25 games; the verifier is grade_predictions
(no oracle). Offline, no LLM/GPU.

  .venv/bin/python scripts/experiments/arc3_m2_active_data.py --train_budget 800 --test_budget 1500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_agi3_world_model import (  # noqa: E402
    GameGraph, grid_of, frame_hash, compute_grid_delta, objects)
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel  # noqa: E402
from carnot.agentic.arc_world_model_synth import grade_predictions  # noqa: E402
import arc3_graph_explore as gx  # noqa: E402
from arc3_m2_world_model import _collect, _auroc  # noqa: E402 (passive baseline collector)


def _state_sig(grid, q=8):
    """Coarse state signature for coverage: sorted quantized object centroids (position-invariance
    is learned by seeing an action applied from MANY distinct signatures)."""
    return tuple(sorted((cy // q, cx // q) for cy, cx in objects(grid)))[:12]


def active_collect(arc, game, budget, episodes, rng, GameAction, GameState):
    """Coverage-driven collection: at each state prefer the action that (a) has been applied from the
    FEWEST distinct contexts, (b) is untested here, (c) for clicks, targets an un-clicked color, while
    avoiding deadly actions. Maximizes effect-space coverage per action -> better invariance learning."""
    by_id = {a.value: a for a in GameAction}
    graph = GameGraph(game)
    transitions = []
    action_counts: Counter = Counter()
    action_sigs: dict = defaultdict(set)        # action_int -> set of state signatures it's fired from
    clicked_colors: set = set()
    total = 0
    for _ in range(episodes):
        env = arc.make(game)
        f = env.reset()
        prev = None
        while total < budget:
            grid = grid_of(f); fh = frame_hash(grid); graph.see_node(fh, f)
            if prev is not None:
                transitions.append((prev[1], prev[2], grid.copy()))
                graph.record(prev[0], prev[2], fh, compute_grid_delta(prev[1], grid), 0, False)
            if getattr(f, "state", None) in (GameState.WIN, GameState.GAME_OVER):
                break
            sig = _state_sig(grid)
            cands = [c for c in gx._candidate_akeys(grid, getattr(f, "available_actions", []))
                     if not graph.is_deadly(fh, c)]
            if not cands:
                break

            def score(c):
                a = c[0]; s = 0.0
                if sig not in action_sigs[a]:
                    s += 3.0                          # NEW context for this action (the key signal)
                if not graph.tried(fh, c):
                    s += 1.0
                s -= 0.15 * action_counts[a]          # balance action usage (prefer under-sampled)
                if a == 6 and len(c) == 3 and 0 <= c[2] < grid.shape[0] and 0 <= c[1] < grid.shape[1]:
                    if int(grid[c[2], c[1]]) not in clicked_colors:
                        s += 2.0                       # an un-clicked color
                return s + rng.random() * 0.4          # tiebreak

            akey = max(cands, key=score)
            a_int = akey[0]
            action_counts[a_int] += 1
            action_sigs[a_int].add(sig)
            if a_int == 6 and 0 <= akey[2] < grid.shape[0] and 0 <= akey[1] < grid.shape[1]:
                clicked_colors.add(int(grid[akey[2], akey[1]]))
            data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
            prev = (fh, grid.copy(), akey)
            f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
            total += 1
            if getattr(f, "state", None) == GameState.GAME_OVER:
                transitions.append((prev[1], akey, grid_of(f).copy()))
                break
        if total >= budget:
            break
    return transitions


def _common_test(transitions, train_keys, max_n=600):
    """Held-out test = transitions whose (frame_hash, action) key is NOT in the train set (true
    generalization), capped."""
    out = []
    for s, a, s2 in transitions:
        if (frame_hash(s), tuple(a)) not in train_keys and (s != s2).any():
            out.append((s, a, s2))
        if len(out) >= max_n:
            break
    return out


def _keys(transitions):
    return {(frame_hash(s), tuple(a)) for s, a, _ in transitions}


def run(games=None, train_budget=800, test_budget=1500, episodes=35, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    sel = [g for g in all_ids if (not games or g.split("-")[0] in set(games))]
    probe = REPO / "results" / "arc3_determinism_probe.json"
    hidden = set(json.loads(probe.read_text()).get("hidden_state_games", [])) if probe.exists() else set()

    per_game = []
    for game in sel:
        short = game.split("-")[0]
        test_all = _collect(arc, game, test_budget, episodes, rng, GameAction, GameState)   # common test pool
        passive = _collect(arc, game, train_budget, episodes, rng, GameAction, GameState)   # passive train
        active = active_collect(arc, game, train_budget, episodes, rng, GameAction, GameState)  # active train
        test = _common_test(test_all, _keys(passive) | _keys(active))
        e_passive = grade_predictions(ObjectDeltaModel(short).fit(passive).predict, test)
        e_active = grade_predictions(ObjectDeltaModel(short).fit(active).predict, test)
        ep, ea = e_passive["energy"], e_active["energy"]
        per_game.append({
            "game": short, "is_hidden_state_truth": short in hidden,
            "energy_passive": ep, "energy_active": ea,
            "improvement": (round(ep - ea, 4) if (ep is not None and ea is not None) else None),
            "n_test": e_passive.get("n_changed_transitions"),
            "n_passive": len(passive), "n_active": len(active),
        })
        print(f"  {short:6s} passive={ep} active={ea} improve={per_game[-1]['improvement']} "
              f"hidden={short in hidden} n_test={e_passive.get('n_changed_transitions')}", flush=True)

    rated = [g for g in per_game if g["improvement"] is not None]

    def _mean(gs, k):
        return round(sum(g[k] for g in gs) / len(gs), 4) if gs else None
    mp, ma = _mean(rated, "energy_passive"), _mean(rated, "energy_active")
    n_improved = sum(1 for g in rated if g["improvement"] > 0.03)
    n_regressed = sum(1 for g in rated if g["improvement"] < -0.03)
    # focus on games where the DSL has traction (passive energy < 0.9 => expressible enough to benefit)
    tract = [g for g in rated if g["energy_passive"] is not None and g["energy_passive"] < 0.9]
    mp_t, ma_t = _mean(tract, "energy_passive"), _mean(tract, "energy_active")

    verdict = (f"complete: m2v4_active_data_meanE_passive{mp}_active{ma}_improved{n_improved}"
               f"_regressed{n_regressed}of{len(rated)}_tractableMeanE_passive{mp_t}_active{ma_t}")
    art = {
        "experiment": "arc3_m2_active_data", "title": "arc3_m2v4_active_vs_passive_data_collection",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_world_model_consistency_energy",
        "claim": ("Active coverage-driven collection vs passive random, same budget, graded by the DSL "
                  "inducer's consistency energy on a COMMON held-out test. Active wins if energy lower."),
        "n_games": len(per_game), "train_budget": train_budget, "test_budget": test_budget,
        "mean_energy_passive": mp, "mean_energy_active": ma,
        "n_improved_gt_0.03": n_improved, "n_regressed_gt_0.03": n_regressed,
        "tractable_games_mean_energy_passive": mp_t, "tractable_games_mean_energy_active": ma_t,
        "n_tractable_games": len(tract),
        "random_seed": seed, "no_llm_used": True, "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1), "per_game": per_game,
        "note": ("M2-v4a active data collection. If active lowers energy (esp. on DSL-tractable games), "
                 "the data hypothesis holds -> apply to codex synthesis (M2-v4b) + add latent-state "
                 "tracking for hidden-state games. If not, the bottleneck is observation/representation, "
                 "not exploration policy -> latent registers become primary."),
    }
    if write:
        (REPO / "results" / "arc3_m2_active_data.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {verdict}")
    print(f"   mean DSL energy: passive={mp} -> active={ma} | improved {n_improved}, regressed "
          f"{n_regressed} of {len(rated)} | tractable games: {mp_t} -> {ma_t}")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="", help="comma-separated short ids; empty = all 25")
    ap.add_argument("--train_budget", type=int, default=800)
    ap.add_argument("--test_budget", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gl = [g.strip() for g in args.games.split(",") if g.strip()] or None
    run(games=gl, train_budget=args.train_budget, test_budget=args.test_budget,
        episodes=args.episodes, seed=args.seed)
