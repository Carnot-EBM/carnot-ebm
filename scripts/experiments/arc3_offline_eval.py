"""Offline ARC-AGI-3 evaluation harness — the quota-gate's measurement loop.

Plays a PLUGGABLE policy over the 25 ARC-AGI-3 games FULLY OFFLINE (air-gapped: SDK
OperationMode.OFFLINE + local environment_files/, no API, no quota), and reports the
north-star metrics in a comparable artifact:
  ACCURACY  = total levels solved (and per-game levels_completed)
  EFFICIENCY = actions used vs EnvironmentInfo.baseline_actions[level] (the reference)

WHY (operator directive 2026-06-08, feedback_arc3_online_gated_on_offline_beating_baselines):
online/scored runs are quota-limited and gated — only go online when an OFFLINE result
beats BOTH the TRM baseline AND our best prior submitted Carnot run. This harness produces
that offline number, consistently, so any candidate agent is measured here FIRST. The stock
ARC-AGI-3-Agents main.py forces an online /api/games call (can't run offline), so we use
our own offline driver.

Add a policy to POLICIES; a policy is policy(frame, ctx, rng) -> (GameAction, data|None).

  .venv/bin/python scripts/experiments/arc3_offline_eval.py --policy random --budget_factor 1.5
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
OUT_TMPL = str(REPO / "results" / "arc3_offline_eval_{policy}.json")


def random_policy(frame, ctx, rng):
    """Uniform over available actions; click actions (ACTION6) get random grid coords."""
    from arcengine.enums import GameAction
    by_id = {a.value: a for a in GameAction}
    av = list(getattr(frame, "available_actions", []) or [])
    if not av:
        return None, None
    a_int = rng.choice(av)
    data = {"x": rng.randrange(ctx["grid_w"]), "y": rng.randrange(ctx["grid_h"])} if a_int == 6 else None
    return by_id.get(a_int, GameAction.ACTION1), data


def _objects(frame):
    """Connected non-background components -> candidate click targets (the action-pruner).
    Returns list of (y, x) representative cells, one per object. Pure perception, no induction."""
    import numpy as np
    try:
        arr = np.array(frame.frame)
        if arr.ndim == 3:
            arr = arr[-1]
    except Exception:
        return []
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[counts.argmax()])
    mask = arr != bg
    if not mask.any():
        return []
    # simple 4-neighbour flood-fill labelling (no scipy dependency)
    h, w = arr.shape
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


def object_click_policy(frame, ctx, rng):
    """Verifier-as-action-pruner: on click games, click OBJECT centroids (round-robin) instead
    of random pixels; on keyboard games, cycle action types systematically. Pure perception +
    structured exploration — NO rule induction (that's the generator's job, TBD)."""
    from arcengine.enums import GameAction
    by_id = {a.value: a for a in GameAction}
    av = list(getattr(frame, "available_actions", []) or [])
    if not av:
        return None, None
    mem = ctx.setdefault("mem", {"obj_i": 0, "kb_i": 0})
    if 6 in av:  # click action available -> click an object, not a random pixel
        objs = _objects(frame)
        if objs:
            y, x = objs[mem["obj_i"] % len(objs)]
            mem["obj_i"] += 1
            # small jitter within the object's neighbourhood
            return GameAction.ACTION6, {"x": int(x), "y": int(y)}
        # no objects -> fall back to a random click
        return GameAction.ACTION6, {"x": rng.randrange(ctx["grid_w"]), "y": rng.randrange(ctx["grid_h"])}
    # keyboard-only: cycle through the available non-reset actions systematically
    kb = [a for a in av if a != 0]
    if kb:
        a_int = kb[mem["kb_i"] % len(kb)]
        mem["kb_i"] += 1
        return by_id.get(a_int, GameAction.ACTION1), None
    return by_id.get(rng.choice(av), GameAction.ACTION1), None


POLICIES = {"random": random_policy, "object_click": object_click_policy}


def _grid_dims(frame):
    try:
        import numpy as np
        arr = np.array(frame.frame)
        if arr.ndim == 3:
            arr = arr[-1]
        return arr.shape  # (h, w)
    except Exception:
        return (64, 64)


def play_game(arc, game_id, baseline, policy, budget, rng):
    """Play one game offline; track levels solved + the action count at each level-up."""
    from arcengine.enums import GameAction, GameState
    env = arc.make(game_id)
    f = env.reset()
    h, w = _grid_dims(f)
    ctx = {"game_id": game_id, "baseline": baseline, "grid_h": h, "grid_w": w}
    actions = 0
    levels = int(getattr(f, "levels_completed", 0) or 0)
    action_at_level = []  # actions taken when each new level was reached
    state = str(getattr(f, "state", "?"))
    while f is not None and actions < budget:
        st = getattr(f, "state", None)
        if st in (GameState.WIN, GameState.GAME_OVER):
            state = str(st)
            break
        action, data = policy(f, ctx, rng)
        if action is None:
            break
        f = env.step(action, data=data)
        actions += 1
        if f is not None:
            nl = int(getattr(f, "levels_completed", 0) or 0)
            while nl > levels:               # a level (or more) was just solved
                levels += 1
                action_at_level.append(actions)
            state = str(getattr(f, "state", state))
    # per-level efficiency: actions spent on each solved level vs its baseline
    per_level_actions, prev = [], 0
    for a in action_at_level:
        per_level_actions.append(a - prev)
        prev = a
    eff = []
    for i, used in enumerate(per_level_actions):
        b = baseline[i] if i < len(baseline) else None
        eff.append(round(used / b, 3) if b else None)
    return {"game_id": game_id, "levels_solved": levels, "win_levels": len(baseline),
            "actions_used": actions, "budget": budget, "final_state": state,
            "per_level_actions": per_level_actions, "baseline_actions": baseline,
            "per_level_action_ratio": eff}


def run(policy_name="random", n_games=25, budget_factor=1.5, budget_cap=3000, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    started = time.time()
    rng = random.Random(seed)
    policy = POLICIES[policy_name]
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    envs = arc.get_environments()
    info = {getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or []) for e in envs}
    games = list(info)[:n_games]

    per_game = []
    for g in games:
        base = info[g] or [60]
        budget = min(int(sum(base) * budget_factor) or 80, budget_cap)
        r = play_game(arc, g, base, policy, budget, rng)
        per_game.append(r)
        print(f"  {g:18s} levels={r['levels_solved']}/{r['win_levels']} "
              f"actions={r['actions_used']}/{budget} state={r['final_state']}", flush=True)

    total_levels = sum(r["levels_solved"] for r in per_game)
    total_win = sum(r["win_levels"] for r in per_game)
    games_with_any = sum(1 for r in per_game if r["levels_solved"] > 0)
    solved_ratios = [x for r in per_game for x in (r["per_level_action_ratio"] or []) if x is not None]
    mean_eff = round(sum(solved_ratios) / len(solved_ratios), 3) if solved_ratios else None

    art = {
        "experiment": "arc3_offline_eval",
        "title": f"arc3_offline_eval_{policy_name}",
        "honest_verdict": (f"complete: arc3_offline_{policy_name}_levels{total_levels}of{total_win}"
                           f"_games_with_progress{games_with_any}_meaneff{mean_eff}"),
        "inference_substrate": "offline_air_gapped_arc_agi3_local_environments",
        "policy": policy_name, "n_games": len(games), "random_seed": seed,
        "budget_factor": budget_factor, "budget_cap": budget_cap,
        # the two north-star axes, for the quota-gate comparison:
        "ACCURACY_total_levels_solved": total_levels,
        "ACCURACY_total_win_levels": total_win,
        "ACCURACY_solve_rate": round(total_levels / total_win, 4) if total_win else 0.0,
        "EFFICIENCY_mean_action_ratio_on_solved": mean_eff,
        "games_with_any_progress": games_with_any,
        "per_game": per_game,
        "submitted_to_leaderboard": False, "no_llm_used": True,
        "duration_s": round(time.time() - started, 1),
        "gate_note": ("Offline, air-gapped. This is the number the quota-gate compares: only go "
                      "ONLINE when ACCURACY_total_levels_solved (then EFFICIENCY) beats BOTH the "
                      "TRM-agent baseline AND our best prior submitted Carnot run."),
    }
    if write:
        Path(OUT_TMPL.format(policy=policy_name)).write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=list(POLICIES), default="random")
    ap.add_argument("--n_games", type=int, default=25)
    ap.add_argument("--budget_factor", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"OFFLINE eval: policy={args.policy} over {args.n_games} games (air-gapped)")
    art = run(policy_name=args.policy, n_games=args.n_games,
              budget_factor=args.budget_factor, seed=args.seed)
    print(f"\n-> {art['honest_verdict']}")
    print(f"   ACCURACY levels={art['ACCURACY_total_levels_solved']}/{art['ACCURACY_total_win_levels']} "
          f"(solve_rate {art['ACCURACY_solve_rate']}) | EFFICIENCY mean_action_ratio={art['EFFICIENCY_mean_action_ratio_on_solved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
