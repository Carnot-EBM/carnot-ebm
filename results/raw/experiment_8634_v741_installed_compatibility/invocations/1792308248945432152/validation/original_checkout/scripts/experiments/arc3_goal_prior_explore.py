"""M1b: GameGraph explorer + LOCAL Gemma-4 GOAL-PRIOR — does a goal signal unblock the 23 zeros?

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M1b). The M1 Family-A floor sweep
(results/arc3_familyA_floor_sweep.json) found pure no-induction exploration solves only 2/25 games
to Level 1 and never progresses — a SPARSE-REWARD ceiling: 23 games give no intermediate signal to
bootstrap search. This experiment isolates the cause: is the blocker the GOAL (nothing to aim at) or
the DYNAMICS (search too weak to plan)? It grafts the cheapest possible goal signal onto the SAME
explorer: a local multimodal Gemma-4 looks at the rendered frame and proposes a short ranked queue of
goal-directed candidate actions. The LLM is the GENERATOR (hypothesizes the goal + promising moves);
the GameGraph is the VERIFIER/pruner (drops deadly + no-effect + already-seen — the Carnot division
of labor). The LLM is called only to REFILL the queue (when it empties / on level-up / when stuck),
never per action, so latency stays bounded.

Read-out: compare ACCURACY_total_levels_solved against the Family-A floor on the SAME games.
- goal-prior LIFTS levels  => the blocker was the GOAL; a cheap prior + existing search suffices.
- goal-prior does NOT help => the blocker is multi-step DYNAMICS planning => build the Family-B
  deterministic-delta world-model synthesizer (M2, the full Carnot thesis) next.

Offline + air-gapped (SDK OperationMode.OFFLINE + local environment_files/). Local GPU only; no
network, no leaderboard. Emits the quota-gate artifact shape for comparability.

  .venv/bin/python scripts/experiments/arc3_goal_prior_explore.py --model E2B \
      --games m0r0,r11l,vc33,ar25,cn04,tr87 --budget 300 --episodes 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_agi3_world_model import (  # noqa: E402
    GameGraph, grid_of, frame_hash, compute_grid_delta, objects)
import arc3_graph_explore as gx  # noqa: E402  (reuse _candidate_akeys / _pick fallback)
import arc3_gemma_policy as gp  # noqa: E402  (reuse _render / _load model cache)


def _full_game_id(arc, short):
    for e in arc.get_environments():
        gid = getattr(e, "game_id", None)
        if gid and gid.split("-")[0] == short:
            return gid
    return short


def goal_prior_actions(frame, ctx, graph, fh, model_key="E2B", k=5):
    """Generator step: Gemma-4 SEES the frame and proposes up to k ranked goal-directed actions.
    Returns a list of action_keys ((a,) or (6,x,y)); [] on parse failure. One LLM call per refill."""
    import torch
    proc, model = gp._load(model_key)
    av = list(getattr(frame, "available_actions", []) or [])
    lv = int(getattr(frame, "levels_completed", 0) or 0)
    mem = ctx.setdefault("mem", {})
    mem.setdefault("notes", "")
    hist = mem.setdefault("history", [])
    # summarize what this node already learned, so the LLM proposes NEW goal-directed moves
    deadly_here = [e for ek, e in graph.edges.items()
                   if e["from"] == fh and ek in graph.deadly]
    img = gp._render(frame)
    click = ("ACTION6 is a CLICK; give x,y in 0-63 (the image is the 64x64 grid upscaled). "
             if 6 in av else "")
    deadly_txt = (", ".join(f"{e['akey']}" for e in deadly_here[:6]) or "(none yet)")
    txt = (
        "You are SOLVING an ARC-AGI-3 interactive grid puzzle. The image is the CURRENT frame.\n"
        f"Levels solved so far: {lv}. Available actions: {av}. {click}\n"
        f"Your running GOAL hypothesis (carry/refine): {mem['notes'] or '(none yet - infer it)'}\n"
        f"Recent action->outcome: {' | '.join(hist[-8:]) if hist else '(none yet)'}\n"
        f"Actions that caused GAME_OVER here (AVOID): {deadly_txt}\n"
        "First infer the WIN GOAL (what configuration ends the level), then propose the most "
        "promising distinct actions toward it, best first. Prefer actions that make visible "
        "progress; avoid the deadly ones.\n"
        "Output EXACTLY:\n"
        "GOAL: <one phrase for the win condition>\n"
        f"Then up to {k} lines, best first, each 'ACTION:<n>' or 'ACTION:6 x=<x> y=<y>'."
    )
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to("cuda")
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.uint8:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    resp = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gm = re.search(r"GOAL:\s*(.+)", resp, re.I)
    if gm:
        mem["notes"] = gm.group(1).strip()[:200]
    akeys = []
    for line in resp.splitlines():
        mm = re.search(r"ACTION:\s*([0-7])(?:\s+x=\s*(\d+)\s+y=\s*(\d+))?", line, re.I)
        if not mm:
            continue
        a = int(mm.group(1))
        if a not in av:
            continue
        if a == 6 and mm.group(2):
            akeys.append((6, int(mm.group(2)) % 64, int(mm.group(3)) % 64))
        elif a != 6:
            akeys.append((a,))
    # de-dup preserving order
    seen, uniq = set(), []
    for k_ in akeys:
        if k_ not in seen:
            seen.add(k_); uniq.append(k_)
    return uniq


def run(model_key="E2B", games=None, budget=300, episodes=8, seed=0, max_llm_per_game=120, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    by_id = {a.value: a for a in GameAction}
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    info = {getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or [])
            for e in arc.get_environments()}
    shorts = games or ["m0r0", "r11l", "vc33", "ar25", "cn04", "tr87"]
    per_game, lat = [], []
    n_llm_calls = 0

    for short in shorts:
        game = _full_game_id(arc, short)
        baseline = info.get(game, [])
        graph = GameGraph(game)
        ctx = {"mem": {"history": [], "notes": ""}}
        total_actions = 0
        max_levels = 0
        first_solve_action = None
        llm_cache: dict[str, list[tuple]] = {}   # frame_hash -> ranked goal-prior proposals (one call/state)
        calls_this_game = 0

        for ep in range(episodes):
            env = arc.make(game)
            f = env.reset()
            prev = None
            while total_actions < budget:
                grid = grid_of(f)
                fh = frame_hash(grid)
                graph.see_node(fh, f)
                cur_lv = int(getattr(f, "levels_completed", 0) or 0)
                if prev is not None:
                    delta = compute_grid_delta(prev[2], grid)
                    ld = cur_lv - prev[3]
                    graph.record(prev[0], prev[1], fh, delta, ld, game_over=False)
                    outcome = ("LEVEL_UP" if ld > 0 else
                               "no_change" if delta.get("n_changed", 0) == 0 else "changed")
                    ctx["mem"]["history"].append(
                        f"{prev[1]}->{outcome}({delta.get('n_changed', 0)})")
                    if cur_lv > max_levels:
                        max_levels = cur_lv
                        first_solve_action = first_solve_action or total_actions
                st = getattr(f, "state", None)
                if st in (GameState.WIN, GameState.GAME_OVER):
                    break
                cands = gx._candidate_akeys(grid, getattr(f, "available_actions", []))
                untested = graph.untested(fh, cands)
                # query the goal prior ONCE per distinct state (cached by frame_hash), capped per game
                if fh not in llm_cache and calls_this_game < max_llm_per_game:
                    t0 = time.time()
                    llm_cache[fh] = goal_prior_actions(f, ctx, graph, fh, model_key=model_key)
                    lat.append(time.time() - t0); n_llm_calls += 1; calls_this_game += 1
                akey = None
                for cand in llm_cache.get(fh, []):     # generator-proposed, goal-directed, verifier-pruned
                    if not graph.tried(fh, cand) and not graph.is_deadly(fh, cand):
                        akey = cand; break
                if akey is None:                       # fall back to Family-A graph exploration
                    if untested:
                        akey = gx._pick(graph, fh, untested, rng)
                    else:
                        frontier = graph.frontier_states(
                            lambda h, n: gx._candidate_akeys_for_node(graph, h))
                        akey = graph.shortest_path_action(fh, frontier - {fh}) if frontier else None
                        if akey is None:
                            break
                a_int = akey[0]
                data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
                prev = (fh, akey, grid, cur_lv)
                f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
                total_actions += 1
                if getattr(f, "state", None) == GameState.GAME_OVER:
                    ng = grid_of(f)
                    graph.record(prev[0], prev[1], frame_hash(ng),
                                 compute_grid_delta(prev[2], ng), 0, True)
                    ctx["mem"]["history"].append(f"{prev[1]}->GAME_OVER")
                    break
                if getattr(f, "state", None) == GameState.WIN:
                    max_levels = max(max_levels, int(getattr(f, "levels_completed", 0) or 0))
                    break
            if total_actions >= budget:
                break

        r = {"game_id": game, "levels_solved": max_levels, "win_levels": len(baseline),
             "actions_used": total_actions, "first_solve_at": first_solve_action,
             "graph_nodes": len(graph.nodes), "deadly": len(graph.deadly),
             "goal_notes": ctx["mem"]["notes"]}
        per_game.append(r)
        print(f"  {short:8s} levels={max_levels}/{len(baseline)} actions={total_actions} "
              f"nodes={len(graph.nodes)} deadly={len(graph.deadly)} solve@{first_solve_action} "
              f"| goal='{ctx['mem']['notes'][:50]}'", flush=True)

    total = sum(r["levels_solved"] for r in per_game)
    total_win = sum(r["win_levels"] for r in per_game)
    art = {
        "experiment": "arc3_goal_prior_explore", "title": f"arc3_goal_prior_{model_key}",
        "honest_verdict": (f"complete: goal_prior_{model_key}_levels{total}of{total_win}"
                           f"_over{len(per_game)}games_beats_floor={total > 0}"),
        "inference_substrate": "offline_arc_agi3_graph_explore_plus_local_gemma4_goal_prior",
        "model": gp.MODELS[model_key], "policy": f"goal_prior_{model_key}",
        "games": shorts, "budget_per_game": budget, "episodes_per_game": episodes,
        "ACCURACY_total_levels_solved": total, "ACCURACY_total_win_levels": total_win,
        "n_llm_calls": n_llm_calls, "mean_llm_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "per_game": per_game, "random_seed": seed,
        "no_gpu_used": False, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1),
        "note": ("M1b goal-prior de-risk: LLM generator proposes goal-directed actions, GameGraph "
                 "verifier prunes deadly/no-effect/seen. Compare vs Family-A floor on the SAME games: "
                 "lift => goal was the blocker; no lift => need the Family-B dynamics synthesizer (M2)."),
    }
    if write:
        (REPO / "results" / f"arc3_goal_prior_{model_key}.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {art['honest_verdict']} | {n_llm_calls} LLM calls, "
          f"mean_latency={art['mean_llm_latency_s']}s", flush=True)
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(gp.MODELS), default="E2B")
    ap.add_argument("--games", default="m0r0,r11l,vc33,ar25,cn04,tr87",
                    help="comma-separated short game ids")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_llm", type=int, default=120, help="cap LLM goal-prior calls per game")
    args = ap.parse_args()
    art = run(model_key=args.model, games=[g.strip() for g in args.games.split(",") if g.strip()],
              budget=args.budget, episodes=args.episodes, seed=args.seed, max_llm_per_game=args.max_llm)
    raise SystemExit(0 if art["ACCURACY_total_levels_solved"] > 0 else 1)
