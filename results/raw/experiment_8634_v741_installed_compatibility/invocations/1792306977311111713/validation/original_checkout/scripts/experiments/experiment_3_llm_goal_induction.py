#!/usr/bin/env python3
"""#3 LLM goal-induction on FIRST CONTACT — inject a goal SIGNAL to steer exploration.

WHY (2026-06-22, the path both #1 and #2 nulls point to): the hard-tail wall is
GOAL-DIRECTED exploration-to-first-win. #2 showed the recombination engine works
(crossover fires) but is goal-starved — exploration never reaches the win region, so
there is no winning fragment to recombine, because coverage-fitness has no goal to
steer toward. The ONE lever that injects a goal from OUTSIDE the exploration loop is
world knowledge: have the local LLM READ the grid + observed action-effects and INDUCE
the most plausible WIN CONDITION, compiled into a dense `goal_progress(grid) -> float`
the search descends. This is the chicken-and-egg breaker — a goal PRIOR before any win
is seen (distinct from arc_gap_fill, which needs a banked win to ground its heuristic).

Reuses arc_gap_fill's proven machinery (LocalGGUFProposer codegen + sandbox-compile +
graph_explore_solve_v2 A* with the heuristic), but the prompt induces the goal from
STRUCTURE, not from a win grid. A/B vs blind BFS at matched budget, reproduction-gated.
Honest, OFFLINE, zero quota. verifier_is_oracle: false (the LLM ESTIMATES the goal;
the env's win logic is the oracle, kept distinct). The perceptual-grounding wall
(Sensi 2603.17683) is the live risk — measured here, not assumed away.

OUTER-LOOP PREP EXPERIMENT (not a conductor task). Runs the local LLM on the iGPU
(NEVER the 3090s). PRECONDITION: the GGUF must be cached / the llama-server must start;
on failure the game records blocked_llm_unavailable (no fabrication).

  .venv/bin/python scripts/experiments/experiment_3_llm_goal_induction.py \
      --games wa30,sb26,su15 --budget 5000 --repo Qwen3.5-9B-MTP
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import (
    graph_explore_solve_v2, trajectory_labels, rich_action_candidates, _warm,
)
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_3_llm_goal_induction.json"


def _ok(frame) -> bool:
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _logical(frame, cell):
    return np.asarray(to_logical(grid_of(frame), cell))


def explore_transitions(game: str, cell, n: int, rng) -> tuple:
    """Warm-up exploration to gather action->effect examples (the dynamics the LLM reasons
    over) + the start logical grid. Pure salient-random walk; no goal yet."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    start = _logical(f, cell)
    trans = []
    for _ in range(n):
        if not _ok(f):
            break
        cands = rich_action_candidates(f)
        if not cands:
            break
        c = cands[rng.randrange(min(len(cands), 8))]
        g0 = _logical(f, cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, int(c.action_id)), data=c.data)
        if nf is None or not _ok(nf):
            break
        trans.append(e3.Transition(g0, int(c.action_id), c.data, _logical(nf, cell),
                                   l0, _levels_completed(nf)))
        f = nf
    return trans, start, list(getattr(f, "available_actions", []) or range(1, 7))


def _delta_examples(trans, k: int = 4) -> str:
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)][:k]
    out = []
    for t in changed:
        d = e3._delta(t.grid, t.next_grid, cap=20)
        out.append(f"  ACTION{t.action}: changed (row,col,from,to) = {d}")
    return "\n".join(out) or "  (no state-changing actions observed yet)"


def induce_prompt(game: str, start_grid: np.ndarray, trans, avail) -> str:
    h, w = start_grid.shape
    colors = sorted(set(int(v) for v in start_grid.flatten().tolist()))
    return f"""You help a SEARCH algorithm solve an ARC-AGI-3 grid-puzzle game on FIRST CONTACT
(no win has been seen yet). From the current grid + the observed action effects, INDUCE the MOST
PLAUSIBLE WIN CONDITION and write a heuristic `def goal_progress(grid) -> float`: 0.0 when the
inferred goal is satisfied, HIGHER = farther. The search descends goal_progress toward a win.

`grid` is a {h}x{w} numpy int array (colors {colors}). `np` and `ndi` (scipy.ndimage) are ALREADY
DEFINED — do not import. Write a SHORT function (<= 15 lines).

Reason about what THIS game likely WANTS. Common ARC-AGI-3 win conditions:
  - cover / fill every cell of a target color, or make a color disappear entirely
  - align or connect matching same-color pairs / regions
  - move a single avatar object onto a distinct goal cell (manhattan distance)
  - make the grid symmetric / complete a partial pattern
Pick the SINGLE most plausible goal from the grid structure + the action effects below, and make
goal_progress measure distance to THAT goal (a graded count/distance that DROPS as the goal is
approached — not a constant).

CRITICAL CONTRACT: goal_progress MUST `return float(...)` for ANY grid — never None, never fall off
the end. No win grid is given; DERIVE the goal from structure. Do NOT hardcode the current grid.

CURRENT GRID ({h}x{w}):
{e3.to_ascii(start_grid)}
OBSERVED ACTION -> EFFECT:
{_delta_examples(trans)}
AVAILABLE ACTIONS: {list(avail)}

Output ONLY one ```python code block with just `def goal_progress(grid):` (np and ndi provided).
```python
"""


def main() -> int:
    import random
    import scipy.ndimage as _ndi

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=str, default="wa30,sb26,su15")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--repo", type=str, default="gemma-4-12B-it",
                    help="GGUF repo substring (iGPU). arc_gap_fill's proven goal-heuristic codegen model; "
                         "swap --repo Qwen3.5-9B-DeepSeek for the live-agent generator")
    ap.add_argument("--warmup-steps", type=int, default=18)
    ap.add_argument("--max-depth", type=int, default=60, help="A* max path depth (raise for deep fills)")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--dump-prompt", action="store_true",
                    help="print the induce-prompt (grid + dynamics) for each game and exit (no LLM)")
    ap.add_argument("--goal-code", type=str, default="",
                    help="path to a Claude-written goal_progress(grid) .py file; bypasses the local LLM "
                         "(the Claude-as-goal-inducer test). Applies to the single --games game.")
    args = ap.parse_args()
    t0 = time.time()
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    proposer = None
    if not args.dump_prompt and not args.goal_code:
        proposer = e3.LocalGGUFProposer(repo_substr=args.repo, max_tokens=1536, timeout=600)
    rows = []
    for game in games:
        rng = random.Random(args.seed + hash(game) % 9999)
        t1 = time.time()
        arc = kit.offline_arcade()
        env0 = arc.make(game, scorecard_id=arc.open_scorecard())
        f0 = _warm(env0, False)
        if not _ok(f0):
            rows.append({"game": game, "honest_verdict": "blocked_degenerate_start"}); continue
        cell = detect_cell(grid_of(f0))
        trans, start_grid, avail = explore_transitions(game, cell, args.warmup_steps, rng)

        def _progress_from_code(code_str):
            ns: dict = {"np": np, "ndi": _ndi}
            exec(code_str, ns)
            return ns["goal_progress"]

        def _validate(code_str) -> bool:
            try:
                gp = _progress_from_code(code_str)
                return math.isfinite(float(gp(start_grid)))
            except Exception:
                return False

        prompt = induce_prompt(game, start_grid, trans, avail)
        if args.dump_prompt:
            print(f"\n===== INDUCE-PROMPT {game} (cell={cell}, {len(trans)} transitions) =====")
            print(prompt)
            print(f"===== END {game} =====\n", flush=True)
            continue
        if args.goal_code:
            code = Path(args.goal_code).read_text()
            ok = _validate(code)
            inducer = f"claude:{Path(args.goal_code).name}"
        else:
            ok, code = proposer.generate(prompt, required=("goal_progress",), validate=_validate)
            inducer = f"local:{args.repo}"
        if not ok:
            rows.append({"game": game, "induced": False, "inducer": inducer,
                         "honest_verdict": "blocked_inducer_unavailable_or_invalid_code",
                         "error": str(code)[:200], "secs": round(time.time() - t1, 1)})
            print(f"  [{game}] goal-induction FAILED ({inducer}): {str(code)[:120]}", flush=True)
            continue
        gp = _progress_from_code(code)

        def heuristic(frame):
            try:
                return float(gp(_logical(frame, cell)))
            except Exception:
                return 1e9

        # A/B: LLM-goal-directed A* vs blind BFS at matched budget, reproduction-gated.
        res = {}
        for label, hf in [("llm_goal_heuristic", heuristic), ("bfs_baseline", None)]:
            arc = kit.offline_arcade()
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            st: dict = {}
            traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=args.budget, max_depth=args.max_depth,
                                               heuristic=hf, stats=st)
            won = bool(traj) and int(lvl) >= 1
            repro = False
            if won:
                def apply(env_, label_, frame):
                    s = json.loads(label_)
                    return env_.step(_game_action(GameAction, s["action"]), data=s.get("data"))
                g = kit.reproduce(game, trajectory_labels(traj), apply, claimed_level=int(lvl))
                repro = bool(g["reproduced"])
            res[label] = {"won": won, "offline_reproduced": repro, "reached_level": int(lvl),
                          "expansions": st.get("expansions"), "actions": len(traj) if traj else 0}

        llm, bfs = res["llm_goal_heuristic"], res["bfs_baseline"]
        row = {
            "game": game, "induced": True, "inducer": inducer, "goal_progress_code": code.strip()[:1200],
            "goal_progress_start_value": round(float(gp(start_grid)), 3),
            "llm_won": llm["won"], "llm_offline_reproduced": llm["offline_reproduced"],
            "llm_reached_level": llm["reached_level"], "llm_expansions": llm["expansions"],
            "bfs_won": bfs["won"], "bfs_offline_reproduced": bfs["offline_reproduced"],
            "bfs_reached_level": bfs["reached_level"], "bfs_expansions": bfs["expansions"],
            "llm_generates_where_bfs_does_not": bool(llm["offline_reproduced"] and not bfs["offline_reproduced"]),
            "honest_verdict": "complete: llm_goal_induction_ran",
            "secs": round(time.time() - t1, 1),
        }
        rows.append(row)
        print(f"  [{game}] induced=True | llm_won={llm['won']} repro={llm['offline_reproduced']} "
              f"(L{llm['reached_level']}, exp={llm['expansions']}) | bfs_won={bfs['won']} "
              f"repro={bfs['offline_reproduced']} (L{bfs['reached_level']}) | "
              f"LLM>BFS={row['llm_generates_where_bfs_does_not']} [{row['secs']}s]", flush=True)

    induced = [r for r in rows if r.get("induced")]
    n_llm_only = sum(1 for r in induced if r.get("llm_generates_where_bfs_does_not"))
    n_llm_repro = sum(1 for r in induced if r.get("llm_offline_reproduced"))
    n_blocked = sum(1 for r in rows if not r.get("induced"))
    if n_blocked == len(rows):
        verdict = "blocked_llm_unavailable_all_games"
    elif n_llm_only >= 2:
        verdict = "success: llm_goal_induction_generates_above_bfs_on_2plus_games"
    elif n_llm_only >= 1:
        verdict = "complete: llm_goal_induction_generates_above_bfs_on_1_game_preliminary"
    elif n_llm_repro >= 1:
        verdict = "complete: llm_goal_induction_reproduces_but_not_above_bfs_honest_null_gap_sharpened"
    else:
        verdict = "complete: llm_goal_induction_no_winner_honest_null_gap_sharpened"

    artifact = {
        "experiment": "experiment_3_llm_goal_induction",
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "inference_substrate": "live_llm_inference",
        "model_specs": {"repo_substr": args.repo, "device": "iGPU (never the 3090s)"},
        "random_seed": args.seed,
        "budget_max_expansions": args.budget,
        "games": games,
        "n_games_llm_generates_above_bfs": n_llm_only,
        "n_games_llm_reproduced": n_llm_repro,
        "n_games_blocked": n_blocked,
        "gate_note": "LLM-induced goal steers search to a reproduced win BFS does NOT find, on >=2 hard games",
        "grounding_wall_note": "Sensi 2603.17683: the live risk is the LLM mis-reading the grid -> wrong goal",
        "rows": rows,
        "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"  LLM-generates-above-BFS on {n_llm_only}/{len(induced)} induced games; "
          f"reproduced {n_llm_repro}; blocked {n_blocked}. -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
