"""LLM GAP-FILLER (not solver). On a game the deterministic search STALLS / is inefficient
on, the iGPU LLM writes ONE focused component — a goal-distance HEURISTIC — that plugs into
the working value-guided explorer (graph_explore_solve_v3). The LLM does NOT solve the game
(LLMs get ~0% on ARC directly); it writes the small missing piece, and the search does the
work. The heuristic is GROUNDED: it must run on real grids and only counts if the search
then SOLVES (reproduction-gated). Per the 'LLM is a gap-filler' reframe (2026-06-17).

Usage: arc_gap_fill.py <game> [--repo gemma-4-12B-it] [--budget 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
from carnot.agentic.arc_graph_explore import graph_explore_solve_v3, trajectory_labels


def _mh():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def characterize(game: str):
    """Collect RAW (64x64) transitions by replaying the banked win + recording each step,
    so the prompt has real dynamics + the full WIN STATE (the goal to score distance to)."""
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    steps = [mh.normalize(a) for a in mh.load_actions(src)] if src else []
    arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset(); trans = []
    for aid, data in steps:
        if aid is None:
            continue
        g0 = grid_of(f); l0 = _levels_completed(f)
        f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
        if f is None:
            break
        trans.append(e3.Transition(g0, int(aid), data, grid_of(f), l0, _levels_completed(f)))
    win = next((t.next_grid for t in trans if t.level_after > t.level_before), None)
    return trans, win


def gap_fill_prompt(game: str, trans, win) -> str:
    h, w = win.shape
    colors = sorted(set(int(v) for v in win.flatten().tolist()))
    return f"""You are helping a SEARCH algorithm solve the ARC-AGI-3 game '{game}'. The search
explores game states but STALLS — it needs a HEURISTIC to know which states are closer to
winning. Write `def goal_distance(grid) -> float`: LOWER = CLOSER to a level-complete (win)
state. `grid` is a {h}x{w} numpy int array (colors {colors}).

Below are observed transitions (compact: one full grid + per-action deltas) and the full WIN
STATE. INFER what makes a state close to the win (pieces aligned to targets, a region filled,
the player adjacent to an exit, etc.) and score it so goal_distance(win_state)≈0 and
far-from-win states score higher. Use only numpy. Output ONLY one ```python code block
containing `import numpy as np` and the complete function `def goal_distance(grid):`.

{e3._transitions_block(trans)}
```python
"""


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("game")
    ap.add_argument("--repo", default="gemma-4-12B-it"); ap.add_argument("--budget", type=int, default=8000)
    args = ap.parse_args(); game = args.game; t0 = time.time()
    print(f"== LLM gap-fill (goal-distance heuristic) for {game} on the iGPU ==", flush=True)

    trans, win = characterize(game)
    if win is None:
        print("  no win-state available (need a banked solve to ground the heuristic)"); return 0
    print(f"  characterized: {len(trans)} transitions, win-state {win.shape}", flush=True)

    import math
    from carnot.agentic import gap_fills

    def _gd_from_code(code_str):
        ns: dict = {}
        exec(code_str, ns)
        return ns["goal_distance"]

    def _validate(code_str) -> bool:   # runtime smoke-test: must run on win+start -> finite floats
        gd_ = _gd_from_code(code_str)
        return math.isfinite(float(gd_(win))) and math.isfinite(float(gd_(trans[0].grid)))

    # AUTOLEARNING reuse-first: a previously-captured heuristic is pre-generated (no LLM call)
    saved = gap_fills.load_heuristic(game)
    if saved is not None:
        gd, code, reused = saved, "<pre-generated>", True
        print("  REUSING pre-generated heuristic (autolearning; no LLM call)", flush=True)
    else:
        reused = False
        proposer = e3.LocalGGUFProposer(repo_substr=args.repo)
        ok, code = proposer.generate(gap_fill_prompt(game, trans, win),
                                     required=("goal_distance",), validate=_validate)
        if not ok:
            print(f"  LLM heuristic generation FAILED: {code[:160]}", flush=True)
            _write(game, {"stage": "generate", "ok": False, "error": code[:200]}, t0); return 0
        gd = _gd_from_code(code)
    d_win = float(gd(win)); d_start = float(gd(trans[0].grid))
    print(f"  heuristic runs: goal_distance(win)={d_win:.2f}, goal_distance(start)={d_start:.2f} "
          f"({'REUSED pre-generated' if reused else 'freshly generated'})", flush=True)

    def verifier(frame):
        try:
            return float(gd(grid_of(frame)))
        except Exception:
            return 1e9

    # A/B: value-guided search WITH the LLM heuristic vs novelty-only
    results = {}
    for label, vf in [("llm_heuristic", verifier), ("novelty_only", None)]:
        arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard())
        t1 = time.time()
        traj, lvl = graph_explore_solve_v3(env, 0, max_expansions=args.budget, max_depth=60, verifier=vf)
        solved = bool(traj)
        gate = False
        if solved:
            def apply(env, label_, frame):
                s = json.loads(label_); return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))
            g = kit.reproduce(game, trajectory_labels(traj), apply, claimed_level=lvl)
            gate = bool(g["reproduced"])
        results[label] = {"solved": solved, "reproduced": gate,
                          "actions": len(traj) if traj else 0, "secs": round(time.time() - t1, 0)}
        print(f"  [{label:13}] solved={solved} reproduced={gate} actions={len(traj) if traj else 0} "
              f"[{time.time()-t1:.0f}s]", flush=True)

    helped = (results["llm_heuristic"]["reproduced"] and not results["novelty_only"]["reproduced"]) or \
             (results["llm_heuristic"]["reproduced"] and results["novelty_only"]["reproduced"] and
              results["llm_heuristic"]["actions"] < results["novelty_only"]["actions"])
    verdict = ("success_gap_fill_heuristic_helped_" + game if helped
               else f"complete_gap_fill_{game}_heuristic_no_gain")
    print(f"  HEURISTIC HELPED: {helped}", flush=True)
    # AUTOLEARNING CAPTURE: a freshly-generated heuristic that HELPED (reproduction-gated) is
    # rolled back in as a pre-generated, deterministic, bundle-able asset for future runs.
    captured = None
    if helped and not reused:
        path = gap_fills.save_heuristic(
            game, code, meta=f"reproduced solve in {results['llm_heuristic']['actions']} actions "
                             f"via {args.repo}; A/B beat novelty-only")
        captured = str(path.relative_to(REPO))
        print(f"  CAPTURED -> {captured} (autolearning: future runs reuse it, no LLM call)", flush=True)
    _write(game, {"win_dist": d_win, "start_dist": d_start, "results": results, "helped": helped,
                  "reused_pregenerated": reused, "captured_to": captured,
                  "honest_verdict": verdict, "proposer": "pre-generated" if reused else args.repo,
                  "inference_substrate": "pregenerated_deterministic" if reused else "live_llm_inference_igpu",
                  "verifier_is_oracle": False, "heuristic_code": code[:2000]}, t0)
    return 0


def _write(game, payload, t0):
    payload.update({"experiment": f"arc_gap_fill_{game}", "game": game,
                    "duration_s": round(time.time() - t0, 1), "run_date": "2026-06-17"})
    (REPO / "results" / f"arc_gap_fill_{game}.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote results/arc_gap_fill_{game}.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
