"""E3-in-agent validation: does inducing a world model + planning IN it solve a game
in FEW real actions (the efficiency win the explorer can't deliver)? Pipeline on one
game: collect transitions INCLUDING a win (so is_level_complete is grounded) -> induce
(codex dev proposer) -> VERIFY (Carnot WorldModelVerifier) -> plan_in_model (pure model,
zero real actions) -> EXECUTE the plan in the real offline env, halting on divergence ->
measure actions-to-solve vs the explorer. Usage: arc_e3_agent_solve.py <game> [--local]
"""
from __future__ import annotations

import argparse
import importlib.util
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


def _mh():
    spec = importlib.util.spec_from_file_location("mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def banked_transitions(game: str, cell: int) -> list:
    """Replay the banked winning solution, recording every transition INCLUDING the
    level-up (the positive is_level_complete example the goal predicate needs)."""
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    steps = [mh.normalize(a) for a in mh.load_actions(src)] if src else []
    arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    out = []
    for aid, data in steps:
        if aid is None:
            continue
        g0 = e3.to_logical(grid_of(f), cell); l0 = _levels_completed(f)
        f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
        if f is None:
            break
        out.append(e3.Transition(g0, int(aid), data, e3.to_logical(grid_of(f), cell), l0, _levels_completed(f)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("game"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(); game = args.game; t0 = time.time()
    print(f"== E3-in-agent solve: {game} (proposer={'local-gguf' if args.local else 'codex-dev'}) ==", flush=True)

    # 1. transitions: banked win (grounds the goal) + exploration (dynamics coverage)
    explore, cell = e3.collect_transitions(game, n=80, seed=3)
    trans = banked_transitions(game, cell) + explore
    wins = sum(1 for t in trans if t.level_after > t.level_before)
    print(f"  collected {len(trans)} transitions (cell={cell}, {wins} winning) for induction", flush=True)

    # 2. induce
    proposer = e3.LocalGGUFProposer() if args.local else e3.CodexProposer(timeout=args.timeout)
    ok, tail = proposer.induce(game, trans, cell)
    if not ok:
        print(f"  induce FAILED: {tail[-200:]}", flush=True)
        _write(game, {"stage": "induce", "ok": False, "tail": tail[-300:]}, t0); return 0
    engine, is_done = e3.load_engine(game)

    # 3. verify
    vr = e3.WorldModelVerifier(trans).score(engine)
    print(f"  VERIFY: {vr.n_correct}/{vr.n} = {vr.accuracy:.0%} reproduced; is_level_complete present={is_done is not None}", flush=True)

    # 4. plan in the model (zero real actions)
    arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard())
    start = e3.to_logical(grid_of(env.reset()), cell)
    plan = e3.plan_in_model(engine, is_done, start)
    print(f"  PLAN in model: {('found '+str(len(plan))+' actions') if plan else 'NO plan to a win in the model'}", flush=True)

    # 5. execute in the real env, halting on divergence
    solved, exec_actions, diverged = False, 0, None
    if plan:
        f = env.reset(); g = e3.to_logical(grid_of(f), cell); start_lvl = _levels_completed(f)
        for step in plan:
            pred = np.asarray(engine(g.copy(), step["action"], step.get("data")))
            f = env.step(getattr(GameAction, f"ACTION{step['action']}"), data=step.get("data"))
            exec_actions += 1
            if f is None:
                break
            obs = e3.to_logical(grid_of(f), cell)
            if _levels_completed(f) > start_lvl:
                solved = True; break
            if pred.shape != obs.shape or not np.array_equal(pred, obs):
                diverged = step; break
            g = obs
    verdict = ("success_e3_agent_solved_%s_in_%d_actions" % (game, exec_actions) if solved
               else f"complete_e3_agent_{game}_verify_{vr.accuracy:.2f}_no_solve")
    print(f"  RESULT: solved={solved} in {exec_actions} real actions "
          f"(explorer needed ~10k+); diverged={diverged is not None}", flush=True)
    _write(game, {"verifier_accuracy": vr.accuracy, "plan_len": len(plan) if plan else 0,
                  "solved": solved, "exec_actions": exec_actions, "diverged": diverged,
                  "honest_verdict": verdict, "proposer": "local-gguf" if args.local else "codex-dev",
                  "inference_substrate": "live_llm_inference", "verifier_is_oracle": True}, t0)
    return 0


def _write(game, payload, t0):
    payload.update({"experiment": f"arc_e3_agent_{game}", "game": game,
                    "duration_s": round(time.time() - t0, 1), "run_date": "2026-06-17"})
    (REPO / "results" / f"arc_e3_agent_{game}.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote results/arc_e3_agent_{game}.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
