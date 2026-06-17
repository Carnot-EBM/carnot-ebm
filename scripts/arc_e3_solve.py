"""E3 runner — Carnot Executable-World-Model solver (induce -> VERIFY -> refactor ->
plan) for one ARC-AGI-3 game, via codex/gpt-5.5 as the proposer and the Carnot
WorldModelVerifier as the grounding. After arXiv:2605.05138. Quota-aware: bounded
codex rounds. Usage: arc_e3_solve.py <game> [--rounds N] [--n-trans M] [--no-plan]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--rounds", type=int, default=2, help="max induce/refactor codex rounds")
    ap.add_argument("--n-trans", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=480, help="per-codex-call timeout (s)")
    ap.add_argument("--no-plan", action="store_true")
    args = ap.parse_args()
    game = args.game
    t0 = time.time()

    print(f"== E3 executable-world-model solver: {game} ==", flush=True)
    trans, cell = e3.collect_transitions(game, n=args.n_trans, seed=1)
    changed = sum(1 for t in trans if not (t.grid == t.next_grid).all())
    print(f"  collected {len(trans)} transitions (cell={cell}, {changed} grid-changing)", flush=True)

    verifier = e3.WorldModelVerifier(trans)
    proposer = e3.CodexProposer(timeout=args.timeout)
    history, best_acc = [], 0.0

    for rnd in range(args.rounds):
        if rnd == 0:
            print(f"  [round {rnd}] codex INDUCE world_model.py ...", flush=True)
            ok, tail = proposer.induce(game, trans, cell)
        else:
            print(f"  [round {rnd}] codex REFACTOR against {len(vr.mismatches)} mismatches ...", flush=True)
            ok, tail = proposer.refactor(game, vr)
        if not ok:
            print(f"    codex call failed/timeout: {tail[-200:]}", flush=True)
            history.append({"round": rnd, "codex_ok": False, "tail": tail[-400:]})
            break
        try:
            engine, is_done = e3.load_engine(game)
        except Exception as ex:
            print(f"    no loadable engine produced: {ex}", flush=True)
            history.append({"round": rnd, "codex_ok": True, "engine_loadable": False})
            break
        vr = verifier.score(engine)
        best_acc = max(best_acc, vr.accuracy)
        print(f"    VERIFY: {vr.n_correct}/{vr.n} = {vr.accuracy:.0%} reproduced "
              f"({len(vr.mismatches)} mismatch artifacts)", flush=True)
        history.append({"round": rnd, "codex_ok": True, "engine_loadable": True,
                        "accuracy": vr.accuracy, "n_correct": vr.n_correct, "n": vr.n})
        if vr.accuracy >= 0.98 or not vr.mismatches:
            break

    plan_out = None
    if not args.no_plan and best_acc > 0.0:
        try:
            engine, is_done = e3.load_engine(game)
            print(f"  PLAN in the verified model + execute (halt on divergence) ...", flush=True)
            plan_out = e3.plan_and_execute(game, engine, is_done)
            print(f"    plan result: {plan_out}", flush=True)
        except Exception as ex:
            plan_out = {"error": repr(ex)[:200]}

    artifact = {
        "experiment": f"arc_e3_{game}",
        "game": game, "method": "executable_world_model_induce_verify_refactor_plan",
        "paper": "arXiv:2605.05138", "proposer": "codex/gpt-5.5",
        "cell": cell, "n_transitions": len(trans), "grid_changing": changed,
        "verifier_best_accuracy": best_acc, "rounds": history, "plan": plan_out,
        "duration_s": round(time.time() - t0, 1),
        "honest_verdict": _verdict(best_acc, plan_out),
        "verifier_is_oracle": False,
        "inference_substrate": "live_llm_inference",
        "notes": "Carnot WorldModelVerifier grounds the codex-induced model against real "
                 "offline transitions; the LLM is the proposer, the verifier is the moat.",
    }
    out = REPO / "results" / f"arc_e3_{game}.json"
    out.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"  wrote {out.relative_to(REPO)}  verdict={artifact['honest_verdict']}", flush=True)
    return 0


def _verdict(best_acc: float, plan_out) -> str:
    if plan_out and plan_out.get("level_up"):
        return "success_executable_world_model_level_up"
    if best_acc >= 0.9:
        return f"complete_world_model_verified_{best_acc:.2f}_no_levelup_yet"
    if best_acc > 0.0:
        return f"complete_harness_validated_partial_model_{best_acc:.2f}"
    return "complete_harness_built_proposer_did_not_yield_engine"


if __name__ == "__main__":
    raise SystemExit(main())
