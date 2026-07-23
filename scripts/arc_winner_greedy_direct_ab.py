#!/usr/bin/env python3
"""Winner-faithful greedy-direct A/B (REQ-ARC-WMTE-5829, operator: match the 27-31B leaderboard leaders).

Runs the Duck-Harness-style greedy-direct agent (carnot.agentic.arc_greedy_direct_agent) with
gemma-4-31B (the model 2 of 3 Milestone-1 winners used) DIRECTLY driving actions -- no search, no
induced world model, adapter-free -- on a set of stalled games, and measures LEVELS DISCOVERED. The
comparison point is our own current stack: the standing arc_live_oracle_gap.json baseline (E3AgentPolicy
frame-only, 9B) scores 0 on these games, and experiment_5722 already showed a 31B AS AN INDUCER in our
architecture also scores 0. The open question this answers: does the 31B used the WINNERS' way (direct
greedy action generator) discover levels where our architecture does not?

Optional `--also-9b` runs the SAME greedy-direct loop with Qwen3.5-9B to isolate model-vs-architecture.

gemma-4-31B (~18.3GB Q4) runs on ONE 24GB RTX 3090 (GPU 1, the outer-loop's card). Writes a dedicated
artifact. Usage:
  arc_winner_greedy_direct_ab.py [--games bp35,lf52] [--budget 120] [--max-turns 4] [--max-seq 5] [--also-9b]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")  # outer-loop owns GPU 1

GEMMA_PORT = 8955
QWEN_PORT = 8957
SEED = 5829
DEFAULT_GAMES = ["bp35", "lf52"]


def _oracle_levels() -> dict:
    import yaml

    d = yaml.safe_load((REPO / "ops" / "arc_solve_registry.yaml").read_text())
    return {
        g["game"]: int(g.get("levels_reproduced", 0))
        for g in d.get("games", [])
        if g.get("reproducibility") == "reproduced" and int(g.get("levels_reproduced", 0)) > 0
    }


def _run_arm(
    label: str, proposer, games, *, budget, max_turns, max_seq, perception, reflect_iv
) -> list[dict]:
    from carnot.agentic.arc_greedy_direct_agent import run_greedy_direct

    rows = []
    for game in games:
        print(f"\n-- [{label}] {game} (budget={budget}, perception={perception}) --", flush=True)
        try:
            r = run_greedy_direct(
                game,
                proposer,
                action_budget=budget,
                max_turns=max_turns,
                max_seq=max_seq,
                seed=SEED,
                perception=perception,
                reflection_interval=reflect_iv,
                goal_verify=goal_verify,
            )
            row = {
                "game": game,
                "levels_gained": r.levels_gained,
                "reached_level": r.reached_level,
                "actions_taken": r.actions_taken,
                "orientations": r.orientations,
                "game_over": r.game_over,
                "wall_s": r.wall_s,
                "transcript_sample": r.transcript_sample,
                "final_notes": r.final_notes,
                "goal_verifier_stats": r.goal_verifier_stats,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - a policy crash on a game is a datum
            row = {"game": game, "error": f"{type(exc).__name__}: {exc}"[:300]}
        print(
            f"   -> levels={row.get('levels_gained')} reached=L{row.get('reached_level')} "
            f"actions={row.get('actions_taken')} orients={row.get('orientations')} "
            f"[{row.get('wall_s')}s] err={row.get('error')}",
            flush=True,
        )
        rows.append(row)
    return rows


def main() -> None:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    argv = sys.argv[1:]

    def _arg(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default

    games = _arg("--games", ",".join(DEFAULT_GAMES)).split(",")
    budget = int(_arg("--budget", "120"))
    max_turns = int(_arg("--max-turns", "4"))
    max_seq = int(_arg("--max-seq", "5"))
    also_9b = "--also-9b" in argv
    perception = _arg("--perception", "objects")
    reflect_iv = int(_arg("--reflect", "0"))
    goal_verify = "--goal-verify" in argv

    print(
        f"== winner greedy-direct A/B: games={games} budget={budget} "
        f"max_turns={max_turns} max_seq={max_seq} perception={perception} reflect={reflect_iv} goal_verify={goal_verify} also_9b={also_9b} ==",
        flush=True,
    )
    t0 = time.time()

    gemma = LocalGGUFProposer(
        repo_substr="gemma-4-31B-it",
        port=GEMMA_PORT,
        mtp=False,
        kv_quant="q8_0",
        n_ctx=8192,
        max_tokens=256,
        no_think_prefix="",
    )
    arms = {}
    print("Loading gemma-4-31B on GPU 1...", flush=True)
    arms["gemma31_greedy_direct"] = _run_arm(
        "gemma31",
        gemma,
        games,
        budget=budget,
        max_turns=max_turns,
        max_seq=max_seq,
        perception=perception,
        reflect_iv=reflect_iv,
        goal_verify=goal_verify,
    )

    if also_9b:
        qwen = LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP",
            port=QWEN_PORT,
            mtp=True,
            kv_quant="q8_0",
            n_ctx=8192,
            max_tokens=256,
            no_think_prefix="",
        )
        arms["qwen9b_greedy_direct"] = _run_arm(
            "qwen9b",
            qwen,
            games,
            budget=budget,
            max_turns=max_turns,
            max_seq=max_seq,
            perception=perception,
            reflect_iv=reflect_iv,
            goal_verify=goal_verify,
        )

    oracle = _oracle_levels()
    gemma_rows = arms["gemma31_greedy_direct"]
    any_level = any((r.get("levels_gained") or 0) > 0 for r in gemma_rows)
    any_depth = any((r.get("reached_level") or 0) > 0 for r in gemma_rows)
    errored = [r["game"] for r in gemma_rows if r.get("error")]

    if errored and not any_level:
        verdict = f"complete_greedy_direct_ab_ran_with_errors_{'_'.join(errored)}"
    elif any_level:
        won = [r["game"] for r in gemma_rows if (r.get("levels_gained") or 0) > 0]
        verdict = f"complete_greedy_direct_gemma31_DISCOVERED_levels_on_{'_'.join(won)}_where_our_stack_gets_0"
    elif any_depth:
        verdict = (
            "complete_greedy_direct_gemma31_reached_depth_but_no_full_levelup_vs_zero_baseline"
        )
    else:
        verdict = "complete_greedy_direct_gemma31_honest_null_no_discovery_matches_our_stack_zero_baseline"

    duration_s = round(time.time() - t0, 1)
    checksum_input = json.dumps(
        [{"game": r["game"], "levels": r.get("levels_gained")} for r in gemma_rows], sort_keys=True
    ).encode()
    artifact = {
        "experiment": "outer_loop_arc_winner_greedy_direct_ab_20260723",
        "schema": "carnot.arc_winner_greedy_direct_ab.v1",
        "run_date": "2026-07-23",
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": "Real gemma-4-31B-it GGUF (18.3GB Q4) loaded on GPU 1 via "
        "LocalGGUFProposer, driving the greedy-direct agent DIRECTLY (no search, no induced world "
        "model), adapter-free on the offline arcade -- the winner architecture. Live LLM inference "
        "throughout (many /completion calls per game).",
        "solve_provenance": "live_agent_self_discovery",
        "solve_provenance_note": "adapter-free greedy-direct discovery (the agent picks every action "
        "from its own play, no GameAdapter, no banked trajectory).",
        "target_model": "unsloth/gemma-4-31B-it-GGUF",
        "random_seed": SEED,
        "reproducibility_checksum": hashlib.sha256(checksum_input).hexdigest(),
        "duration_s": duration_s,
        "config": {
            "budget": budget,
            "max_turns": max_turns,
            "max_seq": max_seq,
            "games": games,
            "perception": perception,
            "reflection_interval": reflect_iv,
            "goal_verify": goal_verify,
        },
        "honest_verdict": verdict,
        "narrative": (
            "Operator directive: match the leaderboard leaders (27-31B, greedy-direct). Tests whether "
            "gemma-4-31B used the WINNERS' way (direct greedy action generator) discovers levels where "
            "our induce-then-plan stack (9B, and 31B-as-inducer in exp5722) scores 0. First real test "
            "of the winner architecture for us -- all prior tests used our own architecture."
        ),
        "acceptance_gate": {
            "condition": "gemma-4-31B greedy-direct discovers >0 levels on a game where our current "
            "stack (arc_live_oracle_gap baseline) scores 0",
            "principle": "The winners top the leaderboard with 27-31B greedy-direct; if a faithful "
            "reproduction discovers a level our stack cannot, matching the leaders is validated and "
            "justifies switching the submission stack.",
            "passed": bool(any_level),
        },
        "baseline_our_stack": {
            "config": "E3AgentPolicy frame-only 9B (arc_live_oracle_gap.json) + 31B-as-inducer (exp5722)",
            "levels_on_these_games": 0,
        },
        "per_game_oracle_levels": {g: oracle.get(g, 0) for g in games},
        "arms": arms,
    }
    out = REPO / "results" / "outer_loop_arc_winner_greedy_direct_ab_20260723.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"\nWrote {out}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
