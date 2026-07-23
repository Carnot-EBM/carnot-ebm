#!/usr/bin/env python3
"""REQ-ARC-WMTE-5828: real empirical test of the tool-calling + multi-step lookahead search vs
the standing induce-then-plan and bare reactive-filter mechanisms, on the SAME 3 worst live/
oracle-gap games (sc25, lf52, bp35).

**Researcher summary:**
    Operator directive following REQ-ARC-WMTE-5827's GAP-ARC-REACTIVE-FILTER-MYOPIC diagnosis: add
    real multi-step lookahead, and allow up to 12 tool-calling/REPL turns per decision (inspect
    history, reason, then commit). This script runs
    `carnot.agentic.arc_tool_loop_lookahead.ToolLoopLookaheadSession` wired into
    `arc_solver_kit.OfflineSolver`'s best-first search, for real, on the SAME three games as the
    prior two mechanisms, and reports honestly against both prior baselines -- no need to re-run
    either (Failed-Experiment Rerun Discipline).

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-5828
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "outer_loop_tool_loop_lookahead_ab_20260723.json")

GAMES = ["sc25", "lf52", "bp35"]
MAX_TURNS_PER_DECISION = 12  # per operator directive, verbatim
MAX_SEARCH_NODES = 15  # bounded lookahead search per game (cost: up to 12 LLM turns/node)
DEPTH_CAP = 15
CUDA_PORT = 8947

PRIOR_BASELINES = {
    "sc25": {
        "induce_then_plan": {
            "levels_gained": 0,
            "source": "outer_loop_holdout_generalization_probe_sc25_20260722.json",
        },
        "reactive_filter": {
            "levels_gained": 0,
            "source": "outer_loop_reactive_verifier_filter_ab_20260722.json",
        },
        "oracle_levels": 6,
    },
    "lf52": {
        "induce_then_plan": {
            "levels_gained": 0,
            "source": "experiment_5727_arc_generalization_live_oracle_gap_v511.json",
        },
        "reactive_filter": {
            "levels_gained": 0,
            "source": "outer_loop_reactive_verifier_filter_ab_20260722.json",
        },
        "oracle_levels": 10,
    },
    "bp35": {
        "induce_then_plan": {
            "levels_gained": 0,
            "source": "experiment_5727_arc_generalization_live_oracle_gap_v511.json",
        },
        "reactive_filter": {
            "levels_gained": 0,
            "source": "outer_loop_reactive_verifier_filter_ab_20260722.json",
        },
        "oracle_levels": 9,
    },
}


def main() -> None:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_solver_kit import OfflineSolver, offline_arcade, frame_level
    from carnot.agentic.arc_tool_loop_lookahead import ToolLoopLookaheadSession

    t0 = time.monotonic()
    print(f"Loading LocalGGUFProposer (Qwen3.5-9B-MTP) on CUDA GPU 1, port {CUDA_PORT}...")
    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        port=CUDA_PORT,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="",
        max_tokens=512,
    )

    per_game = []
    for game in GAMES:
        print(f"\n=== {game} ===")
        t_game0 = time.monotonic()
        session = ToolLoopLookaheadSession(prop, max_turns=MAX_TURNS_PER_DECISION)
        solver = OfflineSolver(
            game,
            action_labels=session.action_labels,
            apply=session.apply,
            state_key=session.state_key,
            verifier=session.verifier,
            warmup_label=session.WARMUP_LABEL,
            max_nodes=MAX_SEARCH_NODES,
            branch_mode="replay",
            move_pruner=session.pruner,
        )
        error = None
        path = None
        states_expanded = 0
        reached_level = 0
        try:
            arc = offline_arcade()
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            path, states_expanded = solver.solve_level(
                env, start_level=0, prefix=[], depth_cap=DEPTH_CAP
            )
            reached_level = frame_level(solver.last_frame) if solver.last_frame is not None else 0
        except Exception as exc:  # noqa: BLE001 - a policy crash on a game is a datum
            error = f"{type(exc).__name__}: {exc}"[:300]

        game_duration_s = round(time.monotonic() - t_game0, 3)
        row = {
            "game": game,
            "solved": path is not None,
            "levels_gained": 1 if path is not None else 0,
            "path_length": len(path) if path else 0,
            "states_expanded": states_expanded,
            "tool_loop_calls": session._tool_loop_calls,
            "final_reached_level": reached_level,
            "error": error,
            "game_wall_s": game_duration_s,
            "prior_baselines": PRIOR_BASELINES[game],
        }
        print(json.dumps(row, indent=2))
        per_game.append(row)

    duration_s = round(time.monotonic() - t0, 3)
    any_new_level = any(r["levels_gained"] > 0 for r in per_game)
    replay_errors = [r for r in per_game if r["error"] is not None]

    if replay_errors:
        verdict = "complete_ab_ran_with_errors_see_error_field"
    elif any_new_level:
        verdict = "complete_tool_loop_lookahead_reached_a_real_levelup_where_both_prior_mechanisms_did_not"
    else:
        verdict = (
            "complete_tool_loop_lookahead_honest_negative_no_levelup_matches_both_prior_baselines"
        )

    checksum_input = json.dumps(
        [{"game": r["game"], "levels_gained": r["levels_gained"]} for r in per_game], sort_keys=True
    ).encode()
    reproducibility_checksum = hashlib.sha256(checksum_input).hexdigest()

    artifact = {
        "experiment": "outer_loop_tool_loop_lookahead_ab_20260723",
        "schema": "carnot.arc_tool_loop_lookahead_ab.v1",
        "run_date": "2026-07-23",
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "live_agent_self_discovery",
        "target_model": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "random_seed": None,
        "random_seed_note": "OfflineSolver's best-first search is deterministic given the LLM's "
        "responses; the LLM itself is sampled at temperature=0.2 (arc_tool_loop_lookahead._completion) "
        "-- no separate seed control exists in this harness.",
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": duration_s,
        "duration_note": f"Real, non-mocked run across {len(GAMES)} games, max_nodes={MAX_SEARCH_NODES}, "
        f"up to {MAX_TURNS_PER_DECISION} tool-calling turns per search-node decision.",
        "honest_verdict": verdict,
        "narrative": (
            "Operator directive following REQ-ARC-WMTE-5827's GAP-ARC-REACTIVE-FILTER-MYOPIC "
            "diagnosis: add real multi-step lookahead (reusing arc_solver_kit.OfflineSolver's "
            "best-first search, not a new algorithm) and up to 12 tool-calling/REPL turns per "
            "decision (a constrained, safe tool API -- not arbitrary code execution, a disclosed "
            "scope reduction from Duck Harness's own full sandboxed REPL). Tested on the SAME 3 "
            "worst live/oracle-gap games as both prior mechanisms."
        ),
        "acceptance_gate": {
            "condition": "any tested game reaches a real levelup that BOTH prior mechanisms "
            "(induce-then-plan, bare reactive-filter) did not reach",
            "principle": "Both prior mechanisms scored 0 new levels across every measurement on "
            "these exact games -- ANY real levelup here is decisive, unambiguous evidence the "
            "added lookahead+orientation closes a real gap, not a noisy delta.",
            "passed": any_new_level,
        },
        "per_game": per_game,
        "preconditions_checked": [
            {"resource": "arc_tool_loop_lookahead_module_shipped", "available": True},
            {
                "resource": "cuda_gpu1_pinned",
                "available": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU") == "1",
            },
        ],
        "field_provenance": {
            "acceptance_gate": {
                "principle": "A single real levelup where BOTH prior mechanisms scored exactly 0 "
                "across multiple independent measurements is decisive evidence with real headroom "
                "(oracle depths 6-10 levels, well above the live baseline of 0 for every mechanism "
                "tried so far)."
            }
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"verdict: {verdict}")
    print(f"any_new_level_found: {any_new_level}")


if __name__ == "__main__":
    main()
