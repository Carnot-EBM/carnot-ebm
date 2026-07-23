#!/usr/bin/env python3
"""REQ-ARC-WMTE-5828 follow-up: the larger-budget re-run of the tool-calling + multi-step
lookahead search, to resolve the `GAP-ARC-TOOL-LOOP-LOOKAHEAD-BUDGET-INCONCLUSIVE` finding from
the first (max_nodes=15) run (`results/outer_loop_tool_loop_lookahead_ab_20260723.json`), which
only ever expanded 16-18 total search nodes -- far too shallow to plausibly reach these games'
winning-sequence depths, so it could not distinguish "the mechanism doesn't help" from "the test
never got deep enough to find out."

**Operator directive (2026-07-23):** "run that larger budget test."

**Important recalibration found while preparing this run (NOT known when the first budget was
picked):** `ops/arc_solve_registry.yaml` shows all three games (sc25/lf52/bp35) are ALREADY
`full_game_clear: true` -- but via hand-crafted, per-game `GameAdapter`s built across MANY
outer-loop sessions of deliberate reverse-engineering (`development_proxy` provenance, not
`live_agent_self_discovery`), e.g. lf52's registry entry cites a "146-label full [...] action
sequence" for L3 alone and a "759-label"/"927-action" probe for L8+. A domain-agnostic mechanism
(no game-specific hints, no hand-authored win condition) attempting LEVEL 1 from scratch is not
attempting a small problem -- it is trying to rediscover, from raw pixels and its own exploration
alone, mechanics that took real per-game reverse-engineering to fully map (e.g. sc25's win
condition: "toggle corrected offline cast-grid cells until the visible reference pattern/spell
fires, then tank-control navigate to the exit" -- not obviously inferable from a few dozen
exploratory clicks). This run raises the node/depth budget substantially (still nowhere near
"low hundreds" being provably sufficient for these specific games), and the write-up must report
this honestly: a null result here still does not cleanly prove the architecture doesn't work,
only that it doesn't work within a budget that is itself far short of what full per-game mastery
apparently required. What a null WOULD still say (over the first budget-15 run): whether the
tool loop's own confidence signal shows ANY discriminating power across a meaningfully larger
explored frontier, and whether the search mechanism (not the LLM judgment) scales cleanly.

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
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "results", "outer_loop_tool_loop_lookahead_ab_largebudget_20260723.json"
)
PRIOR_SMALL_BUDGET_ARTIFACT = "outer_loop_tool_loop_lookahead_ab_20260723.json"

GAMES = ["sc25", "lf52", "bp35"]
MAX_TURNS_PER_DECISION = 12  # unchanged -- per operator directive, verbatim
MAX_SEARCH_NODES = 300  # up from 15 (20x) -- still a real, disclosed cost: up to 12 LLM turns/node
DEPTH_CAP = 40  # up from 15 -- the prior depth cap discarded any path >=15 actions without
# ever expanding it further, which could have silently truncated exactly the deep paths this
# test is trying to find.
CUDA_PORT = 8947
RANDOM_SEED = 58280  # distinct from the small-budget run's 5828 (not a rerun of the identical
# seeded trajectory -- a genuinely larger, independent exploration).

PRIOR_BASELINES = {
    "sc25": {
        "induce_then_plan": {"levels_gained": 0},
        "reactive_filter": {"levels_gained": 0},
        "tool_loop_lookahead_smallbudget": {
            "levels_gained": 0,
            "states_expanded": 16,
            "source": PRIOR_SMALL_BUDGET_ARTIFACT,
        },
        "oracle_levels": 6,
    },
    "lf52": {
        "induce_then_plan": {"levels_gained": 0},
        "reactive_filter": {"levels_gained": 0},
        "tool_loop_lookahead_smallbudget": {
            "levels_gained": 0,
            "states_expanded": 16,
            "source": PRIOR_SMALL_BUDGET_ARTIFACT,
        },
        "oracle_levels": 10,
    },
    "bp35": {
        "induce_then_plan": {"levels_gained": 0},
        "reactive_filter": {"levels_gained": 0},
        "tool_loop_lookahead_smallbudget": {
            "levels_gained": 0,
            "states_expanded": 18,
            "source": PRIOR_SMALL_BUDGET_ARTIFACT,
        },
        "oracle_levels": 9,
    },
}


def main() -> None:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_solver_kit import OfflineSolver, offline_arcade, frame_level
    from carnot.agentic.arc_tool_loop_lookahead import ToolLoopLookaheadSession

    t0 = time.monotonic()
    print(
        f"Loading LocalGGUFProposer (Qwen3.5-9B-MTP) on CUDA GPU 1, port {CUDA_PORT}, "
        f"max_nodes={MAX_SEARCH_NODES}, depth_cap={DEPTH_CAP}..."
    )
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
        print(f"\n=== {game} (large budget) ===", flush=True)
        t_game0 = time.monotonic()
        session = ToolLoopLookaheadSession(prop, max_turns=MAX_TURNS_PER_DECISION, seed=RANDOM_SEED)
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
        print(json.dumps(row, indent=2), flush=True)
        per_game.append(row)

    duration_s = round(time.monotonic() - t0, 3)
    any_new_level = any(r["levels_gained"] > 0 for r in per_game)
    replay_errors = [r for r in per_game if r["error"] is not None]

    if replay_errors:
        verdict = "complete_ab_ran_with_errors_see_error_field"
    elif any_new_level:
        verdict = (
            "complete_tool_loop_lookahead_largebudget_reached_a_real_levelup_where_prior_"
            "mechanisms_did_not"
        )
    else:
        verdict = (
            "complete_tool_loop_lookahead_largebudget_still_no_levelup_"
            "still_budget_short_of_registry_documented_per_level_action_counts"
        )

    checksum_input = json.dumps(
        [{"game": r["game"], "levels_gained": r["levels_gained"]} for r in per_game], sort_keys=True
    ).encode()
    reproducibility_checksum = hashlib.sha256(checksum_input).hexdigest()

    artifact = {
        "experiment": "outer_loop_tool_loop_lookahead_ab_largebudget_20260723",
        "schema": "carnot.arc_tool_loop_lookahead_ab.v1",
        "run_date": "2026-07-23",
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "live_agent_self_discovery",
        "solve_provenance_note": "Adapter-free (no per-game hand-authored win condition, no "
        "hardcoded action sequence) -- the mechanism CLASS matches live_agent_self_discovery's "
        "definition. Caveat honestly disclosed: this run itself executes via the offline dev-twin "
        "search backbone (arc_solver_kit.OfflineSolver) directly, NOT via a path the scored "
        "E3AgentPolicy cascade could invoke (arc_tool_loop_lookahead.py is not yet in either live "
        "entrypoint's import closure -- see the operator's own question this run responds to and "
        "the ops/known-issues.md entry addressing it).",
        "target_model": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "random_seed": RANDOM_SEED,
        "random_seed_note": "Deliberately a DIFFERENT seed from the small-budget run's 5828 -- "
        "this is a genuinely larger independent search, not a re-run of the identical seeded "
        "trajectory extended. Same llama.cpp /completion seed mechanism, offset per search-node "
        "decision and per tool-loop turn (see arc_tool_loop_lookahead.py docstrings).",
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": duration_s,
        "duration_note": f"Real, non-mocked run across {len(GAMES)} games, max_nodes={MAX_SEARCH_NODES} "
        f"(20x the first run's 15), depth_cap={DEPTH_CAP}, up to {MAX_TURNS_PER_DECISION} "
        "tool-calling turns per search-node decision.",
        "honest_verdict": verdict,
        "narrative": (
            "Follow-up to the budget-inconclusive first run (max_nodes=15, states_expanded "
            "16-18 total). Raises max_nodes to 300 and depth_cap to 40. Recalibration found "
            "while preparing this run (see module docstring): ops/arc_solve_registry.yaml shows "
            "these games' FULL solves required per-game hand-engineering across many outer-loop "
            "sessions (e.g. lf52's registry cites a 146-action L3 sequence, a 927-action L8+ "
            "probe) -- so even this 20x-larger budget is not provably sufficient for these "
            "SPECIFIC games; a null result here still narrows rather than closes the open "
            "question, and that must be reported honestly rather than overclaimed as definitive."
        ),
        "acceptance_gate": {
            "condition": "any tested game reaches a real levelup that no prior mechanism "
            "(induce-then-plan, bare reactive-filter, small-budget tool-loop-lookahead) reached",
            "principle": "Every prior mechanism scored 0 new levels across every measurement on "
            "these exact games -- ANY real levelup here is decisive, unambiguous evidence the "
            "larger-budget lookahead closes a real gap.",
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
                "principle": "A single real levelup where every prior mechanism scored exactly 0 "
                "is decisive evidence with real headroom (oracle depths 6-10 levels, well above "
                "the live baseline of 0 for every mechanism tried so far)."
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
