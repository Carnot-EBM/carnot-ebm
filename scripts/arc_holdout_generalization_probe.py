#!/usr/bin/env python3
"""ARC-AGI-3 Generalization-Testing Floor task class 1: held-out live-path measurement on a
never-adaptered public game.

**Researcher summary:**
    Runs the REAL scored live-path mechanism (`E3AgentPolicy` via
    `arc_actions_to_progress.run_bounded_progress`, the same whole-loop
    explore->stall->induce->plan->execute cascade the scored Kaggle submission uses) against a
    game that has NO registered `GameAdapter` in `python/carnot/agentic/arc_game_adapters.py` --
    i.e. no per-game hand-tuned scaffolding exists for it at all. Measures how far the reusable,
    general-purpose live mechanism gets ON ITS OWN, and compares that against the depth the
    per-game hand-derived solve (banked in `ops/arc_solve_registry.yaml`) reached, to see how
    much of the registry's already-solved depth is reachable via generic runtime discovery versus
    per-game specifics that don't transfer.

**Detailed explanation for engineers:**
    Only 3 of the 25 fully-solved public games (sc25, tn36, wa30) have no `GameAdapter` --
    their registry solves were banked via a different one-off mechanism, not the reusable
    `arc_game_adapters.py` scaffolding. This picks `sc25` (registry `levels_reproduced=6`, a
    known-tricky `two_phase_cast_grid_then_tank_exit` mechanic with several documented gotchas
    in `general_gotchas` -- non-idempotent reset, deepcopy-injection unreliable, first-step-
    after-reset consumed -- making it a genuinely informative, not softball, held-out target).

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-5821
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

os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1,0")
# Layer-split across BOTH 3090s: +89.7% decode / +215% prefill vs one card at the
# shipped n_ctx (results/outer_loop_arc_gpu_layer_split_sweep_20260731.json), because it
# avoids the auto-fit's forced CPU offload. Order is "1,0" NOT "0,1": if the conductor
# restarts and holds GPU 0 the split is refused, and the fallback scans this list in
# order -- so the outer loop degrades onto its OWN card (2026-06-27 allocation) rather
# than trying to take the conductor's. setdefault, so an explicit export still wins.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "results", "outer_loop_holdout_generalization_probe_sc25_20260722.json"
)

GAME = "sc25"
SEED = 20260722
CUDA_PORT = 8934  # distinct from exp5720's 8933 to avoid colliding with any leftover server
BUDGET = 120
MAX_INDUCTIONS = 3
WALL_S = 600.0
EXPLORE_BUDGET = 24


def main() -> None:
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_game_adapters import get_adapter

    t0 = time.monotonic()

    preconditions = {
        "game_has_no_registered_adapter": get_adapter(GAME) is None,
        "cuda_gpu1_pinned": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU") == "1",
    }
    if not preconditions["game_has_no_registered_adapter"]:
        artifact = {
            "experiment": "outer_loop_holdout_generalization_probe_sc25_20260722",
            "schema": "carnot.arc_holdout_generalization_probe.v1",
            "honest_verdict": "blocked_game_has_registered_adapter_not_a_valid_holdout_target",
            "duration_s": round(time.monotonic() - t0, 3),
            "preconditions_checked": preconditions,
        }
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
        print("BLOCKED:", artifact["honest_verdict"])
        sys.exit(1)

    print(f"Loading LocalGGUFProposer (Qwen3.5-9B-MTP) on CUDA GPU 1, port {CUDA_PORT}...")
    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        port=CUDA_PORT,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=4096,
        timeout=600,
    )

    print(
        f"Running run_bounded_progress({GAME!r}, arm='frozen', seed={SEED}, "
        f"budget={BUDGET}, max_inductions={MAX_INDUCTIONS}, wall_s={WALL_S})..."
    )
    t_run0 = time.monotonic()
    result = atp.run_bounded_progress(
        GAME,
        "frozen",
        proposer=prop,
        seed=SEED,
        budget=BUDGET,
        max_inductions=MAX_INDUCTIONS,
        wall_s=WALL_S,
        explore_budget=EXPLORE_BUDGET,
    )
    run_duration_s = round(time.monotonic() - t_run0, 3)

    # Read the registry's per-game hand-derived depth for comparison. This is read, never
    # invoked -- the held-out run above has zero access to sc25's registry entry, its solver
    # module, or any per-game gotcha -- reading it AFTER the run for comparison only does not
    # contaminate the held-out measurement.
    import yaml

    with open(os.path.join(REPO_ROOT, "ops", "arc_solve_registry.yaml")) as f:
        registry = yaml.safe_load(f)
    registry_entry = next((g for g in registry["games"] if g.get("game") == GAME), None)
    registry_levels_reproduced = registry_entry.get("levels_reproduced") if registry_entry else None

    duration_s = round(time.monotonic() - t0, 3)
    result_dict = {
        "game": result.game,
        "arm": result.arm,
        "seed": result.seed,
        "start_level": result.start_level,
        "reached_level": result.reached_level,
        "levels_gained": result.levels_gained,
        "solved": result.solved,
        "actions_to_first_solve": result.actions_to_first_solve,
        "total_actions": result.total_actions,
        "noop_frac": result.noop_frac,
        "revisit_frac": result.revisit_frac,
        "start_hv": result.start_hv,
        "best_hv": result.best_hv,
        "hv_progress": result.hv_progress,
        # PER-LEVEL progress (REQ-ARC-WMTE-6045). `hv_progress` / `best_hv` above are GLOBAL and
        # their baseline was never reset at a level-up, so on a run that levelled up they score a
        # later level's board against the FIRST level's starting distance. Kept for continuity;
        # `hv_progress_best_level` is the figure to read.
        "hv_progress_per_level": result.hv_progress_per_level,
        "hv_progress_best_level": result.hv_progress_best_level,
        "n_inductions": result.n_inductions,
        "n_plans_found": result.n_plans_found,
        "plan_found_rate": result.plan_found_rate,
        "mean_heldout_accuracy": result.mean_heldout_accuracy,
        "mean_prefix_accuracy": result.mean_prefix_accuracy,
        "wall_s": result.wall_s,
        "timed_out": result.timed_out,
        "hit_induction_cap": result.hit_induction_cap,
        "error": result.error,
    }

    if result.error is not None:
        verdict = "complete_holdout_run_crashed_see_error_field_a_crash_is_itself_a_datum"
    elif result.solved:
        verdict = (
            "complete_holdout_generic_mechanism_reached_a_real_levelup_without_per_game_scaffolding"
        )
    elif (result.n_inductions or 0) == 0:
        verdict = "complete_holdout_never_stalled_into_induction_within_budget_generic_mechanism_untested_by_this_run"
    else:
        verdict = "complete_holdout_induced_and_planned_but_did_not_reach_a_levelup_within_budget"

    checksum_input = json.dumps(result_dict, sort_keys=True).encode()
    reproducibility_checksum = hashlib.sha256(checksum_input).hexdigest()

    artifact = {
        "experiment": "outer_loop_holdout_generalization_probe_sc25_20260722",
        "schema": "carnot.arc_holdout_generalization_probe.v1",
        "run_date": "2026-07-22",
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "live_agent_self_discovery",
        "target_model": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "random_seed": SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": duration_s,
        "duration_note": f"Includes GGUF server load + the bounded live e3 run itself "
        f"({run_duration_s}s for run_bounded_progress alone).",
        "honest_verdict": verdict,
        "narrative": (
            f"Held-out live-path measurement (ARC-AGI-3 Generalization-Testing Floor task "
            f"class 1): {GAME} has NO registered GameAdapter -- its registry solve was banked "
            f"via a different one-off mechanism, not arc_game_adapters.py's reusable "
            f"scaffolding. This run drives the REAL scored E3AgentPolicy cascade (the same "
            f"mechanism the Kaggle submission uses) against {GAME} cold, with zero per-game "
            f"hand-tuning, to measure how far the generic reusable mechanism gets on its own. "
            f"The registry's hand-derived depth for {GAME} is levels_reproduced="
            f"{registry_levels_reproduced} (read AFTER this run, for comparison only -- this "
            f"run had no access to the registry entry, solver module, or any per-game gotcha)."
        ),
        "comparison_to_registry_hand_derived_depth": {
            "registry_levels_reproduced": registry_levels_reproduced,
            "holdout_run_reached_level": result.reached_level,
            "holdout_run_levels_gained": result.levels_gained,
        },
        "result": result_dict,
        "preconditions_checked": preconditions,
        "field_provenance": {
            "solve_provenance": {
                "principle": "The live agent's own runtime attempt IS the deliverable per "
                "CLAUDE.md's ARC Live-Path Reachability Discipline self-solve provenance "
                "contract -- this run used zero outer-loop RE, zero source-reading, zero "
                "per-game adapter."
            }
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"verdict: {verdict}")
    print(
        f"reached_level={result.reached_level} levels_gained={result.levels_gained} "
        f"n_inductions={result.n_inductions} plan_found_rate={result.plan_found_rate} "
        f"registry_levels_reproduced={registry_levels_reproduced}"
    )


if __name__ == "__main__":
    main()
