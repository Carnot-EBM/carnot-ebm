#!/usr/bin/env python3
"""Build the Phase-3 pre-flight results artifact from the cells and the scored payload."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "pf", "cells")
SCORED = os.path.join(HERE, "pf", "preflight_scored.json")
MAIN_REPO = "/home/ianblenke/github.com/ianblenke/carnot"
OUT = os.path.join(
    MAIN_REPO, "results",
    "outer_loop_arc_phase3_preflight_wired_induce_treatment_20260731.json")

sc = json.load(open(SCORED))


def engine_sha(arm: str, game: str):
    fs = sorted(glob.glob(os.path.join(HERE, "pf", "e3", f"{arm}__{game}__s1", "*",
                                       "world_model.py")))
    if not fs:
        return None
    return hashlib.sha256(open(fs[0], "rb").read()).hexdigest()[:16]


cells, total_wall = {}, 0.0
for f in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
    d = json.load(open(f))
    r = d.get("result") or {}
    g, a = d.get("game"), d.get("arm")
    total_wall += float(d.get("cell_wall_s") or 0)
    cells[f"{a}__{g}"] = {
        "arm": a, "game": g, "status": d.get("status"),
        "trace_len": len(r.get("action_trace") or []),
        "total_actions": r.get("total_actions"),
        "n_inductions": r.get("n_inductions"),
        "n_plans_found": r.get("n_plans_found"),
        "mean_heldout_accuracy": r.get("mean_heldout_accuracy"),
        "levels_gained": r.get("levels_gained"),
        "timed_out": r.get("timed_out"),
        "induced_engine_sha256_16": engine_sha(a, g),
        "induce_repeat_penalty_effective": d.get("induce_repeat_penalty_effective"),
        "induce_defect_reasks_allowed": d.get("induce_defect_reasks_effective"),
        "induce_defect_reasks_observed": d.get("n_induce_defect_reasks_observed"),
        "generator_sampler_seed_effective": d.get("generator_sampler_seed_effective"),
        "server_pid": d.get("server_pid"), "server_port": d.get("server_port"),
        "server_exe": d.get("server_exe"), "observed_n_ctx": d.get("observed_n_ctx"),
        "cell_wall_s": d.get("cell_wall_s"),
    }

engine_shas = [v["induced_engine_sha256_16"] for v in cells.values()
               if v["induced_engine_sha256_16"]]

art = {
    "experiment": "outer_loop_arc_phase3_preflight_wired_induce_treatment",
    "title": ("Phase-3 treatment-activation pre-flight REFUSES the banked-levels grid: the "
              "wired induce treatment changes the induced engine on every game and changes "
              "the live agent's actions on none"),
    "run_date": "2026-07-31",
    "git_head": subprocess.run(["git", "-C", MAIN_REPO, "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "inference_substrate": "live_llm_inference",
    "substrate_note": (
        "16 live agent runs against the OFFLINE arcade, each loading gemma-4-31B-it Q4_K_M on "
        "an RTX 3090 and performing real induce inference (3-4 completion calls per cell). "
        "duration_s is the summed per-cell wall clock and is far above the 60s floor."),
    "duration_s": round(total_wall, 1),
    "random_seed": 1,
    "reproducibility_checksum": hashlib.sha256(
        "|".join(sorted(engine_shas)).encode()).hexdigest()[:32],
    "reproducibility_checksum_note": (
        "sha256 over the sorted induced-engine hashes. NOTE it is NOT expected to reproduce: "
        "this artifact's central instrument finding is that the induce output is "
        "nondeterministic run-to-run even at a fixed seed, so a re-run yields different engine "
        "hashes. It fingerprints THIS run's engine set, nothing more."),
    "model_specs": {
        "generator": "unsloth/gemma-4-31B-it-GGUF (Q4_K_M)",
        "n_ctx": 32768, "kv_quant": "q8_0", "ffn_cpu_layers": 0, "mtp": False,
        "why_this_config": (
            "the shipped _default_induce_n_ctx() of 81920 does not fit a 24 GiB card; the "
            "loader then silently binds the iGPU HIP build and the agent runs LLM-OFF while "
            "REPORTING LLM-ON. Every cell proves its CUDA build from /proc/<pid>/exe and its "
            "context from the server's own /props."),
    },
    "preconditions_checked": [
        {"resource": "gemma-4-31B-it GGUF cached", "available": True},
        {"resource": "CUDA llama-server build (not build-hip)", "available": True,
         "evidence": "server_exe read from /proc/<pid>/exe on every cell"},
        {"resource": "per-PID VRAM residency on a discrete 3090", "available": True,
         "evidence": "~21434 MiB on bus 03:00.0 / 62:00.0"},
        {"resource": "CARNOT_ARC_E3_DIR redirected to scratch and empty at start",
         "available": True,
         "evidence": "asserted per cell; the canonical engine store was never written"},
    ],

    "question": (
        "Phase 2 wired repeat_penalty=1.1 (+repeat_last_n=256) and a defect gate with one plain "
        "re-ask into the live code-only induce path. Phase 3 asks the question the planned "
        "12-cell banked-levels grid depends on: does that reach the LIVE AGENT'S ACTIONS?"),

    "verdict": "REFUSE_12_CELL_GRID_UNDERPOWERED",
    "verdict_basis": (
        "not 'too few cells perturb' but ZERO. 4 of 4 comparable A/B pairs are BYTE-IDENTICAL "
        "action traces. A pair whose arms emit identical actions cannot differ on any endpoint "
        "computed downstream of those actions, so no outcome of the grid could reach alpha."),
    "required_one_way_discordant_pairs": sc["required_one_way_discordant_pairs"],
    "planned_n_cells": sc["planned_n_cells"],
    "n_comparable_cells": sc["n_comparable_cells"],
    "attributable_perturbation_rate": sc["attributable_rate"],
    "best_reachable_p_at_planned_size": sc["min_reachable_p_at_planned_size_charitable"],
    "p_planned_grid_reaches_alpha_charitable": sc["p_planned_grid_reaches_alpha_charitable"],
    "perturbation_rate_95pct_upper_bound_given_zero":
        sc["perturbation_rate_95pct_upper_bound_given_zero"],
    "ab_per_cell": sc["ab_per_cell"],
    "aa_ctl_per_cell": sc["aa_ctl_per_cell"],
    "aa_trt_per_cell": sc["aa_trt_per_cell"],

    "the_treatment_was_active_proven_three_ways": {
        "1_flag_read_back_through_shipped_accessor": (
            "induce_repeat_penalty_effective is 1.1 on every trt/trtb cell and 1.0 on every "
            "ctl/ctlb cell, read via e3._induce_repeat_penalty(), not echoed off os.environ"),
        "2_the_induced_engine_differs_on_every_game": (
            f"{len(set(engine_shas))} distinct engine hashes across {len(engine_shas)} cells -- "
            "no two cells produced the same world_model.py"),
        "3_a_maximal_quality_difference_with_zero_action_difference": (
            "on tn36 the treatment's engine scored mean_heldout_accuracy 1.0 where the control "
            "scored 0.0, and BOTH arms took byte-identically the same 31 actions"),
    },

    "where_the_causal_chain_breaks": {
        "finding": (
            "the break is at induce->plan, not at the sampler. n_plans_found is 0 in every "
            "completed cell except one. The ONLY cell whose action trace diverged (ft09, ctl vs "
            "its own replicate ctlb) is the ONLY cell where n_plans_found differed (0 vs 1)."),
        "implication": (
            "the action stream moves when and only when a plan is found; the treatment never "
            "changed whether a plan was found; run-to-run noise changed it once, i.e. MORE than "
            "the treatment did. Improving induce QUALITY cannot move banked levels while the "
            "planner converts ~0 of those engines into a plan."),
        "recommended_next_target": (
            "plan_in_model / the induce->plan conversion, NOT further induce-payload tuning."),
    },

    "the_reask_half_was_never_exercised": {
        "observed": "induce_defect_reasks_observed == 0 in all 16 cells",
        "why": ("the defect gate re-asks only on an engine that FAILS static validation, and "
                "every live induction here passed it"),
        "consequence": (
            "Phase 1's 22-of-36 defective-accept rate did NOT reproduce on the live path. The "
            "re-ask arm of the treatment is UNTESTED by this probe -- this is 'the tier never "
            "fired', not 'the tier did not help', and the two must not be conflated. Its only "
            "support remains Phase 1's 2 of 13 paired wins."),
    },

    "instrument_findings": {
        "seed_does_not_make_induce_deterministic": (
            "ctl vs ctlb ran at an IDENTICAL sampler seed, identical config, identical code and "
            "the SAME server process, and still produced substantively different engine source "
            "(different generated comments and logic, not a timestamp) -- and on ft09 a "
            "different plan outcome. CARNOT_ARC_GENERATOR_SEED is necessary but not sufficient. "
            "This corroborates the prior session's n=2 observation with more cells."),
        "server_process_guard_added": (
            "the scorer now REFUSES to score any pair whose two arms talked to different "
            "llama-server PIDs, because the seed does not hold across processes. Not "
            "hypothetical: both original servers were replaced mid-session by a concurrent "
            "agent. Audited: every scored pair is intra-process."),
        "harness_bug_found_and_fixed_mid_run": (
            "cell.py's ARM_ENV had no 'trtb' entry when the driver first reached it, so "
            "vc33/trtb exited rc=1 with no record. Fixed mid-run and backfilled; the cell is "
            "present and IDENTICAL to its siblings. Recorded because a gap left by the "
            "harness's own defect is not a datum."),
    },

    "tu93_excluded_by_the_guard_reported_not_scored": (
        "tu93's control ran on server PID 1532116 and its treatment on 1562595 (the original "
        "server was replaced mid-session), so the pair is confounded and the guard excludes it. "
        "Reported for completeness and NOT counted anywhere in the verdict: that pair was ALSO "
        "27/27 byte-identical, plans 0 vs 0, heldout 0.0 vs 0.0. It points the same way as the "
        "four scored cells, but a confounded pair is not evidence and is not treated as any."),

    "known_limits_do_not_over_read": [
        "4 comparable cells, not 6. BOTH tu93 cells RAN to completion but on DIFFERENT server "
        "processes, so the guard excludes the pair as confounded; lp85 truncated on BOTH arms "
        "at the 1500s cap. Both are MISSING observations, never zeros.",
        "0 perturbed of 4-5 does NOT mean 'never perturbs': the one-sided 95% upper bound on "
        f"the rate is {sc['perturbation_rate_95pct_upper_bound_given_zero']}. The refusal rests "
        "on the MECHANISM (n_plans_found 0 across arms) at least as much as on the count.",
        "the two omitted games are the two SLOWEST, i.e. the deepest runs where a treatment "
        "acting through the engine would have the most room to express. They were attempted, "
        "not skipped, and they failed for resource reasons.",
        "a PASS would not have meant 'powered'; a REFUSE means only that this grid cannot "
        "produce a finding, not that the treatment is worthless at the induce level -- Phase 1 "
        "measured a real induce-level effect and that stands.",
    ],

    "cells": cells,
    "honest_verdict": (
        "complete_phase3_preflight_refuses_12cell_banked_levels_grid_zero_of_four_comparable_"
        "cells_perturb_treatment_changes_engine_on_every_game_and_actions_on_none_break_is_at_"
        "induce_to_plan"),
    "what_is_not_claimed": (
        "No moat, verifier-value or efficiency claim is made, so verifier_is_oracle does not "
        "apply. No banked level was claimed or reproduced. No scored or online ARC game was "
        "played and nothing was submitted -- every cell ran against the OFFLINE arcade via "
        "arc_solver_kit.offline_arcade(). No flag was changed on the basis of this run."),
}

with open(OUT, "w") as fh:
    json.dump(art, fh, indent=2, sort_keys=True)
print("wrote", OUT)
print("duration_s", art["duration_s"], "| distinct engines",
      len(set(engine_shas)), "of", len(engine_shas), "| verdict", art["verdict"])
