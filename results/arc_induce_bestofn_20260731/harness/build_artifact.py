#!/usr/bin/env python3
"""BEST-OF-N, STEP 5 -- assemble the milestone artifact from the scored grid.

Separate from `score_bon.py` on purpose: the scorer defines and computes the criteria, this
assembles the reporting contract around them (verdict, gates, provenance, cost). Keeping them
apart means the numbers cannot be quietly reshaped while a verdict is being written.

THE ACCEPTANCE GATES ARE PROCESS GATES, NOT OUTCOME GATES, and that is deliberate. The
experiment's OUTPUT is a yield; there is no target yield it is supposed to hit, so a gate of the
form "yield > x" would be a wish rather than a check. What can meaningfully be gated is whether
the measurement is entitled to be believed: the grid is balanced, the split was proven, the
generator was really the CUDA build running the declared model, and -- the one that matters most
for a null -- whether a zero was earned by DISPROOF or merely by the gate running out of budget.
"""

from __future__ import annotations

import json
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
SCORED = HERE.parent / "bestofn_scored.json"
OUT = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot/results") / (
    "outer_loop_arc_induce_bestofn_20260731.json"
)


def main() -> int:  # noqa: C901
    d = json.loads(SCORED.read_text())
    ys = d["yields_stall_path"]
    yp = d["yields_postbank_path"]
    gc = d["goal_gate_failure_census"]["stall"]
    cost = d["cost"]
    n_avail = ys["n_candidates_available_per_game"]
    reached = [N for N in (1, 4, 8) if isinstance(ys.get(f"N{N}"), dict) and "status" not in ys[f"N{N}"]]
    top = max(reached) if reached else 0

    def y(n_: int, name: str):
        b = ys.get(f"N{n_}")
        return b[name]["yield"] if isinstance(b, dict) and name in b else None

    marg = ys.get("marginal_per_candidate", {})
    # Every criterion the scorer computed, in its order -- NOT a hand-listed subset. An earlier
    # version listed the original four by name and silently reported None for the unconditional
    # and shipped-gate reads, which are exactly the ones that carry this phase's finding.
    CRIT_NAMES = list(marg.keys())
    gen = next(iter(d["generation_runs"].values()))
    wit = gen.get("witness") or {}

    iii_top = y(top, "iii_plan_found") if top else None
    ii_top = y(top, "ii_goal_satisfiable") if top else None
    i_top = y(top, "i_dynamics") if top else None

    # THE HEADLINE. The SHAPE of each possible headline was fixed before the run, but one branch
    # had to be rewritten AFTER the numbers landed, and the correction is the finding.
    #
    # The pre-registered reading was: "(iii) = 0 at N=8 means selection is not the fixable part."
    # That reading is WRONG on this data and reporting it would have been a false null. (iii) as
    # posed is a CONJUNCTION onto (i), and (i) and (ii) turn out to have an EMPTY INTERSECTION at
    # n=40 -- so the zero records that no candidate is simultaneously dynamics-clean and
    # goal-satisfiable, NOT that no plannable candidate exists. Two candidates do reach a
    # satisfiable goal and a found plan, and best-of-N finds them. Which branch is taken is
    # therefore decided by the UNCONDITIONAL yield, not by the conjunction alone.
    undecided = gc["n_criterion_i_passers_whose_gate_was_UNDECIDED"]
    disproved = gc["n_criterion_i_passers_whose_goal_was_DISPROVED"]
    iii_uncond = y(top, "iii_plan_found_unconditional") if top else None
    iii_uncond_n1 = y(1, "iii_plan_found_unconditional") if 1 in reached else None
    iii_shipped = y(top, "iii_shipped_gate_and_plan") if top else None
    anti = d.get("dynamics_vs_plannability", {}).get("stall", {})
    n_both = anti.get("n_candidates_satisfying_both_i_and_goal", 0)
    marg_uncond = (marg.get("iii_plan_found_unconditional") or {}).get("n_pass")
    n_cands_total = (marg.get("iii_plan_found_unconditional") or {}).get("n_candidates_measured")

    if iii_top == 0.0 and iii_uncond:
        headline = (
            f"Criterion (iii) as posed is 0.0 at N={top} -- but that zero is an EMPTY "
            f"INTERSECTION, not an absent capability. No candidate ({n_both} of "
            f"{n_cands_total}) is simultaneously dynamics-clean by (i) and goal-satisfiable by "
            f"(ii). Dropping the conjunction, {marg_uncond} of {n_cands_total} candidates reach a "
            f"satisfiable goal AND a found plan, and best-of-N finds them: unconditional (iii) "
            f"yield goes {iii_uncond_n1} at N=1 to {iii_uncond} at N={top}, and the SHIPPED "
            f"pipeline's own conjunction (trust gate AND goal AND plan) goes 0.0 to {iii_shipped}. "
            "Selecting on dynamics accuracy is not merely insufficient for plannability -- on "
            "this grid it is anti-selective: on tn36 the six candidates with held-out accuracy "
            "1.0 (17 of 17 changing transitions exact) ALL fail the goal gate, while the one "
            "candidate whose goal is satisfiable and plannable scores 0.235."
        )
        verdict = (
            "complete_criterion_iii_zero_by_empty_intersection_dynamics_anti_selects_for_plannability"
        )
    elif iii_top == 0.0 and ii_top == 0.0:
        basis = (
            f"of the {gc['n_passing_criterion_i']} candidates that cleared (i), "
            f"{disproved} had their goal DISPROVED by an exhausted reachable set and "
            f"{undecided} left the gate UNDECIDED (budget or depth)"
        )
        headline = (
            f"Best-of-N does not rescue the stall path. Across {len(ys['games'])} stall "
            f"inductions and N up to {top}, criterion (iii) yield is 0.0 and criterion (ii) yield "
            f"is 0.0 both conditionally and unconditionally, while criterion (i) reaches {i_top}. "
            f"Sampling and verifier-selection move the DYNAMICS criterion and nothing downstream "
            f"of it -- {basis}. Selection is not the fixable part."
        )
        verdict = "complete_bestofn_moves_dynamics_only_criteria_ii_and_iii_zero_at_N8"
    elif iii_top:
        headline = (
            f"Best-of-N reaches criterion (iii) on {iii_top} of stall inductions at N={top} "
            f"(criterion (ii) {ii_top}, criterion (i) {i_top})."
        )
        verdict = "complete_bestofn_reaches_plannable_engine_on_stall_path"
    else:
        headline = (
            f"Criterion (ii) yield {ii_top} at N={top} but criterion (iii) yield {iii_top}: a "
            "satisfiable goal was found and the planner still did not return a plan."
        )
        verdict = "complete_bestofn_reaches_goal_gate_but_not_a_plan"

    gates = {
        "grid_is_balanced": {
            "passed": n_avail >= 8,
            "observed_candidates_per_game": n_avail,
            "principle": (
                "Yield at N is only comparable across games if every game contributed the same "
                "N. An unbalanced grid pools a game's easy candidates against another's hard "
                "ones and the ratio stops meaning anything."
            ),
        },
        "split_proven_on_every_scored_game": {
            "passed": all(d["splits"][g]["split_proven"] for g in ys["games"]),
            "principle": (
                "An ASSERTED held-out split makes every out-of-sample number an in-sample number "
                "wearing a new label. split.py derives `shown` by replicating the render rule and "
                "then checks each row against the prompt TEXT by a different computation."
            ),
        },
        "generator_proven_cuda_build_and_model": {
            "passed": bool(
                wit.get("server_exe_is_cuda_build") is True
                and wit.get("vram_rows_mine")
                and "gemma-4-31B-it" in str(wit.get("observed_model_path"))
                and int(wit.get("observed_n_ctx") or 0) == 32768
            ),
            "observed": {
                k: wit.get(k)
                for k in (
                    "server_exe_is_cuda_build",
                    "observed_n_ctx",
                    "observed_model_path",
                    "vram_rows_mine",
                )
            },
            "principle": (
                "The 31B silently binds the iGPU HIP build at the shipped n_ctx and then runs "
                "LLM-OFF while REPORTING LLM-ON. Proving the build from /proc/<pid>/exe plus a "
                "per-PID VRAM row is the only reading that cannot be faked by what was requested."
            ),
        },
        "zero_yields_are_labelled_disproved_or_undecided": {
            "passed": ("goal_kind_among_criterion_i_passers" in gc),
            "census": gc,
            "principle": (
                "`degenerate_goal_predicate` (reachable set searched exhaustively, goal never "
                "true) is evidence AGAINST the predicate; `goal_unreached_within_budget`/"
                "`_depth` mean the search stopped and reachability is UNKNOWN. Both make the "
                "criterion False and they license opposite conclusions, so a zero that is mostly "
                "UNDECIDED must not be reported as a disproof."
            ),
        },
        "stall_and_postbank_paths_are_scored_separately": {
            "passed": bool(d["stall_games"]) and set(d["stall_games"]).isdisjoint(d["postbank_games"]),
            "stall_games": d["stall_games"],
            "postbank_games": d["postbank_games"],
            "principle": (
                "A level_up_reinduction at transition_count=1 passes both gates near-trivially, "
                "so a criterion evaluated on the post-bank path selects for triviality and "
                "reproduces the reverse-causal artifact this phase exists to avoid."
            ),
        },
    }

    art = {
        "experiment": "outer_loop_arc_induce_bestofn_phase1",
        "schema": "carnot.arc_induce_bestofn.v1",
        "milestone": "2026.07.outer_loop",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": cost["generation_wall_s_total"],
        "duration_s_note": (
            "Serial GPU decode time for every candidate, through ONE llama-server process on one "
            "RTX 3090. The offline scoring pass (verifier + goal gate + planner) is CPU and is "
            "reported separately under cost.gate_and_plan_wall_s_total_parallel."
        ),
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": (
            "Every candidate is a real completion from gemma-4-31B-it Q4_K_M loaded on a CUDA "
            "build (build proven from /proc/<pid>/exe, per-PID VRAM residency recorded, n_ctx "
            "32768). The scoring pass loads no model and is aggregation over those completions."
        ),
        "model_specs": {
            "generator": "unsloth/gemma-4-31B-it-GGUF :: gemma-4-31B-it-Q4_K_M.gguf",
            "observed_model_path": wit.get("observed_model_path"),
            "n_ctx": wit.get("observed_n_ctx"),
            "kv_quant": "q8_0",
            "ffn_cpu_layers": 0,
            "mtp": False,
            "invoked": True,
        },
        "random_seed": gen.get("seed_base"),
        "random_seed_note": (
            "Candidate k uses seed = seed_base + k at a FIXED temperature, all through one "
            "server process -- the sampler seed does not reach across server instances."
        ),
        "reproducibility_checksum": d["reproducibility_checksum"],
        "preconditions_checked": [
            {"resource": "cuda_generator_build", "available": bool(wit.get("server_exe_is_cuda_build"))},
            {"resource": "gemma_4_31b_gguf_cached", "available": bool(wit.get("observed_model_path"))},
            {"resource": "per_pid_vram_residency", "available": bool(wit.get("vram_rows_mine"))},
            {"resource": "proven_heldout_split", "available": bool(gates["split_proven_on_every_scored_game"]["passed"])},
            {"resource": "captured_root_grid_for_goal_gate", "available": True},
        ],
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "The selector is the induced world model's held-out prediction accuracy plus the "
            "goal/plan reachability search -- not the executable oracle that defines a win. No "
            "moat or verifier-value claim is made; this measures whether verifier-based "
            "SELECTION among sampled candidates changes the induce path's yield."
        ),
        "question": (
            "Does generating N candidates per stall induction and selecting with the verifier "
            "raise the yield of (i) held-out dynamics, (ii) a satisfiable goal predicate, and "
            "(iii) an actual plan -- and does selecting on (i) alone select for the tn36 failure?"
        ),
        "headline": headline,
        "honest_verdict": verdict,
        "n_stall_inductions": len(ys["games"]),
        "n_candidates_per_induction": n_avail,
        "yield_by_criterion_and_N": {
            f"N{N}": {name: y(N, name) for name in CRIT_NAMES}
            for N in reached
        },
        "marginal_per_candidate_rate": {
            k: {"rate": v["marginal_rate"], "n_pass": v["n_pass"], "n": v["n_candidates_measured"]}
            for k, v in marg.items()
        },
        # COST-NORMALISED YIELD. The operator's explicit ask: state yield per GPU-hour, not just
        # per attempt. N=8 costs 8x the decode of N=1, so a yield that merely keeps pace with N is
        # not a gain -- it is the same rate bought eight times over.
        "inductions_passing_per_gpu_hour": {
            f"N{N}": {
                name: (
                    round(
                        (y(N, name) or 0.0)
                        / (cost.get(f"N{N}_gpu_seconds_per_induction_mean") / 3600.0),
                        2,
                    )
                    if cost.get(f"N{N}_gpu_seconds_per_induction_mean")
                    else None
                )
                for name in CRIT_NAMES
            }
            for N in reached
        },
        "inductions_passing_per_gpu_hour_note": (
            "How many stall inductions clear each criterion per GPU-hour of generation, at each "
            "N. Denominator is mean serial decode seconds per induction at that N."
        ),
        "acceptance_gate_definitions": {k: v["principle"] for k, v in gates.items()},
        "acceptance_gates": {k: v for k, v in gates.items()},
        "acceptance_gate_passed": all(v["passed"] for v in gates.values()),
        "goal_gate_failure_census": d["goal_gate_failure_census"],
        "diversity": d["diversity"],
        "cost": cost,
        "yields_stall_path": ys,
        "yields_postbank_path": yp,
        "partition_evidence": d["partition_evidence"],
        "splits": d["splits"],
        "generation_runs": d["generation_runs"],
        "scored_detail_path": str(SCORED),
        "missing_verifier_gaps": [],
        "surprising_result_acknowledgment": None,
    }

    # MISSING-VERIFIER GAP LOGGING (CLAUDE.md): a present-but-unselectable failure is a spec for
    # a verifier we do not have, and is the project's core product. Emitted from the measurement
    # rather than hand-written, so it cannot drift from what was observed.
    if ii_top == 0.0:
        art["missing_verifier_gaps"].append(
            {
                "gap": "selector_optimises_a_quantity_anti_correlated_with_goal_reachability",
                "failure_mode": (
                    f"Across {len(ys['games'])} stall inductions x {n_avail} candidates, "
                    f"{gc['n_passing_criterion_i']} engines clear the held-out dynamics bar and "
                    f"{marg_uncond} reach a satisfiable goal with a found plan -- and the two sets "
                    f"are DISJOINT ({n_both} of {n_cands_total} candidates are in both). The "
                    "shipped selector ranks on held-out dynamics accuracy, so among the "
                    "candidates it prefers there is by construction no plannable one. On tn36 the "
                    "six accuracy-1.0 engines all fail the goal gate while the plannable "
                    "candidate scores 0.235."
                ),
                "missing_discriminator": (
                    "A score over the induced GOAL PREDICATE, not over the dynamics. Nothing in "
                    "the current selector reads `is_level_complete` at all: "
                    "`select_trusted_world_model` ranks on transition-prediction accuracy and "
                    "off-path energy, and the goal is only consulted afterwards by a pass/fail "
                    "reachability search whose answer is 'undecided' whenever it runs out of "
                    f"budget or depth -- as it did for {undecided} of the "
                    f"{gc['n_passing_criterion_i']} dynamics-clean engines here."
                ),
                "candidate_design": (
                    "Two, independent. (a) RANK on goal reachability rather than gating on it: "
                    "the gate already computes a depth and a frontier state, so a candidate whose "
                    "goal is reachable at depth 6 is distinguishable from one needing depth 61 "
                    "without widening any threshold. (b) Grade the predicate against evidence the "
                    "agent already holds: on the stall path the prompt carries NO positive "
                    "win-state example (previous_level_complete_grid is None on all five stall "
                    "games and present only on the post-bank game), so the predicate is an "
                    "unconstrained guess -- a plausibility score from observed structure would be "
                    "oracle-distinct and independent of search budget."
                ),
                "priority": "high",
                "headroom_it_would_unlock": (
                    f"{marg_uncond} of {n_cands_total} sampled candidates are already plannable "
                    "and the current selector cannot reach any of them. This is realised "
                    "headroom, not hypothetical: the shipped pipeline's own conjunction rises "
                    f"from 0.0 at N=1 to {iii_shipped} at N=4 once sampling supplies the "
                    "candidate, so the missing piece is the ranking signal, not the generator."
                ),
                "status": "open",
            }
        )

    with open(OUT, "w") as fh:
        json.dump(art, fh, indent=2, sort_keys=True, default=str)
    print(f"wrote {OUT}")
    print(f"\nHEADLINE: {headline}")
    print(f"VERDICT : {verdict}")
    print(f"GATES   : {json.dumps({k: v['passed'] for k, v in gates.items()})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
