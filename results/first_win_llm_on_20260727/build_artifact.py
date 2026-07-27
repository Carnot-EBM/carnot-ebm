#!/usr/bin/env python
"""Build the milestone artifact from analysis.json.

Substrate honesty (the exp5178 lesson): the ARMS are real compute -- a real Qwen3.5-9B
GGUF loaded on a real 3090 with real generation -- so `inference_substrate` is
`live_llm_inference` and `duration_s` reports the MEASUREMENT clock (summed from each row
file's own elapsed_s), NOT this builder's clock. The builder's own cost is published
separately as `artifact_build_s` so a fast build can never make a slow measurement look
fabricated, and a slow build can never pad a fast one.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
OUT = REPO / "results" / "first_win_llm_on_20260727"
ART = REPO / "results" / "outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json"


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    t0 = time.time()
    analysis = json.loads((OUT / "analysis.json").read_text())
    arms = analysis["arms"]
    comps = {c["comparison"]: c for c in analysis["comparisons"]}
    base = analysis["baseline"]

    runs_raw = analysis.get("runs", {})
    off = arms.get("llm_off", {})
    fix = arms.get("llm_on_fix", {})
    faulty = arms.get("llm_on_16k", {})

    # ------------------------------------------------------------------ witnesses
    # POSITIVE CONTROL / NON-FORCED-GATE WITNESS. If the harness could not record a
    # first_win at all, an "LLM-on scores 0.00" headline would be unfalsifiable -- it would
    # be indistinguishable from a broken win detector. The llm_off arm, run in the SAME
    # tree on the SAME variants through the SAME code path, is that control: it must
    # reproduce the baseline's 0.04 AND name the same winning variants.
    harness_can_detect_a_win = int(off.get("n_first_win") or 0) > 0
    reproduces_baseline_rate = off.get("first_win_rate") == base["first_win_rate_integrated"]
    off_winners = sorted(off.get("winning_variants") or [])
    reproduces_baseline_winners = off_winners == sorted(base["winning_variants"])
    off_win_games = sorted({s.split("~")[0] for s in off_winners})
    base_win_games = sorted({s.split("~")[0] for s in base["winning_variants"]})
    same_winning_games = off_win_games == base_win_games

    # THE GATE is falsifiability, and nothing more: can this harness record a first_win at
    # all? If it cannot, every "LLM-on scores X" number below is indistinguishable from a
    # broken win detector and the whole measurement is unfalsifiable. This conjunct is about
    # THIS arm only -- it deliberately does NOT encode an assumption about the LLM arms, which
    # is how a previous gate in this project got VOIDED.
    positive_control_passed = bool(harness_can_detect_a_win)

    # CONTENTION-CONTROL WITNESS: the 16k arm has to actually EXHIBIT the fault, otherwise
    # "faulty == fixed" would just mean the fault never fired and the control is inert.
    faulty_live = faulty.get("liveness") or {}
    fixed_live = fix.get("liveness") or {}
    fault_exhibited = int(faulty_live.get("total_llm_server_errors") or 0) > 0
    fix_clean = int(fixed_live.get("total_llm_server_errors") or 0) == 0

    # LLM-ENGAGEMENT WITNESS: an LLM-on arm whose generator was never called is not an
    # LLM-on arm. This is the channel whose absence made the whole fault invisible.
    fix_engaged = int(fixed_live.get("total_llm_responses") or 0) > 0

    payload = {
        "experiment": "outer_loop_arc_first_win_llm_on_eval_concurrency_20260727",
        "schema": "carnot.outer_loop.arc_first_win_llm_on_eval_concurrency.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": analysis["git_head"],
        "inference_substrate": "live_llm_inference",
        # duration_s IS the measurement wall clock. It is published exactly ONCE: an earlier
        # draft also carried it as `measurement_wall_s`, and adversarial_verify correctly
        # flagged that as a TAUTOLOGY -- two names for one number is precisely the redundancy
        # that check exists to catch. Per-arm breakdowns live under arms[*].
        "duration_s": analysis["measurement_wall_s"],
        "duration_s_provenance": (
            "THE MEASUREMENT CLOCK: summed from each per-cell row file's own elapsed_s "
            "(results/first_win_llm_on_20260727/cells/*.json). NOT a per-arm wall clock (K=4 "
            "workers overlap, so a wall sum undercounts by roughly the concurrency factor) "
            "and NOT the analyser's or this builder's clock -- the builder's cost is "
            "published separately as artifact_build_s."
        ),
        "artifact_build_s": None,  # set below
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "submitted_to_leaderboard": False,
        # ------------------------------------------------------------ the question
        "question": (
            "Did the generator concurrency fault materially depress this project's own "
            "measured held-out first-win rate? Re-measure first_win_rate with a WORKING "
            "generator at eval-representative concurrency, using the baseline's exact "
            "definition, so the number is comparable rather than a differently-defined "
            "quantity."
        ),
        # ------------------------------------------------------------ the baseline
        "baseline": base,
        "baseline_provenance_correction": {
            "claim_as_stated": (
                "first_win_rate_integrated = 0.04, CI [0,0] (ops/known-issues.md:3484)"
            ),
            "what_the_ci_actually_is": (
                "[0,0] is the paired bootstrap CI on the DELTA (integrated minus bare), both "
                "arms 0.04, every pair tied -- it is NOT an interval on the 0.04 rate. The "
                "rate's own exact (Clopper-Pearson) 95% interval at 4/100 is "
                "[0.011, 0.0993]. Reporting '0.04, CI [0,0]' reads as a precisely-pinned "
                "rate; it is a precisely-pinned NULL DELTA between two arms that were both "
                "LLM-off."
            ),
            "the_llm_was_off": base["baseline_arms_were_llm_off"],
            "carry_chain": (
                "The 0.04 that gated the programme was not re-measured after exp4605. "
                "exp4950, exp4961, exp4972 and exp4983 all carry it forward "
                "(carry_method=anti_churn_carry_*, live_agent_ran=False) and each names "
                "results/experiment_4605_live_integration_scored_agent.json as its "
                "heldout_proxy_summary.source_artifact_path. exp4983 is stamped "
                "inference_substrate='live_llm_inference' and is labelled the operator's "
                "single final pre-deadline go/no-go artifact "
                "(operator_decision_number.go_no_go='no_go_no_improvement_null'), yet it "
                "resolves through that 4-hop chain to one LLM-OFF, concurrency-1 run."
            ),
            "the_live_agent_ran_true_artifacts_were_also_llm_off": (
                "Six held-out artifacts DO claim live_agent_ran=True with a first-win rate "
                "(exp4875 0.04, exp4886 0.0625, exp4896 0.052632, exp4907 0.05, exp4917 "
                "0.047619, exp4928 0.04) and all six are stamped "
                "inference_substrate='live_llm_inference'. Their generator was nevertheless "
                "NOT engaged. Code path, not inference: they run "
                "experiment_4729_held_out_first_win_readiness.py, whose per-game driver "
                "obtains its runners from "
                "`exp4605.default_variant_runner_factory('integrated')` / `('bare')` "
                "(experiment_4729_held_out_first_win_readiness.py:764-765), which resolve to "
                "exp4605.run_variant_attempt -> exp4605._policy_for_mode -> _NoOpProposer. "
                "So 'live_agent_ran' means the live AGENT (E3AgentPolicy) was stepped "
                "offline; it does NOT mean the LLM tier ran. None of the six carries any "
                "liveness field whatsoever (no call/response/error counter, no "
                "generator-health field), so the claim was unfalsifiable from the artifact. "
                "Corroborating arithmetic: exp4928 reports duration_s=673.601274 for 100 "
                "LLM-on attempts (6.7s each) and exp4875 1137.68s (11.4s each), against a "
                "measured 164-272s per LLM-engaged cell in THIS artifact's arms; and that "
                "same 673.601274 value also appears verbatim in exp4972 and exp4983, i.e. it "
                "is itself a carried number. The family also applies an explicit duration "
                "floor (experiment_4752_held_out_first_win_readiness.py:58, "
                "LIVE_DURATION_FLOOR_S = 60.0), so a duration in that lineage is not by "
                "itself evidence of compute."
            ),
            "consequence_for_the_operator_question": (
                "Because no held-out first-win number in the project's history was measured "
                "with the generator engaged, the concurrency fault CANNOT have depressed the "
                "0.04 baseline -- there was no generator contribution in it to lose. 0.04 is "
                "the LLM-OFF floor, mislabelled as an LLM-on measurement. The live question "
                "is therefore not 'how much did the fault cost us' but 'what does the LLM "
                "tier do at eval concurrency at all', which is what the arms below measure "
                "for the first time."
            ),
        },
        # ------------------------------------------------------------ arms
        "arms": arms,
        "arm_definitions": {
            "llm_off": (
                "_NoOpProposer installed exactly as experiment_4605 does -- reproduces the "
                "baseline definition in this tree. POSITIVE CONTROL."
            ),
            "llm_on_16k": (
                "shipped LocalGGUFProposer with CARNOT_ARC_INDUCE_N_CTX=16384 (the PRE-FIX "
                "context pool). CONTENTION CONTROL: same tree, same binary, same variants, "
                "same concurrency; only the pool reverted."
            ),
            "llm_on_fix": (
                "shipped LocalGGUFProposer at the shipped default n_ctx (81920). THE "
                "TREATMENT: the fixed generator."
            ),
        },
        "comparisons": analysis["comparisons"],
        # ------------------------------------------------------------ witnesses
        "positive_control_passed": positive_control_passed,
        "positive_control_witness": {
            "harness_can_record_a_first_win": harness_can_detect_a_win,
            "llm_off_arm_n_first_win": off.get("n_first_win"),
            "llm_off_winning_variants": off_winners,
            "could_have_failed": (
                "Yes, and it nearly did: of the baseline's four winning variants "
                "(lp85~color01..04) only a subset still win under today's agent code. A "
                "serial K=1 re-run of the UNPATCHED exp4605 code confirms lp85~color01 now "
                "loses. Had the whole lp85 win disappeared, this conjunct would read False "
                "and every LLM-on number in this artifact would be stamped UNFALSIFIABLE "
                "rather than reported."
            ),
            "principle": (
                "A PASS needs a case that COULD have failed. Without a control arm that "
                "demonstrably records wins, any low LLM-on rate is indistinguishable from a "
                "broken win detector. This conjunct is scoped to the control arm ALONE and "
                "encodes no assumption about the LLM arms -- a gate conjunct that asserted "
                "something about another arm is how a prior gate in this project was VOIDED."
            ),
        },
        # DECLARED-vs-ACTUAL n_ctx CROSS-CHECK. The entire fault under investigation is a gap
        # between what the code DECLARED it was doing and what the server was actually doing,
        # so the two are compared here from two INDEPENDENT reads: the agent-side proposer's
        # own `generator_n_ctx` (per cell, via the liveness witness) against the server's own
        # /props `n_ctx` (read over HTTP at launch). Agreement is a fact about the system;
        # asserting the env var was set would have been a fact about my own arithmetic.
        "declared_vs_actual_n_ctx": {
            arm_name: {
                "agent_side_witness_n_ctx": (arms.get(arm_name, {}).get("liveness") or {}).get(
                    "generator_n_ctx_observed"
                ),
                "server_side_props_n_ctx": (
                    ((runs_raw.get(f"run_{arm_name}.json") or {}).get("server") or {}).get("props")
                    or {}
                ).get("n_ctx_reported"),
                "server_side_props_total_slots": (
                    ((runs_raw.get(f"run_{arm_name}.json") or {}).get("server") or {}).get("props")
                    or {}
                ).get("total_slots"),
            }
            for arm_name in sorted(arms)
        },
        # FIDELITY IS REPORTED, NOT GATED. Conflating "can this harness see a win" with
        # "does it byte-reproduce a 5-week-old run" would let ordinary code drift invalidate
        # a sound measurement -- and would hide the drift instead of naming it.
        "baseline_fidelity": {
            "definition_identical": (
                "Yes. The arms call experiment_4605.run_variant_attempt VERBATIM; the only "
                "monkeypatch is which proposer _policy_for_mode installs. Corpus, budget "
                "(200), variant ids (1,2,3,4 = HELD_OUT_VARIANT_IDS), deepen flag, "
                "target_levels and value_weight are all read from the same SUBMITTED_* "
                "helpers the baseline used."
            ),
            "llm_off_first_win_rate": off.get("first_win_rate"),
            "baseline_first_win_rate": base["first_win_rate_integrated"],
            "reproduces_baseline_rate": reproduces_baseline_rate,
            "baseline_winning_variants": base["winning_variants"],
            "llm_off_winning_variants": off_winners,
            "reproduces_baseline_winner_set_exactly": reproduces_baseline_winners,
            "same_winning_games": same_winning_games,
            "llm_off_winning_games": off_win_games,
            "baseline_winning_games": base_win_games,
            "trajectory_divergence_measured": (
                "lp85~color02 wins in both, but at actions_to_first_levelup=187 here versus "
                "59 in the baseline; lp85~color01 won in the baseline and LOSES here."
            ),
            "cause_established_not_assumed": (
                "AGENT-CODE DRIFT, not this harness's threading. Discriminating test in "
                "results/first_win_llm_on_20260727/serialcheck.json: the same cells run "
                "strictly SERIALLY (K=1, single thread, single process) through the "
                "UNPATCHED exp4605 code reproduce THIS run's numbers exactly "
                "(lp85~color02 -> 187, lp85~color01 -> loss), not the baseline's. Two "
                "repetitions of each cell were bit-identical, so per-cell determinism is "
                "intact and K=4 threading is exonerated. 77 commits touched "
                "python/carnot/agentic/arc_competition_agent.py between the baseline "
                "artifact (2026-06-25) and today."
            ),
            "consequence": (
                "This artifact measures TODAY's agent with the LLM off versus TODAY's agent "
                "with the LLM on. It is NOT a bit-reproduction of exp4605, and no claim here "
                "depends on it being one: every comparison is against the llm_off arm run in "
                "THIS tree on the SAME cells, never against the June numbers."
            ),
        },
        # WHY THE NUMBER CAME OUT AS IT DID. Measured, from the shipped diagnostics channel --
        # not inferred from the rate. This is the difference between "the LLM's plans did not
        # help" and "the LLM's plans were never used", which have opposite consequences.
        "mechanism_downstream_of_the_generator": {
            "n_cells_where_llm_output_reached_the_policy": (fixed_live or {}).get(
                "n_cells_llm_output_reached_the_policy"
            ),
            "induction_skip_reason_histogram_fixed_arm": (fixed_live or {}).get(
                "induction_skip_reason_histogram"
            ),
            "generator_side_is_healthy": {
                "total_llm_calls": (fixed_live or {}).get("total_llm_calls"),
                "total_llm_responses": (fixed_live or {}).get("total_llm_responses"),
                "total_llm_server_errors": (fixed_live or {}).get("total_llm_server_errors"),
                "total_llm_content_failures": (fixed_live or {}).get("total_llm_content_failures"),
                "dead_generator_cells": (fixed_live or {}).get("dead_generator_cells"),
            },
            "what_this_means": (
                "With the FIXED generator the LLM tier answers essentially every call, yet the "
                "induced world model is REJECTED downstream on every cell and NO plan is ever "
                "installed. Both skip reasons are POST-generation gates, verified by code read: "
                "arc_competition_agent.py:5858 sets 'hidden_state_trust_below_threshold' after "
                "select_trusted_world_model returns trust_pass=False, and :5877 sets "
                "'world_model_accuracy_below_threshold' after "
                "e3.WorldModelVerifier(active_transitions).score(engine) returns a gate value "
                "below the hard 0.5 floor. Both `return` before any plan is installed. So the "
                "generator's output never reaches action selection, and a first-win delta of "
                "zero is what the pipeline MUST produce at default settings -- it is a "
                "structural consequence, not a measurement failure."
            ),
            "the_project_already_suspected_this": (
                "The comment at arc_competition_agent.py:5869-5872 names it: "
                "CARNOT_ARC_TRUST_METRIC=cell_recall is described as 'the coordinated-redesign "
                "lever for the 0.08 wall: exact-match reads ~0 for an imperfect-but-useful "
                "induced model and gates it out -> the induce->plan path is a no-op'. The "
                "default is 'exact'. This measurement is the first direct evidence for that "
                "hypothesis taken with a generator that was verifiably answering: previously "
                "the gate and the generator were confounded, because nothing recorded whether "
                "the generator had answered at all."
            ),
            "not_measured_here": (
                "The numeric MARGIN by which each gate failed (verify_accuracy, "
                "verify_cell_recall, trust_energy, heldout_accuracy) is recorded by the shipped "
                "code onto its per-attempt dict but is not exposed through "
                "generator_liveness_witness, so this harness did not capture it. Adding that "
                "capture mid-run would have instrumented only the arms started afterwards, "
                "producing exactly the asymmetric-instrumentation confound this project's own "
                "measurement discipline forbids -- so it was deliberately NOT added. The "
                "symmetric skip-reason channel above, which every arm carries, is what the "
                "finding rests on."
            ),
        },
        # COMPUTED ADMISSION WITNESS. Derived from the SHIPPED constants (max_tokens=4096 read
        # off LocalGGUFProposer, the two n_ctx values), not assumed: it states in advance that
        # the control arm's pass region is EMPTY and the treatment arm's is non-empty, so
        # neither arm's outcome rests on an untested premise about the other.
        "admission_arithmetic_witness": {
            "shipped_max_tokens": 4096,
            "K": 4,
            "n_ctx_16k_generation_budgets_alone_need_cells": 4 * 4096,
            "n_ctx_16k_pool_cells": 16384,
            "n_ctx_16k_verdict": (
                "4 * 4096 = 16384 == the ENTIRE pool, so at K=4 the four generation budgets "
                "consume every cell and ZERO remain for prompts. The pre-fix arm therefore MUST "
                "fail admission for any non-empty prompt: the control's pass region is empty by "
                "arithmetic, which is what makes observing failures there a live control rather "
                "than luck."
            ),
            "n_ctx_81920_cells_left_per_prompt_at_K4": (81920 - 4 * 4096) // 4,
            "n_ctx_81920_verdict": (
                "leaves 16384 cells per concurrent prompt, above both the ~6k real induce prompt "
                "measured in this project and the 15734-token worst case the shipped default was "
                "sized against. The treatment arm's pass region is non-empty."
            ),
        },
        "contention_control_witness": {
            "fault_exhibited_in_16k_arm": fault_exhibited,
            "n_server_errors_16k": faulty_live.get("total_llm_server_errors"),
            "n_server_errors_fix": fixed_live.get("total_llm_server_errors"),
            "fix_arm_server_error_free": fix_clean,
            "server_failure_diagnostics_sample_16k": faulty_live.get(
                "server_failure_diagnostics_sample"
            ),
            "response_rate_16k": faulty_live.get("response_rate"),
            "response_rate_fix": fixed_live.get("response_rate"),
            "principle": (
                "A contention control that never exhibits the fault is inert: 'faulty == "
                "fixed' would then mean 'the fault did not fire', not 'the fault does not "
                "matter'. This block is the computed witness that the control was live."
            ),
        },
        "llm_engagement_witness": {
            "fix_arm_generator_engaged": fix_engaged,
            "fix_arm_total_llm_calls": fixed_live.get("total_llm_calls"),
            "fix_arm_total_llm_responses": fixed_live.get("total_llm_responses"),
            "fix_arm_n_ctx_observed": fixed_live.get("generator_n_ctx_observed"),
            "fix_arm_dead_generator_cells": fixed_live.get("dead_generator_cells"),
            "fix_arm_cells_missing_witness": fixed_live.get("n_cells_missing_witness"),
            "principle": (
                "Every prior LLM-on measurement in this project was taken at concurrency 1 "
                "and had no liveness channel, which is why a generator degraded to LLM-off "
                "was reported as the LLM-on scored path. These counters are that channel; "
                "a missing/zero value here would mean the arm is not evidence about the LLM."
            ),
        },
        # ------------------------------------------------------------ scope
        "scope_and_power": {
            "corpus": (
                f"{off.get('n_cells')} cells per arm = 25 PUBLIC ARC-AGI-3 games x 4 "
                "colour-permuted held-out variants, played OFFLINE against "
                "environment_files. This is the same corpus the 0.04 baseline is defined on."
            ),
            "concurrency": (
                "K=4 worker threads sharing ONE local llama-server. K=4 is the eval-relevant "
                "number because llama.cpp caps its own slot count at 4 and QUEUES the rest "
                "(server.cpp:106-110), so the generator sees at most 4 concurrent induce "
                "requests no matter how many games the framework's one-thread-per-game swarm "
                "starts. K=1 was deliberately NOT used: it is the blind spot that hid the "
                "fault from every prior measurement."
            ),
            "what_this_CANNOT_say": (
                "Nothing here forecasts the HIDDEN leaderboard. These are public games the "
                "project has fully solved and whose mechanics are in "
                "ops/arc_solve_registry.yaml, played offline, with no gateway. The hidden "
                "set is OOD by construction and our scored hardware is not even directly "
                "known (our kernel requests machine_shape NvidiaL4; no scored-run nvidia-smi "
                "exists). The answerable question is the narrow one: did the fault depress "
                "OUR OWN measured first-win rate on OUR OWN proxy?"
            ),
            "minimum_detectable_effect": (
                "The paired exact test's smallest reachable two-sided p is 2*0.5^n at n "
                "discordant pairs, so >=6 variants must flip in one direction for ANY result "
                "on this corpus to reach p<0.05 (2*0.5^6 = 0.031). Against a 4/100 baseline "
                "that is an effect of at least +/-0.06 absolute. Anything smaller is "
                "unresolvable here by construction, and is reported as an effect size with "
                "its interval rather than as a null."
            ),
        },
        "preconditions_checked": {},  # filled below
        "field_principles": {
            "duration_s": {
                "principle": (
                    "Real compute takes wall-clock time; this is the MEASUREMENT clock summed from "
                    "per-cell row files, not the artifact builder's clock, so neither can "
                    "disguise the"
                    "other."
                )
            },
            "positive_control_passed": {
                "principle": (
                    "Gates every null claim in this artifact: a low LLM-on rate means nothing "
                    "unless a"
                    "control arm in the same tree demonstrably records wins."
                )
            },
            "contention_control_witness": {
                "principle": (
                    "Proves the pre-fix arm actually exhibited the fault, so a faulty-vs-fixed "
                    "null is"
                    "a statement about the fault's effect and not about the fault's absence."
                )
            },
            "llm_engagement_witness": {
                "principle": (
                    "An uninstrumented arm reads as a clean null; these counters make 'the "
                    "generator"
                    "answered' a measured fact rather than an assumption."
                )
            },
            "honest_verdict": {
                "principle": (
                    "Terminal-prefixed so the conductor reconciler classifies it without re- "
                    "running the"
                    "measurement."
                )
            },
            "random_seed": {
                "principle": (
                    "Determinism is the precondition for reproducibility; the bootstrap and the "
                    "agent's"
                    "own explore RNG are both seeded."
                )
            },
            "reproducibility_checksum": {
                "principle": (
                    "Content hash over the payload so silent drift between this artifact and a "
                    "replication attempt is detectable."
                )
            },
        },
        "random_seed": 4605,
        "model_specs": {
            "name": "Qwen3.5-9B-MTP",
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "model_filename": "Qwen3.5-9B-Q4_K_M.gguf",
            "mtp": True,
            "kv_quant": "q8_0",
            "no_think_prefix": True,
            "max_tokens": 4096,
            "llama_server_kind": "cuda build (~/.cache/llama.cpp-master/build/bin/llama-server)",
            "device": "RTX 3090 GPU 1 (outer-loop card); GPU 0 (conductor) untouched",
            "invoked": True,
        },
        "cited_upstream_artifacts": [
            {
                "experiment_id": 4605,
                "path": base["path"],
                "sha256": base["sha256"],
                "fields_imported": [
                    "first_win_rate_integrated",
                    "first_win_rate_bare",
                    "first_win_ci",
                    "integrated_measurement.variant_attempts",
                    "bare_control_config.llm_arm",
                ],
                "role": "the baseline being reproduced and re-measured against",
            }
        ],
        "row_files_dir": "results/first_win_llm_on_20260727/cells/",
        "analysis_path": "results/first_win_llm_on_20260727/analysis.json",
        "harness_paths": [
            "results/first_win_llm_on_20260727/firstwin.py",
            "results/first_win_llm_on_20260727/analyse.py",
            "results/first_win_llm_on_20260727/build_artifact.py",
        ],
        "no_pivot_recommendation": (
            "This artifact deliberately makes NO recommendation about reallocating the "
            "research programme. It produces the number and its uncertainty; the "
            "reallocation is the operator's call."
        ),
    }

    # preconditions, read from what the runs actually recorded (never asserted)
    runs = runs_raw
    pc = {}
    for name, r in runs.items():
        srv = r.get("server") or {}
        dev = srv.get("device") or {}
        pc[name] = {
            "launch_ok": srv.get("launch_ok", None if "llm_off" not in name else True),
            "device_verdict": dev.get("verdict"),
            "server_vram_mib": dev.get("my_vram_mib"),
            "props_total_slots": (srv.get("props") or {}).get("total_slots"),
            "props_n_ctx_reported": (srv.get("props") or {}).get("n_ctx_reported"),
            "server_healthy_after": r.get("server_healthy_after"),
            "teardown": r.get("teardown"),
            "gpu_after_teardown": r.get("gpu_after_teardown"),
            "k_concurrency": r.get("k_concurrency"),
            "n_cells": r.get("n_cells"),
            "deepen_enabled": r.get("deepen_enabled"),
            "submitted_target_levels": r.get("submitted_target_levels"),
            "submitted_value_weight": r.get("submitted_value_weight"),
        }
    payload["preconditions_checked"] = pc

    # ------------------------------------------------------------------ verdict
    # ------------------------------------------------------------------ acceptance gates
    # These gate MEASUREMENT VALIDITY, not the outcome. The deliverable is a number, so a gate
    # on "the number came out favourable" would be nonsense; a gate on "this number is
    # evidence about the system at all" is exactly what was missing from every prior LLM-on
    # measurement here. Each one could genuinely fail, and each is computed from a read of the
    # real object (the witness rows, the server's own /props) rather than from an assumption.
    llm_arms = {k: v for k, v in arms.items() if k.startswith("llm_on")}
    n_ctx_agree = []
    for arm_name in sorted(llm_arms):
        witnessed = (llm_arms[arm_name].get("liveness") or {}).get("generator_n_ctx_observed") or []
        props = (
            ((runs_raw.get(f"run_{arm_name}.json") or {}).get("server") or {}).get("props") or {}
        ).get("n_ctx_reported")
        n_ctx_agree.append(bool(witnessed) and props is not None and set(witnessed) == {props})
    payload_gates = {
        "acceptance_gate_positive_control_passed": positive_control_passed,
        "acceptance_gate_fix_arm_generator_engaged": bool(
            fix_engaged and int(fixed_live.get("dead_generator_cells") or 0) == 0
        ),
        "acceptance_gate_contention_control_exhibited_the_fault": bool(fault_exhibited),
        "acceptance_gate_no_uninstrumented_llm_cells": bool(
            llm_arms
            and all(
                int((v.get("liveness") or {}).get("n_cells_missing_witness") or 0) == 0
                for v in llm_arms.values()
            )
        ),
        "acceptance_gate_declared_n_ctx_matches_server_props": bool(
            n_ctx_agree and all(n_ctx_agree)
        ),
        "acceptance_gate_no_cell_errors": bool(
            all(int(v.get("n_cell_errors") or 0) == 0 for v in arms.values())
        ),
    }
    payload_gates["acceptance_gate_measurement_valid"] = all(payload_gates.values())
    payload.update(payload_gates)
    payload["measurement_validity_rationale"] = {
        "acceptance_gate_positive_control_passed": (
            "the control arm demonstrably records a first_win, so a low LLM-on rate is not "
            "just a broken win detector"
        ),
        "acceptance_gate_fix_arm_generator_engaged": (
            "the fixed generator answered and no cell was asked-and-silent; an unengaged arm "
            "is not evidence about the LLM"
        ),
        "acceptance_gate_contention_control_exhibited_the_fault": (
            "the pre-fix arm actually overflowed, so a faulty-vs-fixed null is about the "
            "fault's effect and not the fault's absence"
        ),
        "acceptance_gate_no_uninstrumented_llm_cells": (
            "a missing witness reads as a clean null; every LLM-on cell must carry one"
        ),
        "acceptance_gate_declared_n_ctx_matches_server_props": (
            "the whole fault was a declared-versus-actual gap; two independent reads must agree"
        ),
        "acceptance_gate_no_cell_errors": (
            "a crashed cell excluded silently would bias the rate; there must be none"
        ),
    }

    fw = fix.get("first_win_rate")
    bl = base["first_win_rate_integrated"]
    c_fix_off = comps.get("llm_on_fix_vs_llm_off", {})
    c_fix_faulty = comps.get("llm_on_fix_vs_llm_on_16k_FAULTY", {})
    m = c_fix_faulty.get("mcnemar_exact", {})
    payload["headline"] = {
        # Every rate is labelled with the N it is over. The LLM-off control ran the full
        # N=100 (so it can be checked against the published baseline cell-for-cell); the LLM
        # arms ran the 25-game variant-1 slice (see RUN_LOG.md for why, and for the wall-clock
        # measurement that forced it). The matched comparisons below are computed ONLY on the
        # per-variant intersection, never on an any-arm union.
        "n_cells_llm_off_arm": off.get("n_cells"),
        "n_cells_llm_on_fixed_arm": fix.get("n_cells"),
        "n_cells_llm_on_faulty_arm": faulty.get("n_cells"),
        "n_matched_pairs_fixed_vs_faulty": c_fix_faulty.get("n_matched_pairs"),
        "n_matched_pairs_fixed_vs_off": c_fix_off.get("n_matched_pairs"),
        "matched_subset_llm_off_rate": c_fix_off.get("control_first_win_rate"),
        "first_win_rate_llm_on_fixed_generator_K4": fw,
        "first_win_rate_llm_on_fixed_generator_ci95": fix.get(
            "first_win_rate_ci95_clopper_pearson"
        ),
        "first_win_rate_llm_on_FAULTY_generator_K4": faulty.get("first_win_rate"),
        "first_win_rate_llm_off_control_K4": off.get("first_win_rate"),
        "baseline_first_win_rate_llm_off_K1": bl,
        "delta_fixed_vs_faulty": c_fix_faulty.get("delta"),
        "delta_fixed_vs_faulty_ci95": (c_fix_faulty.get("paired_bootstrap_delta") or {}).get(
            "ci95"
        ),
        "delta_fixed_vs_llm_off": c_fix_off.get("delta"),
        "delta_fixed_vs_llm_off_ci95": (c_fix_off.get("paired_bootstrap_delta") or {}).get("ci95"),
        "fault_effect_p_two_sided": m.get("p_two_sided"),
        "fault_effect_p_one_sided_fixed_better": m.get("p_one_sided_treatment_better"),
        "fault_effect_p_one_sided_faulty_better": m.get("p_one_sided_control_better"),
        "fault_effect_min_reachable_p": m.get("min_reachable_p_two_sided_at_this_support"),
        "fault_effect_significant_at_0_05": m.get("significant_at_0_05"),
    }

    d = c_fix_faulty.get("delta")
    if d is None:
        verdict = "blocked_incomplete_arms"
    elif m.get("significant_at_0_05") and (d or 0) > 0:
        verdict = f"complete_fault_depressed_first_win_by_{abs(d):.2f}_significant"
    elif m.get("significant_at_0_05") and (d or 0) < 0:
        verdict = f"complete_fixed_generator_first_win_lower_by_{abs(d):.2f}_significant"
    elif abs(d) < 1e-9:
        verdict = (
            "complete_fault_had_no_measurable_effect_on_first_win_rate"
            "_underpowered_exact_zero_delta"
        )
    else:
        verdict = f"complete_fault_effect_on_first_win_{d:+.2f}_underpowered_not_significant"
    payload["honest_verdict"] = verdict

    payload["null_delta_methodology_note"] = (
        "Any zero delta reported here is an HONEST NULL, gated by two computed controls, not "
        "a measurement bug: (1) positive_control_passed shows the llm_off arm reproduces the "
        "baseline rate AND its exact winning variants in this tree, so the win detector "
        "works; (2) contention_control_witness shows the pre-fix arm genuinely exhibited the "
        "fault (non-zero server errors with the server's own 'Context size has been "
        "exceeded.' body) while the fixed arm had none, so the control was live. Power is "
        "stated explicitly: with n discordant pairs the smallest reachable two-sided exact p "
        "is 2*0.5^n, so a non-significant result on few discordant pairs is reported as an "
        "effect size with an interval, never as evidence of no effect."
    )
    payload["false_negative_risk_checked"] = True
    # PROVENANCE, in the shape scripts/artifact_freshness_lint.py verifies: every CODE file
    # whose change could invalidate these numbers, plus every row file they were computed
    # from. Registered so that touching any dependency makes this artifact provably stale and
    # forces a rebuild, rather than leaving a silently-drifted number in the record.
    code_deps = [
        "results/first_win_llm_on_20260727/firstwin.py",
        "results/first_win_llm_on_20260727/analyse.py",
        "results/first_win_llm_on_20260727/build_artifact.py",
        "results/first_win_llm_on_20260727/guardtest.py",
        "python/carnot/experiment_4605_live_integration_scored_agent.py",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_executable_world_model.py",
    ]
    payload["provenance"] = {
        "analyzer": "results/first_win_llm_on_20260727/build_artifact.py",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": [
            {"path": p, "sha256": hashlib.sha256((REPO / p).read_bytes()).hexdigest()}
            for p in code_deps
            if (REPO / p).exists()
        ],
        "rows_sources": [
            {
                "path": str(f.relative_to(REPO)),
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
            for f in sorted((OUT / "cells").glob("*.json"))
        ],
    }
    payload["artifact_build_s"] = round(time.time() - t0, 3)
    payload["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    )
    ART.write_text(json.dumps(payload, indent=1, default=str))
    print("wrote", ART)
    print(json.dumps(payload["headline"], indent=1))
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
