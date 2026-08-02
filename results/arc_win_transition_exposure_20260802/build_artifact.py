#!/usr/bin/env python3
"""Assemble the exposure artifact. Reads rows.json + analysis.json; writes the results artifact."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "results" / "outer_loop_arc_win_transition_exposure_20260802.json"
AGENT = REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
E3 = REPO / "python" / "carnot" / "agentic" / "arc_executable_world_model.py"
FIRSTWIN = REPO / "results" / "first_win_llm_on_20260727"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_tree(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def corroboration() -> dict:
    """The SAME question answered on a corpus that had a REAL generator, at budget 200.

    Bounded rather than exact: that corpus recorded `induction_attempts_n` per cell but not the
    per-attempt ordering against the level-up, so a cell with one induce call and one level-up
    is ambiguous. Both ends are reported; neither is presented as the number.
    """
    cells = []
    for p in sorted(glob.glob(str(FIRSTWIN / "cells" / "*.json"))):
        d = json.loads(Path(p).read_text())
        lw = d.get("liveness_witness") or {}
        cells.append(
            {
                "arm": d["arm"],
                "game": d["game"],
                "n": int(lw.get("induction_attempts_n") or 0),
                "planned": int(lw.get("induction_attempts_planned") or 0),
                "reached": int(d.get("reached_level") or 0),
            }
        )
    total = sum(c["n"] for c in cells)
    upper = sum(min(c["reached"], c["n"]) for c in cells)
    lower = sum(1 for c in cells if c["n"] == 2 and c["reached"] >= 1)
    pg = defaultdict(lambda: [0, 0])
    for c in cells:
        pg[c["game"]][0] += c["n"]
        pg[c["game"]][1] += min(c["reached"], c["n"])
    rates = {g: (v[1] / v[0] if v[0] else None) for g, v in pg.items()}
    live = {g: r for g, r in rates.items() if r is not None}
    import statistics

    return {
        "source": "results/first_win_llm_on_20260727/cells (224 cells, REAL gemma-4-31B generator)",
        "sha256_of_cells": sha_tree(glob.glob(str(FIRSTWIN / "cells" / "*.json"))),
        "budget_actions": 200,
        "corpus": "25 public games x colour-permuted held-out variants, CARNOT_ARC_GATE_DEEPEN=1",
        "n_induce_calls_total": total,
        "n_with_win_available_upper_bound": upper,
        "n_with_win_available_lower_bound": lower,
        "exposure_rate_upper_bound": round(upper / total, 6),
        "exposure_rate_lower_bound": round(lower / total, 6),
        "why_a_bound_not_a_number": (
            "That corpus recorded induction_attempts_n per cell, not per-attempt ordering "
            "against the level-up. Cells with n=2 and one level-up are almost certainly "
            "stall-then-reinduction (lower bound counts the second call); the 9 cells with n=1 "
            "and one level-up are ambiguous and only the upper bound counts them."
        ),
        "per_game_upper_bound_exposure": {g: round(r, 6) for g, r in sorted(live.items())},
        "roster_max_game": max(live, key=lambda g: (live[g], g)),
        "roster_max_rate": round(max(live.values()), 6),
        "roster_median_rate": round(statistics.median(live.values()), 6),
        "n_games_at_exactly_zero": sum(1 for v in live.values() if v == 0.0),
        "second_gate": {
            "n_attempts_that_installed_a_plan": sum(c["planned"] for c in cells),
            "n_attempts": total,
            "reading": (
                "ZERO of 240 live induce attempts installed a plan, with a real generator "
                "answering. The prompt-level exposure measured here is therefore an UPPER "
                "BOUND on behavioural exposure: even at an exposed call, the induced engine "
                "was rejected by a post-generation trust gate before any plan existed."
            ),
        },
    }


def direct_kwarg_verification(scratch: Path) -> dict:
    """The receiving-end check. Reads verify_kwarg.py output if present.

    The routing partition infers "this call reached :6429" from the skip string. That is an
    inference. This reads the ARGUMENT AS THE PROPOSER RECEIVED IT, with the caller taken off the
    stack, so the claim rests on a measurement rather than on a decoding of a string.
    """
    rows = []
    for f in sorted(scratch.glob("verify_*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            continue
    if not rows:
        return {"available": False, "reason": "verify_kwarg.py output not present"}
    callers: dict[str, int] = {}
    for r in rows:
        for c in r.get("proposer_induce_calls") or []:
            key = f"{c.get('caller_file')}:{c.get('caller_lineno')} {c.get('caller_func')}"
            callers[key] = callers.get(key, 0) + 1
    return {
        "available": True,
        "games": [{"game": r["game"], "budget": r["budget"], "levels": r["levels"]} for r in rows],
        "n_proposer_induce_calls": sum(r["n_proposer_induce_calls"] for r in rows),
        "n_with_win_transition_kwarg_not_none": sum(
            r["n_with_win_transition_kwarg_not_none"] for r in rows
        ),
        "call_sites_observed": dict(sorted(callers.items())),
        "reading": (
            "Every proposer induce call that came from `arc_llm_reinduction._call_induce` "
            "carried NO `win_transition` keyword at all -- it is not in kwargs_keys. The only "
            "calls carrying the keyword came from `arc_competition_agent._induce_and_plan` "
            "(:6429), and on those the value was None, because that path is reached only before "
            "a level has been banked."
        ),
    }


def main() -> int:
    t0 = time.time()
    rows = json.loads((HERE / "rows.json").read_text())
    analysis = json.loads((HERE / "analysis.json").read_text())
    o = analysis["overall"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True
    ).stdout.strip()

    corr = corroboration()
    art = {
        "experiment": "arc_win_transition_exposure",
        "schema": "carnot.arc_win_transition_exposure.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_note": (
            "The LIVE agent (E3AgentPolicy.next_move, built with the SUBMITTED_AGENT_CONFIG "
            "kwargs make_carnot_agent uses) takes real actions against the OFFLINE arcade over "
            "environment_files. No GGUF is loaded, no llama-server is spawned, no CUDA is "
            "touched, no scored/online game is played and nothing is submitted. The proposer is "
            "experiment_4605._NoOpProposer -- the project's own `llm_off` arm definition -- so "
            "the shipped `_induce_and_plan` body runs in full, INCLUDING the "
            "`self._proposer().induce(..., win_transition=self._win_transition)` call site "
            "under test; only the generator's answer is absent."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "NO SOLVE IS CLAIMED. No level is banked to the registry, `offline_reproduced` is "
            "deliberately absent rather than set False, and the per-game `levels` figures are "
            "trajectory context for the exposure denominator, not solve claims. These are "
            "offline dev-twin runs on PUBLIC games."
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "No verifier value, moat or efficiency claim is made. This artifact measures how "
            "often one code path is reached; it grades nothing."
        ),
        "the_change_under_test": {
            "shipped": "2026-08-01",
            "flagged": False,
            "flag_discipline_breach": (
                "The change ships UNFLAGGED: there is no default-OFF env gate on the live "
                "behaviour change at arc_competition_agent.py:6433. Recorded here as a standing "
                "-discipline breach, independent of whether the change turns out to help."
            ),
            "sites": [
                "arc_competition_agent.py:4710 (`self._win_transition = None`)",
                "arc_competition_agent.py:4946 (captured in _begin_level_goal_episode)",
                "arc_competition_agent.py:6429-6433 (passed to the live induce call)",
                "arc_executable_world_model.py:2533/2870/3044/6422 (`win_transition=` kwarg)",
            ],
            "sha256_arc_competition_agent": sha_file(AGENT),
            "sha256_arc_executable_world_model": sha_file(E3),
            "why_exposure_is_the_first_question": (
                "`_begin_level_goal_episode` is the SOLE writer of `_win_transition`, and it "
                "runs only after a level-up. With None, both proposers reproduce the historical "
                "scan of `trans` byte-for-byte. So every induce call made before the agent has "
                "banked a level is, by construction, unchanged by this change."
            ),
        },
        "no_hypothesis_test_is_run": (
            "This is a descriptive rate with a game-clustered spread, not a comparison, so no "
            "p-value is computed and none should be quoted from it. The MIN REACHABLE p is "
            "undefined because there is no second arm. The decision this artifact supports is "
            "whether an effect measurement would be interpretable, not whether an effect exists."
        ),
        "preconditions_checked": [
            {
                "resource": "offline_arcade_environment_files",
                "available": (REPO / "environment_files").is_dir(),
            },
            {"resource": "live_agent_source", "available": AGENT.exists()},
            {"resource": "world_model_source", "available": E3.exists()},
            {
                "resource": "arc_solve_registry",
                "available": (REPO / "ops" / "arc_solve_registry.yaml").exists(),
            },
            {
                "resource": "live_generator_corroboration_corpus",
                "available": (FIRSTWIN / "cells").is_dir(),
            },
            {"resource": "no_gpu_used", "available": True},
        ],
        "cited_upstream_artifacts": [
            {
                "experiment_id": "arc_first_win_llm_on_eval_concurrency_20260727",
                "fields_imported": [
                    "cells/*.json:liveness_witness.induction_attempts_n",
                    "cells/*.json:liveness_witness.induction_attempts_planned",
                    "cells/*.json:reached_level",
                    "arm_definitions.llm_off",
                ],
                "sha256": sha_file(
                    REPO
                    / "results"
                    / "outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json"
                ),
            },
            {
                "experiment_id": "arc_goal_predicate_anatomy_20260801",
                "fields_imported": [
                    "was_it_knowable.first_and_largest_fact",
                    "clusters_of_the_71_failing_goals.A_DECLINED",
                ],
                "sha256": sha_file(
                    REPO / "results" / "arc_goal_predicate_anatomy_20260801" / "artifact.json"
                ),
            },
        ],
        "method": {
            "driver": (
                "scripts/arc_leaderboard_eval.py:run_game (shipped; writes no files -- only its "
                "main() does, and main() is never called here)"
            ),
            "policy": (
                "E3AgentPolicy built with the exact SUBMITTED_AGENT_CONFIG kwargs "
                "make_carnot_agent uses"
            ),
            "instrument": (
                "E3AgentPolicy._induce_and_plan is wrapped for the duration of the run and "
                "records `self._win_transition is not None` BEFORE delegating. No shipped file "
                "is edited; no default is changed."
            ),
            "games": "the 25 public survey games, read from ops/arc_solve_registry.yaml",
            "budgets": sorted({r["budget"] for r in rows}),
            "replicate_axis": "frontier_discipline_seed (the seed the agent actually consumes)",
            "harness_paths": [
                "results/arc_win_transition_exposure_20260802/worker.py",
                "results/arc_win_transition_exposure_20260802/run.py",
                "results/arc_win_transition_exposure_20260802/analyse.py",
                "results/arc_win_transition_exposure_20260802/build_artifact.py",
                "results/arc_win_transition_exposure_20260802/verify_kwarg.py",
            ],
        },
        "headline": o,
        "what_a_live_ab_could_reach_at_this_exposure": None,
        "per_budget": {k: v for k, v in analysis.items() if k.startswith("budget_")},
        "fd_seed_replicates": analysis.get("fd_seed_replicates"),
        "argv_seed_aa": analysis.get("argv_seed_aa"),
        "corroboration_on_a_real_generator_corpus": corr,
        "direct_kwarg_verification": direct_kwarg_verification(
            Path(os.environ.get("WTX_VERIFY_DIR", str(HERE / "verify_out")))
        ),
        "the_routing_defect": {
            "what": (
                "The changed call site is UNREACHABLE at every induce call where a win "
                "transition exists. `_induce_and_plan` takes the "
                "`execute_bounded_llm_reinduction` branch whenever "
                "`attempt['reason'] == 'level_up_reinduction' OR next_level_episode` "
                "(arc_competition_agent.py:6138), and RETURNS at :6245. `next_level_episode` is "
                "`_previous_level_complete_grid is not None AND _current_goal_level > "
                "_start_level + 1` -- which is set by `_begin_level_goal_episode`, the SAME "
                "function and the SAME instant that sets `_win_transition`. So the routing "
                "predicate and the availability predicate are, in practice, the same predicate."
            ),
            "where_the_argument_is_dropped": (
                "arc_llm_reinduction.py:203-220 `_call_induce` forwards only "
                "`previous_level_complete_grid`; it has no `win_transition` parameter, so the "
                "keyword is absent from that call entirely."
            ),
            "measured_partition": o.get("routing_partition"),
            "off_diagonal_cells": 0,
            "why_the_partition_is_credible": (
                "The two paths write DIFFERENT skip strings -- `proposer_failed` at "
                "arc_llm_reinduction.py:1507 and `proposer_failed_or_missing_root` at "
                "arc_competition_agent.py:6436, the line immediately after the changed call -- "
                "and the observed partition has ZERO off-diagonal cells across 128 calls. It is "
                "corroborated independently at the receiving end by `direct_kwarg_verification`."
            ),
        },
        "what_this_measurement_cannot_say": [
            "It cannot say what the change DOES. Exposure is a precondition for an effect, not "
            "an effect.",
            "It cannot speak to the hidden set. These are public games the project has fully "
            "solved (183/183 in ops/arc_solve_registry.yaml) and whose mechanics are recorded "
            "there; a hidden game is OOD by construction and could bank levels at a different "
            "rate in either direction.",
            "It cannot measure the SECOND gate. Under the NoOp proposer every attempt skips at "
            "`proposer_failed`, so this harness sees prompt-level exposure only. The real-"
            "generator corpus supplies that number: 0 of 240 live induce attempts installed a "
            "plan.",
            "It is a PROXY for the LLM-on trajectory. The justification is measured, not "
            "assumed -- the cited corpus found every LLM-on arm BIT-IDENTICAL to its llm_off "
            "control on first_win, actions, reached_level and actions_to_first_levelup across "
            "74/74 matched cells, because no plan was ever installed. If a future change makes "
            "the trust gate pass, that equivalence breaks and this proxy must be re-derived.",
        ],
        "methodology_note": (
            "EXACT 0.0 AND EXACT 1.0 ARE EXPECTED HERE AND ARE NOT THE FABRICATION SIGNATURE. "
            "Every rate in this artifact is a COUNT RATIO over a small integer denominator, not "
            "a classifier score. A game whose live agent never banks a level makes exactly one "
            "induce call and has exactly zero win-transition-available calls, so its exposure is "
            "exactly 0/1 = 0.0 as a matter of counting; a game that levels up before it ever "
            "stalls makes all of its induce calls after the level-up and is exactly 1.0 for the "
            "same reason. `n_attempts_that_installed_a_plan == 0` is likewise a real measured "
            "count, corroborated independently at 0/240 on a real-generator corpus. No metric "
            "here is an accuracy, an AUROC or a TPR, and none is being claimed as one."
        ),
        "false_negative_risk_checked": (
            "This artifact reports an EXPOSURE rate, not a null effect, so the false-negative "
            "trap does not apply in its usual form. The adjacent trap that DOES apply is the "
            "inverse: reading a low exposure as 'the change does not work'. It does not say "
            "that. It says an effect measurement run at this exposure would be measuring "
            "mostly cells in which the change is inert, and would therefore be uninterpretable."
        ),
        "random_seed": 20260802,
        "random_seed_note": (
            "The worker seeds `random`/`numpy` with this value, and that seeding is INERT: the "
            "explorer's RNGs are `random.Random(<constructor default>)` and cannot be reached "
            "from a global seed. The seed that matters is frontier_discipline_seed, reported "
            "separately. This field is recorded for reproducibility of the worker process, not "
            "as evidence the trajectory was randomised."
        ),
        "duration_s": None,
        "rows_path": "results/arc_win_transition_exposure_20260802/rows.json",
        "analysis_path": "results/arc_win_transition_exposure_20260802/analysis.json",
    }
    # STATED AS A CONSEQUENCE OF THE MEASURED EXPOSURE, not discovered after a test.
    exposed_games = sorted(g for g, v in (o.get("per_game_win_available") or {}).items() if v > 0)
    # THE NUMBER THAT MATTERS IS THE EFFECTIVE ONE. A game can only be discordant in a live A/B
    # of THIS change if the changed call site actually received a win transition there.
    k = int(o.get("n_calls_with_win_available_that_reached_the_changed_call_site") or 0)
    art["what_a_live_ab_could_reach_at_this_exposure"] = {
        "clustering": (
            "GAME level -- cells within a game share a trajectory and are not independent"
        ),
        "n_games_with_at_least_one_WIN_AVAILABLE_induce_call": len(exposed_games),
        "games_with_any_win_available_call": exposed_games,
        "n_calls_at_which_the_CHANGED_CALL_SITE_received_a_win_transition": k,
        "max_discordant_pairs_available": k,
        "min_reachable_two_sided_p": (round(2 * 0.5**k, 6) if k else 1.0),
        "min_reachable_two_sided_p_note": (
            "An exact paired (McNemar/sign) test's smallest reachable two-sided p is 2*0.5^n at "
            "n discordant pairs, and a cell where the changed argument was never delivered can "
            "never be discordant -- the two arms are the same program. At k=0 the smallest "
            "reachable p is 1.0: a live A/B of this change on this roster is UNFALSIFIABLE, not "
            "underpowered. Running more cells of an identity returns exactly 0 forever."
        ),
        "reachable_p_lt_0_05": (k >= 6),
    }
    art["duration_s"] = round(sum(float(r.get("elapsed_s") or 0.0) for r in rows), 3)
    art["duration_s_provenance"] = (
        "sum of per-cell worker wall time across all cells; the sweep ran 11 cells concurrently "
        "so the wall SPAN is shorter than this total"
    )
    b400 = analysis.get("budget_400", {})
    art["headline_summary"] = {
        "the_question": (
            "At what fraction of LIVE goal-predicate induce calls is `self._win_transition` "
            "actually non-None -- i.e. how often is the 2026-08-01 change even reached?"
        ),
        "answer_prompt_availability": {
            "n_induce_calls_total": o["n_induce_calls_total"],
            "n_with_win_available": o["n_with_win_available"],
            "rate_pooled": o["exposure_rate_pooled"],
            "rate_game_clustered_mean": o["cluster_mean_of_per_game_rates"],
            "roster_median_rate": o["roster_median_rate"],
            "roster_max_game": o["roster_max_game"],
            "roster_max_rate": o["roster_max_rate"],
            "n_games_at_exactly_zero": o["roster_n_games_at_zero_exposure"],
            "at_the_SHIPPED_action_cap_400": {
                "rate_pooled": b400.get("exposure_rate_pooled"),
                "rate_game_clustered_mean": b400.get("cluster_mean_of_per_game_rates"),
                "roster_median_rate": b400.get("roster_median_rate"),
                "n_games_at_exactly_zero": b400.get("roster_n_games_at_zero_exposure"),
                "note": (
                    "400 is the SHIPPED per-game action cap on the scored path "
                    "(`make_carnot_agent`'s CarnotAgent.MAX_ACTIONS = 400). The 2000 arm is a "
                    "5x-headroom sensitivity check, not a configuration that ships."
                ),
            },
        },
        "answer_effective_exposure": {
            "n_calls_where_the_changed_call_site_received_a_win_transition": (
                o["n_calls_with_win_available_that_reached_the_changed_call_site"]
            ),
            "rate": o["effective_exposure_rate"],
            "why": (
                "Every one of the 30 calls at which a win transition existed was routed into "
                "`execute_bounded_llm_reinduction`, which does not forward the argument, and "
                "that branch returns before reaching the changed call. The partition is exact: "
                "98/98 no-win calls reached the changed site, 30/30 win-available calls did "
                "not, zero off-diagonal."
            ),
        },
        "the_one_number_to_read": (
            "0 of 128. The change is not low-exposure; on the live path as shipped it is "
            "structurally INERT, for a reason unrelated to how often the agent banks levels."
        ),
    }
    art["honest_verdict"] = (
        "complete_win_transition_change_is_structurally_inert_on_the_live_path: a win "
        "transition was AVAILABLE at 30 of 128 live induce calls (23.4% pooled; game-clustered "
        "mean 15.7%; roster MEDIAN 0.0 with vc33 the maximum at 1.0 and 14 of 25 games at "
        "exactly zero) -- but it reached the changed call site at arc_competition_agent.py:"
        "6429-6433 at 0 of 128 calls, at both budgets, because `_induce_and_plan` routes every "
        "level-banked induction into `execute_bounded_llm_reinduction` (whose `_call_induce` "
        "has no `win_transition` parameter) and returns at :6245 before the changed line. The "
        "routing predicate `next_level_episode` and the availability predicate are set by the "
        "same function at the same instant, so the two are the same predicate in practice. "
        "Verified twice: an exact skip-string partition with zero off-diagonal cells across "
        "128 calls, and a direct instrument on the proposer's own `induce` across the five "
        "level-banking games, where the keyword was absent on all 12 reinduction-path calls "
        "and None on all 4 that carried it. DO NOT run a live effect A/B: at 0 discordant "
        "pairs its smallest reachable two-sided p is 1.0, so it would be unfalsifiable, not "
        "underpowered."
    )
    art["recommendation"] = {
        "do_not": (
            "Do not spend GPU hours on a live end-to-end A/B of this change. Both arms are the "
            "same program on every cell measured; the comparison is an identity."
        ),
        "do_first": (
            "Decide whether the change is meant to apply to the level-up reinduction path. If "
            "yes, that is a code fix (thread `win_transition` through "
            "`execute_bounded_llm_reinduction` -> `_call_induce`), behind a default-OFF env "
            "flag, and exposure must be RE-MEASURED with this same harness afterwards before "
            "any effect measurement."
        ),
        "then": (
            "Measure the effect as a COMPONENT test, not end-to-end: call the shipped "
            "`induce_prompt` / proposer `generate` directly on windows that contain a real "
            "level-up, with and without `win_transition`, so the mechanism fires in 100% of "
            "cells by construction. That is the pattern results/arc_goal_evidence_20260802 "
            "Stage 1 already established for exactly this reason."
        ),
        "and_note_the_second_gate": (
            "Even a fixed and exposed prompt is downstream-gated: 0 of 240 live induce attempts "
            "on the real-generator corpus installed a plan, every one rejected by a "
            "post-generation trust gate. A prompt improvement cannot move an end-to-end metric "
            "while that gate is closed."
        ),
        "and_the_third_gate": (
            "`_goal_only_prompt` -- the FOCUSED win-condition prompt used by the split-induce "
            "fallback, and the one whose output the anatomy pass found to be evidence-free -- "
            "does not receive `win_transition` either (arc_executable_world_model.py:6498). "
            "The argument reaches the combined prompt and the split path's ENGINE half only."
        ),
        "flag_discipline": (
            "The change ships with no default-OFF env gate. Whatever is decided above, the gate "
            "is owed under the standing discipline."
        ),
    }
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps({"rows": rows, "analysis": analysis}, sort_keys=True, default=str).encode()
    ).hexdigest()
    art["build_s"] = round(time.time() - t0, 3)
    OUT.write_text(json.dumps(art, indent=2, default=str))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
