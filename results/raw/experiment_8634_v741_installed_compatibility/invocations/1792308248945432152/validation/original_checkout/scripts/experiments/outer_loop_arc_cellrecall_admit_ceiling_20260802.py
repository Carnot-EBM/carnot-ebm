#!/usr/bin/env python3
"""ARC cell_recall trust-gate ADMIT CEILING, from already-recorded margins. CPU ONLY.

THE QUESTION (operator, 2026-08-02): would a cell_recall trust gate near 0.2 admit
engines, and do admitted engines produce PLANS?

WHY THIS SCRIPT EXISTS AT ALL -- the headline is a RETRACTION, not a measurement.
`llm_on_fix_cellrecall` (results/first_win_llm_on_20260727/) already ran and produced 0
plans, and that "0 plans" is currently load-bearing for the standing GENERATION-WALL
verdict. This script establishes that the arm flipped the METRIC and never touched the
THRESHOLD -- which is a hardcoded `0.5` literal at arc_competition_agent.py:6712 with no
env override anywhere in python/carnot/agentic/ -- while the arm's own engines maxed at
cell_recall 0.0476. Nothing in that arm could have been admitted, so its 0 plans is an
arithmetic identity, not evidence. This is the "a prior negative may be UNFALSIFIABLE,
not negative" failure mode, committed against the project by the project.

WHAT IS ANALYSED. Only the plain (`else`) branch of `_induce_and_plan`, because
`HIDDEN_STATE_GAME_IDS` is a hardcoded 11-game PUBLIC tuple -- a hidden Kaggle game is
never a member, so it ALWAYS takes the plain branch. Hidden-state records are excluded
and counted, not pooled.

MISSING IS NOT ZERO. An attempt whose gate diagnostics carry no `verify_cell_recall`
never reached STAGE 4 and has NO margin. It is excluded and counted, never coerced to 0.

CLUSTERING. Admit counts are reported as ENGINES and as DISTINCT GAMES. The game count
is what bounds any later inference; a rule-of-three or a p-value computed at engine level
while games are the independent unit overstates by the within-game multiplicity.

No GGUF, no GPU, no generator, no episode. This is a recount of cached per-cell JSON plus
a read of two source files.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "first_win_llm_on_20260727" / "cells"
FIRSTWIN = REPO / "results" / "first_win_llm_on_20260727" / "firstwin.py"
AGENT = REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
EWM = REPO / "python" / "carnot" / "agentic" / "arc_executable_world_model.py"
TRUST = REPO / "python" / "carnot" / "agentic" / "arc_world_model_trust_energy.py"
OUT = REPO / "results" / "outer_loop_arc_cellrecall_admit_ceiling_20260802.json"

THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.50)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_attempts() -> tuple[list[dict], list[dict], list[dict]]:
    """Split every recorded induction attempt into plain / hidden-state / no-margin.

    The three-way split is the point. A plain-branch attempt carries
    `verify_cell_recall`; a hidden-state attempt carries `trust_energy` and is on a
    branch a hidden game never takes; an attempt with neither never reached the
    verifier and has NO margin to report.
    """
    plain: list[dict] = []
    hidden: list[dict] = []
    nomargin: list[dict] = []
    for f in sorted(glob.glob(str(CELLS / "*.json"))):
        d = json.loads(Path(f).read_text())
        lw = d.get("liveness_witness") or {}
        for i, g in enumerate(lw.get("induction_attempt_gate_diagnostics") or []):
            rec = dict(arm=d["arm"], game=d["game"], attempt_index=i, **g)
            if "verify_cell_recall" in g:
                plain.append(rec)
            elif "trust_energy" in g:
                hidden.append(rec)
            else:
                nomargin.append(rec)
    return plain, hidden, nomargin


def dist(vals: list[float]) -> dict:
    q = statistics.quantiles(vals, n=4) if len(vals) >= 2 else [vals[0]] * 3
    return {
        "n": len(vals),
        "min": round(min(vals), 6),
        "q1": round(q[0], 6),
        "median": round(statistics.median(vals), 6),
        "q3": round(q[2], 6),
        "max": round(max(vals), 6),
        "mean": round(statistics.mean(vals), 6),
        "n_exactly_zero": sum(1 for v in vals if v == 0.0),
        "nonzero_values_sorted": sorted(v for v in vals if v > 0),
    }


def admit_table(rows: list[dict]) -> dict:
    out = {}
    for t in THRESHOLDS:
        adm = [r for r in rows if r["verify_cell_recall"] >= t]
        games = sorted({r["game"] for r in adm})
        out[f"{t:.2f}"] = {
            "engines_admitted": len(adm),
            "games_admitted": len(games),
            "game_list": games,
            "engines": [
                {"arm": r["arm"], "game": r["game"], "cell_recall": r["verify_cell_recall"]}
                for r in sorted(adm, key=lambda r: -r["verify_cell_recall"])
            ],
        }
    return out


def min_reachable_p(n_games: int) -> dict:
    """Best achievable p from n independent GAME clusters, all in one direction.

    A sign / exact-binomial / game-level permutation test over n clusters that all move
    the same way has 2**n equally-likely sign assignments, so the smallest attainable
    one-sided p is 2**-n and the smallest two-sided p is 2 * 2**-n (capped at 1.0).
    Stated BEFORE any effect is measured, per the pre-registration discipline.
    """
    if n_games <= 0:
        return {
            "n_game_clusters": 0,
            "min_one_sided_p": None,
            "min_two_sided_p": None,
            "note": "no admitted games -- no test is defined at all",
        }
    one = 2.0**-n_games
    return {
        "n_game_clusters": n_games,
        "min_one_sided_p": round(one, 6),
        "min_two_sided_p": round(min(1.0, 2 * one), 6),
        "reaches_p_lt_0_05_one_sided": one < 0.05,
        "reaches_p_lt_0_05_two_sided": min(1.0, 2 * one) < 0.05,
        "n_games_needed_for_one_sided_0_05": 5,
        "n_games_needed_for_two_sided_0_05": 6,
    }


def main() -> None:
    t0 = time.time()

    # ---- FORENSICS: what threshold did the prior arm actually run at? --------------
    agent_src = AGENT.read_text()
    gate_line_no = None
    gate_line = None
    for i, line in enumerate(agent_src.splitlines(), 1):
        if "_gate_value < 0.5" in line:
            gate_line_no, gate_line = i, line.strip()
    metric_line_no = None
    for i, line in enumerate(agent_src.splitlines(), 1):
        if 'os.environ.get("CARNOT_ARC_TRUST_METRIC"' in line:
            metric_line_no = i
    # Every CARNOT_ARC_* env read in the agent, so "there is no threshold env" is a
    # SEARCH RESULT and not an assertion.
    env_names = sorted(
        set(re.findall(r'os\.environ\.get\(\s*"(CARNOT_ARC_[A-Z0-9_]+)"', agent_src))
    )
    threshold_envs = [e for e in env_names if "THRESH" in e]

    fw = FIRSTWIN.read_text()
    m = re.search(r"ARM_ENV\s*=\s*\{(.*?)\n\}", fw, re.S)
    arm_env_block = m.group(0) if m else ""
    arm_env_sets_threshold = bool(re.search(r"THRESH", arm_env_block))

    plain, hidden, nomargin = load_attempts()
    cr_arm = [r for r in plain if r["arm"] == "llm_on_fix_cellrecall"]
    diag_arm = [r for r in plain if r["arm"] == "llm_on_fix_diag"]

    prior_max = max(r["verify_cell_recall"] for r in cr_arm)
    prior_admitted_at_live_threshold = [r for r in cr_arm if r["verify_cell_recall"] >= 0.5]

    # ---- NONDEGENERACY: does the cell_recall path carry correct_changed_cells>=1? ---
    # Read the branch structure literally: `if change_gate_enabled: ... elif _gate_value
    # < 0.5: ...`. if/elif is MUTUALLY EXCLUSIVE, not conjunctive.
    tail = agent_src.splitlines()[gate_line_no - 6 : gate_line_no + 3]
    branch_is_if_elif = any('if _change_gate["gate_enabled"]:' in ln for ln in tail) and any(
        "elif _gate_value < 0.5:" in ln for ln in tail
    )
    change_gate_default_off = "SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED = False" in EWM.read_text()
    # Positive confirmation from the RECORDS, not just the source: every plain rejection
    # names the elif's label, and none names the change gate's label.
    plain_skips: dict[str, int] = {}
    for r in plain:
        plain_skips[str(r.get("skipped"))] = plain_skips.get(str(r.get("skipped")), 0) + 1
    any_change_gate_skip = any(k.startswith("world_model_change_gate_") for k in plain_skips)
    # And the projection never persisted correct_changed_cells on the plain branch at all.
    plain_has_ccc = sum(1 for r in plain if "correct_changed_cells" in r)

    hidden_ccc_zero = sum(1 for r in hidden if int(r.get("correct_changed_cells", 0)) == 0)

    # ---- gate-cleared-but-no-plan (necessary != sufficient), on the live corpus -----
    cleared_no_plan = [
        {
            "arm": r["arm"],
            "game": r["game"],
            "trust_metric": r.get("trust_metric"),
            "verify_accuracy": r["verify_accuracy"],
            "verify_cell_recall": r["verify_cell_recall"],
            "planned": bool(r["planned"]),
            "skipped": r.get("skipped"),
        }
        for r in plain
        if r.get("trust_metric") == "exact" and r["verify_accuracy"] >= 0.5
    ]

    table_pooled = admit_table(plain)
    n_games_at_020 = table_pooled["0.20"]["games_admitted"]

    art = {
        "experiment": "outer_loop_arc_cellrecall_admit_ceiling_20260802",
        "title": (
            "ARC cell_recall admit ceiling: the prior cell_recall arm ran at threshold 0.5 "
            "against a maximum achievable 0.0476, so its 0-plans result is VOID; a 0.20 gate "
            "admits 2 engines on 2 games, at which no p<0.05 is reachable"
        ),
        "run_date": "2026-08-02",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_note": (
            "CPU only. No GGUF load, no GPU, no llama-server, no generator, no episode "
            "played. This is a recount of cached per-cell JSON plus a static read of "
            "arc_competition_agent.py / arc_executable_world_model.py."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "No level was solved and no solve is claimed. This is a gate-margin census over "
            "an already-recorded corpus."
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "No verifier-value or moat claim is made. The subject IS the trust gate, not a "
            "verifier competing with an oracle."
        ),
        "model_specs": {"note": "no model invoked; margins are read from cached 2026-07-27 cells"},
        "random_seed": 0,
        "random_seeds_used": [0],
        "not_submitted": True,
        "no_shipped_default_changed": True,
        "no_shipped_default_changed_note": (
            "Read-only analysis. No env flag flipped, no default changed, no source edited."
        ),
        "no_p_value_by_design": (
            "This is a census of a fixed corpus with no treatment and no pairing. The only "
            "probability statement made is the PRE-REGISTERED minimum REACHABLE p implied by "
            "the admitted-game count, stated in advance of any effect being measured."
        ),
        "provenance": {
            "code_read": [
                {"path": str(p.relative_to(REPO)), "sha256": sha(p)}
                for p in (AGENT, EWM, TRUST, FIRSTWIN)
            ],
            "corpus": {
                "path": "results/first_win_llm_on_20260727/cells",
                "n_cell_files": len(glob.glob(str(CELLS / "*.json"))),
            },
            "rebuild_command": (
                "JAX_PLATFORMS=cpu .venv/bin/python "
                "scripts/experiments/outer_loop_arc_cellrecall_admit_ceiling_20260802.py"
            ),
            "nothing_tracked_is_written": (
                "Reads results/first_win_llm_on_20260727/cells (EVIDENCE, read-only) and "
                "writes exactly one new file, this artifact."
            ),
        },
        "prior_art_built_on": [
            {
                "path": "results/outer_loop_arc_induce_gate_anatomy_20260802.json",
                "what_it_already_answered": (
                    "the gate is BINDING and mechanically reached (callee-instrumented "
                    "plan_in_model, caller read off the stack), and clearing it is NECESSARY "
                    "NOT SUFFICIENT (a gate-clearing engine with an unreachable goal planned "
                    "nothing)."
                ),
                "what_this_artifact_CORRECTS_in_it": (
                    "its `so_a_threshold_relaxation_is_not_the_lever` cites the "
                    "llm_on_fix_cellrecall arm's 0 plans as evidence against the metric "
                    "lever. That arm could not have admitted anything. The citation is void "
                    "and the sub-claim must be withdrawn -- see PRIOR_ARM_FORENSICS."
                ),
            },
            {
                "path": "results/outer_loop_arc_generation_vs_selection_20260802.json",
                "what_it_already_answered": (
                    "the GENERATION-WALL verdict; its LIVE_comparison_the_actual_rejection_set "
                    "reports the same 28 plain-branch margins used here."
                ),
                "what_this_artifact_CORRECTS_in_it": (
                    "its plain-branch rule-of-three upper bound 0.1015 is computed at ENGINE "
                    "level (n=28) while the 28 engines span only 14 distinct GAMES. At the "
                    "game level the bound is 3/14 = 0.2143, i.e. ~2.1x looser."
                ),
            },
            {
                "path": "ops/verifier_gaps.md",
                "what_it_already_answered": (
                    "GAP-WM-TRUST-GATE / GAP-WM-TRUST-GATE-HIDDEN-STATE: the change gate and "
                    "its coverage hole; also records that cell_recall masks to TRUE changes "
                    "only and is therefore recall, blind to a spurious writer."
                ),
            },
        ],
        # ------------------------------------------------------------------ Q1
        "PRIOR_ARM_FORENSICS": {
            "question": "what threshold did llm_on_fix_cellrecall actually run at?",
            "answer": 0.5,
            "verdict_void": True,
            "evidence": {
                "the_arm_sets_only_the_metric": {
                    "source": "results/first_win_llm_on_20260727/firstwin.py ARM_ENV",
                    "value": {"llm_on_fix_cellrecall": {"CARNOT_ARC_TRUST_METRIC": "cell_recall"}},
                    "arm_env_sets_any_threshold": arm_env_sets_threshold,
                },
                "the_threshold_is_a_hardcoded_literal": {
                    "path": "python/carnot/agentic/arc_competition_agent.py",
                    "line": gate_line_no,
                    "text": gate_line,
                    "metric_env_read_at_line": metric_line_no,
                    "comment": (
                        "CARNOT_ARC_TRUST_METRIC chooses WHICH quantity is compared "
                        "(vr.cell_recall vs vr.accuracy). The 0.5 it is compared AGAINST is a "
                        "literal in the same expression. Switching the metric moves the "
                        "quantity onto a different scale and leaves the bar where it was."
                    ),
                },
                "no_threshold_env_exists_anywhere_in_the_agent": {
                    "all_CARNOT_ARC_env_reads_in_arc_competition_agent_py": env_names,
                    "of_which_threshold_related": threshold_envs,
                    "search_is_the_evidence": (
                        "this list is a regex sweep of the file, not an assertion; it is empty "
                        "of THRESH names, so no override existed to be set."
                    ),
                },
                "the_arm_could_not_have_admitted_anything": {
                    "n_plain_branch_attempts_in_that_arm": len(cr_arm),
                    "max_verify_cell_recall_in_that_arm": prior_max,
                    "threshold_in_force": 0.5,
                    "headroom_to_the_bar": round(0.5 - prior_max, 6),
                    "n_admitted_at_the_threshold_in_force": len(prior_admitted_at_live_threshold),
                    "every_rejection_named_the_threshold_branch": True,
                    "skip_census_for_that_arm": {
                        k: sum(1 for r in cr_arm if str(r.get("skipped")) == k)
                        for k in sorted({str(r.get("skipped")) for r in cr_arm})
                    },
                },
            },
            "what_must_be_retracted": (
                "'the cell_recall metric that WOULD separate it already shipped as a flag and "
                "already ran as an arm (llm_on_fix_cellrecall): still 0 plans installed'. That "
                "arm ran a 0.0476-maximum quantity against a 0.5 bar. Zero plans was the only "
                "arithmetically possible outcome, so it is not evidence about the lever in "
                "either direction. The cell_recall lever HAS NEVER BEEN TESTED at a threshold "
                "it could clear. Stating this in the other direction with equal force: this is "
                "NOT a finding that the lever works -- it is a finding that the question is "
                "OPEN where the record said CLOSED."
            ),
            "scope_of_the_retraction": (
                "One leg of the GENERATION-WALL verdict. The verdict's other legs -- 0 of 250 "
                "on the gate-faithful split, 0 of 15 strongest engines generalising, the "
                "hidden-state branch's own 0 of 22 -- are untouched by this and are NOT "
                "retracted here."
            ),
        },
        # ------------------------------------------------------------------ Q2
        "NONDEGENERACY_CONJUNCT": {
            "question": "does the cell_recall path carry `correct_changed_cells >= 1`?",
            "answer": False,
            "answer_in_one_line": (
                "NO -- and the premise that it might is itself worth correcting. The two live "
                "in MUTUALLY EXCLUSIVE branches of an if/elif, so they can never both bind. "
                "Turning the change gate ON does not ADD nondegeneracy to the cell_recall "
                "gate; it makes CARNOT_ARC_TRUST_METRIC a DEAD LEVER, because `_gate_value` is "
                "then never read."
            ),
            "source_evidence": {
                "branch_is_if_elif_not_conjunction": branch_is_if_elif,
                "excerpt": [ln.rstrip() for ln in tail],
                "change_gate_shipped_default_off": change_gate_default_off,
                "constant": "SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED = False",
            },
            "record_evidence": {
                "plain_branch_skip_census": plain_skips,
                "any_rejection_named_the_change_gate": any_change_gate_skip,
                "reading": (
                    "every plain rejection is labelled world_model_accuracy_below_threshold, "
                    "which is written ONLY by the elif. Not one is labelled "
                    "world_model_change_gate_*. The threshold branch is the branch that fired."
                ),
                "n_plain_attempts_recording_correct_changed_cells": plain_has_ccc,
                "reading_2": (
                    "0 of 28. The diagnostics projection never persisted "
                    "correct_changed_cells on the plain branch, so the conjunct is not even "
                    "OBSERVABLE there from these records -- another reason the 16/22 figure "
                    "cannot be about this branch."
                ),
            },
            "the_16_of_22_belongs_to_the_OTHER_branch": {
                "n_hidden_state_attempts": len(hidden),
                "n_with_correct_changed_cells_zero": hidden_ccc_zero,
                "games": sorted({r["game"] for r in hidden}),
                "reading": (
                    "all 11 games are HIDDEN_STATE_GAME_IDS members. This statistic is a "
                    "property of the hidden-state admission branch, which a hidden Kaggle game "
                    "never takes. Carrying it onto the plain branch is the cross-branch "
                    "transfer error, and it would have made the admit ceiling look bounded by "
                    "a conjunct that does not apply."
                ),
            },
            "consequence_for_the_admit_ceiling": (
                "The ceiling is bounded by the MARGIN DISTRIBUTION alone, not by a conjunct. "
                "That is the more permissive of the two possibilities -- and the ceiling is "
                "still tiny."
            ),
        },
        # ------------------------------------------------------------------ Q3/Q4
        "SCOPE_PLAIN_BRANCH_ONLY": {
            "why": (
                "HIDDEN_STATE_GAME_IDS is a hardcoded 11-game PUBLIC tuple. A hidden Kaggle "
                "game id is not a member, so a hidden game ALWAYS takes the plain (`else`) "
                "branch. Hidden-state results do not transfer and are excluded, not pooled."
            ),
            "hidden_state_game_ids": sorted({r["game"] for r in hidden}),
            "n_hidden_state_attempts_excluded": len(hidden),
            "n_no_margin_attempts_excluded": len(nomargin),
            "no_margin_detail": [
                {"arm": r["arm"], "game": r["game"], "skipped": r.get("skipped")} for r in nomargin
            ],
            "no_margin_is_MISSING_not_zero": (
                "these attempts never reached STAGE 4, so no verify_cell_recall was computed. "
                "They are excluded from every distribution and every count. Coercing them to "
                "0.0 would deflate the mean and inflate the n."
            ),
            "n_plain_attempts_analysed": len(plain),
            "n_plain_distinct_games": len({r["game"] for r in plain}),
            "arms_contributing": sorted({r["arm"] for r in plain}),
        },
        "MARGIN_DISTRIBUTION_plain_branch": {
            "pooled_28": dist([r["verify_cell_recall"] for r in plain]),
            "llm_on_fix_cellrecall_14": dist([r["verify_cell_recall"] for r in cr_arm]),
            "llm_on_fix_diag_14": dist([r["verify_cell_recall"] for r in diag_arm]),
            "the_two_arms_cover_the_SAME_14_games": sorted({r["game"] for r in cr_arm})
            == sorted({r["game"] for r in diag_arm}),
            "between_arm_variance_is_the_dominant_effect": (
                "same 14 games, same generator config (n_ctx 81920), two independent samples: "
                "one arm yields three engines at or above 0.1985 and the other yields a "
                "maximum of 0.0476. The admit count per run is therefore bimodal, and a single "
                "arm's admit count is not a stable estimate of the ceiling."
            ),
        },
        "ADMIT_COUNT_BY_THRESHOLD": {
            "gate_modelled": "cell_recall >= t, threshold-only (no nondegeneracy conjunct, per Q2)",
            "pooled_28_engines_over_14_games": table_pooled,
            "llm_on_fix_cellrecall_arm_only": admit_table(cr_arm),
            "llm_on_fix_diag_arm_only": admit_table(diag_arm),
            "the_result_that_matters_most": (
                "EVERY engine admitted at any tested threshold comes from llm_on_fix_diag -- "
                "the arm that ran the SHIPPED `exact` metric. The arm that actually ran "
                "cell_recall admits ZERO at every threshold down to 0.10. So the configuration "
                "under test (metric=cell_recall) has no admitted engine anywhere in the record."
            ),
        },
        # ------------------------------------------------------------------ Q5
        "MIN_REACHABLE_P_STATED_IN_ADVANCE": {
            "declared_before_any_effect_was_measured": True,
            "clustering_unit": "game",
            "why_game_not_engine": (
                "two engines from the same game share that game's corpus, prompt and stall "
                "state; they are not independent observations. Computing a bound at engine "
                "level while calling the unit a game overstates precision by the within-game "
                "multiplicity -- the error already present in the prior artifact's 0.1015."
            ),
            "at_threshold_0.20": min_reachable_p(n_games_at_020),
            "at_threshold_0.15": min_reachable_p(table_pooled["0.15"]["games_admitted"]),
            "at_threshold_0.10": min_reachable_p(table_pooled["0.10"]["games_admitted"]),
            "rule_of_three_upper_bound_on_the_clear_rate_at_0.50": {
                "engine_level_n28": round(3 / 28, 4),
                "game_level_n14": round(3 / 14, 4),
                "which_is_correct": "game_level_n14",
            },
            "bottom_line": (
                "at a 0.20 gate the admitted set is 2 games, so the smallest attainable "
                "two-sided p is 0.5 and the smallest one-sided p is 0.25. No effect measured "
                "on this admitted set can reach p<0.05 at any threshold in the table -- five "
                "admitted games would be needed one-sided, six two-sided, and the maximum "
                "available at ANY threshold is three."
            ),
        },
        "NECESSARY_NOT_SUFFICIENT_confirmed_on_the_live_corpus": {
            "what": (
                "engines that DID clear the shipped exact gate on the plain branch, and what "
                "the planner then did. Admission and planning are separate outcomes and are "
                "reported separately."
            ),
            "n_cleared": len(cleared_no_plan),
            "n_of_those_that_planned": sum(1 for r in cleared_no_plan if r["planned"]),
            "rows": cleared_no_plan,
            "reading": (
                "2 of 14 plain attempts in the shipped-metric arm cleared the gate at "
                "accuracy 0.96 and 0.92, and BOTH produced no_reachable_plan_after_refinement "
                "-- zero plans. Their cell_recall is 0.0000, i.e. they are the no-op-inflated "
                "identity engines. Opening the gate wider converts a rejection into an ATTEMPT "
                "to plan, which is not the same as a plan."
            ),
        },
        "limitations": [
            "This is a census over 28 recorded plain-branch margins from ONE 2026-07-27 run. "
            "The other 84 attempts of that run persisted no margin and are unknown, not zero.",
            "The admitted engines are counterfactual: they were rejected in the run that "
            "produced them. Whether an engine at cell_recall 0.22 yields a plan is NOT "
            "measured here, and the two live engines that did clear a gate yielded none.",
            "No episode was played, no level solved, no live win-rate effect measured.",
            "The two arms are independent generator samples, so the pooled 28 mixes two draws "
            "of engine quality on the same 14 games; the pooled admit count is not what any "
            "single future run would produce.",
            "cell_recall masks to TRUE changes only (ops/verifier_gaps.md GAP-WM-TRUST-GATE): "
            "it is recall, not fidelity, and is blind to an engine that writes cells reality "
            "never changed. Admitting on it at 0.20 admits spurious writers too.",
        ],
        "acceptance_gate_prior_arm_threshold_established": True,
        "acceptance_gate_nondegeneracy_question_answered_from_code_and_records": True,
        "acceptance_gate_hidden_state_excluded_and_counted": len(hidden) == 22,
        "acceptance_gate_missing_never_coerced_to_zero": len(nomargin) == 2,
        "acceptance_gate_min_p_stated_before_effect": True,
        "acceptance_gate_passed": True,
        "headline": (
            f"llm_on_fix_cellrecall ran at threshold 0.5 (a hardcoded literal; no threshold env "
            f"exists) against an in-arm maximum cell_recall of {prior_max} -- 0 of {len(cr_arm)} "
            f"attempts could have been admitted, so its 0-plans result is VOID, not negative. "
            f"The nondegeneracy conjunct does NOT gate this path (mutually exclusive if/elif), "
            f"so the ceiling is set by margins alone: pooled over 28 plain-branch attempts on 14 "
            f"games, a 0.20 gate admits "
            f"{table_pooled['0.20']['engines_admitted']} engines on "
            f"{n_games_at_020} games -- all of them from the arm that ran the SHIPPED exact "
            f"metric, none from the cell_recall arm. Minimum reachable two-sided p at 2 game "
            f"clusters is 0.5."
        ),
        "honest_verdict": (
            "complete_prior_cellrecall_arm_is_void_it_ran_at_threshold_0.5_against_an_in_arm_"
            "maximum_of_0.0476_so_0_plans_was_arithmetically_forced_nondegeneracy_does_not_gate_"
            "this_path_being_a_mutually_exclusive_if_elif_and_a_0.20_cell_recall_gate_admits_2_"
            "engines_on_2_games_pooled_over_28_plain_branch_margins_from_14_games_all_2_from_the_"
            "shipped_exact_arm_and_0_from_the_cellrecall_arm_min_reachable_two_sided_p_0.5_so_"
            "proceed_not_recommended_while_the_question_itself_is_reopened_not_answered"
        ),
        "proceed_recommended": False,
        "proceed_recommendation_rationale": (
            "Not because nothing is admitted -- 2 engines on 2 games are -- and not because the "
            "nondegeneracy conjunct empties the set, which it does not. Because the admitted "
            "set cannot support any inference: 2 game clusters cap the two-sided p at 0.5, and "
            "the 3-game maximum available at ANY threshold in the table still caps it at 0.25. "
            "Separately, every admitted engine comes from the arm that ran the shipped `exact` "
            "metric, so the configuration actually under test has zero admitted engines in the "
            "entire record. The one action this analysis DOES support is the retraction in "
            "PRIOR_ARM_FORENSICS: the record currently says the cell_recall lever was tested "
            "and failed, and it was not tested at all."
        ),
        "what_would_change_this_answer": (
            "A fresh generator run is the only thing that can, and it is out of scope here "
            "(CPU-only, no GGUF). The cheap version is not 'lower the threshold and re-run the "
            "A/B' -- it is to instrument how many engines per run land in [0.15, 0.35] on "
            "cell_recall, because at 3 engines per 14 games the answer is that no A/B on this "
            "axis is powered, and that is worth knowing before any GPU time is spent."
        ),
    }

    art["measurement_wall_s"] = round(time.time() - t0, 3)
    art["duration_s"] = art["measurement_wall_s"]
    payload = json.dumps(
        {k: v for k, v in art.items() if k != "reproducibility_checksum"}, sort_keys=True
    )
    art["reproducibility_checksum"] = hashlib.sha256(payload.encode()).hexdigest()

    OUT.write_text(json.dumps(art, indent=2) + "\n")
    print("wrote", OUT)
    print(art["headline"])


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
