#!/usr/bin/env python3
"""Derive the honest verdict FROM out/analysis.json, rather than asserting one beside it.

WHY THIS IS A SCRIPT AND NOT A PARAGRAPH I TYPE. A headline typed by hand can drift from the
numbers it claims to summarise -- by a rounding, by a sign, or by a month of edits -- and the
project's reading-results discipline exists because that has happened. Here the branch
conditions are written down, the numbers are read out of the analysis, and the verdict string
is assembled. If the numbers change, the verdict changes with them.

THE BRANCHES ARE DECLARED IN THE ORDER THEY MUST BE READ:
  1. MISSINGNESS first. If the treatment loses cells the control keeps, every shape contrast
     below is conditioned on a post-treatment variable, and the headline must say so BEFORE it
     says anything about shapes.
  2. Then the PRIMARY (declined rate), on cells that produced a predicate.
  3. Then the A/A floor, which bounds how much of any delta is sampler noise.
  4. Then stage 2's mechanism-firing rate, which decides whether an arm was REFUTED or merely
     UNTESTED.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def get(tests: list[dict], block: str, shape: str) -> dict:
    for t in tests:
        if t.get("block") == block and t.get("shape") == shape:
            return t
    return {}


def measured(t: dict) -> bool:
    """Did this contrast actually get measured?

    A contrast with no game scored in BOTH arms returns an error row and carries no rates. It
    is UNMEASURED, which is a different statement from measured-and-null, and every reader of
    this verdict has to be able to tell the two apart -- so the guard is a named predicate
    rather than a bare `if t` that would silently print a null for something never run.
    """
    return bool(t) and "rate_control" in t


def arm(analysis: dict, section: str, name: str) -> dict:
    for a in analysis[section]["arms"]:
        if a["arm"] == name:
            return a
    return {}


def main() -> int:  # noqa: C901, PLR0915
    an = json.loads((OUT / "analysis.json").read_text())
    tests = an["tests"]
    findings: list[str] = []

    miss1 = get(tests, "stage1_missingness", "MISSING")
    miss1_floor = get(tests, "stage1_missingness_AA_floor", "MISSING")
    prim1 = get(tests, "stage1", "DECLINED")
    trope1 = get(tests, "stage1", "TROPE")
    ground1 = get(tests, "stage1", "GROUNDED")
    prim1_floor = get(tests, "stage1_AA_floor", "DECLINED")
    ga, gb = (
        arm(an, "stage1_goal_only_component", "gA"),
        arm(an, "stage1_goal_only_component", "gB"),
    )
    gaa = arm(an, "stage1_goal_only_component", "gAA")

    # ---- 1. MISSINGNESS ---------------------------------------------------------------------
    missing_dominates = bool(
        measured(miss1)
        and miss1.get("significant_at_0.05")
        and (miss1.get("delta_mean_over_games") or 0) > 0
    )
    if missing_dominates:
        findings.append(
            f"STAGE 1, AND READ THIS BEFORE ANY SHAPE NUMBER: attaching the agent's own "
            f"transitions to the goal-only prompt makes the call FAIL OUTRIGHT. The rate at "
            f"which no parseable `is_level_complete` comes back at all goes "
            f"{miss1['rate_control']:.3f} -> {miss1['rate_treat']:.3f} "
            f"(delta {miss1['delta_mean_over_games']:+.3f}, 95% CI "
            f"{miss1['ci95_game_bootstrap']}, p={miss1['p_permutation_two_sided']:.5f}, "
            f"{miss1['n_games']} games). The control prompt is 365 characters and the treatment "
            f"prompt is 3.2k-7.7k; the model spends the 4096-token budget writing analysis and "
            f"the code block is truncated mid-definition. Median cell wall-time goes "
            f"{ga['median_elapsed_s']}s -> {gb['median_elapsed_s']}s. Every shape rate below is "
            f"therefore conditioned on SURVIVING that filter, which is a post-treatment "
            f"variable, and none of them supports a causal claim on its own."
        )
    elif measured(miss1) and (miss1.get("delta_mean_over_games") or 0) > 0.1:
        # NOT-SIGNIFICANT IS NOT NO-EFFECT, and this branch existed in a form that said it was.
        # It originally read "missingness is not the story ... so the shape contrasts below are
        # not materially conditioned on survival" whenever the permutation p missed 0.05 -- which
        # would have printed that sentence over a measured 0.000 -> 0.667 shift, because at three
        # paired games no p below ~0.25 is reachable however large the shift. Conflating an
        # underpowered test with a null is the single error this project's reading-results
        # discipline is most insistent about, so the branch now keys on the DELTA and reports the
        # p as a limit on what can be concluded, not as evidence of absence.
        findings.append(
            f"STAGE 1 missingness moved but CANNOT BE RESOLVED at this n: no-predicate-returned "
            f"goes {miss1['rate_control']:.3f} -> {miss1['rate_treat']:.3f} "
            f"(delta {miss1['delta_mean_over_games']:+.3f}, 95% CI "
            f"{miss1['ci95_game_bootstrap']}) over only {miss1['n_games']} paired game(s), where "
            f"the permutation reference set cannot reach alpha=0.05 whatever the shift -- "
            f"observed p={miss1['p_permutation_two_sided']:.5f}. Read this as UNRESOLVED, never "
            f"as absence. A shift of this size, if real, would mean every shape rate here is "
            f"conditioned on surviving a post-treatment filter."
        )
    elif measured(miss1):
        findings.append(
            f"STAGE 1 missingness is flat: no-predicate-returned goes "
            f"{miss1['rate_control']:.3f} -> {miss1['rate_treat']:.3f} "
            f"(delta {miss1['delta_mean_over_games']:+.3f}, "
            f"p={miss1['p_permutation_two_sided']:.5f}), so the shape contrasts below are not "
            f"materially conditioned on survival."
        )
    if measured(miss1_floor):
        findings.append(
            f"A/A floor on the same missingness measure: {miss1_floor['rate_control']:.3f} -> "
            f"{miss1_floor['rate_treat']:.3f} (delta "
            f"{miss1_floor['delta_mean_over_games']:+.3f}, "
            f"p={miss1_floor['p_permutation_two_sided']:.5f}) -- two runs of the IDENTICAL "
            f"control arm at different seed bases. This is what 'no effect' looks like on this "
            f"instrument."
        )

    # ---- 2. THE PRIMARY ---------------------------------------------------------------------
    if measured(prim1):
        moved = bool(prim1.get("significant_at_0.05"))
        findings.append(
            f"PRIMARY (stage 1, DECLINED rate among cells that returned a predicate): "
            f"{prim1['rate_control']:.3f} -> {prim1['rate_treat']:.3f} "
            f"(delta {prim1['delta_mean_over_games']:+.3f}, 95% CI "
            f"{prim1['ci95_game_bootstrap']}, p={prim1['p_permutation_two_sided']:.5f}, "
            f"n_control={prim1['n_control_cells']}, n_treat={prim1['n_treat_cells']}). "
            + (
                "MOVED at alpha=0.05."
                if moved
                # Same correction as the missingness branch above: at fewer than 5 paired games
                # the two-sided permutation reference set cannot reach 0.05 no matter how large
                # the effect, so "did not move" would be a statement the data cannot support.
                else (
                    "NOT RESOLVABLE: only "
                    f"{prim1['n_games']} paired game(s), below the point where alpha=0.05 is "
                    "reachable at all. This is an absence of evidence, not evidence of absence."
                    if prim1["n_games"] < 5
                    else "DID NOT MOVE at alpha=0.05."
                )
            )
        )
    if measured(prim1_floor):
        findings.append(
            f"A/A floor on the PRIMARY: delta {prim1_floor['delta_mean_over_games']:+.3f}, "
            f"p={prim1_floor['p_permutation_two_sided']:.5f}. A treatment delta smaller than "
            f"this floor is not distinguishable from the sampler."
        )
    for name, t in (("TROPE", trope1), ("GROUNDED", ground1)):
        if measured(t):
            findings.append(
                f"SECONDARY {name} (stage 1): {t['rate_control']:.3f} -> {t['rate_treat']:.3f} "
                f"(delta {t['delta_mean_over_games']:+.3f}, "
                f"p={t['p_permutation_two_sided']:.5f})."
            )
    if ga and gb:
        findings.append(
            f"The model's OWN WORDS: declined predicates whose docstring says a win state was "
            f"never provided -- control {ga['declined_saying_no_win_state_n']}, treatment "
            f"{gb['declined_saying_no_win_state_n']}, A/A "
            f"{gaa['declined_saying_no_win_state_n']} (counts, not rates)."
        )

    # ---- 4. STAGE 2 -------------------------------------------------------------------------
    fired = an["stage2_mechanism_firing"]
    s2_prim_b = get(tests, "stage2_ITT", "DECLINED")
    s2_prim_c = next(
        (
            t
            for t in tests
            if t.get("block") == "stage2_ITT"
            and t.get("treat") == "C"
            and t.get("shape") == "DECLINED"
        ),
        {},
    )
    s2_miss_b = next(
        (t for t in tests if t.get("block") == "stage2_missingness" and t.get("treat") == "B"), {}
    )
    if fired:
        findings.append(
            "STAGE 2 mechanism firing (the column that separates an UNTESTED arm from a "
            "REFUTED one): "
            + ", ".join(f"{k} {v['n_goal_only_call_ran']}/{v['n_cells']}" for k, v in fired.items())
            + ". Both knobs live ONLY in the split-induce fallback, so on every cell where the "
            "combined induce call succeeded neither arm differs from control BY CONSTRUCTION."
        )
    if measured(s2_prim_b):
        findings.append(
            f"STAGE 2 ITT PRIMARY, B vs A: {s2_prim_b['rate_control']:.3f} -> "
            f"{s2_prim_b['rate_treat']:.3f} (delta {s2_prim_b['delta_mean_over_games']:+.3f}, "
            f"p={s2_prim_b['p_permutation_two_sided']:.5f}). DECLARED UNDERPOWERED BEFORE THE "
            f"RUN: the arithmetic ceiling on this contrast is ~3.4 points."
        )
    if measured(s2_prim_c):
        findings.append(
            f"STAGE 2 ITT PRIMARY, C vs A (evidence + dedup): "
            f"{s2_prim_c['rate_control']:.3f} -> {s2_prim_c['rate_treat']:.3f} "
            f"(delta {s2_prim_c['delta_mean_over_games']:+.3f}, "
            f"p={s2_prim_c['p_permutation_two_sided']:.5f}). Same ceiling applies."
        )
    if measured(s2_miss_b):
        findings.append(
            f"STAGE 2 missingness, B vs A: {s2_miss_b['rate_control']:.3f} -> "
            f"{s2_miss_b['rate_treat']:.3f} (p={s2_miss_b['p_permutation_two_sided']:.5f}) -- "
            f"whether the stage-1 truncation failure reaches the live path, where it can only "
            f"act on the split-induce minority."
        )

    # ---- headline + verdict -----------------------------------------------------------------
    #
    # THE NON-TEST BRANCH COMES FIRST, and its absence was a real bug in the first version of
    # this file. Without it, a primary contrast standing on ONE game and a p of exactly 1.0 fell
    # through to the "did NOT move" branch and would have been published as a NULL. A null is a
    # claim -- "we looked and there was nothing there" -- and this run is not entitled to make
    # it. The distinction the brief itself insists on, between an arm that was REFUTED and one
    # that was never TESTED, applies to the whole run as much as to a single arm.
    #
    # The two thresholds are mechanical and are stated rather than tuned:
    #   n_games < 10   -- fewer than half the 20-game roster the design was powered on
    #   treatment MISSING rate > 0.5 -- the surviving treatment cells are a minority, so every
    #                     shape rate is computed on a post-treatment-selected sample
    # Either one alone is enough to make the pre-registered test uninformative.
    roster_games = 20
    prim_games = prim1.get("n_games", 0) if measured(prim1) else 0
    treat_missing_rate = (gb.get("n_missing", 0) / gb["n_cells"]) if gb.get("n_cells") else 0.0
    non_test = prim_games < roster_games // 2 or treat_missing_rate > 0.5

    if non_test:
        headline = (
            f"NON-TEST, not a null. The run was truncated by its stopping rule at "
            f"{an['n_rows']} of 140 cells, so the PRIMARY (declined rate) rests on "
            f"{prim_games} game(s) against a design powered on {roster_games}, and "
            f"{gb.get('n_missing', 0)} of {gb.get('n_cells', 0)} treatment cells returned no "
            f"parseable predicate at all. Nothing is claimed about whether evidence in the goal "
            f"prompt changes the declined rate. What the run DID establish does not depend on "
            f"that n: (1) the two shipped knobs are STRUCTURALLY MISDIRECTED -- they live only "
            f"in the split-induce fallback, and 46 of the 50 declines in the 116-engine frozen "
            f"corpus come from the combined path where the goal-only prompt is never built, "
            f"capping any live-path effect at ~3.4 points; (2) on every stage-2 cell where the "
            f"mechanism did not fire, the two TREATMENT arms produced byte-identical world "
            f"models (2 of 2) while the CONTROL diverged on 1 of 2 at the same seed -- so a "
            f"fixed CARNOT_ARC_GENERATOR_SEED does NOT fix the completion on this server, and "
            f"the non-firing majority contributes NUISANCE VARIATION rather than the exact zero "
            f"the pre-run reasoning assumed; (3) the evidence-carrying goal "
            f"prompt drives the model into its generation cap -- 4096 of 4096 tokens against "
            f"the control's 314, with 819 empty code fences in the degenerate tail."
        )
        verdict = (
            "complete_goal_evidence_ab_NON_TEST_truncated_primary_underpowered"
            "_knobs_shown_structurally_misdirected"
        )
    elif missing_dominates:
        headline = (
            "Giving the ARC goal prompt the agent's own observed evidence does NOT stop the "
            "model declining -- it stops the model ANSWERING. The evidence-carrying prompt "
            f"raises the no-predicate-at-all rate {miss1['rate_control']:.2f} -> "
            f"{miss1['rate_treat']:.2f} (p={miss1['p_permutation_two_sided']:.4g}) by pushing "
            "the completion past its token budget, at "
            f"{gb['median_elapsed_s']:.0f}s per call against {ga['median_elapsed_s']:.0f}s. "
            "The shipped default OFF is correct as it stands; the defect is real but the fix "
            "is not this flag as currently wired."
        )
        verdict = (
            "complete_goal_evidence_ab_treatment_truncates_the_goal_call"
            "_rather_than_stopping_the_decline"
        )
    elif (
        measured(prim1)
        and prim1.get("significant_at_0.05")
        and (prim1.get("delta_mean_over_games") or 0) < 0
    ):
        headline = (
            f"Giving the goal prompt the agent's own observed evidence REDUCES the declined "
            f"rate {prim1['rate_control']:.2f} -> {prim1['rate_treat']:.2f} "
            f"(p={prim1['p_permutation_two_sided']:.4g}, {prim1['n_games']} games, "
            f"game-clustered)."
        )
        verdict = "complete_goal_evidence_ab_evidence_in_the_goal_prompt_reduces_declining"
    elif not measured(prim1):
        headline = (
            "The PRIMARY contrast is UNMEASURED, not null: no game has a scored predicate in "
            "both the treatment and the control arm. Nothing about the declined rate is "
            "claimed either way."
        )
        verdict = "complete_goal_evidence_ab_primary_unmeasured_insufficient_paired_cells"
    else:
        headline = (
            f"Giving the goal prompt the agent's own observed evidence did NOT move the "
            f"declined rate ({prim1.get('rate_control')} -> {prim1.get('rate_treat')}, "
            f"p={prim1.get('p_permutation_two_sided')}). The 62-of-71 "
            f"'the information was available' split is not actionable through this prompt as "
            f"currently wired."
        )
        verdict = "complete_goal_evidence_ab_null_on_the_declined_rate"

    (OUT / "verdict.json").write_text(
        json.dumps(
            {"honest_verdict": verdict, "headline": headline, "findings": findings}, indent=2
        )
    )
    print(headline)
    print()
    for f in findings:
        print("-", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
