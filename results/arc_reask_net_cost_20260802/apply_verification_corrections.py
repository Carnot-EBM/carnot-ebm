#!/usr/bin/env python3
"""Apply the corrections that survived an independent verification pass over this run's own data.

WHY A SCRIPT AND NOT A HAND EDIT. The artifact is serialised at `indent=1, sort_keys=True` with
NO trailing newline. Hand-editing it, or re-dumping it at a different indent, rewrites every line
of a 40 KB file -- which is how a sibling artifact turned a six-leaf append into a
+52,572 / -52,562 diff earlier today and stale-ed every freshness acknowledgement that declared
its sha256. This script reads, mutates in memory, and writes back with the EXACT same serialiser
settings, so the diff is only the leaves that actually changed.

NEVER-PRUNE. No original text is deleted. The one factually wrong sentence is preserved verbatim
under `superseded_text` next to its correction, because a record that silently fixes its own
errors is worth less than one that shows what it got wrong.

WHAT IS BEING CORRECTED, and how each was verified against the run's own data rather than against
the summary that reported it:

  C1  A numerator/denominator mismatch. `FINDING_1.shipped_stack_comparison` read "1 hard failure
      in 160 cells", mixing ONE arm's numerator with ALL FOUR arms' denominator. Recount from the
      160 per-cell JSONs: a_off 0, aa 2, b_shipped 1, c_owns 1 = 4 of 160. As written it
      understated the shipped rate 4x and inflated the counterfactual contrast from ~18x to ~71x.
      The qualitative finding (the pre-penalty regime is far more fragile) is unaffected.

  C2  The exposure comparator was the wrong quantity -- though NOT in the way it was first
      flagged. A review asserted that "22/36" is the POST-gate USABLE rate and should have been
      the pre-gate defect rate "23/36 = 64%". That is REFUTED by the founding measurement:
      `confirm_scored.json` carries `control_accepted_a_defective_candidate: 22` of
      `control_attempts_scored: 36`, so 22/36 = 61% IS a real pre-gate defect rate, and 23/36
      appears nowhere in the corpus. The coincidence is genuine and confusing: 22 is BOTH the
      treatment's usable count AND the control's defective-accept count.
      The comparator is nevertheless wrong, for a different reason. The gate cannot fire on a
      CONTROL-arm defect; it fires on a TREATMENT-arm one, and the repeat penalty has already
      removed most defects by then. Recomputed from the 36 raw attempt records: treatment round-0
      defect rate 9/36, `reask_fired` true 9/36. So the gate's own EXPOSURE was 25%, not 61%.

  C3  The pro-gate alternative reading was never named. Gate-OFF arms emitted 3 defective engines
      of 78; gate-ON arms 0 of 78, and 2 of those 3 are 2 of the 3 discordant games driving the
      primary. A reader can conclude the gate silently worked without incrementing its counter.
      It did not, and the rebuttal is mechanical rather than statistical -- but the statistics are
      recorded too, because the mechanical argument is the load-bearing one and should not have to
      carry a p-value it does not need.

  C4  THE ONE THAT MATTERS, and it is not a correction to this run -- it is a correction to what
      this run's subject was. The founding harness (`arc_induce_confirm_20260731/harness/
      confirm_ab.py`) issues the re-ask as an EXTRA `/completion` call; there is no retry ladder
      for it to borrow from. That is `CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS=1` semantics. What
      shipped into `generate()` is the opposite -- the re-ask CONSUMES an attempt. So the shipped
      configuration has never been measured, and the flag that is default-OFF is the one the
      evidence was collected under.

  C5  Two sentences contradicted each other read in isolation (same function, different args).
      Benign, and the consequence was already declared; reconciled explicitly.

  C6  The A/A arm changed the seed base AND the request position at once, so it is not a pure
      noise floor. Direction is conservative (it can only inflate the floor), which is why the
      null it backs is still trustworthy.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "results" / "outer_loop_arc_reask_net_cost_20260802.json"

# The exact serialiser the file was written with. Verified by round-tripping the file on disk:
# json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False) reproduces it BYTE-FOR-BYTE with no
# trailing newline. Changing any of these three rewrites all ~1200 lines.
DUMP = {"indent": 1, "sort_keys": True, "ensure_ascii": False}

WRONG = (
    "1 hard failure in 160 cells on the shipped stack versus 4 in 9 here. The repeat penalty is "
    "doing the work its own source note claims (11 of 13 paired wins), and that is what collapsed "
    "the gate's exposure."
)

RIGHT = (
    "b_shipped hard-failed 1 of its 40 cells (2.5%) against 4 of 9 here (44%) -- roughly 18x. "
    "Across all four arms the shipped stack hard-failed 4 of 160 (2.5%), the same rate. The "
    "repeat penalty is doing the work its own source note claims (11 of 13 paired wins), and that "
    "is what collapsed the gate's exposure."
)


def main() -> int:
    original = ARTIFACT.read_text(encoding="utf-8")
    d = json.loads(original)

    # Guard: refuse if the file is not the serialisation we think it is. A silent reformat is the
    # failure this whole script exists to avoid, so it fails CLOSED rather than writing anyway.
    if json.dumps(d, **DUMP) != original:
        print("REFUSING: artifact does not round-trip at indent=1/sort_keys=True.", file=sys.stderr)
        return 2

    f1 = d["SUPPLEMENTARY_counterfactual_repeat_penalty_off"][
        "FINDING_1_regime_is_far_more_failure_prone"
    ]
    if f1.get("shipped_stack_comparison") != WRONG:
        print(
            "REFUSING: FINDING_1 text is not the one this correction was written against.",
            file=sys.stderr,
        )
        return 2

    f1["shipped_stack_comparison"] = RIGHT
    f1["shipped_stack_comparison_SUPERSEDED_TEXT"] = WRONG
    f1["shipped_stack_comparison_correction_note"] = (
        "Corrected 2026-08-02 by an independent verification pass that recounted from the 160 "
        "per-cell JSONs rather than from the summary. The superseded sentence mixed one arm's "
        "numerator (b_shipped's 1) with all four arms' denominator (160), understating the "
        "shipped hard-failure rate 4x and inflating this contrast from ~18x to ~71x."
    )

    d["CORRECTIONS_20260802_independent_verification_pass"] = {
        "what_this_is": (
            "A second pass that recomputed every headline number from this run's raw per-cell "
            "files and from the founding measurement's raw attempt records, rather than from the "
            "summaries that reported them. Everything below either survived that recomputation or "
            "is a defect the recomputation found. Findings that did NOT survive are recorded too, "
            "under `REFUTED_*`, because a correction that turns out to be wrong is exactly as "
            "important to write down as one that is right."
        ),
        "C1_hard_failure_denominator": {
            "status": "APPLIED",
            "where": (
                "SUPPLEMENTARY_counterfactual_repeat_penalty_off"
                ".FINDING_1_regime_is_far_more_failure_prone"
            ),
            "recount_from_160_per_cell_files": {
                "a_off": 0,
                "aa": 2,
                "b_shipped": 1,
                "c_owns": 1,
                "total": 4,
            },
            "effect_on_conclusions": "none -- the pre-penalty regime is still far more fragile",
        },
        "C2_exposure_comparator_is_the_FIRE_rate_not_the_control_defect_rate": {
            "status": "APPLIED",
            "the_confusing_coincidence": (
                "22/36 is BOTH the treatment arm's usable count in the '13/36 -> 22/36' headline "
                "AND the control arm's defective-accept count. Two different quantities, one "
                "number, and the summary of this run compared against it without saying which."
            ),
            "recomputed_from_the_36_raw_attempt_records": {
                "control_defective_accepts": "22/36 = 61%",
                "treatment_round0_defective": "9/36 = 25%",
                "reask_fired": "9/36 = 25%",
                "source": "results/arc_induce_confirm_20260731/confirm_scored.json -> attempts[]",
            },
            "why_the_fire_rate_is_the_right_comparator": (
                "the gate cannot fire on a CONTROL-arm defect. It sees the TREATMENT arm's "
                "candidate, by which point the repeat penalty has already removed most defects "
                "(61% -> 25%). Comparing this run's 0% fire rate against 61% overstates the "
                "collapse by attributing the penalty's work to the gate's exposure."
            ),
            "like_for_like": (
                "fire rate 25% (9/36, 2026-07-31) -> 0% (0/160, today). "
                "Defect rate 61% -> 1.9% (33x)."
            ),
        },
        "C3_the_pro_gate_alternative_reading_now_named_and_rebutted": {
            "status": "APPLIED",
            "the_alternative": (
                "gate-OFF arms emitted 3 mechanically defective engines of 78; gate-ON arms 0 of "
                "78. Two of those 3 (cn04 r0 missing_return, wa30 r0 engine_raised) are 2 of the "
                "3 discordant games in the primary. A reader can conclude the gate silently "
                "worked and simply failed to increment its counter."
            ),
            "mechanical_rebuttal_load_bearing": (
                "`self.n_induce_defect_reasks += 1` is at arc_executable_world_model.py:6122, "
                "inside the sole `if _defects:` block and immediately before its `continue`. It "
                "is the ONLY increment site in the module (verified by grep: 6122 is the only "
                "`+=` on that name). A firing therefore cannot go unrecorded, so 0 recorded "
                "firings means 0 firings, not 0 recordings."
            ),
            "statistical_rebuttal_secondary": {
                "fisher_exact_two_sided_3_of_78_vs_0_of_78": 0.2452,
                "within_gate_off_arms_a_off_2_of_40_vs_aa_1_of_38": 1.0,
                "reading": (
                    "the two gate-OFF arms already differ from each other by as much as "
                    "they differ from the gate-ON arms"
                ),
            },
        },
        "C4_the_founding_measurement_measured_the_OTHER_flag": {
            "status": "APPLIED -- and this is the largest finding of the verification pass",
            "what_the_founding_harness_does": (
                "results/arc_induce_confirm_20260731/harness/confirm_ab.py lines 405-433: each "
                "scored row is one CONTROL `/completion` call plus one TREATMENT round-0 call, "
                "and -- if round 0 is defective -- one ADDITIONAL `/completion` call whose reply "
                "becomes TREATMENT_final unconditionally. There is no `tries` ladder in that "
                "harness at all; `attempt` is an outer index over seeds and temperatures, and "
                "every attempt is scored as its own independent row."
            ),
            "therefore": (
                "the re-ask was GRANTED its own call. It never competed with the content-failure "
                "retry ladder. That is precisely the semantics of "
                "`CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS=1`, which is DEFAULT OFF."
            ),
            "what_shipped_instead": (
                "`generate()` CONSUMES an attempt: `_reasks_left -= 1` then `continue`, borrowing "
                "from the same `tries=3` ladder the content-failure path needs. The `attempt < "
                "_budget - 1` guard stops only the LAST attempt from falling out of the loop; it "
                "does not stop an earlier re-ask from spending the attempt that would have been "
                "the accept."
            ),
            "consequence": (
                "the shipped configuration of this gate has never been measured. The 13/36 -> "
                "22/36 headline was obtained under grant semantics, and grant semantics is the "
                "flag that is off by default. This inverts the usual reading of arm C: it is not "
                "a speculative fix awaiting evidence, it is the configuration the existing "
                "evidence was collected under."
            ),
            "the_gate_half_of_that_headline_was_never_significant_either": {
                "identified_contrast": (
                    "usable__treatment_final_vs_round0 -- same arm, same seed, same "
                    "penalty, differing ONLY in the re-ask"
                ),
                "result": "2 better / 0 worse, n_discordant = 2, sign test p = 0.5",
                "min_reachable_p_at_n_discordant_2": 0.5,
                "reading": (
                    "the re-ask half of the shipped intervention rests on 2 attempts across 2 "
                    "games, at an n where p <= 0.05 is arithmetically unreachable. The spec's "
                    "'the penalty carries 11 of the 13 paired wins and the re-ask 2' is accurate "
                    "as a decomposition of WINS and should not be read as evidence for the re-ask."
                ),
                "a_contrast_deliberately_NOT_reported_as_a_finding": (
                    "on the 9 fired cells, CONTROL had 4 usable against TREATMENT_final's 2. That "
                    "looks like the gate costing 2 net engines, and this pass initially recorded "
                    "it as such. It is CONFOUNDED and is withdrawn: the cells are selected on "
                    "treatment-arm defectiveness, and control lacks the repeat penalty, so the "
                    "contrast mixes the gate's effect with the penalty's absence. "
                    "final-vs-round0 above is the identified one."
                ),
            },
        },
        "C5_scorer_definition_wording_reconciled": {
            "status": "APPLIED",
            "the_apparent_contradiction": (
                "`components_note` says the scorer uses the SAME definition the shipped gate "
                "uses, while `scorer_asymmetry_declared` says the live gate is stricter."
            ),
            "reconciliation": (
                "both are true of one function called with different arguments. "
                "`usable_worker.py` calls `arc_engine_static_validation.validate_engine_code` -- "
                "the shipped function -- with an explicit `stop_type=None`, because a post-hoc "
                "scorer cannot know how generation terminated. The live gate passes a real "
                "`stop_type` and can therefore additionally raise "
                "`truncated_before_required_symbols`."
            ),
            "why_it_cannot_confound": (
                "the blindness applies identically to all four arms, so it makes every arm's "
                "`usable` count generous in the same direction. Truncation is still counted "
                "against an arm via the content-failure path (induce_ok False -> -1)."
            ),
        },
        "C6_the_A_A_floor_confounds_seed_base_with_request_position": {
            "status": "APPLIED",
            "the_limitation": (
                "arm order was fixed (a_off, b_shipped, c_owns, aa), so the A/A arm changed the "
                "seed base 8100 -> 8200 AND moved from request position 1 to position 4 at the "
                "same time. It is therefore not a pure noise floor."
            ),
            "direction_is_conservative": (
                "adding a second source of variation can only INFLATE the measured floor, never "
                "shrink it. So the floor is an upper bound on noise, which is the direction that "
                "makes 'the design resolves a 3-point swing and saw none between arms' weaker, "
                "not stronger -- and the honest phrasing is the one already in "
                "`pairing_internal_validity`: a 3-point net difference is INSIDE this design's "
                "noise, so identical 38/38/38 cannot discriminate 'costs nothing' from 'costs up "
                "to ~3 net points and this design cannot see it'. The zero-exposure COUNT, not "
                "the net metric, is what carries the headline."
            ),
        },
        "REFUTED_the_23_of_36_replacement_figure": {
            "status": "NOT APPLIED -- refuted by the founding measurement's own data",
            "the_proposed_correction": (
                "a review held that '61% (22/36)' is the POST-gate usable rate and should be "
                "replaced by the pre-gate defect rate '23/36 = 64%'."
            ),
            "why_it_is_refuted": (
                "`confirm_scored.json` records `control_accepted_a_defective_candidate: 22` of "
                "`control_attempts_scored: 36`, recomputed independently from the 36 raw attempt "
                "records as 22 (61.1%). 23/36 appears nowhere in the corpus. 22/36 = 61% IS a "
                "genuine pre-gate defect rate; the reviewer was misled by the same coincidence "
                "recorded in C2."
            ),
            "what_survived_of_it": (
                "the underlying concern -- that the wrong quantity was being compared against -- "
                "was correct, and is applied as C2 with the right replacement (the 25% fire "
                "rate), not the proposed one."
            ),
        },
    }

    ARTIFACT.write_text(json.dumps(d, **DUMP), encoding="utf-8")
    print(f"OK: corrections applied to {ARTIFACT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
