#!/usr/bin/env python3
"""METRIC-HEADROOM, STEP 3 -- assemble the scored artifact from steps 1 and 2.

Pure assembly plus the derived judgements. It runs no engine and loads no model; every number it
writes is read from `metric_scores_raw.json` (the 48-candidate census), `positive_control.json`
(the constructed-quality ladder), or the two upstream artifacts it cross-references
(`experiment_6018_object_perception_heldout_ab.json` and the frozen `bestofn_scored.json`).

THE JUDGEMENT THIS FILE ENCODES, stated up front so it can be argued with rather than inferred.
A metric is RECOMMENDABLE as an A/B primary only if it clears all FIVE of:

  H1  RESOLVES a constructed quality ladder -- weakly monotone with more than one distinct value
      on at least one game where it can be measured at all.
  H2  Does NOT reward inertness. The degenerate identity engine ("nothing ever changes") must not
      outscore a real-but-bad engine. This is the one that eliminates every object-level metric
      and `spurious_changed_cells`, and it is not a theoretical worry: 13 of the 40 stall-path
      candidates in the preceding phase predicted that no action changes anything.
  H3  PENALISES spurious writes. A metric blind to invented changes can be won by an engine that
      writes everywhere. This is the one that eliminates `cell_recall`, whose blindness is
      already documented in `VerifyResult` and is demonstrated here rather than assumed.
  H4  SEPARATES the real frozen candidates on MORE THAN ONE game. Not an aesthetic preference --
      the per-game paired sign test's minimum reachable two-sided p is `2 * 0.5**n_discordant`,
      which is 1.0 at n_discordant = 1. A primary that grades a single game admits no test AT ALL,
      which is the identical failure mode to the one being fixed, relocated from "the metric does
      not vary" to "it varies in only one place". This eliminates `change_exact_accuracy` (gradable
      on tn36 alone) and `spurious_changed_cells`.
  H5  Has ADEQUATE DYNAMIC RANGE: the constructed ladder must traverse at least half the metric's
      scale on EVERY game where it is measurable. A metric that separates a perfect engine from a
      totally wrong one by 0.0002 is technically non-constant and practically useless -- an A/B on
      it needs an n nobody is going to run. This eliminates `exact_match_accuracy` (range 0.0 on
      lp85, 0.07 on sc25, 0.2 on ft09), both `grid_agreement` variants (0.0002-0.0035) and
      `changed_cell_jaccard` (0.14).

      THE 0.5 THRESHOLD IS A JUDGEMENT and is the only one of the five that is. H1-H4 are
      pass/fail from the data. The raw `control_min_dynamic_range` is reported per metric so a
      reader who prefers a different bar can apply it. At any bar from 0.14 to 1.0 the
      surviving set is the same single metric; below 0.139 `changed_cell_jaccard`
      rejoins it, and below 0.0003 so does everything except the object metrics and
      `cell_recall`, which fail H2/H3 at every bar.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent
# `.../<repo>/results/arc_metric_headroom_20260801/harness/build_artifact.py` -> up 3 FROM THE
# FILE. Derived, never hardcoded (CLAUDE.md "Test-Run Record Integrity" rule 4).
REPO = pathlib.Path(__file__).resolve().parents[3]
BON = REPO / "results" / "arc_induce_bestofn_20260731"
EXP6018 = REPO / "results" / "experiment_6018_object_perception_heldout_ab.json"

RECOMMENDED = "change_fidelity"
# See H5 in the module docstring: the one threshold in this file that is a judgement rather
# than a reading. Reported alongside every metric's raw range so it can be re-applied.
MIN_DYNAMIC_RANGE = 0.5


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:  # noqa: C901
    t0 = time.monotonic()
    raw = json.loads((OUT_DIR / "metric_scores_raw.json").read_text())
    pc = json.loads((OUT_DIR / "positive_control.json").read_text())
    e6018 = json.loads(EXP6018.read_text())
    scored = json.loads((BON / "bestofn_scored.json").read_text())

    a = raw["metric_analysis"]
    res = pc["resolution"]
    hbf = e6018["analysis"]["heldout_by_field"]

    # exp6018's own field names -> ours. Only these six have DIRECT two-arm evidence; every other
    # metric below is new in this phase and its two-arm behaviour is UNTESTED, which is said out
    # loud rather than implied by omission.
    e6018_field = {
        "exact_match_accuracy": "accuracy",
        "cell_recall": "cell_recall",
        "change_fidelity": "change_fidelity",
        "correct_changed_cells": "correct_changed_cells",
        "spurious_changed_cells": "spurious_changed_cells",
        "change_exact_accuracy": None,  # 6018's `n_changes_correct` is the COUNT, not the rate
    }

    metrics_tried = []
    for key, an in a.items():
        r = res.get(key, {})
        f6 = e6018_field.get(key)
        two_arm = hbf.get(f6) if f6 else None
        # exp6018's `n_changes_correct` is the count form of our `change_exact_accuracy`; both are
        # zero exactly when no changing transition is predicted exactly, so its floored/not verdict
        # transfers even though the scale does not.
        if key == "change_exact_accuracy":
            two_arm = hbf.get("n_changes_correct")

        h1 = bool(r.get("n_games_ladder_resolved", 0) > 0)
        h2 = not bool(r.get("rewards_inertness_disqualifying"))
        h3 = bool(r.get("penalises_spurious_writes_on_every_measurable_game"))
        # H4: >= 2 gradable games, because 2 * 0.5**1 == 1.0 -- one discordant pair admits no test.
        h4 = bool(an["n_discordant_pairs_available"] > 0 and an["n_gradable_games"] >= 2)
        rng = r.get("min_dynamic_range_over_ladder")
        h5 = bool(rng is not None and rng >= MIN_DYNAMIC_RANGE)

        metrics_tried.append(
            {
                "name": key,
                "definition": an["definition"],
                "family": an["family"],
                "higher_is_better": an["higher_is_better"],
                # ---- census over the 48 FROZEN candidates (this phase's own measurement) ----
                "n_candidates_measured": an["n_measured"],
                "distinct_values": an["distinct_values"],
                "value_min": an["value_min"],
                "value_max": an["value_max"],
                "is_constant_across_candidates": an["is_constant"],
                "is_hard_floored_all_zero": an["is_hard_floored_all_zero"],
                "separates_candidates": an["separates_candidates"],
                "n_discordant_pairs_available": an["n_discordant_pairs_available"],
                "n_within_game_pairs": an["n_within_game_pairs"],
                "n_gradable_games_of_5_stall": an["n_gradable_games"],
                "spearman_vs_exact_match_per_game": an["spearman_vs_exact_match_per_game"],
                "spearman_vs_exact_match_pooled": an[
                    "spearman_vs_exact_match_pooled_within_game_centred"
                ],
                # ---- constructed-quality positive control (this phase) ----
                "control_n_games_measurable": r.get("n_games_measurable"),
                "control_n_games_ladder_resolved": r.get("n_games_ladder_resolved"),
                "control_n_games_ladder_collapsed": r.get("n_games_ladder_collapsed_to_one_value"),
                "control_min_dynamic_range": r.get("min_dynamic_range_over_ladder"),
                "control_identity_engine_outranks_a_real_engine_on_n_games": r.get(
                    "n_games_identity_outranks_a_real_engine"
                ),
                "control_penalises_spurious_writes": r.get(
                    "penalises_spurious_writes_on_every_measurable_game"
                ),
                # ---- exp6018's OWN two-arm evidence, where it measured the same channel ----
                "exp6018_two_arm_measured": bool(two_arm),
                "exp6018_floored_in_both_arms": (
                    bool(two_arm.get("all_cells_exactly_zero_both_arms")) if two_arm else None
                ),
                "exp6018_n_distinct_cell_values_over_168_cells": (
                    two_arm.get("n_distinct_cell_values") if two_arm else None
                ),
                "exp6018_n_discordant_per_game_pairs_of_14": (
                    (two_arm.get("sign_test") or {}).get("n_discordant") if two_arm else None
                ),
                "exp6018_test_was_possible": (
                    (two_arm.get("sign_test") or {}).get("test_was_possible") if two_arm else None
                ),
                "exp6018_min_reachable_two_sided_p": (
                    (two_arm.get("sign_test") or {}).get(
                        "min_reachable_two_sided_p_at_this_discordance"
                    )
                    if two_arm
                    else None
                ),
                # ---- the four criteria ----
                "H1_resolves_quality_ladder": h1,
                "H2_does_not_reward_inertness": h2,
                "H3_penalises_spurious_writes": h3,
                "H4_separates_frozen_candidates_on_2plus_games": h4,
                "H5_adequate_dynamic_range": h5,
                "failed_criteria": [
                    n
                    for n, ok in (
                        ("H1_resolves_quality_ladder", h1),
                        ("H2_does_not_reward_inertness", h2),
                        ("H3_penalises_spurious_writes", h3),
                        ("H4_separates_frozen_candidates_on_2plus_games", h4),
                        ("H5_adequate_dynamic_range", h5),
                    )
                    if not ok
                ],
                "recommendable_as_ab_primary": bool(h1 and h2 and h3 and h4 and h5),
            }
        )
    metrics_tried.sort(key=lambda m: (-int(m["recommendable_as_ab_primary"]), m["name"]))

    rec = next(m for m in metrics_tried if m["name"] == RECOMMENDED)

    # NO-OP CREDIT DECOMPOSITION. Whether exact-match's spread on THIS corpus is dynamics or
    # no-op credit is the single most load-bearing check in the whole analysis, because without it
    # this phase would have concluded "exact-match is fine after all, exp6018 was unlucky".
    rows = [r for r in raw["rows"] if r["is_stall_game"]]
    noop_decomp = {}
    for g in raw["stall_games"]:
        gr = [r for r in rows if r["game"] == g and r.get("status") == "ok"]
        ex = sorted({round(r["exact_match_accuracy"], 6) for r in gr})
        ncc = sorted({r.get("heldout_n_changes_correct") for r in gr})
        s = next(x for x in raw["split_provenance"] if x["game"] == g)
        noop_decomp[g] = {
            "n_heldout": s["n_heldout"],
            "heldout_n_changing": s["heldout_n_changing"],
            "heldout_n_noop": s["n_heldout"] - s["heldout_n_changing"],
            "change_dominated": bool(s["heldout_n_changing"] * 2 > s["n_heldout"]),
            "exact_match_distinct_values": ex,
            "n_changes_correct_across_candidates": ncc,
            "exact_match_varies": len(ex) > 1,
            "every_candidate_got_zero_changing_rows_exact": ncc == [0],
            "exact_match_variation_is_purely_noop_credit": bool(len(ex) > 1 and ncc == [0]),
            "change_fidelity_distinct_values": sorted(
                {round(r["change_fidelity"], 6) for r in gr if r.get("change_fidelity") is not None}
            ),
        }

    # AGREEMENT WITH THE FROZEN RUN'S OWN NUMBERS. This re-scoring re-executes the same engines
    # through the same shipped verifier, so the channels the 2026-07-31 run already recorded
    # (`heldout_accuracy`, `heldout_cell_recall`, `heldout_n_changes_correct`) must come back
    # IDENTICAL. If they do not, something about the corpus, the split or the verifier moved
    # underneath this analysis and every new number is suspect -- which is exactly the risk the
    # k-pin note describes. Checking it is cheap and makes the claim falsifiable rather than
    # assumed.
    frozen_by_key = {f"{c['game']}|{c['candidate']}": c for c in scored["candidates"]}
    agree = {"n_compared": 0, "n_mismatched": 0, "mismatches": []}
    for r in raw["rows"]:
        if r.get("status") != "ok":
            continue
        f = frozen_by_key.get(f"{r['game']}|{r['candidate']}")
        if not f or f.get("score_status") != "ok":
            continue
        for mine, theirs in (
            ("exact_match_accuracy", "heldout_accuracy"),
            ("cell_recall", "heldout_cell_recall"),
            ("heldout_n_changes_correct", "heldout_n_changes_correct"),
            ("heldout_n_changing", "heldout_n_changing"),
        ):
            a_v, b_v = r.get(mine), f.get(theirs)
            if a_v is None and b_v is None:
                continue
            agree["n_compared"] += 1
            if a_v is None or b_v is None or abs(float(a_v) - float(b_v)) > 1e-4:
                agree["n_mismatched"] += 1
                agree["mismatches"].append(
                    {
                        "key": f"{r['game']}|{r['candidate']}",
                        "field": mine,
                        "this_run": a_v,
                        "frozen_run": b_v,
                    }
                )
    agree["identical"] = agree["n_mismatched"] == 0
    agree["what_this_checks"] = (
        "the three held-out channels the frozen 2026-07-31 scoring already recorded re-derive "
        "EXACTLY from the same completions through the same shipped WorldModelVerifier, on the "
        "split re-proven against the frozen prompt text"
    )

    inputs = [
        HERE / "metric_worker.py",
        HERE / "score_metrics.py",
        HERE / "build_artifact.py",
        OUT_DIR / "metric_scores_raw.json",
        OUT_DIR / "positive_control.json",
        BON / "bestofn_scored.json",
        BON / "split.json",
        EXP6018,
    ]
    input_sha = {str(p.relative_to(REPO)): sha256_file(p) for p in inputs if p.exists()}
    checksum = hashlib.sha256(
        json.dumps(input_sha, sort_keys=True).encode()
        + json.dumps(metrics_tried, sort_keys=True).encode()
    ).hexdigest()

    art = {
        "experiment": "arc_metric_headroom_20260801",
        "title": (
            "Is there ANY graded held-out induction metric with headroom? Re-scoring the 48 "
            "frozen best-of-N candidates under 13 metrics, plus a constructed-quality positive "
            "control."
        ),
        "run_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "milestone": "outer-loop 2026-08-01",
        "requirement": "REQ-ARC-WMTE-5830 (blocked on instrumentation, per experiment_6018)",
        "why_this_exists": (
            "experiment_6018 A/B'd object perception and returned "
            "`complete_object_perception_heldout_ab_unmeasurable_instrument_floor_primary_zero_"
            "both_arms_no_test_possible_zero_discordant_pairs_n_support_games_14`. Its "
            "pre-registered primary -- held-out exact-full-grid transition accuracy -- was "
            "exactly 0.0 in BOTH arms on all 168 cells, so there were zero discordant per-game "
            "pairs and the minimum reachable two-sided p was 1.0 whatever the treatment did. "
            "That is an instrument floor, not a null about object perception: it is the "
            "FALSE_NEGATIVE_RISK pattern in which 'the method does not help' and 'the "
            "measurement had no headroom' are indistinguishable. No representation change can be "
            "evaluated until a graded metric with headroom exists, so this phase looks for one "
            "and is explicitly willing to report that none does."
        ),
        "honest_verdict": (
            "complete_graded_metric_with_headroom_found_change_fidelity_39_discordant_within_game"
            "_candidate_pairs_vs_0_for_exact_match_on_change_dominated_splits_object_and_grid_"
            "metrics_disqualified_for_ranking_the_identity_engine_above_real_engines"
        ),
        "has_headroom": True,
        "recommended_metric": RECOMMENDED,
        "recommended_metric_definition": rec["definition"],
        "n_discordant_pairs_available": rec["n_discordant_pairs_available"],
        "min_reachable_two_sided_p": rec["exp6018_min_reachable_two_sided_p"],
        "min_reachable_two_sided_p_basis": (
            "exp6018's OWN two-arm run already computed change_fidelity as an exploratory "
            "secondary on its 14-game roster and recorded n_discordant = 12 of 14 per-game pairs, "
            "test_was_possible = True, min reachable two-sided p = 2 * 0.5**12 = 0.00048828. That "
            "is the figure that matters for the A/B design, because it is measured on the roster "
            "the A/B actually uses. On THIS phase's 5-game stall corpus the same metric grades 3 "
            "games, so its own min reachable p here is only 2 * 0.5**3 = 0.25 -- reported "
            "separately as `recommended_metric_on_this_corpus` and NOT used as the headline, "
            "because a 5-game corpus is not the A/B roster."
        ),
        "recommended_metric_on_this_corpus": {
            "n_gradable_games_of_5_stall": rec["n_gradable_games_of_5_stall"],
            "n_discordant_within_game_candidate_pairs": rec["n_discordant_pairs_available"],
            "n_within_game_candidate_pairs": rec["n_within_game_pairs"],
            "min_reachable_two_sided_p_at_this_gradability": a[RECOMMENDED][
                "min_reachable_two_sided_p_at_this_gradability"
            ],
        },
        "why_change_fidelity_and_not_the_others": {
            "vs_exact_match_accuracy": (
                "Exact-match is not merely coarse, it is ANTI-CORRELATED with dynamics quality "
                "on this corpus. On ft09 it ranks candidate 1 BEST (0.8) while that candidate's "
                "change_fidelity is 0.0 -- it earns the score by predicting that nothing ever "
                "changes on a split that is 16 no-ops to 4 changes. Its between-candidate "
                "variation on ft09 is 100% no-op credit: `n_changes_correct` is 0 for every "
                "candidate on that game. On tu93, whose held-out split is 19 of 19 CHANGING, "
                "exact-match is 0.0 for all five runnable candidates -- exp6018's floor, "
                "reproduced on an independent corpus. It floors exactly where the split is "
                "change-dominated, which is the regime that matters."
            ),
            "vs_change_exact_accuracy": (
                "The honest strict form of exact-match, and it does not floor to a single value "
                "corpus-wide -- but it is gradable on only ONE of five games (tn36) and takes "
                "just 2 distinct values on 3 of the 4 games where the positive control can "
                "measure it. Near-binary is the original problem, restated."
            ),
            "vs_cell_recall": (
                "Identical to change_fidelity on the corruption ladder, and it fails H3: the "
                "positive control's oracle-plus-256-invented-cells engine scores cell_recall 1.0 "
                "on every measurable game. `VerifyResult` documents this blindness; this is the "
                "measurement of it. An arm could win on cell_recall by producing engines that "
                "write everywhere."
            ),
            "vs_object_metrics": (
                "The metrics this phase ADDED, and the finding is negative and worth stating "
                "plainly: all four object metrics rank the DEGENERATE IDENTITY ENGINE above a "
                "real-but-bad engine on 3 of the 4 games where they can be measured. This is "
                "visible on real candidates, not only synthetic ones -- on tu93, object_match_iou "
                "scores candidates 0 and 1 (change_fidelity 0.0, i.e. nothing correct) at "
                "0.996162, ABOVE candidate 5 (change_fidelity 0.112292) at 0.995117. The cause "
                "is structural: a 64x64 ARC frame's connected-component partition is dominated "
                "by large static background objects that an inert engine reproduces perfectly, so "
                "the score is mostly about the part of the board the dynamics never touch. "
                "object_match_iou's entire dynamic range across the quality ladder is 0.0004 to "
                "0.011 depending on game. Object perception may still be the right LEVER -- this "
                "says nothing about the treatment -- but object-partition OVERLAP is the wrong "
                "instrument to measure it with."
            ),
            "vs_grid_agreement": (
                "Does not reward inertness outright, but identity TIES the worst rung of the "
                "quality ladder on tu93 (0.996454 both) and the whole ladder spans 0.0002 to "
                "0.0035. 99.6% of the number is background the engine never wrote. Usable as a "
                "diagnostic, hopeless as a primary."
            ),
            "vs_changed_cell_jaccard": (
                "Deliberately value-blind -- it asks WHERE the engine wrote, not WHAT. It "
                "therefore does not resolve a value-corruption ladder (2 of 4 games) and is "
                "correctly non-monotone there. Keep as the complementary channel that separates "
                "'knows where the action lands but not what it draws' from 'no idea'; it is not a "
                "substitute for the primary."
            ),
            "vs_spurious_changed_cells": (
                "Has the most per-game discordance in exp6018 (14 of 14) but is minimised "
                "trivially by an engine that writes nothing: identity scores 0, the best possible "
                "value, on every game. It fails H2 the hardest of any metric here."
            ),
        },
        "the_control_metric_did_not_floor_on_this_corpus_and_that_is_explained": (
            "Read on its own, this corpus would say exact-match is fine: it takes 8 distinct "
            "values over the 48 candidates and grades 4 of 5 games. That reading is wrong, and "
            "`noop_credit_decomposition` is why. Three of the five stall games have no-op-"
            "dominated held-out splits (ft09 16/20, sc25 13/14, lp85 18/18), and on every one of "
            "them `n_changes_correct` is 0 for every single candidate -- so all of exact-match's "
            "variation there is scored on transitions where the correct answer is 'nothing "
            "happens'. On the two change-dominated games it behaves exactly as exp6018 found: "
            "flat 0.0 across all candidates on tu93, and on tn36 it agrees with change_fidelity "
            "perfectly (Spearman 1.0) because with 17 of 17 changing and no no-ops the two "
            "measures coincide. The instrument floor is real; this corpus merely has three games "
            "where the instrument measures something else instead of flooring."
        ),
        "spearman_reading": (
            "The task's guard is that a graded metric which DISAGREES with exact-match where "
            "exact-match works is measuring something else. Applied naively here it would reject "
            "the right answer: change_fidelity's pooled within-game-centred Spearman against "
            "exact-match is only 0.19. But 3 of the 4 games where exact-match is non-degenerate "
            "are the no-op-credit games above, where exact-match is not working -- it is ranking "
            "inertness. On tn36, the ONLY game where exact-match is non-degenerate for a dynamics "
            "reason, change_fidelity's Spearman against it is exactly 1.0, as is cell_recall's, "
            "changed_cell_jaccard's and correct_changed_cells'. So the graded metrics are graded "
            "versions of exact-match wherever exact-match means what it claims to, and they "
            "diverge from it precisely where it does not."
        ),
        "noop_credit_decomposition": noop_decomp,
        "metrics_tried": metrics_tried,
        "n_metrics_tried": len(metrics_tried),
        "n_metrics_recommendable": sum(
            1 for m in metrics_tried if m["recommendable_as_ab_primary"]
        ),
        # ---- acceptance gates ---------------------------------------------------------------
        "acceptance_gate_at_least_one_metric_has_headroom": bool(
            sum(1 for m in metrics_tried if m["recommendable_as_ab_primary"]) >= 1
        ),
        "acceptance_gate_recommended_metric_separates_candidates": bool(
            rec["n_discordant_pairs_available"] > 0
        ),
        "acceptance_gate_split_provenance_verified": bool(
            all(
                r["matches_frozen_split_json"] and r["split_proven"]
                for r in raw["split_provenance"]
            )
        ),
        "acceptance_gate_reproduces_frozen_run_numbers": bool(agree["identical"]),
        "acceptance_gate_passed": bool(
            sum(1 for m in metrics_tried if m["recommendable_as_ab_primary"]) >= 1
            and rec["n_discordant_pairs_available"] > 0
            and all(
                r["matches_frozen_split_json"] and r["split_proven"]
                for r in raw["split_provenance"]
            )
            and agree["identical"]
        ),
        "is_this_gate_vacuous_honest_answer": (
            "PARTLY, and the part that is not pre-registered is named here rather than left for a "
            "reader to find. The gate COULD have failed: 'no metric has headroom' was an "
            "explicitly permitted -- and arguably more important -- outcome, and it is what would "
            "have been reported if change_fidelity had floored like exact-match did. Two of the "
            "four gates are independent of the metric analysis entirely (the split reproduces the "
            "one proven on 2026-07-31, prompt sha included; and 117 of 117 re-derived values match "
            "the frozen run's own recorded numbers), and either could have failed on its own -- "
            "indeed the shipped `_induce_transitions_k()` default moving that morning is exactly "
            "the kind of thing that breaks the first one. What is NOT pre-registered: the five "
            "H-criteria were fixed after the positive control had been read. H1-H4 are properties "
            "any A/B primary must have on any corpus, and H4's threshold is derived arithmetic "
            "(2 * 0.5**1 == 1.0), not a fitted number -- but H5's 0.5 bar was chosen knowing the "
            "ladder ranges. A reader who distrusts it should apply their own; the per-metric raw "
            "`control_min_dynamic_range` is reported for that purpose, and the surviving set is "
            "unchanged for any bar from 0.14 to 1.0."
        ),
        "corpus": {
            "source": "results/arc_induce_bestofn_20260731 (frozen 2026-07-31, not modified here)",
            "n_candidates_total": raw["n_candidates"],
            "n_candidates_on_stall_path": sum(1 for r in raw["rows"] if r["is_stall_game"]),
            "games": raw["games"],
            "stall_games": raw["stall_games"],
            "postbank_games_excluded_from_headline": sorted(
                set(raw["games"]) - set(raw["stall_games"])
            ),
            "why_stall_only": (
                "A post-bank induction at transition_count=1 passes both gates near-trivially and "
                "has 0 held-out rows, so including vc33 would flatter every metric. Same "
                "partition score_bon.py measured rather than assumed."
            ),
            "candidate_disposition": {
                "n_undetermined_worker_timeout": sum(
                    1 for r in raw["rows"] if str(r.get("status", "")).startswith("worker_timeout")
                ),
                "n_unrunnable_no_engine": sum(
                    1 for r in raw["rows"] if str(r.get("status", "")).startswith("unrunnable")
                ),
                "why_undetermined_is_not_a_zero": (
                    "ft09 candidate 5 produced an engine that does not terminate -- the same "
                    "candidate that wedged the generation loop for 13 minutes on 2026-07-31. "
                    "Nothing about it was measured, so it is None everywhere and leaves both "
                    "numerator and denominator. `unrunnable:*` is different: no engine exists, "
                    "which is a genuine zero on every similarity metric."
                ),
            },
        },
        "split_provenance": raw["split_provenance"],
        "agreement_with_frozen_run": agree,
        "induce_transitions_k_pinned": raw["induce_transitions_k_pinned"],
        "shipped_default_moved_under_this_frozen_corpus": (
            "`_induce_transitions_k()` returned 8 when these 48 candidates were generated "
            "(2026-07-31); commit 253e1b60ed changed its default to None ('show ALL transitions') "
            "on 2026-08-01, the day of this re-analysis. `split.py` derives the held-out set by "
            "replicating `changed[:k-2] + noop[:2]` with the CURRENT resolver, so an unpinned "
            "re-analysis grades against a split the frozen prompts never had. It happens to raise "
            "`TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'` rather than "
            "silently mis-splitting, but relying on that is luck. CARNOT_ARC_INDUCE_TRANSITIONS_K "
            "is pinned to 8 (the resolver's own docstring says 8 'restores the previous prompt "
            "byte-for-byte') and the derived split is then verified field-by-field against the "
            "frozen split.json, prompt sha included -- all 6 games match and all 3 of split.py's "
            "text-level proofs pass. Anyone re-running this MUST keep the pin."
        ),
        "exp6018_leak_prompt_control_is_the_decisive_independent_evidence": {
            "what_exp6018_did": (
                "raised k past the window so EVERY held-out transition was shown in the induction "
                "prompt, then graded on exactly the indices the production k=8 prompt withholds. "
                "The model was handed the answers."
            ),
            "n_cells": 28,
            "max_exact_match_accuracy": 0.0,
            "n_cells_with_nonzero_exact_match_accuracy": 0,
            "max_cell_recall": 0.442505,
            "n_cells_with_nonzero_cell_recall": 12,
            "exp6018_own_field_metric_demonstrably_moves": False,
            "why_this_settles_it": (
                "This is the objection-killer, and it is exp6018's own recorded number rather "
                "than anything this phase computed. The strongest case against the present "
                "recommendation is that on THIS 5-game corpus exact-match actually has MORE "
                "spread than change_fidelity -- 52 discordant within-game candidate pairs across "
                "4 gradable games against 39 across 3 -- so disqualifying it looks like a "
                "threshold chosen to reach a conclusion. But exp6018 ran the maximal possible "
                "treatment, showing the inducer the very transitions it would be graded on, and "
                "held-out exact-match was 0.0 in all 28 cells. An instrument that does not move "
                "when the answer is placed in the prompt cannot detect a representation change; "
                "cell_recall moved in 12 of the same 28 cells. The spread exact-match shows on "
                "this corpus is therefore not the instrument working -- the no-op decomposition "
                "identifies what it is instead."
            ),
        },
        "positive_control": {
            "what_it_is": (
                "Reference engines of CONSTRUCTED quality graded on the same proven held-out "
                "splits: identity, a perfect oracle, a 7-rung ladder of oracles with a fraction p "
                "of the truly-changed cells overwritten with a wrong in-palette colour, and "
                "oracles that additionally invent k writes on cells reality left alone. Candidate "
                "variance alone cannot tell resolution from noise; this can."
            ),
            "ran": True,
            "n_reference_arms_per_game": 12,
            "corruption_ladder": pc["corruption_ladder"],
            "spurious_write_counts": pc["spurious_write_counts"],
            "duration_s": pc["duration_s"],
            "recommended_metric_ladder_by_game": {
                g: v.get("values")
                for g, v in res[RECOMMENDED]["per_game"].items()
                if v.get("measurable")
            },
            "exact_match_ladder_by_game": {
                g: v.get("values")
                for g, v in res["exact_match_accuracy"]["per_game"].items()
                if v.get("measurable")
            },
            "headline": (
                "On the two no-op-dominated games the control metric drops from 1.0 to the "
                "IDENTITY engine's score the moment 5% of changed cells are corrupted and never "
                "moves again -- a 5%-wrong engine and a 100%-wrong engine are the same number. "
                "change_fidelity spans the full [0,1] monotonically over the same ladder on every "
                "game whose changed-cell sets are large enough to corrupt fractionally."
            ),
            "known_artifact_of_the_ladder": (
                "tn36's held-out transitions change roughly one cell each, so round(p * 1) is 0 "
                "for every p below 0.75 and the ladder cannot bite until then -- which is why "
                "tn36 shows 2 distinct values for EVERY metric including change_fidelity. That is "
                "a property of the game's tiny change sets, not of the metrics, and it is "
                "recorded rather than smoothed over."
            ),
        },
        "limitations": [
            "ONE ARM. This corpus was generated with CARNOT_ARC_OBJECT_PERCEPTION at its default "
            "(off), so for the metrics this phase ADDED (object_*, changed_cell_jaccard, "
            "grid_agreement_*) the `floored in BOTH arms` question cannot be answered from this "
            "data at all. It is answered only for the six channels exp6018 itself measured in two "
            "arms, and those fields are named `exp6018_*` to keep the provenance visible. The new "
            "metrics' single-arm spread plus the positive control is the evidence offered; it is "
            "not two-arm evidence and is not presented as such.",
            "FIVE GAMES, of which only three are gradable for change quality and only two are "
            "change-dominated. The 39 discordant within-game candidate pairs are a real count, "
            "not an estimate, but they come from 3 games. exp6018's 14-game roster is the right "
            "venue for the actual A/B and its recorded n_discordant of 12 of 14 for this metric "
            "is the number that should govern the design.",
            "NO CAUSAL CLAIM ABOUT OBJECT PERCEPTION. This phase measures instruments, not the "
            "treatment. That all four object-OVERLAP metrics reward inertness says nothing about "
            "whether feeding an object table to the inducer helps; it says those metrics cannot "
            "be used to find out.",
            "SC25 IS MEASURABLE BUT NOT GRADABLE. Its held-out split contains exactly one "
            "changing transition and every candidate gets it wrong, so change_fidelity is a "
            "constant 0.0 there. A roster for the A/B should require a floor on "
            "heldout_n_changing, not merely that it be non-zero.",
            "The recommended metric is NOT new. It is `WorldModelVerifier.change_fidelity`, "
            "already shipped, already computed by exp6018 as an exploratory secondary, and "
            "already recorded there as non-degenerate. The contribution of this phase is the "
            "evidence that it should be the PRIMARY, plus the elimination of the alternatives -- "
            "not the invention of an instrument.",
        ],
        "what_this_changes_for_the_next_ab": [
            "Pre-register change_fidelity as the PRIMARY and exact-match as a reported secondary, "
            "inverting exp6018's choice. exp6018's own recorded numbers say a 14-game paired sign "
            "test on this primary reaches a minimum two-sided p of 0.00048828, against 1.0 for "
            "the primary it actually used.",
            "Select the game roster on `heldout_n_changing`, not merely on the game existing. "
            "exp6018's own finding[0] already recorded that 6 of 20 games contribute ZERO held-out "
            "transitions; this phase adds that a game with 1 changing held-out row (sc25) or 0 "
            "(lp85) is equally useless for a change-quality primary while looking fine in a count "
            "of games.",
            "Report cell_recall and change_fidelity together, never cell_recall alone. They agree "
            "on the corruption ladder and disagree completely on invented writes, so a gap "
            "between them localises the failure to spurious writing.",
            "Do NOT use object-partition overlap as an outcome measure, in either arm. If the "
            "object hypothesis is to be tested at object granularity, the metric needs to be "
            "restricted to objects the dynamics actually touch -- scoring the whole partition "
            "hands the win to whichever arm produces more inert engines.",
        ],
        "cross_references": {
            "experiment_6018": "results/experiment_6018_object_perception_heldout_ab.json",
            "frozen_corpus": "results/arc_induce_bestofn_20260731/bestofn_scored.json",
            "proven_split": "results/arc_induce_bestofn_20260731/split.json",
            "shipped_metric_source": (
                "python/carnot/agentic/arc_executable_world_model.py:VerifyResult/"
                "WorldModelVerifier.score"
            ),
            "object_segmentation_source": (
                "python/carnot/agentic/arc_color_blob_salience.py:connected_color_blobs/object_hash"
            ),
            "subprocess_isolation_precedent": (
                "python/carnot/agentic/arc_engine_static_validation.py:dry_run_defects"
            ),
        },
        # ---- discipline fields -------------------------------------------------------------
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_note": (
            "No model is loaded and no token is generated. Every candidate's completion text was "
            "frozen on 2026-07-31; this run extracts the engine, executes it against RECORDED "
            "transitions in a killable subprocess, and computes metrics. That is exactly the "
            "`verifier_ensemble_against_cached_candidates` substrate: the verifier scored against "
            "pre-existing (input, candidate, label) triples, no LLM in the loop. The GGUF named in "
            "model_specs is the generator that PRODUCED the frozen candidates and is NOT invoked "
            "here."
        ),
        "model_specs": [
            {
                "name": "Qwen3.5-9B-MTP-GGUF",
                "role": "generator that produced the 48 frozen candidates on 2026-07-31",
                "invoked_by_this_experiment": False,
                "why_listed": (
                    "provenance of the candidates being re-scored, not a compute claim by this run"
                ),
            }
        ],
        "random_seed": 20260801,
        "random_seed_note": (
            "Used only by the positive control's corruption/spurious-write choice. The "
            "48-candidate census is fully deterministic: same frozen code, same recorded "
            "transitions, same arithmetic."
        ),
        "reproducibility_checksum": f"sha256:{checksum}",
        "input_file_sha256": input_sha,
        # Recomputed HERE rather than inherited from the census payload, so the artifact records
        # the sources as they are at artifact-build time. `harness_sha256_at_census_time` is the
        # census run's own copy; the two agreeing is the evidence that no harness source moved
        # between measuring and reporting.
        "harness_sha256": {q.name: sha256_file(q) for q in sorted(HERE.glob("*.py"))},
        "harness_sha256_at_census_time": raw["harness_sha256"],
        "measurement_sources_unchanged_since_census": all(
            sha256_file(HERE / k) == v
            for k, v in raw["harness_sha256"].items()
            if k in {"score_metrics.py", "metric_worker.py"}
        ),
        "preconditions_checked": [
            {
                "resource": "frozen_bestofn_corpus_present",
                "available": (BON / "bestofn_scored.json").exists(),
                "principle": (
                    "the whole phase is a re-analysis of a frozen corpus; without it there is "
                    "nothing to score and the honest verdict would be blocked_corpus_missing"
                ),
            },
            {
                "resource": "proven_split_reproduces_with_k_pinned_to_8",
                "available": all(
                    r["matches_frozen_split_json"] and r["split_proven"]
                    for r in raw["split_provenance"]
                ),
                "principle": (
                    "the shipped induce-prompt default moved on the day of this re-analysis; "
                    "grading against a split the frozen prompts never had would silently "
                    "invalidate every number, so the derived split is checked field-by-field "
                    "against the split proven on 2026-07-31, prompt sha included"
                ),
            },
            {
                "resource": "no_generated_code_executed_in_the_driver_interpreter",
                "available": True,
                "principle": (
                    "a non-terminating induced engine wedged a run for 13 minutes on 2026-07-31 "
                    "and an in-process alarm would be swallowed by the scoring loop's own "
                    "`except Exception`; only an external kill is a real bound"
                ),
            },
            {
                "resource": "cpu_only_no_gpu_claimed",
                "available": True,
                "principle": (
                    "GPU 1 was held by a concurrent workflow at 20726 MiB / 99% util for the "
                    "duration; this run touches neither card, so no GPU-backed claim is made and "
                    "none could be"
                ),
            },
            {
                "resource": "experiment_6018_artifact_present_for_two_arm_cross_reference",
                "available": EXP6018.exists(),
                "principle": (
                    "the `floored in BOTH arms` question can only be answered from a two-arm run; "
                    "without 6018 the answer for every metric would have to be 'unknown'"
                ),
            },
        ],
        "verifier_is_oracle": {
            "value": False,
            "principle": (
                "the graders here compare predicted next grids to RECORDED next grids; the level "
                "counter and the win oracle are never consulted, so nothing measured here can be "
                "circular with a solve"
            ),
        },
        "solve_provenance": {
            "value": "development_proxy",
            "principle": (
                "an offline instrument-calibration measurement on the dev twin. NO level is "
                "claimed, no game is solved, and nothing here is evidence that the live agent "
                "self-discovered anything"
            ),
        },
        "not_submitted": {
            "value": True,
            "principle": (
                "ARC/Kaggle submission is operator-only; this run plays no scored or online game "
                "and reaches no submission gate"
            ),
        },
        "flag_remains_default_off": {
            "value": True,
            "principle": (
                "CARNOT_ARC_OBJECT_PERCEPTION is untouched by this phase. Only the operator flips "
                "a default, and this phase measured instruments rather than the treatment, so it "
                "produced no evidence that could justify one"
            ),
        },
        "missing_verifier_gaps": [
            {
                "gap": (
                    "no held-out metric scores object correctness RESTRICTED to the objects the "
                    "dynamics touch"
                ),
                "failure_mode": (
                    "all four whole-partition object metrics rank the identity engine above a "
                    "real-but-bad engine on 3 of 4 measurable games, because a 64x64 ARC frame's "
                    "partition is dominated by static background an inert engine reproduces "
                    "perfectly"
                ),
                "missing_discriminator": (
                    "the sub-partition of objects that appeared, vanished, moved or recoloured "
                    "between g0 and g1 -- scored per changing object rather than over the whole "
                    "frame, so background cannot pay for the score"
                ),
                "candidate_design": (
                    "diff the g0 and g1 partitions by translation-invariant hash to get the "
                    "CHANGED object set, then score the prediction's best same-colour pixel-IoU "
                    "against that set only; identity would score 0 by construction, as it does on "
                    "change_fidelity"
                ),
                "priority": (
                    "medium -- change_fidelity already unblocks the A/B, so this is about getting "
                    "object-granular resolution, not about unblocking"
                ),
                "status": "open",
            }
        ],
        "duration_s": None,  # filled below
        "measurement_duration_s": {
            "candidate_census": raw["duration_s"],
            "positive_control": pc["duration_s"],
        },
    }
    art["duration_s"] = round(
        float(raw["duration_s"]) + float(pc["duration_s"]) + (time.monotonic() - t0), 3
    )

    out = OUT_DIR / "metric_headroom.json"
    out.write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"recommended: {RECOMMENDED}")
    print(f"n_discordant_pairs_available: {art['n_discordant_pairs_available']}")
    print(f"recommendable metrics: {art['n_metrics_recommendable']} of {art['n_metrics_tried']}")
    for m in metrics_tried:
        if m["recommendable_as_ab_primary"]:
            print(f"   PASS {m['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
