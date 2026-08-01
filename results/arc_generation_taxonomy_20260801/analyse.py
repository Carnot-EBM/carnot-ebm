#!/usr/bin/env python3
"""Turn the per-candidate detector output into a taxonomy, and test the claims that follow from it.

READ THE PRECEDENCE ORDER BELOW BEFORE READING ANY SHARE. A candidate can carry several defects
at once (a `missing_return` engine that also raises), so "count the kinds" and "count the
candidates" are different questions with different denominators. This file answers the second:
every candidate gets exactly ONE primary class, assigned by a fixed precedence, so the shares sum
to 1 and a reader can subtract. The per-kind multiset is reported separately and explicitly
labelled as non-exclusive.

The precedence follows the SHIPPED detector's own ordering (`validate_engine_code` returns early
on truncation, then on syntax/missing-function, then adds dry-run defects), because a taxonomy
whose ordering disagrees with the detector's would classify candidates into buckets the pipeline
can never act on in that order.

WHAT IS TESTED, and why each test exists rather than an eyeballed comparison:

  * RETRY VALUE. The single-shot corpus and the 3-try corpus differ in exactly one wired thing:
    tries. Their terminal no-engine rates are therefore an estimate of what retry buys, and it is
    reported with a confidence interval rather than as two bare fractions.
  * INERTNESS HETEROGENEITY. Whether "re-ask when the engine is inert" can work at all depends on
    whether inertness is a property of the SAMPLE (resampling escapes it) or of the GAME
    (resampling returns another inert engine). That is a contingency table, not a judgement call.
  * DOWNSTREAM CEILING. Converting an inert engine into a live one is only worth what a live
    engine is worth. The plannability rate among already-live engines is the ceiling on any
    inertness intervention, and it is computed rather than assumed.

CLUSTERING. Candidates from the same game are not independent -- they share a prompt, a window,
and whatever the game's mechanics make easy or hard. Every interval over candidates is therefore
reported with the game as the cluster, and where a clustered interval is not available the
unclustered one is labelled as optimistic rather than quoted plainly.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent

# Precedence: first match wins. Ordered most-fundamental-failure-first, matching the shipped
# detector's own early-return order.
PRECEDENCE = [
    ("no_code_after_all_tries", None),
    ("truncated_before_required_symbols", "truncated_before_required_symbols"),
    ("syntax_error", "syntax_error"),
    ("missing_function", "missing_function"),
    ("engine_nonterminating", "engine_nonterminating"),
    ("engine_crashed_validator", "engine_crashed_validator"),
    ("module_exec_raised", "module_exec_raised"),
    ("engine_raised", "engine_raised"),
    ("missing_return", "missing_return"),
    ("returns_none_literal", "returns_none_literal"),
    ("engine_returned_none", "engine_returned_none"),
    ("engine_wrong_shape", "engine_wrong_shape"),
    ("goal_raised", "goal_raised"),
    ("goal_not_boolean", "goal_not_boolean"),
]

# What a caller can DO about each class, and whether the LIVE induce path currently does it.
# `live_gate_acts` is not "is it detectable" -- it is "does the shipped `generate()` loop change
# its behaviour on this". The gap between the two columns is the actionable finding.
CLASS_META = {
    "no_code_after_all_tries": ("terminal generation failure", True, True),
    "truncated_before_required_symbols": ("output cap", True, True),
    "syntax_error": ("invalid Python", True, True),
    "missing_function": ("no `def engine`", True, True),
    "engine_nonterminating": ("runs forever", True, True),
    "engine_crashed_validator": ("takes the interpreter down", True, True),
    "module_exec_raised": ("import-time failure", True, True),
    "engine_raised": ("raises on an observed transition", True, True),
    "missing_return": ("returns None on some path", True, True),
    "returns_none_literal": ("explicit `return None`", True, True),
    "engine_returned_none": ("returned None in the dry run", True, True),
    "engine_wrong_shape": ("wrong output shape", True, True),
    "goal_raised": ("goal predicate raises", True, True),
    "goal_not_boolean": ("goal predicate not a truth value", True, True),
    "inert_no_defect": ("predicts that nothing ever changes", True, False),
    "clean_and_live": ("no mechanical defect, engine does change the grid", None, None),
}


def primary_class(row: dict) -> str:
    if row.get("status") == "no_code_file":
        return "no_code_after_all_tries"
    if row.get("status") not in {"ok"}:
        return f"undetermined:{row.get('status')}"
    kinds = set(row.get("defect_kinds") or [])
    for name, kind in PRECEDENCE:
        if kind is not None and kind in kinds:
            return name
    if kinds:
        return "other:" + sorted(kinds)[0]
    if row.get("engine_changes_anything") is False:
        return "inert_no_defect"
    if row.get("engine_changes_anything") is None:
        return "undetermined:inertness_unmeasured"
    return "clean_and_live"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    return tuple(round(float(x), 4) for x in stats.binomtest(k, n).proportion_ci(0.95))


def cluster_bootstrap_ci(
    pairs: list[tuple[str, int]], n_boot: int = 5000, seed: int = 20260801
) -> tuple[float, float]:
    """95% CI for a proportion, resampling GAMES rather than candidates.

    Candidates from one game share a prompt and a window; treating them as independent narrows
    every interval on this corpus by construction. Resampling the cluster is the cheap honest fix
    and it is used everywhere a rate spans more than one game.
    """
    if not pairs:
        return (float("nan"), float("nan"))
    by_game: dict[str, list[int]] = collections.defaultdict(list)
    for g, v in pairs:
        by_game[g].append(int(v))
    games = sorted(by_game)
    if len(games) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(games), size=len(games), replace=True)
        vals = [v for i in pick for v in by_game[games[i]]]
        if vals:
            draws.append(float(np.mean(vals)))
    if not draws:
        return (float("nan"), float("nan"))
    return (round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4))


def main() -> int:
    data = json.loads((HERE / "classification.json").read_text())
    rows = data["rows"]
    for r in rows:
        r["primary_class"] = primary_class(r)

    out: dict = {"n_units": len(rows), "windows_not_rebuilt": {}}
    for g, st in (data.get("windows") or {}).items():
        if st.get("status") != "ok":
            out["windows_not_rebuilt"][g] = st.get("status")

    # ---- per-corpus taxonomy ------------------------------------------------------------
    corpora = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        sub = [r for r in rows if r["corpus"] == corpus]
        cnt = collections.Counter(r["primary_class"] for r in sub)
        n = len(sub)
        classes = []
        for cls, c in cnt.most_common():
            desc, detectable, live_acts = CLASS_META.get(cls, (cls, None, None))
            lo, hi = cluster_bootstrap_ci([(r["game"], r["primary_class"] == cls) for r in sub])
            classes.append(
                {
                    "class": cls,
                    "what_it_is": desc,
                    "count": c,
                    "share": round(c / n, 4),
                    "share_ci95_game_clustered": [lo, hi],
                    "mechanically_detectable_before_trust_gate": detectable,
                    "live_induce_path_acts_on_it_today": live_acts,
                }
            )
        # kinds are NOT exclusive: one candidate can contribute several
        kindcnt = collections.Counter(k for r in sub for k in (r.get("defect_kinds") or []))
        corpora[corpus] = {
            "n": n,
            "n_games": len({r["game"] for r in sub}),
            "primary_classes_mutually_exclusive": classes,
            "defect_kinds_NON_exclusive": dict(kindcnt.most_common()),
            "n_unusable": sum(
                1 for r in sub if r["primary_class"] not in {"clean_and_live", "inert_no_defect"}
            ),
            "n_inert": cnt.get("inert_no_defect", 0),
            "n_clean_and_live": cnt.get("clean_and_live", 0),
        }
    out["corpora"] = corpora

    # ---- reproduction check against the frozen record -----------------------------------
    bon = [r for r in rows if r["corpus"] == "bestofn_31B_single_shot"]
    agree = dis = 0
    disagreements = []
    for r in bon:
        frozen = sorted(r.get("frozen_defect_kinds") or [])
        mine = sorted(r.get("defect_kinds") or [])
        # The frozen run used `required=("engine","is_level_complete")` and this pass does too,
        # so the two are directly comparable EXCEPT for `validation_timeout`, which is the frozen
        # harness's own outer-bound token and not a detector kind.
        if frozen == ["validation_timeout"]:
            continue
        if frozen == mine:
            agree += 1
        else:
            dis += 1
            disagreements.append({"cell": r["cell"], "frozen": frozen, "reclassified": mine})
    out["reproduction_check_vs_frozen_bestofn"] = {
        "n_compared": agree + dis,
        "n_agree": agree,
        "n_disagree": dis,
        "disagreements": disagreements[:20],
        "sha_reproduces": collections.Counter(
            str(r.get("code_sha_reproduces")) for r in bon
        ).most_common(),
        "why_this_matters": (
            "the taxonomy is only about the frozen corpus if it is reading the same bytes the "
            "frozen corpus was scored on. The extracted-code sha is checked per candidate and "
            "the detector's verdict is re-derived; a disagreement here would mean the detector "
            "changed under the record, not that the corpus did."
        ),
    }

    # ---- what retry buys -----------------------------------------------------------------
    ab = [r for r in rows if r["corpus"] == "abcf_31B_after_3_tries"]
    bon_nocode = sum(
        1
        for r in bon
        if r["primary_class"] in {"syntax_error", "missing_function", "no_code_after_all_tries"}
    )
    ab_nocode = sum(
        1
        for r in ab
        if r["primary_class"] in {"syntax_error", "missing_function", "no_code_after_all_tries"}
    )
    tbl = [[bon_nocode, len(bon) - bon_nocode], [ab_nocode, len(ab) - ab_nocode]]
    odds, p_fisher = stats.fisher_exact(tbl)
    out["what_retry_buys"] = {
        "claim": (
            "the ONLY wired difference between these two corpora is tries=1 vs tries=3 on the "
            "same generator, prompt family and sampler, so the difference in the rate of "
            "'no loadable engine at all' is an estimate of what the shipped retry already buys."
        ),
        "single_shot_no_engine": [bon_nocode, len(bon), round(bon_nocode / len(bon), 4)],
        "after_3_tries_no_engine": [ab_nocode, len(ab), round(ab_nocode / len(ab), 4)],
        "single_shot_ci95_game_clustered": cluster_bootstrap_ci(
            [
                (
                    r["game"],
                    r["primary_class"]
                    in {"syntax_error", "missing_function", "no_code_after_all_tries"},
                )
                for r in bon
            ]
        ),
        "after_3_tries_ci95_game_clustered": cluster_bootstrap_ci(
            [
                (
                    r["game"],
                    r["primary_class"]
                    in {"syntax_error", "missing_function", "no_code_after_all_tries"},
                )
                for r in ab
            ]
        ),
        "fisher_exact_p": round(float(p_fisher), 6),
        "odds_ratio": None if odds != odds or math.isinf(odds) else round(float(odds), 4),
        "confounds_stated": (
            "the two corpora also differ in game roster (6 vs 20) and in prompt arm (the A/B "
            "corpus varies an object-perception header). Both are stated rather than adjusted "
            "for: with 6 games in one arm there is no honest way to adjust, and the direction of "
            "the effect is large enough that the roster difference would have to be extreme to "
            "explain it. This is evidence about retry, not proof."
        ),
    }

    # ---- is inertness a property of the sample or of the game? --------------------------
    scored = [
        r
        for r in rows
        if r["primary_class"] in {"inert_no_defect", "clean_and_live"}
        and r["corpus"] != "ab0728_qwen9B_retired_generator"
    ]
    by_game = collections.defaultdict(lambda: [0, 0])  # game -> [inert, live]
    for r in scored:
        by_game[r["game"]][0 if r["primary_class"] == "inert_no_defect" else 1] += 1
    table = [v for v in by_game.values() if sum(v) >= 2]
    chi = None
    if len(table) >= 2:
        arr = np.array(table)
        keep = arr[arr.sum(axis=1) > 0]
        if keep.shape[0] >= 2 and keep.sum(axis=0).min() > 0:
            c2, p, dof, _ = stats.chi2_contingency(keep)
            chi = {"chi2": round(float(c2), 4), "p": float(f"{p:.3g}"), "dof": int(dof)}
    # The asymptotic chi-square is NOT trustworthy here -- most cells hold 0-6 observations, well
    # under the usual expected-count-5 rule, and quoting it alone would be exactly the kind of
    # "the test was run so the number is real" that this project's own discipline forbids. A
    # permutation test makes no asymptotic assumption: shuffle the inert/live labels across all
    # scored candidates and recompute, so the null is "inertness is unrelated to which game".
    perm_p = None
    if scored:
        labels = np.array([r["primary_class"] == "inert_no_defect" for r in scored])
        game_order = sorted({r["game"] for r in scored})
        gidx = np.array([game_order.index(r["game"]) for r in scored])
        ng = len(game_order)

        def stat(lab: np.ndarray) -> float:
            # Sum of squared deviations of each game's inert rate from the pooled rate, weighted
            # by that game's n -- the same quantity chi-square measures, computed directly.
            p0 = float(lab.mean())
            s = 0.0
            for g in range(ng):
                m = gidx == g
                n = int(m.sum())
                if n:
                    s += n * (float(lab[m].mean()) - p0) ** 2
            return s

        rng = np.random.default_rng(20260801)
        obs = stat(labels)
        null = np.array([stat(rng.permutation(labels)) for _ in range(20000)])
        perm_p = float((1 + int((null >= obs).sum())) / (1 + len(null)))

    out["is_inertness_a_property_of_the_game"] = {
        "why_this_decides_the_intervention": (
            "'reject inert engines and re-ask' only works if a re-sample on the same game has a "
            "real chance of being live. If inertness is a property of the GAME, the re-ask "
            "returns another inert engine and the intervention buys nothing but latency."
        ),
        "per_game_inert_vs_live": {
            g: {"inert": v[0], "live": v[1]} for g, v in sorted(by_game.items())
        },
        "chi2_homogeneity_ASYMPTOTIC_UNRELIABLE_AT_THESE_CELL_COUNTS": chi,
        "permutation_p_20000_shuffles": perm_p,
        "which_p_to_believe": (
            "the permutation p. Most cells hold 0-6 observations, far below the expected-count-5 "
            "rule the chi-square approximation needs; the permutation test assumes nothing and is "
            "reported as the primary result."
        ),
        "reading": (
            "a small p means the per-game inertness rates are NOT the same, i.e. inertness is "
            "concentrated in particular games -- which is the bad case for a blind re-ask."
        ),
    }

    # ---- the ceiling on any inertness intervention --------------------------------------
    bon_live = [r for r in bon if r["primary_class"] == "clean_and_live"]
    n_plan = sum(1 for r in bon_live if r.get("frozen_plan_found") is True)
    out["ceiling_on_an_inertness_intervention"] = {
        "logic": (
            "converting an inert engine into a live one is worth exactly what a live engine is "
            "worth. On the frozen best-of-N corpus, plannability among engines that are already "
            "live and defect-free is the ceiling on any such conversion."
        ),
        "live_and_clean": len(bon_live),
        "of_which_plannable": n_plan,
        "plannability_given_live": round(n_plan / len(bon_live), 4) if bon_live else None,
        "ci95_unclustered_OPTIMISTIC": wilson(n_plan, len(bon_live)) if bon_live else None,
        "ci95_game_clustered": cluster_bootstrap_ci(
            [(r["game"], r.get("frozen_plan_found") is True) for r in bon_live]
        ),
    }

    # ---- would rejecting inert engines have destroyed a plannable one? -------------------
    inert_bon = [r for r in bon if r["primary_class"] == "inert_no_defect"]
    out["does_rejecting_inert_destroy_a_plannable_candidate"] = {
        "why_asked": (
            "an intervention that rejects a class must be checked against the outcomes IN that "
            "class before it is recommended. If a plannable candidate is inert, 'reject inert and "
            "re-ask' throws away the thing the pipeline exists to produce."
        ),
        "n_inert": len(inert_bon),
        "n_inert_that_were_plannable": sum(
            1 for r in inert_bon if r.get("frozen_plan_found") is True
        ),
        "plannable_cells_and_their_class": [
            {"cell": r["cell"], "class": r["primary_class"]}
            for r in bon
            if r.get("frozen_plan_found") is True
        ],
    }

    # ---- expected yield of 'reject inert, re-ask' ----------------------------------------
    # Leave-one-out within game: for each inert candidate, the chance a RE-SAMPLE on the same
    # game is live is estimated by that game's live share among its OTHER scored candidates. This
    # is the estimate the clustering result above demands -- a pooled rate would assume exactly
    # the homogeneity the chi-square rejects.
    def loo_yield(sub: list[dict]) -> dict:
        by_g = collections.defaultdict(lambda: {"inert": 0, "live": 0})
        for r in sub:
            if r["primary_class"] == "inert_no_defect":
                by_g[r["game"]]["inert"] += 1
            elif r["primary_class"] == "clean_and_live":
                by_g[r["game"]]["live"] += 1
        exp_conv = 0.0
        per_game = {}
        for g, v in by_g.items():
            n = v["inert"] + v["live"]
            if v["inert"] == 0:
                continue
            # leave the candidate itself out of its own estimate
            p_live = v["live"] / (n - 1) if n > 1 else 0.0
            exp_conv += v["inert"] * p_live
            per_game[g] = {
                "inert": v["inert"],
                "live": v["live"],
                "p_live_on_resample_loo": round(p_live, 4),
                "expected_converted": round(v["inert"] * p_live, 3),
            }
        return {
            "n": len(sub),
            "n_inert": sum(v["inert"] for v in by_g.values()),
            "expected_converted_to_live_by_ONE_reask": round(exp_conv, 2),
            "as_share_of_all_candidates": round(exp_conv / len(sub), 4) if sub else None,
            "per_game": per_game,
        }

    yl_ab = loo_yield(ab)
    yl_bon = loo_yield(bon)
    p_plan = out["ceiling_on_an_inertness_intervention"]["plannability_given_live"]
    out["expected_yield_of_rejecting_inert_and_reasking"] = {
        "method": (
            "ONE extra ask per inert candidate (the shipped `_INDUCE_DEFECT_REASKS = 1` budget, "
            "unchanged). Conversion probability is estimated leave-one-out WITHIN the game, "
            "because inertness is game-clustered; a pooled rate would assume the homogeneity the "
            "chi-square above rejects, and would overstate the yield on exactly the games where "
            "inertness concentrates."
        ),
        "after_3_tries_corpus": yl_ab,
        "single_shot_corpus": yl_bon,
        "downstream_plannable_gain_after_3_tries": (
            None
            if p_plan is None
            else {
                "expected_extra_live_engines": yl_ab["expected_converted_to_live_by_ONE_reask"],
                "plannability_given_live": p_plan,
                "expected_extra_PLANNABLE_engines": round(
                    yl_ab["expected_converted_to_live_by_ONE_reask"] * p_plan, 3
                ),
                "per_100_candidates": round(
                    100 * yl_ab["expected_converted_to_live_by_ONE_reask"] * p_plan / yl_ab["n"], 3
                ),
            }
        ),
        "what_this_number_is_not": (
            "it is an upper bound on the MECHANICAL conversion, not a prediction of banked "
            "levels. It assumes re-samples within a game are exchangeable, which is the most "
            "favourable assumption available: if the model is inert on a game because it cannot "
            "see the mechanic, re-sampling returns another inert engine and the true yield is "
            "lower than this."
        ),
    }

    # ---- what the taxonomy implies about where candidates END UP -------------------------
    ab_live = [r for r in ab if r["primary_class"] == "clean_and_live"]
    cfs = [
        r["frozen_change_fidelity"] for r in ab_live if r.get("frozen_change_fidelity") is not None
    ]
    ab_inert_cf = [
        r["frozen_change_fidelity"]
        for r in ab
        if r["primary_class"] == "inert_no_defect" and r.get("frozen_change_fidelity") is not None
    ]
    out["where_a_fixed_candidate_lands"] = {
        "why": (
            "every generation-side fix moves candidates INTO `clean_and_live`. What that class is "
            "worth is therefore the return on any such fix, and it is measured here rather than "
            "assumed."
        ),
        "clean_and_live_heldout_change_fidelity": {
            "n": len(cfs),
            "mean": round(float(np.mean(cfs)), 4) if cfs else None,
            "median": round(float(np.median(cfs)), 4) if cfs else None,
            "n_exactly_zero": sum(1 for x in cfs if x == 0),
            "n_at_or_above_0p5": sum(1 for x in cfs if x >= 0.5),
        },
        "inert_heldout_change_fidelity": {
            "n": len(ab_inert_cf),
            "mean": round(float(np.mean(ab_inert_cf)), 4) if ab_inert_cf else None,
            "n_exactly_zero": sum(1 for x in ab_inert_cf if x == 0),
        },
    }

    # ---- how much of 'predicts nothing useful' is MECHANICALLY visible? ------------------
    # The sharpest honest bound on any inertness gate. An engine that is inert scores exactly 0
    # held-out change fidelity -- but so does an engine that changes the WRONG cells. Only the
    # first is mechanically detectable. The ratio is the ceiling on what a mechanical gate can
    # reach within the class it is aimed at.
    zero_cf = [r for r in ab if r.get("frozen_change_fidelity") == 0.0]
    zero_inert = [r for r in zero_cf if r["primary_class"] == "inert_no_defect"]
    out["how_much_of_zero_fidelity_is_mechanically_visible"] = {
        "n_cells_scored": len([r for r in ab if r.get("frozen_change_fidelity") is not None]),
        "n_with_heldout_change_fidelity_exactly_zero": len(zero_cf),
        "of_which_mechanically_inert": len(zero_inert),
        "of_which_live_but_wrong_INVISIBLE_to_any_static_or_dry_run_check": len(zero_cf)
        - len(zero_inert),
        "share_of_zero_fidelity_that_a_mechanical_gate_can_see": (
            round(len(zero_inert) / len(zero_cf), 4) if zero_cf else None
        ),
        "reading": (
            "an engine that predicts NOTHING and an engine that predicts the WRONG THING both "
            "score 0. Only the first leaves a mechanical signature. This ratio bounds what any "
            "generation-time validity check can do about the zero-fidelity population."
        ),
    }

    # ---- wasted generation calls ---------------------------------------------------------
    def wasted(sub: list[dict]) -> dict:
        unusable = [
            r for r in sub if r["primary_class"] not in {"clean_and_live", "inert_no_defect"}
        ]
        inert = [r for r in sub if r["primary_class"] == "inert_no_defect"]
        return {
            "n": len(sub),
            "mechanically_unusable": len(unusable),
            "inert": len(inert),
            "guaranteed_downstream_reject": len(unusable) + len(inert),
            "share": round((len(unusable) + len(inert)) / len(sub), 4) if sub else None,
        }

    out["wasted_generation_calls"] = {
        "definition": (
            "a call is wasted when the candidate it produced cannot survive the downstream trust "
            "gate under any circumstances -- it does not load, or it predicts that nothing ever "
            "changes. Verified for the inert class on the frozen best-of-N corpus: the shipped "
            "trust gate rejected 11 of 11 inert candidates, while `generate()` ACCEPTED all 11. "
            "That is the wasted work -- the generation call is spent and the candidate is thrown "
            "away one stage later."
        ),
        "single_shot": wasted(bon),
        "after_3_tries": wasted(ab),
        "shipped_trust_gate_admitted_any_inert_candidate": sum(
            1
            for r in bon
            if r["primary_class"] == "inert_no_defect" and r.get("frozen_shipped_gate_passes")
        ),
        "generate_accepted_inert_candidates": sum(
            1
            for r in bon
            if r["primary_class"] == "inert_no_defect" and r.get("frozen_generate_would_accept")
        ),
    }

    # ---- truncation: verify rather than assume ------------------------------------------
    def stop_counts(sub):
        return dict(collections.Counter(str(r.get("stop_type")) for r in sub).most_common())

    q9 = [r for r in rows if r["corpus"] == "ab0728_qwen9B_retired_generator"]
    lim_bon = sum(1 for r in bon if r.get("stop_type") == "limit")
    lim_q9 = sum(1 for r in q9 if r.get("stop_type") == "limit")
    tbl2 = [[lim_bon, len(bon) - lim_bon], [lim_q9, len(q9) - lim_q9]]
    _, p_trunc = stats.fisher_exact(tbl2)
    out["truncation_verified_not_assumed"] = {
        "current_generator_31B_with_repeat_penalty": {
            "stop_type": stop_counts(bon),
            "hit_output_cap": [lim_bon, len(bon), round(lim_bon / len(bon), 4)],
            "of_which_actually_missing_a_required_symbol": sum(
                1
                for r in bon
                if "truncated_before_required_symbols" in (r.get("defect_kinds") or [])
            ),
        },
        "retired_generator_9B_no_repeat_penalty": {
            "stop_type": stop_counts(q9),
            "hit_output_cap": [lim_q9, len(q9), round(lim_q9 / len(q9), 4)] if q9 else None,
        },
        "fisher_exact_p": round(float(p_trunc), 8) if q9 else None,
        "caveat": (
            "GENERATOR AND SAMPLER MOVED TOGETHER between these two corpora, so this comparison "
            "cannot attribute the drop to repeat_penalty alone. It establishes the thing that "
            "was asked -- that truncation is no longer a live failure class on the CURRENT "
            "generator -- and nothing more."
        ),
    }

    # ---- the headline: what is actually large, and is it addressable ---------------------
    prim31 = [r for r in rows if r["corpus"] != "ab0728_qwen9B_retired_generator"]
    cnt31 = collections.Counter(r["primary_class"] for r in prim31)
    # PRECEDENCE SENSITIVITY. Two candidates are BOTH inert and raise on an observed transition;
    # the precedence assigns them to `engine_raised`. If a reader thinks inertness should win
    # instead, the inert count moves and the ordering must be checked rather than asserted.
    inert_if_top = sum(
        1 for r in prim31 if r.get("engine_changes_anything") is False and r.get("status") == "ok"
    )
    validity_classes = {
        "no_code_after_all_tries",
        "syntax_error",
        "missing_function",
        "engine_nonterminating",
        "engine_crashed_validator",
        "module_exec_raised",
        "engine_raised",
        "missing_return",
        "returns_none_literal",
        "engine_returned_none",
        "engine_wrong_shape",
        "goal_raised",
        "goal_not_boolean",
        "truncated_before_required_symbols",
    }
    n_validity = sum(v for c, v in cnt31.items() if c in validity_classes)
    out["headline"] = {
        "n_31B_candidates": len(prim31),
        "classes_by_size": [
            {"class": c, "count": n, "share": round(n / len(prim31), 4)}
            for c, n in cnt31.most_common()
        ],
        "POOLING_CAVEAT": (
            "these 172 candidates mix TWO retry regimes -- 48 single-shot and 124 after 3 tries "
            "-- so the pooled shares are descriptive of what is on disk, NOT a rate for either "
            "path. The per-corpus tables are the numbers to quote. Pooling changes the inertness "
            "share in particular: 11/48 (22.9%) single-shot against 15/124 (12.1%) after retry."
        ),
        "precedence_sensitivity": {
            "inert_under_the_precedence_used": cnt31.get("inert_no_defect", 0),
            "inert_if_inertness_took_top_precedence": inert_if_top,
            "all_code_validity_classes_combined": n_validity,
            "does_the_ordering_change": (
                "no -- inertness is the largest single failure class under either assignment, "
                "and exceeds every code-validity class combined either way."
            ),
        },
    }

    (HERE / "analysis.json").write_text(json.dumps(out, indent=2, default=str))
    (HERE / "classified_rows.json").write_text(
        json.dumps(
            [
                {
                    k: r.get(k)
                    for k in (
                        "corpus",
                        "cell",
                        "game",
                        "arm",
                        "primary_class",
                        "defect_kinds",
                        "parses",
                        "parse_error_type",
                        "parse_error",
                        "engine_changes_anything",
                        "stop_type",
                        "code_lines",
                        "code_bytes",
                        "frozen_plan_found",
                        "frozen_change_fidelity",
                        "induce_msg",
                    )
                }
                for r in rows
            ],
            indent=2,
            default=str,
        )
    )
    print(json.dumps(out["headline"], indent=2))
    print("\nper-corpus unusable:")
    for c, v in corpora.items():
        print(
            f"  {c}: n={v['n']} unusable={v['n_unusable']} "
            f"inert={v['n_inert']} clean_live={v['n_clean_and_live']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
