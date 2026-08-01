#!/usr/bin/env python3
"""Assemble the scored artifact from `analysis.json`. No compute, no generated code.

Everything numeric here is COPIED from `analysis.json`, which is itself derived from
`scored.json`. The prose is composed from those numbers rather than written alongside them, so
the headline cannot drift from the measurement it describes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR_ARTIFACT = HERE / "artifact.json"
OUT_TOPLEVEL = REPO / "results" / "outer_loop_arc_metric_validity_20260801.json"


def sha_of_inputs() -> str:
    """Content hash over every input and harness file this result depends on.

    Inputs first (the frozen engines and their scores), then the harness that read them. A
    checksum over only the outputs would not detect a changed input; a checksum over only the
    harness would not detect a changed corpus.
    """
    h = hashlib.sha256()
    for p in sorted(
        [
            HERE / "scored.json",
            HERE / "analysis.json",
            HERE / "run.py",
            HERE / "score_worker.py",
            HERE / "window_worker.py",
            HERE / "bon_window_worker.py",
            HERE / "analyse.py",
            HERE / "preregistration.json",
        ]
    ):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


def fmt(x, nd: int = 4):
    return None if x is None else round(float(x), nd)


def main() -> int:  # noqa: C901, PLR0915
    a = json.loads((HERE / "analysis.json").read_text())
    s = json.loads((HERE / "scored.json").read_text())
    pre = json.loads((HERE / "preregistration.json").read_text())

    cf = a["predictors"]["change_fidelity"]
    floor = a["power_floor_stated_before_the_association"]
    n = floor["n_analysable"]
    k = floor["n_plannable"]
    minp = floor["min_reachable_two_sided_p"]

    pooled = cf.get("auc_pooled")
    cl_ci = cf.get("cluster_bootstrap_ci95_game_resample")
    eng_ci = cf.get("bootstrap_ci95_engine_resample")
    wg = cf.get("within_game") or {}
    wg_p = (cf.get("within_game_perm_p") or {}).get("p")

    # --- which rival, if any, beats the primary --------------------------------------------
    # Predictors the analysis marked `definitionally_coupled_to_outcome` are EXCLUDED from this
    # ranking. The shipped goal gate is the same bounded BFS from the same root as plan_in_model
    # (they even share the depth resolver, deliberately), so `goal_satisfiable` does not predict
    # plannability -- it recomputes it. Calling it the best predictor would be the circularity the
    # Circularity / Oracle-Distinctness Discipline exists to catch.
    coupled = set(a.get("definitionally_coupled_predictors") or [])
    ranked = sorted(
        (
            (kk, vv)
            for kk, vv in a["predictors"].items()
            if vv.get("auc_pooled") is not None and kk != "change_fidelity" and kk not in coupled
        ),
        key=lambda kv: -abs(kv[1]["auc_pooled"] - 0.5),
    )
    best_key, best = ranked[0] if ranked else (None, None)

    # --- verdict composition, driven by the numbers -----------------------------------------
    # TWO independent ways this run can fail to answer its question, and both must be checked
    # before a chance-level result may be called a NULL (preregistration ADDENDUM 1):
    #   * the design's FLOOR -- the smallest two-sided p reachable at this n and k even in the
    #     single most extreme arrangement. A floor above 0.05 means no arrangement of these data
    #     could have been significant.
    #   * the design's POWER -- how often an injected association of a decision-relevant size is
    #     actually detected. The floor is a best-case statement and the power is an average-case
    #     one; a design can pass the first and fail the second, and then a chance-level result is
    #     absence of evidence rather than evidence of absence.
    pc = a.get("power_positive_control") or {}
    power75 = pc.get("power_at_auc_0.75")
    underpowered_by_floor = minp is not None and minp > 0.05
    underpowered_by_power = power75 is not None and float(power75) < 0.8
    underpowered = bool(underpowered_by_floor or underpowered_by_power)
    ci_excludes_half = cl_ci is not None and (min(cl_ci) > 0.5 or max(cl_ci) < 0.5)
    # A DETECTED association stands on its own: power governs how a NEGATIVE may be read, not
    # whether a positive that actually cleared its CI counts.
    #
    # DIRECTION IS NOT COSMETIC. The A/B that motivated this run treats change_fidelity as
    # higher-is-better, so the only result that would license flipping a lever on it is an
    # association in the POSITIVE direction. An AUC significantly BELOW 0.5 means higher fidelity
    # goes with LESS plannability -- the metric is informative and pointing the wrong way, which
    # for the decision at hand is a stronger negative than a null, not a positive. Collapsing the
    # two into "there is an association" would invert the recommendation.
    detected = bool(pooled is not None and ci_excludes_half and abs(pooled - 0.5) >= 0.15)
    predicts = bool(detected and pooled > 0.5)
    inverse = bool(detected and pooled < 0.5)

    if predicts:
        assoc_state = "PREDICTS"
    elif inverse:
        assoc_state = "INVERSE_HIGHER_FIDELITY_LESS_PLANNABLE"
    elif underpowered:
        assoc_state = "UNESTABLISHED_UNDERPOWERED"
    else:
        assoc_state = "NULL_DOES_NOT_PREDICT"

    dc = a["driver_check_tn36_removed"]["change_fidelity"]
    deg = a["degeneracy_audit_of_the_top_of_the_metric"]

    verdict_bits = [
        "complete_change_fidelity_metric_validity",
        {
            "PREDICTS": "PREDICTS_plannability",
            "INVERSE_HIGHER_FIDELITY_LESS_PLANNABLE": (
                "INVERSELY_associated_higher_fidelity_LESS_plannable"
            ),
            "NULL_DOES_NOT_PREDICT": "does_not_predict_plannability",
            "UNESTABLISHED_UNDERPOWERED": "unestablished_underpowered",
        }[assoc_state],
        f"auc_{pooled}",
        f"cluster_ci95_{cl_ci[0] if cl_ci else None}_{cl_ci[1] if cl_ci else None}",
        f"n_{n}_plannable_{k}",
        f"min_reachable_p_{minp}",
        f"within_game_auc_{wg.get('auc')}_over_{wg.get('n_informative_games')}_games",
        f"power_at_injected_auc075_{power75}",
    ]
    if best_key is not None:
        verdict_bits.append(f"best_rival_{best_key}_auc_{best['auc_pooled']}")
    honest_verdict = "_".join(str(b) for b in verdict_bits)

    headline = _headline(
        assoc_state,
        n,
        k,
        pooled,
        cl_ci,
        wg,
        minp,
        deg,
        best_key,
        best,
        dc,
        power75,
        _degenerate_summary(a),
    )

    artifact = {
        "schema": "carnot.arc.metric_validity.v1",
        "experiment": "outer_loop_arc_change_fidelity_metric_validity",
        "milestone": "2026.08.outer_loop",
        "run_date": _utc(),
        "question": pre["question"],
        "headline": headline,
        "honest_verdict": honest_verdict,
        "duration_s": round(float(s.get("duration_s") or 0.0), 2),
        "duration_s_note": (
            "wall time of the scoring sweep: one killable subprocess per engine, each running a "
            "held-out verifier pass, an in-sample pass, the shipped trust gate, and then a goal "
            "gate plus a 20000-node planner from each available root. No model is loaded."
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_note": (
            "No LLM is invoked and no GPU is touched (CUDA_VISIBLE_DEVICES is emptied in every "
            "worker). The engines are frozen LLM completions already on disk from two prior runs; "
            "this pass executes that cached code against cached transitions and inside a search. "
            "The GGUF strings in model_specs name the generator that WROTE those engines, with "
            "invoked=false -- they are provenance, not a live-inference claim."
        ),
        "random_seed": 20260801,
        "random_seed_note": (
            "seeds the permutation and bootstrap resampling only. The scoring itself is "
            "deterministic: the engines, the transitions and the search are all fixed."
        ),
        "reproducibility_checksum": sha_of_inputs(),
        "verifier_is_oracle": {
            "value": False,
            "principle": (
                "nothing here consults the environment's level counter or win oracle. The "
                "outcome is whether plan_in_model finds a path to the engine's OWN induced "
                "is_level_complete inside the engine's OWN rollout, and the exposure is "
                "prediction accuracy against RECORDED transitions. This measures whether one "
                "internal quantity predicts another; it makes no solve claim and cannot be "
                "circular with one."
            ),
        },
        "preconditions_checked": _preconditions(a, s),
        "model_specs": [
            {
                "name": "gemma-4-31B-it-qat-GGUF",
                "role": "wrote the 116 object-perception engines (frozen 2026-08-01)",
                "invoked": False,
                "why_not_invoked": "reads that run's committed engine text; generates nothing",
            },
            {
                "name": "gemma-4-31B-it (llama-server, CUDA)",
                "role": "wrote the 48 best-of-N candidates (frozen 2026-07-31)",
                "invoked": False,
                "why_not_invoked": "replays that run's cached completions; generates nothing",
            },
        ],
        "cited_upstream_artifacts": [
            {
                "experiment_id": (
                    "outer_loop_arc_object_perception_heldout_ab_change_fidelity_20260801"
                ),
                "fields_imported": "engine text (116 cells), rows.json roster",
                "path": "results/arc_object_perception_ab_change_fidelity_20260801/",
            },
            {
                "experiment_id": "outer_loop_arc_induce_bestofn_phase1",
                "fields_imported": (
                    "48 cached completions, the proven split, root_grid, "
                    "frozen plan_found at depth 40"
                ),
                "path": "results/arc_induce_bestofn_20260731/",
            },
        ],
        # ---- the answer -------------------------------------------------------------------
        "association_state": assoc_state,
        "change_fidelity_predicts_plannability": predicts,
        "change_fidelity_predicts_plannability_means": (
            "higher held-out change_fidelity goes with MORE plannable, which is the only "
            "direction that would license optimising it. False covers both a null and an INVERSE "
            "association; read association_state for which."
        ),
        "change_fidelity_inversely_associated": inverse,
        "primary": {
            "exposure": "held-out change_fidelity",
            "outcome": "plan_in_model finds a plan from the level root at the shipped defaults",
            "n_engines_scored": n,
            "n_plannable": k,
            "n_games": a["corpus"]["n_games"],
            "auc_pooled": pooled,
            "auc_pooled_note": (
                "anticonservative: engines cluster by game, so the effective n is nearer the "
                "game count than the engine count. Read the cluster CI, not this one."
            ),
            "perm_p_pooled": cf.get("perm_p_pooled"),
            "bootstrap_ci95_engine_resample": eng_ci,
            "cluster_bootstrap_ci95_game_resample": cl_ci,
            "cluster_ci_is_the_honest_one": True,
            "point_biserial": cf.get("point_biserial"),
            "logistic": cf.get("logistic"),
            "mean_change_fidelity_when_plannable": cf.get("mean_when_plannable"),
            "mean_change_fidelity_when_unplannable": cf.get("mean_when_unplannable"),
            "within_game_auc": wg.get("auc"),
            "within_game_n_informative_games": wg.get("n_informative_games"),
            "within_game_perm_p": wg_p,
            "within_game_is_the_decision_relevant_estimate": (
                "the decision this metric would be used for is: given several candidates induced "
                "for THE SAME game, pick one. Only games containing both a plannable and an "
                "unplannable engine can inform that."
            ),
        },
        "estimator_selftest": a["estimator_selftest"],
        "power_floor_stated_before_the_association": floor,
        "driver_check_tn36_removed": a["driver_check_tn36_removed"],
        "degeneracy_audit_of_the_top_of_the_metric": deg,
        "objperc_change_fidelity_reproduction_check": a[
            "objperc_change_fidelity_reproduction_check"
        ],
        "degenerate_goal_predicate_audit": _degenerate_summary(a),
        "inertness_floor": _inertness_summary(a),
        "root_substitution_check": a["root_substitution_check"],
        "depth_40_to_80_check": a["depth_40_to_80_check"],
        "rival_predictors": {
            "why_this_half_matters_more": pre["rival_predictors_tested_on_the_same_outcome"][
                "why_this_half_matters_more"
            ],
            "ranked_by_distance_from_chance": [
                {
                    "predictor": kk,
                    "what_it_is": vv["what_it_is"],
                    "auc_pooled": vv["auc_pooled"],
                    "cluster_bootstrap_ci95_game_resample": vv.get(
                        "cluster_bootstrap_ci95_game_resample"
                    ),
                    "within_game_auc": (vv.get("within_game") or {}).get("auc"),
                    "within_game_n_informative_games": (vv.get("within_game") or {}).get(
                        "n_informative_games"
                    ),
                    "perm_p_pooled": (vv.get("perm_p_pooled") or {}).get("p"),
                    "mean_when_plannable": vv.get("mean_when_plannable"),
                    "mean_when_unplannable": vv.get("mean_when_unplannable"),
                }
                for kk, vv in sorted(
                    a["predictors"].items(),
                    key=lambda kv: -abs((kv[1].get("auc_pooled") or 0.5) - 0.5),
                )
            ],
            "best_rival": best_key,
            "best_rival_auc": None if best is None else best["auc_pooled"],
            "MULTIPLICITY_ADJUSTED": a["family_multiplicity_check"],
            "read_the_adjusted_p_not_the_winners_own": (
                "the best rival was SELECTED as the furthest from chance among the whole family, "
                "so its own p-value is a selection effect. The family-wise max-statistic p above "
                "is the one to compare against 0.05."
            ),
        },
        # ---- false-negative-risk discipline ------------------------------------------------
        "false_negative_risk_checked": True,
        "positive_control_passed": _positive_control(a)["passed"],
        "positive_control": _positive_control(a),
        "preregistration_path": str(HERE / "preregistration.json"),
        "scored_detail_path": str(HERE / "scored.json"),
        "analysis_detail_path": str(HERE / "analysis.json"),
        "limitations": _limitations(a),
        "missing_verifier_gaps": _gaps(assoc_state, best_key, best, a["family_multiplicity_check"]),
        "surprising_result_acknowledgment": None,
        "acceptance_gates": {
            "metric_is_worth_scoring_on": {
                "condition": (
                    "pooled AUC materially above 0.5 with a game-clustered CI excluding 0.5, "
                    "surviving removal of tn36"
                ),
                "principle": (
                    "a metric that does not order engines by whether they can be planned with is "
                    "not a reason to flip a lever, however cleanly an A/B moves it."
                ),
                "met": predicts,
            },
            "corpus_can_answer_the_question": {
                "condition": (
                    "min reachable two-sided p <= 0.05 AND power >= 0.80 at an injected AUC of 0.75"
                ),
                "principle": (
                    "reporting a p above the design's own floor as 'trending' is how an "
                    "unanswerable question becomes a claim -- and a design that would rarely "
                    "detect a real moderate effect cannot report a chance-level result as a null."
                ),
                "met": bool(detected or not underpowered),
                "met_note": (
                    "a DETECTED association satisfies this gate regardless of the power figure: "
                    "power governs how a NEGATIVE may be read, not whether an association that "
                    "actually cleared its interval counts as an answer."
                ),
                "min_reachable_two_sided_p": minp,
                "underpowered_by_floor": underpowered_by_floor,
                "underpowered_by_power": underpowered_by_power,
                "power_at_injected_auc_0_75": power75,
            },
        },
        "acceptance_gate_passed": bool(detected or not underpowered),
        "acceptance_gate_passed_note": (
            "this gate is about whether the RUN could answer its question, not about whether the "
            "answer was the hoped-for one. A measured null passes it."
        ),
    }

    OUT_DIR_ARTIFACT.write_text(json.dumps(artifact, indent=2))
    OUT_TOPLEVEL.write_text(json.dumps(artifact, indent=2))
    print(honest_verdict)
    print()
    print(headline)
    return 0


def _utc() -> str:
    return subprocess.run(  # noqa: S603
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _degenerate_summary(a: dict) -> dict:
    """The outcome variable's own false positives, and the primary re-estimated without them.

    Found while reading the scored rows, not anticipated in the plan, and reported as a labelled
    post-hoc sensitivity rather than substituted for the preregistered primary.
    """
    dg = a["degenerate_goal_predicate_audit"]
    ranked = sorted(
        (
            (kk, vv)
            for kk, vv in dg["best_rival_re_estimated"].items()
            if vv.get("auc_pooled") is not None and kk != "change_fidelity"
        ),
        key=lambda kv: -abs(kv[1]["auc_pooled"] - 0.5),
    )
    prim = dg["primary_re_estimated_with_the_gate_applied_first"]
    return {
        "why": dg["why"],
        "post_hoc": True,
        "n_with_root_true_goal": dg["n_with_root_true_goal"],
        "n_of_those_counted_plannable_by_the_raw_planner": dg[
            "n_of_those_counted_plannable_by_the_raw_planner"
        ],
        "their_plan_lengths": dg["their_plan_lengths"],
        "n_plannable_before_gating": dg["n_plannable_before_gating"],
        "n_plannable_after_gating": dg["n_plannable_after_gating"],
        "min_reachable_two_sided_p_after_gating": dg["min_reachable_two_sided_p_after_gating"],
        "change_fidelity_after_gating": {
            kk: prim.get(kk)
            for kk in (
                "auc_pooled",
                "cluster_bootstrap_ci95_game_resample",
                "perm_p_pooled",
                "mean_when_plannable",
                "mean_when_unplannable",
                "n_plannable",
                "n_scored",
            )
        },
        "change_fidelity_within_game_after_gating": (prim.get("within_game") or {}).get("auc"),
        "best_rival_after_gating": ranked[0][0] if ranked else None,
        "best_rival_auc_after_gating": ranked[0][1]["auc_pooled"] if ranked else None,
        "ranked_after_gating": [
            {
                "predictor": kk,
                "auc_pooled": vv["auc_pooled"],
                "cluster_bootstrap_ci95_game_resample": vv.get(
                    "cluster_bootstrap_ci95_game_resample"
                ),
                "within_game_auc": (vv.get("within_game") or {}).get("auc"),
            }
            for kk, vv in ranked
        ],
        "an_independent_finding_worth_recording": (
            "plan_in_model tests is_level_complete on successors only, never on the start grid, "
            "while _goal_satisfiability_check tests the root explicitly and rejects it as "
            "goal_predicate_true_at_root. The two therefore DISAGREE on this class: the gate "
            "vetoes, the planner returns a length-1 plan. Production is safe because it runs the "
            "gate first -- but any caller that reaches plan_in_model without the gate would take "
            "a degenerate predicate as a solved level."
        ),
    }


def _inertness_summary(a: dict) -> dict:
    """The floor every structural predictor gets for free, and what survives it.

    An engine that changes nothing at the root cannot plan by construction, so any predictor
    that is really an inertness detector clears chance without predicting anything. The live-only
    ranking is the one that says whether a selection rule would have anything to select ON.
    """
    il = a["inertness_floor"]
    ranked = sorted(
        (
            (kk, vv)
            for kk, vv in il["predictors_within_live_engines_only"].items()
            if vv.get("auc_pooled") is not None
        ),
        key=lambda kv: -abs(kv[1]["auc_pooled"] - 0.5),
    )
    return {
        "why": il["why"],
        "n_inert_at_root": il["n_inert_at_root"],
        "n_inert_that_are_plannable": il["n_inert_that_are_plannable"],
        "n_live": il["n_live"],
        "n_live_plannable": il["n_live_plannable"],
        "min_reachable_two_sided_p_within_live": il["min_reachable_two_sided_p_within_live"],
        "change_fidelity_within_live": {
            kk: il["predictors_within_live_engines_only"]["change_fidelity"].get(kk)
            for kk in (
                "auc_pooled",
                "cluster_bootstrap_ci95_game_resample",
                "perm_p_pooled",
                "mean_when_plannable",
                "mean_when_unplannable",
            )
        },
        "ranked_within_live": [
            {
                "predictor": kk,
                "auc_pooled": vv["auc_pooled"],
                "cluster_bootstrap_ci95_game_resample": vv.get(
                    "cluster_bootstrap_ci95_game_resample"
                ),
                "within_game_auc": (vv.get("within_game") or {}).get("auc"),
            }
            for kk, vv in ranked
        ],
    }


def _positive_control(a: dict) -> dict:
    """A null on the primary is only informative if the design CAN detect a signal.

    TWO CONTROLS, and the PRIMARY one is the power analysis, deliberately. Pointing at a real
    rival predictor that happens to separate the classes is the weaker argument: if no rival had
    worked, the null would be uninterpretable, and whether a rival works is a fact about induced
    ARC engines rather than about whether this test can see. So the load-bearing control injects a
    synthetic predictor with a KNOWN association into the real labels, clustering and class
    balance, and reports how often the identical estimator detects it -- i.e. this design's
    empirical power. The real-rival evidence is reported alongside as corroboration.
    """
    pc = a.get("power_positive_control") or {}
    coupled = set(a.get("definitionally_coupled_predictors") or [])
    best = None
    for kk, vv in a["predictors"].items():
        if kk == "change_fidelity" or kk in coupled or vv.get("auc_pooled") is None:
            continue
        ci = vv.get("cluster_bootstrap_ci95_game_resample")
        sep = ci is not None and (min(ci) > 0.5 or max(ci) < 0.5)
        if sep and (best is None or abs(vv["auc_pooled"] - 0.5) > abs(best[1] - 0.5)):
            best = (kk, vv["auc_pooled"], ci, (vv.get("perm_p_pooled") or {}).get("p"))
    return {
        "primary_control_is_the_power_analysis": pc,
        "what_it_is": (
            "(1) an injected-signal power analysis on the real labels and clustering, and (2) the "
            "rival predictors run through the SAME AUC, permutation and game-clustered bootstrap "
            "code on the SAME plan_found outcome."
        ),
        "passed_requires": (
            "the power analysis detecting a moderate (AUC 0.75) injected association at least 80% "
            "of the time. The rival evidence corroborates but does not substitute for it."
        ),
        "definitionally_coupled_predictors_excluded": sorted(coupled),
        "why_they_are_excluded": (
            "the shipped goal gate runs the same bounded BFS from the same root as plan_in_model "
            "and shares its depth resolver by design, so goal_satisfiable separates the classes "
            "by construction. Using it as the positive control would prove only that the "
            "estimator can detect an identity."
        ),
        "why_it_licenses_reading_a_null": (
            "if a rival separates plannable from unplannable engines, the outcome varies, the "
            "corpus has contrast, and the estimator can detect a real signal -- so a chance-level "
            "result on change_fidelity is a fact about change_fidelity, not a dead test."
        ),
        "passed": bool(pc.get("passed")),
        "power_at_injected_auc_0_75": pc.get("power_at_auc_0.75"),
        "a_rival_also_separates": best is not None,
        "separating_predictor": None if best is None else best[0],
        "separating_predictor_auc": None if best is None else best[1],
        "separating_predictor_cluster_ci95": None if best is None else best[2],
        "separating_predictor_perm_p": None if best is None else best[3],
        "if_not_passed": (
            "the design lacks the power to detect a moderate association at this n and clustering, "
            "so a chance-level result on change_fidelity is ABSENCE OF EVIDENCE, not evidence of "
            "absence. Reported as such rather than as a null."
        ),
    }


def _headline(  # noqa: PLR0913
    state, n, k, pooled, cl_ci, wg, minp, deg, best_key, best, dc, power, dgs
) -> str:
    parts = []
    if state == "UNESTABLISHED_UNDERPOWERED":
        parts.append(
            f"UNESTABLISHED, not null. {k} of {n} scoreable engines are plannable; the AUC point "
            f"estimate is {pooled} with a game-clustered 95% CI of {cl_ci}, but this design does "
            f"not have the power to call a chance-level result a null "
            f"(min reachable two-sided p {minp}; detection rate {power} at an injected AUC of "
            f"0.75, against a 0.80 bar). This is absence of evidence, not evidence of absence."
        )
    elif state == "PREDICTS":
        parts.append(
            f"Held-out change_fidelity DOES order engines by plannability: AUC {pooled} over "
            f"{n} engines ({k} plannable), game-clustered 95% CI {cl_ci}."
        )
    elif state == "INVERSE_HIGHER_FIDELITY_LESS_PLANNABLE":
        parts.append(
            f"Held-out change_fidelity is associated with plannability THE WRONG WAY: AUC "
            f"{pooled} over {n} scoreable engines ({k} plannable), game-clustered 95% CI "
            f"{cl_ci}, entirely below chance. Higher fidelity goes with LESS plannability, so "
            f"optimising this metric is not neutral -- it selects against the property the "
            f"induced engine is needed for."
        )
    else:
        parts.append(
            f"Held-out change_fidelity does NOT predict plannability. AUC {pooled} over {n} "
            f"scoreable engines ({k} plannable), game-clustered 95% CI {cl_ci} -- an interval "
            f"that contains chance, in a design that detects an injected AUC-0.75 association "
            f"{power} of the time."
        )
    if wg.get("n_informative_games"):
        parts.append(
            f"Within-game (the decision-relevant comparison -- pick one of several candidates "
            f"for the SAME game) the AUC is {wg.get('auc')} over "
            f"{wg['n_informative_games']} informative games."
        )
    else:
        parts.append(
            "No game contains both a plannable and an unplannable engine, so there is NO "
            "within-game comparison at all: any pooled association would be entirely between "
            "games, i.e. it would measure which GAMES are plannable rather than which ENGINES."
        )
    parts.append(
        f"{deg['n_at_or_above_0999']} engines sit at change_fidelity >= 0.999 and "
        f"{deg['n_of_those_plannable']} of them are plannable."
    )
    parts.append(
        f"Removing tn36 leaves {dc['n_plannable']} plannable of {dc['n_scored']} and moves the "
        f"AUC to {dc['auc_pooled']}."
    )
    if dgs.get("n_of_those_counted_plannable_by_the_raw_planner"):
        parts.append(
            f"{dgs['n_of_those_counted_plannable_by_the_raw_planner']} of the plannable engines "
            f"have a goal predicate that is TRUE AT THE ROOT -- the shipped gate vetoes exactly "
            f"those and the live pipeline runs it before the planner, so applying the gate first "
            f"leaves {dgs['n_plannable_after_gating']} plannable and moves the AUC to "
            f"{dgs['change_fidelity_after_gating'].get('auc_pooled')}."
        )
    if best_key:
        parts.append(
            f"The strongest rival predictor on the same outcome is {best_key} at AUC "
            f"{best['auc_pooled']}."
        )
    return " ".join(parts)


def _limitations(a: dict) -> list[dict]:
    rs = a["root_substitution_check"]
    return [
        {
            "limitation": "plan_found depends on the induced GOAL PREDICATE, not the engine alone",
            "detail": (
                "plan_in_model searches for a state where the engine's own is_level_complete "
                "returns True. An engine with perfect dynamics and a broken goal predicate is "
                "unplannable, and an engine with sloppy dynamics and a loose predicate is "
                "plannable. So this measures whether change_fidelity predicts the plannability "
                "of the WHOLE induced model, which is the quantity the live agent consumes -- "
                "but it is not a clean test of the dynamics half in isolation."
            ),
        },
        {
            "limitation": "plannable is not the same as CORRECT",
            "detail": (
                "a plan found inside the model is a plan the model believes in. Nothing here "
                "executes it against a real environment, so no engine is shown to actually clear "
                "a level. Plannability is a NECESSARY condition for the induce->plan path to do "
                "anything at all, which is why it is the right outcome for a metric-validity "
                "question, but a metric that predicted plannability perfectly would still not be "
                "shown to predict solving."
            ),
        },
        {
            "limitation": "the object-perception corpus has no recorded root_grid",
            "detail": (
                "its plan root is reconstructed as the first grid of the level-up-straddling "
                "window. On the best-of-N corpus, where the real E3AgentPolicy.root_grid exists, "
                f"the two roots agree on plan_found for {rs['n_agree_on_plan_found']} of "
                f"{rs['n_engines_with_both_roots']} engines "
                f"(rate {rs['agreement_rate']}). That is the evidence for the substitution; it "
                "is not a proof that it holds on the 20 games where only the reconstruction "
                "exists."
            ),
        },
        {
            "limitation": "held-out sets are small and CHANGING-only",
            "detail": (
                "every roster game's held-out tail is changing rows plus a level-up row, so "
                "n_noop is 0 nearly everywhere and no-op hallucination cannot be graded. Engine "
                "behaviour on 'nothing should happen' -- half of what a forward rollout needs -- "
                "is unobserved, on both corpora."
            ),
        },
        {
            "limitation": "goal_energy was not supplied to the planner",
            "detail": (
                "matching the frozen best-of-N. The live stall path may install a best-first "
                "heuristic, which can only reach a goal in FEWER nodes, so every plan_found here "
                "is a lower bound on what the live planner would find. The direction of that bias "
                "is toward FEWER plannable engines, which makes the power floor worse and the "
                "association harder to detect -- it cannot manufacture one."
            ),
        },
    ]


def _gaps(state: str, best_key, best, fam: dict | None = None) -> list[dict]:
    gaps = []
    if state != "PREDICTS":
        gaps.append(
            {
                "gap": "no validated selection signal for induced world models",
                "failure_mode": (
                    "change_fidelity is the metric the current induction A/Bs are scored on, and "
                    "it does not order engines by whether the downstream planner can use them. "
                    "Every representation A/B scored on it inherits that."
                ),
                "missing_discriminator": (
                    "a scalar computable from an induced engine + its goal predicate + the "
                    "observed transitions that correlates with the engine being USABLE by "
                    "plan_in_model -- most plausibly a joint dynamics-and-goal quantity rather "
                    "than a pure dynamics-accuracy one, since plannability depends on both."
                ),
                "candidate_design": (
                    "score the pair, not the engine: reachability of the induced goal within the "
                    "shipped horizon, combined with the engine's state-graph branching, "
                    "penalised by disagreement with held-out changing transitions."
                ),
                "priority": "high -- it gates what any future induction A/B can be scored on",
                "status": "open",
            }
        )
    if best_key:
        gaps.append(
            {
                "gap": (
                    f"the best available plannability predictor ({best_key}) "
                    "is not wired into selection"
                ),
                "failure_mode": (
                    "the shipped trust gate selects on held-out dynamics accuracy. If a different "
                    f"quantity ({best_key}, AUC {best['auc_pooled']}) orders engines by "
                    "plannability better, the gate is selecting on the wrong axis."
                ),
                "missing_discriminator": (
                    "none -- this one is measurable today; what is missing is the wiring and a "
                    "prospective test that selecting on it raises banked levels rather than just "
                    "raising plannability."
                ),
                "do_not_wire_it_yet_and_here_is_why": (
                    "it was SELECTED as the furthest from chance among the whole family scored "
                    "here, so its margin is partly a selection effect. The family-wise "
                    "max-statistic p is "
                    f"{(fam or {}).get('fwer_adjusted_p_for_strongest')} over a family of "
                    f"{(fam or {}).get('family_size')}. Treat it as the best HYPOTHESIS this run "
                    "produces, to be confirmed prospectively on engines this run never saw -- not "
                    "as a validated selector."
                ),
                "candidate_design": (
                    "add it as a secondary in select_trusted_world_model and A/B the banked-level "
                    "rate, not the metric."
                ),
                "priority": "high",
                "status": "open",
            }
        )
    return gaps


def _preconditions(a: dict, s: dict) -> list[dict]:
    wr = s.get("window_rebuild") or {}
    op = wr.get("objperc") or {}
    bon = wr.get("bestofn") or {}
    n_op_ok = sum(1 for v in op.values() if v.get("status") == "ok")
    n_bon_ok = sum(1 for v in bon.values() if v.get("status") == "ok")
    n_bon_repro = sum(1 for v in bon.values() if v.get("reproduces_frozen_split"))
    return [
        {
            "resource": "frozen object-perception engines on disk",
            "available": True,
            "principle": (
                "this pass generates nothing; if the committed engine text were missing the run "
                "would be measuring nothing at all."
            ),
            "evidence": f"{a['corpus']['by_corpus'].get('objperc', 0)} objperc engines analysable",
        },
        {
            "resource": "frozen best-of-N completions + proven split + real root_grid",
            "available": True,
            "principle": (
                "the real root is the only thing that makes the reconstructed root on the other "
                "corpus checkable rather than assumed."
            ),
            "evidence": (
                f"{n_bon_ok} of {len(bon)} splits rebuilt, {n_bon_repro} of them reproducing "
                "split.json's row counts exactly"
            ),
        },
        {
            "resource": "no GPU used",
            "available": True,
            "principle": (
                "the machine is shared. Every worker empties CUDA_VISIBLE_DEVICES and sets "
                "JAX_PLATFORMS=cpu, so this run cannot contend for a card another session owns."
            ),
            "evidence": ("CUDA_VISIBLE_DEVICES='' set in run.py's worker env and in every worker"),
        },
        {
            "resource": "progress windows rebuildable",
            "available": n_op_ok > 0,
            "principle": (
                "build_progress_window steps a real env and has no internal bound; a game it "
                "cannot rebuild inside the timeout is DROPPED with its reason recorded, never "
                "scored as a zero."
            ),
            "evidence": f"{n_op_ok} of {len(op)} objperc windows rebuilt",
        },
        {
            "resource": "shipped search defaults unmodified",
            "available": True,
            "principle": (
                "widening a budget to manufacture plannable engines would inflate exactly the "
                "outcome being predicted."
            ),
            "evidence": (
                "plan_in_model and _goal_satisfiability_check are both called with no max_nodes "
                "or max_depth, so both read the shipped resolvers (max_depth 80, max_nodes 20000)"
            ),
        },
    ]


if __name__ == "__main__":
    raise SystemExit(main())
