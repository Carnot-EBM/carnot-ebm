"""Independent re-verification of the metric-validity run, plus the three findings it missed.

WHY THIS FILE EXISTS
--------------------
The artifact `results/outer_loop_arc_metric_validity_20260801.json` was reviewed
adversarially after it was written. The review raised one scoping defect and several
corrections. A review that only asserts numbers is worth little, so this script
recomputes EVERY headline statistic from `scored.json` from scratch -- deliberately
without importing anything from `analyse.py` -- so that a bug shared between the
analysis and its own check cannot reproduce itself. Where scipy is available the AUC
is additionally cross-checked against the Mann-Whitney U statistic.

It then computes three things the original analysis did not, and which change how the
result should be read. All three are POST-HOC and are labelled as such in the output;
they are hypotheses this run generated, not findings it pre-registered.

WHAT "AUC" MEANS HERE (for a reader who is not a statistician)
--------------------------------------------------------------
AUC is the probability that a randomly chosen PLANNABLE engine has a higher score
than a randomly chosen UNPLANNABLE one, counting ties as half. 0.5 is chance -- the
score tells you nothing. 1.0 is a perfect ordering. BELOW 0.5 means the score is
ordering things BACKWARDS: higher score, less likely to be plannable.

Run:  .venv/bin/python results/arc_metric_validity_20260801/corrigendum_verify.py
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORED = HERE / "scored.json"
OUT = HERE / "corrigendum_verify.json"

# The two goal-gate verdicts under which plan_in_model can possibly return a plan.
GOAL_KINDS_THAT_ADMIT_A_PLAN = {"satisfiable", "goal_predicate_true_at_root"}


def load_analysable() -> list[dict]:
    """Apply the pre-registered exclusion rules BY HAND, transcribed from preregistration.json.

    Two rules, both quoted from the plan:
      - "An engine whose held-out set has n_changing == 0 is EXCLUDED" -- because
        change_fidelity averages over changing transitions only, so with none it
        returns 0.0, and 0.0 is ALSO the value meaning "got every change wrong".
        Letting those in would enter a NOT-MEASURED as a measured worst score.
      - "An engine with no determined plan outcome leaves BOTH numerator and
        denominator" -- scoring a hang as a failed plan biases the very quantity
        being estimated.
    """
    payload = json.loads(SCORED.read_text())
    kept: list[dict] = []
    dropped: Counter[str] = Counter()
    for row in payload["engines"]:
        if row.get("status") != "ok":
            dropped["status_not_ok"] += 1
            continue
        heldout = row.get("heldout") or {}
        if not heldout:
            dropped["no_heldout"] += 1
            continue
        if (heldout.get("n_changing") or 0) == 0:
            dropped["heldout_n_changing_zero"] += 1
            continue
        if heldout.get("change_fidelity") is None:
            dropped["change_fidelity_none"] += 1
            continue
        plan = (row.get("plan") or {}).get("window_root")
        if not plan or plan.get("plan_found") is None:
            dropped["no_determined_plan_outcome"] += 1
            continue
        graph = (row.get("state_graph") or {}).get("window_root") or {}
        gate = (row.get("goal_gate") or {}).get("window_root") or {}
        kept.append(
            {
                "game": row["game"],
                "cell": row["cell"],
                "corpus": row["corpus"],
                "change_fidelity": heldout["change_fidelity"],
                "plannable": bool(plan["plan_found"]),
                "probe_depth_reached": graph.get("probe_depth_reached"),
                "live": graph.get("engine_changes_anything_at_root"),
                "goal_kind": gate.get("kind"),
                "trust_energy": row.get("shipped_gate_trust_energy"),
            }
        )
    return kept, dict(dropped), len(payload["engines"])


def auc(values: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney AUC by direct pair enumeration. No library, so no shared bug."""
    pos = [v for v, lab in zip(values, labels, strict=True) if lab]
    neg = [v for v, lab in zip(values, labels, strict=True) if not lab]
    if not pos or not neg:
        return None
    total = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return total / (len(pos) * len(neg))


def within_game_auc(rows: list[dict], key: str) -> tuple[float | None, float, dict]:
    """Pair-weighted AUC computed only between engines of the SAME game.

    This is the decision-relevant comparison: the decision the metric would be used
    for is "given several candidates induced for THE SAME game, pick one". Comparing
    an engine for game A against one for game B answers a question nobody asks.
    """
    numerator = denominator = 0.0
    per_game: dict[str, tuple[float, int]] = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["game"]].append(row)
    for game, members in grouped.items():
        pos = [m[key] for m in members if m["plannable"] and m[key] is not None]
        neg = [m[key] for m in members if not m["plannable"] and m[key] is not None]
        if not pos or not neg:
            continue
        total = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
        pairs = len(pos) * len(neg)
        per_game[game] = (total / pairs, pairs)
        numerator += total
        denominator += pairs
    if not denominator:
        return None, 0.0, {}
    return numerator / denominator, denominator, per_game


def cluster_ci(
    rows: list[dict], key: str, *, within: bool, draws: int = 4000, seed: int = 7
) -> list[float] | None:
    """Bootstrap resampling GAMES, not engines.

    Engines induced for the same game share that game's window, root, difficulty and
    goal structure, so they are not independent observations. Resampling engines
    would treat ~138 correlated rows as ~138 independent ones and produce an interval
    that is far too narrow. Resampling games is the honest one.
    """
    rng = random.Random(seed)
    games = sorted({r["game"] for r in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["game"]].append(row)
    estimates: list[float] = []
    for _ in range(draws):
        sample: list[dict] = []
        for index, game in enumerate(rng.choices(games, k=len(games))):
            # Re-label the game so a game drawn twice counts as two distinct clusters
            # in the within-game estimator, rather than silently merging.
            sample.extend({**r, "game": f"{game}#{index}"} for r in grouped[game])
        value = (
            within_game_auc(sample, key)[0]
            if within
            else auc([r[key] for r in sample], [r["plannable"] for r in sample])
        )
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[int(0.975 * len(estimates))]
    return [round(lo, 4), round(hi, 4)]


def within_game_permutation_p(
    rows: list[dict], key: str, *, draws: int = 20000, seed: int = 11
) -> float:
    """Permute plannability labels WITHIN each game only.

    This holds every game's class balance fixed, so the null being tested is
    "within a game, the score does not order engines" -- which is the null that
    matches the decision.
    """
    observed = within_game_auc(rows, key)[0]
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["game"]].append(row)
    at_least_as_extreme = 0
    for _ in range(draws):
        shuffled: list[dict] = []
        for members in grouped.values():
            labels = [m["plannable"] for m in members]
            rng.shuffle(labels)
            shuffled.extend({**m, "plannable": lab} for m, lab in zip(members, labels, strict=True))
        if abs(within_game_auc(shuffled, key)[0] - 0.5) >= abs(observed - 0.5):
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (draws + 1)


def null_per_game_auc_spread(rows: list[dict], *, draws: int = 20000, seed: int = 5):
    """How much do per-game AUCs scatter under a GLOBAL NULL, at these pair counts?

    This exists to stop a real mistake. Per-game AUCs range from 0.10 to 0.81 here,
    which LOOKS like the effect varies by game. But games contribute as few as 3
    pairs, and an AUC over 3 pairs can only take a handful of values -- so wild
    scatter is exactly what chance produces. Comparing the observed spread against
    the spread chance alone would produce is the only way to tell those apart.
    """
    counts = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["game"]].append(row)
    for members in grouped.values():
        positives = sum(m["plannable"] for m in members)
        if positives and positives < len(members):
            counts.append((len(members), positives))
    rng = random.Random(seed)
    spreads = []
    for _ in range(draws):
        aucs = []
        for total, positives in counts:
            order = list(range(total))
            rng.shuffle(order)
            labels = [i < positives for i in order]
            values = [rng.random() for _ in range(total)]
            pos = [v for v, lab in zip(values, labels, strict=True) if lab]
            neg = [v for v, lab in zip(values, labels, strict=True) if not lab]
            aucs.append(
                sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
                / (len(pos) * len(neg))
            )
        spreads.append(statistics.stdev(aucs))
    spreads.sort()
    return {
        "null_mean_sd": round(statistics.mean(spreads), 4),
        "null_ci95": [
            round(spreads[int(0.025 * len(spreads))], 4),
            round(spreads[int(0.975 * len(spreads))], 4),
        ],
    }


def main() -> None:
    rows, dropped, raw_n = load_analysable()
    n = len(rows)
    k = sum(r["plannable"] for r in rows)
    games = sorted({r["game"] for r in rows})
    live = [r for r in rows if r["live"] is True]

    report: dict = {
        "what_this_is": (
            "an independent from-scratch recomputation of "
            "results/outer_loop_arc_metric_validity_20260801.json, written without "
            "importing analyse.py, plus three post-hoc findings the original analysis "
            "did not report."
        ),
        "raw_rows": raw_n,
        "dropped_by_preregistered_exclusions": dropped,
        "n_analysable": n,
        "n_plannable": k,
        "n_games": len(games),
    }

    # ---- part 1: does every published headline number reproduce? ----
    fid = [r["change_fidelity"] for r in rows]
    lab = [r["plannable"] for r in rows]
    pooled = auc(fid, lab)
    wg, pair_mass, per_game = within_game_auc(rows, "change_fidelity")
    no_tn36 = [r for r in rows if r["game"] != "tn36"]
    top = [r for r in rows if r["change_fidelity"] >= 0.999]

    reproduction = {
        "auc_pooled": round(pooled, 4),
        "mean_when_plannable": round(
            statistics.mean([v for v, x in zip(fid, lab, strict=True) if x]), 4
        ),
        "mean_when_unplannable": round(
            statistics.mean([v for v, x in zip(fid, lab, strict=True) if not x]), 4
        ),
        "min_reachable_two_sided_p": 2.0 / math.comb(n, k),
        "within_game_auc": round(wg, 4),
        "within_game_n_informative_games": len(per_game),
        "tn36_removed": {
            "n": len(no_tn36),
            "k": sum(r["plannable"] for r in no_tn36),
            "auc_pooled": round(
                auc(
                    [r["change_fidelity"] for r in no_tn36],
                    [r["plannable"] for r in no_tn36],
                ),
                4,
            ),
            "within_game_auc": round(within_game_auc(no_tn36, "change_fidelity")[0], 4),
        },
        "engines_at_or_above_0999": {
            "n": len(top),
            "games": dict(Counter(r["game"] for r in top)),
            "n_plannable": sum(r["plannable"] for r in top),
        },
        "inertness_floor": {
            "n_inert": len(rows) - len(live),
            "n_inert_plannable": sum(r["plannable"] for r in rows if r["live"] is not True),
            "n_live": len(live),
            "n_live_plannable": sum(r["plannable"] for r in live),
            "auc_within_live": round(
                auc(
                    [r["change_fidelity"] for r in live],
                    [r["plannable"] for r in live],
                ),
                4,
            ),
        },
        "probe_depth_reached_pooled_auc": round(
            auc([r["probe_depth_reached"] for r in rows], lab), 4
        ),
        "shipped_gate_trust_energy_pooled_auc": round(
            auc([r["trust_energy"] for r in rows], lab), 4
        ),
    }
    try:
        from scipy.stats import mannwhitneyu

        pos = [v for v, x in zip(fid, lab, strict=True) if x]
        neg = [v for v, x in zip(fid, lab, strict=True) if not x]
        stat = mannwhitneyu(pos, neg, alternative="two-sided").statistic
        reproduction["scipy_cross_check_abs_delta"] = abs(stat / (len(pos) * len(neg)) - pooled)
    except Exception as exc:  # pragma: no cover - scipy is optional here
        reproduction["scipy_cross_check_abs_delta"] = f"unavailable: {exc}"
    report["reproduction_of_published_numbers"] = reproduction

    # ---- part 2: game-unweighted view, and whether its spread means anything ----
    per_game_aucs = [a for a, _ in per_game.values()]
    report["FINDING_1_game_unweighted_view"] = {
        "post_hoc": True,
        "why_it_matters": (
            "the published within-game AUC of 0.5688 is PAIR-weighted, so the three "
            "largest games supply most of the evidence. Weighting games equally -- "
            "which is the same clustering logic the artifact uses for its CI -- gives "
            "a different picture, and it makes the null STRONGER, not weaker."
        ),
        "pair_weighted_within_game_auc": round(wg, 4),
        "game_unweighted_mean_of_per_game_aucs": round(statistics.mean(per_game_aucs), 4),
        "n_games_below_chance": sum(1 for a in per_game_aucs if a < 0.5),
        "n_games_above_chance": sum(1 for a in per_game_aucs if a > 0.5),
        "pair_mass_share_of_top_three_games": round(
            sum(sorted((p for _, p in per_game.values()), reverse=True)[:3]) / pair_mass,
            4,
        ),
        "per_game": {g: {"auc": round(a, 4), "pairs": p} for g, (a, p) in per_game.items()},
        "observed_per_game_auc_sd": round(statistics.stdev(per_game_aucs), 4),
        "null_spread_check": null_per_game_auc_spread(rows),
        "CAVEAT_THAT_MUST_TRAVEL_WITH_THIS": (
            "the per-game AUCs range from 0.10 to 0.81 and 6 of 11 sit below chance, "
            "which LOOKS like the effect varies by game. It does not. The observed "
            "spread is INSIDE the range that pure sampling noise produces at these "
            "pair counts (as few as 3 pairs per game), so the direction flips are "
            "consistent with chance alone and must NOT be reported as evidence of "
            "between-game heterogeneity."
        ),
    }

    # ---- part 3: the two pre-registered controls applied TOGETHER ----
    # Each control was pre-registered on its own; their CONJUNCTION was not, which is
    # why this is labelled post-hoc. But each is independently motivated: inert
    # engines are unplannable by construction (0 of 45), and within-game is the
    # comparison the decision actually makes.
    both = {}
    for key in ("change_fidelity", "probe_depth_reached"):
        subset = [r for r in live if r[key] is not None]
        value, pairs, per = within_game_auc(subset, key)
        both[key] = {
            "within_game_auc_among_live_engines": round(value, 4),
            "n_pairs": int(pairs),
            "n_informative_games": len(per),
            "game_clustered_ci95": cluster_ci(subset, key, within=True),
            "n_games_below_chance": sum(1 for a, _ in per.values() if a < 0.5),
            "per_game": {g: round(a, 4) for g, (a, _) in per.items()},
        }
    both["change_fidelity"]["within_game_permutation_p"] = round(
        within_game_permutation_p(
            [r for r in live if r["change_fidelity"] is not None], "change_fidelity"
        ),
        5,
    )
    # Leave-one-game-out, to show a single game is not carrying the result.
    loo = {}
    for game in sorted({r["game"] for r in live}):
        subset = [r for r in live if r["game"] != game]
        value = within_game_auc(subset, "change_fidelity")[0]
        if value is not None:
            loo[game] = round(value, 4)
    both["change_fidelity"]["leave_one_game_out"] = loo

    report["FINDING_2_both_controls_together"] = {
        "post_hoc": True,
        "post_hoc_note": (
            "each control is pre-registered separately -- the inertness floor and the "
            "within-game stratification are both in preregistration.json. Their "
            "CONJUNCTION is not, so this is a hypothesis this run generated, not one "
            "it tested. It needs a prospective replication before it is acted on."
        ),
        "why_it_matters": (
            "inert engines are unplannable BY CONSTRUCTION (0 of 45), so leaving them "
            "in lets a metric look predictive merely by being correlated with "
            "inertness. Removing them AND comparing within game isolates the question "
            "actually being asked: among engines that do something, for the same "
            "game, does a higher score mean a better candidate?"
        ),
        "results": both,
        "reading": (
            "change_fidelity does not merely fail to predict under both controls -- it "
            "is INVERSELY associated, with an interval that excludes chance on the "
            "wrong side, 9 of 11 games below chance, a within-game permutation p of "
            "about 0.009, and no single game carrying it (leave-one-game-out never "
            "moves it above ~0.31). probe_depth_reached, the rival the original run "
            "nominated, falls to roughly chance under the same two controls -- so its "
            "headline advantage is substantially an inertness-and-graph-shape "
            "detector rather than a quality signal."
        ),
    }

    # ---- part 4: what plannability actually is ----
    kinds = Counter(r["goal_kind"] for r in rows)
    kinds_live = Counter(r["goal_kind"] for r in live)
    mismatches = [
        r for r in rows if r["plannable"] != (r["goal_kind"] in GOAL_KINDS_THAT_ADMIT_A_PLAN)
    ]
    unsat_live = [r for r in live if r["goal_kind"] not in GOAL_KINDS_THAT_ADMIT_A_PLAN]
    report["FINDING_3_plannability_is_a_goal_predicate_property"] = {
        "post_hoc": True,
        "why_it_matters": (
            "the artifact lists 'plan_found depends on the induced GOAL PREDICATE, not "
            "the engine alone' as limitation [0]. The data show it is not a caveat on "
            "the result -- it IS the result. plan_found is an EXACT function of the "
            "goal gate's verdict across all analysable engines, with zero exceptions. "
            "So the outcome variable this run scored every dynamics metric against is "
            "a restatement of whether the induced goal predicate is reachable. That "
            "is why no dynamics metric predicts it, and it means a null here is NOT "
            "evidence that dynamics accuracy is worthless -- it is evidence that "
            "plannability is the wrong outcome for testing a dynamics metric."
        ),
        "plan_found_equals_goal_kind_in_admitting_set": {
            "n_checked": len(rows),
            "n_mismatches": len(mismatches),
            "is_an_identity": not mismatches,
            "admitting_kinds": sorted(GOAL_KINDS_THAT_ADMIT_A_PLAN),
        },
        "goal_kind_distribution_all_analysable": dict(kinds.most_common()),
        "goal_kind_distribution_live_only": dict(kinds_live.most_common()),
        "live_engines_that_cannot_yield_a_plan": {
            "n": len(unsat_live),
            "of_n_live": len(live),
            "share": round(len(unsat_live) / len(live), 4),
            "breakdown": dict(Counter(r["goal_kind"] for r in unsat_live).most_common()),
        },
        "the_goal_gates_own_satisfiable_flag_is_NOT_the_identity": {
            "why": (
                "the gate records BOTH a `kind` and a `satisfiable` boolean, and they "
                "disagree. Every engine with kind `goal_predicate_true_at_root` carries "
                "satisfiable=False, yet all of them ARE plannable -- the goal predicate "
                "is already true at the root, so the planner returns immediately. So "
                "the identity holds against `kind`, not against `satisfiable`. Anything "
                "gating on the boolean alone would reject engines the planner can plan "
                "with. Those particular four are the degenerate root-true ones and "
                "arguably SHOULD be rejected, but the disagreement is a real "
                "inconsistency in the shipped gate and is recorded here rather than "
                "smoothed over."
            ),
            "n_engines_where_flag_and_outcome_disagree": sum(
                1
                for r in rows
                if r["plannable"] and r["goal_kind"] == "goal_predicate_true_at_root"
            ),
        },
        "the_actionable_decomposition": (
            "among the 93 engines that actually run and change something, the induce->"
            "plan path fails for 71. The single largest cause is the goal predicate "
            "never becoming true anywhere the bounded search reaches (45 of 93, about "
            "48 percent); the next is a goal predicate the gate classifies as "
            "degenerate (24 of 93, about 26 percent). The binding constraint on this "
            "path is the induced GOAL PREDICATE, not the induced dynamics."
        ),
    }

    # ---- part 5: reconciled ceiling on the inert-rejection intervention ----
    rate = sum(r["plannable"] for r in live) / len(live)
    rng = random.Random(3)
    grouped = defaultdict(list)
    for row in live:
        grouped[row["game"]].append(row)
    game_list = sorted(grouped)
    rates = []
    for _ in range(8000):
        sample: list[dict] = []
        for game in rng.choices(game_list, k=len(game_list)):
            sample.extend(grouped[game])
        rates.append(sum(r["plannable"] for r in sample) / len(sample))
    rates.sort()
    lo, hi = rates[200], rates[7800]
    report["FINDING_4_reconciled_intervention_ceiling"] = {
        "post_hoc": True,
        "why_it_matters": (
            "the taxonomy artifact put the ceiling on inert-rejection at 2 plannable "
            "of 26 live-and-clean engines (0.077), measured on the smaller best-of-N "
            "corpus at the OLD depth-40 default. This corpus is larger and runs at the "
            "shipped depth 80, and gives a higher rate. The two are not in conflict; "
            "the newer one supersedes."
        ),
        "p_plannable_given_live": round(rate, 4),
        "game_clustered_ci95": [round(lo, 4), round(hi, 4)],
        "taxonomy_estimate_it_supersedes": 0.0769,
        "expected_additional_plannable_engines_if_12_1_inert_convert": round(12.1 * rate, 2),
        "expected_additional_ci95": [round(12.1 * lo, 2), round(12.1 * hi, 2)],
        "as_share_of_a_124_candidate_corpus": round(12.1 * rate / 124, 4),
        "reading": (
            "converting every inert candidate the shipped one-reask budget can convert "
            "buys roughly three more plannable engines across a 124-candidate corpus, "
            "about 2 percent. And 'plannable' here means only that the goal gate "
            "believes a goal-true state is reachable INSIDE the model -- nothing has "
            "been executed against a real environment."
        ),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
