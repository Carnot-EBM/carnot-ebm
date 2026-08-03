"""Analysis: per-arm distributions and paired arm-vs-base contrasts, CLUSTERED AT THE GAME.

CLUSTERING (failure mode 8). Replicates within a game share a window, a split and a prompt; they
are not independent draws. Every replicate of a (game, arm) is averaged into ONE per-game mean
BEFORE any test or interval. 20 games x 3 replicates is 20 units, not 60. Treating replicates as
trials inflated a p from 0.125 to 0.049 elsewhere in this project and had to be corrected; a
rule-of-three computed at engine level was up to 12x overstated.

PAIRING (failure mode 6 -- missing is never zero). A (game, replicate) contributes to an
arm-vs-base contrast only if BOTH that arm and base produced a MEASURABLE score in that replicate.
An unmatched cell is counted and named, never averaged into one side. A cell whose engine failed
to load, whose worker timed out, or whose induce never completed is MISSING: excluded from the
metric aggregates and reported in its own ledger. It is separately counted in the induce_ok /
usable-engine rates, where "the arm produced nothing" is the honest outcome rather than a gap.

NO CELL IS SILENTLY CENSORED (failure mode 9). Every queued cell lands in exactly one of:
scored, missing-with-a-reason, or not-run-with-a-reason. The three counts plus the queue size are
reported and must reconcile.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CELLS = OUT / "cells"
SCORED = OUT / "scored"

ARM_ORDER = ["base", "think", "antiid", "delta"]
CONTROL = "base"
HIDDEN_STATE = {
    "ar25",
    "cd82",
    "cn04",
    "dc22",
    "g50t",
    "ka59",
    "m0r0",
    "re86",
    "sc25",
    "sk48",
    "wa30",
}
# `shown_train` is TRAINING accuracy on the very rows the prompt contained. It is NEVER a
# generalization number and is labelled as such everywhere it appears. It is measured because it
# separates two completely different diagnoses that a held-out zero cannot tell apart: "the model
# induced a rule that does not generalize" versus "the model cannot even reproduce the transitions
# it was literally shown". Only the first is a generalization problem.
BLOCKS = [
    "tail",
    "fresh",
    "shown_train",
    # STRATIFIED held-out blocks. `*_substantive` = rows where reality changed >= 2 cells;
    # `*_trivial_1cell` = rows where exactly one cell moved (a HUD/counter tick). Added
    # after bp35__r1__antiid cleared the brief's target at change_accuracy 0.5662 on 219
    # clean rows, and decomposition showed all 124 of its exact matches were one-cell
    # counter ticks while every multi-cell row was wrong. Both strata are reported with
    # their denominators; nothing is censored.
    "tail_substantive",
    "fresh_substantive",
    "tail_trivial_1cell",
    "fresh_trivial_1cell",
]
FIELDS = ["change_accuracy", "cell_recall", "change_fidelity", "accuracy"]
SEED = 20260802


def _binom_tail_ge(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / float(2**n)


def min_reachable_two_sided_p(n_disc: int) -> float:
    if n_disc <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**n_disc))


def sign_test(deltas: list[float]) -> dict:
    """Exact two-sided sign test. Ties are dropped and REPORTED: a run whose every pair ties has
    had NO TEST, and that must be legible as such rather than as a p of 1.0 meaning 'no effect'."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    d = pos + neg
    if d == 0:
        return {
            "n_pairs": len(deltas),
            "n_positive": 0,
            "n_negative": 0,
            "n_ties": ties,
            "n_discordant": 0,
            "p_two_sided": None,
            "test_was_possible": False,
            "reading": "every pair tied -- NO TEST WAS POSSIBLE. This is not 'no significant "
            "difference'; it is 'the metric did not move in either arm'.",
            "min_reachable_two_sided_p_at_this_discordance": 1.0,
        }
    p = min(1.0, 2.0 * _binom_tail_ge(max(pos, neg), d))
    return {
        "n_pairs": len(deltas),
        "n_positive": pos,
        "n_negative": neg,
        "n_ties": ties,
        "n_discordant": d,
        "p_two_sided": round(p, 8),
        "test_was_possible": True,
        "min_reachable_two_sided_p_at_this_discordance": round(min_reachable_two_sided_p(d), 8),
    }


def bootstrap_ci(values: list[float], *, seed: int = SEED, n: int = 10000) -> dict:
    """Percentile bootstrap over the UNIT OF INDEPENDENCE the caller passes in -- per-GAME means,
    never per-cell rows."""
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return {
        "mean": round(sum(values) / k, 6),
        "lo": round(means[int(0.025 * n)], 6),
        "hi": round(means[min(n - 1, int(0.975 * n))], 6),
        "n": k,
        "n_resamples": n,
    }


def quantiles(xs: list[float]) -> dict:
    """min/q1/median/q3/max. Failure mode 7: one game is not the roster."""
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)

    def q(f):
        i = f * (n - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)

    return {
        "n": n,
        "min": round(s[0], 6),
        "q1": round(q(0.25), 6),
        "median": round(q(0.5), 6),
        "q3": round(q(0.75), 6),
        "max": round(s[-1], 6),
        "mean": round(sum(s) / n, 6),
    }


def load() -> tuple[list[dict], dict]:
    cells = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*.json"))]
    scored = {p.stem: json.loads(p.read_text()) for p in sorted(SCORED.glob("*.json"))}
    return cells, scored


def main() -> int:
    cells, scored = load()
    prep = json.loads((OUT / "prep_meta.json").read_text())
    res: dict = {}

    # ---- LEDGER: every cell lands in exactly one bucket -----------------------------------
    ledger = {"n_cells_run": len(cells), "n_scored_ok": 0, "missing": [], "not_scored": []}
    merged: list[dict] = []
    for c in cells:
        cid = c["cell_id"]
        s = scored.get(cid)
        rec = dict(c)
        if not c.get("induce_ok"):
            ledger["missing"].append(
                {
                    "cell_id": cid,
                    "arm": c["arm"],
                    "game": c["game"],
                    "reason": c.get("missing_reason") or "induce_failed",
                    "induce_msg": c.get("induce_msg", "")[:120],
                }
            )
            rec["scored"] = None
        elif s is None:
            ledger["not_scored"].append(
                {"cell_id": cid, "arm": c["arm"], "reason": "no_score_file"}
            )
            rec["scored"] = None
        elif s.get("status") != "ok":
            ledger["missing"].append(
                {
                    "cell_id": cid,
                    "arm": c["arm"],
                    "game": c["game"],
                    "reason": s.get("status"),
                    "error": str(s.get("error"))[:120],
                }
            )
            rec["scored"] = None
        else:
            ledger["n_scored_ok"] += 1
            rec["scored"] = s
        merged.append(rec)
    ledger["n_missing"] = len(ledger["missing"])
    ledger["n_not_scored"] = len(ledger["not_scored"])
    ledger["reconciles"] = ledger["n_scored_ok"] + ledger["n_missing"] + ledger[
        "n_not_scored"
    ] == len(cells)
    ledger["missing_reason_counts"] = dict(Counter(m["reason"] for m in ledger["missing"]))
    ledger["missing_by_arm"] = dict(Counter(m["arm"] for m in ledger["missing"]))
    res["cell_ledger"] = ledger

    # ---- ARM INTEGRITY: was the arm REACHED, per cell, on the wire? -----------------------
    integrity = {}
    for arm in ARM_ORDER:
        rs = [c for c in cells if c["arm"] == arm]
        integrity[arm] = {
            "n_cells": len(rs),
            "n_with_wire_calls": sum(1 for c in rs if (c.get("n_wire_calls") or 0) > 0),
            "n_directive_consistent": sum(
                1 for c in rs if c.get("arm_directive_consistent") is True
            ),
            "n_directive_inconsistent": sum(
                1 for c in rs if c.get("arm_directive_consistent") is False
            ),
            "n_marker_consistent": sum(1 for c in rs if c.get("arm_marker_consistent") is True),
            "n_marker_inconsistent": sum(1 for c in rs if c.get("arm_marker_consistent") is False),
            "n_predict_values_on_wire": sorted(
                {v for c in rs for v in (c.get("wire_n_predict") or [])}
            ),
        }
        integrity[arm]["arm_was_delivered_on_every_cell"] = bool(
            rs
            and integrity[arm]["n_directive_consistent"] == len(rs)
            and integrity[arm]["n_marker_inconsistent"] == 0
        )
    res["arm_integrity_on_the_wire"] = integrity
    res["arm_integrity_note"] = (
        "Read off the LITERAL payload POSTed to /completion (urlopen wrapped), not off the "
        "configuration that was supposed to produce it. A fix shipped elsewhere in this project "
        "was delivered 0 of 128 times while its call site looked correct."
    )

    # ---- PER-CELL VALUES ------------------------------------------------------------------
    def val(rec, block, field):
        s = rec.get("scored")
        if not s:
            return None
        b = s.get(block) or {}
        if not b.get("measurable"):
            return None
        return b.get(field)

    # ---- PER-GAME MEANS -------------------------------------------------------------------
    def per_game(block, field, arm):
        acc: dict[str, list[float]] = {}
        for r in merged:
            if r["arm"] != arm:
                continue
            v = val(r, block, field)
            if v is None:
                continue
            acc.setdefault(r["game"], []).append(float(v))
        return {g: sum(v) / len(v) for g, v in acc.items()}, acc

    # ---- DISTRIBUTIONS + the >0 counts ----------------------------------------------------
    dists: dict = {}
    for block in BLOCKS:
        dists[block] = {}
        for field in FIELDS:
            dists[block][field] = {}
            for arm in ARM_ORDER:
                pg, raw = per_game(block, field, arm)
                cellvals = [v for vs in raw.values() for v in vs]
                dists[block][field][arm] = {
                    "per_game_mean_quantiles": quantiles(list(pg.values())),
                    "per_cell_quantiles": quantiles(cellvals),
                    "n_cells": len(cellvals),
                    "n_games": len(pg),
                    "n_cells_above_zero": sum(1 for v in cellvals if v > 0.0),
                    "n_games_with_any_cell_above_zero": sum(
                        1 for vs in raw.values() if any(v > 0.0 for v in vs)
                    ),
                    "n_cells_at_or_above_0.5": sum(1 for v in cellvals if v >= 0.5),
                    "n_games_with_any_cell_at_or_above_0.5": sum(
                        1 for vs in raw.values() if any(v >= 0.5 for v in vs)
                    ),
                    "max_cell": max(cellvals) if cellvals else None,
                    "all_cells_exactly_zero": bool(cellvals) and all(v == 0.0 for v in cellvals),
                    "per_game_means": {g: round(v, 6) for g, v in sorted(pg.items())},
                }
    res["distributions"] = dists

    # ---- PAIRED CONTRASTS (matched by game AND replicate) ---------------------------------
    def paired(block, field, arm, games_filter=None):
        by = {}
        for r in merged:
            v = val(r, block, field)
            if v is None:
                continue
            by[(r["game"], r["replicate"], r["arm"])] = float(v)
        games = sorted({g for (g, _, _) in by})
        if games_filter is not None:
            games = [g for g in games if g in games_filter]
        deltas, means, unmatched = [], {}, []
        for g in games:
            reps = sorted({rp for (gg, rp, _) in by if gg == g})
            a_vals, b_vals = [], []
            for rp in reps:
                a = by.get((g, rp, arm))
                b = by.get((g, rp, CONTROL))
                if a is None or b is None:
                    if a is not None or b is not None:
                        unmatched.append(f"{g}__r{rp}__{arm if a is not None else CONTROL}")
                    continue
                a_vals.append(a)
                b_vals.append(b)
            if not a_vals:
                continue
            ma, mb = sum(a_vals) / len(a_vals), sum(b_vals) / len(b_vals)
            deltas.append(ma - mb)
            means[g] = {
                "arm": round(ma, 6),
                CONTROL: round(mb, 6),
                "delta": round(ma - mb, 6),
                "n_matched_replicates": len(a_vals),
            }
        return deltas, means, unmatched

    plain_ex_tn36 = sorted({c["game"] for c in cells} - HIDDEN_STATE - {"tn36"})
    strata = {
        "PRIMARY_plain_branch_ex_tn36": plain_ex_tn36,
        "all_games_ex_tn36": sorted({c["game"] for c in cells} - {"tn36"}),
        "hidden_state_stratum": sorted({c["game"] for c in cells} & HIDDEN_STATE),
    }
    contrasts: dict = {}
    for sname, gset in strata.items():
        contrasts[sname] = {}
        for block in BLOCKS:
            contrasts[sname][block] = {}
            for field in FIELDS:
                contrasts[sname][block][field] = {}
                for arm in ARM_ORDER:
                    if arm == CONTROL:
                        continue
                    d, m, un = paired(block, field, arm, set(gset))
                    contrasts[sname][block][field][f"{arm}_vs_base"] = {
                        "per_game": m,
                        "mean_delta_over_games": round(sum(d) / len(d), 6) if d else None,
                        "delta_quantiles": quantiles(d),
                        "sign_test": sign_test(d),
                        "bootstrap_ci_over_games": bootstrap_ci(d),
                        "n_unmatched_cells_excluded": len(un),
                        "unmatched_cells": un[:20],
                    }
    res["paired_contrasts"] = contrasts
    res["strata"] = {k: {"games": v, "n": len(v)} for k, v in strata.items()}
    res["multiplicity"] = {
        "primary": "change_accuracy on the tail block, PRIMARY_plain_branch_ex_tn36 stratum, "
        "3 arm-vs-base contrasts",
        "bonferroni_threshold_for_the_3_primary_contrasts": round(0.05 / 3, 6),
        "bonferroni_threshold_for_the_6_secondaries": round(0.05 / 6, 6),
        "note": "the four fields are strongly correlated (all functions of the same predicted "
        "grids), so Bonferroni is conservative; the honest reading of any secondary hit "
        "is 'worth one confirmatory run', not 'established'.",
    }

    # ---- TARGET: did anything reach change_accuracy >= 0.5 on a non-tn36 game? -------------
    # HELD-OUT BLOCKS ONLY. `shown_train` scores the rows the prompt CONTAINED and must never
    # enter the target: an exact match there is memorisation of visible evidence, not
    # generalization. This is not hypothetical -- the first draft of this scan iterated BLOCKS,
    # which had `shown_train` appended to it later, and it duly reported sb26__r0__think at
    # change_accuracy 0.5 as a TARGET HIT. It was a training score. Caught only because every hit
    # records the block it came from; recorded here rather than silently corrected, because the
    # near-miss is the whole reason this run labels its blocks.
    held_out_blocks = ["tail", "fresh", "tail_substantive", "fresh_substantive"]
    hits = []
    for r in merged:
        if r["game"] == "tn36":
            continue
        for block in held_out_blocks:
            v = val(r, block, "change_accuracy")
            if v is not None and v >= 0.5:
                s = r["scored"][block]
                hits.append(
                    {
                        "cell_id": r["cell_id"],
                        "game": r["game"],
                        "arm": r["arm"],
                        "block": block,
                        "change_accuracy": v,
                        "n_changing": s.get("n_changing"),
                        "n_changes_correct": s.get("n_changes_correct"),
                        "branch": "hidden_state" if r["game"] in HIDDEN_STATE else "plain",
                    }
                )
    # The same scan over the TRAINING block, reported separately and labelled, because "can the
    # model exactly reproduce a row it was shown?" is a different and independently useful
    # question from the target.
    train_exact = []
    for r in merged:
        v = val(r, "shown_train", "change_accuracy")
        if v is not None and v > 0.0:
            s = r["scored"]["shown_train"]
            train_exact.append(
                {
                    "cell_id": r["cell_id"],
                    "game": r["game"],
                    "arm": r["arm"],
                    "change_accuracy": v,
                    "n_changing": s.get("n_changing"),
                    "n_changes_correct": s.get("n_changes_correct"),
                }
            )
    res["TRAINING_block_exact_matches_NOT_a_target_hit"] = {
        "n_cells_with_any_exact_training_match": len(train_exact),
        "cells": train_exact,
        "why_separate": "these are the rows the prompt CONTAINED. An exact match here is "
        "reproduction of visible evidence, never generalization, and it is "
        "excluded from the target by construction.",
    }
    # THE HONEST TARGET. A hit on an unstratified block can be carried entirely by one-cell
    # counter ticks (see the bp35 decomposition), so the target is reported at BOTH readings and
    # the substantive one is the one that means "the engine modelled a dynamic".
    # "MORE THAN A HANDFUL OF ROWS" is part of the target as the brief states it, and it is
    # load-bearing: the production tail leaves 2-3 gradable changing rows per game, where
    # change_accuracy can only take the values {0, 1/2, 1} or {0, 1/3, 2/3, 1}. A single lucky
    # exact match on a 2-row denominator reads as 0.5 and clears the bar while meaning almost
    # nothing. The threshold is applied explicitly at 10 rows rather than left implicit, and every
    # hit records its own n_changing so a reader can apply a different threshold.
    min_rows_for_a_real_hit = 10
    subs_hits = [h for h in hits if h["block"].endswith("_substantive")]
    subs_hits_powered = [
        h for h in subs_hits if (h.get("n_changing") or 0) >= min_rows_for_a_real_hit
    ]
    res["TARGET_on_SUBSTANTIVE_transitions_only"] = {
        "n_hits_any_row_count": len(subs_hits),
        "hits_any_row_count": subs_hits,
        "min_rows_for_a_real_hit": min_rows_for_a_real_hit,
        "n_hits_with_more_than_a_handful_of_rows": len(subs_hits_powered),
        "hits_with_more_than_a_handful_of_rows": subs_hits_powered,
        "TARGET_AS_THE_BRIEF_STATES_IT_reached": bool(subs_hits_powered),
        "n_hits": len(subs_hits),
        "hits": subs_hits,
        "target_reached_on_substantive_rows": bool(subs_hits),
        "definition": "change_accuracy >= 0.5 restricted to held-out rows where reality changed "
        ">= 2 cells, i.e. excluding one-cell HUD/counter ticks",
        "why_this_is_the_meaningful_reading": "change_accuracy weights a one-cell counter tick "
        "identically to a 47-cell state transition. An "
        "engine that induces only the progress counter can "
        "therefore clear 0.5 without modelling any game "
        "mechanic -- which is exactly what the single "
        "unstratified hit in this run turned out to be.",
    }
    res["TARGET_change_accuracy_ge_0.5_non_tn36"] = {
        "n_hits": len(hits),
        "hits": hits,
        "blocks_considered": held_out_blocks,
        "READ_WITH": "TARGET_on_SUBSTANTIVE_transitions_only -- an unstratified hit may be "
        "entirely one-cell counter ticks",
        "blocks_excluded": [
            "shown_train (training rows -- see TRAINING_block_exact_matches_NOT_a_target_hit)"
        ],
        "target_reached": bool(hits),
        "definition": "change_accuracy = n_changes_correct / n_changing, WHOLE-GRID EXACT matches "
        "over changing held-out rows",
    }

    # ---- MECHANISM: identity rate per arm (the antiid arm's own witness) -------------------
    ident = {}
    for arm in ARM_ORDER:
        rs = [r for r in merged if r["arm"] == arm and r.get("scored")]
        ip = [r["scored"].get("identity_probe") or {} for r in rs]
        n_id = sum(1 for p in ip if p.get("is_identity"))
        by_game: dict[str, list[bool]] = {}
        for r, p in zip(rs, ip, strict=False):
            by_game.setdefault(r["game"], []).append(bool(p.get("is_identity")))
        rates = [sum(v) / len(v) for v in by_game.values()]
        ident[arm] = {
            "n_engines_probed": len(ip),
            "n_identity": n_id,
            "identity_fraction_pooled": round(n_id / len(ip), 6) if ip else None,
            "per_game_identity_fraction_quantiles": quantiles(rates),
            "per_game": {g: round(sum(v) / len(v), 4) for g, v in sorted(by_game.items())},
        }
    # paired per-game identity contrast
    for arm in ARM_ORDER:
        if arm == CONTROL:
            continue
        by = {}
        for r in merged:
            if not r.get("scored"):
                continue
            p = r["scored"].get("identity_probe") or {}
            if "is_identity" not in p:
                continue
            by[(r["game"], r["replicate"], r["arm"])] = float(bool(p["is_identity"]))
        d = []
        for g in sorted({k[0] for k in by}):
            reps = sorted({k[1] for k in by if k[0] == g})
            av, bv = [], []
            for rp in reps:
                a, b = by.get((g, rp, arm)), by.get((g, rp, CONTROL))
                if a is None or b is None:
                    continue
                av.append(a)
                bv.append(b)
            if av:
                d.append(sum(av) / len(av) - sum(bv) / len(bv))
        ident[f"{arm}_vs_base_identity_delta"] = {
            "mean_delta_over_games": round(sum(d) / len(d), 6) if d else None,
            "sign_test": sign_test(d),
            "bootstrap_ci_over_games": bootstrap_ci(d),
        }
    res["identity_mechanism"] = ident

    # ---- COST + YIELD ----------------------------------------------------------------------
    cost = {}
    for arm in ARM_ORDER:
        rs = [c for c in cells if c["arm"] == arm]
        ok = [c for c in rs if c.get("induce_ok")]
        loaded = [r for r in merged if r["arm"] == arm and r.get("scored")]
        cost[arm] = {
            "n_cells": len(rs),
            "induce_ok_rate": round(len(ok) / len(rs), 4) if rs else None,
            "usable_scored_engine_rate": round(len(loaded) / len(rs), 4) if rs else None,
            "elapsed_s": quantiles([float(c["elapsed_s"]) for c in rs]),
            "generated_tokens": quantiles([float(c["generated_tokens"]) for c in rs]),
            "completion_calls": quantiles([float(c["completion_calls_delta"]) for c in rs]),
            "stop_types": dict(Counter(c.get("last_stop_type") for c in rs)),
            "n_prompt_truncated": sum(1 for c in rs if c.get("prompt_truncated")),
        }
    res["cost_and_yield"] = cost
    # Was the 8192 token budget binding for base? If base never approaches the shipped 4096, then
    # lifting the cap to equalise it with the think arm changed nothing about base.
    base_tok = [float(c["generated_tokens"]) for c in cells if c["arm"] == "base"]
    res["budget_lift_was_inert_for_base"] = {
        "shipped_max_tokens": 4096,
        "max_tokens_used_here": 8192,
        "base_generated_tokens_max": max(base_tok) if base_tok else None,
        "base_n_cells_above_4096": sum(1 for v in base_tok if v > 4096),
        "inert": bool(base_tok) and max(base_tok) <= 4096,
        "why_this_matters": "every arm got 8192 so the directive contrast could not be confounded "
        "with the token budget. If base never exceeds the shipped 4096 the "
        "lift is inert for base and `base` is the shipped prompt in effect.",
    }

    # ---- DISTANCE TO EXACT: how far from an exact match is each engine? --------------------
    # change_accuracy is an EXACT-match rate, so it reports 0.0000 identically for an engine that
    # returns its input and for one that is two cells away. This channel separates them, and it is
    # the difference between "the model declines to model" and "the model is one named mechanic
    # short". Aggregated per ARM over engines; the per-engine unit is that engine's MEDIAN wrong-
    # cell count over its own changing held-out rows.
    dist: dict = {}
    for block, key in (("tail", "distance_tail"), ("fresh", "distance_fresh")):
        dist[block] = {}
        for arm in ARM_ORDER:
            meds, mins, any2, any10, nrows, exact = [], [], 0, 0, 0, 0
            n_eng = 0
            for r in merged:
                if r["arm"] != arm or not r.get("scored"):
                    continue
                d = r["scored"].get(key) or {}
                if not d.get("n_rows"):
                    continue
                n_eng += 1
                meds.append(float(d["wrong_cells_median"]))
                mins.append(float(d["wrong_cells_min"]))
                nrows += int(d["n_rows"])
                exact += int(d.get("n_rows_exact", 0))
                any2 += 1 if d.get("n_rows_within_2", 0) > 0 else 0
                any10 += 1 if d.get("n_rows_within_10", 0) > 0 else 0
            dist[block][arm] = {
                "n_engines_with_gradable_rows": n_eng,
                "per_engine_median_wrong_cells": quantiles(meds),
                "per_engine_best_row_wrong_cells": quantiles(mins),
                "n_engines_with_any_row_within_2_cells": any2,
                "n_engines_with_any_row_within_10_cells": any10,
                "total_gradable_rows_scored": nrows,
                "total_rows_exact": exact,
            }
    dist["_reading"] = (
        "A high 'any row within 2 cells' count alongside change_accuracy 0.0000 means the engines "
        "are NEAR-MISS machines, not identity machines: they model most of the transition and miss "
        "a small number of cells. That is a different diagnosis from 'declines to model', and it "
        "is what the 2026-08-01 census found on one engine (ls20, 50 of 52 changed cells right, "
        "wrong on two counter cells) generalised here across the whole corpus and every arm."
    )
    res["distance_to_exact"] = dist

    # ---- BIT-IDENTITY (failure mode 10): are replicates independent observations? ----------
    # The sampler is seeded per replicate, so two replicates of the same (game, arm) SHOULD differ.
    # If they are bit-identical the replicate carries no extra information and averaging it in
    # would double-count one observation. Counted and reported rather than assumed away.
    dup = {}
    for arm in ARM_ORDER:
        by_ga: dict[str, list[str]] = {}
        for c in cells:
            if c["arm"] != arm or not c.get("engine_sha256"):
                continue
            by_ga.setdefault(c["game"], []).append(c["engine_sha256"])
        collapsed = {g: v for g, v in by_ga.items() if len(v) > 1 and len(set(v)) == 1}
        allsha = [s for v in by_ga.values() for s in v]
        dup[arm] = {
            "n_engines": len(allsha),
            "n_distinct_sha256": len(set(allsha)),
            "n_games_whose_replicates_are_all_bit_identical": len(collapsed),
            "games_with_collapsed_replicates": sorted(collapsed),
        }
    # cross-arm: did two ARMS produce a bit-identical engine for the same (game, replicate)?
    cross = []
    by_key: dict[tuple, dict[str, str]] = {}
    for c in cells:
        if c.get("engine_sha256"):
            by_key.setdefault((c["game"], c["replicate"]), {})[c["arm"]] = c["engine_sha256"]
    for k, m in by_key.items():
        inv: dict[str, list[str]] = {}
        for arm, s in m.items():
            inv.setdefault(s, []).append(arm)
        for _sha, arms in inv.items():
            if len(arms) > 1:
                cross.append({"game": k[0], "replicate": k[1], "arms": sorted(arms)})
    dup["_cross_arm_identical_engines"] = cross
    dup["_reading"] = (
        "A cross-arm bit-identical pair means those two arms produced the SAME engine for that "
        "cell, so that cell contributes an exact tie to their contrast by construction rather "
        "than by measurement. Reported so a tie-heavy sign test can be read correctly."
    )
    res["bit_identity_check"] = dup

    # ---- DIRECTIVE COMPLIANCE: does the code-only directive achieve its stated purpose? ----
    # `_L2_CODEONLY_DIRECTIVE` instructs "Do NOT write step-by-step analysis, explanation, or
    # commentary -- not even as comments" and "Skip all reasoning". Whether gemma-4-31B OBEYS that
    # is a separate question from whether obeying it helps, and it is answerable statically from
    # the emitted code. This is a census of the source text; nothing is executed.
    engdir = OUT / "engines"
    comp = {}
    for arm in ARM_ORDER:
        rows_c, frac, leak, doc = [], [], 0, 0
        for r in merged:
            if r["arm"] != arm:
                continue
            p = engdir / f"{r['cell_id']}.py"
            if not p.exists():
                continue
            try:
                src = p.read_text()
            except Exception:  # noqa: BLE001
                continue
            lines = [ln for ln in src.splitlines() if ln.strip()]
            if not lines:
                continue
            ncom = sum(1 for ln in lines if ln.strip().startswith("#"))
            rows_c.append(ncom)
            frac.append(ncom / len(lines))
            # control-token leakage: a chat-template marker that reached the emitted source
            if "<|channel" in src or "<|start" in src or "<|end" in src:
                leak += 1
            if '"""' in src or "'''" in src:
                doc += 1
        comp[arm] = {
            "n_engines": len(rows_c),
            "comment_lines": quantiles([float(x) for x in rows_c]),
            "comment_line_fraction": quantiles(frac),
            "n_with_control_token_leak": leak,
            "n_with_docstring": doc,
        }
    comp["_reading"] = (
        "The base/antiid/delta arms carry the code-only directive, which forbids commentary "
        "'not even as comments'. If their emitted code still carries a comparable comment "
        "fraction to the think arm (which was given no such instruction), the directive is not "
        "achieving its stated purpose on this generator -- it is displacing the reasoning into "
        "comments rather than suppressing it. That is a statement about COMPLIANCE, and is "
        "independent of whether compliance would have helped accuracy."
    )
    res["directive_compliance_static_census"] = comp

    res["prep_meta_summary"] = {
        g: {
            k: prep[g][k]
            for k in (
                "n_shown",
                "n_tail",
                "tail_gradable_changing",
                "n_fresh_kept",
                "fresh_gradable_changing",
                "fresh_dropped_content_collision_with_shown",
                "fresh_dropped_rendered_line_in_prompt",
                "tail_rows_colliding_with_shown_content",
                "tail_rows_whose_line_is_in_prompt",
            )
            if k in prep[g]
        }
        for g in prep
        if g != "_prep" and prep[g].get("built")
    }
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(
        json.dumps(
            {
                "cells": ledger["n_cells_run"],
                "scored_ok": ledger["n_scored_ok"],
                "missing": ledger["n_missing"],
                "reconciles": ledger["reconciles"],
                "target_reached": res["TARGET_change_accuracy_ge_0.5_non_tn36"]["target_reached"],
                "n_target_hits": res["TARGET_change_accuracy_ge_0.5_non_tn36"]["n_hits"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
