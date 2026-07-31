#!/usr/bin/env python3
"""Score the Phase-3 treatment-activation pre-flight and emit the PASS / REFUSE verdict.

Three quantities, in increasing order of how much they are allowed to license:

  1. RAW A/B PERTURBATION -- did the two arms' action traces differ at all. On its own this
     is not evidence of anything: a nondeterministic harness makes every cell perturb.

  2. ATTRIBUTABLE PERTURBATION -- perturbs under A/B AND byte-identical under BOTH arms'
     A/A replicates. This is what `preflight_verdict` uses as the ceiling on discordant pairs.

  3. INDUCTION-ATTRIBUTABLE PERTURBATION -- attributable AND the first divergence lands at or
     after the first induction. This is the one that matters here and it is NOT a refinement
     for its own sake. The treatment lives inside `induce()`. `explore_budget=24` transitions
     are collected BEFORE the first induce, so a divergence at action index < 24 cannot have
     been caused by it -- it is harness noise or upstream drift wearing the treatment's name.
     The 2026-07-30 composite probe found 3 of its 4 attributable cells diverged pre-induction,
     so reporting (2) alone would repeat an overstatement that has already happened once on
     this exact path.

The boundary is deliberately CONSERVATIVE and code-grounded rather than read from a recorded
index (no induction event carries an action index): each explored transition costs at least one
action, so the first induction cannot occur before action `explore_budget`. Using >= 24 as the
cutoff therefore over-credits the treatment if anything, which is the safe direction for a
quantity used to REFUSE.
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "pf", "cells")
OUT = os.path.join(HERE, "pf", "preflight_scored.json")
MAIN_REPO = "/home/ianblenke/github.com/ianblenke/carnot"

sys.path.insert(0, os.path.join(MAIN_REPO, "python"))
from carnot.analysis.treatment_activation_preflight import (  # noqa: E402
    IDENTICAL,
    PERTURBED,
    classify_trace_pair,
    format_report,
    min_one_way_discordant_pairs,
    preflight_verdict,
    two_sided_sign_test_p,
)

GAMES = ["ft09", "tn36", "tu93", "vc33", "sc25", "lp85"]
SEED = 1
PLANNED_N_CELLS = 12
EXPLORE_BUDGET = 24
ALPHA = 0.05


def _load(arm: str, game: str):
    p = os.path.join(CELLS, f"{arm}__{game}__s{SEED}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _trace_and_complete(d):
    if d is None:
        return None, False
    res = d.get("result") or {}
    return res.get("action_trace"), (d.get("status") == "ok" and not res.get("timed_out"))


def _same_server_process(a: str, b: str, g: str):
    """Did these two arms talk to the SAME llama-server PROCESS?

    Load-bearing, not hygiene. Measured on this box: the sampler seed does NOT reach across
    server processes -- an identical config on a second server gives different output, while
    within one process it holds byte-exactly. So a pair whose arms crossed a process boundary
    differs by sampler variance alone and its classification means nothing, in EITHER
    direction: a spurious PERTURBED would be credited to the treatment, and a spurious
    IDENTICAL would understate it.

    This is not hypothetical here. Both original servers were replaced partway through the
    session (a concurrent agent claimed a card), so whether any given pair is intra-process is a
    fact that has to be READ from what each cell recorded, never assumed from the port it asked
    for. Returns None when either side has no record to check.
    """
    da, db = _load(a, g), _load(b, g)
    if not da or not db:
        return None
    pa, pb = da.get("server_pid"), db.get("server_pid")
    if pa is None or pb is None:
        return None
    return pa == pb


def _pairs(a: str, b: str) -> dict:
    out = {}
    for g in GAMES:
        ta, ca = _trace_and_complete(_load(a, g))
        tb, cb = _trace_and_complete(_load(b, g))
        rec = dict(classify_trace_pair(ta, tb, a_complete=ca, b_complete=cb))
        same = _same_server_process(a, b, g)
        rec["same_server_process"] = same
        if same is False:
            # Downgrade to a MISSING OBSERVATION rather than score a confounded pair. Scoring
            # it either way would be the error this whole module exists to prevent.
            rec["cls_before_server_process_guard"] = rec.get("cls")
            rec["cls"] = "MISSING"
            rec["why"] = ("arms crossed a llama-server PROCESS boundary, where the sampler seed "
                          "does not hold; the pair is confounded by sampler variance and is "
                          "excluded as a missing observation rather than scored")
        out[g] = rec
    return out


def _binom_at_least(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no scipy dependency."""
    if p <= 0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1:
        return 1.0 if k <= n else 0.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main() -> int:
    ab = _pairs("ctl", "trt")
    aa_ctl = _pairs("ctl", "ctlb")
    aa_trt = _pairs("trt", "trtb")

    verdict = preflight_verdict(
        ab, alpha=ALPHA, planned_n_cells=PLANNED_N_CELLS,
        noise_pairs=aa_ctl, noise_pairs_b=aa_trt,
    )

    # ---- the induction boundary ----------------------------------------------------------
    attributable = [
        g for g in GAMES
        if ab[g].get("cls") == PERTURBED
        and aa_ctl[g].get("cls") == IDENTICAL
        and aa_trt[g].get("cls") == IDENTICAL
    ]
    induction_attr, pre_induction_only, boundary_detail = [], [], {}
    for g in attributable:
        idx = ab[g].get("first_divergence_index")
        if idx is None:
            # One arm's trace is a strict PREFIX of the other and it stopped of its own
            # accord. The divergence IS the early stop, located at the shorter length.
            idx = min(ab[g].get("len_a") or 0, ab[g].get("len_b") or 0)
            kind = "early_stop_at_prefix_end"
        else:
            kind = "differing_action"
        after = idx >= EXPLORE_BUDGET
        boundary_detail[g] = {
            "first_divergence_index": idx, "divergence_kind": kind,
            "explore_budget": EXPLORE_BUDGET,
            "at_or_after_first_induction": after,
            "why": ("at or after the first induce, so this treatment could have caused it"
                    if after else
                    "BEFORE the first induce could have run, so it cannot be this treatment"),
        }
        (induction_attr if after else pre_induction_only).append(g)

    # ---- did the tier actually FIRE, and did it change the induction --------------------
    # Separates "the tier did not help" from "the tier never fired" -- routinely confused, and
    # no trace diff can tell them apart on its own.
    tier = {}
    for g in GAMES:
        row = {}
        for arm in ("ctl", "trt"):
            d = _load(arm, g)
            if d is None:
                row[arm] = None
                continue
            res = d.get("result") or {}
            row[arm] = {
                "repeat_penalty_effective": d.get("induce_repeat_penalty_effective"),
                "reasks_allowed": d.get("induce_defect_reasks_effective"),
                "reasks_observed": d.get("n_induce_defect_reasks_observed"),
                "n_inductions": res.get("n_inductions"),
                "n_plans_found": res.get("n_plans_found"),
                "mean_heldout_accuracy": res.get("mean_heldout_accuracy"),
                "total_actions": res.get("total_actions"),
                "levels_gained": res.get("levels_gained"),
                "timed_out": res.get("timed_out"),
                "status": d.get("status"),
            }
        c, t = row.get("ctl"), row.get("trt")
        row["induction_internals_differ"] = bool(
            c and t and any(c.get(k) != t.get(k) for k in
                            ("n_plans_found", "mean_heldout_accuracy"))
        )
        row["treatment_tier_fired"] = bool(t and (t.get("reasks_observed") or 0) > 0)
        tier[g] = row

    # ---- instrument cross-check: the external wrapper vs the shipped native recorder -------
    # These two DISAGREE byte-for-byte by construction -- the wrapper emits
    # `ACTION6|{"x":14,"y":54}` where the shipped recorder emits
    # `{"action":6,"data":{"x":14,"y":54}}` -- so comparing the raw strings reports a mismatch
    # on every data-carrying action and proves nothing. (The 2026-07-30 probe shipped exactly
    # that bug and had to correct it.) Normalising both to (kind, sorted-data) is what makes
    # this an actual check that the instrument sees what the agent did.
    def _norm(lbl: str):
        if lbl in ("RESET", "NONE"):
            return (lbl, None)
        if lbl.startswith("ACTION") and "|" in lbl:
            k, _, rest = lbl[6:].partition("|")
            try:
                return (str(k), json.dumps(json.loads(rest), sort_keys=True))
            except Exception:
                return (str(k), rest)
        try:
            o = json.loads(lbl)
            data = o.get("data")
            return (str(o.get("action")), json.dumps(data, sort_keys=True) if data else None)
        except Exception:
            return (lbl, None)

    xcheck = {}
    for g in GAMES:
        for arm in ("ctl", "trt", "ctlb", "trtb"):
            d = _load(arm, g)
            if not d:
                continue
            ext = (d.get("result") or {}).get("action_trace") or []
            nat = d.get("native_action_trace") or []
            xcheck[f"{arm}__{g}"] = {
                "len_external": len(ext), "len_native": len(nat),
                "agree_after_normalisation": [_norm(x) for x in ext] == [_norm(x) for x in nat],
            }
    all_agree = all(v["agree_after_normalisation"] for v in xcheck.values()) if xcheck else None

    n_comparable = sum(1 for g in GAMES if ab[g].get("cls") != "MISSING")
    rate_attr = len(attributable) / n_comparable if n_comparable else 0.0
    rate_ind = len(induction_attr) / n_comparable if n_comparable else 0.0
    required = min_one_way_discordant_pairs(ALPHA)

    def _cells_for(rate: float, target: float = 0.5) -> int | None:
        """Smallest grid size whose CHARITABLE ceiling reaches significance with prob >= target.

        Charitable on purpose: it assumes every induction-attributable perturbed cell also
        turns out DISCORDANT and points the SAME WAY. Both are unmeasured here, so this is a
        best case, not an estimate -- if even this number is out of reach, the refusal is
        beyond argument.
        """
        if rate <= 0:
            return None
        for n in range(required, 4001):
            if _binom_at_least(n, required, rate) >= target:
                return n
        return None

    # ---- how much "0 perturbed" is actually allowed to claim ------------------------------
    # A zero count over a handful of cells is NOT "the treatment never perturbs". The
    # one-sided 95% Clopper-Pearson upper bound for 0 successes in n trials is 1-0.05**(1/n)
    # -- at n=5 that is 0.451, i.e. the true attributable rate could still be as high as ~45%
    # and this probe would have seen zero with probability >=5%. Stating the bound keeps the
    # refusal honest: it rests on the MECHANISM (plans=0 either way) at least as much as on
    # the count, and the count alone would be thin.
    ci_upper = (1 - 0.05 ** (1 / n_comparable)) if n_comparable else None

    payload = {
        "perturbation_rate_95pct_upper_bound_given_zero": (
            round(ci_upper, 4) if ci_upper is not None else None),
        "perturbation_rate_upper_bound_note": (
            "one-sided 95% Clopper-Pearson bound for 0 perturbed of "
            f"{n_comparable}; the refusal rests on the mechanism (n_plans_found identical and "
            "0 across arms) as much as on the count"),
        "probe": "phase3_treatment_activation_preflight_wired_induce_treatment",
        "games": GAMES, "seed": SEED, "alpha": ALPHA,
        "required_one_way_discordant_pairs": required,
        "planned_n_cells": PLANNED_N_CELLS,
        "explore_budget": EXPLORE_BUDGET,
        "n_comparable_cells": n_comparable,
        "ab_per_cell": ab,
        "aa_ctl_per_cell": aa_ctl,
        "aa_trt_per_cell": aa_trt,
        "raw_ab_perturbed": [g for g in GAMES if ab[g].get("cls") == PERTURBED],
        "aa_ctl_perturbed": [g for g in GAMES if aa_ctl[g].get("cls") == PERTURBED],
        "aa_trt_perturbed": [g for g in GAMES if aa_trt[g].get("cls") == PERTURBED],
        "attributable_cells": attributable,
        "attributable_rate": round(rate_attr, 4),
        "induction_attributable_cells": induction_attr,
        "induction_attributable_rate": round(rate_ind, 4),
        "pre_induction_only_cells_NOT_this_treatment": pre_induction_only,
        "boundary_detail": boundary_detail,
        "tier_diagnostics": tier,
        "instrument_cross_check": xcheck,
        "instrument_external_and_native_agree_everywhere": all_agree,
        # Reported BEFORE any endpoint is measured, which is the whole point of a power
        # pre-flight: the best p the planned grid could ever return, given how many of its
        # cells the treatment is even capable of moving.
        "min_reachable_p_at_planned_size_charitable": two_sided_sign_test_p(
            min(required, round(rate_ind * PLANNED_N_CELLS))
        ),
        "p_planned_grid_reaches_alpha_charitable": round(
            _binom_at_least(PLANNED_N_CELLS, required, rate_ind), 4),
        "cells_for_50pct_chance_charitable": _cells_for(rate_ind),
        "cells_for_50pct_chance_span_attributable": _cells_for(rate_attr),
        "module_verdict": verdict,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)

    print(format_report(verdict, title="Phase-3 pre-flight: wired induce treatment"))
    print()
    print(f"raw A/B perturbed            : {payload['raw_ab_perturbed']}")
    print(f"A/A floor ctl perturbed      : {payload['aa_ctl_perturbed']}")
    print(f"A/A floor trt perturbed      : {payload['aa_trt_perturbed']}")
    print(f"attributable (both floors)   : {attributable}  rate={rate_attr:.3f}")
    print(f"  of which POST-induction    : {induction_attr}  rate={rate_ind:.3f}")
    print(f"  pre-induction (NOT ours)   : {pre_induction_only}")
    print(f"P(12-cell grid reaches a<.05): {payload['p_planned_grid_reaches_alpha_charitable']}"
          "   [charitable ceiling]")
    print(f"cells for 50% chance         : {payload['cells_for_50pct_chance_charitable']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
