#!/usr/bin/env python3
"""Answer Q1/Q2/Q3 off the frozen masked re-score. Pure arithmetic -- no engine is executed.

EVERY NUMBER IS REPORTED MASKED AND UNMASKED SIDE BY SIDE. A masked column alone cannot be
read: the mask is `applied` on only some games, and on the rest a masked arm is BY CONSTRUCTION
identical to the unmasked one. Printing those identical values without the status next to them
is the "second column of identical numbers wearing a different name" the A/B warned about, so
`hud_mask_status` travels with every figure.
"""

from __future__ import annotations

import json
import math
import pathlib
from itertools import combinations

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent
REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
AB = REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801"

ARMS = (
    "unmasked",
    "default_swallow_full",
    "default_swallow_slice",
    "default_forced_guard_bypassed",
    "conditional_swallow_full",
    "conditional_swallow_slice",
    "conditional_forced_guard_bypassed",
)
HEADLINE_ARM = "default_swallow_full"


def sign_test(deltas: list[float]) -> dict:
    """Exact two-sided paired sign test, ties dropped. The pre-registered primary test."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    n = pos + neg
    if n == 0:
        return {
            "n_pairs": len(deltas),
            "n_positive": pos,
            "n_negative": neg,
            "n_ties": ties,
            "n_discordant": 0,
            "p_two_sided": None,
            "test_was_possible": False,
            "why_not": "every pair is a tie -- the sign test has nothing to test",
        }
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {
        "n_pairs": len(deltas),
        "n_positive": pos,
        "n_negative": neg,
        "n_ties": ties,
        "n_discordant": n,
        "p_two_sided": min(1.0, 2 * tail),
        "test_was_possible": True,
        "min_reachable_two_sided_p_at_this_discordance": min(1.0, 2 * 0.5**n),
    }


def bootstrap_ci(deltas: list[float], n_resamples: int = 20000, seed: int = 20260801) -> dict:
    """Percentile bootstrap over GAMES -- the same resampling unit as the clustering."""
    if not deltas:
        return {"lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    a = np.asarray(deltas, dtype=float)
    means = a[rng.integers(0, len(a), size=(n_resamples, len(a)))].mean(axis=1)
    return {
        "mean": round(float(a.mean()), 8),
        "lo": round(float(np.percentile(means, 2.5)), 8),
        "hi": round(float(np.percentile(means, 97.5)), 8),
        "n": len(a),
        "n_resamples": n_resamples,
        "excludes_zero": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0),
    }


def signflip(deltas: list[float]) -> dict:
    """Exact sign-flip permutation on the per-game deltas; reported alongside the sign test
    because it uses the magnitudes the sign test throws away. Enumerated when <= 2**20."""
    nz = [d for d in deltas if d != 0]
    if not nz or len(nz) > 20:
        return {"test_was_possible": False, "n_nonzero": len(nz)}
    obs = abs(float(np.mean(nz)))
    a = np.asarray(nz, dtype=float)
    n = len(a)
    signs = np.array(
        [[1 if (i >> b) & 1 == 0 else -1 for b in range(n)] for i in range(2**n)], dtype=float
    )
    means = np.abs((signs * a).mean(axis=1))
    return {
        "test_was_possible": True,
        "n_nonzero": n,
        "observed_mean": round(float(np.mean(nz)), 8),
        "p_two_sided": round(float((means >= obs - 1e-12).mean()), 8),
        "n_enumerated": int(2**n),
    }


# ---------------------------------------------------------------------------------------
def per_game_means(cells: list[dict], arm: str, rows: list[dict]) -> dict:
    """Replicates averaged FIRST, then paired. The clustering the preregistration fixes.

    THE UNIT SET IS `rows.json`, NOT the re-scored cells, AND THAT DISTINCTION IS THE WHOLE
    REASON THIS FUNCTION TAKES BOTH. 8 of the A/B's 120 on/off cells have NO committed
    `world_model.py`: `engine_file_exists: false` with `missing: false`, i.e. the generator
    returned a complete response that never wrote an engine. The preregistration's
    MISSING_VS_ZERO rule puts those in the ZERO class, not the EXCLUDED class -- "a complete
    response whose code does not import" is a real failure of the treatment and scores 0.0 --
    and the published analysis paired all 3 replicates on all 20 games accordingly.

    Driving the pairing off the re-scored cells alone silently dropped those 8 and moved the
    unmasked mean delta from +0.07208 to +0.10059 while leaving the sign test unchanged, which
    is exactly the shape of error that survives a casual check. They are re-inserted at 0.0
    here, which is not an assumption but a definition: there is no engine, so there is no
    prediction, under any mask. Masking cannot change the score of an engine that does not
    exist, so these 8 are common-mode across every arm.

    A game is paired only where BOTH arms have the SAME replicate indices, exactly as the
    original analysis did.
    """
    scored = {
        c["cell"]: ((c.get("arms") or {}).get(arm) or {}).get("change_fidelity")
        for c in cells
        if c.get("status") == "ok"
    }
    by: dict = {}
    for r in rows:
        tag = r["tag"]
        if tag not in ("on", "off"):  # AA control cells are not part of the primary
            continue
        cell = f"{r['game']}__r{r['replicate']}__{tag}"
        if not r.get("engine_file_exists"):
            v: float | None = 0.0
        else:
            v = scored.get(cell)
        if v is None:
            continue
        by.setdefault(r["game"], {}).setdefault(tag, {})[int(r["replicate"])] = float(v)
    out = {}
    for game, d in by.items():
        on, off = d.get("on", {}), d.get("off", {})
        shared = sorted(set(on) & set(off))
        if not shared:
            continue
        o = float(np.mean([on[r] for r in shared]))
        f = float(np.mean([off[r] for r in shared]))
        out[game] = {
            "off": round(f, 6),
            "on": round(o, 6),
            "delta": round(o - f, 6),
            "n_matched_replicates": len(shared),
            "matched_replicates": shared,
        }
    return out


def primary(cells: list[dict], arm: str, rows: list[dict]) -> dict:
    pg = per_game_means(cells, arm, rows)
    games = sorted(pg)
    deltas = [pg[g]["delta"] for g in games]
    # WHICH games this arm actually masked. Without it a reader cannot tell a real null from an
    # arm that changed nothing because it had no mask to apply.
    applied = sorted(
        {
            c["game"]
            for c in cells
            if c.get("status") == "ok"
            and ((c.get("arms") or {}).get(arm) or {}).get("hud_mask_status") == "applied"
        }
    )
    return {
        "arm": arm,
        "n_games_paired": len(games),
        "n_games_where_mask_was_APPLIED": len(applied),
        "games_where_mask_was_APPLIED": applied,
        "games_where_this_arm_is_identical_to_unmasked_by_construction": [
            g for g in games if g not in applied
        ],
        "mean_delta_over_games": round(float(np.mean(deltas)), 8) if deltas else None,
        "sign_test": sign_test(deltas),
        "signflip_test": signflip(deltas),
        "bootstrap_ci_over_games": bootstrap_ci(deltas),
        "per_game": pg,
    }


# ---------------------------------------------------------------------------------------
def auc_with_ci(pos: list[float], neg: list[float], seed: int = 20260801) -> dict:
    """P(a plannable engine outranks an unplannable one), ties counted as 1/2.

    0.5 is "no association". Reported with an EXHAUSTIVE bootstrap CI because n_positive is 2:
    at that size a bootstrap resamples the same two points over and over, so the interval is
    honest only if it is read as "this is what two points can support", which is why the
    exact count of concordant/discordant/tied pairs is reported next to it.
    """
    if not pos or not neg:
        return {"auc": None, "n_pos": len(pos), "n_neg": len(neg), "computable": False}
    conc = sum(1 for a in pos for b in neg if a > b)
    disc = sum(1 for a in pos for b in neg if a < b)
    tied = sum(1 for a in pos for b in neg if a == b)
    total = len(pos) * len(neg)
    auc = (conc + 0.5 * tied) / total
    rng = np.random.default_rng(seed)
    pos_a, neg_a = np.asarray(pos, float), np.asarray(neg, float)
    boots = []
    for _ in range(20000):
        p = pos_a[rng.integers(0, len(pos_a), len(pos_a))]
        q = neg_a[rng.integers(0, len(neg_a), len(neg_a))]
        c = float((p[:, None] > q[None, :]).sum())
        t = float((p[:, None] == q[None, :]).sum())
        boots.append((c + 0.5 * t) / (len(p) * len(q)))
    b = np.asarray(boots)
    return {
        "auc": round(float(auc), 6),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_concordant_pairs": conc,
        "n_discordant_pairs": disc,
        "n_tied_pairs": tied,
        "n_pairs_total": total,
        "bootstrap_ci_95": [
            round(float(np.percentile(b, 2.5)), 6),
            round(float(np.percentile(b, 97.5)), 6),
        ],
        "ci_excludes_0.5": bool(np.percentile(b, 2.5) > 0.5 or np.percentile(b, 97.5) < 0.5),
        "computable": True,
    }


def ranking(bon: list[dict], arm: str) -> list[dict]:
    rows = []
    for r in bon:
        if r.get("status") != "ok":
            continue
        a = (r.get("arms") or {}).get(arm) or {}
        if a.get("change_fidelity") is None:
            continue
        rows.append(
            {
                "game": r["game"],
                "cand": r["cand"],
                "change_fidelity": a["change_fidelity"],
                "accuracy": a.get("accuracy"),
                "n_changing": a.get("n_changing"),
                "plan_found": bool(r.get("plan_found")),
                "goal_satisfiable": bool(r.get("goal_satisfiable")),
                "hud_mask_status": a.get("hud_mask_status"),
                "hud_mask_cells": a.get("hud_mask_cells"),
            }
        )
    rows.sort(key=lambda r: -r["change_fidelity"])
    for i, r in enumerate(rows, 1):
        r["rank"] = f"{i}/{len(rows)}"
    return rows


def q3(bon: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        rows = ranking(bon, arm)
        pos = [r["change_fidelity"] for r in rows if r["plan_found"]]
        neg = [r["change_fidelity"] for r in rows if not r["plan_found"]]
        top = [r for r in rows if r["change_fidelity"] >= 0.999]
        n_applied = sum(1 for r in rows if r["hud_mask_status"] == "applied")
        out[arm] = {
            "n_candidates_scored": len(rows),
            "n_candidates_where_mask_was_APPLIED": n_applied,
            "n_plannable": len(pos),
            "where_the_plannable_ones_rank": [
                {
                    "game": r["game"],
                    "candidate": r["cand"],
                    "change_fidelity": r["change_fidelity"],
                    "ordinal_rank_TIE_ORDER_IS_ARBITRARY": r["rank"],
                    # TIE-SAFE POSITION. Under the forced arm 25 of 31 candidates sit at exactly
                    # 0.0, so an ordinal rank among them is decided by input order and means
                    # nothing -- reading "rank 11/31" as better than "rank 13/31" there would be
                    # reading sort stability as a result. These two fields are the honest
                    # statement of position and are what any claim must be made from.
                    "n_candidates_strictly_above": sum(
                        1 for x in rows if x["change_fidelity"] > r["change_fidelity"]
                    ),
                    "n_candidates_tied_with_it": sum(
                        1 for x in rows if x["change_fidelity"] == r["change_fidelity"]
                    )
                    - 1,
                    "strictly_beaten_by_fewer_than_half": sum(
                        1 for x in rows if x["change_fidelity"] > r["change_fidelity"]
                    )
                    < len(rows) / 2,
                    "hud_mask_status": r["hud_mask_status"],
                }
                for r in rows
                if r["plan_found"]
            ],
            "n_perfect_fidelity_candidates": len(top),
            "n_perfect_fidelity_candidates_that_are_plannable": sum(
                1 for r in top if r["plan_found"]
            ),
            "association_plannable_vs_change_fidelity": auc_with_ci(pos, neg),
            "n_candidates_at_exactly_zero": sum(1 for r in rows if r["change_fidelity"] == 0.0),
            "n_candidates_with_zero_changing_transitions": sum(
                1 for r in rows if r["n_changing"] == 0
            ),
        }
    out["ranking_headline_arm"] = ranking(bon, HEADLINE_ARM)
    out["ranking_unmasked"] = ranking(bon, "unmasked")
    out["ranking_conditional_forced"] = ranking(bon, "conditional_forced_guard_bypassed")
    return out


def q3_rank_stability(ab_cells: list[dict], bon: list[dict]) -> dict:
    return {
        "why": (
            "Q3's plannability join has n_plannable = 2 and cannot establish anything. This asks "
            "the same question where n is not 2: does the mask REORDER engines, or only move the "
            "numbers? A masked score that is a monotone transform of the unmasked one ranks every "
            "engine identically, so turning masking on would change no selection the metric is "
            "used for."
        ),
        "ab_116_engines": {a: rank_stability(ab_cells, a, "cell") for a in ARMS if a != "unmasked"},
        "bestofn_31_candidates": {
            a: rank_stability(bon, a, "game") for a in ARMS if a != "unmasked"
        },
    }


def rank_stability(cells: list[dict], arm: str, key: str = "cell") -> dict:
    """Does the mask REORDER engines, or only rescale them?

    Q3 as posed rides on n_plannable = 2, which cannot establish anything. This is the same
    question -- "does masking change what the metric MEANS" -- asked where the sample is not 2:
    if the masked score is a monotone transform of the unmasked one, the metric ranks engines
    identically and masking changes only the numbers on the axis. If it reorders them, masking
    is measuring something genuinely different. Restricted to the units where the mask was
    actually APPLIED, because a unit that was never masked contributes a guaranteed tie and
    would inflate the correlation toward 1 for free.
    """
    pairs = [
        (
            (c["arms"]["unmasked"]).get("change_fidelity"),
            (c["arms"][arm]).get("change_fidelity"),
            c.get(key),
        )
        for c in cells
        if c.get("status") == "ok"
        and (c["arms"].get(arm) or {}).get("hud_mask_status") == "applied"
        and (c["arms"].get(arm) or {}).get("change_fidelity") is not None
    ]
    if len(pairs) < 3:
        return {"n_units_masked": len(pairs), "computable": False}
    u = np.asarray([p[0] for p in pairs], float)
    m = np.asarray([p[1] for p in pairs], float)

    # Spearman without scipy: Pearson on ranks (average ranks for ties).
    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[order] = np.arange(1, len(a) + 1, dtype=float)
        for v in np.unique(a):
            idx = a == v
            r[idx] = r[idx].mean()
        return r

    ru, rm = rank(u), rank(m)
    sd = ru.std() * rm.std()
    rho = float(((ru - ru.mean()) * (rm - rm.mean())).mean() / sd) if sd > 0 else None
    both = list(zip(u, m, strict=True))
    inversions = sum(1 for (a1, b1), (a2, b2) in combinations(both, 2) if (a1 - a2) * (b1 - b2) < 0)
    return {
        "n_units_masked": len(pairs),
        "computable": True,
        "spearman_rho_masked_vs_unmasked": None if rho is None else round(rho, 6),
        "n_pairwise_rank_inversions": inversions,
        "n_pairs_compared": len(pairs) * (len(pairs) - 1) // 2,
        "n_units_whose_score_moved_at_all": int((np.abs(u - m) > 1e-9).sum()),
        "mean_absolute_score_shift": round(float(np.abs(u - m).mean()), 6),
        "mean_signed_score_shift": round(float((m - u).mean()), 6),
    }


def q1(bon: list[dict]) -> dict:
    """The six perfect-`change_fidelity` tn36 bar-tickers, every arm, plus the whole game."""
    tn = [r for r in bon if r.get("game") == "tn36" and r.get("status") == "ok"]
    perfect = [
        r
        for r in tn
        if ((r.get("arms") or {}).get("unmasked") or {}).get("change_fidelity", 0) >= 0.999
    ]

    def row(r):
        rec = {
            "candidate": r["cand"],
            "plan_found": bool(r.get("plan_found")),
            "goal_satisfiable": bool(r.get("goal_satisfiable")),
        }
        for arm in ARMS:
            a = (r.get("arms") or {}).get(arm) or {}
            rec[arm] = {
                "change_fidelity": a.get("change_fidelity"),
                "accuracy": a.get("accuracy"),
                "n_changing": a.get("n_changing"),
                "n_noop": a.get("n_noop"),
                "hud_mask_status": a.get("hud_mask_status"),
                "hud_mask_cells": a.get("hud_mask_cells"),
            }
        return rec

    sw = {}
    if tn:
        sw = (tn[0].get("full_corpus_swallow_check") or {}).get("conditional") or {}
    return {
        "n_perfect_unmasked": len(perfect),
        "perfect_candidates": [row(r) for r in perfect],
        "all_tn36_candidates": [row(r) for r in tn],
        "tn36_conditional_mask_swallow_verdict_on_the_bestofn_corpus": sw,
    }


def reproduction_gate(mine: dict, orig: dict) -> dict:
    """The UNMASKED arm, re-derived here, must equal the published unmasked primary.

    This is the gate that makes every masked number believable. The masked arms differ from the
    published analysis in two ways at once -- a rebuilt window and a mask -- and only one of
    them is the thing being measured. If the unmasked arm reproduces exactly, the window is the
    window the A/B graded and the only remaining difference IS the mask. If it does not, the
    pass is void rather than partly reported: a masked delta computed on a different window is
    not a masked delta.
    """
    st, ost = mine["sign_test"], orig["sign_test"]
    checks = {
        "mean_delta": (mine["mean_delta_over_games"], orig["mean_delta_over_games"]),
        "n_games_paired": (mine["n_games_paired"], orig["n_games_paired"]),
        "n_positive": (st["n_positive"], ost["n_positive"]),
        "n_negative": (st["n_negative"], ost["n_negative"]),
        "n_ties": (st["n_ties"], ost["n_ties"]),
        "p_two_sided": (st["p_two_sided"], ost["p_two_sided"]),
        "bootstrap_lo": (
            mine["bootstrap_ci_over_games"]["lo"],
            orig["bootstrap_ci_over_games"]["lo"],
        ),
        "bootstrap_hi": (
            mine["bootstrap_ci_over_games"]["hi"],
            orig["bootstrap_ci_over_games"]["hi"],
        ),
    }
    out = {"per_field": {}, "all_reproduce": True}
    for k, (a, b) in checks.items():
        ok = (
            abs(float(a) - float(b)) < 1e-6
            if isinstance(a, float) or isinstance(b, float)
            else a == b
        )
        out["per_field"][k] = {"rederived": a, "published": b, "reproduces": bool(ok)}
        # The bootstrap bounds are the ONE place a mismatch is not a defect: they are a
        # resampling estimate and this pass does not claim the original's RNG stream. They are
        # reported and excluded from the gate rather than quietly dropped.
        if not ok and not k.startswith("bootstrap_"):
            out["all_reproduce"] = False
    out["bootstrap_bounds_excluded_from_the_gate_because"] = (
        "a percentile bootstrap depends on the resampling RNG stream, which this pass does not "
        "claim to reproduce. The point estimate and the exact test ARE gated."
    )
    return out


def main() -> int:
    raw = json.loads((OUT / "rescore_masked_raw.json").read_text())
    ab_cells, bon = raw["ab_cells"], raw["bon_candidates"]
    rows = json.loads((AB / "rows.json").read_text())
    orig = json.loads((AB / "analysis.json").read_text())["PRIMARY"]
    arms = {a: primary(ab_cells, a, rows) for a in ARMS}

    result = {
        "Q1_does_the_degeneracy_die": q1(bon),
        "Q2_does_the_object_perception_effect_survive": {
            "ORIGINAL_UNMASKED_AS_PUBLISHED": {
                "mean_delta": orig["mean_delta_over_games"],
                "sign_test": orig["sign_test"],
                "bootstrap_ci": orig["bootstrap_ci_over_games"],
            },
            "REPRODUCTION_GATE_unmasked_arm_vs_published": reproduction_gate(
                arms["unmasked"], orig
            ),
            "arms": arms,
        },
        "Q3_does_masking_change_what_the_metric_means": q3(bon),
        "Q3b_rank_stability_the_same_question_where_n_is_not_2": q3_rank_stability(ab_cells, bon),
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=1))

    g = result["Q2_does_the_object_perception_effect_survive"][
        "REPRODUCTION_GATE_unmasked_arm_vs_published"
    ]
    print(f"REPRODUCTION GATE (unmasked arm vs published): all_reproduce={g['all_reproduce']}")
    for k, v in g["per_field"].items():
        if not v["reproduces"]:
            print(f"  MISMATCH {k}: rederived={v['rederived']} published={v['published']}")

    p = result["Q2_does_the_object_perception_effect_survive"]["arms"]
    print("\nQ2 -- per arm (change_fidelity, game-clustered, replicates averaged first)")
    for a in ARMS:
        r = p[a]
        st = r["sign_test"]
        ci = r["bootstrap_ci_over_games"]
        print(
            f"  {a:36} applied_on={r['n_games_where_mask_was_APPLIED']:>2}/20  "
            f"mean={r['mean_delta_over_games']:+.5f}  "
            f"+{st['n_positive']}/-{st['n_negative']}/={st['n_ties']}  "
            f"p={st['p_two_sided']}  CI=[{ci['lo']:+.5f},{ci['hi']:+.5f}]"
        )
    print("\nQ1 -- tn36 best-of-N")
    for r in result["Q1_does_the_degeneracy_die"]["all_tn36_candidates"]:
        print(
            f"  k{r['candidate']} plan={str(r['plan_found']):<5} "
            f"unmasked={r['unmasked']['change_fidelity']:<9} "
            f"cond_guard={r['conditional_swallow_full']['change_fidelity']:<9}"
            f"({r['conditional_swallow_full']['hud_mask_status']}) "
            f"forced={r['conditional_forced_guard_bypassed']['change_fidelity']} "
            f"nch {r['unmasked']['n_changing']}"
            f"->{r['conditional_forced_guard_bypassed']['n_changing']}"
        )
    print("\nQ3 -- plannability join")
    for a in ARMS:
        r = result["Q3_does_masking_change_what_the_metric_means"][a]
        au = r["association_plannable_vs_change_fidelity"]
        print(
            f"  {a:36} applied={r['n_candidates_where_mask_was_APPLIED']:>2}"
            f"/{r['n_candidates_scored']} "
            f"perfect={r['n_perfect_fidelity_candidates']} "
            f"perfect_plannable={r['n_perfect_fidelity_candidates_that_are_plannable']} "
            f"AUC={au['auc']} CI={au.get('bootstrap_ci_95')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
