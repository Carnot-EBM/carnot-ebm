"""Analyse the object-perception A/B. Clustered at the GAME level, exactly as pre-registered.

THE PSEUDO-REPLICATION TRAP THIS AVOIDS. 20 games x 3 replicates is 120 cells but only 20
INDEPENDENT UNITS: the replicates re-sample the same generator on the same prompt for the
same game, so they estimate within-game noise, not between-game variation. Treating them as
120 trials inflated a p from 0.125 to 0.049 on 2026-07-31 and had to be corrected. Replicates
are therefore AVERAGED into one per-game mean per arm BEFORE anything is paired, and only
replicates present in BOTH arms are averaged (an unmatched replicate would let the arms be
compared on different samples).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

PRIMARY = "change_fidelity"
SECONDARIES = [
    "accuracy",
    "cell_recall",
    "change_accuracy",
    "correct_changed_cells",
    "spurious_changed_cells",
    "n_changes_correct",
    # ---- ADDED 2026-08-01 after adversarial review, from `rescore.json` -----------------
    # The six above omit EVERY channel that names the one engine `change_fidelity` is
    # structurally blind to: correct on every CHANGING transition, hallucinating a change on
    # every NO-OP. Reproduced on frozen split data before being acted on -- on sc25 that
    # engine scores change_fidelity 1.0000 at full-grid accuracy 0.0714, and
    # `spurious_changed_cells` reads a clean 0 because it only accumulates INSIDE changing
    # transitions. These are field copies out of the same `VerifyResult`, not new compute.
    "n_noop",
    "n_noop_hallucinated",
    "noop_hallucination_rate",
    "invented_changed_cells",
]

# Games whose held-out tail grades exactly ONE changing transition, so their per-game mean IS
# that single transition. They stay in the pre-registered primary (dropping them after seeing
# outcomes would be a post-hoc roster edit) but they are NAMED, and the >=3-row sensitivity
# roster that excludes them is reported as a CO-PRIMARY rather than a footnote.
SINGLE_ROW_GAMES_NOTE = (
    "per-game mean rests on ONE graded transition; see sensitivity_well_supported_roster"
)


def merge_rescore(rows: list[dict], out_dir: Path) -> dict:
    """Fold `rescore.json`'s added channels into each row's `heldout` dict.

    GATED ON THE REPRODUCTION CHECK. `rescore.py` re-derives the two fields run_ab.py already
    recorded from a rebuilt window. If any cell disagrees, the rebuilt window is not the window
    that was graded, and every added field is void -- so nothing is merged at all rather than
    some cells silently carrying fields derived from a different split.
    """
    p = out_dir / "rescore.json"
    if not p.exists():
        return {"merged": False, "reason": "rescore.json absent"}
    rs = json.loads(p.read_text())
    check = rs.get("reproduction_check", {})
    if not check.get("all_reproduce_run_ab_change_fidelity"):
        return {"merged": False, "reason": "reproduction check FAILED", "check": check}
    by = {c["cell"]: c for c in rs.get("cells", []) if c.get("status") == "ok"}
    n = 0
    for r in rows:
        c = by.get(f"{r['game']}__r{r['replicate']}__{r['tag']}")
        if not c:
            continue
        h = r.get("heldout") or {}
        if not h.get("measurable"):
            continue
        for k in (
            "n_noop",
            "n_noop_hallucinated",
            "noop_hallucination_rate",
            "noop_channel_measurable",
            "invented_changed_cells",
            "invented_change_rate",
            "hud_mask_status",
        ):
            h[k] = c["full"][k]
        h["behaviourally_blind"] = c.get("behaviourally_blind")
        h["reads_data_param"] = c.get("reads_data_param")
        r["heldout"] = h
        n += 1
    return {
        "merged": True,
        "n_cells_merged": n,
        "reproduction_check": check,
        "per_game_disqualifier_checks": rs.get("per_game_disqualifier_checks", {}),
        "baseline_summary": {
            g: {
                "identity": b["baselines"]["IDENTITY"]["change_fidelity"],
                "modal_shown_delta_replay": b["baselines"]["MODAL_SHOWN_DELTA_REPLAY"][
                    "change_fidelity"
                ],
                "oracle_ceiling": b["baselines"]["ORACLE_ceiling"]["change_fidelity"],
            }
            for g, b in rs.get("baselines", {}).items()
            if b.get("status") == "ok"
        },
        "noop_channel_roster_wide": rs.get("noop_channel_roster_wide", {}),
    }


def _binom_tail_ge(k: int, n: int) -> float:
    return sum(comb(n, i) for i in range(k, n + 1)) / (2.0**n)


def min_reachable_two_sided_p(n_disc: int) -> float:
    return 1.0 if n_disc <= 0 else min(1.0, 2.0 * (0.5**n_disc))


def sign_test(deltas):
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    nd = pos + neg
    if nd == 0:
        return {
            "n_pairs": len(deltas),
            "n_positive": pos,
            "n_negative": neg,
            "n_ties": ties,
            "n_discordant": 0,
            "p_two_sided": 1.0,
            "test_was_possible": False,
            "min_reachable_two_sided_p_at_this_discordance": 1.0,
        }
    p = min(1.0, 2.0 * _binom_tail_ge(max(pos, neg), nd))
    return {
        "n_pairs": len(deltas),
        "n_positive": pos,
        "n_negative": neg,
        "n_ties": ties,
        "n_discordant": nd,
        "p_two_sided": round(p, 8),
        "test_was_possible": True,
        "min_reachable_two_sided_p_at_this_discordance": round(min_reachable_two_sided_p(nd), 12),
    }


def signflip_exact(deltas, max_n: int = 22):
    """Exact two-sided sign-flip (randomisation) test on the MAGNITUDES. Uses the effect
    sizes the sign test throws away, so a consistent tiny effect and a single huge one are
    distinguishable."""
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return {"n_nonzero": 0, "p_two_sided": 1.0, "test_was_possible": False}
    obs = abs(sum(nz) / n)
    if n > max_n:
        return {
            "n_nonzero": n,
            "test_was_possible": False,
            "note": f"exact enumeration capped at {max_n}",
        }
    count = 0
    for mask in range(1 << n):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(nz))
        if abs(s / n) >= obs - 1e-15:
            count += 1
    return {
        "n_nonzero": n,
        "observed_mean": round(sum(nz) / n, 8),
        "p_two_sided": round(count / (1 << n), 8),
        "test_was_possible": True,
        "n_enumerated": 1 << n,
    }


def bootstrap_ci(deltas, n_resamples: int = 20000, seed: int = 6100):
    import random

    if not deltas:
        return {"n": 0}
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(n_resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": round(sum(deltas) / n, 8),
        "lo": round(means[int(0.025 * n_resamples)], 8),
        "hi": round(means[int(0.975 * n_resamples)], 8),
        "n": n,
        "n_resamples": n_resamples,
        "alpha": 0.05,
    }


POOL_TRUNCATION = "TRUNCATED BY SHARED CONTEXT POOL"
BUDGET_LIMIT = "HIT n_predict"


def is_missing(r) -> tuple[bool, str | None]:
    """MISSING (absent measurement) vs ZERO (real failure of the treatment).

    THE ASYMMETRY THIS EXISTS FOR. The ON arm's prompt is 1382-7389 characters LONGER than
    the OFF arm's -- that is the treatment. If a long prompt eats the server's shared context
    pool, llama-server returns HTTP 200 with a SILENTLY TRUNCATED completion, which the
    proposer records as a CONTENT failure, not a server failure. Scoring those as 0.0 would
    penalise the treatment arm in proportion to how much treatment it received: a confound
    pointing the wrong way, and invisible in the headline number.

    So pool truncation is MISSING. A budget limit (the model generated its whole max_tokens
    and was still rambling) is NOT missing -- the infrastructure did its job and the model
    failed to write a compact engine, which is a real zero. `_limit_diagnostic()` emits
    distinct strings for the two, and `induce_msg` carries them.
    """
    if r.get("server_failures_delta", 0) > 0:
        return True, "server_failure"
    if r.get("exception"):
        return True, "harness_exception"
    msg = r.get("induce_msg") or ""
    if POOL_TRUNCATION in msg:
        return True, "shared_context_pool_truncation"
    return False, None


def failure_asymmetry(rows) -> dict:
    """Per-arm induction-failure accounting, so a difference in FAILURE RATE cannot hide
    inside a difference in SCORE."""
    out = {}
    for arm in ("off", "on"):
        sel = [r for r in rows if r.get("tag") == arm]
        out[arm] = {
            "n_cells": len(sel),
            "n_induce_ok": sum(1 for r in sel if r.get("induce_ok")),
            "n_engine_loaded": sum(1 for r in sel if r.get("engine_loaded")),
            "n_content_failure_cells": sum(
                1 for r in sel if r.get("content_failures_delta", 0) > 0
            ),
            "n_server_failure_cells": sum(1 for r in sel if r.get("server_failures_delta", 0) > 0),
            "n_pool_truncation": sum(
                1 for r in sel if POOL_TRUNCATION in (r.get("induce_msg") or "")
            ),
            "n_budget_limit": sum(1 for r in sel if BUDGET_LIMIT in (r.get("induce_msg") or "")),
            "mean_prompt_chars": round(sum(r["prompt_chars"] for r in sel) / len(sel), 1)
            if sel
            else None,
            "max_prompt_chars": max((r["prompt_chars"] for r in sel), default=None),
        }
    out["reading"] = (
        "n_pool_truncation must be 0 in BOTH arms for the scores to be comparable; a nonzero "
        "count in the ON arm alone would mean the treatment was penalised for being longer. "
        "n_budget_limit is a real model failure and is scored as a zero, not excluded."
    )
    return out


def per_game_pairs(rows, field, roster):
    """(deltas, per_game) with replicates averaged inside each game, matched replicates only."""
    by = defaultdict(dict)  # game -> (rep, arm) -> value
    excluded = []
    for r in rows:
        if r.get("tag") not in ("off", "on"):
            continue  # A/A cells are not part of the contrast
        miss, why = is_missing(r)
        if miss:
            excluded.append(
                {"game": r["game"], "replicate": r["replicate"], "arm": r["arm"], "reason": why}
            )
            continue
        h = r.get("heldout") or {}
        if not h.get("measurable"):
            excluded.append(
                {
                    "game": r["game"],
                    "replicate": r["replicate"],
                    "arm": r["arm"],
                    "reason": "not_measurable",
                }
            )
            continue
        v = h.get(field)
        if v is None:
            continue
        by[r["game"]][(r["replicate"], r["arm"])] = float(v)

    deltas, per_game = [], {}
    for game in roster:
        cells = by.get(game, {})
        reps = sorted({rep for (rep, _a) in cells})
        matched = [rep for rep in reps if (rep, "off") in cells and (rep, "on") in cells]
        if not matched:
            per_game[game] = {"n_matched_replicates": 0, "excluded_from_pairing": True}
            continue
        off = statistics.mean(cells[(rep, "off")] for rep in matched)
        on = statistics.mean(cells[(rep, "on")] for rep in matched)
        per_game[game] = {
            "off": round(off, 6),
            "on": round(on, 6),
            "delta": round(on - off, 6),
            "n_matched_replicates": len(matched),
            "matched_replicates": matched,
            "off_per_replicate": [round(cells[(r_, "off")], 6) for r_ in matched],
            "on_per_replicate": [round(cells[(r_, "on")], 6) for r_ in matched],
        }
        deltas.append(on - off)
    return deltas, per_game, excluded


def within_arm_replicate_noise(rows, field=PRIMARY) -> dict:
    """How much does the SAME arm on the SAME game move between replicates?

    THE NUMBER THE EFFECT MUST BEAT. Each replicate re-runs an IDENTICAL prompt in an
    IDENTICAL arm under a different generator seed, so the spread between replicates is pure
    generator variability with the treatment held fixed. If the between-ARM delta is smaller
    than the within-ARM spread, the design cannot separate the treatment from resampling the
    generator, whatever the sign test says -- and a reader deciding whether to flip a shipped
    default needs that comparison stated, not left to be inferred from a CI.

    This is DISTINCT from the A/A control, and both are needed. A/A repeats the same arm at
    the SAME seed and asks "is the pipeline deterministic". This asks "how big is the
    seed-to-seed swing", which is the noise floor an effect has to clear. A/A can be perfectly
    byte-identical while this is enormous.
    """
    by = defaultdict(dict)
    for r in rows:
        if r.get("tag") not in ("off", "on"):
            continue
        h = r.get("heldout") or {}
        if not h.get("measurable"):
            continue
        v = h.get(field)
        if v is not None:
            by[(r["game"], r["tag"])][r["replicate"]] = float(v)
    spreads, sds = [], []
    for vals in by.values():
        xs = list(vals.values())
        if len(xs) >= 2:
            spreads.append(max(xs) - min(xs))
            sds.append(statistics.pstdev(xs))
    if not spreads:
        return {"computed": False, "why": "no (game, arm) cell has >=2 replicates yet"}
    return {
        "computed": True,
        "field": field,
        "n_game_arm_cells_with_2plus_replicates": len(spreads),
        "mean_spread": round(statistics.mean(spreads), 6),
        "median_spread": round(statistics.median(spreads), 6),
        "max_spread": round(max(spreads), 6),
        "n_cells_with_zero_spread": sum(1 for x in spreads if x == 0),
        "mean_within_cell_sd": round(statistics.mean(sds), 6),
        "reading": (
            "compare mean_spread against the primary's mean_delta_over_games. If the noise "
            "floor is the larger of the two, a per-game sign flip is as easily a reseed as a "
            "treatment effect, and the honest conclusion is 'not separable at this n' rather "
            "than 'no effect'."
        ),
    }


def dose_response(per_game: dict, out_dir: Path) -> dict:
    """POST-HOC, EXPLORATORY, NOT PRE-REGISTERED: does MORE treatment produce MORE effect?

    The treatment is an inserted block of text whose size varies 5x across the roster
    (1382-7389 characters, measured by the independent witness). If the object block works by
    telling the model something useful about the game, a bigger block on a more
    object-rich board is more of the thing that works, so the per-game delta should have SOME
    relationship with block size. If instead the per-game deltas are sampler noise, block size
    should predict nothing.

    Neither direction is conclusive on its own -- a real effect can saturate, and a null
    correlation over ~20 games is weak evidence. It is reported because it is nearly free and
    because it is the only handle this design has on "is the delta a mechanism or is it
    noise" that does not require more compute. Labelled exploratory so it is never read as a
    pre-registered result.
    """
    p = out_dir / "treatment_witness_indep.json"
    if not p.exists():
        return {"computed": False, "why": "independent witness (block sizes) not available"}
    sizes = {
        r["game"]: r["inserted_span_chars"]
        for r in json.loads(p.read_text())
        if r.get("status") == "ok"
    }
    pts = [
        (sizes[g], v["delta"])
        for g, v in per_game.items()
        if g in sizes and isinstance(v.get("delta"), (int, float))
    ]
    if len(pts) < 4:
        return {"computed": False, "why": f"only {len(pts)} paired games with a known block size"}
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]

    def _pearson(u, v):
        mu, mv = statistics.mean(u), statistics.mean(v)
        num = sum((a - mu) * (b - mv) for a, b in zip(u, v, strict=False))
        den = (sum((a - mu) ** 2 for a in u) * sum((b - mv) ** 2 for b in v)) ** 0.5
        return round(num / den, 4) if den else None

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for k, i in enumerate(order):
            out[i] = k
        return out

    return {
        "computed": True,
        "PRE_REGISTERED": False,
        "n_games": len(pts),
        "block_chars_min": min(xs),
        "block_chars_max": max(xs),
        "pearson_r_blocksize_vs_delta": _pearson(xs, ys),
        "spearman_rho_blocksize_vs_delta": _pearson(_rank(xs), _rank(ys)),
        "mean_delta_large_blocks_ge_4000": round(
            statistics.mean([b for a, b in pts if a >= 4000]), 6
        )
        if any(a >= 4000 for a, _ in pts)
        else None,
        "mean_delta_small_blocks_lt_4000": round(
            statistics.mean([b for a, b in pts if a < 4000]), 6
        )
        if any(a < 4000 for a, _ in pts)
        else None,
        "reading": (
            "a correlation near zero means the amount of treatment does not predict the size "
            "of the effect, which is what noise looks like and is not what a working mechanism "
            "usually looks like. It is suggestive, not decisive: a real effect can saturate."
        ),
    }


def analyse_field(rows, field, roster):
    deltas, per_game, excluded = per_game_pairs(rows, field, roster)
    vals = []
    for g in per_game.values():
        if "off" in g:
            vals += [g["off"], g["on"]]
    return {
        "field": field,
        "games_paired": [g for g, v in per_game.items() if "off" in v],
        "n_games_paired": len(deltas),
        "per_game": per_game,
        "mean_delta_over_games": round(sum(deltas) / len(deltas), 8) if deltas else None,
        "sign_test": sign_test(deltas),
        "signflip_test": signflip_exact(deltas),
        "bootstrap_ci_over_games": bootstrap_ci(deltas),
        "all_values_zero_both_arms": bool(vals) and all(v == 0 for v in vals),
        "n_distinct_values": len(set(vals)),
        "excluded_cells": excluded,
    }


def main() -> int:
    rows = json.loads((OUT / "rows.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text()) if (OUT / "meta.json").exists() else {}
    roster = meta.get("roster") or sorted({r["game"] for r in rows})
    well = meta.get("roster_well_supported", [])
    rescore_meta = merge_rescore(rows, OUT)

    res = {
        "rescore_merge": rescore_meta,
        "n_rows": len(rows),
        "n_missing": sum(1 for r in rows if is_missing(r)[0]),
        "missing_detail": [
            {
                "game": r["game"],
                "replicate": r["replicate"],
                "arm": r["arm"],
                "reason": is_missing(r)[1],
                "induce_msg": (r.get("induce_msg") or "")[:200],
            }
            for r in rows
            if is_missing(r)[0]
        ],
        "failure_asymmetry_by_arm": failure_asymmetry(rows),
        "arm_prompt_consistent_all": all(
            r.get("arm_prompt_consistent", False)
            for r in rows
            if r.get("tag") in ("off", "on", "offAA")
        ),
        "induce_ok": {
            arm: {
                "n": sum(1 for r in rows if r.get("tag") == arm),
                "n_ok": sum(1 for r in rows if r.get("tag") == arm and r.get("induce_ok")),
                "n_engine_loaded": sum(
                    1 for r in rows if r.get("tag") == arm and r.get("engine_loaded")
                ),
            }
            for arm in ("off", "on")
        },
        "PRIMARY": analyse_field(rows, PRIMARY, roster),
        "secondaries": {f: analyse_field(rows, f, roster) for f in SECONDARIES},
        "sensitivity_well_supported_roster": (analyse_field(rows, PRIMARY, well) if well else None),
    }

    res["within_arm_replicate_noise"] = within_arm_replicate_noise(rows)
    res["dose_response_EXPLORATORY"] = dose_response(res["PRIMARY"]["per_game"], OUT)

    # ---- CO-PRIMARY, promoted from a footnote after adversarial review 2026-08-01 --------
    # Four roster games (tn36, cd82, su15, lp85) grade exactly ONE changing held-out
    # transition each, so their per-game mean IS that single transition. They stay in the
    # pre-registered primary -- removing them after seeing outcomes would be a post-hoc roster
    # edit, which is the thing pre-registration exists to prevent -- but the >=3-row roster
    # that excludes them is reported at equal prominence, and the two are compared explicitly.
    # If they disagree in DIRECTION, the primary is being carried by single transitions and
    # the artifact must say so.
    single_row = sorted(
        g
        for g, m in (meta.get("split_meta") or {}).items()
        if g in roster and int(m.get("heldout_gradable_changing", 0)) <= 1
    )
    s = res["sensitivity_well_supported_roster"]
    pm = res["PRIMARY"]["mean_delta_over_games"]
    sm = s["mean_delta_over_games"] if s else None
    res["CO_PRIMARY_roster_comparison"] = {
        "why": (
            "the pre-registered primary roster includes games whose per-game mean rests on a "
            "single graded transition; the >=3-gradable-row roster does not. Reporting only "
            "the first would let one transition per game carry a headline."
        ),
        "single_gradable_row_games_in_primary": single_row,
        "n_single_gradable_row_games": len(single_row),
        "primary_roster_n_games": res["PRIMARY"]["n_games_paired"],
        "primary_mean_delta": pm,
        "primary_p": res["PRIMARY"]["sign_test"]["p_two_sided"],
        "sensitivity_roster_n_games": (s["n_games_paired"] if s else None),
        "sensitivity_mean_delta": sm,
        "sensitivity_p": (s["sign_test"]["p_two_sided"] if s else None),
        "same_direction": (
            None
            if (pm is None or sm is None)
            else bool((pm > 0) == (sm > 0) or (pm == 0 and sm == 0))
        ),
        "both_null_at_alpha_0.05": (
            None
            if s is None
            else bool(
                res["PRIMARY"]["sign_test"]["p_two_sided"] >= 0.05
                and s["sign_test"]["p_two_sided"] >= 0.05
            )
        ),
    }

    # ---- A/A control: same arm, same seed, re-run. Tests the sampler seeding actually
    # removed the documented 40% run-to-run divergence. If it did not, that divergence is a
    # FLOOR under any effect claim and is reported as such rather than assumed away.
    aa = []
    for r in rows:
        if r.get("tag") != "offAA":
            continue
        base = next(
            (
                x
                for x in rows
                if x["game"] == r["game"]
                and x["replicate"] == r["replicate"]
                and x.get("tag") == "off"
            ),
            None,
        )
        if base is None:
            continue
        aa.append(
            {
                "game": r["game"],
                "seed": r["seed"],
                "prompt_identical": base["prompt_sha256"] == r["prompt_sha256"],
                "engine_identical": base.get("engine_sha256") == r.get("engine_sha256"),
                "base_engine_sha256": base.get("engine_sha256"),
                "repeat_engine_sha256": r.get("engine_sha256"),
                "base_change_fidelity": (base.get("heldout") or {}).get(PRIMARY),
                "repeat_change_fidelity": (r.get("heldout") or {}).get(PRIMARY),
            }
        )
    n_ident = sum(1 for a in aa if a["engine_identical"])
    res["AA_control"] = {
        "n": len(aa),
        "n_engine_byte_identical": n_ident,
        "all_prompts_identical": all(a["prompt_identical"] for a in aa) if aa else None,
        "rows": aa,
        "reading": (
            "byte-identical engines under a repeated arm+seed means the seeding removed the "
            "documented run-to-run nondeterminism, so an OFF-vs-ON difference is attributable "
            "to the treatment"
            if aa and n_ident == len(aa)
            else "NOT all byte-identical: residual nondeterminism survives the seed, and it is a "
            "FLOOR under any effect claim -- an observed per-game flip may be sampler noise"
        ),
    }

    # ---- THE FLOOR THE PRE-REGISTRATION DEMANDS ----------------------------------------
    # prereg.AA_CONTROL: "If A/A is not byte-identical the residual nondeterminism is reported
    # as a FLOOR on any effect claim." It was not byte-identical, so the floor is computed
    # here rather than left for a reader to infer -- and it is computed as a DISTRIBUTION
    # comparison, because "the A/A moved at all" and "the A/A moved as much as the treatment"
    # are completely different verdicts on the same failed check.
    aa_abs = [
        abs((r["repeat_change_fidelity"] or 0) - (r["base_change_fidelity"] or 0)) for r in aa
    ]
    tr_abs = [abs(v["delta"]) for v in res["PRIMARY"]["per_game"].values() if "delta" in v]
    res["AA_FLOOR_vs_EFFECT"] = {
        "why": (
            "the A/A control repeats the SAME arm at the SAME seed, so any movement is "
            "residual nondeterminism with the treatment held fixed. It is the floor an effect "
            "must clear to be attributable to the treatment at all."
        ),
        "n_aa_pairs": len(aa_abs),
        "n_aa_engine_byte_identical": sum(1 for r in aa if r["engine_identical"]),
        "aa_abs_delta_values": [round(x, 6) for x in aa_abs],
        "aa_abs_delta_mean": round(statistics.mean(aa_abs), 6) if aa_abs else None,
        "aa_abs_delta_max": round(max(aa_abs), 6) if aa_abs else None,
        "treatment_abs_delta_mean": round(statistics.mean(tr_abs), 6) if tr_abs else None,
        "treatment_abs_delta_max": round(max(tr_abs), 6) if tr_abs else None,
        "n_games_whose_abs_delta_exceeds_the_aa_max": (
            sum(1 for x in tr_abs if x > max(aa_abs)) if aa_abs and tr_abs else None
        ),
        "n_games_total": len(tr_abs),
        "effect_clears_the_determinism_floor": (
            bool(statistics.mean(tr_abs) > max(aa_abs)) if aa_abs and tr_abs else None
        ),
        "reading": (
            "a mean per-game |delta| several times the A/A maximum means the observed movement "
            "is larger than the pipeline's own irreproducibility, so the sign test is not "
            "measuring nondeterminism. It does NOT make the A/A failure harmless: engines that "
            "are not byte-identical at a fixed seed mean this experiment is not exactly "
            "reproducible, and a re-run will not return these numbers to the last digit."
        ),
    }

    (OUT / "analysis.json").write_text(json.dumps(res, indent=2))

    p = res["PRIMARY"]
    st = p["sign_test"]
    print("=" * 72)
    print(f"PRIMARY: {PRIMARY}  (pre-registered)")
    print(f"  games paired         : {p['n_games_paired']}")
    print(f"  mean delta (on-off)  : {p['mean_delta_over_games']}")
    print(
        f"  sign test            : +{st['n_positive']} / -{st['n_negative']} / "
        f"={st['n_ties']}  discordant={st['n_discordant']}"
    )
    print(f"  p (two-sided)        : {st['p_two_sided']}   test_possible={st['test_was_possible']}")
    print(f"  min reachable p here : {st['min_reachable_two_sided_p_at_this_discordance']}")
    print(f"  signflip p           : {p['signflip_test'].get('p_two_sided')}")
    print(f"  bootstrap CI         : {p['bootstrap_ci_over_games']}")
    print(
        f"  floored both arms?   : {p['all_values_zero_both_arms']}  "
        f"(distinct values {p['n_distinct_values']})"
    )
    print(f"  missing cells        : {res['n_missing']}")
    s = res["sensitivity_well_supported_roster"]
    if s:
        print(
            f"CO-PRIMARY (>=3 gradable rows, {s['n_games_paired']} games): "
            f"delta={s['mean_delta_over_games']} p={s['sign_test']['p_two_sided']} "
            f"disc={s['sign_test']['n_discordant']}"
        )
    cp = res["CO_PRIMARY_roster_comparison"]
    print(
        f"  single-gradable-row games in primary: {cp['n_single_gradable_row_games']} "
        f"{cp['single_gradable_row_games_in_primary']}  same_direction={cp['same_direction']}"
    )
    wn = res["within_arm_replicate_noise"]
    if wn.get("computed"):
        pmd = res["PRIMARY"]["mean_delta_over_games"]
        print(
            f"NOISE FLOOR: within-arm replicate spread mean={wn['mean_spread']} "
            f"median={wn['median_spread']} max={wn['max_spread']} "
            f"over {wn['n_game_arm_cells_with_2plus_replicates']} (game,arm) cells "
            f"-- vs between-arm mean delta {pmd}"
        )
    dr = res["dose_response_EXPLORATORY"]
    if dr.get("computed"):
        print(
            f"DOSE-RESPONSE (exploratory, not pre-registered): "
            f"pearson={dr['pearson_r_blocksize_vs_delta']} "
            f"spearman={dr['spearman_rho_blocksize_vs_delta']} over {dr['n_games']} games "
            f"(block {dr['block_chars_min']}-{dr['block_chars_max']} chars)"
        )
    dq = rescore_meta.get("per_game_disqualifier_checks") or {}
    if dq:
        bad = [g for g, v in dq.items() if v["a_non_model_outranks_every_real_engine"]]
        blindwin = [g for g, v in dq.items() if v["blind_outranks_aware"]]
        meas = sum(1 for v in dq.values() if v["noop_channel_measurable"])
        print("DISQUALIFIER CHECKS (the criterion the object metrics were rejected on):")
        print(f"  games where a NON-MODEL outranks every real engine : {len(bad)} {bad}")
        print(f"  games where a BLIND engine outranks an aware one   : {len(blindwin)} {blindwin}")
        print(f"  games where the no-op channel is MEASURABLE at all : {meas}/{len(dq)}")
    print(
        f"A/A control: {res['AA_control']['n_engine_byte_identical']}"
        f"/{res['AA_control']['n']} byte-identical"
    )
    f = res["AA_FLOOR_vs_EFFECT"]
    print(
        f"  A/A |delta| mean={f['aa_abs_delta_mean']} max={f['aa_abs_delta_max']}  vs  "
        f"treatment |delta| mean={f['treatment_abs_delta_mean']} max={f['treatment_abs_delta_max']}"
    )
    print(
        f"  games whose |delta| exceeds the A/A max: "
        f"{f['n_games_whose_abs_delta_exceeds_the_aa_max']}/{f['n_games_total']}  "
        f"clears_floor={f['effect_clears_the_determinism_floor']}"
    )
    print("SECONDARIES (exploratory, Bonferroni 0.00833):")
    for f, r in res["secondaries"].items():
        print(
            f"  {f:24} delta={str(r['mean_delta_over_games']):>12} "
            f"p={r['sign_test']['p_two_sided']:<10} disc={r['sign_test']['n_discordant']:<3} "
            f"floored={r['all_values_zero_both_arms']}"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
