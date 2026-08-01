"""Analyse the inert-rejection A/B exactly as pre-registered. Runs ONCE, after collection stops.

Every test here clusters at the GAME: replicates within a game are averaged into ONE per-game
mean before pairing, so 20 games x 4 replicates is 20 independent units, not 80. Treating
replicates as trials inflated a p from 0.125 to 0.049 on 2026-07-31 and had to be corrected.

Only (game, replicate) pairs where BOTH arms produced a non-missing observation enter the
analysis. A server failure, a harness exception, or a scoring worker that had to be killed is a
MISSING OBSERVATION and is counted and reported -- never scored as a zero.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def binom_tail_ge(k: int, n: int) -> float:
    return sum(comb(n, i) for i in range(k, n + 1)) / (2.0**n)


def min_reachable_two_sided_p(n_disc: int) -> float:
    if n_disc <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**n_disc))


def sign_test(deltas: dict[str, float]) -> dict:
    """Exact two-sided paired sign test over games, with the reachability floor reported.

    `min_reachable_two_sided_p_at_this_discordance` is the smallest p this comparison COULD have
    returned given how many pairs were discordant. When it exceeds alpha, a p above alpha says
    the design could not have detected the effect, which is a different statement from "there is
    no effect" and must not be reported as the same one.
    """
    vals = list(deltas.values())
    pos = sum(1 for d in vals if d > 0)
    neg = sum(1 for d in vals if d < 0)
    ties = sum(1 for d in vals if d == 0)
    nd = pos + neg
    if nd == 0:
        return {
            "n_pairs": len(vals),
            "n_positive": 0,
            "n_negative": 0,
            "n_ties": ties,
            "n_discordant": 0,
            "p_two_sided": 1.0,
            "test_was_possible": False,
            "min_reachable_two_sided_p_at_this_discordance": 1.0,
            "mean_delta": round(statistics.fmean(vals), 6) if vals else None,
            "per_game_delta": {k: round(v, 6) for k, v in sorted(deltas.items())},
        }
    p = min(1.0, 2.0 * binom_tail_ge(max(pos, neg), nd))
    return {
        "n_pairs": len(vals),
        "n_positive": pos,
        "n_negative": neg,
        "n_ties": ties,
        "n_discordant": nd,
        "p_two_sided": round(p, 8),
        "test_was_possible": True,
        "min_reachable_two_sided_p_at_this_discordance": round(min_reachable_two_sided_p(nd), 10),
        "mean_delta": round(statistics.fmean(vals), 6),
        "per_game_delta": {k: round(v, 6) for k, v in sorted(deltas.items())},
    }


def per_game_means(cells: dict, metric, arms=("off", "on")) -> tuple[dict, dict]:
    """(deltas, detail). A game contributes only replicates where BOTH arms are measurable."""
    by_game: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"off": [], "on": []})
    used: dict[str, list[int]] = defaultdict(list)
    for (game, rep), pair in sorted(cells.items()):
        if not all(a in pair for a in arms):
            continue
        vals = {a: metric(pair[a]) for a in arms}
        if any(v is None for v in vals.values()):
            continue
        for a in arms:
            by_game[game][a].append(float(vals[a]))
        used[game].append(rep)
    deltas, detail = {}, {}
    for game, d in by_game.items():
        if not d["off"]:
            continue
        m_off, m_on = statistics.fmean(d["off"]), statistics.fmean(d["on"])
        deltas[game] = m_on - m_off
        detail[game] = {
            "n_replicates_used": len(d["off"]),
            "replicates_used": sorted(used[game]),
            "mean_off": round(m_off, 6),
            "mean_on": round(m_on, 6),
            "delta_on_minus_off": round(m_on - m_off, 6),
        }
    return deltas, detail


def main() -> int:  # noqa: C901
    rows = json.loads((OUT / "rows.json").read_text())
    scored = {r["cell_id"]: r for r in json.loads((OUT / "scored.json").read_text())}
    meta = json.loads((OUT / "meta.json").read_text())
    prereg = json.loads((OUT / "preregistration.json").read_text())

    # --- attach scoring, decide missingness -------------------------------------------
    for r in rows:
        cid = f"{r['game']}__r{r['replicate']}__{r['tag']}"
        s = scored.get(cid, {})
        r["score_status"] = s.get("status")
        r["heldout"] = s.get("heldout")
        r["state_graph"] = s.get("state_graph")
        # A scoring worker killed at its bound is a MISSING observation for the DOWNSTREAM
        # metrics only. The PRIMARY was computed at collection time by a check that is itself
        # subprocess-bounded, so it is still a real observation.
        r["score_missing"] = s.get("status") not in ("ok",)

    aa = [r for r in rows if r["tag"] == "offAA"]
    ab = [r for r in rows if r["tag"] in ("off", "on")]

    cells: dict[tuple[str, int], dict] = defaultdict(dict)
    for r in ab:
        cells[(r["game"], r["replicate"])][r["arm"]] = r

    complete = {k: v for k, v in cells.items() if "off" in v and "on" in v}

    # --- A/A NONDETERMINISM FLOOR ------------------------------------------------------
    # Seeding is a claim about the SAMPLER, not about the server. Without this floor an effect
    # is uninterpretable: the 2026-08-01 sibling run's A/A matched on only 1 of 4 cells.
    aa_rows = []
    for r in aa:
        base = cells.get((r["game"], 0), {}).get("off")
        if base is None:
            continue
        aa_rows.append(
            {
                "game": r["game"],
                "engine_sha_a": base.get("engine_sha256"),
                "engine_sha_aa": r.get("engine_sha256"),
                "byte_identical": base.get("engine_sha256") == r.get("engine_sha256"),
                "usable_a": base.get("usable"),
                "usable_aa": r.get("usable"),
                "usable_agrees": base.get("usable") == r.get("usable"),
                "inert_a": base.get("engine_inert"),
                "inert_aa": r.get("engine_inert"),
            }
        )
    n_aa = len(aa_rows)
    aa_summary = {
        "n_cells": n_aa,
        "n_byte_identical": sum(1 for a in aa_rows if a["byte_identical"]),
        "n_primary_agrees": sum(1 for a in aa_rows if a["usable_agrees"]),
        "byte_identical_rate": round(sum(a["byte_identical"] for a in aa_rows) / n_aa, 4)
        if n_aa
        else None,
        "primary_agreement_rate": round(sum(a["usable_agrees"] for a in aa_rows) / n_aa, 4)
        if n_aa
        else None,
        "rows": aa_rows,
        "how_to_read": "byte-identical means the seeded sampler reproduced exactly. Where it does "
        "not, the PRIMARY may still agree -- what matters for a yield claim is "
        "whether the same seed reaches the same usable/not verdict, not whether it "
        "emits the same bytes. Both are reported; the weaker one is the floor.",
        "direction_of_bias": "CONSERVATIVE. The A/A cells run at the END of collection, hours "
        "after the rep-0 control they are compared against and with a "
        "completely different server KV-cache state, so they are if "
        "anything HARDER to reproduce than a back-to-back repeat would be. "
        "That means this floor may OVERSTATE the nondeterminism, which is "
        "the safe direction for a floor: it cannot make an effect look more "
        "real than it is.",
        "reference_point": "The 2026-08-01 object-perception A/B, on the same generator and the "
        "same seeding scheme, came back byte-identical on 1 of 4 A/A cells "
        "(verified directly from its rows.json, not quoted from memory).",
    }

    # --- MECHANISTIC WITNESS -----------------------------------------------------------
    # THE DESIGN ASSUMPTION THIS WITNESS WAS WRITTEN UNDER TURNED OUT TO BE FALSE, and saying so
    # is more useful than the witness itself. The pre-registration expected that with the sampler
    # seeded, the two arms would draw the SAME attempt-0 completion and could therefore only
    # diverge where the treatment fired. Within the first three (game, replicate) pairs -- ls20,
    # s5i5, tu93 -- all three produced DIFFERENT engines across arms at byte-identical prompt
    # hashes, identical seeds, and exactly one completion call each. So they did not diverge
    # because of the treatment; they diverged because the generator did not reproduce.
    #
    # A direct probe of the same live server (two identical short `/completion` requests at a
    # fixed seed) returned byte-identical output, and the server runs a single slot with no
    # `--parallel`, so this is not batch nondeterminism and the seed IS honoured. The remaining
    # candidate is `cache_prompt: true`: reusing a KV prefix computed under a different preceding
    # state changes GEMM shapes and therefore floating-point accumulation order, which at
    # temperature 0.2 is enough to flip a near-tied token. Recorded as the leading hypothesis, NOT
    # as an established cause -- it is not tested here.
    #
    # WHAT IT MEANS FOR THE RESULT. The experiment is still a valid RANDOMIZED A/B: both arms draw
    # from the same distribution under the same conditions, and the (game, replicate) pairing is
    # a blocking factor rather than a matched sample. It is NOT the tighter matched design the
    # pre-registration described, so it is noisier and the effective power is lower than the
    # already-pessimistic estimate stated up front. `n_diverged_without_trigger` measures how
    # often this happened and is reported next to `n_fired` precisely so the two can be compared.
    witness = {"fired": [], "diverged_without_trigger": [], "trigger_but_no_reask": []}
    for (game, rep), pair in sorted(complete.items()):
        off, on = pair["off"], pair["on"]
        # TRIGGER = the treatment's OWN condition, which is `not defects and inert`, NOT merely
        # `inert`. The harness records `engine_inert` from `engine_changes_anything_bounded`,
        # which reports False for an engine that never RAN (raises on everything) as well as for
        # one that ran and changed nothing. `engine_inertness_defect` requires at least one usable
        # prediction before it fires, so a never-ran engine is NOT a trigger and its absence of a
        # re-ask is correct behaviour, not a treatment failure. Requiring `usable` here reproduces
        # that condition exactly: `_engine_defects` only reaches the inertness probe when
        # `validate_engine_code` came back empty. Found by an adversarial review pass; without it
        # the witness and the treatment could disagree and the disagreement would read as a bug.
        trigger = bool(off.get("usable")) and off.get("engine_inert") is True
        diverged = off.get("engine_sha256") != on.get("engine_sha256")
        reasked = (on.get("defect_reasks_delta") or 0) > (off.get("defect_reasks_delta") or 0)
        item = {
            "game": game,
            "replicate": rep,
            "control_engine_inert": trigger,
            "engines_diverged": diverged,
            "treatment_reasked_more": reasked,
            "off_usable": off.get("usable"),
            "on_usable": on.get("usable"),
            "off_live": off.get("live"),
            "on_live": on.get("live"),
            "off_defects": off.get("defect_kinds"),
            "on_defects": on.get("defect_kinds"),
        }
        if trigger and reasked:
            witness["fired"].append(item)
        elif trigger and not reasked:
            witness["trigger_but_no_reask"].append(item)
        elif diverged:
            witness["diverged_without_trigger"].append(item)
    witness["n_fired"] = len(witness["fired"])
    witness["n_diverged_without_trigger"] = len(witness["diverged_without_trigger"])
    witness["n_trigger_but_no_reask"] = len(witness["trigger_but_no_reask"])
    # HOW OFTEN DID THE TREATMENT ACT, measured on the arm it acts on. Because the arms do not
    # share a draw (see below), `n_fired` -- which keys on the CONTROL being inert -- is the wrong
    # denominator. A defect re-ask fires in BOTH arms for ordinary code defects; only the ON arm
    # can additionally re-ask for inertness. The DIFFERENCE in re-ask rate is therefore the
    # inertness-driven share, and it is the honest measure of how much treatment there was.
    reask_rate = {}
    for arm in ("off", "on"):
        sel = [r for r in ab if r["arm"] == arm and not r.get("missing")]
        n_re = sum(1 for r in sel if (r.get("defect_reasks_delta") or 0) > 0)
        reask_rate[arm] = {
            "n_cells": len(sel),
            "n_cells_with_a_reask": n_re,
            "rate": round(n_re / len(sel), 4) if sel else None,
        }
    if reask_rate["off"]["rate"] is not None and reask_rate["on"]["rate"] is not None:
        reask_rate["excess_reask_rate_attributable_to_inertness"] = round(
            reask_rate["on"]["rate"] - reask_rate["off"]["rate"], 4
        )
    reask_rate["how_to_read"] = (
        "If the excess is ~0 the treatment essentially never acted and every downstream null is a "
        "statement about exposure, not about the intervention. The taxonomy's base rate predicts "
        "an excess of roughly 0.12-0.15."
    )
    witness["reask_rate_by_arm"] = reask_rate

    witness["pairing_is_randomization_not_matched_sampling"] = (
        "READ THIS BEFORE READING n_fired. The pre-registration assumed a seeded sampler would "
        "give both arms the SAME attempt-0 completion, making every cross-arm difference "
        "attributable to the treatment. It does not: at byte-identical prompts, identical seeds "
        "and one completion call per arm, the arms still draw different engines. The (game, "
        "replicate) pair is therefore a BLOCKING factor, not a matched sample, and this remains a "
        "valid randomized A/B that is noisier and less powerful than the one that was designed. "
        "The treatment can only fire when the TREATMENT arm's own draw is clean-and-inert, not "
        "when the control's is -- so n_fired is a property of the ON arm alone."
    )
    witness["how_to_read"] = (
        "n_fired is how many times the treatment actually did something. It bounds every effect "
        "this experiment can detect: with n_fired small, a null is a statement about power, not "
        "about the intervention. n_diverged_without_trigger is the nondeterminism floor measured "
        "inside the A/B itself, and is directly comparable to n_fired -- if it is of similar "
        "size, the arms differ about as often for no reason as for the treatment."
    )

    # --- THE TESTS ----------------------------------------------------------------------
    # A DEVIATION BETWEEN THE PRE-REGISTRATION AND THE COLLECTION CODE, recorded rather than
    # quietly resolved. The prereg's MISSING_VS_ZERO clause names "a completion truncated by the
    # token cap" as a MISSING observation. `run_ab.py` does not implement that: it sets `missing`
    # only on a server failure or a harness exception, so a truncated completion becomes a
    # `truncated_before_required_symbols` defect and scores a real ZERO on usable-engine yield.
    #
    # Neither reading is obviously wrong. Truncation IS the generation path failing, which argues
    # for missing; it is also check #1 inside `validate_engine_code`, which is the very definition
    # the primary is measured with, which argues for zero. Choosing after seeing the results would
    # be the offence, so BOTH are computed: the prereg-faithful version is the PRIMARY (the
    # contract was written first), and the as-collected version is reported beside it. If they
    # disagree, that disagreement is itself a finding and is stated in the artifact.
    trunc_kind = "truncated_before_required_symbols"

    def _truncated(r) -> bool:
        return trunc_kind in (r.get("defect_kinds") or [])

    def m_usable(r):
        """PRIMARY, prereg-faithful: a truncated completion is a MISSING observation."""
        if r.get("missing") or _truncated(r):
            return None
        return 1.0 if r.get("usable") else 0.0

    def m_usable_as_collected(r):
        """SENSITIVITY: truncation scored as the real defect `validate_engine_code` calls it."""
        return None if r.get("missing") else (1.0 if r.get("usable") else 0.0)

    def m_live(r):
        if r.get("missing") or _truncated(r):
            return None
        return 1.0 if r.get("live") else 0.0

    def m_cf(r):
        h = r.get("heldout") or {}
        if r.get("missing") or r.get("score_missing") or not h.get("measurable"):
            return None
        return float(h.get("change_fidelity"))

    def m_depth(r):
        sg = r.get("state_graph") or {}
        if r.get("missing") or r.get("score_missing") or sg.get("probe_error"):
            return None
        v = sg.get("probe_depth_reached")
        return None if v is None else float(v)

    def m_acc(r):
        h = r.get("heldout") or {}
        if r.get("missing") or r.get("score_missing") or not h.get("measurable"):
            return None
        return float(h.get("accuracy"))

    def m_calls(r):
        return None if r.get("missing") else float(r.get("completion_calls_delta") or 0)

    def m_wall(r):
        return None if r.get("missing") else float(r.get("elapsed_s") or 0)

    tests = {}
    for name, fn in [
        ("PRIMARY_usable_engine_yield", m_usable),
        ("SENSITIVITY_usable_engine_yield_truncation_as_zero", m_usable_as_collected),
        ("SECONDARY_live_engine_yield", m_live),
        ("SECONDARY_heldout_change_fidelity", m_cf),
        ("SECONDARY_probe_depth_reached", m_depth),
        ("SECONDARY_heldout_accuracy", m_acc),
        ("COST_completion_calls", m_calls),
        ("COST_wall_seconds", m_wall),
    ]:
        deltas, detail = per_game_means(complete, fn)
        tests[name] = {**sign_test(deltas), "per_game": detail}

    missing = {
        "n_cells_collected": len(rows),
        "n_ab_cells": len(ab),
        "n_complete_pairs": len(complete),
        "n_incomplete_pairs": len(cells) - len(complete),
        "n_generation_missing": sum(1 for r in ab if r.get("missing")),
        "generation_missing_reasons": sorted(
            {r.get("missing_reason") for r in ab if r.get("missing")} - {None}
        ),
        "n_score_missing": sum(1 for r in ab if r.get("score_missing")),
        "score_missing_statuses": sorted(
            {r.get("score_status") for r in ab if r.get("score_missing")} - {None}
        ),
        "n_truncated": sum(1 for r in ab if _truncated(r)),
        "truncation_handling": "The pre-registration calls a truncated completion MISSING; the "
        "collection code scored it as the defect validate_engine_code "
        "reports. The PRIMARY follows the pre-registration (excluded); "
        "SENSITIVITY_usable_engine_yield_truncation_as_zero follows the "
        "collection code. Both are reported because choosing between them "
        "after seeing the numbers would be the offence.",
    }

    raw_rates = {}
    for arm in ("off", "on"):
        sel = [r for r in ab if r["arm"] == arm and not r.get("missing")]
        raw_rates[arm] = {
            "n": len(sel),
            "usable": round(sum(1 for r in sel if r.get("usable")) / len(sel), 4) if sel else None,
            "live": round(sum(1 for r in sel if r.get("live")) / len(sel), 4) if sel else None,
            "inert": round(sum(1 for r in sel if r.get("engine_inert") is True) / len(sel), 4)
            if sel
            else None,
            "mean_completion_calls": round(
                statistics.fmean([r.get("completion_calls_delta") or 0 for r in sel]), 3
            )
            if sel
            else None,
            "mean_wall_s": round(statistics.fmean([r.get("elapsed_s") or 0 for r in sel]), 1)
            if sel
            else None,
        }
    raw_rates["note"] = (
        "POOLED over cells, NOT the test. Games contribute unequal numbers of cells and cells "
        "within a game are correlated, so these are descriptive only. Every inferential claim "
        "comes from the game-clustered sign tests above."
    )

    out = {
        "prereg_sha256": meta.get("prereg_sha256"),
        "prereg_declared_primary": prereg["PRIMARY"]["metric"],
        "server_witness": meta.get("server_witness"),
        "collection": missing,
        "aa_nondeterminism_floor": aa_summary,
        "mechanistic_witness": witness,
        "raw_pooled_rates": raw_rates,
        "tests": tests,
    }
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2))

    print(
        f"complete pairs: {len(complete)}  |  treatment fired: {witness['n_fired']}  |  "
        f"diverged w/o trigger: {witness['n_diverged_without_trigger']}"
    )
    print(
        f"A/A byte-identical: {aa_summary['n_byte_identical']}/{aa_summary['n_cells']}  "
        f"primary-agrees: {aa_summary['n_primary_agrees']}/{aa_summary['n_cells']}"
    )
    for name, t in tests.items():
        print(
            f"{name:38} d={t['mean_delta']} disc={t['n_discordant']} "
            f"(+{t['n_positive']}/-{t['n_negative']}/={t['n_ties']}) p={t['p_two_sided']} "
            f"minp={t['min_reachable_two_sided_p_at_this_discordance']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
