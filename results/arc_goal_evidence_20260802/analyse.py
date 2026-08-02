#!/usr/bin/env python3
"""Score the goal-evidence A/B: per-arm shape rates, game-clustered permutation tests, A/A floor.

CLUSTERING IS AT THE GAME, NOT THE CELL. Replicates within a game are not independent trials --
they share the window, the transitions and the prompt. Treating them as independent inflated a
sibling experiment's p from 0.125 to 0.049 on 2026-07-31 and had to be corrected. Here the
statistic is the mean over GAMES of a within-game rate difference and the permutation reshuffles
the arm label WITHIN each game, so a game with many cells cannot dominate.

MISSING IS NEVER ZERO. A cell that produced no parseable `is_level_complete` -- server failure,
truncation, an exception, an unparseable completion -- is EXCLUDED from the denominator and
counted per arm as `n_missing`. Scoring it as a non-decline would credit the treatment for a
failure, and scoring it as a decline would blame it for one.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classify import GROUNDABLE_CLUSTERS  # noqa: E402

OUT = HERE / "out"
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
RNG = np.random.default_rng(20260802)

SHAPES = ("DECLINED", "TROPE", "GROUNDED", "OTHER")


def per_game_vectors(rows: list[dict], arm: str, shape: str, ran_only: bool):
    """{game: 0/1 array} for cells in `arm`.

    For a real shape the denominator is cells with a RECOVERABLE predicate -- missing is never
    zero, so an unrecoverable cell is dropped rather than scored as a non-decline.

    `shape="MISSING"` is the deliberate exception and is a MEASURE, not a nuisance: it scores
    every cell, 1 if no predicate could be recovered. DIFFERENTIAL MISSINGNESS IS A TREATMENT
    EFFECT. A longer prompt that makes the model ramble past its token budget converts cells
    that would have succeeded into hard failures, and if that happens only in the treatment arm
    then the surviving treatment cells are a SELECTED sample and every other contrast is
    conditioned on a post-treatment variable. The sibling goal-defect A/B measured exactly this
    failure the night before -- 17 of 21 treatment cells hard-failed against 1 of 22 in control
    -- so it is tested here rather than discovered in the residuals.
    """
    out: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["arm"] != arm:
            continue
        if ran_only and not r.get("goal_only_call_ran"):
            continue
        s = r.get("pred_shape")
        if shape == "MISSING":
            out[r["game"]].append(1 if s is None else 0)
            continue
        if s is None:
            continue
        out[r["game"]].append(1 if s == shape else 0)
    return {g: np.asarray(v, dtype=float) for g, v in out.items() if v}


def arm_summary(rows: list[dict], arm: str, ran_only: bool = False) -> dict:
    cells = [r for r in rows if r["arm"] == arm and (not ran_only or r.get("goal_only_call_ran"))]
    scored = [r for r in cells if r.get("pred_shape") is not None]
    n = len(scored)
    d = {
        "arm": arm,
        "n_cells": len(cells),
        "n_scored": n,
        "n_missing": len(cells) - n,
        "n_games": len({r["game"] for r in scored}),
        "mechanism_fired_rate": (
            round(sum(1 for r in cells if r.get("goal_only_call_ran")) / len(cells), 4)
            if cells
            else None
        ),
        "median_elapsed_s": round(float(np.median([r["elapsed_s"] for r in cells])), 1)
        if cells
        else None,
    }
    for s in SHAPES:
        d[f"{s.lower()}_rate"] = (
            round(sum(1 for r in scored if r["pred_shape"] == s) / n, 4) if n else None
        )
        d[f"{s.lower()}_n"] = sum(1 for r in scored if r["pred_shape"] == s)
    d["constant_true_n"] = sum(1 for r in scored if r.get("pred_other_kind") == "constant_true")
    d["dual_definition_n"] = sum(1 for r in scored if (r.get("pred_n_defs") or 0) > 1)
    # THE SHARPEST SINGLE COLUMN IN THE RUN, if it moves. A declined predicate whose own
    # docstring says a win state was never provided is the model EXPLICITLY diagnosing the
    # information gap this intervention exists to close -- not an inferred motive, the model's
    # own words. Four such predicates in the 71-engine taxonomy are what made the gap
    # actionable at all. If the treatment removes these specifically, the causal story is
    # direct; if declines stay flat but these vanish, the model has stopped SAYING it and not
    # stopped doing it, which is worth knowing and would be invisible without this column.
    d["declined_saying_no_win_state_n"] = sum(
        1
        for r in scored
        if r["pred_shape"] == "DECLINED" and r.get("pred_docstring_says_no_win_state")
    )
    return d


def perm_test(rows: list[dict], treat: str, control: str, shape: str, ran_only: bool) -> dict:
    """Game-clustered stratified permutation on the mean within-game rate difference."""
    treat_v = per_game_vectors(rows, treat, shape, ran_only)
    ctrl_v = per_game_vectors(rows, control, shape, ran_only)
    games = sorted(set(treat_v) & set(ctrl_v))
    if not games:
        # The identifying fields are carried on the error too. A row that says only "error"
        # forces the reader to infer WHICH contrast is empty from its position in a list, and a
        # test that cannot be named cannot be audited.
        return {
            "treat": treat,
            "control": control,
            "shape": shape,
            "mechanism_fired_only": ran_only,
            "error": "no game has both arms scored -- this contrast is UNMEASURED, which is not "
            "the same as measured-and-null",
            "n_games": 0,
        }
    obs = float(np.mean([treat_v[g].mean() - ctrl_v[g].mean() for g in games]))

    pooled = {g: np.concatenate([treat_v[g], ctrl_v[g]]) for g in games}
    nt = {g: len(treat_v[g]) for g in games}
    chunk = 20_000
    exceed = 0
    done = 0
    while done < DRAWS:
        d = min(chunk, DRAWS - done)
        acc = np.zeros(d)
        for g in games:
            p = pooled[g]
            n = len(p)
            k = nt[g]
            idx = np.argsort(RNG.random((d, n)), axis=1)
            perm = p[idx]
            acc += perm[:, :k].mean(axis=1) - perm[:, k:].mean(axis=1)
        acc /= len(games)
        exceed += int(np.sum(np.abs(acc) >= abs(obs) - 1e-12))
        done += d
    p = (exceed + 1) / (DRAWS + 1)

    # Game-clustered bootstrap CI on the same statistic: resample GAMES with replacement.
    boot = np.empty(4000)
    garr = np.asarray(games)
    for i in range(4000):
        sel = RNG.choice(garr, size=len(garr), replace=True)
        boot[i] = np.mean([treat_v[g].mean() - ctrl_v[g].mean() for g in sel])
    return {
        "treat": treat,
        "control": control,
        "shape": shape,
        "mechanism_fired_only": ran_only,
        "n_games": len(games),
        "n_treat_cells": int(sum(len(treat_v[g]) for g in games)),
        "n_control_cells": int(sum(len(ctrl_v[g]) for g in games)),
        "rate_treat": round(float(np.mean([treat_v[g].mean() for g in games])), 4),
        "rate_control": round(float(np.mean([ctrl_v[g].mean() for g in games])), 4),
        "delta_mean_over_games": round(obs, 4),
        "ci95_game_bootstrap": [
            round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4),
        ],
        "p_permutation_two_sided": round(p, 5),
        "draws": DRAWS,
        "significant_at_0.05": bool(p < 0.05),
    }


def main() -> int:
    rows = json.loads((OUT / "rows.json").read_text())
    s1 = [r for r in rows if r["stage"] == 1]
    s2 = [r for r in rows if r["stage"] == 2]

    res: dict = {
        "n_rows": len(rows),
        "stage1_goal_only_component": {"arms": [arm_summary(s1, a) for a in ("gA", "gB", "gAA")]},
        "stage2_live_induce_ITT": {"arms": [arm_summary(s2, a) for a in ("A", "B", "C", "AA")]},
        "stage2_live_induce_mechanism_fired_only": {
            "arms": [arm_summary(s2, a, ran_only=True) for a in ("A", "B", "C", "AA")]
        },
    }

    tests = []
    # SHAPES + MISSING. The missingness test runs FIRST, before any shape contrast, because if
    # it is significant every shape contrast below is conditioned on a post-treatment variable
    # and must be read as such. Order here is the reading order.
    shapes_and_missing = ("MISSING", *SHAPES)
    for shape in ("MISSING",):
        tests.append({"block": "stage1_missingness", **perm_test(s1, "gB", "gA", shape, False)})
        tests.append(
            {"block": "stage1_missingness_AA_floor", **perm_test(s1, "gAA", "gA", shape, False)}
        )
        for treat in ("B", "C"):
            tests.append({"block": "stage2_missingness", **perm_test(s2, treat, "A", shape, False)})
        tests.append(
            {"block": "stage2_missingness_AA_floor", **perm_test(s2, "AA", "A", shape, False)}
        )
    assert shapes_and_missing[0] == "MISSING"
    # STAGE 1 -- the powered contrast. PRIMARY shape first, then the secondaries.
    for shape in SHAPES:
        tests.append({"block": "stage1", **perm_test(s1, "gB", "gA", shape, False)})
    # THE A/A FLOOR. Mandatory: the generator sends no seed to the sampler by default and A/A
    # has failed repeatedly this week. Any treatment delta must be read against this.
    for shape in SHAPES:
        tests.append({"block": "stage1_AA_floor", **perm_test(s1, "gAA", "gA", shape, False)})
    # STAGE 2 -- declared underpowered before the run; reported in full anyway.
    for treat in ("B", "C"):
        for shape in SHAPES:
            tests.append({"block": "stage2_ITT", **perm_test(s2, treat, "A", shape, False)})
    for shape in SHAPES:
        tests.append({"block": "stage2_AA_floor", **perm_test(s2, "AA", "A", shape, False)})
    for treat in ("B", "C"):
        for shape in SHAPES:
            tests.append(
                {"block": "stage2_mechanism_fired", **perm_test(s2, treat, "A", shape, True)}
            )
    res["tests"] = tests

    # Cluster cross-tab: WHERE the shapes come from, so a null can be attributed.
    xt: dict = {}
    for name, sub in (("stage1", s1), ("stage2", s2)):
        c: dict = defaultdict(lambda: defaultdict(int))
        for r in sub:
            if r.get("pred_shape") is not None:
                c[r["arm"]][r.get("pred_cluster", "?")] += 1
        xt[name] = {a: dict(v) for a, v in c.items()}
    res["cluster_crosstab"] = xt

    # The mechanism-firing audit, per stage-2 arm. An arm whose mechanism never fired is an
    # UNTESTED arm, not a refuted one, and this is the column that says which.
    fired: dict = {}
    for a in ("A", "B", "C", "AA"):
        cells = [r for r in s2 if r["arm"] == a]
        fired[a] = {
            "n_cells": len(cells),
            "n_goal_only_call_ran": sum(1 for r in cells if r.get("goal_only_call_ran")),
            "rate": round(sum(1 for r in cells if r.get("goal_only_call_ran")) / len(cells), 4)
            if cells
            else None,
        }
    res["stage2_mechanism_firing"] = fired

    # BYTE-IDENTITY ON NON-FIRING CELLS. This turns "the treatment was diluted" from a
    # statistical hedge into a STRUCTURAL FACT. B and C share A's seed within a (game,
    # replicate), so on a cell where the combined induce call succeeded and the goal-only call
    # was never built, the three arms should not merely be similar -- they should produce the
    # SAME BYTES. If they do, those cells contribute EXACTLY ZERO to any arm difference, and the
    # ITT contrast is a weighted average over the firing minority with a known-zero remainder
    # rather than a noisy wash. If they do NOT, something other than the declared knobs is
    # varying between arms, and that is a defect in the harness which must be found before any
    # contrast is believed -- so the disagreeing cells are listed, not just counted.
    ident: dict = {"n_compared": 0, "n_identical": 0, "disagreeing_cells": []}
    by_cell: dict[tuple, dict] = {}
    for r in s2:
        by_cell[(r["game"], r["replicate"], r["arm"])] = r
    for (game, rep, a), row in sorted(by_cell.items(), key=str):
        if a == "A" or row.get("goal_only_call_ran"):
            continue
        base = by_cell.get((game, rep, "A"))
        if base is None or base.get("goal_only_call_ran") or a == "AA":
            # AA is EXCLUDED on purpose: it is the same arm as A at a DIFFERENT seed, so it is
            # expected to differ and including it would manufacture a false disagreement.
            continue
        if "engine_sha256" not in row or "engine_sha256" not in base:
            continue
        ident["n_compared"] += 1
        if row["engine_sha256"] == base["engine_sha256"]:
            ident["n_identical"] += 1
        else:
            ident["disagreeing_cells"].append(f"{game}__r{rep}__{a}")
    ident["all_identical"] = ident["n_compared"] > 0 and ident["n_identical"] == ident["n_compared"]
    ident["reading"] = (
        "compares B and C against A on cells where the goal-only call did NOT run, at the same "
        "seed. Identity there is the structural proof that the knobs contribute exactly zero "
        "outside the split-induce minority."
    )
    res["stage2_nonfiring_byte_identity"] = ident

    # SENSITIVITY, NOT A REPLACEMENT: GROUNDED with trivial literals excluded.
    #
    # DISCLOSED PROVENANCE -- this was added AFTER inspecting three landed control/treatment
    # pairs, so it is a post-hoc sensitivity and is labelled as one. It does NOT replace the
    # pre-registered GROUNDED rate anywhere, and the pre-registered number is still the one the
    # verdict branches on. Swapping in a definition devised after seeing which arm it favours
    # would be exactly the instrument-tuning this project's disciplines exist to prevent.
    #
    # WHAT IT FIXES. `_int_literals` counts every small int, so `0` and `1` -- which appear in
    # essentially every numpy predicate as indexing and background constants -- can satisfy the
    # grounding test on their own. Observed concretely on bp35: the CONTROL predicate, which
    # invents "one solid rectangle of a single colour" and references nothing the agent saw,
    # scored GROUNDED on literals [0, 1] (hit row 1, colour 0). The TREATMENT predicate on the
    # same game and seed scored GROUNDED on literals [0, 63], where 63 is the bottom row it read
    # a progress bar out of -- a real reference to a real observed region. Both are "GROUNDED"
    # under the pre-registered rule and only one of them means anything.
    #
    # So this variant requires a hit on a literal >= 2. It is reported for BOTH arms, and its
    # direction of bias is stated: it removes accidental grounding from whichever arm has it.
    strict: dict = {}
    for name, sub in (("stage1", s1), ("stage2", s2)):
        per: dict = {}
        for a in sorted({r["arm"] for r in sub}):
            scored = [r for r in sub if r["arm"] == a and r.get("pred_shape") is not None]
            n_strict = 0
            for r in scored:
                if r.get("pred_shape") != "GROUNDED":
                    continue
                h = r.get("pred_grounding_hits") or {}
                hits = (
                    set(h.get("rows") or [])
                    | set(h.get("cols") or [])
                    | set(h.get("colours") or [])
                )
                if any(v >= 2 for v in hits):
                    n_strict += 1
            per[a] = {
                "n_scored": len(scored),
                "n_grounded_preregistered": sum(1 for r in scored if r["pred_shape"] == "GROUNDED"),
                "n_grounded_strict_nontrivial_literal": n_strict,
                "grounded_strict_rate": round(n_strict / len(scored), 4) if scored else None,
            }
        strict[name] = per
    res["SENSITIVITY_grounded_excluding_trivial_literals"] = {
        "provenance": "POST-HOC. Added after inspecting three landed pairs; reported alongside "
        "the pre-registered GROUNDED rate and never substituted for it.",
        "rule": "a GROUNDED predicate counts only if some matched literal is >= 2, so the "
        "numpy-ubiquitous constants 0 and 1 cannot ground a predicate by themselves",
        "by_stage": strict,
    }

    # GROUNDING AUDIT. `_int_literals` is deliberately INCLUSIVE (any small int anywhere in the
    # predicate can match a row, a column or a colour), which makes GROUNDED easy to pass. That
    # is the conservative direction for a NULL but the OPTIMISTIC direction for a positive, so
    # the two components are separated here rather than left fused:
    #   groundable_rate      -- did the model write a region/object-naming predicate at all
    #   grounded_given_groundable -- and did the thing it named actually appear in its own deltas
    # A treatment that moves only the first has changed the model's STYLE; one that moves the
    # second has changed what the model is READING. They are different claims.
    audit: dict = {}
    for name, sub in (("stage1", s1), ("stage2", s2)):
        per: dict = {}
        for a in sorted({r["arm"] for r in sub}):
            scored = [r for r in sub if r["arm"] == a and r.get("pred_shape") is not None]
            groundable = [
                r
                for r in scored
                if r.get("pred_cluster") in GROUNDABLE_CLUSTERS or r.get("pred_shape") == "GROUNDED"
            ]
            gr = [r for r in scored if r["pred_shape"] == "GROUNDED"]
            by = {"rows": 0, "cols": 0, "colours": 0, "colour_only": 0}
            for r in gr:
                h = r.get("pred_grounding_hits") or {}
                for k in ("rows", "cols", "colours"):
                    if h.get(k):
                        by[k] += 1
                if h.get("colours") and not h.get("rows") and not h.get("cols"):
                    by["colour_only"] += 1
            per[a] = {
                "n_scored": len(scored),
                "n_groundable_cluster": len(groundable),
                "groundable_rate": round(len(groundable) / len(scored), 4) if scored else None,
                "n_grounded": len(gr),
                "grounded_given_groundable": round(len(gr) / len(groundable), 4)
                if groundable
                else None,
                "grounded_hit_kinds": by,
            }
        audit[name] = per
    res["grounding_audit"] = audit

    # PER-GAME, so a reader can see whether an effect is carried by two games or spread across
    # twenty. The game-clustered permutation already stops one big game dominating the p-value,
    # but it cannot show CONCENTRATION, and concentration is the difference between "this works"
    # and "this works on cd82".
    pg: dict = {}
    for name, sub in (("stage1", s1), ("stage2", s2)):
        t: dict = {}
        for g in sorted({r["game"] for r in sub}):
            t[g] = {}
            for a in sorted({r["arm"] for r in sub}):
                sc = [
                    r
                    for r in sub
                    if r["game"] == g and r["arm"] == a and r.get("pred_shape") is not None
                ]
                if not sc:
                    t[g][a] = None
                    continue
                t[g][a] = {
                    s.lower(): round(sum(1 for r in sc if r["pred_shape"] == s) / len(sc), 3)
                    for s in SHAPES
                }
                t[g][a]["n"] = len(sc)
        pg[name] = t
    res["per_game"] = pg

    # KNOWN CONFOUNDS, found by adversarially reviewing this harness against its own brief and
    # written into the OUTPUT rather than only into a note, so they travel with the numbers.
    res["known_confounds"] = {
        "GROUNDED_is_partially_determined_by_the_treatment": {
            "severity": "HIGH -- this is the circularity class the brief warns about",
            "what": "GROUNDED is defined as 'the predicate names a literal that appears in the "
            "agent's observed deltas'. The TREATMENT prompt is precisely those deltas, rendered "
            "as text. A model that copies a row index out of its prompt scores GROUNDED without "
            "having understood anything, and the control -- which is shown no deltas -- can only "
            "score GROUNDED by coincidence.",
            "so_what": "the GROUNDED contrast is NOT a clean read on goal quality and must not "
            "be reported as one. It is reported as what it is: whether the treatment gets the "
            "model to REFERENCE its own observations.",
            "which_metrics_are_clean": "DECLINED (the PRIMARY) is unaffected -- a constant-False "
            "predicate references nothing, so no prompt content can mechanically produce or "
            "prevent it. TROPE is unaffected for the same reason: a whole-board uniformity claim "
            "names nothing observed by construction. MISSING is unaffected -- it is a parse "
            "outcome. The circularity is confined to GROUNDED.",
            "why_it_was_not_fixed": "removing it would mean changing the pre-registered outcome "
            "definition after seeing data. It is disclosed instead.",
        },
        "arm_ORDER_within_a_game_is_fixed_not_randomised": {
            "severity": "MEDIUM, and confined to the TIMING measure",
            "what": "cells run A, B, C, AA within a game (and gA, gB, gAA in stage 1), always in "
            "that order. Anything that drifts with position -- llama.cpp prompt-cache warmth, "
            "GPU thermal state -- is therefore confounded with the arm label.",
            "so_what_for_SHAPE": "nothing. Every arm sends the SAME seed within a (game, "
            "replicate), and a fixed seed fixes the draw; the byte-identity check on non-firing "
            "stage-2 cells is direct evidence that order is not perturbing the sampled output.",
            "so_what_for_TIMING": "median_elapsed_s must be read descriptively and NEVER as a "
            "treatment effect with a p-value. Observed directly in this run: on ar25 the control "
            "took 111.4s and both treatments 55.2/55.1s, while on bp35 the control took 44.5s "
            "and both treatments 67.5/67.6s -- the sign flips between games, which is what "
            "position-plus-noise looks like.",
            "where_timing_IS_load_bearing": "stage 1, where the treatment prompt is 10-20x the "
            "control's and the gap is 3s vs 340s. A cache or thermal effect cannot produce a "
            "100x difference in that direction -- warmth makes calls FASTER -- and the failures "
            "are explained by a mechanism visible in the artefact itself (three full 4096-token "
            "attempts, code block truncated mid-definition).",
            "the_fix_for_a_future_run": "randomise or counterbalance arm order within a game.",
        },
    }

    (OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "tests"}, indent=2, default=str)[:4000])
    print("\n--- tests ---")
    for t in tests:
        if t.get("error"):
            print(t["block"], t.get("shape"), "ERR", t["error"])
            continue
        print(
            f"{t['block']:28s} {t['shape']:9s} {t['treat']} vs {t['control']}: "
            f"{t['rate_control']:.3f} -> {t['rate_treat']:.3f} "
            f"d={t['delta_mean_over_games']:+.4f} ci={t['ci95_game_bootstrap']} "
            f"p={t['p_permutation_two_sided']:.4f} n_games={t['n_games']} "
            f"nT={t['n_treat_cells']} nC={t['n_control_cells']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
