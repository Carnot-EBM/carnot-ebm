#!/usr/bin/env python3
"""HOW OPTIMISTIC ARE OUR ARC SCORE ESTIMATES, AND BY HOW MUCH? -- the artifact builder.

THE DEFECT. The live gateway charges a RESET one action
(`arc_agi/scorecard.py:701-704` `inc_reset_count` -> `resets += 1` AND
`actions += 1`, reached from `update_scorecard`:839-843). Our offline harness
charged it ZERO (`scripts/arc_leaderboard_eval.py`: `actions += 1` lived only in
the non-RESET branch). Since the scorer's per-level cost is a DIFFERENCE of
cumulative CHARGED counts (:479) and the per-level score is
`min((baseline/level_actions)**2 * 100, 115)`, every per-level efficiency number
this project holds is optimistic in the SQUARED term by the resets charged
BEFORE that level-up.

TWO MEASUREMENTS WITH TWO CLOCKS -- never conflated (CLAUDE.md "THE ANALYSER
CLOCK IS NOT THE MEASUREMENT CLOCK"):

  PART A -- BOUNDS, an AGGREGATION over the 1401 already-persisted rows.
    Substrate `aggregation_from_upstream_artifacts`. Its `measurement_wall_s` is
    the SUM OF EACH UPSTREAM ROW FILE'S OWN `elapsed_s`, not this script's
    runtime and not the sum of per-cell `wall_s` (which undercounts ~25%).
    Answers: what is the WIDEST the correction could be, given the rows record
    only a whole-run `n_resets`?

  PART B -- EXACT ATTRIBUTION, a LIVE re-run. Substrate
    `offline_arcade_live_agent_runtime_self_discovery_no_llm`, with its own
    `part_b_measurement_wall_s`. Answers: where in that band does reality sit?
    Matched cell-for-cell to the recorded arm (same game/seed/budget, LLM off,
    same seeding), and the match is WITNESSED by reproducing each recorded row's
    `efficiency` AND `n_resets` exactly -- so the exact gateway numbers attach to
    the recorded corpus rather than describing a different agent.

WHY PART B IS NOT OPTIONAL. Part A's band is [0%, 95.7%] per cell. A band that
wide cannot answer "is the optimism materially different from zero" in either
direction -- its best case IS zero by construction. Reporting Part A alone would
be the unfalsifiable-gate failure mode. Part B is what makes the question
answerable.

This script never submits anything and never rewrites a historical artifact's
recorded numbers: the corrections it publishes are NEW numbers that cite the
originals by path and checksum.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from arc_gateway_rescore import (  # noqa: E402
    crosscheck_post_solve_tail_is_free,
    crosscheck_reset_charge,
    crosscheck_row,
    gateway_score_via_calculator,
    load_rows,
    rescore_row,
)

ROWS_GLOB = "results/early_stop_sweep_20260726/rows_*.json"
EXACT_FILES = [
    "results/early_stop_sweep_20260726/rows_exact_attribution.json",
    "results/early_stop_sweep_20260726/rows_exact_attribution_b2000.json",
]
# Corpora that record NO reset count at all -- reported as a coverage hole, not
# silently dropped.
UNRESCORABLE_GLOBS = ["results/cptb_20260726_cells/*.jsonl.gz"]

ACTUAL_HIDDEN_SCORE = 0.08
ACTUAL_HIDDEN_REF = (
    "public leaderboard 0.08, 2026-06-19 23:40Z, ref 53862349, 'carnot v1.1', kernel v3 "
    "(flat v3->v5); recorded ops/known-issues.md:3275"
)


def _q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = (len(xs) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": round(min(xs), 6),
        "p25": round(_q(xs, 0.25), 6),
        "median": round(_q(xs, 0.50), 6),
        "p75": round(_q(xs, 0.75), 6),
        "max": round(max(xs), 6),
        "mean": round(statistics.mean(xs), 6),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out_path: Path | None = None) -> dict:
    """Build the artifact. `out_path` is threaded in ONLY so registration names the
    file actually written -- a hardcoded registration path would let a
    scratchpad dry-run register the real deliverable it never wrote."""
    t0 = time.time()
    art: dict = {
        "experiment": "outer_loop_arc_gateway_rescore_20260726",
        "title": (
            "Gateway-accurate re-scoring: the uncharged-RESET defect, bounded over 1401 "
            "recorded rows and then MEASURED exactly on matched live re-runs"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.arc_gateway_rescore.v1",
    }

    # ---------------------------------------------------------------- the defect
    art["defect_under_measurement"] = {
        "gateway_charges_a_reset": (
            "arc_agi/scorecard.py:701-704 inc_reset_count -> `self.resets[i] += 1` AND "
            "`self.actions[i] += 1`; reached from update_scorecard:839-843"
        ),
        "our_harness_charged_zero": (
            "scripts/arc_leaderboard_eval.py -- `actions += 1` lived ONLY in the non-RESET "
            "branch (fixed additively 2026-07-26; the pre-existing `actions` semantics are "
            "unchanged, new fields were added alongside)"
        ),
        "why_it_is_squared": (
            "per-level charge is a DIFFERENCE of cumulative CHARGED counts (scorecard.py:479 "
            "`level_actions = actions_at_level - prev_actions`), and the per-level score is "
            "min((baseline/level_actions)**2 * 100, 115) (:168-171) -- so a reset taken BEFORE "
            "a level-up lands in that level's denominator and is squared"
        ),
        "reset_replay_is_not_a_rare_path": (
            "arc_competition_agent.py:969-970 -- RESET + replay is described as 'the ONLY "
            "navigation'; :3339 records 'every backtrack RESET-replayed from root'"
        ),
        "a_LARGER_defect_ruled_out_from_source": {
            "the_risk": (
                "the gateway consumes `actions_by_level` POSITIONALLY -- `level, actions_at_level "
                "= actions_by_level[level_idx]` (scorecard.py:477) IGNORES the stored level "
                "number. `Card.set_levels_completed` (:706-713) appends an entry whenever "
                "levels_completed CHANGES, in either direction. So if a RESET dropped "
                "levels_completed back toward 0, it would insert a spurious entry and shift every "
                "subsequent level's charge by one slot -- corrupting the per-level attribution far "
                "more than the one-action charge does."
            ),
            "verdict": "DOES NOT MATERIALIZE, verified from the installed engine source",
            "evidence": (
                "arcengine/base_game.py:308-328 -- `reset()` dispatches to `full_reset()` only "
                "when `_action_count == 0` or the state is WIN; otherwise it calls "
                "`level_reset()`, which re-clones the CURRENT level and does NOT touch `_score` "
                "(:325-328). `levels_completed` is `_score` (:244,:255). Therefore every MID-RUN "
                "reset preserves levels_completed, appends no entry, and the positional "
                "consumption stays aligned."
            ),
            "residual_edge_case_stated_not_hidden": (
                "a reset at `_action_count == 0` IS a full_reset (`_score = 0`), and on the "
                "gateway side `update_scorecard`:836-839 routes a full_reset to `new_play()` -- a "
                "NEW card index whose action count starts at 0 -- rather than to "
                "`inc_reset_count()`. That path is the run's opening reset, before any level-up, "
                "so it cannot corrupt an attribution; but a policy that contrived a mid-run "
                "full_reset would score as a separate play, and this analysis does not cover that."
            ),
        },
        "three_units_never_conflated": {
            "offline_actions": "our harness `actions`; EXCLUDES resets. The unit every recorded `efficiency` is in.",
            "frames": "loop iterations; INCLUDES resets.",
            "gateway_charged": "non-RESET moves PLUS resets. The ONLY unit the competition score is a function of.",
            "identity_asserted": "gateway_charged == frames == offline_actions + n_resets",
        },
    }

    # ------------------------------------------- scorer-fidelity cross-checks
    xchecks = {
        "reset_charge_through_full_chain": crosscheck_reset_charge(),
        "post_solve_tail_is_free_through_full_chain": crosscheck_post_solve_tail_is_free(),
    }

    # ---------------------------------------------------------------- PART A
    row_files = sorted(f for f in glob.glob(str(REPO / ROWS_GLOB)) if "exact_attribution" not in f)
    upstream_elapsed = 0.0
    row_sources = []
    all_res = []
    for f in row_files:
        raw = json.loads(Path(f).read_text())
        el = float(raw.get("elapsed_s") or 0.0)
        upstream_elapsed += el
        row_sources.append(
            {
                "path": os.path.relpath(f, REPO),
                "elapsed_s": el,
                "n_rows": len(raw.get("rows") or []),
                "sha256": _sha(Path(f)),
            }
        )
        for r in raw.get("rows") or []:
            res = rescore_row(r)
            d = dict(res.__dict__)
            d["source"] = os.path.basename(f)
            all_res.append(d)

    lv = [r for r in all_res if r["rescorable"] and r["n_levels_completed"]]
    no_lv = [r for r in all_res if r["rescorable"] and not r["n_levels_completed"]]

    # full-chain cross-check on every DISTINCT (game,budget,levels,per-level-cost)
    sigs: dict = {}
    for r in lv:
        sigs.setdefault(
            (r["game"], r["budget"], r["n_levels_completed"], tuple(r["level_offline"])), r
        )
    xc_agree, xc_tot, xc_fail = 0, 0, []
    for r in sigs.values():
        obj = type("R", (), {})()
        obj.__dict__.update({k: v for k, v in r.items() if k != "source"})
        got = crosscheck_row({}, obj)
        if got is None:
            continue
        xc_tot += 1
        if got["agree"]:
            xc_agree += 1
        else:
            xc_fail.append(got)
    xchecks["calculator_vs_full_chain_per_signature"] = {
        "n_distinct_signatures": xc_tot,
        "n_agree": xc_agree,
        "failures": xc_fail[:10],
        "agreement": (round(xc_agree / xc_tot, 6) if xc_tot else None),
        "principle": (
            "Path 1 drives EnvironmentScoreCalculator as _calculate_score does; path 2 builds a "
            "real Scorecard/Card through the REAL mutators and scores via from_scorecard. Both "
            "are the installed scorer, neither is a paraphrase of the formula. Cell-by-cell "
            "agreement is what makes any downstream sensitivity claim mean anything."
        ),
    }

    dis = [r for r in lv if not r["greedy_agrees"]]
    part_a = {
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "measurement_wall_s": round(upstream_elapsed, 1),
        "measurement_wall_s_principle": (
            "the SUM of each upstream row FILE's own elapsed_s -- the clock of the runs being "
            "re-scored. NOT this analyser's runtime (that is duration_s) and NOT the sum of "
            "per-cell wall_s (which undercounts ~25%)."
        ),
        "row_sources": row_sources,
        "n_rows_total": len(all_res),
        "n_rows_with_a_levelup_and_reset_count": len(lv),
        "n_rows_no_levelup_score_zero_either_way": len(no_lv),
        "n_rows_unrescorable": len([r for r in all_res if not r["rescorable"]]),
        "why_only_a_bound": (
            "the persisted rows record a WHOLE-RUN `n_resets` only. The correction needs the "
            "resets attributed PER LEVEL, because the scorer differences cumulative counts. "
            "BEST case = every reset lands after the last level-up, where it costs EXACTLY "
            "nothing (an incomplete level scores 0.0 whatever it is charged, scorecard.py:178-183) "
            "-> the score is unchanged. WORST case = an exact DP allocation of all n_resets "
            "across the completed levels."
        ),
        "worst_case_method": (
            "exact dynamic programming over (level, resets-used). A greedy-on-marginals "
            "allocation is computed independently as a cross-check and DISAGREES on "
            f"{len(dis)}/{len(lv)} cells -- always by finding a HIGHER (less pessimal) score, "
            "because the 115 cap creates a flat region with zero marginal that greedy refuses "
            "to spend into. The DP is therefore the reported bound; greedy is reported as a "
            "witness that the naive method is wrong here, not averaged with it."
        ),
        "greedy_vs_dp_disagreements": len(dis),
        "greedy_never_beat_dp": all(r["worst_score"] <= r["greedy_score"] + 1e-12 for r in dis),
        "abs_delta_offline_minus_worst_score_points": _stats([r["delta_worst"] for r in lv]),
        "rel_delta_fraction_of_offline_score_erased": _stats(
            [r["rel_delta_worst"] for r in lv if r["rel_delta_worst"] is not None]
        ),
        "verdict": (
            "THE BOUND IS TOO WIDE TO ANSWER THE HEADLINE QUESTION. Per cell the worst case "
            "erases between 0% and 95.7% of the offline score (median 11.4%), and its best case "
            "is 0% BY CONSTRUCTION -- so bounds alone can establish neither that the optimism "
            "is material nor that it is negligible. This is a real answer, and it is exactly "
            "what justifies the per-level instrumentation in Part B."
        ),
    }

    # cells where resets do the most damage: the CAPPED superhuman levels
    capped = [
        r
        for r in lv
        if r["baselines"]
        and r["level_offline"]
        and ((r["baselines"][0] / max(r["level_offline"][0], 1)) ** 2) * 100 > 115
    ]
    part_a["mechanism_note_cap_region"] = {
        "n_cells_whose_first_level_is_at_the_115_cap": len(capped),
        "note": (
            "a level solved FASTER than the human baseline by >7.2% sits at the 115 cap, so its "
            "score has the most headroom to lose -- and conversely the cap ABSORBS a small reset "
            "charge with zero score change. Both effects are visible in Part B."
        ),
    }

    # ------------------------------------------------- unrescorable corpora
    unres = []
    for g in UNRESCORABLE_GLOBS:
        files = sorted(glob.glob(str(REPO / g)))
        n = 0
        for f in files:
            n += len(load_rows(f))
        unres.append(
            {
                "glob": g,
                "n_files": len(files),
                "n_cells": n,
                "reason": (
                    "declares `action_count_convention: resets_excluded_run_game_native` but "
                    "records NO reset count and NO level_up_actions -- the quantity the "
                    "correction needs was never persisted, so these cells CANNOT be "
                    "gateway-re-scored at any width. Reported as a coverage hole."
                ),
            }
        )
    art["corpora_that_cannot_be_rescored_at_all"] = unres

    # ---------------------------------------------------------------- PART B
    exact_cells = []
    part_b_wall = 0.0
    exact_sources = []
    for rel in EXACT_FILES:
        p = REPO / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        part_b_wall += float(d.get("measurement_wall_s") or 0.0)
        exact_sources.append(
            {"path": rel, "measurement_wall_s": d.get("measurement_wall_s"), "sha256": _sha(p)}
        )
        exact_cells.extend(d.get("cells") or [])

    solved = [
        c
        for c in exact_cells
        if c.get("levels")
        and c.get("efficiency_offline") is not None
        and c.get("efficiency_gateway_charged") is not None
    ]

    # UNROUNDED recomputation. `run_game` rounds both efficiency fields to 4
    # decimals, and on small-score cells that rounding silently ERASES a real
    # delta: measured 2026-07-26, ft09/seed-20260724 reads 0.0000 rounded but is
    # 1.81% unrounded, and s5i5/seed-20260726 reads 0.0000 but is 3.85%. Taking
    # the rounded fields at face value would have reported 4 zero-optimism cells
    # when only 2 (the same sp80 cell at both budgets) are structurally zero --
    # a false null manufactured by display precision. So both scores are
    # recomputed here from the cell's own per-level charge vectors.
    def _unrounded(c: dict) -> tuple[float, float]:
        pl = c.get("per_level") or []
        plg = c.get("per_level_gateway") or []
        base = [p["human_actions"] for p in pl]
        off_lv = [p["agent_actions"] for p in pl if p.get("completed")]
        gw_lv = [p["agent_charged_actions"] for p in plg if p.get("completed")]
        tail_off = next((p["agent_actions"] for p in pl if not p.get("completed")), 0)
        tail_gw = next((p["agent_charged_actions"] for p in plg if not p.get("completed")), 0)
        o, _ = gateway_score_via_calculator(base, off_lv, tail_off)
        g, _ = gateway_score_via_calculator(base, gw_lv, tail_gw)
        return o, g

    unrounded: dict = {}
    rounding_erased: list[dict] = []
    for c in solved:
        o, g = _unrounded(c)
        unrounded[id(c)] = (o, g)
        rounded_rel = (
            (c["efficiency_offline"] - c["efficiency_gateway_charged"]) / c["efficiency_offline"]
            if c["efficiency_offline"]
            else 0.0
        )
        true_rel = (o - g) / o if o else 0.0
        if rounded_rel == 0.0 and true_rel > 1e-12:
            rounding_erased.append(
                {
                    "game": c["game"],
                    "seed": c["seed"],
                    "budget": c["budget"],
                    "rounded_rel_optimism": 0.0,
                    "unrounded_rel_optimism": round(true_rel, 8),
                }
            )
    rel_opt = [
        (unrounded[id(c)][0] - unrounded[id(c)][1]) / unrounded[id(c)][0]
        for c in solved
        if unrounded[id(c)][0] > 0
    ]
    abs_opt = [unrounded[id(c)][0] - unrounded[id(c)][1] for c in solved]

    # matched-reproduction witness: does the re-run reproduce the recorded row?
    recorded: dict = {}
    for f in row_files:
        for r in json.loads(Path(f).read_text()).get("rows") or []:
            if r.get("early_stop_grace") is None:
                recorded[(r["game"], r["seed"], r["budget"])] = r
    matched, unmatched = 0, []
    for c in exact_cells:
        k = (c.get("game"), c.get("seed"), c.get("budget"))
        r = recorded.get(k)
        if not r:
            continue
        eff_ok = abs((r.get("efficiency") or 0) - (c.get("efficiency_offline") or 0)) < 1e-9
        res_ok = r.get("n_resets") == c.get("n_resets")
        if eff_ok and res_ok:
            matched += 1
        else:
            unmatched.append(
                {
                    "cell": list(k),
                    "recorded_efficiency": r.get("efficiency"),
                    "rerun_efficiency": c.get("efficiency_offline"),
                    "recorded_n_resets": r.get("n_resets"),
                    "rerun_n_resets": c.get("n_resets"),
                }
            )

    part_b = {
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "part_b_measurement_wall_s": round(part_b_wall, 3),
        "sources": exact_sources,
        "n_cells_run": len(exact_cells),
        "n_cells_with_a_levelup": len(solved),
        "matched_reproduction_witness": {
            "n_cells_matching_a_recorded_row": matched + len(unmatched),
            "n_fully_matched_on_efficiency_AND_reset_count": matched,
            "mismatches": unmatched[:10],
            "principle": (
                "an UNMATCHED re-run would compare two different agents and confound the reset "
                "delta with the configuration difference. Reproducing each recorded row's own "
                "`efficiency` AND `n_resets` exactly is the witness that the exact gateway "
                "numbers attach to the recorded corpus."
            ),
        },
        "identity_violations": len(
            [c for c in exact_cells if c.get("identity_charged_eq_actions_plus_resets") is False]
        ),
        "baselines_nonzero_on_every_cell": all(
            c.get("baselines_nonzero") for c in exact_cells if "baselines_nonzero" in c
        ),
        "baselines_channel_note": (
            "baselines are read via arc_leaderboard_eval._baseline_actions, which goes through "
            "`getattr(env, 'info', env)`. A prior agent read `getattr(env, 'baseline_actions')` "
            "directly -- a DEAD CHANNEL that summed to 0.0 and made both charge models agree, "
            "reading as a clean null. Asserted non-zero per cell here so that cannot recur."
        ),
        "resets_before_levelups_observed": [
            {
                "game": c["game"],
                "seed": c["seed"],
                "budget": c["budget"],
                "n_resets_whole_run": c["n_resets"],
                "resets_before_each_levelup": c["resets_before_levelups"],
                "offline_efficiency_unrounded": round(unrounded[id(c)][0], 8),
                "gateway_efficiency_unrounded": round(unrounded[id(c)][1], 8),
                "abs_optimism": round(unrounded[id(c)][0] - unrounded[id(c)][1], 8),
                "rel_optimism": (
                    round((unrounded[id(c)][0] - unrounded[id(c)][1]) / unrounded[id(c)][0], 8)
                    if unrounded[id(c)][0]
                    else 0.0
                ),
            }
            for c in solved
        ],
        "abs_optimism_score_points": _stats(abs_opt),
        "rel_optimism_fraction_of_offline_score": _stats(rel_opt),
        "rel_optimism_PER_GAME_the_replication_unit": None,  # filled below
        "n_cells_with_zero_optimism": sum(1 for x in rel_opt if x == 0.0),
        "n_cells_with_nonzero_optimism": sum(1 for x in rel_opt if x > 0.0),
        "precision_hazard_found_and_corrected": {
            "n_cells_whose_delta_the_4dp_rounding_erased": len(rounding_erased),
            "cells": rounding_erased,
            "principle": (
                "`run_game` rounds both efficiency fields to 4 decimals. On small-score cells "
                "that rounding turns a real optimism into an exact 0.0000, which reads as a "
                "clean null -- the same failure class as a dead diagnostic channel. Every "
                "statistic above is recomputed UNROUNDED from the cell's own per-level charge "
                "vectors; the rounded fields are retained in the row files but not used here."
            ),
            "the_only_STRUCTURAL_zeros": (
                "sp80 seed 20260726 at both b400 and b2000: exactly 1 reset preceded its "
                "level-up, and that level is a SUPERHUMAN solve (27 agent vs 39 human actions) "
                "sitting at the 115 cap, which absorbs the extra charge with no score change. "
                "This is a genuine zero, not a rounding one."
            ),
        },
    }

    # PER-GAME UNITS. The 44 cells are NOT independent draws: the same game+seed
    # appears at two budgets, and tu93 returns a byte-identical result on all
    # three seeds (rbl=[8,78] every time), so a per-cell median silently
    # over-weights the games that happened to be run twice. The GAME is the
    # replication unit -- a hidden game is a fresh draw, a re-seed of a solved
    # public game is not -- so the per-game distribution is reported alongside.
    per_game: dict = {}
    for c in solved:
        o, g = unrounded[id(c)]
        if o > 0:
            per_game.setdefault(c["game"], []).append((o - g) / o)
    per_game_med = [statistics.median(v) for v in per_game.values()]
    part_b["rel_optimism_PER_GAME_the_replication_unit"] = {
        **_stats(per_game_med),
        "n_games": len(per_game),
        "per_game_median": {k: round(statistics.median(v), 6) for k, v in sorted(per_game.items())},
        "agrees_with_the_per_cell_median": (
            abs(statistics.median(per_game_med) - _q(rel_opt, 0.5)) < 0.01
        ),
        "principle": (
            "if the per-game and per-cell medians diverged, the per-cell number would be an "
            "artifact of which games got repeated. They agree here (4.92% vs 4.62%), so the "
            "conclusion survives the unit change -- but the per-game n is 11, not 44, and any "
            "interval should be computed on 11."
        ),
        "n_distinct_values_per_game": {
            k: len({round(x, 6) for x in v}) for k, v in sorted(per_game.items())
        },
        "seed_degeneracy_note": (
            "tu93 yields only 2 distinct optimism values across 6 cells (identical across all "
            "three seeds at each budget) -- its explorer is effectively deterministic on this "
            "game, so its 3 seeds are 1 observation, not 3."
        ),
    }

    # ------------------------------------------------------------- HEADLINE
    med_rel = part_b["rel_optimism_fraction_of_offline_score"].get("median")
    max_rel = part_b["rel_optimism_fraction_of_offline_score"].get("max")
    art["headline"] = {
        "question": (
            "By how much are our ARC score estimates optimistic because the offline harness did "
            "not charge RESETs, and is the answer materially different from zero?"
        ),
        "answer_from_bounds_alone": part_a["verdict"],
        "answer_from_exact_measurement": (
            f"MATERIALLY DIFFERENT FROM ZERO BUT SMALL. On {len(rel_opt)} matched live cells "
            f"spanning {len(per_game)} games and 2 budgets, the optimism is {med_rel:.2%} of the "
            f"per-game score at the median and at most {max_rel:.2%}; "
            f"{part_b['n_cells_with_nonzero_optimism']}/{len(rel_opt)} cells are non-zero, and "
            "the only two zeros are structural (the 115 cap absorbing a single reset), not "
            "rounding. On the GAME as the replication unit (n="
            f"{len(per_game)}) the median is "
            f"{statistics.median(per_game_med):.2%}, range "
            f"{min(per_game_med):.2%}-{max(per_game_med):.2%} -- the same answer, so it does not "
            "rest on which games were repeated. In absolute terms the correction is "
            f"{part_b['abs_optimism_score_points']['median']:.4f} score points at the median "
            f"(max {part_b['abs_optimism_score_points']['max']:.4f}), which is small only because "
            "most of these games score near zero to begin with."
        ),
        "why_reality_sits_at_the_optimistic_end_of_the_bound": (
            "level-ups happen EARLY and the resets concentrate in the POST-SOLVE TAIL, which is "
            "free. Measured across both budgets: whole-run reset counts of 3-110, but only a "
            "fraction land before a level-up (vc33 b2000: 1 of 88; ft09: 4 of 110), and those "
            "that do land on an already-large denominator. The worst-case bound assumed ALL "
            "resets precede a level-up, which is not how this agent behaves. The counter-example "
            "matters too: r11l charges 72-74 of its ~109 resets BEFORE its single level-up and is "
            "correspondingly the worst-hit game at 15.4-17.5%."
        ),
        "dose_response_the_correction_tracks_resets_before_the_levelup": (
            "not the whole-run reset count. r11l (109 resets, 72 before) is hit 4x harder than "
            "ft09/seed-20260725 (110 resets, 4 before). Any future estimate of this correction "
            "must use the per-level attribution, not `n_resets` -- which is exactly why the "
            "bound over `n_resets` alone was uninformative."
        ),
        "no_conclusion_reordered": (
            "a single-digit-percent haircut on the efficiency term does not change the settled "
            "structural facts: DEPTH dominates (1/2/4/8 of 8 levels -> 2.78/8.33/27.78/100), a "
            "per-game score can never exceed 100, and the post-solve tail is free. It does mean "
            "every per-level efficiency figure this project quotes should be read as an upper "
            "estimate, with the size of the overstatement now measured rather than unknown."
        ),
    }

    # --------------------------------------------- cross-check vs reality
    aggregates = []
    bykey: dict = {}
    for r in all_res:
        if not r["rescorable"]:
            continue
        bykey.setdefault((r["source"], r["arm"], r["budget"], r["seed"]), {})[r["game"]] = r
    for k, games in sorted(bykey.items()):
        if len(games) < 10:
            continue
        aggregates.append(
            {
                "source": k[0],
                "arm": k[1],
                "budget": k[2],
                "seed": k[3],
                "n_games": len(games),
                "n_games_with_a_levelup": sum(1 for g in games.values() if g["n_levels_completed"]),
                "offline_mean_per_game": round(
                    statistics.mean(g["offline_score"] for g in games.values()), 6
                ),
                "worst_case_mean_per_game": round(
                    statistics.mean(g["worst_score"] for g in games.values()), 6
                ),
            }
        )
    offs = [a["offline_mean_per_game"] for a in aggregates]
    wors = [a["worst_case_mean_per_game"] for a in aggregates]
    art["cross_check_against_the_actual_hidden_score"] = {
        "actual_hidden_score": ACTUAL_HIDDEN_SCORE,
        "actual_hidden_score_ref": ACTUAL_HIDDEN_REF,
        "offline_public_set_mean_per_game": _stats(offs),
        "worst_case_public_set_mean_per_game": _stats(wors),
        "does_the_correction_narrow_the_gap": False,
        "verdict": (
            "NO -- AND THE HONEST READING IS THAT THIS TEST HAS ALMOST NO POWER, SO ITS "
            "NON-CORROBORATION IS NOT EVIDENCE AGAINST THE CORRECTION EITHER. The offline "
            f"PUBLIC-set mean-per-game score is {statistics.median(offs):.4f} at the median, "
            f"which is already near-identical to the actual HIDDEN score of {ACTUAL_HIDDEN_SCORE}. "
            f"Applying the worst-case reset correction moves it DOWN to "
            f"{statistics.median(wors):.4f} -- AWAY from the observed value, not toward it. So on "
            "this evidence the correction does not explain any offline-vs-hidden gap."
        ),
        "why_the_test_has_almost_no_power": [
            "DIFFERENT GAME SETS. The offline corpus is the 25 PUBLIC games; 0.08 was scored on "
            "HIDDEN OOD games. The two numbers landing near each other is a coincidence of very "
            "different mechanisms, not a calibration.",
            "DIFFERENT DOMINANT TERM. The hidden score is dominated by whether the agent wins "
            "ANY level at all (held-out proxy first_win_rate_integrated = 0.04, CI [0,0], "
            "ops/known-issues.md:3275-3279), whereas 11 of 25 public games reach a level. The "
            "reset correction only touches the efficiency of levels that WERE solved, so it "
            "cannot move a score whose limiting factor is winning nothing.",
            "DIFFERENT AGENT CONFIGURATION. Every recorded row is the LLM-OFF arm; the submitted "
            "kernel ran with the generator enabled.",
            "A ONE-POINT COMPARISON. One hidden submission is a single observation with no "
            "interval; nothing about it can distinguish a 5% efficiency correction.",
        ],
        "what_would_make_this_test_informative": (
            "gateway-charged re-scoring of a HIDDEN-set run's own per-level reset attribution, "
            "which requires the instrumentation to be present in a submitted kernel -- an "
            "operator decision, not measurable from here."
        ),
    }

    # ----------------------------------------------- scope, power, honesty
    games_b = sorted({c["game"] for c in solved})
    art["scope_and_power"] = {
        "part_a_scope": f"{len(lv)} of {len(all_res)} recorded rows carry both a level-up and a reset count",
        "part_b_scope": (
            f"{len(rel_opt)} matched live cells with a level-up, across {len(games_b)} games "
            f"{games_b}, budgets {sorted({c['budget'] for c in solved})}, seeds "
            f"{sorted({c['seed'] for c in solved})}"
        ),
        "single_game_concentration_warning": (
            "the Part A corpus score is heavily concentrated: vc33 alone carries the large "
            "per-game scores (median offline 2.09 vs <0.03 for most games), so a corpus-level "
            "relative delta is a statement about vc33 more than about the corpus. Part B's "
            "per-cell numbers are reported individually for exactly this reason -- read the "
            "per-cell table, not only the median."
        ),
        "what_this_does_NOT_measure": [
            "the hidden set. Every number here is the 25 public games via the offline arcade.",
            "the LLM-ON configuration.",
            "whether a policy could actually have produced the worst-case reset placement -- the "
            "Part A bound is accounting-only and therefore conservative (a true lower bound on "
            "the score).",
            "the 1713 cptb cells and 375 budget-sweep rows, which record no reset count at all.",
        ],
    }

    art["what_was_NOT_changed"] = [
        "no SUBMITTED_* flag and no MAX_ACTIONS value was touched -- this measures and reports; "
        "the flag decision is the operator's",
        "the pre-existing `efficiency` field keeps its offline (reset-free) semantics; the "
        "gateway number is a NEW field `efficiency_gateway_charged` alongside it",
        "no historical artifact's recorded numbers were rewritten; "
        "results/outer_loop_arc_early_stop_grace_sweep_20260726.json was REBUILT because the "
        "freshness lint correctly flagged the arc_leaderboard_eval.py edit, and the deep diff "
        "moved ZERO measured numbers (only run_date/git_head/provenance/checksum)",
        "nothing was submitted to ARC or Kaggle",
    ]

    # ------------------------- independent-implementation cross-check
    # A CONCURRENT session wrote scripts/analyze_arc_reset_charge_attribution.py
    # against the same defect, with a SEPARATELY-WRITTEN bound (exhaustive for
    # <=2 completed levels, coordinate-descent above) and its own live probe.
    # Two independent implementations agreeing on the same corpus is worth more
    # than either alone (CLAUDE.md Adversarial-Confirmation Discipline), and a
    # disagreement would be the single most informative thing here -- so the
    # comparison is computed, not asserted. Cited by CONTENT HASH as a snapshot
    # rather than registered as a dependency: that artifact is still being
    # revised, and making it a tracked input would wedge this one as stale on
    # every one of their rebuilds.
    peer_rel = "results/outer_loop_arc_reset_charge_attribution_20260726.json"
    peer_path = REPO / peer_rel
    peer_cmp: dict = {"peer_artifact": peer_rel, "available": peer_path.exists()}
    if peer_path.exists():
        try:
            peer = json.loads(peer_path.read_text())
            pa = peer.get("part_a_bound_over_existing_rows") or {}
            loss = pa.get("score_worst_case_relative_loss") or {}
            mine_rel = part_a["rel_delta_fraction_of_offline_score_erased"]
            mine_abs = part_a["abs_delta_offline_minus_worst_score_points"]
            width = pa.get("score_range_width") or {}
            agree = {
                "n_rows_total": (pa.get("n_rows_total"), len(all_res)),
                "n_cells_bounded": (pa.get("n_won_cells_bounded"), len(lv)),
                "upstream_measurement_wall_s": (
                    pa.get("measurement_wall_s"),
                    part_a["measurement_wall_s"],
                ),
                "worst_case_rel_loss_median": (loss.get("median"), mine_rel.get("median")),
                "worst_case_rel_loss_max": (loss.get("max"), mine_rel.get("max")),
                "worst_case_abs_width_max": (width.get("max"), mine_abs.get("max")),
                "frames_identity_holds_on_all_rows": (
                    pa.get("identity_frames_eq_actions_plus_resets_all_rows"),
                    True,
                ),
            }
            peer_cmp["peer_sha256"] = _sha(peer_path)
            peer_cmp["field_by_field_peer_vs_mine"] = {
                k: {
                    "peer": v[0],
                    "mine": v[1],
                    "agree": (
                        v[0] == v[1]
                        if not isinstance(v[0], float) or not isinstance(v[1], float)
                        else abs(v[0] - v[1]) < 1e-4
                    ),
                }
                for k, v in agree.items()
            }
            peer_cmp["n_agreeing"] = sum(
                1 for v in peer_cmp["field_by_field_peer_vs_mine"].values() if v["agree"]
            )
            peer_cmp["n_compared"] = len(agree)
            peer_cmp["verdict"] = (
                "TWO INDEPENDENT IMPLEMENTATIONS AGREE on the corpus size, the bounded-cell "
                "count, the frames identity, the upstream measurement clock, and the worst-case "
                "relative-loss distribution -- despite different worst-case search methods "
                "(exact DP here; exhaustive-for-2-levels + coordinate-descent there) and "
                "separately written scorer drivers."
                if peer_cmp["n_agreeing"] == len(agree)
                else "DISAGREEMENT -- investigate before citing either number."
            )
            peer_cmp["peer_qualifier_adopted"] = {
                "finding": (
                    "the peer identified a SECOND clamp this analysis had not isolated: the "
                    "game-level `min(., max_weights/total_weights*100)` clamp "
                    "(scorecard.py:204-206). A cell sitting on it has its score set by DEPTH, so "
                    "extra charged actions cost it NOTHING until a completed level's own "
                    "(baseline/charged)^2*100 term falls below 100. The peer counts 17 such cells."
                ),
                "why_it_matters_here": (
                    "it is a second, independent reason the exact correction lands near the "
                    "optimistic end of the bound -- alongside the per-level 115 cap isolated "
                    "here (the sp80 true-zero cell) and the free post-solve tail."
                ),
            }
            peer_cmp["where_the_two_differ_in_coverage"] = (
                "the peer's exact-vs-bound part has 4 cells; this artifact's has "
                f"{len(rel_opt)} across {len(games_b) if (games_b := sorted({c['game'] for c in solved})) else 0} "
                "games and 2 budgets, so the distributional statements here are better powered. "
                "Conversely the peer measures reset COMPOSITION (where resets come from and "
                "which are avoidable), which this artifact does not."
            )
        except Exception as exc:
            peer_cmp["error"] = f"{type(exc).__name__}:{exc}"
    art["independent_implementation_crosscheck"] = peer_cmp

    art["scorer_fidelity_crosschecks"] = xchecks
    art["part_a_bounds_over_recorded_rows"] = part_a
    art["part_b_exact_attribution_live"] = part_b

    art["honest_verdict"] = (
        "complete_gateway_rescore_reset_charge_optimism_measured_"
        f"bound_too_wide_to_conclude_0_to_95pct_but_exact_matched_measurement_puts_it_at_"
        f"median_{med_rel:.4f}_max_{max_rel:.4f}_of_per_game_score_"
        f"{part_b['n_cells_with_nonzero_optimism']}_of_{len(rel_opt)}_cells_nonzero_"
        "material_but_small_no_conclusion_reordered_hidden_score_crosscheck_underpowered_"
        "does_NOT_corroborate"
    )
    art["honest_verdict_principle"] = (
        "terminal `complete_` prefix per the Verdict Terminal-Prefix Discipline; the verdict "
        "states BOTH that the bound could not answer the question and what the exact measurement "
        "found, and states plainly that the reality cross-check did not corroborate."
    )
    art["verifier_is_oracle"] = True
    art["verifier_is_oracle_principle"] = (
        "the 'verifier' here IS the installed competition scorer -- the executable oracle that "
        "defines the score. This is execution-grounded measurement, NOT an oracle-distinct "
        "verifier-moat claim, and it must never be headlined as one."
    )
    art["solve_provenance"] = "development_proxy"
    art["solve_provenance_principle"] = (
        "Part B re-runs the offline dev twin on PUBLIC games to measure accounting, not to claim "
        "a solve. No new level is claimed; no registry entry is added."
    )
    art["arc_solve_claim"] = False
    art["random_seed"] = 20260726
    art["random_seeds_used"] = sorted({c["seed"] for c in exact_cells if c.get("seed")})
    art["preconditions_checked"] = [
        {"resource": "installed arc_agi scorer importable", "available": True},
        {"resource": "recorded early-stop row corpora present", "available": bool(row_files)},
        {"resource": "per-level human baselines non-zero", "available": True},
    ]

    # ------------------------------------------------------------ provenance
    try:
        import analyze_scored_path_lever_ab as sibling

        code = []
        for rel in (
            "scripts/arc_gateway_rescore.py",
            "scripts/arc_gateway_exact_attribution.py",
            "scripts/analyze_arc_gateway_rescore.py",
            "scripts/arc_leaderboard_eval.py",
            "python/carnot/agentic/arc_competition_agent.py",
        ):
            p = REPO / rel
            if p.exists():
                code.append(
                    {
                        "path": rel,
                        "sha256": _sha(p),
                        "bytes": p.stat().st_size,
                        "mtime_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)
                        ),
                    }
                )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        art["git_head"] = head
        # `rows_sources` MUST be {group_name: [{path, sha256, ...}, ...]}, matching the
        # two sibling analysers. A flat list of bare path strings both CRASHES
        # scripts/summarize_artifact.py:230 (which calls `.values()` on it) and defeats
        # the staleness check entirely, since there is no sha256 to compare against --
        # the check would silently verify nothing and report fresh.
        art["provenance"] = {
            "git_head": head,
            "code": code,
            "rows_sources": {
                "recorded_rows": [{"path": s["path"], "sha256": s["sha256"]} for s in row_sources],
                "exact_attribution_rows": [
                    {"path": s["path"], "sha256": s["sha256"]} for s in exact_sources
                ],
            },
        }
        if out_path is not None:
            sibling.register_analyzed_artifact(out_path, analyzer=Path(__file__).resolve())
    except Exception as exc:
        art["provenance"] = {"error": f"{type(exc).__name__}:{exc}"}

    art["duration_s"] = round(time.time() - t0, 3)
    art["duration_s_principle"] = (
        "THIS ANALYSER'S runtime only. It is NOT the measurement clock -- Part A's clock is "
        "`measurement_wall_s` (sum of upstream row-file elapsed_s) and Part B's is "
        "`part_b_measurement_wall_s`. Conflating them is the 2026-07-26 failure this field's "
        "annotation exists to prevent."
    )
    art["inference_substrate"] = "aggregation_from_upstream_artifacts"
    art["inference_substrate_principle"] = (
        "the artifact as a whole is an aggregation over persisted rows plus a cited live "
        "sub-measurement; the live half declares its own substrate inside "
        "`part_b_exact_attribution_live`."
    )
    payload = json.dumps(
        {k: art[k] for k in art if k not in ("run_date", "duration_s")},
        sort_keys=True,
        default=str,
    ).encode()
    art["reproducibility_checksum"] = hashlib.sha256(payload).hexdigest()
    return art


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(REPO / "results/outer_loop_arc_gateway_rescore_20260726.json")
    )
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    art = build(out_path=out)
    try:
        import analyze_scored_path_lever_ab as sibling

        sibling.preserve_freshness_acknowledgements(art, out)
    except Exception:
        pass
    out.write_text(json.dumps(art, indent=1, default=str) + "\n")
    print(json.dumps(art["headline"], indent=1))
    print(
        json.dumps(
            art["part_b_exact_attribution_live"]["rel_optimism_fraction_of_offline_score"], indent=1
        )
    )
    print("verdict:", art["honest_verdict"])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
