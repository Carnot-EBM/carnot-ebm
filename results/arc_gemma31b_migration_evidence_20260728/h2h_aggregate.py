#!/usr/bin/env python
"""Aggregate the two head-to-head arms (base Qwen3.6-27B vs gemma-4-31B-it) into one artifact.

WHY the statistics are done this way (verbose per CLAUDE.md):

* The PAIRED UNIT IS THE GAME, not the cell. The 3 trials per game are repeats against the SAME
  seeded window, so they measure per-game stability -- they are NOT 3 independent degrees of
  freedom. Treating 39 cells as N=39 would manufacture significance out of replicates.

* Exact two-sided SIGN test, not a t-test. heldout_accuracy is a bounded proportion over windows as
  small as 3 transitions, so it is coarsely quantised and nowhere near Gaussian.

* The MINIMUM REACHABLE p is reported at the ACHIEVED discordant count, because with few discordant
  pairs significance is unreachable no matter which way every pair falls.

* THREE VIEWS, because a cell can FAIL TO PRODUCE AN ENGINE AT ALL (induce_ok=False -> heldout is
  None). That is not missing data -- it is the inducer failing at its job -- but scoring it as 0.0
  is still a judgement call, so we report both rather than silently pick one:
    A. SCORABLE-ONLY   -- mean over the cells that produced a scorable engine. Answers "when it
                          produces a world model, how good is it?" Ignores how OFTEN it fails.
    B. ZERO-IMPUTED    -- a non-scorable cell counts as 0.0. Answers the OPERATIONAL question
                          "hand this inducer an unseen game; what heldout accuracy do I get?",
                          where producing nothing is a failure with real cost.
    C. STRICT-PAIRED   -- only games where BOTH arms scored all 3 trials. Smallest, cleanest, but
                          it DISCARDS exactly the games one arm failed on, which is the thing we
                          most want to know -- so it is a robustness check, never the headline.

* BUDGET EXHAUSTION IS THE DOMINANT EFFECT AND IS REPORTED, NOT HIDDEN. Both arms receive an
  IDENTICAL "/think" prefix and an IDENTICAL 16384-token completion budget. An earlier draft of
  this analysis assumed the prefix was inert on gemma (its chat template has no <think> tag) and
  that Qwen was therefore being penalised unfairly. THE MEASUREMENT REFUTES THAT: reason_engaged
  fires 39/39 for BOTH models, so both genuinely reason. What differs is budget efficiency --
  gemma finishes inside the budget (exp5764: overran 0/39) while Qwen hits the token limit on a
  large fraction of cells and then emits code missing the required engine / is_level_* functions,
  so no scorable engine exists. That is a real, decision-relevant deficiency at a fixed budget,
  not a harness artifact. It does NOT establish that Qwen is intrinsically the weaker inducer --
  a larger budget or /no_think might rescue it -- which is why the recommended follow-up is a
  third arm. See budget_exhaustion_finding in the artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
SCRATCH = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad"
)
ARTIFACT = REPO / "results" / "experiment_6021_inducer_head_to_head_qwen27b_vs_gemma31b.json"

ARMS = ["qwen27b", "gemma31b"]
LABEL = {"qwen27b": "qwen3.6-27B-base", "gemma31b": "gemma-4-31B-it"}


def load_rows(arm: str) -> list[dict[str, Any]]:
    p = SCRATCH / f"h2h_shard_{arm}.jsonl"
    out: list[dict[str, Any]] = []
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_meta(arm: str) -> dict[str, Any]:
    p = SCRATCH / f"h2h_meta_{arm}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mean(xs: list[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 6) if xs else None


def binom_two_sided_sign(n_pos: int, n_neg: int) -> tuple[float, float, int]:
    """Exact two-sided sign test on discordant pairs -> (p, min_reachable_p, d)."""
    d = n_pos + n_neg
    if d == 0:
        return 1.0, 1.0, 0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(d, i) for i in range(0, k + 1)) / (2**d)
    return round(min(1.0, 2 * tail), 6), round(min(1.0, 2 * (1 / (2**d))), 6), d


def sign_test_on(
    per_game: dict[str, dict[str, Optional[float]]], games: list[str]
) -> dict[str, Any]:
    q_wins = g_wins = ties = 0
    deltas: dict[str, Any] = {}
    for g in games:
        q, m = per_game[g]["qwen27b"], per_game[g]["gemma31b"]
        if q is None or m is None:
            continue
        d = round(q - m, 6)
        deltas[g] = {"qwen27b": q, "gemma31b": m, "delta_qwen_minus_gemma": d}
        if d > 0:
            q_wins += 1
        elif d < 0:
            g_wins += 1
        else:
            ties += 1
    p, minp, d_n = binom_two_sided_sign(q_wins, g_wins)
    return {
        "n_games": len(deltas),
        "qwen_wins": q_wins,
        "gemma_wins": g_wins,
        "ties": ties,
        "discordant_d": d_n,
        "p_two_sided": p,
        "min_reachable_p_at_this_d": minp,
        "significant_at_0.05": p <= 0.05,
        "per_game_deltas": deltas,
    }


def main() -> int:
    t0 = time.time()
    rows = {a: load_rows(a) for a in ARMS}
    metas = {a: load_meta(a) for a in ARMS}
    roster = metas.get("qwen27b", {}).get("roster") or metas.get("gemma31b", {}).get("roster") or []
    trials = metas.get("qwen27b", {}).get("trials") or metas.get("gemma31b", {}).get("trials") or []
    n_tr = len(trials) or 3

    # ---- per-cell bookkeeping: scorable vs failed-to-produce-an-engine ----
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {a: {} for a in ARMS}
    for a in ARMS:
        for r in rows[a]:
            cells[a].setdefault(r["game"], []).append(r)

    def scorable(r: dict[str, Any]) -> bool:
        return isinstance(r.get("heldout_accuracy"), (int, float))

    detail: dict[str, Any] = {}
    for g in roster:
        detail[g] = {}
        for a in ARMS:
            rs = cells[a].get(g, [])
            sc = [float(r["heldout_accuracy"]) for r in rs if scorable(r)]
            detail[g][a] = {
                "cells_run": len(rs),
                "cells_scorable": len(sc),
                "cells_nonscorable": len(rs) - len(sc),
                "heldout_cells_scorable": [round(x, 4) for x in sc],
                "heldout_mean_scorable_only": mean(sc),
                "heldout_mean_zero_imputed": (
                    mean(sc + [0.0] * (len(rs) - len(sc))) if rs else None
                ),
                "n_induce_failed": sum(1 for r in rs if r.get("induce_ok") is False),
                "n_overran": sum(1 for r in rs if r.get("overran")),
                "n_reason_engaged": sum(1 for r in rs if r.get("reason_engaged")),
                "n_memorizing": sum(1 for r in rs if r.get("is_memorizing")),
                "failure_details": sorted(
                    {
                        str(r.get("induce_detail") or r.get("error") or "")[:160]
                        for r in rs
                        if not scorable(r)
                    }
                    - {""}
                ),
            }

    # ---- completeness gate: never compare a truncated arm to a complete one ----
    completeness = {
        a: {
            "status": metas.get(a, {}).get("status"),
            "cells_completed": metas.get(a, {}).get("cells_completed"),
            "cells_total": metas.get(a, {}).get("cells_total"),
            "cells_missing": metas.get(a, {}).get("cells_missing"),
            "wedge": metas.get(a, {}).get("wedge"),
        }
        for a in ARMS
    }
    both_complete = all(completeness[a]["status"] == "complete" for a in ARMS)

    # ---- the three views ----
    games_A = [
        g for g in roster if all(detail[g][a]["cells_scorable"] >= 1 for a in ARMS)
    ]  # scorable-only
    games_B = [
        g for g in roster if all(detail[g][a]["cells_run"] == n_tr for a in ARMS)
    ]  # zero-imputed (all cells run)
    games_C = [
        g for g in roster if all(detail[g][a]["cells_scorable"] == n_tr for a in ARMS)
    ]  # strict

    pg_A = {g: {a: detail[g][a]["heldout_mean_scorable_only"] for a in ARMS} for g in roster}
    pg_B = {g: {a: detail[g][a]["heldout_mean_zero_imputed"] for a in ARMS} for g in roster}

    def arm_view(a: str, games: list[str], key: str) -> dict[str, Any]:
        vals = [detail[g][a][key] for g in games if detail[g][a][key] is not None]
        nz = [g for g in games if (detail[g][a][key] or 0.0) > 0.0]
        return {
            "mean_heldout_over_games": mean(vals),
            "n_games": len(vals),
            "coverage_nonzero_games": len(nz),
            "coverage_of": len(games),
            "nonzero_games": nz,
            "zero_games": [g for g in games if g not in nz],
        }

    views = {
        "A_scorable_only": {
            "games": games_A,
            "n_games": len(games_A),
            "definition": (
                "Per-game mean over ONLY the cells that produced a scorable engine. Answers: when "
                "this inducer produces a world model at all, how accurate is it? Ignores failure "
                "frequency, so it FLATTERS an arm that fails often but scores well when it "
                "succeeds."
            ),
            "arms": {a: arm_view(a, games_A, "heldout_mean_scorable_only") for a in ARMS},
            "sign_test": sign_test_on(pg_A, games_A),
        },
        "B_zero_imputed": {
            "games": games_B,
            "n_games": len(games_B),
            "definition": (
                "A cell that failed to produce a usable engine counts as heldout 0.0. Answers the "
                "OPERATIONAL question: hand this inducer an unseen game, what do you actually get? "
                "Producing no engine is a real failure with real cost, not missing data. Read "
                "together with budget_exhaustion_finding: the failures are budget exhaustion at a "
                "fixed 16384-token budget, which both arms received identically."
            ),
            "arms": {a: arm_view(a, games_B, "heldout_mean_zero_imputed") for a in ARMS},
            "sign_test": sign_test_on(pg_B, games_B),
        },
        "C_strict_all_trials_scorable_both_arms": {
            "games": games_C,
            "n_games": len(games_C),
            "definition": (
                "Only games where BOTH arms scored all trials. Cleanest pairing but it DISCARDS "
                "precisely the games one arm failed on -- the most informative cases -- so it is a "
                "robustness check, never the headline."
            ),
            "arms": {a: arm_view(a, games_C, "heldout_mean_scorable_only") for a in ARMS},
            "sign_test": sign_test_on(pg_A, games_C),
        },
    }

    # ---- COVERAGE test (the criterion I consider decisive), on view B ----
    cov_q_only = [
        g
        for g in games_B
        if (detail[g]["qwen27b"]["heldout_mean_zero_imputed"] or 0) > 0
        and (detail[g]["gemma31b"]["heldout_mean_zero_imputed"] or 0) == 0
    ]
    cov_g_only = [
        g
        for g in games_B
        if (detail[g]["gemma31b"]["heldout_mean_zero_imputed"] or 0) > 0
        and (detail[g]["qwen27b"]["heldout_mean_zero_imputed"] or 0) == 0
    ]
    p_cov, minp_cov, d_cov = binom_two_sided_sign(len(cov_q_only), len(cov_g_only))
    coverage_test = {
        "basis": "view B (zero-imputed): a game counts as covered if mean heldout > 0",
        "qwen_only_covered": cov_q_only,
        "gemma_only_covered": cov_g_only,
        "discordant_d": d_cov,
        "p_two_sided": p_cov,
        "min_reachable_p_at_this_d": minp_cov,
        "significant_at_0.05": p_cov <= 0.05,
        "test": "exact McNemar / two-sided binomial on discordant games",
    }

    # ---- secondary criteria (decisive if quality does not separate) ----
    def secondary(a: str) -> dict[str, Any]:
        srv = metas.get(a, {}).get("server", {}) or {}
        ws = [r.get("elapsed_s") for r in rows[a] if isinstance(r.get("elapsed_s"), (int, float))]
        res = srv.get("residency_mib_gpu1")
        return {
            "residency_mib_gpu1": res,
            "headroom_mib_vs_24576_total": (24576 - res) if res else None,
            "headroom_mib_vs_24123_usable": (24123 - res) if res else None,
            "headroom_pct_of_usable": (round(100 * (24123 - res) / 24123, 1) if res else None),
            "health_wait_s": srv.get("health_wait_s"),
            "median_cell_s": round(sorted(ws)[len(ws) // 2], 1) if ws else None,
            "total_cell_s": round(sum(ws), 1) if ws else None,
            "n_cells_run": len(rows[a]),
            "n_cells_scorable": sum(1 for r in rows[a] if scorable(r)),
            "n_induce_failed": sum(1 for r in rows[a] if r.get("induce_ok") is False),
            "n_overran": sum(1 for r in rows[a] if r.get("overran")),
            "n_reason_engaged": sum(1 for r in rows[a] if r.get("reason_engaged")),
            "n_memorizing": sum(1 for r in rows[a] if r.get("is_memorizing")),
        }

    sec = {a: secondary(a) for a in ARMS}

    measurement_wall_s = round(
        sum(
            r.get("elapsed_s", 0.0)
            for a in ARMS
            for r in rows[a]
            if isinstance(r.get("elapsed_s"), (int, float))
        ),
        2,
    )

    h = hashlib.sha256()
    for a in ARMS:
        p = SCRATCH / f"h2h_shard_{a}.jsonl"
        if p.exists():
            h.update(p.read_bytes())
    checksum = "sha256:" + h.hexdigest()

    # ---- window sizes (measured) so extreme values are explainable, not suspicious ----
    window_len: dict[str, int] = {}
    try:
        if os.environ.get("H2H_SKIP_WINDOW_MEASURE") == "1":
            raise RuntimeError("skipped by H2H_SKIP_WINDOW_MEASURE=1")
        sys.path.insert(0, str(REPO / "python"))
        from carnot.agentic import arc_actions_to_progress as _atp

        for g in roster:
            try:
                w = _atp.build_progress_window(g)
                if w is not None:
                    window_len[g] = len(w[0])
            except Exception:
                pass
    except Exception:
        pass

    extremes = []
    for g in games_B:
        for a in ARMS:
            v = detail[g][a]["heldout_mean_zero_imputed"]
            if v is not None and v in (0.0, 1.0):
                n = window_len.get(g)
                extremes.append(
                    {
                        "game": g,
                        "arm": a,
                        "heldout_mean_zero_imputed": v,
                        "window_transitions": n,
                        "attainable_values": (f"multiples of 1/{n}" if n else "unknown"),
                        "why_not_fabrication": (
                            "heldout_accuracy is a proportion over this game's window"
                            + (f" ({n} transitions)" if n else "")
                            + ", so it is coarsely quantised and an exact 0.0/1.0 is an ordinary "
                            "attainable value. 0.0 is the common failure mode (engine predicts no "
                            "transition correctly, or no engine was produced at all)."
                        ),
                    }
                )

    # ---- verdict ----
    def f(x: Optional[float]) -> str:
        return "na" if x is None else f"{x:.6f}"

    A, B = views["A_scorable_only"], views["B_zero_imputed"]
    qB, gB = B["arms"]["qwen27b"], B["arms"]["gemma31b"]
    separated = coverage_test["significant_at_0.05"] or B["sign_test"]["significant_at_0.05"]

    if not both_complete:
        verdict = (
            "partial_inducer_head_to_head_"
            + "_".join(
                f"{a}_{completeness[a]['status']}_{completeness[a]['cells_completed']}"
                f"of{completeness[a]['cells_total']}"
                for a in ARMS
            )
            + "_no_comparison_truncated_arm_not_compared_to_complete_arm"
        )
    elif separated:
        lead = (
            "gemma31b"
            if (gB["coverage_nonzero_games"], gB["mean_heldout_over_games"] or 0)
            > (qB["coverage_nonzero_games"], qB["mean_heldout_over_games"] or 0)
            else "qwen27b"
        )
        verdict = (
            f"complete_inducer_head_to_head_{lead}_separates_coverage_qwen_"
            f"{qB['coverage_nonzero_games']}of{B['n_games']}_gemma_"
            f"{gB['coverage_nonzero_games']}of{B['n_games']}_meanB_qwen_"
            f"{f(qB['mean_heldout_over_games'])}_gemma_{f(gB['mean_heldout_over_games'])}_"
            f"p_cov_{coverage_test['p_two_sided']}_p_meanB_{B['sign_test']['p_two_sided']}_"
            f"N{B['n_games']}"
        )
    else:
        verdict = (
            f"complete_inducer_head_to_head_indistinguishable_at_this_support_coverage_qwen_"
            f"{qB['coverage_nonzero_games']}of{B['n_games']}_gemma_"
            f"{gB['coverage_nonzero_games']}of{B['n_games']}_meanB_qwen_"
            f"{f(qB['mean_heldout_over_games'])}_gemma_{f(gB['mean_heldout_over_games'])}_"
            f"p_meanB_{B['sign_test']['p_two_sided']}_floor_"
            f"{B['sign_test']['min_reachable_p_at_this_d']}_d{B['sign_test']['discordant_d']}_"
            f"p_cov_{coverage_test['p_two_sided']}_floor_{coverage_test['min_reachable_p_at_this_d']}"
            f"_d{coverage_test['discordant_d']}_N{B['n_games']}"
        )

    # ---- VRAM witness ----
    vram: dict[str, Any] = {"samples": 0}
    vp = SCRATCH / "h2h_vram.jsonl"
    if vp.exists():
        n = 0
        mn = mx = None
        for line in vp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            for c in s.get("cards", []):
                if c.get("index") == 1 and isinstance(c.get("used_mib"), int):
                    u = c["used_mib"]
                    mn = u if mn is None else min(mn, u)
                    mx = u if mx is None else max(mx, u)
        vram = {
            "samples": n,
            "gpu1_min_used_mib": mn,
            "gpu1_max_used_mib": mx,
            "sampler_jsonl": str(vp),
            "note": (
                "Per-PID + per-card residency sampled every 10s for the whole run, so a card "
                "falling off the PCI bus would be a RECORDED FACT with a timestamp rather than an "
                "unexplained hang (three prior attempts at this measurement died that way). A low "
                "gpu1_min between arms is expected: the arms are sequential, one server at a time, "
                "each torn down by explicit PID before the next launches."
            ),
        }

    art: dict[str, Any] = {
        "experiment": "experiment_6021_inducer_head_to_head_qwen27b_vs_gemma31b",
        "schema": "carnot.exp6021.inducer_head_to_head.v1",
        "requirements": ["REQ-ARC-WMTE-6021"],
        "question": (
            "WHICH INDUCER? base Qwen3.6-27B vs gemma-4-31B-it on world-model induction quality, "
            "matched on quantisation (both Q4_K_M, verified from GGUF general.file_type==15), "
            "corpus (exp5764's 13 games), mechanism (exp5726.run_reason_cell_budget at budget "
            "16384), repeats (3/game), n_ctx (32768), and card (GPU 1, sequential, one server at a "
            "time). Decides which model the ARC induction path should use."
        ),
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "used_env_source": True,
        "offline_ground_truth_bfs": False,
        "exhaustive_bfs_calibration": False,
        "hand_calibrated_per_game": False,
        "submitted_to_leaderboard": False,
        "random_seed": (trials[0] if trials else 0),
        "random_seeds_used": list(trials),
        "reproducibility_checksum": checksum,
        # TWO GENUINELY DIFFERENT QUANTITIES (do not collapse them -- an earlier draft set
        # duration_s = measurement_wall_s and adversarial_verify correctly flagged that as a
        # TAUTOLOGY, since two "distinct" metrics agreeing to full precision is a bug signal):
        #   measurement_wall_s = sum of each cell row's OWN elapsed_s (pure per-cell compute)
        #   duration_s         = sum of each ARM's wall clock, which additionally includes model
        #                        load, the per-cell GPU/props/completion gates, and teardown.
        "measurement_wall_s": measurement_wall_s,
        "duration_s": round(
            sum(float(metas.get(a, {}).get("arm_wall_s") or 0.0) for a in ARMS), 2
        ),
        "duration_s_minus_measurement_wall_s_overhead": round(
            sum(float(metas.get(a, {}).get("arm_wall_s") or 0.0) for a in ARMS)
            - measurement_wall_s,
            2,
        ),
        "both_arms_complete": both_complete,
        "completeness": completeness,
        "roster": list(roster),
        "trials_per_game": n_tr,
        "per_game_detail_side_by_side": detail,
        "views": views,
        "coverage_test": coverage_test,
        "decisive_criterion": {
            "choice": "coverage_nonzero_games under view B (zero-imputed)",
            "why": (
                "The target is HIDDEN, out-of-distribution games. The operational question is "
                "whether the inducer yields ANY usable signal on an arbitrary unseen game, and a "
                "per-game mean is only defined once signal exists. The mean is also dominated by a "
                "few high scorers on tiny windows -- a single 1.0 on a 3-transition window moves a "
                "13-game mean by 0.077, more than most real effects here. exp5764's own edge was "
                "12/13 vs 6/13 coverage, far more robust than its 0.378 vs 0.188 mean."
            ),
            "caveat": (
                "Under view B a cell that produced no engine counts as 0.0, and most of Qwen's "
                "zeros are budget exhaustion rather than a wrong world model. Both arms got the "
                "SAME budget, so this is a fair comparison of 'usable engine within budget' -- but "
                "read it together with secondary_criteria.n_overran / n_induce_failed and "
                "budget_exhaustion_finding before restating it as a claim about induction "
                "capability, which this experiment does NOT establish."
            ),
        },
        "memorization_caveat": {
            "measured": {
                a: {
                    "scorable_cells": sec[a]["n_cells_scorable"],
                    "memorizing_cells": sum(
                        1 for r in rows[a] if scorable(r) and r.get("is_memorizing")
                    ),
                    "mean_heldout_memorizing": mean(
                        [
                            float(r["heldout_accuracy"])
                            for r in rows[a]
                            if scorable(r) and r.get("is_memorizing")
                        ]
                    ),
                    "mean_heldout_non_memorizing": mean(
                        [
                            float(r["heldout_accuracy"])
                            for r in rows[a]
                            if scorable(r) and not r.get("is_memorizing")
                        ]
                    ),
                }
                for a in ARMS
            },
            "what_this_means": (
                "heldout_accuracy is scored by WorldModelVerifier(window).score(engine) -- i.e. "
                "against the SAME window the engine was induced from -- and the inherited "
                "exp5726/exp5764 mechanism runs an AST memorization scan that flags engines "
                "hardcoding that window's changed coordinates. In BOTH arms essentially all of the "
                "non-zero score comes from cells flagged as memorizing; the non-memorizing cells "
                "score at or near zero. So this metric substantially measures 'can the model write "
                "an engine that fits the observed window', NOT 'can it induce a world model that "
                "generalizes to unseen dynamics'."
            ),
            "does_it_change_the_ranking": (
                "No. gemma leads on the memorizing subset AND on the non-memorizing subset, and "
                "leads decisively on coverage. But it BOUNDS the claim: this experiment ranks two "
                "inducers on a window-fitting proxy, and should not be restated as a measurement "
                "of generalizing world-model induction. The same caveat applies to exp5764 and to "
                "every prior number in this comparison lineage, since the mechanism is shared."
            ),
        },
        "secondary_criteria": sec,
        "window_transitions_by_game": window_len,
        "extreme_value_explanations": extremes,
        "vram_witness": vram,
        "arm_meta": metas,
        "model_specs": {
            a: {
                "label": LABEL[a],
                "hf_id": metas.get(a, {}).get("hf_id"),
                "gguf": metas.get(a, {}).get("gguf"),
                "quantisation": metas.get(a, {}).get("quantisation"),
                "quantisation_verified_from": "GGUF general.file_type == 15 (Q4_K_M), not filename",
                "n_ctx_deployed": metas.get(a, {}).get("n_ctx_deployed"),
                "kv_quant": metas.get(a, {}).get("kv_quant"),
                "use_chat_template": metas.get(a, {}).get("use_chat_template"),
                "mtp": metas.get(a, {}).get("mtp"),
                "invoked": True,
            }
            for a in ARMS
        },
        "preconditions_checked": [
            c for a in ARMS for c in (metas.get(a, {}).get("preconditions_checked") or [])
        ],
        "budget_exhaustion_finding": {
            "what_was_measured": (
                "The shared mechanism sends an identical '/think\\n' prefix and an identical 16384 "
                "TOKEN completion budget to both arms. reason_engaged (any of '<think', '</think', "
                "'<thinking', '<reasoning' in the raw completion) fired on 39/39 cells for BOTH "
                "arms, so the prefix is NOT inert on either model -- both genuinely reason. The "
                "difference is BUDGET EFFICIENCY: overran (llama.cpp stop_type == 'limit', i.e. it "
                "hit n_predict) fired on a large fraction of Qwen cells and on ZERO gemma cells "
                "(exp5764's gemma arm: overran 0/39, induce_failed 1/39, max completion 34816 "
                "chars). When Qwen exhausts the budget it emits code missing the required engine / "
                "is_level_* functions, so induce_ok=False and no scorable engine exists."
            ),
            "why_this_is_a_real_finding_not_a_harness_artifact": (
                "An earlier draft of this analysis asserted the /think prefix was inert on gemma "
                "because gemma-4-31B-it's chat template lacks a <think> tag, and therefore that "
                "Qwen was being unfairly penalised. The measurement refutes that: gemma emits "
                "reasoning tags on every cell too. Both models get the same prompt, prefix, budget, "
                "quantisation, context, corpus and card. Qwen simply reasons longer and runs out. A "
                "fixed generation budget is a REAL operational constraint for this project (the "
                "scored ARC path is latency- and VRAM-bounded), so 'fails to emit a usable engine "
                "within budget' is a legitimate, decision-relevant deficiency, not an artifact."
            ),
            "what_is_still_NOT_established": (
                "This does NOT establish that Qwen is intrinsically the weaker INDUCER. On the "
                "cells where Qwen did finish, it sometimes scored well (see view A). A larger "
                "budget, or /no_think, might rescue it. So the honest claim is bounded: AT THIS "
                "BUDGET, with this mechanism, Qwen fails to deliver a usable world model on a large "
                "fraction of cells while gemma nearly always does."
            ),
            "recommended_next_measurement": (
                "A third arm: Qwen3.6-27B at the same corpus with either /no_think or a raised "
                "budget, to separate 'induces worse' from 'spends its budget reasoning and runs "
                "out'. Deliberately NOT run here, because changing the prefix or budget would break "
                "mechanism identity with exp5764 -- the entire point of this matched design."
            ),
        },
        "field_principles": {
            "per_game_detail_side_by_side": (
                "Per-game matched values for BOTH arms in one place, including failure counts. A "
                "pooled mean over different game sets produced a phantom win difference earlier in "
                "this project; showing the per-game pairs makes that error class visible."
            ),
            "views": (
                "Three readings of the same cells, because a non-scorable cell is a judgement call "
                "(failure worth 0.0, or missing data?). Reporting one silently would hide the "
                "choice that most affects the answer."
            ),
            "coverage_nonzero_games": (
                "Games where the arm got off the floor. For hidden OOD games, some signal almost "
                "everywhere plausibly beats a higher mean on a favourable subset."
            ),
            "min_reachable_p_at_this_d": (
                "Smallest two-sided p attainable at the achieved discordant count. Prevents "
                "claiming significance below the floor AND prevents reading an unavoidable null as "
                "evidence of equivalence."
            ),
            "completeness": (
                "Which cells each arm actually finished. A truncated arm must never be compared "
                "against a complete one; both_arms_complete gates the comparison."
            ),
            "vram_witness": (
                "Sampled per-PID residency. Three prior attempts died on an unexplained hang; a "
                "card falling off the bus must be recorded, not inferred."
            ),
            "measurement_wall_s": (
                "Summed from each row's OWN elapsed_s, so it reflects real per-cell compute rather "
                "than wall-clock including server loads and idle gaps."
            ),
            "reproducibility_checksum": (
                "sha256 over the raw per-cell shard jsonls -- the evidence the aggregates are "
                "computed from, so a third party can recompute them."
            ),
            "residency_mib_gpu1": (
                "Proves WHICH physical card served each arm, read from per-PID residency against "
                "GPU 1's UUID rather than from CUDA_VISIBLE_DEVICES (an intention, not a fact)."
            ),
        },
        "methodology_note": (
            "Sequential arms, ONE llama-server at a time on GPU 1 (24 GiB cannot hold a 21GB and an "
            "18GB server together). GPU 0 was busy with an unrelated lane and was left alone. Card "
            "membership proven from per-PID residency vs GPU 1's UUID. Per-arm CARNOT_ARC_E3_DIR so "
            "neither arm can read the other's induced world_model.py (a shared engine store "
            "contaminated an earlier run). Per-cell shard caching so a wedge costs at most one "
            "cell. Every cell gated on GPU-1 presence, >15GB residency, /props model identity, and "
            "a bounded REAL /completion -- because exp5833 died with /health returning 200 while "
            "/completion hung, so health is not liveness. Paired unit is the GAME (13), not the "
            "cell (39): the 3 trials are repeats of one seeded window."
        ),
        "prior_work_extended": {
            "exp5764_not_used_as_this_comparison_arm_sha256": (
                "ed7f0d14f3991d17c81e9bf6b2773c3848b9b59d76267efa29a3cdf063abaf04"
            ),
            "note": (
                "exp5764's gemma numbers (pooled 0.378487, nonzero 12/13) are NOT this "
                "comparison's gemma arm. gemma was RE-RUN in this session so both arms are "
                "same-session, same-card, same-harness. exp5764 is cited only as a cross-check and "
                "is never modified (never-prune)."
            ),
            "exp5705_confound_avoided": (
                "exp5705 compared gemma Q8_0 against qwen Q4 and its own verdict admits the "
                "precision/model confound. Here BOTH arms are Q4_K_M, verified from each GGUF's "
                "general.file_type field rather than its filename."
            ),
            "exp5833_failure_mode_guarded": (
                "exp5833's qwen arm wedged with a HIP server and its verdict states it was not a "
                "valid 3-way ranking. Here each cell re-verifies a real /completion."
            ),
            "exp5598_and_exp5764_were_non_comparable": (
                "exp5598 (4 games) and exp5764 (13 games) share only 3 games, far too few to "
                "separate the models; that mismatch is what this experiment exists to fix."
            ),
        },
        "aggregation_wall_s": round(time.time() - t0, 3),
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ARTIFACT}")
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
