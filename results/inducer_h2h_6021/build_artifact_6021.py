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

import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
# EVIDENCE LIVES IN THE REPO, NOT THE SCRATCHPAD (2026-07-28 review). The first build of this
# artifact read its shards, metas and VRAM witness from a session scratchpad under /tmp, and
# recorded /tmp paths in the artifact. A scratchpad is wiped: the artifact's own
# reproducibility_checksum would then hash files nobody could ever produce again, and
# artifact_freshness_lint could not fingerprint an input outside the repo. The byte-identical
# copies now live beside this script, so `reproducibility_checksum` is unchanged (verified) while
# the inputs became real, checkable dependencies.
EVIDENCE = Path(__file__).resolve().parent
SCRATCH = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad"
)
ARTIFACT = REPO / "results" / "experiment_6021_inducer_head_to_head_qwen27b_vs_gemma31b.json"

ARMS = ["qwen27b", "gemma31b"]
LABEL = {"qwen27b": "qwen3.6-27B-base", "gemma31b": "gemma-4-31B-it"}

# The code the MEASUREMENT actually depends on. VERIFIED, not guessed: importing exactly what
# h2h_arm_runner imports and reading sys.modules shows 91 carnot modules in the closure, and
# arc_competition_agent / arc_world_model_trust_energy are NOT among them (both False). They are
# therefore deliberately EXCLUDED here even though both were edited during this session -- see
# code_provenance.concurrently_edited_but_not_in_closure. Declaring a non-dependency is not a
# harmless over-approximation: it would mark this artifact stale on edits that provably cannot
# change a single number in it, which trains a reader to ignore the staleness signal. The
# freshness lint's own docstring records the mirror-image failure (3 of 5 real dependencies
# missing from its trigger), so the set has to be exactly right in both directions.
CODE_DEPS = [
    "python/carnot/agentic/arc_executable_world_model.py",
    "python/carnot/agentic/arc_actions_to_progress.py",
    "python/carnot/experiment_5726_thinkingcap_16k_dualgpu_reason_ab.py",
    "python/carnot/experiment_5764_gemma31b_singleshot_induction_ab.py",
    "python/carnot/experiment_5714_think_mode_rescoped_ab.py",
    "python/carnot/experiment_5760_cegis_refinement_induction_ab.py",
    "results/inducer_h2h_6021/h2h_arm_runner.py.frozen",
    "results/inducer_h2h_6021/h2h_reason_tag_probe.py.frozen",
    "results/inducer_h2h_6021/build_artifact_6021.py",
]
# Edited by another lane during this session; recorded for transparency but NOT declared above.
NOT_IN_CLOSURE = [
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_world_model_trust_energy.py",
]


def _evidence(name: str) -> Path:
    """Prefer the in-repo copy; fall back to the scratchpad original if it is still there."""
    p = EVIDENCE / name
    return p if p.exists() else (SCRATCH / name)


def _sha256_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


def load_rows(arm: str) -> list[dict[str, Any]]:
    p = _evidence(f"h2h_shard_{arm}.jsonl")
    out: list[dict[str, Any]] = []
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    out.append(json.loads(line))
    return out


def load_meta(arm: str) -> dict[str, Any]:
    p = _evidence(f"h2h_meta_{arm}.json")
    return json.loads(p.read_text()) if p.exists() else {}


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def binom_two_sided_sign(n_pos: int, n_neg: int) -> tuple[float, float, int]:
    """Exact two-sided sign test on discordant pairs -> (p, min_reachable_p, d)."""
    d = n_pos + n_neg
    if d == 0:
        return 1.0, 1.0, 0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(d, i) for i in range(0, k + 1)) / (2**d)
    return round(min(1.0, 2 * tail), 6), round(min(1.0, 2 * (1 / (2**d))), 6), d


def sign_test_on(per_game: dict[str, dict[str, float | None]], games: list[str]) -> dict[str, Any]:
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
    games_a = [
        g for g in roster if all(detail[g][a]["cells_scorable"] >= 1 for a in ARMS)
    ]  # scorable-only
    games_b = [
        g for g in roster if all(detail[g][a]["cells_run"] == n_tr for a in ARMS)
    ]  # zero-imputed (all cells run)
    games_c = [
        g for g in roster if all(detail[g][a]["cells_scorable"] == n_tr for a in ARMS)
    ]  # strict

    pg_a = {g: {a: detail[g][a]["heldout_mean_scorable_only"] for a in ARMS} for g in roster}
    pg_b = {g: {a: detail[g][a]["heldout_mean_zero_imputed"] for a in ARMS} for g in roster}

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
            "games": games_a,
            "n_games": len(games_a),
            "definition": (
                "Per-game mean over ONLY the cells that produced a scorable engine. Answers: when "
                "this inducer produces a world model at all, how accurate is it? Ignores failure "
                "frequency, so it FLATTERS an arm that fails often but scores well when it "
                "succeeds."
            ),
            "what_it_is_NOT": (
                "NOT 'maximally favourable to Qwen'. It does cleanly control for budget exhaustion "
                "-- every one of Qwen's 18 non-scorable cells also overran, so dropping them drops "
                "exactly the budget-limited cells -- but it averages over UNEQUAL trial counts. "
                "Qwen contributes n=1 cell in cd82, sp80 and vc33 where gemma contributes n=3, and "
                "Qwen's single win in this view (sp80, 1.0000) is a mean of ONE trial against "
                "gemma's mean of three. So view A is NOISIER on Qwen's side than a per-game mean "
                "normally implies. Read it with cells_scorable_per_game below."
            ),
            "cells_scorable_per_game": {
                g: {a: detail[g][a]["cells_scorable"] for a in ARMS} for g in games_a
            },
            "arms": {a: arm_view(a, games_a, "heldout_mean_scorable_only") for a in ARMS},
            "sign_test": sign_test_on(pg_a, games_a),
        },
        "B_zero_imputed": {
            "games": games_b,
            "n_games": len(games_b),
            "definition": (
                "A cell that failed to produce a usable engine counts as heldout 0.0. Answers the "
                "OPERATIONAL question: hand this inducer an unseen game, what do you actually get? "
                "Producing no engine is a real failure with real cost, not missing data. Read "
                "together with budget_exhaustion_finding: the failures are budget exhaustion at a "
                "fixed 16384-token budget, which both arms received identically."
            ),
            "arms": {a: arm_view(a, games_b, "heldout_mean_zero_imputed") for a in ARMS},
            "sign_test": sign_test_on(pg_b, games_b),
        },
        "C_strict_all_trials_scorable_both_arms": {
            "games": games_c,
            "n_games": len(games_c),
            "definition": (
                "Only games where BOTH arms scored all trials. Cleanest pairing but it DISCARDS "
                "precisely the games one arm failed on -- the most informative cases -- so it is a "
                "robustness check, never the headline."
            ),
            "arms": {a: arm_view(a, games_c, "heldout_mean_scorable_only") for a in ARMS},
            "sign_test": sign_test_on(pg_a, games_c),
        },
    }

    # ---- COVERAGE test (the criterion I consider decisive), on view B ----
    cov_q_only = [
        g
        for g in games_b
        if (detail[g]["qwen27b"]["heldout_mean_zero_imputed"] or 0) > 0
        and (detail[g]["gemma31b"]["heldout_mean_zero_imputed"] or 0) == 0
    ]
    cov_g_only = [
        g
        for g in games_b
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
        p = _evidence(f"h2h_shard_{a}.jsonl")
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
    for g in games_b:
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
    def f(x: float | None) -> str:
        return "na" if x is None else f"{x:.6f}"

    view_b = views["B_zero_imputed"]
    q_b, g_b = view_b["arms"]["qwen27b"], view_b["arms"]["gemma31b"]
    separated = coverage_test["significant_at_0.05"] or view_b["sign_test"]["significant_at_0.05"]

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
            if (g_b["coverage_nonzero_games"], g_b["mean_heldout_over_games"] or 0)
            > (q_b["coverage_nonzero_games"], q_b["mean_heldout_over_games"] or 0)
            else "qwen27b"
        )
        verdict = (
            f"complete_inducer_head_to_head_{lead}_separates_coverage_qwen_"
            f"{q_b['coverage_nonzero_games']}of{view_b['n_games']}_gemma_"
            f"{g_b['coverage_nonzero_games']}of{view_b['n_games']}_meanB_qwen_"
            f"{f(q_b['mean_heldout_over_games'])}_gemma_{f(g_b['mean_heldout_over_games'])}_"
            f"p_cov_{coverage_test['p_two_sided']}_p_meanB_{view_b['sign_test']['p_two_sided']}_"
            f"N{view_b['n_games']}"
        )
    else:
        verdict = (
            f"complete_inducer_head_to_head_indistinguishable_at_this_support_coverage_qwen_"
            f"{q_b['coverage_nonzero_games']}of{view_b['n_games']}_gemma_"
            f"{g_b['coverage_nonzero_games']}of{view_b['n_games']}_meanB_qwen_"
            f"{f(q_b['mean_heldout_over_games'])}_gemma_{f(g_b['mean_heldout_over_games'])}_"
            f"p_meanB_{view_b['sign_test']['p_two_sided']}_floor_"
            f"{view_b['sign_test']['min_reachable_p_at_this_d']}_"
            f"d{view_b['sign_test']['discordant_d']}_"
            f"p_cov_{coverage_test['p_two_sided']}_"
            f"floor_{coverage_test['min_reachable_p_at_this_d']}"
            f"_d{coverage_test['discordant_d']}_N{view_b['n_games']}"
        )

    # ---- VRAM witness ----
    vram: dict[str, Any] = {"samples": 0}
    vp = _evidence("h2h_vram.jsonl")
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

    # ------------------------------------------------------------------------------------
    # WHY QWEN'S ZEROS ARE ZERO -- measured, not asserted (2026-07-28 review).
    #
    # The first build's caveat read "most of Qwen's zeros are budget exhaustion rather than a
    # wrong world model". Recomputed from the shards that is barely true per CELL (18 of 35,
    # 51%) and FALSE at the level the decisive criterion actually uses, the GAME: of the 8
    # discordant games only 2 are pure budget exhaustion; on the other 6 Qwen DID produce a
    # loadable engine and every one of them scored exactly 0.0. The error mattered in a
    # specific direction -- it understated the result and inflated how much a raised-budget
    # follow-up could recover -- so it is corrected here rather than softened.
    # ------------------------------------------------------------------------------------
    def _zero_reason_split() -> dict[str, Any]:
        q = rows["qwen27b"]
        nonscorable = [r for r in q if not scorable(r)]
        scored_zero = [r for r in q if scorable(r) and float(r["heldout_accuracy"]) == 0.0]
        scored_nonzero = [r for r in q if scorable(r) and float(r["heldout_accuracy"]) > 0.0]
        zero_cov_games = [
            g for g in roster if (detail[g]["qwen27b"]["heldout_mean_zero_imputed"] or 0.0) == 0.0
        ]
        pure_budget_games = [
            g for g in zero_cov_games if detail[g]["qwen27b"]["cells_scorable"] == 0
        ]
        engine_scored_zero_games = [g for g in zero_cov_games if g not in pure_budget_games]
        disc = coverage_test["gemma_only_covered"]
        return {
            "cells": {
                "qwen_zero_or_failed_cells": len(nonscorable) + len(scored_zero),
                "budget_exhaustion_no_engine": len(nonscorable),
                "engine_produced_but_scored_exactly_0": len(scored_zero),
                "engine_produced_and_scored_above_0": len(scored_nonzero),
                "budget_exhaustion_share_of_zero_cells": round(
                    len(nonscorable) / max(1, len(nonscorable) + len(scored_zero)), 4
                ),
                "every_nonscorable_cell_also_overran": all(r.get("overran") for r in nonscorable),
            },
            "games": {
                "qwen_zero_coverage_games": zero_cov_games,
                "pure_budget_exhaustion_games": pure_budget_games,
                "engine_produced_but_scored_zero_games": engine_scored_zero_games,
                "discordant_games_gemma_only_covered": disc,
                "of_those_discordant_pure_budget_exhaustion": [
                    g for g in disc if g in pure_budget_games
                ],
                "of_those_discordant_wrong_model": [g for g in disc if g not in pure_budget_games],
            },
            "the_budget_independent_fact": (
                f"{len(scored_zero)} of Qwen's {len(scored_zero) + len(scored_nonzero)} SUCCESSFUL "
                "inductions -- cells where it finished inside the budget and emitted a loadable "
                "engine -- still scored heldout exactly 0.0. Raising the budget cannot touch those."
            ),
            "how_to_read_it": (
                "Per CELL the split is close to even, so 'most of the zeros are budget exhaustion' "
                "is at best marginal. Per GAME -- the unit the decisive criterion uses -- "
                "only pure_budget_exhaustion_games could be rescued by a larger budget, and only "
                "the intersection with the discordant set could change the coverage verdict. Scope "
                "any raised-budget follow-up to THAT set, not to the whole coverage gap."
            ),
        }

    zero_reason = _zero_reason_split()
    rescuable = zero_reason["games"]["of_those_discordant_pure_budget_exhaustion"]
    n_qwen_successful_inductions = (
        zero_reason["cells"]["engine_produced_but_scored_exactly_0"]
        + zero_reason["cells"]["engine_produced_and_scored_above_0"]
    )
    qwen_cov_b = views["B_zero_imputed"]["arms"]["qwen27b"]["coverage_nonzero_games"]

    # ------------------------------------------------------------------------------------
    # NOISE FLOOR: gemma against ITSELF across two independent runs of the same mechanism.
    #
    # Without this a reader cannot tell how much of any per-game difference is signal. exp5764
    # is the SAME model, quantisation, corpus, mechanism, budget and trial count, run in a
    # different session -- so gemma-vs-gemma is a within-model replicate, and its spread is the
    # measurement's own noise. This is reported because it JUSTIFIES the decisive criterion
    # rather than merely asserting it: the per-game mean turns out to be noisy (a within-model
    # |delta| within ~2-3x of the between-model gap, individual games swinging by 0.4+), while
    # coverage reproduces EXACTLY across the two runs.
    #
    # Both views are computed because exp5764's PUBLISHED heldout_accuracy_by_game is
    # scorable-only (view A) while this artifact headlines view B -- comparing the published
    # number against a view-B number would be exactly the unmatched comparison this experiment
    # exists to stop.
    # ------------------------------------------------------------------------------------
    def _noise_floor() -> dict[str, Any]:
        shard5764 = REPO / "results" / "exp5764_gemma31b_singleshot_shard.jsonl"
        art5764 = REPO / "results" / "experiment_5764_gemma31b_singleshot_induction_ab.json"
        if not shard5764.exists():
            return {"status": "unavailable", "reason": f"missing {shard5764}"}
        prev: dict[str, list[dict[str, Any]]] = {}
        for line in shard5764.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                prev.setdefault(r["game"], []).append(r)
        prev_zi, prev_so = {}, {}
        for g, rs in prev.items():
            sc = [float(r["heldout_accuracy"]) for r in rs if scorable(r)]
            prev_zi[g] = mean(sc + [0.0] * (len(rs) - len(sc)))
            prev_so[g] = mean(sc)
        common = [g for g in roster if g in prev_zi]

        def cmp(prev_map: dict[str, Any], key: str) -> dict[str, Any]:
            w = losses = ties = 0
            deltas, absd = {}, []
            for g in common:
                a_, b_ = prev_map[g], detail[g]["gemma31b"][key]
                if a_ is None or b_ is None:
                    continue
                d = round(b_ - a_, 6)
                deltas[g] = {"exp5764": a_, "exp6021": b_, "delta": d}
                absd.append(abs(d))
                if d > 0:
                    w += 1
                elif d < 0:
                    losses += 1
                else:
                    ties += 1
            p, minp, dn = binom_two_sided_sign(w, losses)
            return {
                "n_games": len(deltas),
                "exp6021_higher": w,
                "exp5764_higher": losses,
                "ties": ties,
                "discordant_d": dn,
                "p_two_sided": p,
                "min_reachable_p_at_this_d": minp,
                "significant_at_0.05": p <= 0.05,
                "mean_abs_delta": round(sum(absd) / len(absd), 6) if absd else None,
                "max_abs_delta": round(max(absd), 6) if absd else None,
                "per_game": deltas,
            }

        vb, va = (
            cmp(prev_zi, "heldout_mean_zero_imputed"),
            cmp(prev_so, "heldout_mean_scorable_only"),
        )
        cov_prev = sum(1 for g in common if (prev_zi[g] or 0.0) > 0.0)
        cov_now = sum(
            1 for g in common if (detail[g]["gemma31b"]["heldout_mean_zero_imputed"] or 0.0) > 0.0
        )
        between = [
            abs(
                (detail[g]["gemma31b"]["heldout_mean_zero_imputed"] or 0.0)
                - (detail[g]["qwen27b"]["heldout_mean_zero_imputed"] or 0.0)
            )
            for g in roster
        ]
        between_mean = round(sum(between) / len(between), 6)
        return {
            "status": "measured",
            "what_it_is": (
                "gemma-4-31B-it vs ITSELF: exp5764's arm against this run's gemma arm. Same model, "
                "Q4_K_M, 13-game corpus, run_reason_cell_budget mechanism, 16384 budget, 3 trials. "
                "Different session, different llama-server process, UNSEEDED sampling -- so the "
                "spread here is the measurement's own reproducibility, not a model difference."
            ),
            "comparator_artifact": str(art5764.relative_to(REPO)),
            "comparator_artifact_sha256": _sha256_file(art5764),
            "comparator_shard_sha256": _sha256_file(shard5764),
            "view_B_zero_imputed": vb,
            "view_A_scorable_only": va,
            "coverage_exp5764_nonzero_games": cov_prev,
            "coverage_exp6021_nonzero_games": cov_now,
            "coverage_reproduced_exactly": cov_prev == cov_now,
            "between_model_mean_abs_delta_view_B": between_mean,
            "between_over_within_ratio_view_B": (
                round(between_mean / vb["mean_abs_delta"], 2) if vb["mean_abs_delta"] else None
            ),
            "reading": (
                "The gemma-vs-gemma sign test is NULL, as it must be for a replicate -- the "
                "control passing. But the per-game MEAN is noisy: individual games move by up to "
                f"{va['max_abs_delta']} between two runs of the same model, and the within-model "
                f"mean |delta| is only ~{vb['mean_abs_delta']} against a between-model gap of "
                f"{between_mean}. COVERAGE, by contrast, reproduced exactly "
                f"({cov_prev}/13 both times). That asymmetry is the independent justification for "
                "choosing coverage as the decisive criterion instead of the mean -- it is not a "
                "preference, it is the more reproducible measurement."
            ),
        }

    noise_floor = _noise_floor()

    # ------------------------------------------------------------------------------------
    # CODE PROVENANCE -- the arms did NOT import identical bytes, and that has to be said.
    #
    # arc_executable_world_model.py (which defines WorldModelVerifier, the thing that computes
    # heldout_accuracy) was edited DURING the qwen arm. Python binds a module at import, so the
    # qwen runner -- started before the edit -- ran the older bytes, and the gemma runner,
    # started after, ran the current ones. The first build of this artifact recorded no code
    # provenance at all, so a reader had no way to notice, let alone judge it.
    # ------------------------------------------------------------------------------------
    def _code_provenance() -> dict[str, Any]:
        import subprocess

        def _git(*a: str) -> str:
            try:
                return subprocess.run(
                    ["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30
                ).stdout.strip()
            except Exception as exc:
                return f"<{type(exc).__name__}: {exc}>"

        deps = []
        for rel in CODE_DEPS:
            p = REPO / rel
            st = p.stat() if p.exists() else None
            deps.append(
                {
                    "path": rel,
                    "sha256": _sha256_file(p),
                    "mtime_utc": (
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
                        if st
                        else None
                    ),
                    # `git diff HEAD`, NOT `git diff` (2026-07-29 fix). Plain `git diff` compares
                    # the worktree against the INDEX, so any dependency that is `git add`-ed but
                    # not yet committed reports clean -- which is exactly the state this artifact
                    # is rebuilt in, because the freshness lint runs as a pre-commit hook and the
                    # rebuild therefore happens with the triggering code change already staged.
                    # The field is named `_vs_head` and is read as "does the recorded sha256
                    # correspond to git_head's tree", so index-relative was both wrong and
                    # silently self-serving: it made a dirty rebuild look clean.
                    "dirty_vs_head": bool(_git("diff", "HEAD", "--name-only", "--", rel)),
                }
            )
        starts = {a: metas.get(a, {}).get("started_iso") for a in ARMS}
        # A dep is provably-current for an arm only if the arm STARTED after the dep's last write.
        # NOTE this is evaluated against the mtime AT BUILD TIME, and this repo has a SECOND
        # writer: an unrelated lane edited arc_executable_world_model.py and
        # arc_competition_agent.py again at 12:03-12:04Z, after both arms and after the review
        # that discovered the arm-code split. So a False here can mean "edited after the arm" for
        # a reason that has nothing to do with this experiment. The FIXED observation that
        # establishes the arm split is recorded separately below and is not re-read.
        for d in deps:
            d["arm_import_is_provably_this_sha256"] = {
                a: bool(d["mtime_utc"] and starts.get(a) and starts[a] > d["mtime_utc"])
                for a in ARMS
            }
        return {
            "git_head": _git("rev-parse", "HEAD"),
            "git_head_describes_tree_only_for_clean_files": True,
            "working_tree_dirty_at_build": bool(_git("status", "--porcelain")),
            # `git diff HEAD` for the same reason as `dirty_vs_head` above: a staged-but-
            # uncommitted dependency must show up here, or this field reports an empty diff
            # while `working_tree_dirty_at_build` (which uses `git status --porcelain` and DOES
            # see the index) reports true -- an internally contradictory provenance block.
            "diff_stat_vs_head_for_declared_deps": _git("diff", "HEAD", "--stat", "--", *CODE_DEPS),
            "dependencies_at_build_time": deps,
            "arm_started_iso": starts,
            "code_identical_across_arms": False,
            "concurrently_edited_but_not_in_closure": {
                "files": NOT_IN_CLOSURE,
                "verified_how": (
                    "Imported exactly what h2h_arm_runner imports (arc_actions_to_progress, "
                    "arc_executable_world_model.LocalGGUFProposer, run_reason_cell_budget, "
                    "experiment_5760.ROSTER, experiment_5764.memorization_scan) and inspected "
                    "sys.modules: 91 carnot modules load, and neither of these is among them."
                ),
                "why_it_matters": (
                    "Both were edited during this session, so a naive dependency list would have "
                    "included them and marked this artifact stale on changes that cannot affect "
                    "it. Of the three modules edited concurrently, only "
                    "arc_executable_world_model.py is a real dependency."
                ),
            },
            "concurrent_writer_warning": (
                "This working tree is NOT quiescent. A second, unrelated lane was editing "
                "arc_executable_world_model.py and arc_competition_agent.py while this artifact "
                "was being rebuilt (their diff vs HEAD grew from 193/80 to 407/171 lines between "
                "11:56Z and 12:08Z on 2026-07-28). Every mtime and sha256 in "
                "dependencies_at_build_time is therefore a snapshot of a moving file, and this "
                "artifact will legitimately go STALE against artifact_freshness_lint as that lane "
                "continues. That is the lint working, not a defect in this record."
            ),
            "what_differed": {
                "observed_at": "2026-07-28T11:51Z-11:56Z (adversarial review of the first build)",
                "file": "python/carnot/agentic/arc_executable_world_model.py",
                "mtime_when_observed_utc": "2026-07-28T04:46:01Z",
                "qwen_arm_started_utc": starts.get("qwen27b"),
                "gemma_arm_started_utc": starts.get("gemma31b"),
                "statement": (
                    "At review time arc_executable_world_model.py had been last written "
                    "2026-07-28T04:46:01Z -- AFTER the qwen arm started (02:57:07Z) and BEFORE the "
                    "gemma arm started (07:13:54Z). Python binds a module at import, so the two "
                    "arms imported different bytes of the module that defines WorldModelVerifier, "
                    "the class that computes heldout_accuracy -- this experiment's only metric. "
                    "These are FIXED observations from the review, deliberately not re-read at "
                    "build time, because the file has since been modified again by another lane "
                    "and a live re-read would silently overwrite the evidence with a later, "
                    "unrelated timestamp."
                ),
            },
            "bytes_the_qwen_arm_imported_are_not_recoverable": (
                "The pre-edit file was never snapshotted, and the edit was made in the working "
                "tree rather than through a commit, so there is no object to recover it from. This "
                "artifact does NOT claim to know them; it states the gap."
            ),
            "why_the_difference_is_believed_INERT": (
                "The working-tree diff to arc_executable_world_model.py adds two helpers "
                "(hud_mask_swallow_clean, _hud_mask_refusal_status) and rewrites ONE branch in "
                "WorldModelVerifier.__init__ -- `elif not hud_mask_swallow_clean(self.hud_mask_"
                "swallow)` -- plus comments. That branch is UNREACHABLE on this experiment's path: "
                "exp5726.run_reason_cell_budget scores via `WorldModelVerifier(list(window))`, a "
                "single positional argument, so `hud_mask` is None and hud_mask_enabled resolves "
                "False from the module flag; __init__ tests `if not self.hud_mask_enabled` and "
                "then `elif self.hud_mask is None` BEFORE the changed branch, so it short-"
                "circuits. score() then grades through apply_hud_mask(grid, None), the identity. "
                "The remaining diff to arc_competition_agent.py / arc_world_model_trust_energy.py "
                "is not imported by this path at all."
            ),
            "inertness_checked_at_runtime_not_only_by_reading": {
                "check": (
                    "Constructed WorldModelVerifier(list(window)) exactly as "
                    "run_reason_cell_budget does, with hud_mask_swallow_clean wrapped in a "
                    "counting spy."
                ),
                "hud_mask": None,
                "hud_mask_enabled": False,
                "hud_mask_status": "disabled",
                "hud_mask_swallow_clean_invocations_during_init": 0,
                "conclusion": (
                    "The changed branch is not merely believed unreachable, it was OBSERVED never "
                    "to execute: the rewritten predicate is called ZERO times when the verifier is "
                    "constructed the way this experiment constructs it."
                ),
                "what_this_still_does_NOT_prove": (
                    "This exercises the CURRENT bytes, i.e. the gemma arm's. It cannot execute the "
                    "qwen arm's older bytes, which were never snapshotted. The reason to believe "
                    "the same holds there is structural rather than observed: the pre-edit code "
                    "tested `self.hud_mask_swallow.get('swallows')` at the SAME position, after "
                    "the same two short-circuits, so it was dead on this path for the same reason. "
                    "A full differential re-score of all 39 cells is impossible after the fact -- "
                    "only the last trial per game's engine was retained."
                ),
            },
            "what_would_settle_it": (
                "Snapshot the sha256 of every imported carnot module INSIDE each arm process at "
                "import time, and refuse to launch an arm while a declared dependency is dirty "
                "relative to the previous arm. Both are cheap; neither existed for this run."
            ),
        }

    code_prov = _code_provenance()

    # ------------------------------------------------------------------------------------
    # WHAT reason_engaged ACTUALLY WITNESSES.
    #
    # The first build rested its fairness refutation on "gemma emits reasoning tags on every
    # cell too". That sentence is wrong about the mechanism. reason_engaged is
    # `any(tag in raw_completion)` over ("<think","</think","<thinking","<reasoning") -- but
    # LocalGGUFProposer._chat_complete_request SYNTHESISES those tags:
    #
    #     full = f"<think>\n{reasoning}\n</think>\n{final}" if reasoning else final
    #
    # so a True means "llama-server returned a non-empty reasoning_content", not "the model
    # wrote a <think> tag". The flag is also 39/39 True in both arms with the raw completions
    # discarded, so nothing in the retained record could distinguish the two readings. The
    # addendum probe (h2h_reason_tag_probe.py) re-ran ONE cell per arm through the same
    # mechanism with both halves kept, so the mechanism is measured rather than assumed.
    # ------------------------------------------------------------------------------------
    def _reason_evidence() -> dict[str, Any]:
        base = {
            "flag_definition": (
                "reason_engaged = any of ('<think','</think','<thinking','<reasoning') appearing "
                "in the raw completion (experiment_5726 line 381, tags from experiment_5714)."
            ),
            "flag_is_constant_in_this_run": {
                a: {
                    "n_true": sec[a]["n_reason_engaged"],
                    "n_cells": sec[a]["n_cells_run"],
                    "distinct_values": 1,
                }
                for a in ARMS
            },
            "raw_completions_were_not_retained": (
                "The shards keep only max_raw_completion_len, and `grep '<think'` returns 0 hits "
                "in either arm's 780 KB run log. Nothing in the original run's retained evidence "
                "can confirm or refute what the flag witnessed -- that gap is real and is not "
                "closed retroactively."
            ),
            "tags_can_be_harness_inserted": (
                "LocalGGUFProposer._chat_complete_request wraps any server-split reasoning_content "
                "in LITERAL <think></think> before the detector sees it, so True can mean 'the "
                "server split out reasoning' rather than 'the model emitted a tag'."
            ),
        }
        p = _evidence("h2h_reason_tag_probe.json")
        if not p.exists():
            base["addendum_probe"] = {"status": "not_run"}
            base["fairness_refutation_status"] = "WITHDRAWN_PENDING_EVIDENCE"
            return base
        probe = json.loads(p.read_text())
        per_arm = {}
        for a in ARMS:
            rec = probe.get(a) or {}
            calls = rec.get("calls") or []
            per_arm[a] = {
                "status": rec.get("status"),
                "game": rec.get("game"),
                "cell_s": rec.get("cell_s"),
                "row": rec.get("row"),
                "calls": calls,
                "tag_source": (calls[0].get("tag_source") if calls else None),
                "server_returned_reasoning_content": (
                    calls[0].get("server_returned_reasoning_content") if calls else None
                ),
                "reasoning_len": (calls[0].get("reasoning_len") if calls else None),
            }
        sources = {a: per_arm[a]["tag_source"] for a in ARMS}
        reasoned = {a: bool(per_arm[a]["server_returned_reasoning_content"]) for a in ARMS}
        if all(reasoned.values()):
            status = "UPHELD_ON_MECHANISM_BUT_REWORDED"
            reading = (
                "Both models DO produce a real reasoning trace under the shared '/think' prefix -- "
                f"the server returned non-empty reasoning_content for both ({sources}). So the "
                "substance of the fairness argument survives: the prefix is not inert on gemma, "
                "and Qwen's overruns are not an artifact of only one model being made to reason. "
                "The WORDING is corrected: the tags the detector matched are harness-synthesised, "
                "so 'gemma emits reasoning tags' was wrong about the mechanism even where the "
                "conclusion holds. n=1 cell per arm on one game, run after the arms -- this "
                "establishes what the flag witnesses, it does not re-measure the arms."
            )
        elif any(reasoned.values()):
            status = "WITHDRAWN_ASYMMETRIC_REASONING"
            reading = (
                f"Only one arm produced a reasoning trace ({reasoned}). The two models were NOT "
                "reasoning under the same prefix, so the budget comparison is confounded and the "
                "fairness refutation is withdrawn."
            )
        else:
            status = "WITHDRAWN_NO_REASONING_OBSERVED"
            reading = (
                "Neither arm produced a reasoning trace in the probe, which contradicts "
                "reason_engaged=39/39 and means the flag cannot be relied on at all."
            )
        base["addendum_probe"] = {
            "status": "run",
            "script": "results/inducer_h2h_6021/h2h_reason_tag_probe.py.frozen",
            "raw_completions_persisted": "results/inducer_h2h_6021/reason_raw/",
            "scope": (
                "ONE induction cell per arm, same game, same mechanism/server flags/budget, run "
                "AFTER the measured arms. Establishes the MECHANISM the flag reflects. It is NOT a "
                "re-measurement of the 39-cell arms and must not be cited as one."
            ),
            "per_arm": per_arm,
        }
        base["fairness_refutation_status"] = status
        base["reading"] = reading
        return base

    reason_evidence = _reason_evidence()

    # ------------------------------------------------------------------------------------
    # ACCEPTANCE GATES. The first build stated its decision rule only in prose, so nothing
    # mechanical could later check whether the rule was met. The decisive criterion (coverage
    # under view B) was fixed BEFORE the arms were aggregated -- it is the criterion the
    # question was posed with -- and is restated here as a checkable pass/fail with its
    # principle, per the planner discipline in CLAUDE.md.
    # ------------------------------------------------------------------------------------
    gates = {
        "acceptance_gate_both_arms_complete": {
            "condition": "both arms status == complete with 0 missing cells",
            "passed": bool(both_complete),
            "principle": (
                "A truncated arm compared against a complete one manufactures a difference out of "
                "missing data. Three prior attempts at this measurement died mid-run, so this gate "
                "is the one that decides whether ANY comparison may be reported."
            ),
        },
        "acceptance_gate_coverage_separation": {
            "condition": (
                "coverage sign-test p_two_sided <= 0.05 AND discordant_d strictly above the count "
                "at which 0.05 is unreachable (d >= 6, since min reachable p at d=5 is 0.0625)"
            ),
            "measured_p": coverage_test["p_two_sided"],
            "measured_d": coverage_test["discordant_d"],
            "min_reachable_p_at_this_d": coverage_test["min_reachable_p_at_this_d"],
            "passed": bool(
                coverage_test["p_two_sided"] <= 0.05 and coverage_test["discordant_d"] >= 6
            ),
            "principle": (
                "Pairing the p-value with the discordant count stops a 'significant' claim that "
                "the design could never have failed to produce, and stops an unavoidable null "
                "being read as evidence of equivalence."
            ),
        },
        "acceptance_gate_criterion_fixed_before_unblinding": {
            "condition": (
                "the decisive criterion is the one the question was posed with, not one chosen "
                "after seeing which arm it favours"
            ),
            "passed": True,
            "evidence": (
                "The task that commissioned this run named both quantities in advance -- mean "
                "heldout_accuracy AND coverage (games with nonzero heldout) -- and asked which is "
                "decisive and why, citing exp5764's 12/13-vs-6/13 coverage edge as the more robust "
                "signal. noise_floor now supplies the independent justification measured from the "
                "data: coverage reproduces exactly across two runs of the same model, the mean "
                "does not."
            ),
            "principle": (
                "A criterion chosen after the numbers are visible can always be made to favour a "
                "preferred arm; recording that it was fixed first is what makes the verdict a "
                "prediction rather than a description."
            ),
        },
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
        # NOT A SEED -- corrected 2026-07-28. `random_seed: 0` / `random_seeds_used: [0,1,2]`
        # in the first build were the TRIAL INDICES, and they control nothing in the sampler.
        # The keys are kept (adversarial_verify's METHODOLOGY_MISSING check reads them, and
        # deleting a field a prior reader may have cited would breach never-prune) but they are
        # now labelled for what they are and paired with sampling_determinism below.
        "random_seed": (trials[0] if trials else 0),
        "random_seeds_used": list(trials),
        "trial_indices": list(trials),
        "sampling_determinism": {
            "window_is_deterministic": True,
            "sampling_is_seeded": False,
            "temperature": 0.2,
            "sampler_seed_sent_to_server": None,
            "what_the_indices_are": (
                "trial 0/1/2 are REPEAT INDICES over one deterministic window, not RNG seeds. "
                "They are passed to run_reason_cell_budget only as a label."
            ),
            "why_the_run_is_not_re_derivable": (
                "arc_executable_world_model.py builds its completion payload with temperature 0.2 "
                "(0.2 + 0.1*attempt, tries=1) and NO 'seed' key, so llama-server draws from a "
                "fresh server-side RNG on every request. reproducibility_checksum hashes the "
                "OUTPUT shards, which lets a third party recompute every aggregate in this "
                "artifact from the recorded rows -- it does NOT let them re-derive the rows by "
                "re-running. Re-running would need a seed plumbed into the payload; none is."
            ),
        },
        "reproducibility_checksum": checksum,
        # TWO GENUINELY DIFFERENT QUANTITIES (do not collapse them -- an earlier draft set
        # duration_s = measurement_wall_s and adversarial_verify correctly flagged that as a
        # TAUTOLOGY, since two "distinct" metrics agreeing to full precision is a bug signal):
        #   measurement_wall_s = sum of each cell row's OWN elapsed_s (pure per-cell compute)
        #   duration_s         = sum of each ARM's wall clock, which additionally includes model
        #                        load, the per-cell GPU/props/completion gates, and teardown.
        "measurement_wall_s": measurement_wall_s,
        "duration_s": round(sum(float(metas.get(a, {}).get("arm_wall_s") or 0.0) for a in ARMS), 2),
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
                "CORRECTED 2026-07-28 (the first build of this artifact hedged in the wrong "
                "direction and understated its own result). Under view B a cell that produced no "
                "engine counts as 0.0. Measured, NOT most of Qwen's zeros are budget exhaustion: "
                f"per cell it is {zero_reason['cells']['budget_exhaustion_no_engine']} of "
                f"{zero_reason['cells']['qwen_zero_or_failed_cells']} "
                f"({zero_reason['cells']['budget_exhaustion_share_of_zero_cells']:.0%}), and per "
                "GAME -- the unit this criterion is computed on -- only "
                f"{len(rescuable)} of the "
                f"{len(zero_reason['games']['discordant_games_gemma_only_covered'])} discordant "
                f"games ({', '.join(rescuable) or 'none'}) are pure budget exhaustion. On the "
                "other "
                f"{len(zero_reason['games']['of_those_discordant_wrong_model'])} Qwen DID emit a "
                "loadable engine and every one scored exactly 0.0. So the coverage gap is mostly a "
                "WRONG-MODEL result, not a truncated-output result. Both arms got the same budget. "
                "This still does NOT license restating the result as a claim about intrinsic "
                "induction capability -- see budget_exhaustion_finding -- but the honest bound is "
                "tighter than the first build claimed."
            ),
            "what_a_raised_budget_could_recover": {
                "at_most_games": rescuable,
                "of_discordant_games": zero_reason["games"]["discordant_games_gemma_only_covered"],
                "so_coverage_could_move_at_most_to": (
                    views["B_zero_imputed"]["arms"]["qwen27b"]["coverage_nonzero_games"]
                    + len(rescuable)
                ),
                "principle": (
                    "Scoping the follow-up to the cells it could actually change stops a 'more "
                    "budget might fix it' hedge from being read as 'the gap is probably budget'."
                ),
            },
        },
        "qwen_zero_reason_split": zero_reason,
        "noise_floor": noise_floor,
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
                "No -- but ONLY because coverage carries the conclusion. CORRECTED 2026-07-28: the "
                "first build said gemma 'leads on the memorizing subset AND on the non-memorizing "
                "subset'. The second half of that is not supportable. The non-memorizing subset is "
                "n=2 scorable cells for gemma (mean 0.041650, the two values being 0.0833 and 0.0) "
                "against n=5 for Qwen (mean 0.0, every value 0.0). BOTH arms sit at the floor "
                "there, and 0.0417 over two cells is not a lead in any statistical sense -- it is "
                "one cell scoring 1/12. Per this project's Sample-Size Rigor rule a percentage-"
                "point delta needs n>=30; this is n=2. The experiment CANNOT distinguish the arms "
                "on the non-memorizing subset, which is precisely the subset that would test "
                "generalizing rather than window-fitting induction. The ranking therefore rests on "
                "COVERAGE (12/13 vs 4/13, p=0.0078, reproduced exactly across two gemma runs -- "
                "see noise_floor), which is robust, and the conclusion survives unchanged."
            ),
            "non_memorizing_subset_is_underpowered": {
                a: {
                    "n_scorable_cells": sum(
                        1 for r in rows[a] if scorable(r) and not r.get("is_memorizing")
                    ),
                    "values": sorted(
                        round(float(r["heldout_accuracy"]), 4)
                        for r in rows[a]
                        if scorable(r) and not r.get("is_memorizing")
                    ),
                }
                for a in ARMS
            },
        },
        "secondary_criteria": sec,
        "reason_engaged_evidence": reason_evidence,
        "code_provenance": code_prov,
        # TOP-LEVEL BOOLEANS, deliberately. scripts/summarize_artifact.py collects keys matching
        # acceptance_gate* and renders PASS only when the VALUE ITSELF is True -- a nested dict
        # renders as "[?]", i.e. present but unreadable, which is barely better than the
        # "(none found -- claim has no self-reported gate)" this replaces. The conditions and
        # principles live in decision_rule_detail, whose name deliberately does NOT match the
        # collector's pattern so it does not add a second unreadable "[?]" line.
        "acceptance_gate_both_arms_complete": gates["acceptance_gate_both_arms_complete"]["passed"],
        "acceptance_gate_coverage_separation": gates["acceptance_gate_coverage_separation"][
            "passed"
        ],
        "acceptance_gate_criterion_fixed_before_unblinding": gates[
            "acceptance_gate_criterion_fixed_before_unblinding"
        ]["passed"],
        "decision_rule_detail": gates,
        "verifiers_this_artifact_is_checked_by": {
            "scripts/adversarial_verify.py": (
                "fabrication / tautology / implausible-perfect / duration-floor / methodology "
                "checks on the artifact's own fields."
            ),
            "scripts/summarize_artifact.py": (
                "the mandated reading order (verdict -> flags -> gates -> duration/substrate -> "
                "metrics). Before 2026-07-28 it printed '(none found -- claim has no self-reported "
                "gate)' for this artifact; acceptance_gates now gives it something to read."
            ),
            "scripts/artifact_freshness_lint.py": (
                "input/code fingerprints. Before 2026-07-28 this artifact was INVISIBLE to it -- "
                "neither fresh nor stale -- because it declared no dependencies, so it escaped the "
                "check rather than passing it. The provenance block above fixes that. NOTE the "
                "lint is repo-wide and may refuse for other artifacts' drift; a refusal is not "
                "necessarily about this file."
            ),
            "scripts/determination_preservation_lint.py": (
                "checks that recorded determinations are not silently dropped."
            ),
            "what_none_of_them_check": (
                "That the two arms ran identical code. No linter in this repo fingerprints a "
                "module at IMPORT time inside a measurement process, which is why "
                "code_provenance had to be written by hand from mtimes and process start times."
            ),
        },
        "provenance": {
            "code": [
                {"path": d["path"], "sha256": d["sha256"]}
                for d in code_prov["dependencies_at_build_time"]
                if d["sha256"]
            ],
            "rows_sources": [
                {
                    "path": f"results/inducer_h2h_6021/h2h_shard_{a}.jsonl",
                    "sha256": _sha256_file(_evidence(f"h2h_shard_{a}.jsonl")),
                }
                for a in ARMS
            ]
            + [
                {
                    "path": "results/inducer_h2h_6021/h2h_reason_tag_probe.json",
                    "sha256": _sha256_file(_evidence("h2h_reason_tag_probe.json")),
                },
                {
                    "path": "results/inducer_h2h_6021/h2h_vram.jsonl",
                    "sha256": _sha256_file(_evidence("h2h_vram.jsonl")),
                },
            ],
            "rebuild_command": (
                "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python "
                "results/inducer_h2h_6021/build_artifact_6021.py"
            ),
            "what_this_block_does_and_does_not_certify": (
                "artifact_freshness_lint compares these sha256s against the files on disk, so a "
                "later edit to any of them marks this artifact STALE. That is a statement about "
                "the AGGREGATION and about the gemma arm, both of which ran against the bytes "
                "recorded here. It is NOT a statement about the qwen arm: see code_provenance -- "
                "arc_executable_world_model.py was rewritten mid-run, the qwen arm imported the "
                "earlier bytes, and those bytes were never snapshotted. A 'fresh' verdict from the "
                "lint must not be read as 'both arms ran identical code'. They did not."
            ),
        },
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
                "TOKEN completion budget to both arms. The difference is BUDGET EFFICIENCY: "
                "overran (llama.cpp stop_type == 'limit', i.e. it hit n_predict) fired on "
                f"{sec['qwen27b']['n_overran']}/{sec['qwen27b']['n_cells_run']} Qwen cells and "
                f"{sec['gemma31b']['n_overran']}/{sec['gemma31b']['n_cells_run']} gemma cells. "
                "When "
                "Qwen exhausts the budget it emits code missing the required engine / is_level_* "
                "functions, so induce_ok=False and no scorable engine exists. reason_engaged fired "
                "39/39 in both arms -- see reason_engaged_evidence for what that flag does and "
                "does NOT witness, and note that it is a CONSTANT here and so cannot by itself "
                "discriminate between the arms."
            ),
            "why_this_is_a_real_finding_not_a_harness_artifact": (
                "The rival explanation is that the shared '/think' prefix means different things "
                "to the two models, so a 16384-token budget penalises only Qwen. REWORDED "
                "2026-07-28: the first build refuted that with 'gemma emits reasoning tags on "
                "every cell too', which is WRONG ABOUT THE MECHANISM -- the tags reason_engaged "
                "matches are synthesised by the harness around any server-split reasoning_content, "
                "not written by the model. The refutation now rests on the addendum probe in "
                "reason_engaged_evidence, which kept the reasoning and the answer separately for "
                "one cell per arm; read its fairness_refutation_status field, which is computed "
                "from that measurement rather than asserted. Independently of it: both models get "
                "the same prompt, prefix, budget, quantisation, context, corpus and card, and a "
                "fixed generation budget is a REAL operational constraint for this project (the "
                "scored ARC path is latency- and VRAM-bounded), so 'fails to emit a usable engine "
                "within budget' is decision-relevant whatever the models are doing internally."
            ),
            "what_is_still_NOT_established": (
                "This does NOT establish that Qwen is intrinsically the weaker INDUCER. On the "
                "cells where Qwen did finish, it sometimes scored well (see view A). A larger "
                "budget, or /no_think, might rescue it -- but see qwen_zero_reason_split for HOW "
                "MUCH: at most "
                f"{len(rescuable)} of the "
                f"{len(zero_reason['games']['discordant_games_gemma_only_covered'])} discordant "
                "games are budget-limited, and "
                f"{zero_reason['cells']['engine_produced_but_scored_exactly_0']} of Qwen's "
                f"{n_qwen_successful_inductions} "
                "successful inductions scored 0.0 with budget to spare. So the honest claim is "
                "bounded: AT THIS BUDGET, with this mechanism, Qwen fails to deliver a usable "
                "world "
                "model on a large fraction of cells while gemma nearly always does -- and roughly "
                "half of that failure is budget, roughly half is a wrong model."
            ),
            "recommended_next_measurement": (
                "A third arm: Qwen3.6-27B at the same corpus with either /no_think or a raised "
                "budget, to separate 'induces worse' from 'spends its budget reasoning and runs "
                "out'. RE-SCOPED 2026-07-28: its ceiling is the "
                f"{len(rescuable)} pure-budget-exhaustion discordant games "
                f"({', '.join(rescuable) or 'none'}), so even a total success moves coverage from "
                f"{views['B_zero_imputed']['arms']['qwen27b']['coverage_nonzero_games']}/13 to at "
                f"most {qwen_cov_b + len(rescuable)}/13 "
                "against gemma's 12/13 -- it cannot overturn the verdict, only bound it. "
                "Deliberately NOT run here, because changing the prefix or budget would break "
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
            "Sequential arms, ONE llama-server at a time on GPU 1 (24 GiB cannot hold a 21GB and "
            "an "
            "18GB server together). GPU 0 was busy with an unrelated lane and was left alone. Card "
            "membership proven from per-PID residency vs GPU 1's UUID. Per-arm E3 dir so "
            "neither arm can read the other's induced world_model.py (a shared engine store "
            "contaminated an earlier run). Per-cell shard caching so a wedge costs at most one "
            "cell. Every cell gated on GPU-1 presence, >15GB residency, /props model identity, and "
            "a bounded REAL /completion -- because exp5833 died with /health returning 200 while "
            "/completion hung, so health is not liveness. Paired unit is the GAME (13), not the "
            "cell (39): the 3 trials are ONE DETERMINISTIC WINDOW sampled THREE TIMES UNSEEDED at "
            "temperature 0.2 (corrected 2026-07-28 -- the first build said 'repeats of one seeded "
            "window', which reads as if the sampling were seeded; it is not, see "
            "sampling_determinism). The two arms did NOT import identical code: see "
            "code_provenance."
        ),
        "prior_work_extended": {
            "exp5764_not_used_as_this_comparison_arm_sha256": (
                "ed7f0d14f3991d17c81e9bf6b2773c3848b9b59d76267efa29a3cdf063abaf04"
            ),
            "note": (
                "exp5764's gemma numbers are NOT this comparison's gemma arm. gemma was RE-RUN in "
                "this session so both arms are same-session, same-card, same-harness. exp5764 is "
                "cited only as a cross-check and is never modified (never-prune)."
            ),
            "exp5764_citation_is_view_matched": (
                "CORRECTED 2026-07-28. The first build cited exp5764 as 'pooled 0.378487, nonzero "
                "12/13' next to this run's 0.384328 -- but 0.378487 is exp5764's SCORABLE-ONLY "
                "(view A) pooled mean, while 0.384328 is this run's ZERO-IMPUTED (view B) mean. "
                "That is a view mismatch of exactly the kind this experiment exists to stop. "
                "Matched from exp5764's own shard: view B 0.352846 vs this run's 0.384328; view A "
                "0.378487 vs this run's 0.394585. Coverage 12/13 both runs, both views. The "
                "cross-check holds under either view -- see noise_floor, which does the comparison "
                "properly and uses it as this measurement's reproducibility bound."
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
