#!/usr/bin/env python3
"""LIVE capture of PER-LEVEL reset attribution across a small ARC corpus.

WHAT THIS MEASURES AND WHY IT NEEDED A NEW CAPTURE
==================================================
Every per-level efficiency number this project holds is recorded in OFFLINE ACTIONS, a unit that
EXCLUDES resets. The live gateway CHARGES a reset an action (`arc_agi/scorecard.py:701-704`
`inc_reset_count` does `resets += 1` AND `actions += 1`, reached from `update_scorecard`:839-843),
and the per-level score is `min((baseline / charged)**2 * 100, 115)` (:166-173) over a DIFFERENCE of
cumulative charged counts (:479). So the optimism in any per-level number is quadratic in the
resets charged BEFORE that level-up -- and a whole-run `n_resets` cannot be apportioned across
levels after the fact. The attribution has to be recorded per inter-level-up SPAN while the run
happens, which `arc_leaderboard_eval.run_game` now does.

THREE UNITS, NEVER CONFLATED. Each span is emitted in all three:
  offline_actions  -- `actions`; EXCLUDES resets. The unit every historical number is in.
  frames           -- loop iterations; INCLUDES resets. The unit the early-stop window counts in.
  gateway_charged  -- offline_actions + resets. The ONLY unit the competition score depends on.

WHAT THIS IS NOT. This is a LOCAL, OFFLINE measurement on PUBLIC games with the LLM disabled. It
submits nothing, flips no flag, and rewrites no historical artifact's recorded numbers -- the
re-scored figures here are emitted as NEW fields alongside the originals, which are preserved
verbatim from the same run.

Per-seed MATCHED: every (game, seed) is its own cell and the corpus verdict is computed per seed,
never as an any-seed union -- a union lets one lucky seed carry a claim.

Usage:
  arc_per_level_reset_attribution_capture.py --games vc33,tu93,... --seeds a,b,c --budget 400 --out F
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

# The LLM is OFF: this is a navigation/accounting measurement, not an induction measurement. Set
# before importing the agent so no proposer is constructed.
os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"


def _score_from_charged(spans: list[int], baselines: list[int], n_levels_total: int) -> float:
    """Drive the INSTALLED scorer over a per-level charged-action vector.

    Uses `arc_agi.scorecard.EnvironmentScoreCalculator` exactly as the gateway drives it rather
    than reimplementing the formula -- a 2026-06-20 review caught a paraphrase of this formula
    being wrong on three separate counts, so the installed scorer is treated as the definition.
    """

    from arc_agi.scorecard import EnvironmentScoreCalculator

    calc = EnvironmentScoreCalculator()
    for li in range(n_levels_total):
        completed = li < len(spans)
        calc.add_level(
            level_index=li + 1,
            completed=completed,
            actions_taken=int(spans[li]) if completed else 0,
            baseline_actions=int(baselines[li]),
        )
    return round(float(calc.to_score(include_levels=False).score), 6)


def run_cell(game: str, seed: int, budget: int) -> dict:
    """One SCORED-path cell: E3AgentPolicy -> StepwiseExplorer, offline arcade, LLM off."""

    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    random.seed(seed)
    np.random.seed(seed % (2**32))
    t0 = time.time()
    policy = E3AgentPolicy(game, frontier_discipline_seed=seed)
    r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
    wall = round(time.time() - t0, 3)

    attr = r.get("level_reset_attribution") or {}
    nav = r.get("navigation_diagnostics") or {}
    spans_offline = list(attr.get("segment_offline_actions") or [])
    spans_charged = list(attr.get("segment_gateway_charged") or [])
    spans_resets = list(attr.get("segment_resets") or [])
    spans_frames = list(attr.get("segment_frames") or [])

    cell = {
        "game": game,
        "seed": seed,
        "budget": budget,
        "wall_s": wall,
        "levels": int(r["levels"]),
        "reached": int(r["reached"]),
        # --- the three units, whole run
        "run_offline_actions": int(r["actions"]),
        "run_resets": int(r.get("n_resets_run_game") or 0),
        "run_frames": len(r.get("frame_sequence") or []),
        "run_gateway_charged": int(r.get("charged_actions") or 0),
        # --- PER-SPAN attribution, all three units
        "segment_offline_actions": spans_offline,
        "segment_resets": spans_resets,
        "segment_frames": spans_frames,
        "segment_gateway_charged": spans_charged,
        "tail_offline_actions": attr.get("tail_offline_actions"),
        "tail_resets": attr.get("tail_resets"),
        "tail_frames": attr.get("tail_frames"),
        "tail_gateway_charged": attr.get("tail_gateway_charged"),
        "resets_in_completed_segments": attr.get("resets_in_completed_segments"),
        "resets_in_tail": attr.get("resets_in_tail"),
        "attribution_reconciles": attr.get("reconciles"),
        "attribution_discrepancies": attr.get("discrepancies"),
        # --- the scores the two units produce (originals PRESERVED, not overwritten)
        "efficiency_offline_recorded": r.get("efficiency"),
        "efficiency_gateway_charged": r.get("efficiency_gateway_charged"),
        "efficiency_gateway_charged_error": r.get("efficiency_gateway_charged_error"),
        "efficiency_optimism_vs_gateway": r.get("efficiency_optimism_vs_gateway"),
        # --- navigation channel (previously dropped entirely by the row projection)
        "navdiag_instrumented": nav.get("instrumented"),
        "navdiag_uninstrumented_reason": nav.get("uninstrumented_reason"),
        "navdiag_attempts": nav.get("navigation_attempts"),
        "navdiag_reset_replay_fallbacks": nav.get("reset_replay_fallbacks"),
        "navdiag_exact_hits": nav.get("exact_shortest_path_hits"),
        "navdiag_partial_hits": nav.get("partial_forward_walk_hits"),
        "navdiag_similarity_hits": nav.get("similarity_forward_walk_hits"),
        "navdiag_forward_walk_hit_rate": nav.get("forward_walk_hit_rate"),
        "navdiag_forward_edges_recorded": nav.get("forward_edges_recorded"),
    }

    # The identity that says the two accountings close: frames == offline_actions + resets.
    cell["identity_frames_eq_actions_plus_resets"] = (
        cell["run_frames"] == cell["run_offline_actions"] + cell["run_resets"]
    )
    # Whole-run resets MINUS resets attributable to completed spans = resets in the post-solve
    # tail, which the gateway charges but which cost ZERO score (an incomplete level scores 0
    # regardless of actions charged, scorecard.py:178-183). Splitting them is the entire point.
    if cell["resets_in_completed_segments"] is not None:
        cell["resets_that_cost_score"] = int(cell["resets_in_completed_segments"])
        cell["resets_that_cost_nothing"] = int(cell["resets_in_tail"] or 0)

    # Per-span RELATIVE optimism: how much bigger the charged span is than the recorded one.
    cell["per_span_charge_inflation"] = [
        (round(c / o, 6) if o else None) for o, c in zip(spans_offline, spans_charged)
    ]
    return cell


def summarize(cells: list[dict], *, games: list[str], seeds: list[int], budget: int) -> dict:
    """Corpus verdict, PER-SEED MATCHED, with scope and power hoisted beside it."""

    won = [c for c in cells if c["levels"] > 0]
    scored = [
        c
        for c in won
        if c.get("efficiency_gateway_charged") is not None
        and c.get("efficiency_offline_recorded") is not None
    ]
    # Relative score loss per won+scored cell: (offline - gateway) / offline.
    rel_losses = []
    for c in scored:
        off = float(c["efficiency_offline_recorded"] or 0.0)
        gw = float(c["efficiency_gateway_charged"] or 0.0)
        if off > 0:
            rel_losses.append(
                {"cell": f"{c['game']}@{c['seed']}", "rel_loss": round(1 - gw / off, 6)}
            )

    # PER-SEED matched: the verdict is computed within each seed, never pooled across seeds.
    per_seed = {}
    for s in seeds:
        s_cells = [c for c in scored if c["seed"] == s]
        losses = [
            1 - float(c["efficiency_gateway_charged"]) / float(c["efficiency_offline_recorded"])
            for c in s_cells
            if float(c["efficiency_offline_recorded"] or 0) > 0
        ]
        per_seed[str(s)] = {
            "n_cells": len([c for c in cells if c["seed"] == s]),
            "n_won": len([c for c in cells if c["seed"] == s and c["levels"] > 0]),
            "n_scored": len(s_cells),
            "median_rel_score_loss": round(statistics.median(losses), 6) if losses else None,
            "max_rel_score_loss": round(max(losses), 6) if losses else None,
            "games_won": sorted({c["game"] for c in cells if c["seed"] == s and c["levels"] > 0}),
        }

    # The headline the attribution makes possible and a whole-run total does not: for each won
    # cell, HOW MANY of its resets landed before a level-up (costing score, quadratically) versus
    # in the post-solve tail (costing nothing).
    split = [
        {
            "cell": f"{c['game']}@{c['seed']}",
            "run_resets": c["run_resets"],
            "resets_that_cost_score": c.get("resets_that_cost_score"),
            "resets_that_cost_nothing": c.get("resets_that_cost_nothing"),
            "segment_resets": c["segment_resets"],
            "segment_offline_actions": c["segment_offline_actions"],
            "segment_gateway_charged": c["segment_gateway_charged"],
        }
        for c in won
    ]
    costly = [s["resets_that_cost_score"] for s in split if s["resets_that_cost_score"] is not None]
    free = [
        s["resets_that_cost_nothing"] for s in split if s["resets_that_cost_nothing"] is not None
    ]

    n_recon = sum(1 for c in cells if c.get("attribution_reconciles") is True)
    n_nav = sum(1 for c in cells if c.get("navdiag_instrumented") is True)

    # IS THE WHOLE-RUN TOTAL A USABLE PROXY FOR THE COSTLY SHARE? Computed, not asserted. For each
    # game with >=2 won cells, compare the spread of `run_resets` against the spread of
    # `resets_that_cost_score`, and check whether ordering by the total preserves ordering by the
    # costly share. If it does not, the total is not merely imprecise -- it is misleading, which is
    # the whole case for per-span attribution.
    proxy = {}
    for g in sorted({c["game"] for c in won}):
        gc = sorted(
            [c for c in won if c["game"] == g and c.get("resets_that_cost_score") is not None],
            key=lambda c: c["seed"],
        )
        if len(gc) < 2:
            continue
        totals = [int(c["run_resets"]) for c in gc]
        costly_g = [int(c["resets_that_cost_score"]) for c in gc]
        by_total = [costly_g[i] for i in sorted(range(len(gc)), key=lambda i: totals[i])]
        proxy[g] = {
            "n_seeds": len(gc),
            "seeds": [c["seed"] for c in gc],
            "run_resets_per_seed": totals,
            "resets_that_cost_score_per_seed": costly_g,
            "run_resets_spread_ratio": (
                round(max(totals) / min(totals), 4) if min(totals) > 0 else None
            ),
            "costly_spread_ratio": (
                round(max(costly_g) / min(costly_g), 4) if min(costly_g) > 0 else None
            ),
            "ordering_by_total_preserves_ordering_by_costly": by_total == sorted(costly_g),
            "seed_invariant_trajectory": len(set(totals)) == 1 and len(set(costly_g)) == 1,
        }

    return {
        "scope_and_power": {
            "games": games,
            "seeds": seeds,
            "budget": budget,
            "n_cells": len(cells),
            "n_won_cells": len(won),
            "n_scored_cells": len(scored),
            "per_seed_matched": True,
            "what_this_cannot_support": (
                "A small deliberately-chosen corpus at ONE budget with the LLM OFF. The per-cell "
                "attribution is exact for the cells measured; the reset SHARES are single "
                "measurements per (game, seed) and must NOT be read as corpus point estimates. "
                "The plan-execution reset source (arc_competition_agent.py:5314) fires only with "
                "induction ON and is therefore still UNMEASURED here."
            ),
        },
        "instrumentation_health": {
            "cells_with_reconciling_attribution": n_recon,
            "cells_total": len(cells),
            "attribution_reconciles_everywhere": n_recon == len(cells),
            "cells_with_instrumented_nav_channel": n_nav,
            "nav_channel_live_everywhere": n_nav == len(cells),
            "identity_holds_all_cells": all(
                c["identity_frames_eq_actions_plus_resets"] for c in cells
            ),
            "why_these_are_reported": (
                "A dead channel reads as a clean null, so the population of every added field is "
                "asserted rather than assumed. `attribution_reconciles` is the cross-check of the "
                "in-loop accumulators against an independent frame_sequence re-derivation AND "
                "against the whole-run counters; anything but True on every cell means one of the "
                "two accountings drifted."
            ),
        },
        "the_split_a_whole_run_total_cannot_make": {
            "note": (
                "Resets before a level-up are charged INTO that level's squared-efficiency "
                "denominator; resets in the post-solve tail are charged but cost nothing, because "
                "an incomplete level scores 0 regardless of actions charged. Only per-span "
                "attribution separates them."
            ),
            "per_cell": split,
            "median_resets_that_cost_score": (
                round(statistics.median(costly), 3) if costly else None
            ),
            "median_resets_that_cost_nothing": round(statistics.median(free), 3) if free else None,
            "total_resets_that_cost_score": sum(costly) if costly else 0,
            "total_resets_that_cost_nothing": sum(free) if free else 0,
        },
        "score_loss_from_charged_resets": {
            "unit": "relative loss = (offline_recorded_score - gateway_charged_score)/offline",
            "per_cell": sorted(rel_losses, key=lambda d: -d["rel_loss"]),
            "median": (
                round(statistics.median([d["rel_loss"] for d in rel_losses]), 6)
                if rel_losses
                else None
            ),
            "max": round(max(d["rel_loss"] for d in rel_losses), 6) if rel_losses else None,
        },
        "whole_run_total_is_not_a_proxy_for_the_costly_share": {
            "per_game": proxy,
            "games_where_ordering_by_total_fails": sorted(
                g
                for g, v in proxy.items()
                if not v["ordering_by_total_preserves_ordering_by_costly"]
            ),
            "power_note": (
                "This is a WITNESS that the whole-run total mis-ranks the costly share on the "
                "cells measured, not a correlation estimate. Each game has at most 3 seeds, so no "
                "rank test here could reach p<0.05 (a 3-point support cannot). The claim being "
                "made is existential -- 'a cell with MORE total resets can have FEWER costly ones' "
                "-- and one counterexample settles it."
            ),
        },
        "per_seed_matched": per_seed,
    }


def _sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


# The code this capture's numbers depend on. Recorded so `scripts/artifact_freshness_lint.py` can
# refuse a commit that edits any of it without rebuilding -- a stale artifact whose dependency has
# silently moved is a published number nobody can reproduce.
_CODE_DEPS = (
    "scripts/arc_per_level_reset_attribution_capture.py",
    "scripts/arc_leaderboard_eval.py",
    "python/carnot/agentic/arc_competition_agent.py",
)


def _provenance(games: list[str], seeds: list[int], budget: int, out: str) -> dict:
    """Fingerprint the code, and record the exact command that rebuilds this artifact.

    There are no `rows_sources` here: this is a LIVE capture, so its inputs are the games'
    environment files plus the agent code, not a persisted row file. The rebuild command is
    therefore a re-MEASUREMENT, which is stated plainly rather than implying a cheap re-analysis.
    """

    return {
        "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,
        "code": [
            {"path": d, "sha256": _sha256(REPO / d), "bytes": (REPO / d).stat().st_size}
            for d in _CODE_DEPS
            if (REPO / d).exists()
        ],
        "rebuild_command": (
            f"{sys.executable} scripts/arc_per_level_reset_attribution_capture.py "
            f"--games {','.join(games)} --seeds {','.join(str(s) for s in seeds)} "
            f"--budget {budget} --out {out}"
        ),
        "rebuild_is_a_re_measurement_not_a_re_analysis": True,
        "submitted_nothing": True,
        "flags_unchanged": True,
        "max_actions_untouched": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="vc33,tu93,sc25,dc22,r11l,cd82")
    ap.add_argument("--seeds", default="20260724,20260725,20260726")
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    games = [g for g in a.games.split(",") if g]
    seeds = [int(s) for s in a.seeds.split(",") if s]

    # THE CLOCK. This project has published an ANALYSER's runtime as if it were the measurement's
    # wall clock, so the distinction is declared explicitly. Here the script IS the measurement --
    # the loop below drives the live agent -- so the analyser clock and the measurement clock are
    # the SAME quantity. That is asserted as a boolean rather than by emitting the same float
    # twice: two identical floats under two names is a fake second metric, and this project's own
    # adversarial linter correctly flags it as a TAUTOLOGY.
    t0 = time.time()
    cells = []
    # SEED is the OUTER loop and game the inner, so a (game, seed) cell is never adjacent to its
    # own replicate -- any drift in machine state spreads across games rather than concentrating
    # in one game's seed series.
    for seed in seeds:
        for game in games:
            try:
                cells.append(run_cell(game, seed, a.budget))
                c = cells[-1]
                print(
                    f"  {game}@{seed}: levels={c['levels']} off={c['run_offline_actions']} "
                    f"resets={c['run_resets']} charged={c['run_gateway_charged']} "
                    f"spans_off={c['segment_offline_actions']} spans_chg={c['segment_gateway_charged']} "
                    f"recon={c['attribution_reconciles']} nav={c['navdiag_instrumented']} "
                    f"({c['wall_s']}s)",
                    flush=True,
                )
            except Exception as exc:  # a dead cell is RECORDED, never silently dropped
                cells.append(
                    {
                        "game": game,
                        "seed": seed,
                        "budget": a.budget,
                        "levels": 0,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                        "identity_frames_eq_actions_plus_resets": False,
                    }
                )
                print(f"  {game}@{seed}: ERROR {type(exc).__name__}", flush=True)
    measurement_wall_s = round(time.time() - t0, 3)

    ok = [c for c in cells if "error" not in c]
    art = {
        "experiment": "arc_per_level_reset_attribution_capture",
        "title": "PER-LEVEL reset attribution, live capture across a small ARC corpus",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seeds[0],
        "random_seeds_used": seeds,
        # THE LIVE measurement clock. `duration_s` here IS the measurement wall clock, because the
        # loop above drives the agent -- there is no separate analyser pass whose runtime could be
        # mistaken for it. Asserted as a flag rather than duplicated as a second identical float.
        "duration_s": measurement_wall_s,
        "measurement_clock_is_the_analyser_clock_because_this_is_a_live_capture": True,
        # Published ONLY to make its own shortfall visible: summing per-cell wall clocks
        # UNDERCOUNTS (per-cell timing excludes policy construction and inter-cell overhead), so it
        # must never stand in for the measurement clock -- that substitution is what this project
        # shipped once before, at a ~25% shortfall.
        "sum_per_cell_wall_s": round(sum(float(c.get("wall_s") or 0.0) for c in cells), 3),
        "sum_per_cell_wall_s_is_an_undercount_do_not_use_as_measurement_clock": True,
        "inference_substrate": SUBSTRATE,
        "inference_substrate_note": (
            "The live agent takes real actions against the OFFLINE arcade with the LLM disabled "
            "(CARNOT_ARC_DISABLE_INDUCTION=1): pure Python env-stepping plus verifier-routed "
            "search. No GGUF is loaded, so model_specs is not applicable to this substrate."
        ),
        "llm_enabled": False,
        "n_cells": len(cells),
        "n_cells_errored": len(cells) - len(ok),
        "cells": cells,
        **summarize(ok, games=games, seeds=seeds, budget=a.budget),
        "honest_verdict": "",  # filled below
        "provenance": _provenance(games, seeds, a.budget, a.out),
    }
    recon_ok = art["instrumentation_health"]["attribution_reconciles_everywhere"]
    nav_ok = art["instrumentation_health"]["nav_channel_live_everywhere"]
    n_won = art["scope_and_power"]["n_won_cells"]
    art["honest_verdict"] = (
        f"complete_per_level_reset_attribution_captured_{n_won}_won_cells_"
        f"reconciles_{str(recon_ok).lower()}_nav_channel_live_{str(nav_ok).lower()}"
    )
    art["acceptance_gates"] = {
        "gate_1_attribution_reconciles_on_every_cell": {
            "passed": bool(recon_ok),
            "principle": (
                "Two independent accountings that must agree is the only way an off-by-one in a "
                "counting loop announces itself; a single accounting merely looks plausible."
            ),
        },
        "gate_2_nav_channel_populated_not_dead": {
            "passed": bool(nav_ok),
            "principle": (
                "An absent channel that reports 0 reads as a measured zero -- the defect that made "
                "a dead getattr(env,'baseline_actions') look like a clean null."
            ),
        },
        "gate_3_enough_won_cells_for_the_attribution_to_be_real": {
            "passed": bool(n_won >= 4),
            "observed_won_cells": n_won,
            "principle": (
                "Attribution only exists where a level-up exists. A capture with no won cell "
                "demonstrates plumbing, not attribution."
            ),
        },
    }
    payload = json.dumps(art, indent=1, sort_keys=True)
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=1, sort_keys=True))
    # Register under THIS script, not the helper's own file. The helper's `analyzer` default is a
    # documented trap: the first external reuser registered its artifact under the wrong analyser,
    # which sends a future reader chasing the wrong rebuild command for a drifted artifact.
    try:
        from analyze_scored_path_lever_ab import register_analyzed_artifact

        register_analyzed_artifact(out, analyzer=Path(__file__))
    except Exception as exc:  # registration is a guard, not the measurement -- never fail the run
        print(f"  WARNING: freshness registration failed ({type(exc).__name__}); register manually")
    print(f"\nwrote {out}  ({measurement_wall_s}s, {len(ok)}/{len(cells)} cells ok)")
    print(f"verdict: {art['honest_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
