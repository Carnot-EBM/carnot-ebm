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


def _traj_sha(r: dict) -> str | None:
    """SHA-256 of the ordered (kind, data) move sequence the run actually issued."""
    seq = r.get("frame_sequence") or []
    if not seq:
        return None
    blob = json.dumps(
        [
            [
                f.get("loop_index"),
                (f.get("move") or {}).get("kind"),
                (f.get("move") or {}).get("data"),
            ]
            for f in seq
        ],
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def _level_seq_sha(r: dict) -> str | None:
    """SHA-256 of the ordered per-frame level sequence -- the outcome side of the trajectory."""
    seq = r.get("frame_sequence") or []
    if not seq:
        return None
    blob = json.dumps([f.get("level") for f in seq], sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


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
        # UNROUNDED companions (2026-07-27). The two fields above are rounded to 4 dp, and this
        # lane's headline divides one by the other: for tu93 that was literally 0.0003/0.0072 =
        # 1/24 = 0.041667, a six-decimal number carrying ONE significant figure. Every ratio below
        # is now computed from these.
        "efficiency_offline_precise": r.get("efficiency_precise"),
        "efficiency_gateway_charged_precise": r.get("efficiency_gateway_charged_precise"),
        "efficiency_optimism_vs_gateway_precise": r.get("efficiency_optimism_vs_gateway_precise"),
        # --- the MEASURED gateway charge, read off the arcade's own scorecard Card rather than
        #     modelled as offline_actions + resets. A sibling lane found the model wrong on 17 of 44
        #     cells because post-death actions return frame=[] and are never billed.
        "card_actions": r.get("gateway_card_actions"),
        "card_resets": r.get("gateway_card_resets"),
        "card_actions_by_level": r.get("gateway_card_actions_by_level"),
        "efficiency_gateway_card": r.get("efficiency_gateway_card"),
        "efficiency_gateway_card_error": r.get("efficiency_gateway_card_error"),
        "gateway_card_vs_model_charged_delta": r.get("gateway_card_vs_model_charged_delta"),
        "empty_frame_actions": r.get("empty_frame_actions"),
        "observed_full_resets": r.get("observed_full_resets"),
        "consecutive_reset_pairs": r.get("consecutive_reset_pairs"),
        # --- TRAJECTORY FINGERPRINTS. Two SHA-256s over what the agent actually did, so a claim of
        #     "pure addition" (this capture changed no behaviour) is checkable by a third party
        #     instead of asserted in prose.
        "trajectory_move_sha256": _traj_sha(r),
        "trajectory_level_sequence_sha256": _level_seq_sha(r),
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

    def _off_gw(c: dict) -> tuple[float | None, float | None]:
        """Offline / gateway scores at FULL precision, falling back to the 4-dp fields.

        The 4-dp fields exist for continuity with a long history of recorded numbers, but a RATIO
        of two 4-dp values is badly quantised -- tu93's rel_loss was exactly 1/24 (0.0003/0.0072),
        a six-decimal figure carrying one significant figure. The precise fields are used whenever
        present, and whether the fallback fired is recorded so a reader knows which they are seeing.
        """
        off = c.get("efficiency_offline_precise")
        gw = c.get("efficiency_gateway_charged_precise")
        if off is None:
            off = c.get("efficiency_offline_recorded")
        if gw is None:
            gw = c.get("efficiency_gateway_charged")
        return (None if off is None else float(off)), (None if gw is None else float(gw))

    # Relative score loss per won+scored cell: (offline - gateway) / offline, at FULL precision.
    rel_losses = []
    n_fell_back_to_rounded = 0
    for c in scored:
        off, gw = _off_gw(c)
        if (
            c.get("efficiency_offline_precise") is None
            or c.get("efficiency_gateway_charged_precise") is None
        ):
            n_fell_back_to_rounded += 1
        if off and off > 0 and gw is not None:
            rel_losses.append(
                {
                    "cell": f"{c['game']}@{c['seed']}",
                    "game": c["game"],
                    "seed": c["seed"],
                    "rel_loss": round(1 - gw / off, 9),
                    "trajectory_move_sha256": c.get("trajectory_move_sha256"),
                    # The MEASURED-charge counterpart, for the cells where the Card was read: the
                    # modelled charge (offline + resets) overstates wherever post-death actions were
                    # taken, so this is the number a corpus claim should cite going forward.
                    "rel_loss_from_card": (
                        round(1 - float(c["efficiency_gateway_card"]) / off, 9)
                        if (c.get("efficiency_gateway_card") is not None and off)
                        else None
                    ),
                }
            )

    # PER-SEED matched: the verdict is computed within each seed, never pooled across seeds.
    per_seed = {}
    for s in seeds:
        s_cells = [c for c in scored if c["seed"] == s]
        losses = []
        for c in s_cells:
            off, gw = _off_gw(c)
            if off and off > 0 and gw is not None:
                losses.append(1 - gw / off)
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

    # DISTINCT-MEASUREMENT COLLAPSE (2026-07-27). Cells whose move-trajectory SHA is identical are
    # the SAME measurement replicated by a seed that the agent's search never consumed. Collapsing
    # them by trajectory (falling back to the (game, spans) signature when a fingerprint is absent)
    # gives the effective support behind the median.
    def _sig(d: dict) -> str:
        return d.get("trajectory_move_sha256") or f"{d['game']}|{d['rel_loss']}"

    groups: dict[str, list[dict]] = {}
    for d in rel_losses:
        groups.setdefault(_sig(d), []).append(d)
    collapsed = [g[0]["rel_loss"] for g in groups.values()]
    distinct_summary = {
        "basis": "identical trajectory_move_sha256 (fallback: game + rel_loss)",
        "n_cells": len(rel_losses),
        "n_distinct": len(groups),
        "duplicate_groups": [
            {
                "n_cells": len(g),
                "cells": [x["cell"] for x in g],
                "rel_loss": g[0]["rel_loss"],
            }
            for g in groups.values()
            if len(g) > 1
        ],
        "median_over_distinct_trajectories": (
            round(statistics.median(collapsed), 9) if collapsed else None
        ),
        "max_over_distinct_trajectories": round(max(collapsed), 9) if collapsed else None,
        "why": (
            "a per-cell median triple-counts a seed-invariant game. The distinct-trajectory median "
            "is the one whose support equals its measurement count."
        ),
    }

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
            "unit": "relative loss = (offline_score - gateway_charged_score)/offline, FULL PRECISION",
            "precision_note": (
                "computed from the UNROUNDED scorer outputs (`efficiency_*_precise`). The 4-dp "
                "fields are preserved on every cell for continuity but must not be divided: a ratio "
                "of two 4-dp values quantises badly (tu93's was exactly 1/24)."
            ),
            "n_cells_that_fell_back_to_the_rounded_fields": n_fell_back_to_rounded,
            "per_cell": sorted(rel_losses, key=lambda d: -d["rel_loss"]),
            # BOTH aggregations published side by side (2026-07-27). The per-CELL median
            # triple-counts any game whose trajectory is seed-invariant -- this corpus has one
            # (tu93: byte-identical spans at all three seeds, flagged by
            # `whole_run_total_is_not_a_proxy_for_the_costly_share.per_game.<g>.seed_invariant_
            # trajectory`), so 3 of 7 cells were ONE measurement. A reader must be able to see the
            # effective support, not just the cell count.
            "median_per_cell": (
                round(statistics.median([d["rel_loss"] for d in rel_losses]), 9)
                if rel_losses
                else None
            ),
            "max_per_cell": round(max(d["rel_loss"] for d in rel_losses), 9)
            if rel_losses
            else None,
            "median": (
                round(statistics.median([d["rel_loss"] for d in rel_losses]), 9)
                if rel_losses
                else None
            ),
            "max": round(max(d["rel_loss"] for d in rel_losses), 9) if rel_losses else None,
            "distinct_measurements": distinct_summary,
            "effective_support": {
                "n_cells": len(rel_losses),
                "n_distinct_trajectories": distinct_summary.get("n_distinct"),
                "n_distinct_games": len({d["game"] for d in rel_losses}),
                "reading": (
                    "quote the DISTINCT-trajectory median beside the per-cell one. Where they "
                    "differ, the per-cell figure is weighting one trajectory by its seed count."
                ),
            },
            "median_from_the_CARD_measured_charge": (
                round(
                    statistics.median(
                        [
                            d["rel_loss_from_card"]
                            for d in rel_losses
                            if d["rel_loss_from_card"] is not None
                        ]
                    ),
                    9,
                )
                if any(d["rel_loss_from_card"] is not None for d in rel_losses)
                else None
            ),
            "card_vs_model_note": (
                "`median` above is the MODELLED charge (offline + resets). "
                "`median_from_the_CARD_measured_charge` is read off the gateway's own Card and is "
                "the number a forward claim should cite; the modelled one overstates wherever "
                "post-death (empty-frame) actions were taken."
            ),
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


def _recheck_cell(c: dict) -> list[str]:
    """INDEPENDENT re-derivation of every invariant a cell claims, from the cell's fields alone.

    Deliberately does NOT call `run_game`'s own reconciler: the point is a second implementation
    that can disagree. Returns the list of violated invariant names (empty == clean).
    """
    bad: list[str] = []
    off = c.get("run_offline_actions")
    res = c.get("run_resets")
    frames = c.get("run_frames")
    chg = c.get("run_gateway_charged")
    if None not in (off, res, frames) and frames != off + res:
        bad.append("frames_eq_offline_plus_resets")
    if None not in (off, res, chg) and chg != off + res:
        bad.append("charged_eq_offline_plus_resets")
    seg_off = c.get("segment_offline_actions") or []
    seg_res = c.get("segment_resets") or []
    seg_chg = c.get("segment_gateway_charged") or []
    if len(seg_off) == len(seg_chg) == len(seg_res):
        for i, (o, r_, g) in enumerate(zip(seg_off, seg_res, seg_chg)):
            if g != o + r_:
                bad.append(f"span_{i}_charged_eq_offline_plus_resets")
    else:
        bad.append("span_vectors_have_different_lengths")
    ric, rit = c.get("resets_in_completed_segments"), c.get("resets_in_tail")
    if None not in (ric, rit, res) and int(ric) + int(rit) != int(res):
        bad.append("resets_split_sums_to_run_resets")
    if None not in (seg_off, off) and sum(seg_off) + int(c.get("tail_offline_actions") or 0) != off:
        bad.append("span_offline_plus_tail_sums_to_run_offline")
    for k in ("empty_frame_actions", "observed_full_resets", "consecutive_reset_pairs"):
        v = c.get(k)
        if v is not None and int(v) < 0:
            bad.append(f"{k}_is_negative")
    offp = c.get("efficiency_offline_precise")
    gwp = c.get("efficiency_gateway_charged_precise")
    if None not in (offp, gwp) and float(gwp) > float(offp) + 1e-9:
        # charging MORE actions can never score HIGHER: the per-level score is
        # min((baseline/charged)**2*100, 115), monotonically decreasing in `charged`.
        bad.append("gateway_charged_score_exceeds_offline_score")
    ca = c.get("card_actions")
    if ca is not None and None not in (off, res):
        # the card can only ever charge <= offline+resets (free full resets, free post-death actions)
        if int(ca) > int(off) + int(res):
            bad.append("card_charge_exceeds_offline_plus_resets")
        if int(ca) < 0:
            bad.append("card_charge_is_negative")
    return bad


def _mutation_proofs(cells: list[dict]) -> dict:
    """Prove the invariant re-checker CATCHES a deliberately corrupted cell, mutation by mutation.

    WHY (2026-07-27). This lane previously claimed in prose that "10 mutations were applied and
    proved caught" while the shipped artifact contained no record of any of them -- the load-bearing
    safety claim was the one an auditor could not check. Every mutation below is applied to a deep
    COPY of a real cell, the independent re-checker is run, and what caught it is recorded. A
    mutation that ESCAPES is recorded as an escape, not dropped.
    """
    import copy

    base = None
    for c in cells:
        if c.get("levels", 0) > 0 and (c.get("segment_resets") or []):
            base = c
            break
    if base is None:
        return {"ran": False, "reason": "no won cell with span vectors to mutate"}

    def _bump(field, delta=1, index=None):
        def _f(d):
            if index is None:
                d[field] = (d.get(field) or 0) + delta
            else:
                d[field][index] = d[field][index] + delta

        return _f

    mutations = [
        ("run_resets_plus_1", _bump("run_resets")),
        ("run_frames_minus_1", _bump("run_frames", -1)),
        ("run_gateway_charged_plus_7", _bump("run_gateway_charged", 7)),
        ("segment_resets_0_plus_1", _bump("segment_resets", 1, 0)),
        ("segment_gateway_charged_0_minus_1", _bump("segment_gateway_charged", -1, 0)),
        ("segment_offline_actions_0_plus_3", _bump("segment_offline_actions", 3, 0)),
        ("tail_resets_plus_1", _bump("resets_in_tail")),
        ("empty_frame_actions_negative", lambda d: d.__setitem__("empty_frame_actions", -1)),
        (
            "gateway_score_beats_offline_score",
            lambda d: d.__setitem__(
                "efficiency_gateway_charged_precise",
                float(d.get("efficiency_offline_precise") or 1.0) * 1.5,
            ),
        ),
        (
            "card_charge_exceeds_offline_plus_resets",
            lambda d: d.__setitem__(
                "card_actions",
                int(d.get("run_offline_actions") or 0) + int(d.get("run_resets") or 0) + 100,
            ),
        ),
    ]
    clean = _recheck_cell(base)
    results = []
    for name, fn in mutations:
        m = copy.deepcopy(base)
        try:
            fn(m)
        except Exception as exc:
            results.append(
                {"mutation": name, "applied": False, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        found = _recheck_cell(m)
        new_violations = [v for v in found if v not in clean]
        results.append(
            {
                "mutation": name,
                "applied": True,
                "caught": bool(new_violations),
                "caught_by_invariants": new_violations,
            }
        )
    escaped = [r["mutation"] for r in results if r.get("applied") and not r.get("caught")]
    return {
        "ran": True,
        "mutated_cell": f"{base['game']}@{base['seed']}",
        "baseline_violations_on_the_unmutated_cell": clean,
        "n_mutations": len(mutations),
        "n_caught": sum(1 for r in results if r.get("caught")),
        "n_escaped": len(escaped),
        "escaped": escaped,
        "results": results,
        "principle": (
            "A safety claim that cannot be checked from the artifact is prose. Each mutation is "
            "applied to a deep copy of a real cell and the independent re-checker must produce a "
            "NEW violation; escapes are published, not hidden."
        ),
    }


def _pure_addition_proof(cells: list[dict], prior_path: str | None) -> dict:
    """Prove this capture ADDED fields and changed no recorded number, against the prior artifact.

    This edits `scripts/arc_leaderboard_eval.py`, a freshness-tracked dependency of five committed
    artifacts, so "pure addition" is the load-bearing claim. It is checked here rather than
    asserted: every key the PRIOR artifact recorded on a cell must be present with an EQUAL value in
    the newly measured cell of the same (game, seed).
    """
    if not prior_path:
        return {"ran": False, "reason": "no prior artifact given"}
    # Read the COMMITTED version from git, not the working-tree file. The working-tree file is
    # OVERWRITTEN by this very rebuild, so comparing against it would make the proof vacuous on every
    # run after the first (0 fields added, trivially "pure"). `git show HEAD:<path>` is the stable
    # control: the last version anyone actually published.
    source = f"git show HEAD:{prior_path}"
    raw = os.popen(f"git show HEAD:{prior_path} 2>/dev/null").read()
    if not raw.strip():
        if not Path(prior_path).exists():
            return {"ran": False, "reason": f"prior artifact not in git or on disk: {prior_path}"}
        raw = Path(prior_path).read_text()
        source = f"working tree {prior_path} (NOT in git at HEAD -- weaker control)"
    try:
        prior = json.loads(raw)
    except Exception as exc:
        return {"ran": False, "reason": f"prior artifact unreadable: {type(exc).__name__}: {exc}"}
    pcells = {f"{c.get('game')}@{c.get('seed')}": c for c in (prior.get("cells") or [])}
    new = {f"{c.get('game')}@{c.get('seed')}": c for c in cells}
    # `wall_s` is this capture's own LIVE clock: a re-measurement necessarily lands on a different
    # wall time, and counting that as a moved measurement number would make the boolean permanently
    # False and therefore useless. It is bucketed separately and reported, not silently ignored.
    LIVE_CLOCKS = {"wall_s"}
    matched, diffs, clock_moves, added = [], [], [], set()
    for key, pc in pcells.items():
        nc = new.get(key)
        if nc is None:
            diffs.append({"cell": key, "field": "<cell missing in new capture>"})
            continue
        matched.append(key)
        for f, pv in pc.items():
            if f not in nc:
                diffs.append({"cell": key, "field": f, "prior": pv, "new": "<ABSENT>"})
            elif nc[f] != pv:
                (clock_moves if f in LIVE_CLOCKS else diffs).append(
                    {"cell": key, "field": f, "prior": pv, "new": nc[f]}
                )
        added |= set(nc) - set(pc)
    return {
        "ran": True,
        "prior_artifact": prior_path,
        "prior_artifact_source": source,
        "n_cells_matched": len(matched),
        "n_prior_cells": len(pcells),
        "n_number_bearing_diffs": len(diffs),
        "diffs": diffs[:40],
        "n_live_clock_moves_EXPECTED_on_a_re_measurement": len(clock_moves),
        "live_clock_fields_excluded_from_the_boolean": sorted(LIVE_CLOCKS),
        "n_fields_ADDED": len(added),
        "fields_ADDED": sorted(added),
        "pure_addition": bool(not diffs),
        "principle": (
            "The prior artifact's own recorded values are the control. If any of them moved, this "
            "was not pure addition and the five dependent artifacts cannot be trusted to rebuild "
            "identically."
        ),
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
    ap.add_argument(
        "--compare-prior",
        default="results/arc_per_level_reset_attribution_20260726.json",
        help=(
            "prior artifact whose recorded cell values are the control for the PURE-ADDITION proof; "
            "pass '' to skip"
        ),
    )
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
        # PROOFS, not prose (2026-07-27). The prior run of this lane claimed byte-identical
        # trajectory fingerprints, 10 caught mutations, and 0 number-bearing diffs on 5 rebuilt
        # artifacts -- and shipped an artifact containing none of it. Both proofs are now computed
        # and recorded here, including any mutation that escaped.
        "tests_and_mutation_proofs": {
            "independent_invariant_recheck": {
                "n_cells_checked": len(ok),
                "n_cells_clean": sum(1 for c in ok if not _recheck_cell(c)),
                "violations_by_cell": {
                    f"{c['game']}@{c['seed']}": _recheck_cell(c) for c in ok if _recheck_cell(c)
                },
                "note": (
                    "a SECOND implementation of every invariant, derived from the cell fields alone "
                    "and never calling run_game's own reconciler -- two implementations that can "
                    "disagree is the only way a counting bug announces itself"
                ),
            },
            "mutation_proofs": _mutation_proofs(ok),
            "pure_addition_vs_prior_artifact": _pure_addition_proof(ok, a.compare_prior or None),
            "trajectory_fingerprints": {
                f"{c['game']}@{c['seed']}": {
                    "move_sha256": c.get("trajectory_move_sha256"),
                    "level_sequence_sha256": c.get("trajectory_level_sequence_sha256"),
                }
                for c in ok
            },
        },
        "provenance": _provenance(games, seeds, a.budget, a.out),
    }
    recon_ok = art["instrumentation_health"]["attribution_reconciles_everywhere"]
    nav_ok = art["instrumentation_health"]["nav_channel_live_everywhere"]
    n_won = art["scope_and_power"]["n_won_cells"]
    art["honest_verdict"] = (
        f"complete_per_level_reset_attribution_captured_{n_won}_won_cells_"
        f"reconciles_{str(recon_ok).lower()}_nav_channel_live_{str(nav_ok).lower()}"
    )
    ih = art["instrumentation_health"]
    # GATE_3's threshold, published (2026-07-27). It used to live only in the source, so a reader
    # could see `observed_won_cells: 7` and had no way to know what it was compared against.
    GATE3_MIN_WON_CELLS = 4
    n_distinct_won = (
        (art.get("score_loss_from_charged_resets") or {}).get("effective_support") or {}
    ).get("n_distinct_trajectories")
    art["acceptance_gates"] = {
        # Every gate now carries a COMPUTED witness at its own aggregation level. Previously gates 1
        # and 2 published only `passed` + `principle`, so their pass regions could not be checked in
        # place -- the counts existed but only under `instrumentation_health`, several keys away.
        "gate_1_attribution_reconciles_on_every_cell": {
            "passed": bool(recon_ok),
            "witness": {
                "n_cells": ih["cells_total"],
                "n_cells_with_reconciling_attribution": ih["cells_with_reconciling_attribution"],
                "n_cells_NOT_reconciling": ih["cells_total"]
                - ih["cells_with_reconciling_attribution"],
                "cells_not_reconciling": [
                    f"{c['game']}@{c['seed']}"
                    for c in cells
                    if c.get("attribution_reconciles") is not True
                ],
                "identity_frames_eq_actions_plus_resets_holds_all_cells": ih[
                    "identity_holds_all_cells"
                ],
            },
            "principle": (
                "Two independent accountings that must agree is the only way an off-by-one in a "
                "counting loop announces itself; a single accounting merely looks plausible."
            ),
        },
        "gate_2_nav_channel_populated_not_dead": {
            "passed": bool(nav_ok),
            "witness": {
                "n_cells": ih["cells_total"],
                "n_cells_with_instrumented_nav_channel": ih["cells_with_instrumented_nav_channel"],
                "n_cells_with_a_DEAD_nav_channel": ih["cells_total"]
                - ih["cells_with_instrumented_nav_channel"],
                "uninstrumented_reasons": sorted(
                    {
                        str(c.get("navdiag_uninstrumented_reason"))
                        for c in cells
                        if c.get("navdiag_instrumented") is not True
                    }
                ),
                "n_cells_carrying_a_gateway_charge_error": sum(
                    1 for c in cells if c.get("efficiency_gateway_charged_error")
                ),
            },
            "principle": (
                "An absent channel that reports 0 reads as a measured zero -- the defect that made "
                "a dead getattr(env,'baseline_actions') look like a clean null."
            ),
        },
        "gate_3_enough_won_cells_for_the_attribution_to_be_real": {
            "passed": bool(n_won >= GATE3_MIN_WON_CELLS),
            "observed_won_cells": n_won,
            "threshold": GATE3_MIN_WON_CELLS,
            "witness": {
                "n_won_cells": n_won,
                "threshold": GATE3_MIN_WON_CELLS,
                # The threshold is evaluated against the CELL count for continuity, but the
                # DISTINCT-trajectory count is the effective support and is published beside it: a
                # seed-invariant game contributes one measurement, not three.
                "n_distinct_won_trajectories": n_distinct_won,
                "passes_on_distinct_count_too": (
                    None if n_distinct_won is None else bool(n_distinct_won >= GATE3_MIN_WON_CELLS)
                ),
            },
            "principle": (
                "Attribution only exists where a level-up exists. A capture with no won cell "
                "demonstrates plumbing, not attribution."
            ),
        },
        "gate_4_the_card_was_read_not_modelled": {
            # NEW 2026-07-27. The modelled charge (offline + resets) was found wrong on 17 of 44
            # cells in a sibling lane because post-death actions return frame=[] and are never
            # billed. This gate requires the gateway's OWN bookkeeping to have been consulted.
            "passed": bool(ok and all(c.get("card_actions") is not None for c in ok)),
            "witness": {
                "n_cells": len(ok),
                "n_cells_with_a_card_read": sum(1 for c in ok if c.get("card_actions") is not None),
                "n_cells_where_model_and_card_DISAGREE": sum(
                    1 for c in ok if int(c.get("gateway_card_vs_model_charged_delta") or 0) != 0
                ),
                "total_post_death_uncharged_actions": sum(
                    int(c.get("empty_frame_actions") or 0) for c in ok
                ),
            },
            "principle": (
                "A charge MODEL cannot detect the error it does not model. Two reconstructions of "
                "the same assumption agreeing is not independence."
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
