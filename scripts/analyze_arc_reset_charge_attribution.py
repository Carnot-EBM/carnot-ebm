#!/usr/bin/env python3
"""WHERE DO RESETS COME FROM ON THE SCORED PATH, AND WHICH ONES ARE AVOIDABLE?

This is an ANALYSER + a small LIVE MEASUREMENT, kept in one script so the two halves
cannot drift apart. Read the two halves as two different measurements with two
different clocks (CLAUDE.md "THE ANALYSER CLOCK IS NOT THE MEASUREMENT CLOCK"):

PART A -- AGGREGATION over the 1401 already-persisted sweep rows
  (results/early_stop_sweep_20260726/rows_*.json). Bounds the damage the uncharged-RESET
  defect does to every efficiency number this project currently holds, WITHOUT new
  instrumentation. Substrate: aggregation_from_upstream_artifacts. Its
  `measurement_wall_s` is the sum of each upstream row FILE's own `elapsed_s` -- NOT this
  script's runtime, and NOT the sum of per-cell `wall_s` (that undercounts ~25%).

PART B -- LIVE scored-path runs (offline arcade, no LLM) that measure the reset
  composition the rows cannot show, because the lever harness never projected
  `navigation_diagnostics` into its rows. Substrate:
  offline_arcade_live_agent_runtime_self_discovery_no_llm. Its own wall clock is
  measured separately as `part_b_measurement_wall_s`.

PART C -- the OFFLINE TWIN (arc_solver_kit.OfflineSolver) reset count, on the same games,
  to answer whether the dev twin is a faithful proxy for gateway-charged score at all.

THE DEFECT UNDER MEASUREMENT
  The live gateway charges a RESET one action:
    arc_agi/scorecard.py:701-704 inc_reset_count -> `resets += 1` AND `actions += 1`,
    reached from update_scorecard:839-843 -> reset()/new_play() (:762,:768).
  Our offline harness charges a RESET ZERO:
    scripts/arc_leaderboard_eval.py:308-313 -- `actions += 1` lives ONLY in the
    non-RESET branch.
  Per-level charge is a DIFFERENCE of reset-INCLUSIVE cumulative counts
  (scorecard.py:706-713 set_levels_completed appends (level, self.actions[index]);
  :475-479 level_actions = actions_at_level - prev_actions), so resets taken BEFORE a
  level-up land inside that level's denominator and are squared (:168-171).

THREE UNITS, NEVER CONFLATED. Every number this script emits is labelled with one:
  offline_actions  -- our harness `actions`; EXCLUDES resets. What `level_up_actions` is in.
  frames           -- loop iterations; INCLUDES resets.
  gateway_charged  -- non-RESET moves PLUS resets. The only unit the score is a function of.
An identity worth stating, and asserted below on every live run: because the gateway
charges exactly one per RESET and exactly one per action, and our loop emits exactly one
of the two per iteration,
    gateway_charged_total == frames == offline_actions + n_resets.
So `n_frames` in the existing rows ALREADY IS the whole-run gateway-charged total. What
is missing is not the total -- it is the PER-LEVEL SPLIT.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS_DIR = os.path.join(REPO, "results", "early_stop_sweep_20260726")

# ---------------------------------------------------------------------------------------
# THE SCORER. Never re-implement the competition formula and trust it by eye: implement a
# fast version for the allocation SEARCH (which needs thousands of evaluations), then
# ASSERT it agrees with the installed authoritative scorer on every point actually
# reported. A search that is fast but wrong is worse than no search.
# ---------------------------------------------------------------------------------------


def authoritative_game_score(baselines: list[int], charged: list[int], n_completed: int) -> float:
    """Drive the INSTALLED arc_agi scorer. `charged[i]` is level i's gateway-charged
    action count (resets included). Mirrors scorecard.py:465-487's per-level loop."""
    from arc_agi.scorecard import EnvironmentScoreCalculator

    calc = EnvironmentScoreCalculator(id="probe")
    for i, base in enumerate(baselines):
        calc.add_level(
            level_index=i + 1,
            completed=(i < n_completed),
            actions_taken=int(charged[i]) if i < len(charged) else 0,
            baseline_actions=int(base),
        )
    return float(calc.to_score(include_levels=True).score)


def fast_game_score(baselines: list[int], charged: list[int], n_completed: int) -> float:
    """Same arithmetic, no object churn. scorecard.py:168-171 (per level) + :192-206
    (index-weighted mean with the max_weights clamp)."""
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for i in range(len(baselines)):
        w = i + 1
        total_weights += w
        if i < n_completed:
            a = int(charged[i]) if i < len(charged) else 0
            s = min((float(baselines[i]) / a) ** 2 * 100.0, 115.0) if a > 0 else 0.0
        else:
            s = 0.0
        total_score += s * w
        if s > 0:
            max_weights += w
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100.0)


# ---------------------------------------------------------------------------------------
# PART A -- the bound over existing rows.
# ---------------------------------------------------------------------------------------


def _segments_offline(level_up_actions: list[int]) -> list[int]:
    """Offline-action length of each completed inter-level-up segment.
    level_up_actions is CUMULATIVE offline actions at each level-up
    (arc_leaderboard_eval.py:301,:321-324)."""
    segs = []
    prev = 0
    for cum in level_up_actions:
        segs.append(int(cum) - prev)
        prev = int(cum)
    return segs


def _worst_allocation(
    baselines: list[int], segs: list[int], n_completed: int, budget: int
) -> tuple[float, list[int]]:
    """Distribute `budget` resets across the completed segments to MINIMISE the game
    score. The tail (post-last-level-up) is a free sink and costs exactly nothing
    (today's finding), so the worst case necessarily puts every reset before a level-up.

    Exhaustive for 1-2 completed segments; coordinate-descent from every all-in-one
    corner for 3+, which is adequate because the objective is separable and each term is
    monotone in its own allocation -- the reported number is therefore an UPPER bound on
    the true worst case's score, i.e. the reported RANGE is if anything conservative
    (too narrow), never overstated."""
    n = n_completed
    if n <= 0 or budget <= 0:
        return fast_game_score(baselines, segs, n_completed), [0] * max(n, 1)
    if n == 1:
        alloc = [budget]
        return fast_game_score(baselines, [segs[0] + budget], n_completed), alloc
    if n == 2:
        best_s, best_a = None, None
        for k in range(budget + 1):
            ch = [segs[0] + k, segs[1] + (budget - k)]
            s = fast_game_score(baselines, ch, n_completed)
            if best_s is None or s < best_s:
                best_s, best_a = s, [k, budget - k]
        return float(best_s), list(best_a or [])
    # 3+ completed segments: start from each corner, then coordinate-descend.
    best_s, best_a = None, None
    for corner in range(n):
        alloc = [0] * n
        alloc[corner] = budget
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(n):
                    if i == j or alloc[i] == 0:
                        continue
                    for step in (max(1, alloc[i] // 4), 1):
                        if alloc[i] < step:
                            continue
                        cand = list(alloc)
                        cand[i] -= step
                        cand[j] += step
                        ch = [segs[k] + cand[k] for k in range(n)]
                        s = fast_game_score(baselines, ch, n_completed)
                        cur = fast_game_score(
                            baselines, [segs[k] + alloc[k] for k in range(n)], n_completed
                        )
                        if s < cur - 1e-12:
                            alloc = cand
                            improved = True
                            break
        s = fast_game_score(baselines, [segs[k] + alloc[k] for k in range(n)], n_completed)
        if best_s is None or s < best_s:
            best_s, best_a = s, list(alloc)
    return float(best_s), list(best_a or [])


def bound_one_row(row: dict) -> Optional[dict]:
    """Bound a single won cell's gateway-charged score range.

    KNOWN from the row: offline segment lengths (level_up_actions), per-level baselines
    (per_level[i].human_actions), and the WHOLE-RUN reset count (n_resets).
    UNKNOWN from the row: how those resets distribute across segments. Hence a range.

    The BEST case is not zero resets in segment 1: the loop's very first move is
    structurally a RESET (arc_competition_agent.py:3806-3807 bootstrap, and
    StepwiseExplorer's own reset at :3798), verified empirically in Part B, so at least
    one charged reset precedes the first action. Best case therefore = 1 in segment 1 and
    every other reset in the free tail."""
    lua = row.get("level_up_actions")
    per_level = row.get("per_level")
    n_resets = row.get("n_resets")
    if not lua or not per_level or n_resets is None:
        return None
    baselines = [int(p.get("human_actions") or 0) for p in per_level]
    if not baselines or min(baselines) <= 0:
        return None  # dead baseline channel -> refuse to score rather than emit a fake number
    segs = _segments_offline([int(x) for x in lua])
    n_completed = len(segs)
    if n_completed > len(baselines):
        return None
    R = int(n_resets)
    if R < 1:
        return None  # structurally impossible (bootstrap reset); treat as unusable

    claimed = fast_game_score(baselines, segs, n_completed)  # what offline accounting implies
    best_charged = list(segs)
    best_charged[0] += 1
    best = fast_game_score(baselines, best_charged, n_completed)
    worst, worst_alloc = _worst_allocation(baselines, segs, n_completed, R)

    # Authoritative cross-check on the three reported points.
    a_claim = authoritative_game_score(baselines, segs, n_completed)
    a_best = authoritative_game_score(baselines, best_charged, n_completed)
    worst_charged = [
        segs[k] + (worst_alloc[k] if k < len(worst_alloc) else 0) for k in range(n_completed)
    ]
    a_worst = authoritative_game_score(baselines, worst_charged, n_completed)
    for fast_v, auth_v in ((claimed, a_claim), (best, a_best), (worst, a_worst)):
        assert abs(fast_v - auth_v) < 1e-9, (
            f"fast scorer disagrees with installed scorer: {fast_v} vs {auth_v}"
        )

    seg1_charged_min = segs[0] + 1
    seg1_charged_max = segs[0] + R
    return {
        "game": row.get("game"),
        "seed": row.get("seed"),
        "arm": row.get("arm"),
        "budget": row.get("budget"),
        "sweep_tag": row.get("sweep_tag"),
        "levels_completed": n_completed,
        "n_total_levels": len(baselines),
        "n_resets_whole_run": R,
        "offline_actions": row.get("actions"),
        "n_frames": row.get("n_frames"),
        "gateway_charged_total_is_n_frames": (
            int(row["n_frames"]) == int(row["actions"]) + R
            if row.get("n_frames") is not None and row.get("actions") is not None
            else None
        ),
        "segments_offline_actions": segs,
        "baselines": baselines[:n_completed],
        "segment1_offline_actions": segs[0],
        "segment1_gateway_charged_min": seg1_charged_min,
        "segment1_gateway_charged_max": seg1_charged_max,
        "segment1_inflation_factor_max": round(seg1_charged_max / segs[0], 4) if segs[0] else None,
        "score_offline_claimed": round(claimed, 6),
        "score_gateway_best_case": round(best, 6),
        "score_gateway_worst_case": round(worst, 6),
        "score_range_width": round(best - worst, 6),
        "score_worst_case_relative_loss": (
            round((claimed - worst) / claimed, 6) if claimed > 0 else None
        ),
        "worst_case_reset_allocation": worst_alloc,
        "clamp_binds_at_claimed": _clamp_binds(baselines, segs, n_completed),
    }


def _clamp_binds(baselines: list[int], charged: list[int], n_completed: int) -> bool:
    """Is the score sitting on the max_weights clamp (scorecard.py:204-206)? If so the
    raw efficiency term is NOT what sets the score, and a few extra charged actions cost
    NOTHING until a level's own term drops below 100. This is the mechanism that makes
    the damage bound much narrower than the raw action inflation suggests, and it must be
    reported rather than left implicit."""
    total_w = sum(i + 1 for i in range(len(baselines)))
    raw = 0.0
    max_w = 0
    for i in range(len(baselines)):
        w = i + 1
        if i < n_completed:
            a = charged[i] if i < len(charged) else 0
            s = min((float(baselines[i]) / a) ** 2 * 100.0, 115.0) if a > 0 else 0.0
        else:
            s = 0.0
        raw += s * w
        if s > 0:
            max_w += w
    if total_w == 0:
        return False
    return (max_w / total_w * 100.0) <= (raw / total_w) + 1e-12


# ---------------------------------------------------------------------------------------
# PART A' -- THE PER-LEVEL ATTRIBUTION READER. This is the task-4 deliverable, written as
# a PURE FUNCTION OF `frame_sequence`, which arc_leaderboard_eval.py ALREADY BUILDS
# (:328-335) with exactly the three fields needed: move.kind, action_count, and
# levels_completed. No agent change, no eval-loop change, no behaviour change -- the data
# already exists and is simply discarded at row-build time
# (arc_scored_path_lever_harness.py:842-846 aggregates it to a single whole-run count).
# ---------------------------------------------------------------------------------------


def attribute_resets_per_segment(frame_sequence: list[dict]) -> dict:
    """EXACT per-level reset attribution in ALL THREE UNITS, from frame_sequence alone.

    A "segment" is the span of loop iterations between two consecutive level-ups (the
    first segment runs from the start of the run). The gateway's per-level denominator is
    exactly this segment measured in gateway_charged units (scorecard.py:475-479 on
    reset-inclusive cumulative counts).

    Level JUMPS are handled the way arc_leaderboard_eval.py:321-324 handles them: a jump
    of k levels in one frame closes k segments, the first carrying the whole cost and the
    rest zero -- because that is what the gateway's actions_by_level list does
    (set_levels_completed appends one entry per observed change, so a jump appends ONE
    entry and the scorer's `level_idx < len(actions_by_level)` walk charges the remaining
    levels from the tail).
    """
    segments: list[dict] = []
    cur = {"offline_actions": 0, "resets": 0, "frames": 0}
    prev_level: Optional[int] = None
    for fr in frame_sequence:
        kind = (fr.get("move") or {}).get("kind")
        cur["frames"] += 1
        if kind == "RESET":
            cur["resets"] += 1
        elif kind is not None:
            cur["offline_actions"] += 1
        lvl = fr.get("levels_completed")
        if lvl is None:
            continue
        lvl = int(lvl)
        if prev_level is None:
            prev_level = lvl
            continue
        if lvl > prev_level:
            jump = lvl - prev_level
            for j in range(jump):
                if j == 0:
                    seg = dict(cur)
                else:
                    seg = {"offline_actions": 0, "resets": 0, "frames": 0}
                seg["gateway_charged"] = seg["offline_actions"] + seg["resets"]
                seg["level_completed"] = prev_level + j + 1
                segments.append(seg)
            cur = {"offline_actions": 0, "resets": 0, "frames": 0}
        prev_level = lvl
    tail = dict(cur)
    tail["gateway_charged"] = tail["offline_actions"] + tail["resets"]
    return {
        "segments": segments,
        "tail": tail,
        "n_segments": len(segments),
        "resets_in_completed_segments": sum(s["resets"] for s in segments),
        "resets_in_tail": tail["resets"],
    }


# ---------------------------------------------------------------------------------------
# PART B -- LIVE scored-path measurement of the reset COMPOSITION.
# ---------------------------------------------------------------------------------------


def run_live_cell(game: str, seed: int, budget: int) -> dict:
    """One SCORED-path cell (E3AgentPolicy -> StepwiseExplorer), offline arcade, LLM off.
    Captures the FULL navigation diagnostics (the row projection drops all but two keys:
    arc_leaderboard_eval.py:191-194) plus exact per-level reset attribution."""
    import random

    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"  # no LLM: this is a navigation measurement
    random.seed(seed)
    np.random.seed(seed % (2**32))
    t0 = time.time()
    policy = E3AgentPolicy(game, frontier_discipline_seed=seed)
    explorer = policy.explorer
    r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
    wall = time.time() - t0

    fs = r.get("frame_sequence") or []
    attrib = attribute_resets_per_segment(fs)
    n_resets = sum(1 for f in fs if (f.get("move") or {}).get("kind") == "RESET")
    n_frames = len(fs)
    actions = int(r["actions"])
    # THE IDENTITY, asserted rather than assumed.
    identity_holds = n_frames == actions + n_resets
    nav = explorer.navigation_diagnostics() if hasattr(explorer, "navigation_diagnostics") else {}

    first_move_kind = (fs[0].get("move") or {}).get("kind") if fs else None
    return {
        "game": game,
        "seed": seed,
        "budget": budget,
        "wall_s": round(wall, 3),
        "levels": int(r["levels"]),
        "offline_actions": actions,
        "n_frames": n_frames,
        "n_resets": n_resets,
        "gateway_charged_total": actions + n_resets,
        "identity_frames_eq_actions_plus_resets": bool(identity_holds),
        "first_move_is_reset": first_move_kind == "RESET",
        "navigation_diagnostics_full": nav,
        "per_level_attribution": attrib,
        "level_up_actions_offline": r.get("actions_to_first_levelup"),
        "reset_accounted_by_nav_fallbacks": int(nav.get("reset_replay_fallbacks") or 0),
        "resets_unexplained_by_nav": n_resets - int(nav.get("reset_replay_fallbacks") or 0),
    }


# ---------------------------------------------------------------------------------------
# PART D -- WHY does the cheap forward-walk fail? The IRREDUCIBLE / FIXABLE split.
#
# `_exact_shortest_path` (arc_competition_agent.py:3130-3150) BFSes the KNOWN FORWARD edges
# from cur to dst. A node's `path` is its action sequence FROM ROOT, so a target SHALLOWER
# than cur cannot be a descendant of cur: forward reachability is structurally impossible
# and no amount of extra edge-recording fixes it. `_partial_forward_path` (:3291) cannot
# rescue that case either -- it looks for an ancestor OF DST that is forward-reachable FROM
# CUR, and every ancestor of a shallower node is shallower still. Those resets are
# IRREDUCIBLE without a behaviour change. A miss on an equal-or-deeper target means the
# connecting edges merely have not been OBSERVED yet -- FIXABLE.
#
# POPULATION HAZARD (this probe was wrong once before it was right): `_shortest_path` is
# called from TWO places, and the ORDERING caller dominates the call count by ~100x.
#   :3125 `_frontier_navigation_cost_key`  -> allow_similarity=False   (ORDERING, not executed)
#   :3889 the nav decision in `next_move`  -> allow_similarity default True (EXECUTION)
# Classifying the raw call population answers a question nobody asked. The split below is
# keyed on `allow_similarity`, and `exec_calls` is cross-checked against the explorer's own
# `navigation_attempts` so a future refactor that changes either caller shows up as a
# mismatch instead of silently re-corrupting the population.
# ---------------------------------------------------------------------------------------


def probe_nav_failure_reasons(plan: list[tuple[str, int, int]]) -> dict:
    import random

    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    log: list[dict] = []
    original = StepwiseExplorer._shortest_path

    def wrapped(self, src, dst, *, allow_similarity=True):
        out = original(self, src, dst, allow_similarity=allow_similarity)
        cur_depth = len(self.graph.get(src, {}).get("path", [])) if src else 0
        dst_depth = len(self.graph.get(dst, {}).get("path", []))
        log.append(
            {
                "is_exec": bool(allow_similarity),
                "hit": out is not None,
                "cur_depth": cur_depth,
                "dst_depth": dst_depth,
                "shallower": dst_depth < cur_depth,
                "same": dst_depth == cur_depth,
                "deeper": dst_depth > cur_depth,
            }
        )
        return out

    StepwiseExplorer._shortest_path = wrapped  # type: ignore[method-assign]
    cells = []
    try:
        for game, seed, budget in plan:
            log.clear()
            random.seed(seed)
            np.random.seed(seed % (2**32))
            policy = E3AgentPolicy(game, frontier_discipline_seed=seed)
            explorer = policy.explorer
            t0 = time.time()
            r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
            nav = explorer.navigation_diagnostics()
            ex_calls = [x for x in log if x["is_exec"]]
            miss = [x for x in ex_calls if not x["hit"]]
            cells.append(
                {
                    "game": game,
                    "seed": seed,
                    "budget": budget,
                    "levels": int(r["levels"]),
                    "wall_s": round(time.time() - t0, 2),
                    "exec_calls": len(ex_calls),
                    "ordering_calls_excluded": len(log) - len(ex_calls),
                    "navigation_attempts_reported": int(nav["navigation_attempts"]),
                    # A GAME-OVER navigation decision NEVER CALLS the cheap path at all:
                    # :3889 increments _nav_attempts unconditionally, then :3890 guards the
                    # call with `if not over`. So the shortfall between the explorer's own
                    # attempt counter and the calls we intercepted IS the game-over-forced
                    # reset count -- a third reset class, found because these two numbers
                    # disagreed rather than because anything in the code said so.
                    "game_over_forced_resets_IRREDUCIBLE": int(nav["navigation_attempts"])
                    - len(ex_calls),
                    "reset_fallbacks_reported": int(nav["reset_replay_fallbacks"]),
                    "exec_misses": len(miss),
                    "miss_shallower_target_IRREDUCIBLE": sum(1 for x in miss if x["shallower"]),
                    "miss_same_depth_FIXABLE": sum(1 for x in miss if x["same"]),
                    "miss_deeper_target_FIXABLE": sum(1 for x in miss if x["deeper"]),
                    "fixable_share_of_reset_fallbacks": (
                        round(
                            sum(1 for x in miss if x["same"] or x["deeper"])
                            / int(nav["reset_replay_fallbacks"]),
                            4,
                        )
                        if int(nav["reset_replay_fallbacks"])
                        else None
                    ),
                    "irreducible_share_of_reset_fallbacks": (
                        round(
                            (
                                sum(1 for x in miss if x["shallower"])
                                + (int(nav["navigation_attempts"]) - len(ex_calls))
                            )
                            / int(nav["reset_replay_fallbacks"]),
                            4,
                        )
                        if int(nav["reset_replay_fallbacks"])
                        else None
                    ),
                    "forward_walk_hit_rate": round(float(nav["forward_walk_hit_rate"]), 4),
                    "partial_forward_walk_hits": int(nav["partial_forward_walk_hits"]),
                    "reset_replay_steps": int(nav["reset_replay_steps"]),
                    "forward_navigation_steps": int(nav["forward_navigation_steps"]),
                    "forward_edges_recorded": int(nav["forward_edges_recorded"]),
                }
            )
    finally:
        StepwiseExplorer._shortest_path = original  # type: ignore[method-assign]
    fix = [
        c["fixable_share_of_reset_fallbacks"]
        for c in cells
        if c["fixable_share_of_reset_fallbacks"] is not None
    ]
    irr = [
        c["irreducible_share_of_reset_fallbacks"]
        for c in cells
        if c["irreducible_share_of_reset_fallbacks"] is not None
    ]
    return {
        "substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "n_cells": len(cells),
        "fixable_share_of_reset_fallbacks": {
            "min": min(fix) if fix else None,
            "median": round(statistics.median(fix), 4) if fix else None,
            "max": max(fix) if fix else None,
        },
        "irreducible_share_of_reset_fallbacks": {
            "min": min(irr) if irr else None,
            "median": round(statistics.median(irr), 4) if irr else None,
            "max": max(irr) if irr else None,
        },
        "accounting_note": (
            "fixable + irreducible should sum to ~1.0 per cell. Where it does not, the residual "
            "is a fallback whose cheap-path call was intercepted but whose depth comparison was "
            "against a node with no recorded path (depth 0), and it is reported rather than "
            "silently folded into either bucket."
        ),
        "cells": cells,
    }


# ---------------------------------------------------------------------------------------
# PART C -- the OFFLINE TWIN's reset count.
# ---------------------------------------------------------------------------------------


def count_offline_twin_resets(game: str, depth_cap: int = 6, max_nodes: int = 400) -> dict:
    """OfflineSolver (arc_solver_kit.py:5181) navigates by REPLAY-FROM-RESET: `_replay`
    (:5287-5296) calls env.reset() and is invoked once per heap pop (:5311) AND once per
    sibling candidate to restore state (:5344). So its reset count scales with the number
    of candidate expansions, not with navigation failures. Counted here by wrapping
    env.reset, so the number is measured rather than inferred from reading the code."""
    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic import arc_solver_kit as kit

    ad = adapters.get_adapter(game)
    if ad is None:
        return {"game": game, "error": "no_adapter_registered"}
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
    except Exception as exc:
        return {"game": game, "error": f"{type(exc).__name__}:{exc}"}

    counters = {"resets": 0, "steps": 0}
    real_reset = env.reset
    real_step = env.step

    def counted_reset(*a, **k):
        counters["resets"] += 1
        return real_reset(*a, **k)

    def counted_step(*a, **k):
        counters["steps"] += 1
        return real_step(*a, **k)

    env.reset = counted_reset  # type: ignore[method-assign]
    env.step = counted_step  # type: ignore[method-assign]

    solver = kit.OfflineSolver(
        game,
        ad.action_labels,
        ad.apply,
        ad.state_key,
        warmup_label=ad.warmup_label,
        branch_mode=getattr(ad, "branch_mode", "replay"),
    )
    t0 = time.time()
    try:
        solver.max_nodes = max_nodes
        f = solver._replay(env, [])
        start_level = kit.frame_level(f)
        path, nodes = solver.solve_level(env, start_level, [], depth_cap)
    except Exception as exc:
        return {
            "game": game,
            "error": f"{type(exc).__name__}:{exc}",
            "resets_before_error": counters["resets"],
            "steps_before_error": counters["steps"],
        }
    mode = getattr(ad, "branch_mode", "replay")
    # DEAD-CHANNEL STAMP, not a number. `branch_mode='fresh_env'` mints a BRAND-NEW env per
    # candidate evaluation (_solve_level_fresh -> _fresh_env, arc_solver_kit.py:5414-5432), so a
    # wrapper installed on THIS env instance sees almost nothing. Reporting the resulting
    # env_resets=1 as "this game barely resets" would be exactly the clean-looking null this
    # project has shipped before. fresh_env is if anything STRICTLY WORSE than 'replay' in
    # gateway terms -- a fresh env plus a full prefix replay for every evaluation.
    instrumented = mode != "fresh_env"
    return {
        "game": game,
        "wall_s": round(time.time() - t0, 3),
        "branch_mode": mode,
        "reset_channel_instrumented": instrumented,
        "uninstrumented_reason": (
            None
            if instrumented
            else "branch_mode=fresh_env mints a new env per evaluation; the wrapper on this env "
            "instance cannot observe them. Counts below are a FLOOR, not a measurement."
        ),
        "nodes_expanded": int(nodes),
        "env_resets": counters["resets"],
        "env_steps": counters["steps"],
        "resets_per_node": round(counters["resets"] / nodes, 4) if nodes else None,
        "gateway_charged_if_this_were_live": counters["resets"] + counters["steps"],
        "charged_inflation_vs_steps": (
            round((counters["resets"] + counters["steps"]) / counters["steps"], 4)
            if counters["steps"]
            else None
        ),
        "solved": path is not None,
    }


# ---------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    analyser_t0 = time.time()
    do_live = "--no-live" not in argv
    out_path = os.path.join(
        REPO, "results", "outer_loop_arc_reset_charge_attribution_20260726.json"
    )

    # ---- PART A -------------------------------------------------------------------
    row_files = sorted(glob.glob(os.path.join(ROWS_DIR, "rows_*.json")))
    upstream = []
    measurement_wall_s = 0.0
    all_rows: list[dict] = []
    for p in row_files:
        with open(p) as fh:
            blob = fh.read()
        d = json.loads(blob)
        el = float(d.get("elapsed_s") or 0.0)
        measurement_wall_s += el
        upstream.append(
            {
                # `path` (not `file`) is REQUIRED: artifact_freshness_lint.py:111 reads
                # entry["path"] and reports an unreadable "." for anything else -- i.e. the
                # artifact silently degrades to UNVERIFIABLE, which the lint's own docstring
                # names as the failure mode ("an unknown is not a pass"). Getting this key
                # wrong is a dead instrumentation channel in the freshness guard itself.
                "path": p,
                "file": os.path.relpath(p, REPO),
                "n_rows": len(d.get("rows") or []),
                "elapsed_s": el,
                "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            }
        )
        all_rows.extend(d.get("rows") or [])

    bounds = []
    unusable = {"no_levels": 0, "no_baselines_or_fields": 0}
    for row in all_rows:
        if (row.get("levels") or 0) < 1:
            unusable["no_levels"] += 1
            continue
        b = bound_one_row(row)
        if b is None:
            unusable["no_baselines_or_fields"] += 1
            continue
        bounds.append(b)

    identity_checks = [b["gateway_charged_total_is_n_frames"] for b in bounds]
    widths = [b["score_range_width"] for b in bounds]
    rel_losses = [
        b["score_worst_case_relative_loss"]
        for b in bounds
        if b["score_worst_case_relative_loss"] is not None
    ]
    infl = [
        b["segment1_inflation_factor_max"] for b in bounds if b["segment1_inflation_factor_max"]
    ]
    clamped = sum(1 for b in bounds if b["clamp_binds_at_claimed"])
    zero_width = sum(1 for b in bounds if b["score_range_width"] <= 1e-9)

    part_a = {
        "substrate": "aggregation_from_upstream_artifacts",
        "measurement_wall_s": round(measurement_wall_s, 1),
        "measurement_wall_s_provenance": (
            "sum of each upstream row FILE's own elapsed_s -- NOT this analyser's runtime, "
            "and NOT the sum of per-cell wall_s (which undercounts ~25%)"
        ),
        "n_row_files": len(row_files),
        "n_rows_total": len(all_rows),
        "n_won_cells_bounded": len(bounds),
        "n_unusable": unusable,
        "identity_frames_eq_actions_plus_resets_all_rows": (
            all(x is True for x in identity_checks) if identity_checks else None
        ),
        "identity_n_checked": len(identity_checks),
        "segment1_max_inflation_factor": {
            "min": round(min(infl), 4) if infl else None,
            "median": round(statistics.median(infl), 4) if infl else None,
            "max": round(max(infl), 4) if infl else None,
        },
        "score_range_width": {
            "min": round(min(widths), 6) if widths else None,
            "median": round(statistics.median(widths), 6) if widths else None,
            "max": round(max(widths), 6) if widths else None,
            "n_zero_width": zero_width,
            "n_nonzero_width": len(widths) - zero_width,
        },
        "score_worst_case_relative_loss": {
            "min": round(min(rel_losses), 6) if rel_losses else None,
            "median": round(statistics.median(rel_losses), 6) if rel_losses else None,
            "max": round(max(rel_losses), 6) if rel_losses else None,
        },
        "n_cells_on_max_weights_clamp": clamped,
        "clamp_note": (
            "A cell sitting on the max_weights clamp (scorecard.py:204-206) has a score set by "
            "DEPTH, not by efficiency -- extra charged actions cost it NOTHING until a completed "
            "level's own (baseline/charged)^2*100 term falls below 100. This is why the bound is "
            "far narrower than the raw action inflation implies, and it is the single most "
            "important qualifier on this whole analysis."
        ),
        "per_cell": bounds,
    }

    # ---- PART B -------------------------------------------------------------------
    part_b: dict[str, Any] = {"skipped": True}
    if do_live:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        b_t0 = time.time()
        cells = []
        # Games chosen because the existing sweep shows them REACHING levels (so a
        # segment boundary exists to attribute against) across three budgets.
        plan = [
            ("vc33", 20260724, 400),
            ("vc33", 20260724, 2000),
            ("r11l", 20260724, 400),
            ("r11l", 20260725, 2000),
            ("tu93", 20260724, 400),
            ("lp85", 20260724, 400),
            ("sc25", 20260724, 400),
            ("dc22", 20260724, 400),
        ]
        for game, seed, budget in plan:
            try:
                cells.append(run_live_cell(game, seed, budget))
            except Exception as exc:
                cells.append(
                    {
                        "game": game,
                        "seed": seed,
                        "budget": budget,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
        ok = [c for c in cells if "error" not in c]
        part_b = {
            "substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "part_b_measurement_wall_s": round(time.time() - b_t0, 2),
            "n_cells": len(cells),
            "n_cells_ok": len(ok),
            "identity_holds_all_cells": all(c["identity_frames_eq_actions_plus_resets"] for c in ok)
            if ok
            else None,
            "first_move_is_reset_all_cells": all(c["first_move_is_reset"] for c in ok)
            if ok
            else None,
            "forward_walk_hit_rate": {
                c["game"] + "@" + str(c["budget"]): round(
                    float(c["navigation_diagnostics_full"].get("forward_walk_hit_rate") or 0.0), 4
                )
                for c in ok
            },
            "cells": cells,
        }

    # ---- PART D -------------------------------------------------------------------
    part_d: dict[str, Any] = {"skipped": True}
    if do_live:
        part_d = probe_nav_failure_reasons(
            [
                ("vc33", 20260724, 2000),
                ("r11l", 20260724, 2000),
                ("tu93", 20260724, 2000),
                ("dc22", 20260724, 2000),
                ("lp85", 20260724, 2000),
                ("sc25", 20260724, 2000),
            ]
        )

    # ---- PART C -------------------------------------------------------------------
    part_c: dict[str, Any] = {"skipped": True}
    if do_live:
        twin = []
        for game in ("vc33", "tu93", "lp85"):
            try:
                twin.append(count_offline_twin_resets(game))
            except Exception as exc:
                twin.append({"game": game, "error": f"{type(exc).__name__}:{exc}"})
        part_c = {
            "substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "cells": twin,
        }

    # ---- PART E -- the instrumentation SPEC (task 4). Specified, deliberately NOT applied:
    # a sibling lane holds an in-flight task with the same target file/function, and two lanes
    # editing run_game's accumulator loop concurrently is how a measurement harness acquires a
    # silent off-by-one. The reader in this file (attribute_resets_per_segment) needs NO source
    # change at all and is what Part B actually used, so nothing is blocked on the spec landing.
    instrumentation_spec = {
        "status": "CHANGE_1_LANDED_BY_SIBLING_LANE_CHANGES_2_AND_3_STILL_OPEN",
        "why_not_applied_by_this_lane": (
            "A sibling lane held 'Instrument run_game for EXACT per-level reset attribution' as "
            "in_progress against scripts/arc_leaderboard_eval.py:run_game -- the same accumulator "
            "loop. Concurrent edits to a counting loop are how an accounting harness picks up a "
            "silent off-by-one, so this lane specified and deferred. That lane then landed Change 1 "
            "independently and its implementation MATCHES this spec field-for-field "
            "(`resets`, `resets_before_levelups`, `level_up_charged`, plus an added "
            "`efficiency_gateway_charged`). Two lanes converging on the same minimal diff without "
            "coordination is corroboration that the diff is in fact minimal."
        ),
        "review_finding_against_the_landed_change_1": {
            "severity": "dead-channel hazard, reads as a MAXIMAL signal rather than as an error",
            "where": "scripts/arc_leaderboard_eval.py, the added gateway-efficiency block's `except Exception: eff_gateway = 0.0`, plus the same 0.0 default when `base` is empty",
            "problem": (
                "`efficiency_optimism_vs_gateway` is computed as `eff - eff_gateway`. If the scorer "
                "import raises, or the baseline channel is empty (`base == {}` -> `baseline_list == "
                "[]`), `eff_gateway` stays 0.0 and the optimism delta silently becomes the FULL "
                "value of `eff`. A broken measurement therefore does not read as null -- it reads as "
                "'offline accounting is 100% optimistic', the most alarming possible finding. This "
                "is the sibling of the failure this project already shipped once with "
                "`getattr(env,'baseline_actions')` against a field living on `env.info`."
            ),
            "suggested_fix": (
                "Default `eff_gateway` and `efficiency_optimism_vs_gateway` to None, not 0.0, and "
                "emit an explicit `efficiency_gateway_charged_error` string on the exception path. A "
                "consumer can then distinguish 'measured, no optimism' from 'never measured'."
            ),
        },
        "already_possible_with_zero_source_change": (
            "arc_leaderboard_eval.py:328-335 ALREADY records, per loop iteration, move.kind + "
            "action_count + levels_completed into frame_sequence. attribute_resets_per_segment() "
            "in this script derives EXACT per-level reset attribution in all three units from "
            "that alone -- no agent change, no eval-loop change, no behaviour change. It is what "
            "Part B measured with. The spec below only removes the dependency on frame_sequence "
            "being retained, which matters because the 1401 persisted rows do NOT retain it "
            "(arc_scored_path_lever_harness.py:842-847 aggregates it to two scalars and drops it)."
        ),
        "change_1_pure_addition_LANDED": {
            "landed_by": "sibling lane, same session, verified against this spec by reading the diff",
            "file": "scripts/arc_leaderboard_eval.py",
            "function": "run_game",
            "anchor_lines": "298-335 (accumulator init + loop), 400-412 (return dict)",
            "diff_intent": [
                "init: add `resets = 0`, `level_up_charged: list[int] = []`, `level_up_frames: list[int] = []` beside the existing `level_up_actions` at :301-303",
                'loop: in the EXISTING `if kind == "RESET":` branch at :308-309, add `resets += 1` -- do NOT touch `actions`, which must keep meaning offline_actions',
                "loop: in the EXISTING level-up block at :321-324, alongside `level_up_actions.append(actions)` add `level_up_charged.append(actions + resets)` and `level_up_frames.append(len(frames))`",
                "return: add `n_resets=resets`, `gateway_charged_total=actions + resets`, `level_up_charged=level_up_charged`, `level_up_frames=level_up_frames`",
            ],
            "why_pure_addition": (
                "`actions` is never read differently and never written differently; no branch "
                "predicate changes; no existing key changes meaning or value. Every added name is "
                "new. The agent is not touched at all."
            ),
            "built_in_consistency_check": (
                "The new `n_resets` counter and the harness's independent frame_sequence-derived "
                "reset sum must agree exactly. Assert it; a mismatch means one of the two "
                "accountings drifted, which is the failure this whole lane exists to catch."
            ),
        },
        "change_2_pure_addition": {
            "file": "scripts/arc_scored_path_lever_harness.py",
            "anchor_lines": "842-847 (the reset aggregation) and 855-867 (row.update)",
            "diff_intent": [
                "call attribute_resets_per_segment(r['frame_sequence']) and emit per-segment lists: segment_offline_actions, segment_resets, segment_frames, segment_gateway_charged, plus tail_* and resets_in_completed_segments / resets_in_tail",
                "KEEP n_resets and n_frames byte-identical -- never rewrite an existing field's meaning (never-prune)",
                "project the nav diagnostics the row currently drops entirely: nav_attempts, nav_reset_replay_fallbacks, nav_exact_hits, nav_partial_hits, nav_similarity_hits, nav_forward_walk_hit_rate",
            ],
            "naming_hazard_to_avoid": (
                "The row ALREADY has `n_nav_actions` / `nav_fraction` (:861-862) and they mean "
                "NON-CLICK actions, NOT navigation-replay actions. vc33 shows n_nav_actions=0 "
                "while carrying 115 resets. Any new navigation field must carry a distinct prefix "
                "or the two will be conflated by the next reader."
            ),
            "why_this_matters_most": (
                "Without it the reset composition is unattributable from persisted rows, which is "
                "precisely why Part B/D of this analysis had to re-run the agent live. run_game "
                "computes navigation_diagnostics at :394 and returns reset_replay_steps + "
                "forward_walk_hit_rate at :406-407; the lever harness row copies NONE of it."
            ),
        },
        "change_3_projection_only": {
            "file": "scripts/arc_leaderboard_eval.py",
            "anchor_lines": "181-194 (_navigation_diagnostics)",
            "diff_intent": [
                "_navigation_diagnostics currently narrows the explorer's 24-key diagnostics dict down to TWO keys, discarding reset_replay_fallbacks / navigation_attempts / exact vs partial vs similarity hit counts -- the exact fields needed to classify a reset. Return the full dict (it is already a flat dict of ints) and keep the two legacy keys for compatibility.",
            ],
        },
    }

    # ---- PART F -- WHERE IN THE BOUND DOES REALITY ACTUALLY LAND? -------------------
    # Part A can only bound, because a persisted row records the WHOLE-RUN reset count and
    # nothing about its distribution. Part B measured the distribution exactly on 8 cells.
    # Putting the two side by side is the whole argument for instrumenting: if reality sat
    # reliably at one end of the bound, the bound would be usable and instrumentation would
    # be optional. It does not.
    part_f: dict[str, Any] = {"skipped": True}
    if do_live and not part_b.get("skipped"):
        rows_f = []
        for c in part_b.get("cells", []):
            if "error" in c or c["levels"] < 1:
                continue
            attrib = c["per_level_attribution"]
            segs = attrib["segments"]
            if not segs:
                continue
            # Baselines for this game, read from the persisted corpus rather than re-derived,
            # so a dead baseline channel cannot silently zero this comparison.
            baselines = None
            for row in all_rows:
                if row.get("game") == c["game"] and row.get("per_level"):
                    cand = [int(p.get("human_actions") or 0) for p in row["per_level"]]
                    if cand and min(cand) > 0:
                        baselines = cand
                        break
            if baselines is None or len(segs) > len(baselines):
                continue
            offline_segs = [s["offline_actions"] for s in segs]
            charged_segs = [s["gateway_charged"] for s in segs]
            n_done = len(segs)
            claimed = authoritative_game_score(baselines, offline_segs, n_done)
            exact = authoritative_game_score(baselines, charged_segs, n_done)
            R = c["n_resets"]
            best_ch = list(offline_segs)
            best_ch[0] += 1
            best = fast_game_score(baselines, best_ch, n_done)
            worst, _ = _worst_allocation(baselines, offline_segs, n_done, R)
            span = best - worst
            rows_f.append(
                {
                    "game": c["game"],
                    "budget": c["budget"],
                    "levels": c["levels"],
                    "n_resets_whole_run": R,
                    "resets_in_completed_segments_MEASURED": attrib["resets_in_completed_segments"],
                    "resets_in_free_tail_MEASURED": attrib["resets_in_tail"],
                    "share_of_resets_that_cost_score": (
                        round(attrib["resets_in_completed_segments"] / R, 4) if R else None
                    ),
                    "segments_offline_actions": offline_segs,
                    "segments_gateway_charged_MEASURED": charged_segs,
                    "score_offline_claimed": round(claimed, 6),
                    "score_gateway_EXACT": round(exact, 6),
                    "score_bound_from_row_only": [round(worst, 6), round(best, 6)],
                    "exact_position_in_bound_0_worst_1_best": (
                        round((exact - worst) / span, 4) if span > 1e-12 else None
                    ),
                    "actual_relative_loss": (
                        round((claimed - exact) / claimed, 6) if claimed > 0 else None
                    ),
                }
            )
        positions = [
            r["exact_position_in_bound_0_worst_1_best"]
            for r in rows_f
            if r["exact_position_in_bound_0_worst_1_best"] is not None
        ]
        part_f = {
            "substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "n_cells": len(rows_f),
            "exact_position_in_bound": {
                "min": min(positions) if positions else None,
                "max": max(positions) if positions else None,
                "spread": round(max(positions) - min(positions), 4) if positions else None,
            },
            "verdict": (
                "THE BOUND IS NOT A USABLE SUBSTITUTE FOR ATTRIBUTION if the exact positions span "
                "a wide fraction of the bound: it would mean the row-only range cannot predict "
                "which end a given game sits at, so every recorded per-level efficiency number is "
                "individually unknown rather than uniformly-slightly-optimistic."
            ),
            "cells": rows_f,
        }

    artifact = {
        "experiment": "outer_loop_arc_reset_charge_attribution",
        "title": "Where resets come from on the scored path, and which are avoidable",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analyser_duration_s": round(time.time() - analyser_t0, 3),
        "analyser_duration_note": (
            "THIS IS THE ANALYSER CLOCK, NOT THE MEASUREMENT CLOCK. Part A's measurement "
            "clock is part_a.measurement_wall_s; Part B/C's is part_b.part_b_measurement_wall_s."
        ),
        "random_seed": 20260724,
        "inference_substrate": {
            "principle": (
                "Two substrates in one artifact, declared separately so neither borrows the "
                "other's credibility. Part A reads persisted rows (aggregation); Parts B/C/D drive "
                "the real agent against the offline arcade with no LLM. Collapsing them to one "
                "label is how a 2.54h measurement came to ship declaring duration_s 7.884."
            ),
            "part_a": "aggregation_from_upstream_artifacts",
            "parts_b_c_d": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "scope_and_power": {
            "principle": (
                "Hoist scope and power BESIDE the verdict, never below it: a single-game witness "
                "reported as a corpus property is a failure this project has already shipped."
            ),
            "part_a_scope": "485 won cells of 1401, 25 games, 4 budgets -- the whole persisted corpus",
            "parts_b_d_scope": "8 (Part B) and 6 (Part D) cells, 6 distinct games, ONE seed each",
            "parts_b_d_power": (
                "SINGLE SEED per cell. The reset-composition IDENTITY (n_resets == "
                "reset_replay_fallbacks + 1) held in 8/8 cells and is structural, not statistical, "
                "so it is safe to generalise. The reducible/irreducible SHARES are per-game "
                "quantities measured once each and MUST NOT be read as corpus point estimates -- "
                "they range 0%-23% fixable across only 6 games."
            ),
            "unmeasured_and_why_it_matters": (
                "ALL 1401 persisted rows carry llm_enabled=False and induction_planned=0, so the "
                "plan-execution reset at arc_competition_agent.py:5314 has NEVER fired anywhere in "
                "this corpus, and Parts B/D (LLM off) could not exercise it either. The SUBMITTED "
                "configuration runs induction ON. Its reset contribution is therefore UNMEASURED, "
                "and any statement of the form 'resets are fully accounted for' is scoped to the "
                "LLM-off navigation regime ONLY."
            ),
        },
        "part_a_bound_over_existing_rows": part_a,
        "part_b_live_reset_composition": part_b,
        "part_c_offline_twin_reset_count": part_c,
        "part_d_why_the_cheap_path_fails": part_d,
        "part_e_instrumentation_spec": instrumentation_spec,
        "part_f_exact_vs_bound": part_f,
    }

    # PROVENANCE, so scripts/artifact_freshness_lint.py can answer "was this built from the
    # code now on disk". An artifact with no provenance block is reported UNKNOWN rather than
    # STALE, which is precisely the false-clean this lane is meant not to produce.
    def _fingerprint(rel: str) -> dict:
        p = os.path.join(REPO, rel)
        try:
            with open(p, "rb") as fh:
                data = fh.read()
            st = os.stat(p)
            return {
                "path": p,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
            }
        except Exception as exc:
            return {"path": p, "error": f"{type(exc).__name__}:{exc}"}

    import subprocess

    try:
        head = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        head = ""
    artifact["provenance"] = {
        "git_head": head,
        "code": [
            _fingerprint("scripts/analyze_arc_reset_charge_attribution.py"),
            _fingerprint("scripts/arc_leaderboard_eval.py"),
            _fingerprint("scripts/arc_scored_path_lever_harness.py"),
            _fingerprint("python/carnot/agentic/arc_competition_agent.py"),
            _fingerprint("python/carnot/agentic/arc_solver_kit.py"),
        ],
        "rows_sources": {"rows": upstream},
        "rebuild_command": (
            "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python "
            "scripts/analyze_arc_reset_charge_attribution.py"
        ),
        "concurrent_edit_disclosure": (
            "scripts/arc_leaderboard_eval.py was edited by a CONCURRENT lane during this session "
            "(a pure addition of reset counters). Parts B/C/D were re-run end-to-end AFTER that "
            "edit so every live number in this artifact comes from one single version of the "
            "harness. The reads Parts B/D depend on -- `actions`, `frame_sequence`, "
            "`navigation_diagnostics` -- are untouched by that edit in any case."
        ),
    }

    # Copy forward any `provenance.freshness_acknowledgements` this output path already
    # carries on disk, so a rebuild never silently erases that append-only audit log (see
    # analyze_scored_path_lever_ab.py:preserve_freshness_acknowledgements for the full
    # rationale -- every OTHER provenance field above is correctly overwritten fresh).
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from analyze_scored_path_lever_ab import preserve_freshness_acknowledgements

    preserve_freshness_acknowledgements(artifact, Path(out_path))

    artifact["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(artifact, sort_keys=True, default=str).encode()).hexdigest()
    )
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=1, default=str)
    print(f"wrote {out_path}")
    print(json.dumps({k: v for k, v in part_a.items() if k != "per_cell"}, indent=1))
    if do_live:
        print(json.dumps({k: v for k, v in part_b.items() if k != "cells"}, indent=1))
        print(json.dumps({k: v for k, v in part_d.items() if k != "cells"}, indent=1))
        for c in part_d.get("cells", []):
            print(
                "NAVFAIL",
                c["game"],
                f"fallbacks={c['reset_fallbacks_reported']}",
                f"gameover={c['game_over_forced_resets_IRREDUCIBLE']}",
                f"shallower={c['miss_shallower_target_IRREDUCIBLE']}",
                f"fixable={c['miss_same_depth_FIXABLE'] + c['miss_deeper_target_FIXABLE']}",
                f"fixable_share={c['fixable_share_of_reset_fallbacks']}",
            )
        for c in part_c.get("cells", []):
            print("TWIN", json.dumps(c))
        if not part_f.get("skipped"):
            print(json.dumps({k: v for k, v in part_f.items() if k != "cells"}, indent=1))
            for c in part_f["cells"]:
                print(
                    "EXACTvBOUND",
                    f"{c['game']}@{c['budget']}",
                    f"R={c['n_resets_whole_run']}",
                    f"costing={c['resets_in_completed_segments_MEASURED']}",
                    f"free_tail={c['resets_in_free_tail_MEASURED']}",
                    f"claimed={c['score_offline_claimed']}",
                    f"exact={c['score_gateway_EXACT']}",
                    f"bound={c['score_bound_from_row_only']}",
                    f"pos={c['exact_position_in_bound_0_worst_1_best']}",
                    f"actual_loss={c['actual_relative_loss']}",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
