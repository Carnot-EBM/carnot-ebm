#!/usr/bin/env python3
"""Turn the per-cell records into an EXPOSURE rate with its roster distribution beside it.

THREE THINGS THIS REFUSES TO DO, each because it has already produced a wrong answer here:
  * report a pooled mean without the per-game spread (a 73.8% headline was one game's property);
  * treat a game that made ZERO induce calls as a game with 0% exposure (missing is not zero --
    those two facts have different fixes, so they are counted separately);
  * call an identically-zero seed-to-seed difference a noise floor (the agent's RNGs are
    `random.Random(<constructor default>)`; a global `random.seed` cannot reach them, so an A/A
    over argv seeds is expected to be exactly zero BY CONSTRUCTION and proves nothing).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _is_gap(r: dict) -> bool:
    """Derive the gap flag rather than trust it.

    The driver's cached-cell fast path returns the record as written and does not stamp
    `coverage_gap`, so a CACHED error cell would arrive with the field absent. Absent is not
    False. A cell counts as a gap if it says so OR if it carries an error OR if it is missing
    the two fields this whole analysis is a function of."""
    if r.get("coverage_gap"):
        return True
    if r.get("error"):
        return True
    return r.get("n_induce_calls") is None or r.get("n_with_win_transition") is None


def summarise(rows: list[dict], label: str) -> dict:
    gaps = [r for r in rows if _is_gap(r)]
    ok = [r for r in rows if not _is_gap(r)]
    n_calls = sum(int(r.get("n_induce_calls") or 0) for r in ok)
    n_win = sum(int(r.get("n_with_win_transition") or 0) for r in ok)

    per_game_calls: dict[str, int] = defaultdict(int)
    per_game_win: dict[str, int] = defaultdict(int)
    per_game_levels: dict[str, list[int]] = defaultdict(list)
    for r in ok:
        per_game_calls[r["game"]] += int(r.get("n_induce_calls") or 0)
        per_game_win[r["game"]] += int(r.get("n_with_win_transition") or 0)
        per_game_levels[r["game"]].append(int(r.get("levels") or 0))

    games_with_calls = sorted(g for g in per_game_calls if per_game_calls[g] > 0)
    games_without_calls = sorted(g for g in per_game_calls if per_game_calls[g] == 0)
    per_game_rate = {g: per_game_win[g] / per_game_calls[g] for g in games_with_calls}
    rates = sorted(per_game_rate.values())

    # CONSISTENCY CHECK, not an assumption: `_begin_level_goal_episode` is the sole writer of
    # both `_win_transition` and `level_induction_events`, so `win_transition_available` must
    # equal `n_level_induction_events_before > 0` on every call unless the empty-transitions
    # guard fired. A mismatch is reported, never smoothed over.
    mismatches = []
    for r in ok:
        for c in r.get("induce_calls") or []:
            if bool(c["win_transition_available"]) != (
                int(c.get("n_level_induction_events_before") or 0) > 0
            ):
                mismatches.append({"game": r["game"], "budget": r["budget"], **c})

    # THE SECOND GATE. `induce_calls[i]` and `induction_attempts[i]` are the same attempt --
    # `_induce_and_plan` appends its attempt row on entry, and the instrumentation records on
    # entry too -- so the skip reason can be joined to the exposure state by index. Under the
    # NoOp proposer every attempt necessarily skips `proposer_failed`; that is a PROPERTY OF
    # THIS HARNESS and is why this harness can only speak to PROMPT-level exposure, never to
    # whether an exposed prompt would have changed the installed plan.
    skip_by_exposure: dict[str, dict[str, int]] = {
        "win_available": defaultdict(int),
        "no_win": defaultdict(int),
    }
    for r in ok:
        skips = r.get("induction_attempts_skipped") or []
        for i, c in enumerate(r.get("induce_calls") or []):
            key = "win_available" if c["win_transition_available"] else "no_win"
            skip_by_exposure[key][str(skips[i]) if i < len(skips) else "<unrecorded>"] += 1

    # THE ROUTING PARTITION -- the measurement that turns "how often is the win transition
    # AVAILABLE" into "how often does the CHANGED CALL SITE actually receive it".
    #
    # `_induce_and_plan` has TWO exits. When `reason == "level_up_reinduction" OR
    # next_level_episode`, it calls `execute_bounded_llm_reinduction` and RETURNS at
    # arc_competition_agent.py:6245 -- and that module's `_call_induce`
    # (arc_llm_reinduction.py:203-220) does NOT forward `win_transition`. Only the fall-through
    # STALL / first-contact path reaches the changed call at :6429-:6433.
    #
    # The skip string is the observable that separates them, because the two paths write
    # DIFFERENT strings: `proposer_failed` is written by arc_llm_reinduction.py:1507, and
    # `proposer_failed_or_missing_root` by arc_competition_agent.py:6436 -- i.e. by the line
    # immediately after the changed call. This is corroborated INDEPENDENTLY by verify_kwarg.py,
    # which instruments the receiving end and reads the caller off the stack.
    routing = {
        "reached_changed_call_site": defaultdict(int),
        "routed_to_reinduction": defaultdict(int),
    }
    n_win_reached_site = 0
    n_reached_site = 0
    for r in ok:
        skips = r.get("induction_attempts_skipped") or []
        for i, c in enumerate(r.get("induce_calls") or []):
            s = str(skips[i]) if i < len(skips) else "<unrecorded>"
            reached = s == "proposer_failed_or_missing_root"
            key = "reached_changed_call_site" if reached else "routed_to_reinduction"
            routing[key]["win_available" if c["win_transition_available"] else "no_win"] += 1
            n_reached_site += int(reached)
            n_win_reached_site += int(reached and c["win_transition_available"])

    # Where a win transition IS available, which induction reason brought us there.
    reason_win: dict[str, int] = defaultdict(int)
    reason_all: dict[str, int] = defaultdict(int)
    for r in ok:
        for c in r.get("induce_calls") or []:
            reason_all[str(c.get("pending_induction_reason"))] += 1
            if c["win_transition_available"]:
                reason_win[str(c.get("pending_induction_reason"))] += 1

    return {
        "label": label,
        "n_cells": len(rows),
        "n_cells_ok": len(ok),
        "n_coverage_gaps": len(gaps),
        "coverage_gaps": [
            {"game": g["game"], "budget": g.get("budget"), "reason": str(g.get("gap_reason"))[:200]}
            for g in gaps
        ],
        "n_induce_calls_total": n_calls,
        "n_with_win_available": n_win,
        "exposure_rate_pooled": round(n_win / n_calls, 6) if n_calls else None,
        "exposure_rate_pooled_note": (
            "cells are not independent -- several seeds/budgets of the same game share a "
            "trajectory, so the pooled rate is reported WITH the per-game spread, never alone"
        ),
        "n_games_with_at_least_one_induce_call": len(games_with_calls),
        "n_games_with_zero_induce_calls": len(games_without_calls),
        "games_with_zero_induce_calls": games_without_calls,
        "per_game_exposure_rate": {g: round(per_game_rate[g], 6) for g in games_with_calls},
        "per_game_induce_calls": {g: per_game_calls[g] for g in games_with_calls},
        "per_game_win_available": {g: per_game_win[g] for g in games_with_calls},
        "per_game_levels_banked": {g: sorted(per_game_levels[g]) for g in sorted(per_game_levels)},
        "roster_max_game": (
            max(per_game_rate, key=lambda g: (per_game_rate[g], g)) if per_game_rate else None
        ),
        "roster_max_rate": round(max(rates), 6) if rates else None,
        "roster_median_rate": round(statistics.median(rates), 6) if rates else None,
        "roster_median_game": (
            sorted(games_with_calls, key=lambda g: (per_game_rate[g], g))[len(rates) // 2]
            if rates
            else None
        ),
        "roster_n_games_at_zero_exposure": sum(1 for v in rates if v == 0.0),
        "roster_n_games_at_full_exposure": sum(1 for v in rates if v == 1.0),
        "cluster_mean_of_per_game_rates": round(statistics.fmean(rates), 6) if rates else None,
        "induction_reason_census_all_calls": dict(sorted(reason_all.items())),
        "induction_reason_census_win_available_calls": dict(sorted(reason_win.items())),
        "routing_partition": {k: dict(sorted(v.items())) for k, v in routing.items()},
        "n_calls_that_reached_the_changed_call_site": n_reached_site,
        "n_calls_with_win_available_that_reached_the_changed_call_site": n_win_reached_site,
        "effective_exposure_rate": (
            round(n_win_reached_site / n_calls, 6) if (n_calls := n_calls) else None
        ),
        "effective_exposure_note": (
            "n_with_win_available / n_induce_calls_total is the PROMPT-AVAILABILITY rate. "
            "effective_exposure_rate is the fraction of live induce calls at which the changed "
            "call site at arc_competition_agent.py:6429-6433 actually RECEIVED a non-None "
            "win transition. They differ because the routing branch above sends every "
            "level-banked induction down a path that drops the argument."
        ),
        "skip_reason_by_exposure_state": {
            k: dict(sorted(v.items())) for k, v in skip_by_exposure.items()
        },
        "n_attempts_that_installed_a_plan": sum(
            int(r.get("induction_attempts_planned") or 0) for r in ok
        ),
        "structural_consistency_mismatches": mismatches,
        "n_structural_consistency_mismatches": len(mismatches),
    }


def main() -> int:
    rows = json.loads((HERE / "rows.json").read_text())
    out: dict = {"overall": summarise(rows, "all budgets, all seeds")}
    for b in sorted({r["budget"] for r in rows}):
        out[f"budget_{b}"] = summarise([r for r in rows if r["budget"] == b], f"budget={b}")

    # SEED A/A. Expected to be EXACTLY zero because argv seeds cannot reach the agent's RNGs.
    # Recorded so that zero is read as "the axis was not varied", not "the noise floor is zero".
    aa: dict = {}
    probe: dict = defaultdict(dict)
    for r in rows:
        if _is_gap(r):
            continue
        key = (r["game"], r["budget"], r.get("frontier_discipline_seed"))
        probe[key][r.get("seed")] = (
            r.get("levels"),
            r.get("actions"),
            r.get("n_induce_calls"),
            r.get("n_with_win_transition"),
        )
    aa = {
        f"{g}__b{b}__fd{fd}": {
            "per_argv_seed": {str(k): v for k, v in sorted(d.items())},
            "varied": len(set(d.values())) > 1,
        }
        for (g, b, fd), d in sorted(probe.items())
        if len(d) > 1
    }
    # THE REACHABLE REPLICATE AXIS.
    fd_rep: dict = {}
    by_budget_game: dict = defaultdict(dict)
    for r in rows:
        if _is_gap(r):
            continue
        by_budget_game[(r["budget"], r["game"])][r.get("frontier_discipline_seed")] = (
            r.get("levels"),
            r.get("n_induce_calls"),
            r.get("n_with_win_transition"),
        )
    for (b, g), d in sorted(by_budget_game.items()):
        if len(d) < 2:
            continue
        fd_rep[f"{g}__b{b}"] = {
            "per_fd_seed": {str(k): v for k, v in sorted(d.items())},
            "varied": len(set(d.values())) > 1,
        }
    out["fd_seed_replicates"] = {
        "n_game_budget_cells_with_multiple_fd_seeds": len(fd_rep),
        "n_that_varied": sum(1 for v in fd_rep.values() if v["varied"]),
        "detail": fd_rep,
        "interpretation": (
            "`frontier_discipline_seed` is the seed the agent actually consumes "
            "(arc_competition_agent.py:1310). Variation here is a REAL trajectory noise floor; "
            "the argv-seed A/A below is not."
        ),
    }
    out["argv_seed_aa"] = {
        "result": aa,
        "interpretation": (
            "A zero here is NOT a noise floor. Every RNG the live explorer uses is constructed as "
            "`random.Random(<constructor default>)` -- arc_competition_agent.py:1310 "
            "(frontier_discipline_seed=20260724) and :1397 (Random(20260621)) -- so a global "
            "`random.seed()`/`np.random.seed()` in the worker cannot reach them. An identical "
            "outcome across argv seeds is the EXPECTED consequence of that, and it means the seed "
            "axis was not exercised at all. The reachable knob is the "
            "`frontier_discipline_seed` constructor argument; see fd_seed_replicates."
        ),
    }
    (HERE / "analysis.json").write_text(json.dumps(out, indent=2, default=str))
    o = out["overall"]
    print(
        json.dumps(
            {
                k: o[k]
                for k in (
                    "n_cells_ok",
                    "n_coverage_gaps",
                    "n_induce_calls_total",
                    "n_with_win_available",
                    "exposure_rate_pooled",
                    "n_games_with_zero_induce_calls",
                    "roster_max_game",
                    "roster_max_rate",
                    "roster_median_game",
                    "roster_median_rate",
                    "roster_n_games_at_zero_exposure",
                    "n_structural_consistency_mismatches",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
