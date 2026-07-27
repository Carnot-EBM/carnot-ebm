#!/usr/bin/env python3
"""GATEWAY-ACCURATE re-scorer for recorded ARC-AGI-3 offline sweep rows.

WHY THIS EXISTS (the defect being measured)
-------------------------------------------
Our offline harness (`scripts/arc_leaderboard_eval.py:run_game`) charges an
action ONLY in the non-RESET branch::

    if kind == "RESET":
        latest = env.reset()          # <-- no `actions += 1`
    else:
        latest = env.step(...); actions += 1

The LIVE gateway charges a RESET as an action. From the INSTALLED scorer source
(`arc_agi/scorecard.py`)::

    def inc_reset_count(self, guid):        # :701-704
        self.resets[i] += 1
        self.actions[i] += 1                # <-- RESET IS CHARGED AN ACTION

reached from `Scorecard.update_scorecard` (:839-843) on every RESET frame.

Because the per-level charge is a DIFFERENCE of cumulative charged actions::

    level_actions = actions_at_level - prev_actions      # :479

and the per-level score is `min((baseline/level_actions)**2 * 100, 115)`, every
per-level efficiency number this project holds is OPTIMISTIC in the SQUARED
term by the number of resets charged BEFORE that level-up. RESET-replay is not a
rare path: `arc_competition_agent.py:969-970` documents it as "the ONLY
navigation" for reaching untested actions.

THREE UNITS -- NEVER CONFLATE THEM
----------------------------------
  (1) OFFLINE ACTIONS   our harness `actions`; EXCLUDES resets. The unit
                        `level_up_actions` and every recorded `efficiency` is in.
  (2) FRAMES            loop iterations; INCLUDES resets. Verified identity over
                        all 1401 recorded early-stop rows: n_frames == actions
                        + n_resets, 0 exceptions.
  (3) GATEWAY-CHARGED   non-RESET moves PLUS resets. The ONLY unit the
                        competition score is a function of. Equals (2).

WHAT THIS MODULE DOES
---------------------
1. `gateway_score_via_calculator` -- drives the INSTALLED
   `arc_agi.scorecard.EnvironmentScoreCalculator` exactly as
   `_calculate_score` (:474-491) drives it. Never a paraphrase of the formula.
2. `gateway_score_via_full_chain` -- builds a real `Scorecard`/`Card` by calling
   the REAL mutators (`new_play` / `take_action` / `reset` /
   `set_levels_completed`) frame by frame, then scores it through
   `EnvironmentScorecard.from_scorecard`. This is the whole installed chain
   including `inc_reset_count`'s action charge. Used to CROSS-CHECK path 1
   cell-by-cell so a divergence cannot hide.
3. `bound_gateway_score` -- per-cell BOUNDS, because PER-LEVEL RESET
   ATTRIBUTION WAS NEVER RECORDED (only whole-run `n_resets`):
     * BEST case  == the offline score. All resets land AFTER the last level-up,
       where they cost EXACTLY NOTHING (an incomplete level scores 0.0 whatever
       it is charged -- scorecard.py:178-183).
     * WORST case == an exact allocation of all `n_resets` across the COMPLETED
       levels that minimises the index-weighted score, by dynamic programming
       (exact; greedy-on-marginals is also computed as a cross-check because the
       115 cap makes the marginal sequence non-monotone in principle).
   The bound is accounting-only: it does not ask whether a policy could have
   produced that reset placement, so WORST is a true lower bound on the score.

HONESTY CONTRACT
----------------
This module NEVER rewrites a historical artifact. It emits new numbers that
cite the originals. It never submits anything anywhere.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# --------------------------------------------------------------------------
# Path 1: the installed EnvironmentScoreCalculator, driven as _calculate_score
#         drives it.
# --------------------------------------------------------------------------


def gateway_score_via_calculator(
    baselines: Sequence[int],
    level_charged: Sequence[int],
    tail_charged: int,
    *,
    game_won: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    """Per-game score from per-level CHARGED action counts.

    Args:
        baselines: the env's `baseline_actions`, one per level (human reference).
        level_charged: charged actions consumed by each COMPLETED level, in
            order. Length == number of completed levels.
        tail_charged: charged actions spent after the last level-up (the
            in-progress level). Scored 0.0 regardless -- passed through only so
            the calculator sees the same `actions_taken` the real scorer would.
        game_won: whether the whole game reached GameState.WIN. When true the
            installed scorer treats EVERY level as completed
            (`level_completed = level_idx < len(actions_by_level) or completed`).

    Returns:
        (score, per_level_rows). Mirrors arc_agi/scorecard.py:474-491.
    """
    from arc_agi.scorecard import EnvironmentScoreCalculator

    calc = EnvironmentScoreCalculator()
    rows: list[dict[str, Any]] = []
    n_done = len(level_charged)
    for li, baseline in enumerate(baselines):
        if li < n_done:
            charged = int(level_charged[li])
            done = True
        else:
            # Only the FIRST unattempted level absorbs the tail, matching the
            # scorer's `card.actions[idx] - prev_actions` then prev := actions.
            charged = int(tail_charged) if li == n_done else 0
            done = bool(game_won)
        calc.add_level(
            level_index=li + 1,
            completed=done,
            actions_taken=charged,
            baseline_actions=int(baseline),
        )
        rows.append(
            {
                "level": li,
                "charged_actions": charged,
                "human_actions": int(baseline),
                "completed": done,
            }
        )
    return float(calc.to_score(include_levels=False).score), rows


# --------------------------------------------------------------------------
# Path 2: the FULL installed chain -- real Card mutators, real from_scorecard.
# --------------------------------------------------------------------------


def gateway_score_via_full_chain(
    game_id: str,
    baselines: Sequence[int],
    frame_kinds: Sequence[str],
    level_after_frame: Sequence[int],
) -> float:
    """Score by replaying a frame sequence through the REAL Scorecard chain.

    This exercises `inc_reset_count` (which charges an action for a RESET),
    `inc_action_count`, and `set_levels_completed` -- i.e. the gateway's own
    bookkeeping -- then scores via `EnvironmentScorecard.from_scorecard`.

    Args:
        frame_kinds: per loop iteration, either "RESET" or "ACTION".
        level_after_frame: levels_completed observed after each frame.
    """
    from arc_agi.models import EnvironmentInfo
    from arc_agi.scorecard import EnvironmentScorecard, Scorecard

    guid = "rescore-guid"
    sc = Scorecard(card_id="rescore")
    sc.new_play(game_id, guid)
    for kind, lvl in zip(frame_kinds, level_after_frame):
        if kind == "RESET":
            sc.reset(game_id, guid)
        else:
            sc.take_action(game_id, guid)
        sc.set_levels_completed(game_id, guid, int(lvl))
    info = EnvironmentInfo(game_id=game_id, baseline_actions=[int(b) for b in baselines])
    env_sc = EnvironmentScorecard.from_scorecard(sc, [info])
    for env in env_sc.environments:
        for run in env.runs:
            return float(run.score)
    return 0.0


# --------------------------------------------------------------------------
# Bounds: exact worst-case reset allocation over the completed levels.
# --------------------------------------------------------------------------


def _level_score(baseline: int, charged: int) -> float:
    """One level's score, mirroring EnvironmentScoreCalculator.add_level."""
    if charged <= 0:
        return 0.0
    return min(((float(baseline) / float(charged)) ** 2) * 100.0, 115.0)


def worst_case_allocation(
    baselines: Sequence[int],
    level_offline: Sequence[int],
    n_resets: int,
    tail_charged: int,
    *,
    game_won: bool = False,
) -> tuple[float, list[int]]:
    """Exact DP: place `n_resets` across completed levels to MINIMISE the score.

    The per-game aggregation is an index-weighted mean with a further
    `min(., max_weights/total_weights*100)` clamp, and both the weights and the
    clamp depend only on WHICH levels scored > 0 -- not on how much. Adding
    resets never turns a completed level's score to 0 (a completed level with
    charged > 0 always scores > 0), so the weight set is invariant under the
    allocation and minimising the weighted SUM of completed-level scores is
    equivalent to minimising the final score. That is what makes a separable DP
    valid here.

    Returns (worst_score, allocation).
    """
    n_done = len(level_offline)
    if n_done == 0 or n_resets <= 0:
        score, _ = gateway_score_via_calculator(
            baselines, level_offline, tail_charged, game_won=game_won
        )
        return score, [0] * n_done

    weights = [li + 1 for li in range(n_done)]  # level_index used as the weight
    # DP over levels x resets-used, minimising the weighted sum.
    INF = float("inf")
    # best[r] = min weighted-sum using the first i levels and exactly r resets
    best = [0.0] + [INF] * n_resets
    choice: list[list[int]] = []
    for i in range(n_done):
        base_i, off_i, w_i = int(baselines[i]), int(level_offline[i]), weights[i]
        cost = [w_i * _level_score(base_i, off_i + r) for r in range(n_resets + 1)]
        nxt = [INF] * (n_resets + 1)
        pick = [0] * (n_resets + 1)
        for r in range(n_resets + 1):
            if best[r] == INF:
                continue
            for extra in range(n_resets - r + 1):
                v = best[r] + cost[extra]
                if v < nxt[r + extra]:
                    nxt[r + extra] = v
                    pick[r + extra] = extra
        best, ch = nxt, pick
        choice.append(ch)

    # Recover the allocation that used exactly n_resets (all resets before the
    # last level-up is the pessimal placement; using fewer can only score >=).
    alloc = [0] * n_done
    r = n_resets
    for i in range(n_done - 1, -1, -1):
        extra = choice[i][r]
        alloc[i] = extra
        r -= extra
    charged = [int(level_offline[i]) + alloc[i] for i in range(n_done)]
    score, _ = gateway_score_via_calculator(baselines, charged, tail_charged, game_won=game_won)
    return score, alloc


def greedy_worst_case(
    baselines: Sequence[int],
    level_offline: Sequence[int],
    n_resets: int,
    tail_charged: int,
    *,
    game_won: bool = False,
) -> tuple[float, list[int]]:
    """Greedy-on-marginals cross-check of `worst_case_allocation`.

    Kept as an independent implementation so a DP bug cannot pass silently. The
    115 cap makes the marginal sequence non-monotone in principle (a flat
    capped region has marginal 0), so greedy is NOT guaranteed optimal -- any
    disagreement with the DP is reported, not averaged away.
    """
    n_done = len(level_offline)
    alloc = [0] * n_done
    weights = [li + 1 for li in range(n_done)]
    for _ in range(n_resets):
        best_gain, best_i = 0.0, -1
        for i in range(n_done):
            b, a = int(baselines[i]), int(level_offline[i]) + alloc[i]
            gain = weights[i] * (_level_score(b, a) - _level_score(b, a + 1))
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0:
            break
        alloc[best_i] += 1
    charged = [int(level_offline[i]) + alloc[i] for i in range(n_done)]
    score, _ = gateway_score_via_calculator(baselines, charged, tail_charged, game_won=game_won)
    return score, alloc


# --------------------------------------------------------------------------
# Row-level API
# --------------------------------------------------------------------------


@dataclass
class RescoreResult:
    game: str = ""
    arm: str = ""
    seed: int | None = None
    budget: int | None = None
    n_levels_completed: int = 0
    n_resets: int = 0
    offline_actions: int = 0
    frames: int | None = None
    recorded_efficiency: float | None = None
    offline_score: float = 0.0
    worst_score: float = 0.0
    worst_alloc: list[int] = field(default_factory=list)
    greedy_score: float = 0.0
    greedy_agrees: bool = True
    delta_worst: float = 0.0  # offline - worst  (>=0 => offline is optimistic)
    rel_delta_worst: float | None = None
    exact_score: float | None = None  # only when per-level attribution exists
    exact_resets_before_levelups: list[int] | None = None
    rescorable: bool = True
    reason: str = ""
    baselines: list[int] = field(default_factory=list)
    level_offline: list[int] = field(default_factory=list)


def _baselines_from_row(row: dict) -> list[int]:
    """Per-level human baselines as recorded in the row's `per_level` block.

    The recorded rows carry `human_actions` per level, produced upstream by
    `arc_leaderboard_eval._baseline_actions`, which reads
    `getattr(getattr(env, "info", env), attr)` -- i.e. through `env.info`. A
    prior agent read `getattr(env, "baseline_actions")` directly, a DEAD
    CHANNEL that summed to 0.0 and made both charge models agree, reading as a
    clean null. We assert non-zero here so that failure mode cannot recur
    silently.
    """
    per_level = row.get("per_level") or []
    return [int(p.get("human_actions") or 0) for p in per_level]


def rescore_row(row: dict) -> RescoreResult:
    """Re-score one recorded sweep row under both charge models."""
    res = RescoreResult(
        game=str(row.get("game") or ""),
        arm=str(row.get("arm") or row.get("condition") or ""),
        seed=row.get("seed"),
        budget=row.get("budget"),
        n_resets=int(row.get("n_resets") or 0),
        offline_actions=int(row.get("actions") or 0),
        frames=row.get("n_frames"),
        recorded_efficiency=row.get("efficiency"),
    )
    if row.get("n_resets") is None:
        res.rescorable = False
        res.reason = "no_reset_count_recorded"
        return res

    baselines = _baselines_from_row(row)
    if not baselines or not any(baselines):
        res.rescorable = False
        res.reason = "no_human_baselines_in_row"
        return res
    res.baselines = baselines

    lua = [int(x) for x in (row.get("level_up_actions") or [])]
    # Per-level OFFLINE action costs: successive differences of the cumulative
    # level-up action counts.
    level_offline, prev = [], 0
    for at in lua:
        level_offline.append(at - prev)
        prev = at
    res.level_offline = level_offline
    res.n_levels_completed = len(level_offline)
    tail = int(res.offline_actions) - prev

    game_won = len(level_offline) >= len(baselines) and len(baselines) > 0

    res.offline_score, _ = gateway_score_via_calculator(
        baselines, level_offline, tail, game_won=game_won
    )
    res.worst_score, res.worst_alloc = worst_case_allocation(
        baselines, level_offline, res.n_resets, tail, game_won=game_won
    )
    res.greedy_score, _ = greedy_worst_case(
        baselines, level_offline, res.n_resets, tail, game_won=game_won
    )
    res.greedy_agrees = abs(res.greedy_score - res.worst_score) < 1e-9
    res.delta_worst = res.offline_score - res.worst_score
    if res.offline_score > 0:
        res.rel_delta_worst = res.delta_worst / res.offline_score

    # EXACT path, available only when the row carries per-level reset
    # attribution (emitted by the instrumented run_game; absent from every
    # pre-2026-07-26 corpus).
    rb = row.get("resets_before_levelups")
    if isinstance(rb, list) and len(rb) == len(level_offline):
        charged, prev_r = [], 0
        for i, cum_r in enumerate(rb):
            charged.append(level_offline[i] + int(cum_r) - prev_r)
            prev_r = int(cum_r)
        res.exact_resets_before_levelups = [int(x) for x in rb]
        res.exact_score, _ = gateway_score_via_calculator(
            baselines, charged, tail, game_won=game_won
        )
    return res


# --------------------------------------------------------------------------
# Cross-check: path 1 vs path 2, cell by cell.
# --------------------------------------------------------------------------


def crosscheck_row(row: dict, res: RescoreResult) -> dict[str, Any] | None:
    """Rebuild the row's OFFLINE score through the FULL installed chain.

    Constructs a frame sequence with ZERO resets whose per-level charged action
    counts equal the row's recorded offline per-level counts, replays it through
    the real `Scorecard` mutators, and scores it via `from_scorecard`. If path 1
    and path 2 disagree, our calculator-driving is wrong.
    """
    if not res.rescorable or not res.baselines:
        return None
    kinds: list[str] = []
    levels: list[int] = []
    lvl = 0
    for cost in res.level_offline:
        for _ in range(cost):
            kinds.append("ACTION")
            levels.append(lvl)
        lvl += 1
        if levels:
            levels[-1] = lvl  # the level-up is observed on the last action frame
    tail = res.offline_actions - sum(res.level_offline)
    for _ in range(max(0, tail)):
        kinds.append("ACTION")
        levels.append(lvl)
    if not kinds:
        return None
    full = gateway_score_via_full_chain(res.game or "xx00", res.baselines, kinds, levels)
    return {
        "game": res.game,
        "calculator_score": round(res.offline_score, 6),
        "full_chain_score": round(full, 6),
        "agree": abs(full - res.offline_score) < 1e-6,
    }


def crosscheck_reset_charge(game_id: str = "zz00") -> dict[str, Any]:
    """Independent confirmation that the installed chain charges a RESET.

    Two replays reaching level 1 after the same number of ACTION frames, one
    with N extra RESET frames interleaved BEFORE the level-up. If the gateway
    did not charge resets the two scores would be equal.
    """
    baselines = [10, 20]
    kinds_a = ["ACTION"] * 10
    levels_a = [0] * 9 + [1]
    kinds_b = ["RESET"] * 5 + ["ACTION"] * 10
    levels_b = [0] * 5 + [0] * 9 + [1]
    a = gateway_score_via_full_chain(game_id, baselines, kinds_a, levels_a)
    b = gateway_score_via_full_chain(game_id, baselines, kinds_b, levels_b)
    return {
        "no_resets_score": round(a, 6),
        "five_resets_before_levelup_score": round(b, 6),
        "reset_is_charged": b < a,
        "note": "5 resets before the level-up inflate level 1 from 10 to 15 charged actions",
    }


def crosscheck_post_solve_tail_is_free(game_id: str = "zz01") -> dict[str, Any]:
    """Independent confirmation the post-solve tail costs nothing.

    Reproduces the settled 2026-07-26 charge-model result through this module's
    own chain driver, so the sensitivity model rests on a re-derived fact.
    """
    baselines = [10, 20]
    short = gateway_score_via_full_chain(game_id, baselines, ["ACTION"] * 10, [0] * 9 + [1])
    long_kinds = ["ACTION"] * 10 + ["RESET"] * 30 + ["ACTION"] * 500
    long_levels = [0] * 9 + [1] + [1] * 530
    long = gateway_score_via_full_chain(game_id, baselines, long_kinds, long_levels)
    return {
        "tail_0_score": round(short, 6),
        "tail_530_frames_incl_30_resets_score": round(long, 6),
        "tail_is_free": abs(short - long) < 1e-9,
    }


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------


def load_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl.gz"):
        out = []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, list):
        return d
    for key in ("rows", "cells"):
        if isinstance(d.get(key), list) and d[key]:
            return d[key]
    return []


def discover_corpora() -> list[str]:
    pats = [
        os.path.join(REPO, "results/early_stop_sweep_20260726/*.json"),
        os.path.join(REPO, "results/cptb_20260726_cells/*.jsonl.gz"),
    ]
    out: list[str] = []
    for p in pats:
        out.extend(sorted(glob.glob(p)))
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="row files (default: discovered corpora)")
    ap.add_argument("--selftest", action="store_true", help="run the chain cross-checks")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        print(json.dumps(crosscheck_reset_charge(), indent=2))
        print(json.dumps(crosscheck_post_solve_tail_is_free(), indent=2))
        return 0

    paths = args.paths or discover_corpora()
    for p in paths:
        rows = load_rows(p)
        if args.limit:
            rows = rows[: args.limit]
        results = [rescore_row(r) for r in rows]
        ok = [r for r in results if r.rescorable and r.n_levels_completed]
        print(f"{os.path.basename(p):34s} rows={len(rows):5d} rescorable_with_levelup={len(ok):4d}")
        for r in ok[:3]:
            print(
                f"   {r.game} arm={r.arm} resets={r.n_resets} offline={r.offline_score:.4f}"
                f" worst={r.worst_score:.4f} delta={r.delta_worst:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
