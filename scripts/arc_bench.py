#!/usr/bin/env python3
"""One ARC number that can move, measured the way the competition measures.

WHY THIS EXISTS
---------------
The ARC loop stopped being able to tell whether it was improving.

`reproducible_total_levels` is 183 of 183 -- every public game is cleared and hand-adaptered, so
the metric that steered this work for months is pinned and can never move again. Nothing replaced
it. Measured 2026-08-13 over the last 10 milestones: 16 ARC tasks, 13 of them ending in
`ready_no_solve_claim` or `default_off`. Three were named "holdout" -- exp6295, exp6308, exp6401 --
and their metric sets share ZERO keys. Three protocols, three vocabularies, nothing comparable.
You cannot subtract milestone .542 from milestone .550.

A loop with 101 `CARNOT_ARC_*` flags and no comparable number is not self-improving. It is
accumulating options nobody can choose between.

WHAT IT MEASURES, AND WHY NOT A PROXY
--------------------------------------
ARC-AGI-3 scores levels cleared against actions spent, on games the agent has never seen. So this
measures exactly that, locally, rather than inventing a proxy that would then need calibrating
against a scored submission -- and there is essentially one scored submission on record, so no
calibration could be established anyway. If the benchmark IS the metric, there is nothing to
calibrate.

The held-out surface is the adapter-free path: `graph_explore_solve_v2` driving the offline
arcade, with every hand-written `GameAdapter` bypassed. That is the same first-contact mechanism
the live agent uses on a game it has never seen.

Two numbers per game, because the competition cares about both:

    levels_cleared   did the agent get anywhere at all
    actions_spent    how much did it burn doing so (the scored metric squares efficiency,
                     so a brute-force clear is worth little)

Verified to have real variance before this file was written, at max_expansions=3000:

    ls20  level 1   13-step solution   17,197 actions   22s
    ft09  level 1    4-step solution    8,768 actions    6s
    vc33  level 1    3-step solution      232 actions    0s
    ka59  level 0                       13,855 actions    8s
    dc22  level 0                       11,737 actions   10s

Three of five clear, two do not, and the action spend ranges 232 to 17,197. A constant would have
been useless; this discriminates.

THE WEAKNESS, STATED OUT LOUD BECAUSE IT IS REAL
-------------------------------------------------
These are the 25 PUBLIC games. Their mechanics are known to whoever wrote the agent, and their
adapters were authored by reading them. Disabling the adapter removes the hand-written route, not
the knowledge that produced it. So this is the best held-out surface that exists here, and it is
still a proxy for a hidden game. Every report prints that sentence. Do not quote a number from
this file as a hidden-game result.

WHAT IT IS NOT. Not a gate. It never blocks a commit, never fails the conductor, never edits an
artifact. It produces one comparable number per milestone so the promotion machinery
(`scripts/arc_flag_ledger.py`) has something to select on.

Usage:
    python3 scripts/arc_bench.py                        # rotating subset (default 8 games)
    python3 scripts/arc_bench.py --all                  # full 25-game sweep
    python3 scripts/arc_bench.py --games ls20,vc33      # explicit
    python3 scripts/arc_bench.py --out /tmp/bench.json  # where to write (never a fixed path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ROTATION = REPO / "ops" / ".arc_bench_rotation.json"

SCHEMA = "carnot.arc_bench.v1"

# Budget. Chosen from the measurement above: 3000 expansions clears a level on the games that can
# be cleared adapter-free, and the whole 25-game sweep lands near five minutes. Cheap enough to
# run every milestone AND on every promotion, which is what the regression gate needs.
DEFAULT_MAX_EXPANSIONS = 3000
DEFAULT_MAX_DEPTH = 60
DEFAULT_SUBSET = 8

# The SCORED engine's defaults, taken from what actually ships rather than chosen here.
# `make_carnot_agent`'s CarnotAgent sets MAX_ACTIONS = 2000 (raised from 400 on 2026-08-07); the
# frontier seed is the shipped `frontier_discipline_seed`. Measuring at a budget the agent does not
# use would produce numbers that do not describe the agent.
SHIPPED_MAX_ACTIONS = 2000
SHIPPED_FRONTIER_SEED = 20260724

CAVEAT = (
    "These are the 25 PUBLIC games with their hand-written GameAdapter bypassed. Disabling the "
    "adapter removes the hand-written route, not the knowledge that produced it, so this is a "
    "proxy for hidden-game performance and not a hidden-game result."
)


def roster() -> list[str]:
    """The 25 public games, from the registry rather than a list written out here.

    A hardcoded roster is a pattern list that drifts narrower than its concept, which is this
    project's most-repeated bug. Reading the registry means adding a game to the registry adds it
    to the benchmark.
    """
    import yaml

    data = yaml.safe_load(REGISTRY.read_text()) or {}
    games = data.get("games") or []
    if isinstance(games, dict):
        names = list(games)
    else:
        names = [g.get("game") for g in games if isinstance(g, dict)]
    return sorted(n for n in names if n)


def _roster_signature(games: list[str]) -> str:
    """Hash of the roster, so rotation resets when the roster changes.

    Straight from the QA-layer audit's `units_signature` lesson: an offset persisted against a
    roster that has since changed will silently skip the newly-added entries, sometimes for many
    runs. A changed roster resolves the offset to 0. Re-measuring some games is cheap; a new game
    never being benchmarked is the failure that matters.
    """
    return hashlib.sha256("|".join(games).encode()).hexdigest()[:16]


def select(games: list[str], subset: int) -> list[str]:
    """Rotate through the roster so successive runs cover it, without running all 25 every time.

    Steering wants a cheap number every milestone; confirmation wants the full sweep, which is
    what `--all` is for and what the promotion gate uses.
    """
    if subset >= len(games):
        return games
    sig = _roster_signature(games)
    offset = 0
    try:
        state = json.loads(ROTATION.read_text())
        if state.get("roster_signature") == sig:
            offset = int(state.get("offset", 0)) % len(games)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        offset = 0  # unreadable state means start over, not skip
    picked = [games[(offset + i) % len(games)] for i in range(subset)]
    try:
        ROTATION.write_text(
            json.dumps(
                {"roster_signature": sig, "offset": (offset + subset) % len(games)}, indent=2
            )
        )
    except OSError:
        pass  # rotation is an optimisation; failing to persist must not fail the benchmark
    return picked


def run_game_scored(
    game: str,
    *,
    budget: int = SHIPPED_MAX_ACTIONS,
    llm: bool = False,
    seed: int = SHIPPED_FRONTIER_SEED,
) -> dict[str, Any]:
    """Run ONE game through the SCORED path: `E3AgentPolicy` driven by `lb.run_game`.

    WHY A SECOND ENGINE. The `explore` engine drives `graph_explore_solve_v2`, and only 48 of the
    95 tracked flags live inside that import closure. The other 47 are read by the cascade --
    induction, the world-model verifier, planning, the frontier tiers -- and setting one while
    running `explore` produces a byte-identical sweep that the promotion rule reads as "worthless".
    `scripts/arc_flag_ledger.py` refuses to measure those flags for exactly that reason. This
    engine is what un-refuses them.

    It is also the path that actually ships. `make_carnot_agent(Agent)` builds `E3AgentPolicy`, so
    a flag measured here is measured on the agent the competition runs.

    `charged_actions`, NOT `actions`. The live gateway bills resets; `lb.run_game` reports both and
    they differ (vc33: 387 against 400). Reporting the cheaper number would flatter every
    efficiency comparison by whatever the reset count happens to be.

    LLM OFF BY DEFAULT, and this is a real limitation rather than a default nobody thought about.
    A sweep with induction live needs a healthy llama-server, takes far longer, and can silently
    degrade mid-run -- `generate()` returns a failure tuple instead of raising, so the agent logs
    `skipped: proposer_failed` and carries on emitting rows labelled as LLM-on. See
    `scripts/arc_scored_path_lever_harness.py`, which instruments that case properly. Until this
    engine carries the same generator-liveness witness per row, `--llm` is available but a flag
    whose whole effect is inside induction should be measured with that harness, not this one.
    """
    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    row: dict[str, Any] = {
        "game": game,
        "adapter_used": False,
        "engine": "scored",
        "llm": llm,
        "budget": budget,
        "levels_cleared": 0,
        "actions_spent": 0,
        "solution_len": 0,
        "wall_s": 0.0,
        "error": None,
    }
    prev = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    if llm:
        os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    else:
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    t0 = time.time()
    try:
        import random

        random.seed(seed)
        policy = E3AgentPolicy(game, frontier_discipline_seed=seed)
        r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
        row["levels_cleared"] = int(r.get("levels") or 0)
        row["actions_spent"] = int(r.get("charged_actions") or r.get("actions") or 0)
        row["efficiency"] = r.get("efficiency")
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        # Restore rather than delete: a caller may have set it deliberately, and leaving this
        # process-global flipped would silently change every later cell in the same sweep.
        if prev is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = prev
    row["wall_s"] = round(time.time() - t0, 2)
    return row


def run_game(
    game: str,
    *,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict[str, Any]:
    """Run ONE game adapter-free and report levels cleared and actions spent.

    Actions are counted by wrapping `env.step`, because the solution length the solver returns is
    the length of the path it FOUND, not what it SPENT finding it -- and the scored metric cares
    about what was spent. On ls20 those differ by three orders of magnitude (13 versus 17,197).

    Never raises. A game that errors is reported as an error row rather than vanishing, because a
    benchmark that silently drops its hard cases reports a rising average while getting worse.
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

    row: dict[str, Any] = {
        "game": game,
        "adapter_used": False,
        "levels_cleared": 0,
        "actions_spent": 0,
        "solution_len": 0,
        "wall_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        counter = {"n": 0}
        original_step = env.step

        def counted(*a: Any, **k: Any) -> Any:
            counter["n"] += 1
            return original_step(*a, **k)

        env.step = counted  # type: ignore[method-assign]
        traj, level = graph_explore_solve_v2(
            env, 0, max_expansions=max_expansions, max_depth=max_depth
        )
        row["levels_cleared"] = int(level or 0)
        row["actions_spent"] = counter["n"]
        row["solution_len"] = len(traj) if traj else 0
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    row["wall_s"] = round(time.time() - t0, 2)
    return row


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Collapse the rows into the comparable numbers, and never hide what failed.

    `actions_per_level_cleared` is the efficiency half. It is None rather than 0 or infinity when
    nothing cleared, because "infinitely inefficient" and "did not run" are different facts and a
    sentinel number would be averaged into a later comparison as if it meant something.
    """
    ok = [r for r in rows if not r.get("error")]
    cleared = [r for r in ok if r["levels_cleared"] > 0]
    total_levels = sum(r["levels_cleared"] for r in ok)
    total_actions = sum(r["actions_spent"] for r in ok)
    return {
        "games_run": len(rows),
        "games_errored": len(rows) - len(ok),
        "games_cleared_at_least_one_level": len(cleared),
        "total_levels_cleared": total_levels,
        "total_actions_spent": total_actions,
        "clear_rate": round(len(cleared) / len(ok), 4) if ok else None,
        "actions_per_level_cleared": (
            round(total_actions / total_levels, 1) if total_levels else None
        ),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="full sweep instead of a rotating subset")
    ap.add_argument("--games", help="comma-separated explicit roster")
    ap.add_argument("--subset", type=int, default=DEFAULT_SUBSET)
    ap.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument(
        "--engine",
        choices=["explore", "scored"],
        default="explore",
        help="explore = graph_explore_solve_v2 (fast, 48 of 95 flags reachable); "
        "scored = E3AgentPolicy, the path that actually ships (all flags reachable)",
    )
    ap.add_argument("--budget", type=int, default=SHIPPED_MAX_ACTIONS, help="scored engine only")
    ap.add_argument(
        "--llm",
        action="store_true",
        help="scored engine only: run induction live. Needs a healthy llama-server, and this "
        "engine does NOT yet carry a per-row generator-liveness witness -- see run_game_scored.",
    )
    ap.add_argument("--quiet", action="store_true", help="silence the arcade's INFO logging")
    args = ap.parse_args(argv)

    if args.quiet:
        logging.disable(logging.INFO)

    all_games = roster()
    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    elif args.all:
        games = all_games
    else:
        games = select(all_games, args.subset)

    knob = (
        f"budget={args.budget} llm={args.llm}"
        if args.engine == "scored"
        else f"max_expansions={args.max_expansions}"
    )
    print(f"arc-bench {SCHEMA}: {len(games)} game(s), engine={args.engine}, {knob}")
    print(f"  adapter-free (held-out) path. {CAVEAT}\n")

    rows = []
    for g in games:
        r = (
            run_game_scored(g, budget=args.budget, llm=args.llm)
            if args.engine == "scored"
            else run_game(g, max_expansions=args.max_expansions)
        )
        rows.append(r)
        mark = "ERR" if r["error"] else f"L{r['levels_cleared']}"
        print(
            f"  {g:6} {mark:4} actions={r['actions_spent']:>7} "
            f"soln={r['solution_len']:>3} {r['wall_s']:>6.1f}s"
            + (f"  {r['error']}" if r["error"] else "")
        )

    agg = aggregate(rows)
    print("\n  " + json.dumps(agg))

    report = {
        "schema": SCHEMA,
        "engine": args.engine,
        "budget": args.budget if args.engine == "scored" else None,
        "llm": args.llm if args.engine == "scored" else None,
        "max_expansions": args.max_expansions,
        "max_depth": DEFAULT_MAX_DEPTH,
        "roster_size": len(all_games),
        "held_out_caveat": CAVEAT,
        "per_game_rows": rows,
        **agg,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
