"""A/B: recent-action-suffix state keying vs the shipped whole-frame key (REQ-ARC-GE-6110).

WHY THIS EXISTS
---------------
exp6094 established that the adapter-free explorer's failure on sc25/g50t is a FRONTIER
COLLAPSE, not a search wall: sc25 stopped at 24 expansions / 1 distinct state and g50t at
4100 / 843, identically at 6000 and 30000 budgets. The generic `node_id` hashes the whole
visible grid, and on these games behaviourally distinct states render identical frames
(diagnosed 2026-08-05: every sc25 root action is visually inert on its FIRST application —
the game consumes it while hidden state advances — so every successor aliases into the root
node and the frontier drains).

THE TREATMENT
-------------
`graph_explore_solve_v2(..., state_key_action_suffix_k=k)` (default OFF; env flag
`CARNOT_ARC_STATE_KEY_SUFFIX_K`): node identity = frame hash + the last k actions of the
arriving path — the classic k-th-order remedy for a non-Markov observation. Generic by
construction: frames + the agent's own actions only, no game ids, no per-game constants.

THE MEASUREMENT
---------------
2 collapse games (sc25, g50t) + 2 non-collapse controls (ls20, tu93), adapter-free, the
census budget (6000 expansions), max_depth 60, warmup False — the exact call the
adapter-free dev-twin cell made, plus stats={} so every number is a measurement (the
exp6094 lesson: never print a budget as a cost). The search is deterministic (no RNG in
v2); random_seed recorded as 0 for the artifact schema.

WIN CONDITION (pre-stated): collapse games discover more real states / stop collapsing
(explored_out flips to budget_exhausted or a level is reached) WITHOUT the control games
regressing (still solve their level, comparable expansions). Failure mode watched for:
uncontrolled state-space blowup wasting the budget (states >> distinct_frames).

Read-only against the offline arcade; CPU only; no game played on any scored endpoint;
writes ONLY its own artifact JSON.

Usage: outer_loop_arc_state_key_suffix_ab_20260805.py [--games sc25,g50t,ls20,tu93]
                                                      [--ks 0,1] [--budget 6000] [--out X]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # this probe must never take a GPU
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2  # noqa: E402


def run_cell(arc, game: str, k: int, budget: int) -> dict:
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    stats: dict = {}
    t0 = time.time()
    traj, lvl = graph_explore_solve_v2(
        env,
        0,
        max_expansions=budget,
        max_depth=60,
        warmup=False,
        state_key_action_suffix_k=k,
        stats=stats,
    )
    wall = time.time() - t0
    expansions = int(stats.get("expansions") or 0)
    return {
        "game": game,
        "k": k,
        "budget": budget,
        "expansions": expansions,
        "states": int(stats.get("states") or 0),
        "distinct_frames": int(stats.get("distinct_frames") or 0),
        # explored_out: the frontier emptied BEFORE the budget with no advance — the
        # collapse signature. An arm that stops early explored LESS, not better.
        "explored_out": traj is None and expansions < budget,
        "budget_exhausted": expansions >= budget,
        "advanced": traj is not None,
        "reached_level": int(lvl),
        "actions": len(traj) if traj is not None else None,
        "wall_s": round(wall, 2),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="sc25,g50t,ls20,tu93")
    ap.add_argument("--ks", default="0,1")
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument(
        "--out",
        default=str(REPO / "results" / "outer_loop_arc_state_key_suffix_ab_20260805.json"),
    )
    args = ap.parse_args(argv[1:])

    arc = kit.offline_arcade()
    rows = []
    for game in args.games.split(","):
        for k in (int(x) for x in args.ks.split(",")):
            row = run_cell(arc, game.strip(), k, args.budget)
            rows.append(row)
            print(json.dumps(row), flush=True)

    artifact = {
        "experiment": "outer_loop_arc_state_key_suffix_ab",
        "schema": "carnot.arc_state_key_suffix_ab.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "Does keying explorer node identity on (frame_hash, last-k-actions) fix the "
            "sc25/g50t frontier collapse without regressing non-collapse games?"
        ),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": 0,  # graph_explore_solve_v2 is deterministic; recorded for schema
        "config": {
            "budget": args.budget,
            "max_depth": 60,
            "warmup": False,
            "adapter_free": True,
            "arms": args.ks,
        },
        "rows": rows,
        "duration_s": round(sum(r["wall_s"] for r in rows), 2),
        "reproducibility_checksum": hex(abs(hash(json.dumps(rows, sort_keys=True, default=str)))),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
