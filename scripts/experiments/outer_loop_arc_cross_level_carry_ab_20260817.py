#!/usr/bin/env python3
"""Paired live-path A/B for the cross-level verified engine carry (REQ-ARC-XLEVEL-CARRY-1).

DESIGN
======
Arms differ in ONE bit: `CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY` (off = shipped default,
on = the new lever). Everything else is held fixed, including the proposer: a REPLAY
proposer that answers every induce() call by writing this game's ARCHIVED real
LLM-induced engine (from the recorded h2h runs) into the engine store. This keeps
induction quality constant across arms and needs no GPU, so the A/B isolates the carry
stage itself on the LIVE path (E3AgentPolicy -> _induce_and_plan), not a harness twin.

METRIC: per-level actions (actions-to-next-levelup) from the same `run_game` harness
every recorded per-level number uses, plus the carry stage's own fire/decline counters.
A cell where the ON arm never even RECORDS a carry attempt is an invalid cell, not a
null -- the lever-fired discipline this project learned the hard way (exp5836).

EXPECTED, stated before running: with today's archived engines (verify accuracy
0.0-0.78 on a next level, all below the 1.0 bar) the gate should DECLINE every carry,
so both arms should act identically and the honest headline is the populated decline
counters. An optional LAX arm (--lax-bar) lowers the accuracy bar via env to measure
the failure-mode cost of carrying a WEAK engine on purpose.

Offline, public games, no LLM server, no flag flipped, no tracked path written --
output goes only to this experiment's own results directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

# Engine store redirect BEFORE any agent import: the replay proposer writes engines, and
# nothing in this process may touch the real store.
_E3_TMP = tempfile.mkdtemp(prefix="arc_xlevel_ab_e3_")
os.environ["CARNOT_ARC_E3_DIR"] = _E3_TMP

SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"

ENGINE_ARCHIVES = [
    REPO / "results/arc_qwen38_h2h_stopped_20260817/engine_archive",
    REPO / "results/arc_qwen38_h2h_partial_20260817/engine_snapshots",
]


def _archived_source(game: str) -> tuple[str, str] | None:
    for archive in ENGINE_ARCHIVES:
        if not archive.is_dir():
            continue
        hits = sorted(archive.glob(f"*__{game}__*.py"))
        if hits:
            return hits[0].read_text(), str(hits[0].relative_to(REPO))
    return None


class ReplayProposer:
    """Answers induce() by replaying a RECORDED real LLM-induced engine for the game.

    Holds induction quality constant across arms with zero GPU cost. refactor() declines
    so the refinement loop runs exactly one round per induction call -- the call COUNT
    (the carry's wall-clock target) stays a clean unit.
    """

    include_playbook_exemplars: bool | str = False
    model_specs = "replay_of_recorded_qwen38_27b_engine_no_llm"

    def __init__(self, game: str, source: str) -> None:
        self.game = game
        self.source = source
        self.induce_calls = 0
        self.refactor_calls = 0

    def induce(self, game: str, transitions, cell: int, **kwargs) -> tuple[bool, str]:
        self.induce_calls += 1
        slot = Path(_E3_TMP) / game
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "world_model.py").write_text(self.source)
        return True, "replayed_recorded_engine"

    def refactor(self, game: str, result) -> tuple[bool, str]:
        self.refactor_calls += 1
        return False, "replay_proposer_has_no_refactor"


def run_cell(game: str, seed: int, budget: int, arm: str, lax_bar: float | None) -> dict:
    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    # Arm env, resolved by E3AgentPolicy.__init__ (constructed AFTER these are set).
    if arm == "on":
        os.environ["CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY"] = "1"
        os.environ.pop("CARNOT_ARC_CROSS_LEVEL_CARRY_MIN_ACCURACY", None)
    elif arm == "on_lax":
        os.environ["CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY"] = "1"
        os.environ["CARNOT_ARC_CROSS_LEVEL_CARRY_MIN_ACCURACY"] = str(lax_bar)
    else:
        os.environ.pop("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", None)
        os.environ.pop("CARNOT_ARC_CROSS_LEVEL_CARRY_MIN_ACCURACY", None)

    found = _archived_source(game)
    if found is None:
        return {"game": game, "seed": seed, "arm": arm, "error": "no_archived_engine"}
    source, source_path = found

    random.seed(seed)
    np.random.seed(seed % (2**32))
    proposer = ReplayProposer(game, source)
    policy = E3AgentPolicy(game, proposer=proposer, frontier_discipline_seed=seed)
    t0 = time.time()
    r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
    wall = time.time() - t0

    attempts = list(getattr(policy, "induction_attempts", []) or [])
    carry_rows = [
        {"reason_field": a.get("reason"), **a["cross_level_engine_carry"]}
        for a in attempts
        if "cross_level_engine_carry" in a
    ]
    levelup_attempts = [a for a in attempts if a.get("reason") == "level_up_reinduction"]
    return {
        "game": game,
        "seed": seed,
        "arm": arm,
        "budget": budget,
        "engine_source_replayed": source_path,
        "wall_s": round(wall, 2),
        "levels_reached": int(r.get("reached") or 0),
        "per_level_actions": r.get("per_level"),
        "actions_total": int(r.get("actions") or 0),
        "carry_flag_resolved": bool(policy.cross_level_engine_carry_enabled),
        "n_induction_attempts": len(attempts),
        "n_levelup_reinduction_attempts": len(levelup_attempts),
        "proposer_induce_calls": proposer.induce_calls,
        "proposer_refactor_calls": proposer.refactor_calls,
        "carry_attempt_rows": carry_rows,
        "carry_fires": sum(1 for c in carry_rows if c.get("fired")),
        "attempt_engine_sources": [a.get("engine_source") for a in attempts],
        "attempt_skipped": [a.get("skipped") for a in attempts],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--seeds", default="20260724,20260725,20260726")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--lax-bar", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    t0 = time.time()
    cells = []
    for game in games:
        for seed in seeds:
            for arm in arms:
                print(f"== {game} seed={seed} arm={arm}", flush=True)
                try:
                    cells.append(run_cell(game, seed, args.budget, arm, args.lax_bar))
                except Exception as exc:
                    cells.append({"game": game, "seed": seed, "arm": arm, "error": repr(exc)[:300]})
                row = cells[-1]
                print(
                    f"   levels={row.get('levels_reached')} fires={row.get('carry_fires')} "
                    f"induce_calls={row.get('proposer_induce_calls')} wall={row.get('wall_s')}s",
                    flush=True,
                )

    # Paired deltas per (game, seed): ON minus OFF in total actions and induce calls.
    paired = []
    by_key = {(c["game"], c["seed"], c["arm"]): c for c in cells if "error" not in c}
    for game in games:
        for seed in seeds:
            off = by_key.get((game, seed, "off"))
            on = by_key.get((game, seed, "on"))
            if off and on:
                paired.append(
                    {
                        "game": game,
                        "seed": seed,
                        "levels_off": off["levels_reached"],
                        "levels_on": on["levels_reached"],
                        "actions_off": off["actions_total"],
                        "actions_on": on["actions_total"],
                        "actions_delta_on_minus_off": on["actions_total"] - off["actions_total"],
                        "induce_calls_off": off["proposer_induce_calls"],
                        "induce_calls_on": on["proposer_induce_calls"],
                        "carry_fires_on": on["carry_fires"],
                        "carry_engaged_on": len(on["carry_attempt_rows"]) > 0,
                    }
                )

    agent_path = REPO / "python/carnot/agentic/arc_competition_agent.py"
    artifact = {
        "experiment": "outer_loop_arc_cross_level_carry_ab_20260817",
        "title": "Paired live-path A/B: cross-level verified engine carry (replay proposer)",
        "run_date": time.strftime("%Y-%m-%d"),
        "inference_substrate": SUBSTRATE,
        "llm_enabled": False,
        "random_seeds_used": seeds,
        "budget": args.budget,
        "arms": arms,
        "duration_s": round(time.time() - t0, 2),
        "provenance": {
            "agent_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
            "harness": "scripts/arc_leaderboard_eval.py:run_game",
            "proposer": "ReplayProposer over recorded h2h engine sources (no LLM)",
        },
        "n_cells": len(cells),
        "paired": paired,
        "cells": cells,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
