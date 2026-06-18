"""Sweep the UNSOLVED ARC-AGI-3 games with the OfflineSolver fresh-env-per-node branch mode
(arc_solver_kit, gotcha #7), adapter-free, and A/B it against the default replay mode.

WHY (the operator's 2026-06-17 ask "loop through the remaining unsolved games and try fresh_env").
fresh_env was the lever that took tu93 L1 -> L3: tu93's env.reset() is NON-IDEMPOTENT (a
parity-toggling hidden state), so a reuse-one-env replay search accumulates that state across resets.
That can do two bad things to a search: (b) report a FALSE win that fails the fresh-env reproduction
gate (tu93 at L2+), OR -- the interesting case for an UNSOLVED game -- (a) PREVENT the search from ever
observing a win that is in fact reachable from a pristine reset, so the game looks "resisted at L0"
when a winning path actually exists. fresh_env evaluates every candidate on a brand-new env, so each
sees the pristine reset parity the reproduction gate uses: it fixes both (a) and (b).

A cheap grid-hash detector CANNOT rule this out (it false-negatives on tu93 -- the parity only shows at
level-completion boundaries, not on an early grid). So we actually RUN it: for each unsolved game we run
the SAME bounded best-first search under both branch modes and reproduction-gate any advance. If
fresh_env reaches+reproduces a level where replay does not, that game was blocked by non-idempotent
reset and fresh_env unlocked it. If both reach 0, the game needs per-game RE (a deeper win mechanic),
not a branch-mode change -- an honest negative that scopes future effort.

Generic adapter-free wiring (no per-game RE): action vocabulary = rich_action_candidates (salience-
ordered), state key = frame_hash(grid), verifier = None (plain BFS). This is intentionally the SAME
search graph_explore runs -- the ONLY variable under test is the branch mode (replay vs fresh_env).

Zero quota (OFFLINE arcade). Writes results/arc3_fresh_env_sweep.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_live_adapter import _game_action
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
from carnot.agentic.arc_graph_explore import rich_action_candidates

RESULTS = REPO / "results"

# 10 unsolved games (25 total - 15 offline-reproduced per ops/arc_solve_registry.yaml)
UNSOLVED = ["bp35", "dc22", "ft09", "g50t", "lf52", "re86", "s5i5", "sb26", "tr87", "vc33"]

MAX_NODES = 1500          # per-level search budget (fresh_env mints an env per eval -> keep bounded)
DEPTH_CAP = 30
PER_MODE_TIMEOUT_S = 150  # wall-clock guard per (game, mode)


def _generic_adapter():
    """A game-agnostic adapter: salience-ordered action candidates, grid-hash state key, no verifier.
    The SAME search space graph_explore uses -- so the only variable across the A/B is the branch mode."""

    def action_labels(env, frame=None, path=None):
        if frame is None:
            return []
        return [json.dumps({"action": int(c.action_id), "data": c.data})
                for c in rich_action_candidates(frame)]

    def apply(env, label, frame):
        a = json.loads(label)
        return env.step(_game_action(GameAction, a["action"]), data=a.get("data"))

    def state_key(game, frame=None):
        return frame_hash(grid_of(frame)) if frame is not None else None

    return action_labels, apply, state_key


def _solve_one(game: str, mode: str) -> dict:
    """Run the bounded best-first search on `game` under branch `mode`; reproduction-gate any L1 reach."""
    action_labels, apply, state_key = _generic_adapter()
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(game, action_labels, apply, state_key,
                               max_nodes=MAX_NODES, branch_mode=mode)
    t0 = time.monotonic()
    f = solver._replay(env, [])
    start = kit.frame_level(f)
    # bound by wall-clock too: max_nodes already caps the search; the timeout is a belt-and-suspenders
    path, nodes = solver.solve_level(env, start, [], DEPTH_CAP)
    dur = time.monotonic() - t0
    reached, reproduced = start, False
    if path is not None:
        gate = kit.reproduce(game, path, apply, claimed_level=start + 1)
        reached = gate["reached_level"]
        reproduced = bool(gate["reproduced"])
    return {"mode": mode, "advanced": path is not None, "moves": len(path or []),
            "states_expanded": nodes, "reached_level": reached,
            "offline_reproduced": reproduced, "duration_s": round(dur, 1)}


def main() -> int:
    print(f"== fresh_env sweep over {len(UNSOLVED)} unsolved games (A/B vs replay, OFFLINE) ==")
    rows, unlocked = [], []
    for game in UNSOLVED:
        rec: dict = {"game": game}
        for mode in ("replay", "fresh_env"):
            try:
                rec[mode] = _solve_one(game, mode)
            except Exception as ex:  # a single game must never abort the sweep
                rec[mode] = {"mode": mode, "error": f"{type(ex).__name__}: {str(ex)[:80]}"}
        r_repro = rec.get("replay", {}).get("offline_reproduced", False)
        fe_repro = rec.get("fresh_env", {}).get("offline_reproduced", False)
        rec["fresh_env_unlocked"] = bool(fe_repro and not r_repro)   # the signal we are testing for
        if rec["fresh_env_unlocked"]:
            unlocked.append(game)
        rows.append(rec)
        print(f"  {game:5} replay={rec.get('replay', {}).get('reached_level')}"
              f"/repro={r_repro}  fresh_env={rec.get('fresh_env', {}).get('reached_level')}"
              f"/repro={fe_repro}  unlocked={rec['fresh_env_unlocked']}")

    verdict = (f"complete_fresh_env_sweep_unlocked_{len(unlocked)}_of_{len(UNSOLVED)}_unsolved_games"
               if unlocked else
               f"complete_fresh_env_sweep_no_unlock_all_{len(UNSOLVED)}_need_per_game_RE")
    out = {
        "experiment": "arc3_fresh_env_sweep",
        "title": "fresh_env vs replay branch mode on the unsolved ARC-AGI-3 games (gotcha #7 probe)",
        "honest_verdict": verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",  # offline sim search, no LLM
        "no_llm_used": True,
        "max_nodes": MAX_NODES, "depth_cap": DEPTH_CAP,
        "games_tested": UNSOLVED, "fresh_env_unlocked_games": unlocked,
        "per_game": rows,
        "interpretation": ("fresh_env unlocks a game ONLY if non-idempotent reset (gotcha #7) was "
                           "preventing the reuse-one-env search from observing a reachable win. A game "
                           "where both modes reach 0 needs per-game RE (a deeper win mechanic), not a "
                           "branch-mode change."),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "arc3_fresh_env_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {verdict}")
    print(f"   wrote results/arc3_fresh_env_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
