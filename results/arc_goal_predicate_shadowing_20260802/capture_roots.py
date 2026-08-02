"""Capture a real level-1 root grid for every game in the shadowing corpus.

WHY THIS EXISTS. The comparison in `measure_worker.py` runs the SHIPPED goal gate
(`arc_llm_reinduction._goal_satisfiability_check`), and that gate is a bounded
reachability search: it needs a START GRID to search from. The AB corpus
(`results/arc_object_perception_ab_change_fidelity_20260801/`) stored engines and
heldout-transition scores but NO grids, and the only committed root-grid pickles in
the repo cover 6 games (`results/arc_induce_bestofn_20260731/harness/capture/`),
which would have limited the comparison to 5 of the 23 double-definition cells.

WHAT THIS IS NOT. This does not play a scored or online game. `offline_arcade()` is
the zero-quota, no-network OFFLINE Arcade over the local `environment_files/`
checkout, which the ARC Live-Path Reachability Discipline explicitly permits as a
PUBLIC-game development proxy. Nothing here is or feeds a hidden-game submission.

FAITHFULNESS CAVEAT, stated up front because it bounds every number downstream.
The grid captured here is the board at `env.reset()` -- the game's OPENING board.
The grid the live planner actually searches from is `E3AgentPolicy.root_grid`, the
state the agent had reached when reinduction fired, which is generally NOT the
opening board. `verify_against_captured.py` measures that gap directly on the three
games where both are available rather than assuming it away.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))

OUT = HERE / "roots"


def _grid_from_frame(frame: object) -> np.ndarray:
    """The logical grid from an arc_agi frame.

    `frame.frame` is a list of layers; the LAST is the composited visible board, which
    is what `arc_competition_agent` feeds the world model. Falls back to treating the
    object itself as an array so a shape change in the SDK surfaces as a loud error
    here rather than as a silently wrong grid three files away.
    """
    raw = getattr(frame, "frame", None)
    if raw is None:
        return np.asarray(frame)
    return np.asarray(raw[-1])


def main() -> int:
    games = sorted(json.loads((HERE / "corpus_games.json").read_text()))
    from carnot.agentic.arc_solver_kit import offline_arcade

    OUT.mkdir(parents=True, exist_ok=True)
    arc = offline_arcade()
    manifest = []
    for game in games:
        try:
            env = arc.make(game)
            reset = env.reset()
            frame = reset[0] if isinstance(reset, (list, tuple)) else reset
            grid = _grid_from_frame(frame)
            arr = np.ascontiguousarray(np.asarray(grid, dtype=int))
            (OUT / f"{game}.pkl").write_bytes(pickle.dumps(arr))
            manifest.append(
                {
                    "game": game,
                    "ok": True,
                    "shape": list(arr.shape),
                    "n_colors": int(len(np.unique(arr))),
                    "colors": [int(c) for c in np.unique(arr)],
                    "sha256_16": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                    "levels_completed": int(getattr(frame, "levels_completed", 0) or 0),
                }
            )
        except Exception as exc:  # noqa: BLE001 - a failed game must not abort the sweep
            manifest.append({"game": game, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    (HERE / "roots_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ok = sum(1 for m in manifest if m["ok"])
    print(f"captured {ok}/{len(manifest)} root grids -> {OUT}")
    return 0 if ok == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
