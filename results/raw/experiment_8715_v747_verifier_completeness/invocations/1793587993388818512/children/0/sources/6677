"""Experiment 1210: Phase 4 vs BFS on BFS-intractable scrambled-grid puzzles.

This experiment fixes the puzzle generator so that the initial energy of
every generated puzzle is strictly greater than zero. Exp 1189 (the
predecessor) generated puzzles whose verifier-based initial energy was
identically zero because the legal-action set always contained the
correct action; Phase 4 then trivially picked the correct action and
BFS solved the puzzle in well under 100 explored states. The headline
"Phase 4 vs BFS" comparison was uninformative.

The new puzzle generator builds 15 cell-flip puzzles on a 15x15 mod-2
grid by applying 50 random flip actions in reverse from the all-zero
goal grid. The action set has 225 actions per state; BFS therefore hits
the 100,000-state cap inside roughly three depth levels, so it cannot
reach a goal that is at Hamming distance ~40 from the initial state.
Phase 4, which gradient-descends free energy = Hamming distance, can
solve the puzzle in ~40 actions because the energy gradient points at
any cell currently set to 1.

Spec: REQ-KONA-016, SCENARIO-KONA-016
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.phase3.scrambled_grid_puzzle import (  # noqa: E402
    BFS_INTRACTABLE_STATE_LIMIT,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_PUZZLES,
    DEFAULT_N_SCRAMBLE_STEPS,
    run_experiment,
    write_artifact,
)

RESULT_PATH = REPO_ROOT / "results" / "experiment_1210_phase4_bfs_intractable_puzzles_v2.json"


def main() -> dict[str, Any]:
    t0 = time.time()
    artifact = run_experiment(
        n_puzzles=DEFAULT_N_PUZZLES,
        grid_size=DEFAULT_GRID_SIZE,
        n_scramble_steps=DEFAULT_N_SCRAMBLE_STEPS,
        bfs_state_limit=BFS_INTRACTABLE_STATE_LIMIT,
        max_phase4_actions=200,
        n_gibbs_sweeps=40,
    )
    elapsed = time.time() - t0
    artifact["wall_clock_seconds"] = float(elapsed)
    write_artifact(artifact, RESULT_PATH)
    print(
        f"exp1210: n={artifact['n_puzzles_total']} "
        f"grid={artifact['grid_size']}x{artifact['grid_size']} "
        f"nonzero_init={artifact['initial_energy_nonzero_fraction']:.2f} "
        f"bfs_intractable={artifact['bfs_intractable_fraction']:.2f} "
        f"phase4_solved_intractable={artifact['phase4_solved_on_intractable']} "
        f"verdict={artifact['honest_verdict']} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
