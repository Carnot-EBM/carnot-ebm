"""Experiment 1189: Phase 4 stronger BFS baseline at 5x5 and 10x10.

This experiment closes paper ISSUE-9 (paper-v5-integrity-remediation.md) by
replacing the random-legal-action baseline used by Exp 1165 with BFS-to-goal,
which is the gold standard for shortest-path finding on deterministic puzzles.
The comparison is run at two grid sizes: 5x5 (where BFS is expected to win or
tie because the puzzles are tractable) and 10x10 (where BFS may hit the
100,000-state intractability cap and Phase 4's variational free-energy
minimization may produce the only actionable answer).

Spec coverage: REQ-KONA-015, SCENARIO-KONA-015
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

from carnot.phase3.active_inference_pilot import (  # noqa: E402
    ARC3PuzzleEnv,
    ActiveInferencePilot,
    BFSBaseline,
    BFS_INTRACTABLE_STATE_LIMIT,
    LATENT_DIM,
    build_default_k5_ensemble_energies,
    build_stronger_baseline_artifact,
    run_phase4_vs_bfs,
    write_experiment_artifact,
)
from carnot.phase3.snap_validity import snap_to_action  # noqa: E402
from carnot.samplers.phase4_sampler import Phase4Sampler  # noqa: E402

RESULT_PATH = REPO_ROOT / "results" / "experiment_1189_phase4_stronger_baseline_10x10.json"
SEED = 1189
N_GIBBS_SWEEPS = 40
MAX_PHASE4_ACTIONS_5X5 = 30
MAX_PHASE4_ACTIONS_10X10 = 60


def _build_pilot() -> ActiveInferencePilot:
    """Build the Phase 4 pilot with the same blocked-Gibbs sampler used by Exp 1165."""
    sampler = Phase4Sampler(
        algorithm="blocked_gibbs",
        seed=SEED,
        step_size=0.01,
        temperature=0.25,
        discrete_indices=tuple(range(LATENT_DIM)),
        continuous_indices=(),
        hmc_regime_used="C",
    )
    return ActiveInferencePilot(
        build_default_k5_ensemble_energies(),
        snap_to_action,
        sampler,
        latent_dim=LATENT_DIM,
        rng_seed=SEED,
    )


def main() -> dict[str, Any]:
    pilot = _build_pilot()
    bfs = BFSBaseline(state_limit=BFS_INTRACTABLE_STATE_LIMIT)

    env_5x5 = ARC3PuzzleEnv(grid_size=5)
    t0 = time.time()
    rows_5x5 = run_phase4_vs_bfs(
        pilot,
        env_5x5,
        bfs,
        max_actions=MAX_PHASE4_ACTIONS_5X5,
        n_gibbs_sweeps=N_GIBBS_SWEEPS,
    )
    print(f"5x5 phase4-vs-bfs took {time.time() - t0:.1f}s", file=sys.stderr)

    env_10x10 = ARC3PuzzleEnv(grid_size=10)
    t0 = time.time()
    rows_10x10 = run_phase4_vs_bfs(
        pilot,
        env_10x10,
        bfs,
        max_actions=MAX_PHASE4_ACTIONS_10X10,
        n_gibbs_sweeps=N_GIBBS_SWEEPS,
    )
    print(f"10x10 phase4-vs-bfs took {time.time() - t0:.1f}s", file=sys.stderr)

    artifact = build_stronger_baseline_artifact(
        rows_5x5,
        rows_10x10,
        experiment_id=1189,
        blocked_gibbs_params={
            "n_sweeps": int(N_GIBBS_SWEEPS),
            "n_blocks": LATENT_DIM,
            "step_size": 0.01,
            "bfs_state_limit": int(BFS_INTRACTABLE_STATE_LIMIT),
        },
    )
    write_experiment_artifact(artifact, RESULT_PATH)
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
