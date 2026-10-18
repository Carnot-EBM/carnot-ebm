#!/usr/bin/env python3
"""Exp 3929 ARC-AGI-3 synthetic action-efficiency artifact.

Spec refs: REQ-PHASE4-007, SCENARIO-PHASE4-007.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.agentic.arc_agi3_action_efficiency import (  # noqa: E402
    MAX_STEPS,
    N_EPISODES,
    RANDOM_SEED,
    build_blocked_artifact,
    build_episode_configs,
    build_result_artifact,
    check_preconditions,
    probe_real_benchmark_access,
    run_action_efficiency_measurement,
    write_result_artifact,
    RichSyntheticArcEnv,
)


OUTPUT_PATH = REPO_ROOT / "results" / "experiment_3929_arc_agi3_action_efficiency.json"


def main() -> dict[str, Any]:
    started_at = time.perf_counter()
    preconditions = check_preconditions()
    real_benchmark_preflight = probe_real_benchmark_access()

    if preconditions.blocked_resource:
        env = RichSyntheticArcEnv(build_episode_configs(1, random_seed=RANDOM_SEED)[0])
        artifact = build_blocked_artifact(
            preconditions=preconditions,
            real_benchmark_preflight=real_benchmark_preflight,
            duration_s=time.perf_counter() - started_at,
            final_observation=env.reset(),
        )
        write_result_artifact(artifact, OUTPUT_PATH)
        return artifact

    measurement = run_action_efficiency_measurement(
        n_episodes=N_EPISODES,
        random_seed=RANDOM_SEED,
        max_steps=MAX_STEPS,
    )
    artifact = build_result_artifact(
        measurement,
        preconditions=preconditions,
        real_benchmark_preflight=real_benchmark_preflight,
        duration_s=time.perf_counter() - started_at,
    )
    write_result_artifact(artifact, OUTPUT_PATH)
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
