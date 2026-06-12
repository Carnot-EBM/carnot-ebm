"""Exp 4082: ninth ARC-AGI-3 game solve retry via explore-first.

Spec refs: REQ-PHASE4-044, SCENARIO-PHASE4-044.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4082_ninth_game_explore_first.json"
RANDOM_SEED = 4082

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp4070_ninth_game_explore_first as exp4070  # noqa: E402
from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines  # noqa: E402
from carnot.agentic.arc_exp4082_ninth_game_explore_first import (  # noqa: E402
    INFERENCE_SUBSTRATE,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    select_exp4082_candidate_from_survey,
)

_confirm_arc_env_reachable = exp4070._confirm_arc_env_reachable
_load_offline_arcade = exp4070._load_offline_arcade
_load_online_arcade = exp4070._load_online_arcade
_run_ft09_explore_first = exp4070._run_ft09_explore_first


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(*, seed: int = RANDOM_SEED, write: bool = True) -> dict[str, Any]:
    started = time.time()
    try:
        arc_env_count = _confirm_arc_env_reachable()
    except Exception:
        artifact = blocked_artifact(
            random_seed=seed,
            duration_s=round(time.time() - started, 3),
            inference_substrate=INFERENCE_SUBSTRATE,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    survey = json.loads((REPO / "results" / "arc3_win_condition_survey.json").read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")
    candidate = select_exp4082_candidate_from_survey(survey, baselines)
    offline_arcade = _load_offline_arcade()
    online_arcade = _load_online_arcade()
    outcome = _run_ft09_explore_first(
        offline_arcade,
        online_arcade,
        candidate,
        arc_env_count=arc_env_count,
    )
    artifact = build_artifact(
        outcome,
        candidate,
        random_seed=seed,
        duration_s=round(time.time() - started, 3),
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = run(seed=args.seed, write=True)
    print(f"-> {result['honest_verdict']}")
    sys.exit(0 if result["honest_verdict"].startswith(("success:", "complete:", "blocked_")) else 1)
