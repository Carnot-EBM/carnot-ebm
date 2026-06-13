"""Exp 4120: thirteenth ARC-AGI-3 strict non-spatial explore-first attempt.

Spec refs: REQ-PHASE4-048, SCENARIO-PHASE4-048.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4120_thirteenth_game_explore_first.json"
RANDOM_SEED = 4120

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines  # noqa: E402
from carnot.agentic.arc_exp4120_thirteenth_game_explore_first import (  # noqa: E402
    NoUnsolvedNonSpatialCandidate,
    artifact_schema_errors,
    blocked_artifact,
    build_no_solve_artifact,
    select_exp4120_candidate_from_survey,
)


def _write_artifact(artifact: dict[str, Any]) -> None:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_offline_arcade() -> Any:
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )


def _arc_env_count(offline_arcade: Any) -> int:
    return len(offline_arcade.get_environments())


def run(*, seed: int = RANDOM_SEED, write: bool = True) -> dict[str, Any]:
    started = time.time()
    survey_path = REPO / "results" / "arc3_win_condition_survey.json"
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")
    if not baselines:
        artifact = blocked_artifact(
            random_seed=seed,
            duration_s=round(time.time() - started, 3),
            reason="offline fixtures unavailable",
        )
        if write:
            _write_artifact(artifact)
        return artifact

    offline_arcade = _load_offline_arcade()
    arc_env_count = _arc_env_count(offline_arcade)
    try:
        candidate = select_exp4120_candidate_from_survey(survey, baselines)
    except NoUnsolvedNonSpatialCandidate as exc:
        artifact = build_no_solve_artifact(
            exc.report,
            random_seed=seed,
            duration_s=round(time.time() - started, 3),
            offline_driver_available=True,
            arc_env_count=arc_env_count,
        )
    else:
        artifact = blocked_artifact(
            random_seed=seed,
            duration_s=round(time.time() - started, 3),
            reason=f"strict non-spatial target {candidate.game_id} selected but solver arm not implemented",
        )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise RuntimeError("; ".join(errors))
    if write:
        _write_artifact(artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = run(write=not args.no_write)
    print(result["honest_verdict"])
    raise SystemExit(0 if result["honest_verdict"].startswith(("success:", "complete:", "blocked_")) else 1)
