"""Exp 4050: ArcMemo v7 cross-game concept-library transfer.

Spec refs: REQ-LEARN-4050, SCENARIO-LEARN-4050.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4050_arcmemo_cross_game_transfer_v7.json"
PRIOR_SOLVED_GAME_SOURCES = (
    ("r11l", "results/experiment_3946_r11l_first_solve.json"),
    ("lp85", "results/experiment_3954_second_game_solve.json"),
    ("sc25", "results/experiment_3966_third_game_first_solve.json"),
    ("su15", "results/experiment_4004_fourth_game_explore_first.json"),
    ("tn36", "results/experiment_4015_fifth_game_explore_first.json"),
    ("cd82", "results/experiment_4024_fifth_game_explore_first.json"),
    ("dc22", "results/experiment_4038_seventh_game_explore_first.json"),
)

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_arcmemo_cross_game_transfer_v7 import (  # noqa: E402
    artifact_schema_errors,
    build_cross_game_transfer_artifact,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_solved_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_game, relative_path in PRIOR_SOLVED_GAME_SOURCES:
        rows.append(
            {
                "source_game": source_game,
                "source_artifact": relative_path,
                "payload": _read_json(REPO / relative_path),
            }
        )
    return rows


def _write_artifact(artifact: dict[str, Any]) -> Path:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(*, write: bool = True) -> dict[str, Any]:
    started = time.time()
    exp4049 = _read_json(REPO / "results" / "experiment_4049_eighth_game_explore_first.json")
    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_artifacts(),
        exp4049=exp4049,
        duration_s=round(time.time() - started, 3),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
