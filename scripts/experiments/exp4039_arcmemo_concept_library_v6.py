"""Exp 4039: ArcMemo v6 compressed concept-library transfer on `.373`.

Spec refs: REQ-LEARN-4039, SCENARIO-LEARN-4039.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4039_arcmemo_concept_library_v6.json"
PRIOR_ARCMEMO_RESULTS = (
    "experiment_3982_arcmemo_solve_transfer.json",
    "experiment_3994_arcmemo_solve_transfer_v2.json",
    "experiment_4005_arcmemo_solve_transfer_v3.json",
    "experiment_4016_arcmemo_solve_transfer_v4.json",
    "experiment_4025_arcmemo_solve_transfer_v5.json",
)
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_arcmemo_concept_library_v6 import (  # noqa: E402
    artifact_schema_errors,
    build_transfer_artifact,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> Path:
    out = REPO / "results" / RESULT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(*, write: bool = True) -> dict[str, Any]:
    started = time.time()
    prior = [_read_json(REPO / "results" / name) for name in PRIOR_ARCMEMO_RESULTS]
    exp4035 = _read_json(REPO / "results" / "experiment_4035_hierarchical_search_over_vc33_wm.json")
    exp4038 = _read_json(REPO / "results" / "experiment_4038_seventh_game_explore_first.json")
    artifact = build_transfer_artifact(
        prior_artifacts=prior,
        exp4035=exp4035,
        exp4038=exp4038,
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
