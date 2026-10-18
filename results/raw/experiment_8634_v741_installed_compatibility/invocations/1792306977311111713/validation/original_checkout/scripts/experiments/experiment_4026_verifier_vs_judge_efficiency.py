"""Exp 4026: corrected verifier-vs-judge efficiency artifact.

Spec refs: REQ-PHASE4-034, SCENARIO-PHASE4-034.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4026_verifier_vs_judge_efficiency.json"
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_verifier_vs_judge_efficiency_corrigendum import (  # noqa: E402
    artifact_schema_errors,
    build_corrected_artifact,
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
    exp4013 = _read_json(REPO / "results" / "experiment_4013_verifier_vs_judge_efficiency.json")
    artifact = build_corrected_artifact(
        exp4013=exp4013,
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
