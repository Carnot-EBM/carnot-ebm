"""Run Exp 1474 T-SKM linear constraint projection smoke.

Spec: REQ-VERIFY-1474, SCENARIO-VERIFY-1474.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from carnot.verify.skm_projection import (
    HELPER_PATH,
    REQUIRED_ARTIFACT_FIELDS,
    evaluate_toy_cases,
)


DEFAULT_OUTPUT_PATH = Path("results/experiment_1474_tskm_linear_constraint_projection_smoke.json")


def _write_json(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_in_progress_artifact(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, object]:
    """Seed the required deliverable before evaluating the projection smoke."""

    artifact: dict[str, object] = {
        "status": "in_progress",
        "toy_cases_evaluated": 0,
        "zero_violation_projection": False,
        "max_constraint_violation": None,
        "baseline_verifier_agreement": False,
        "projection_iterations_p50": None,
        "projection_iterations_p95": None,
        "helper_path": HELPER_PATH,
        "tests_run": [],
        "honest_verdict": "in_progress",
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing artifact fields: {sorted(missing)}")
    _write_json(output_path, artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    tests_run: Sequence[str] = (),
) -> dict[str, object]:
    """Run the bounded CPU-only projection smoke and write the final artifact."""

    write_in_progress_artifact(output_path)
    summary = evaluate_toy_cases()
    artifact = summary.to_artifact(tests_run=tests_run)
    _write_json(output_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through run_experiment tests.
    run_experiment(output_path=DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":  # pragma: no cover
    main()
