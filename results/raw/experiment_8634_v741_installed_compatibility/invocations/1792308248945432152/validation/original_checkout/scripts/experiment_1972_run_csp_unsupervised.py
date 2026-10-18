#!/usr/bin/env python3
"""Exp 1972: unsupervised RUN-CSP binary-CSP solver.

Spec: REQ-SAMPLE-1972, SCENARIO-SAMPLE-1972
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.inference.run_csp import DEFAULT_RESULT_PATH, run_experiment as _run_experiment


def run_experiment(output_path: Path | str = DEFAULT_RESULT_PATH) -> dict[str, Any]:
    """Run the CPU-only RUN-CSP experiment and write the results artifact."""

    return _run_experiment(output_path=output_path)


def main() -> None:
    """CLI entry point for the research conductor."""

    artifact = run_experiment(DEFAULT_RESULT_PATH)
    eval_1000 = artifact["evaluations"]["1000"]
    print(
        artifact["experiment_id"],
        eval_1000["num_variables"],
        eval_1000["satisfaction_rate"],
        eval_1000["normalized_energy"],
        artifact["honest_verdict"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
