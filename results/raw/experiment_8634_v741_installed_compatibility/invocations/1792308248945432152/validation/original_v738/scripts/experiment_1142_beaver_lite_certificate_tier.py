#!/usr/bin/env python3
"""Run Exp 1142: BEAVER-lite arithmetic certificate tier.

Spec: REQ-VERIFY-1142, SCENARIO-VERIFY-1142
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_lite import (  # noqa: E402
    BEAVERLiteBounder,
    LogprobProvider,
    build_experiment_artifact,
    write_experiment_artifact,
)


OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1142_beaver_lite_certificate_tier.json"
SAMPLE_QUESTION = "If Alice has 3 apples and Bob gives her 4, how many does she have?"


def run_experiment(
    output_path: Path = OUTPUT_PATH,
    provider: LogprobProvider | None = None,
) -> dict[str, object]:
    """Run the bounder once and write the required Exp 1142 artifact."""

    bounder = BEAVERLiteBounder(provider=provider, top_k=50)
    result = bounder.bound_prefix_violation(SAMPLE_QUESTION)
    artifact = build_experiment_artifact(result)
    write_experiment_artifact(artifact, output_path)
    return artifact


def main() -> int:
    """CLI entry point for the conductor."""

    artifact = run_experiment()
    print(
        "[exp1142] "
        f"honest_verdict={artifact['honest_verdict']} "
        f"unsafe_mass_bound={artifact['unsafe_mass_bound']:.6f} "
        f"empirical_violation_rate={artifact['empirical_violation_rate']:.6f} "
        f"bound_gap={artifact['bound_gap']:.6f} "
        f"mock_logprobs_used={artifact['mock_logprobs_used']} "
        f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
