#!/usr/bin/env python3
"""Run Exp 1158: BEAVER-lite live-or-Zipf logprob certificates.

Spec: REQ-VERIFY-1158, SCENARIO-VERIFY-1158
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_lite import LogprobProvider  # noqa: E402
from carnot.verify.beaver_lite_live import (  # noqa: E402
    run_beaver_lite_live_logprob_experiment,
)


OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1158_beaver_lite_live_logprobs.json"


def run_experiment(
    output_path: str | Path = OUTPUT_PATH,
    llama_cpp_available_override: bool | None = None,
    model_path: str | None = None,
    live_provider_factory: object | None = None,
    top_k: int = 10,
    max_tokens: int = 8,
) -> dict[str, object]:
    """Run the Exp 1158 workflow and write the requested artifact."""

    return run_beaver_lite_live_logprob_experiment(
        output_path=output_path,
        llama_cpp_available_override=llama_cpp_available_override,
        model_path=model_path,
        live_provider_factory=live_provider_factory,  # type: ignore[arg-type]
        top_k=top_k,
        max_tokens=max_tokens,
    )


def main() -> int:
    """CLI entry point for the conductor."""

    artifact = run_experiment()
    print(
        "[exp1158] "
        f"honest_verdict={artifact['honest_verdict']} "
        f"unsafe_mass_bound_live={artifact['unsafe_mass_bound_live']:.6f} "
        f"empirical_violation_rate_live={artifact['empirical_violation_rate_live']:.6f} "
        f"mock_logprobs_used={artifact['mock_logprobs_used']} "
        f"zipf_mock_used={artifact['zipf_mock_used']} "
        f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
