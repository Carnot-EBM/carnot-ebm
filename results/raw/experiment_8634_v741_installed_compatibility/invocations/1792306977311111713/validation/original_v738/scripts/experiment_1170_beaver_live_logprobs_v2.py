#!/usr/bin/env python3
"""Run Exp 1170: BEAVER-lite llama.cpp completion logprobs v2.

Spec: REQ-VERIFY-1170, SCENARIO-VERIFY-1170
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_lite_live import (  # noqa: E402
    run_beaver_live_logprobs_v2_experiment,
)


OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1170_beaver_live_logprobs_v2.json"
FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"


def run_experiment(
    output_path: str | Path = OUTPUT_PATH,
    fover_corpus_path: str | Path = FOVER_PATH,
    llama_cpp_available_override: bool | None = None,
    model_path: str | None = None,
    live_provider_factory: object | None = None,
    top_k: int = 1,
    max_tokens: int = 6,
) -> dict[str, object]:
    """Run Exp 1170 and write the requested artifact."""

    return run_beaver_live_logprobs_v2_experiment(
        output_path=output_path,
        fover_corpus_path=fover_corpus_path,
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
        "[exp1170] "
        f"honest_verdict={artifact['honest_verdict']} "
        f"mock_logprobs_used={artifact['mock_logprobs_used']} "
        f"logprobs_source={artifact['logprobs_source']} "
        f"bound_is_sound={artifact['bound_is_sound']} "
        f"n_test_prompts_run={artifact['n_test_prompts_run']} "
        f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
