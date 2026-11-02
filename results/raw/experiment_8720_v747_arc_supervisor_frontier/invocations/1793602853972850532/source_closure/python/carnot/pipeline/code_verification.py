"""Packaged code-verification helpers for end users.

This module exposes a small, stable wrapper around the generated-code
verification path so CLI, MCP, and Python callers can share the same additive
static + PBT behavior.

Spec: REQ-CODE-019, REQ-CODE-010, SCENARIO-CODE-016
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike


def verify_code(
    code: str,
    *,
    entry_point: str,
    prompt: str | None = None,
    official_tests: str = "",
    include_static: bool = True,
    include_pbt: bool = True,
    include_specs: bool = False,
    task_id: str | None = None,
    spec_corpus_path: str | PathLike[str] | None = None,
    trace_paths: Sequence[str | PathLike[str]] | None = None,
) -> VerificationResult:
    """Verify packaged Python code with the additive generated-code path.

    Args:
        code: Python source code containing the candidate function.
        entry_point: Function name to verify within ``code``.
        prompt: Optional HumanEval-style prompt or signature context. When not
            provided, the source code is reused so signature-derived PBT checks
            still run.
        official_tests: Optional official test harness for prompt-implied PBT.
        include_static: Whether to include static code constraints.
        include_pbt: Whether to include the Hypothesis-backed verifier.
        include_specs: Whether to opt in to the explicit spec-aware verifier.
        task_id: Optional checked-in task id used to resolve an explicit spec row.
        spec_corpus_path: Optional override for the checked-in explicit spec corpus.
        trace_paths: Optional override for the checked-in trace-learning artifacts.

    Returns:
        Pipeline-compatible ``VerificationResult`` with additive PBT metadata in
        ``certificate["pbt_summary"]`` and, when enabled, explicit-spec
        metadata in ``certificate["spec_summary"]``.
    """
    normalized_prompt = prompt if prompt and prompt.strip() else code
    pipeline = VerifyRepairPipeline()
    return pipeline.verify_generated_code(
        code,
        normalized_prompt,
        entry_point,
        official_tests,
        include_static=include_static,
        include_pbt=include_pbt,
        include_specs=include_specs,
        task_id=task_id,
        spec_corpus_path=spec_corpus_path,
        trace_paths=trace_paths,
    )
