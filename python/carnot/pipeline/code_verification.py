"""Packaged code-verification helpers for end users.

This module exposes a small, stable wrapper around the generated-code
verification path so CLI, MCP, and Python callers can share the same additive
static + PBT behavior.

Spec: REQ-CODE-019, REQ-CODE-010, SCENARIO-CODE-016
"""

from __future__ import annotations

from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


def verify_code(
    code: str,
    *,
    entry_point: str,
    prompt: str | None = None,
    official_tests: str = "",
    include_static: bool = True,
    include_pbt: bool = True,
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

    Returns:
        Pipeline-compatible ``VerificationResult`` with additive PBT metadata in
        ``certificate["pbt_summary"]``.
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
    )
