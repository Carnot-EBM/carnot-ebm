"""LLM-EBM inference: verify and repair LLM output using energy-based models.

**Researcher summary:**
    Bridges LLM text output to EBM verification. Parses LLM answers,
    scores them against constraint energy, repairs violations via gradient
    descent, and issues verification certificates.

**Detailed explanation for engineers:**
    This package implements the "anti-hallucination" pipeline:

    1. Parse LLM text output into a JAX array (constraint configuration)
    2. Score it against a ComposedEnergy (total constraint violations)
    3. If violations detected, run gradient repair on violated constraints
    4. Round to discrete domain (binary for SAT, integer for coloring)
    5. Re-verify and issue a certificate

    The constraint definitions live in ``carnot.verify`` (sat.py,
    graph_coloring.py). This package provides the glue and benchmark
    harness.

Spec: REQ-INFER-003, REQ-INFER-004, REQ-INFER-005
"""

from carnot.inference.verify_and_repair import (
    VerifyRepairResult,
    parse_llm_coloring,
    parse_llm_sat_assignment,
    verify_and_repair,
)

__all__ = [
    "VerifyRepairResult",
    "parse_llm_coloring",
    "parse_llm_sat_assignment",
    "verify_and_repair",
]
