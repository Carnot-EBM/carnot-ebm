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

Spec: REQ-INFER-003, REQ-INFER-004, REQ-INFER-005, REQ-INFER-006, REQ-CODE-004
"""

from carnot.inference.code_verifier import (
    CodeVerificationResult,
    CodeVerifierConfig,
    compare_learned_vs_handcoded_code,
    generate_code_training_data,
    train_code_verifier,
    verify_python_function,
)
from carnot.inference.learned_verifier import (
    ComparisonResult,
    LearnedEnergyWrapper,
    LearnedVerifierConfig,
    build_learned_sat_energy,
    compare_learned_vs_handcoded,
    train_sat_verifier,
)
from carnot.inference.llm_solver import (
    LLMSolverConfig,
    run_llm_coloring_experiment,
    run_llm_sat_experiment,
    solve_coloring_with_llm,
    solve_sat_with_llm,
)
from carnot.inference.verify_and_repair import (
    VerifyRepairResult,
    parse_llm_coloring,
    parse_llm_sat_assignment,
    verify_and_repair,
)

__all__ = [
    "CodeVerificationResult",
    "CodeVerifierConfig",
    "ComparisonResult",
    "LLMSolverConfig",
    "LearnedEnergyWrapper",
    "LearnedVerifierConfig",
    "VerifyRepairResult",
    "build_learned_sat_energy",
    "compare_learned_vs_handcoded",
    "compare_learned_vs_handcoded_code",
    "generate_code_training_data",
    "parse_llm_coloring",
    "parse_llm_sat_assignment",
    "run_llm_coloring_experiment",
    "run_llm_sat_experiment",
    "solve_coloring_with_llm",
    "solve_sat_with_llm",
    "train_code_verifier",
    "train_sat_verifier",
    "verify_and_repair",
    "verify_python_function",
]
