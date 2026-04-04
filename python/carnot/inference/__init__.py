"""LLM-EBM inference: verify and repair LLM output using energy-based models.

**Researcher summary:**
    Bridges LLM text output to EBM verification. Parses LLM answers,
    scores them against constraint energy, repairs violations via gradient
    descent, and issues verification certificates. Includes semantic energy
    hallucination detection, multi-start repair, ARM-EBM bijection, reasoning
    energy, and diffusion generation.

**Detailed explanation for engineers:**
    This package implements the "anti-hallucination" pipeline plus research
    extensions from arxiv papers (P1-P9 in the roadmap).

Spec: REQ-INFER-003 through REQ-INFER-012, REQ-CODE-004
"""

from carnot.inference.arm_ebm_bridge import (
    TokenEnergyAnalysis,
    analyze_token_energy,
    compute_sequence_energy,
    extract_token_rewards,
    extract_token_rewards_from_logprobs,
    identify_hallucination_tokens,
)
from carnot.inference.code_verifier import (
    CodeVerificationResult,
    CodeVerifierConfig,
    compare_learned_vs_handcoded_code,
    generate_code_training_data,
    train_code_verifier,
    verify_python_function,
)
from carnot.inference.diffusion import (
    DiffusionConfig,
    DiffusionResult,
    diffusion_generate,
    diffusion_generate_coloring,
    diffusion_generate_sat,
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
from carnot.inference.multi_start import (
    MultiStartResult,
    multi_start_repair,
)
from carnot.inference.reasoning_energy import (
    ReasoningEnergyResult,
    ReasoningVerifierConfig,
    train_reasoning_energy,
    verify_reasoning_chain,
)
from carnot.inference.semantic_energy import (
    SemanticEnergyResult,
    classify_hallucination,
    compute_semantic_energy,
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
    "DiffusionConfig",
    "DiffusionResult",
    "LLMSolverConfig",
    "LearnedEnergyWrapper",
    "LearnedVerifierConfig",
    "MultiStartResult",
    "ReasoningEnergyResult",
    "ReasoningVerifierConfig",
    "SemanticEnergyResult",
    "TokenEnergyAnalysis",
    "VerifyRepairResult",
    "analyze_token_energy",
    "build_learned_sat_energy",
    "classify_hallucination",
    "compare_learned_vs_handcoded",
    "compare_learned_vs_handcoded_code",
    "compute_semantic_energy",
    "compute_sequence_energy",
    "diffusion_generate",
    "diffusion_generate_coloring",
    "diffusion_generate_sat",
    "extract_token_rewards",
    "extract_token_rewards_from_logprobs",
    "generate_code_training_data",
    "identify_hallucination_tokens",
    "multi_start_repair",
    "parse_llm_coloring",
    "parse_llm_sat_assignment",
    "run_llm_coloring_experiment",
    "run_llm_sat_experiment",
    "solve_coloring_with_llm",
    "solve_sat_with_llm",
    "train_code_verifier",
    "train_reasoning_energy",
    "train_sat_verifier",
    "verify_and_repair",
    "verify_python_function",
    "verify_reasoning_chain",
]
