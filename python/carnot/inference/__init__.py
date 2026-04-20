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

Spec: REQ-INFER-003 through REQ-INFER-013, REQ-CODE-004
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
from carnot.inference.composite_scorer import (
    CompositeEnergyConfig,
    CompositeEnergyScorer,
)
from carnot.inference.diffusion import (
    DiffusionConfig,
    DiffusionResult,
    diffusion_generate,
    diffusion_generate_coloring,
    diffusion_generate_sat,
)
from carnot.inference.dual_gpu import (
    DualGPUExecutionContext,
    DualGPUExecutionResult,
    DualGPURunner,
    estimate_model_size_billions,
    requires_device_map_auto,
)
from carnot.inference.ebm_loader import (
    KNOWN_MODELS,
    get_model_info,
    load_ebm,
)
from carnot.inference.ebm_rejection import (
    EBMCandidateScore,
    EBMRejectionConfig,
    EBMRejectionResult,
    ebm_rejection_sample,
    score_activations_with_ebm,
)
from carnot.inference.guided_decoding import (
    EnergyGuidedSampler,
    GuidedDecodingResult,
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
    RefinementResult,
    RejectionSampleResult,
    iterative_refine_code,
    iterative_refine_with_properties,
    logprob_rejection_sample,
    run_llm_coloring_experiment,
    run_llm_sat_experiment,
    solve_coloring_with_llm,
    solve_sat_with_llm,
)
from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    SotaModelSpec,
    default_pair,
    flagship_dense,
    flagship_moe,
)
from carnot.inference.model_loader import (
    ModelLoadError,
    ServerBackedModelHandle,
    clear_model_server,
    generate,
    load_model,
    register_model_server,
)
from carnot.inference.model_server import (
    ModelServer,
    WarmServerBenchmarkResult,
    benchmark_cold_load_vs_warm_server,
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
from carnot.inference.tensorrt_backend import (
    TRTBackendStatus,
    TRTLLMBackend,
    TRTLLMBenchmarkResult,
    benchmark_huggingface_vs_tensorrt,
    load_trt_backend,
)
from carnot.inference.jepa_pure_loss import (
    JEPAChainScore,
    PUREMinFormLoss,
    pairs_to_pure_chains,
)
from carnot.inference.jepa_cpmi_pairs import (
    JEPACPMIPair,
    JEPACPMIPairBuilder,
    CPMIContrastiveLoss,
)
from carnot.inference.verify_and_repair import (
    VerifyRepairResult,
    parse_llm_coloring,
    parse_llm_sat_assignment,
    verify_and_repair,
)

__all__ = [
    "JEPAChainScore",
    "PUREMinFormLoss",
    "pairs_to_pure_chains",
    "JEPACPMIPair",
    "JEPACPMIPairBuilder",
    "CPMIContrastiveLoss",
    "EnergyGuidedSampler",
    "GuidedDecodingResult",
    "CodeVerificationResult",
    "CodeVerifierConfig",
    "ComparisonResult",
    "CompositeEnergyConfig",
    "CompositeEnergyScorer",
    "EBMCandidateScore",
    "EBMRejectionConfig",
    "EBMRejectionResult",
    "DualGPUExecutionContext",
    "DualGPUExecutionResult",
    "DualGPURunner",
    "KNOWN_MODELS",
    "DiffusionConfig",
    "DiffusionResult",
    "RefinementResult",
    "LLMSolverConfig",
    "iterative_refine_code",
    "iterative_refine_with_properties",
    "LearnedEnergyWrapper",
    "LearnedVerifierConfig",
    "MultiStartResult",
    "ReasoningEnergyResult",
    "ReasoningVerifierConfig",
    "SemanticEnergyResult",
    "ServerBackedModelHandle",
    "TokenEnergyAnalysis",
    "TRTBackendStatus",
    "TRTLLMBackend",
    "TRTLLMBenchmarkResult",
    "ModelLoadError",
    "ModelServer",
    "VerifyRepairResult",
    "WarmServerBenchmarkResult",
    "analyze_token_energy",
    "benchmark_cold_load_vs_warm_server",
    "build_learned_sat_energy",
    "classify_hallucination",
    "clear_model_server",
    "ebm_rejection_sample",
    "benchmark_huggingface_vs_tensorrt",
    "generate",
    "get_model_info",
    "compare_learned_vs_handcoded",
    "compare_learned_vs_handcoded_code",
    "compute_semantic_energy",
    "compute_sequence_energy",
    "diffusion_generate",
    "diffusion_generate_coloring",
    "diffusion_generate_sat",
    "estimate_model_size_billions",
    "extract_token_rewards",
    "extract_token_rewards_from_logprobs",
    "generate_code_training_data",
    "identify_hallucination_tokens",
    "multi_start_repair",
    "parse_llm_coloring",
    "parse_llm_sat_assignment",
    "RejectionSampleResult",
    "run_llm_coloring_experiment",
    "load_ebm",
    "load_model",
    "logprob_rejection_sample",
    "score_activations_with_ebm",
    "run_llm_sat_experiment",
    "requires_device_map_auto",
    "solve_coloring_with_llm",
    "solve_sat_with_llm",
    "register_model_server",
    "load_trt_backend",
    "train_code_verifier",
    "train_reasoning_energy",
    "train_sat_verifier",
    "verify_and_repair",
    "verify_python_function",
    "verify_reasoning_chain",
    "SOTA_GGUF_MODELS",
    "SotaModelSpec",
    "default_pair",
    "flagship_dense",
    "flagship_moe",
]
