"""Constraint extraction pipeline for automated verification.

Extracts verifiable constraints from text, code, and natural language,
then maps them to energy terms for Ising-model verification.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
"""

from carnot.pipeline.activation_jailbreak_probe import (  # noqa: E402
    ActivationJailbreakProbe,
    ProbeMetadata,
)
from carnot.pipeline.adaptive_thresholds import (
    ModelAdaptiveThresholds,
    PerModelFPTracker,
    SelectiveConsolidation,
)
from carnot.pipeline.adaptrack_repairer import (  # noqa: E402
    AdapTrackRepairer,
    BacktrackEvent,
)
from carnot.pipeline.agentic import (
    AgentStep,
    ConstraintState,
    FactStatus,
    TrackedFact,
    propagate,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.batching_audit import (  # noqa: E402
    BatchingEnforcementAudit,
    BatchingViolation,
)
from carnot.pipeline.batching_hook_runner import BatchingHookRunner  # noqa: E402
from carnot.pipeline.boltzmann_repair import (  # noqa: E402
    BoltzmannRepairBridge,
    LinearSpinAdapter,
    RepairDirection,
)
from carnot.pipeline.capo_calibration import CAPOCalibrationLoss  # noqa: E402
from carnot.pipeline.case_memory import (
    CaseEntry,
    CaseKey,
    CaseMatch,
    CaseMemory,
    CaseProvenance,
    CaseQuery,
    CaseRecord,
)
from carnot.pipeline.causal_reasoning_verifier import (  # noqa: E402
    CausalEntailmentResult,
    CausalReasoningVerifier,
)
from carnot.pipeline.code_learning import (
    CodeVerificationTrace,
    LearningCurvePoint,
    LearningImprovement,
    ProblemTypeScore,
    PropertyFailureTrace,
    PropertyRanker,
    PropertyScore,
    RepairStrategy,
    StrategyRecommendation,
    TraceAnalysis,
    TraceAnalyzer,
    VerificationTraceStep,
)
from carnot.pipeline.code_verification import verify_code
from carnot.pipeline.conductor_dedup import (  # noqa: E402
    ConductorDedupCheck,
    PartialResultHandoff,
)
from carnot.pipeline.confidence_weighted_repair import (
    ConfidenceRepairResult,
    ConfidenceWeightedRepair,
    ViolationConfidence,
    compute_energy_variance_confidence,
    compute_expression_confidence,
)
from carnot.pipeline.constraint_addition import (  # noqa: E402
    ConstraintAdditionFromMemory,
    ViolationPattern,
)
from carnot.pipeline.constraint_template_library import (
    ConstraintTemplate,
    ConstraintTemplateLibrary,
    DEFAULT_MANIPULABILITY_PRIORS,
    carry_check_template,
    comparison_direction_template,
    manipulable_signal_dependency_template,
    sign_check_template,
    unit_consistency_template,
)
from carnot.pipeline.cot_circuit_verifier import (
    CoTCircuit,
    CoTCircuitVerifier,
    CoTStep,
    build_circuit,
    extract_cot_steps,
    find_broken_links,
)
from carnot.pipeline.cpmi_builder import (  # noqa: E402
    CPMIContrastivePairBuilder,
    CPMITriple,
    compute_cpmi_score,
    generate_hard_negative,
)
from carnot.pipeline.cross_session_relay import (  # noqa: E402
    CrossSessionResult,
    compute_relay_verdict,
    simulate_session,
)
from carnot.pipeline.deliverable_guard import (  # noqa: E402
    DeliverableGuard,
    DocOnlyClassifier,
)
from carnot.pipeline.deliverable_validator import (  # noqa: E402
    CloudGPUInstructions,
    DeliverableContentValidator,
    build_cloud_gpu_instructions,
    generate_cloud_gpu_script,
)
from carnot.pipeline.dsvd_adapter import (  # noqa: E402
    DSVDAdapter,
    DSVDLinearProbe,
    DSVDProbeResult,
)
from carnot.pipeline.dsvd_live_trainer import (  # noqa: E402
    DSVDLiveTrainer,
    DSVDLiveTrainPair,
    TemporalWindowLabeler,
)
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner  # noqa: E402
from carnot.pipeline.dual_gpu_harness import (  # noqa: E402
    AuditFinding,
    DualGPUHarness,
    HarnessAudit,
)
from carnot.pipeline.dual_gpu_health import (  # noqa: E402
    DualGPUHealthResult,
    build_gpu_fix_artifact,
    check_dual_gpu_health,
)
from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor, GPUProcessInfo
from carnot.pipeline.dualgpu_retrain import (  # noqa: E402
    DualGPURetrain,
    DualGPURetrainConfig,
)
from carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    ConstraintSPOTuple,
    EmbeddingConstraintStore,
)
from carnot.pipeline.energy_guided_decoder import (  # noqa: E402
    EnergyGuidedConfig,
    EnergyGuidedDecoder,
)
from carnot.pipeline.energy_magnitude_replay import (  # noqa: E402
    EnergyMagnitudeBuffer,
    EnergyMagnitudeReplay,
)
from carnot.pipeline.ensemble_gate_v3 import (  # noqa: E402
    EnsembleGateV3Result,
    EnsembleRecallGateV3,
)
from carnot.pipeline.env_autofix import (  # noqa: E402
    EnvironmentAutoFix,
    apply_env_autofix,
    build_env_autofix_artifact,
)
from carnot.pipeline.eorm_rectifier import (  # noqa: E402
    EORMAdaptiveRectifier,
    RectifierResult,
)
from carnot.pipeline.errors import (
    CarnotError,
    ExtractionError,
    ModelLoadError,
    PipelineTimeoutError,
    RepairError,
    VerificationError,
)
from carnot.pipeline.exclusion_manifest import (  # noqa: E402
    ExclusionEntry,
    ExclusionManifest,
    build_default_manifest,
)
from carnot.pipeline.expanded_gpu_reaper import (  # noqa: E402
    ExpandedGPUReaper,
    ExpandedGPUReaperConfig,
    ExpandedGPUReapResult,
)
from carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutResult,
    ExperimentTimeoutWatchdog,
    build_timeout_artifact,
    get_timeout_minutes,
)
from carnot.pipeline.extract import (
    ArithmeticExtractor,
    AutoExtractor,
    CodeExtractor,
    ConstraintExtractor,
    ConstraintResult,
    LogicExtractor,
    NLExtractor,
)
from carnot.pipeline.fact_e_probe import (  # noqa: E402
    CausalStepDependency,
    FACTEFaithfulnessProbe,
)
from carnot.pipeline.flip_calibrator import (  # noqa: E402
    FLIPRepairTriple,
    FLIPRewardCalibrator,
)
from carnot.pipeline.fover_annotator import (  # noqa: E402
    FOVERAnnotator,
    FOVERCoTStep,
    annotate_step_with_z3,
    parse_cot_into_steps,
)
from carnot.pipeline.fover_corpus import (  # noqa: E402
    FOVERCorpusEntry,
    balance_corpus,
    compute_corpus_diversity,
    merge_fover_sources,
)
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader  # noqa: E402
from carnot.pipeline.gemma_isolation import (  # noqa: E402
    VRAMEvictionResult,
    evict_gpu_vram,
    load_gemma4_on_gpu1,
)
from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: E402
from carnot.pipeline.gguf_cache import (  # noqa: E402
    GGUFCacheConfig,
    GGUFCacheResolver,
    GGUFModelNotFoundError,
    resolve_gguf_path,
)
from carnot.pipeline.gpu_thermal_gate import (  # noqa: E402
    GPUThermalGate,
    GPUThermalThrottleError,
    ThermalStatus,
)
from carnot.pipeline.gpu_vram_gate import (  # noqa: E402
    GPUVRAMGate,
    GPUVRAMInsufficientError,
    VRAMStatus,
)
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2  # noqa: E402
from carnot.pipeline.gpu_zombie_fix import (  # noqa: E402
    ZombieFixResult,
    build_zombie_fix_artifact,
    build_zombie_fix_strategy,
)
from carnot.pipeline.gpu_zombie_killer import (  # noqa: E402
    GPUZombieResult,
    get_gpu_memory_pids,
    kill_gpu_zombies,
)
from carnot.pipeline.hallucination_basin import (  # noqa: E402
    BasinEstimate,
    HallucinationBasinDetector,
    estimate_basin_depth,
)
from carnot.pipeline.hallufield_detector import (  # noqa: E402
    HalluFieldDetector,
    HalluFieldResult,
)
from carnot.pipeline.halp_probe import (  # noqa: E402
    HALPProbe,
    HALPProbeResult,
)
from carnot.pipeline.hardware_energy_probe import (  # noqa: E402
    EORMHardwareCorrelation,
    HardwareEnergyProbe,
    HardwareEnergyReading,
    compute_eorm_hardware_correlation,
)
from carnot.pipeline.harness_patcher import (  # noqa: E402
    HarnessPatcher,
    HarnessPatchResult,
)
from carnot.pipeline.hermes_adapter import (  # noqa: E402
    HermesVerificationStep,
    HermesVerifierAdapter,
)
from carnot.pipeline.hermes_v2_live_loop import (  # noqa: E402
    HermesV2GenerationResult,
    HermesV2LiveLoop,
    HermesV2StepResult,
)
from carnot.pipeline.hermes_v2_structured_loop import (  # noqa: E402
    HermesV2StructuredLoop,
    HermesV2StructuredResult,
)
from carnot.pipeline.hisr_weights import (  # noqa: E402
    HISRViolationWeight,
    HISRWeighter,
)
from carnot.pipeline.interleaved_verifier import (  # noqa: E402
    InterleavedLogicVerifier,
    InterleavedStepResult,
)
from carnot.pipeline.interwhen_monitor import (  # noqa: E402
    InterWhenMonitor,
    InterWhenViolation,
)
from carnot.pipeline.ising_constraint_injector import (  # noqa: E402
    ConstraintInjectionResult,
    ExternalFieldEnergyResult,
    IsingConstraintInjector,
)
from carnot.pipeline.jepa_fast_path import JepaGate
from carnot.pipeline.jepa_wiring_guard import (  # noqa: E402
    JepaWiringCheckResult,
    check_cpmi_wiring,
)
from carnot.pipeline.jit_vram_check import (  # noqa: E402
    JITVRAMCheck,
    JITVRAMResult,
)
from carnot.pipeline.latent_cot_calibrator import (  # noqa: E402
    LatentCoTCalibrationResult,
    LatentCoTEBMCalibrator,
)
from carnot.pipeline.live_assertion import (  # noqa: E402
    assert_live_gpu_available,
    assert_live_or_ci_skip,
)
from carnot.pipeline.live_gpu_gate import (  # noqa: E402
    LiveGPUGate,
    build_session_startup_script,
    check_session_startup_exists,
)
from carnot.pipeline.llm_extractor import LLMConstraintExtractor
from carnot.pipeline.llm_z3_formalizer import (
    LLMz3Formalizer,
    Z3FormalizationResult,
    build_z3_formalization_prompt,
    parse_z3_snippet,
)
from carnot.pipeline.long_run_executor import (  # noqa: E402
    BenchmarkBatch,
    LongRunBenchmarkExecutor,
    LongRunBenchmarkResult,
    get_batch_size,
)
from carnot.pipeline.lsebm_replayer import (  # noqa: E402
    LSEBMConstraintReplayer,
    ViolationDistribution,
)
from carnot.pipeline.lsebmcl_replay import (  # noqa: E402
    LSEBMCLReplayBuffer,
    ReplaySession,
)
from carnot.pipeline.lw_jepa_trainer import (  # noqa: E402
    LeWorldModelJEPATrainer,
    LeWorldModelLoss,
    gaussian_kl_regularization,
)
from carnot.pipeline.manifest_enforcer import ExclusionManifestEnforcer  # noqa: E402
from carnot.pipeline.mars_margin_gate import (  # noqa: E402
    MARSMarginGate,
    MARSMarginResult,
    compute_logit_margin,
)
from carnot.pipeline.memory import PATTERN_THRESHOLD, ConstraintMemory
from carnot.pipeline.metajuls_adapter import (  # noqa: E402
    ExtractorPolicy,
    MetaJuLSAdapter,
)
from carnot.pipeline.mining import (
    FailureAnalyzer,
    FailureReport,
    FalseNegative,
)
from carnot.pipeline.mise_calibrator import (  # noqa: E402
    MISECalibrator,
    MISETriple,
)
from carnot.pipeline.multi_agent_arbiter import (  # noqa: E402
    AgentScore,
    ArbiterResult,
    MultiAgentArbiter,
)
from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor, Z3Result
from carnot.pipeline.npu_entropy_probe import (  # noqa: E402
    NPUBenchmarkResult,
    NPUEntropyProbe,
)
from carnot.pipeline.nup_probe import (  # noqa: E402
    ContinuationEntropy,
    NUPProbe,
    NUPProbeResult,
)
from carnot.pipeline.nup_probe import (
    score_with_latency as nup_score_with_latency,
)
from carnot.pipeline.nup_probe_v2 import (  # noqa: E402
    BayesianEntropyEstimator,
    EntropyEstimate,
    NUPProbeV2,
    NUPProbeV2Result,
)
from carnot.pipeline.nup_probe_v3 import (  # noqa: E402
    CLAPFeatureExtractor,
    CLAPFeatures,
    NUPProbeV3,
)
from carnot.pipeline.nup_probe_v4 import (  # noqa: E402
    ContrastivePairLoss,
    NUPProbeV4,
)
from carnot.pipeline.oracle_corpus_builder import (  # noqa: E402
    OracleChain,
    OracleCorpusBuilder,
    StepLabel,
)
from carnot.pipeline.otv_verifier import (  # noqa: E402
    OTVVerificationToken,
    OTVVerifier,
)
from carnot.pipeline.pbt_code_verifier import (
    PBTCodeVerificationResult,
    PBTCodeVerifier,
    PBTDerivedProperty,
    PBTPropertyFailure,
)
from carnot.pipeline.pps_constraint_learner import (  # noqa: E402
    ConstraintDomain,
    DomainParameterPartition,
    PartitionIsolationScore,
    PPSConstraintLearner,
)
from carnot.pipeline.ppsebm_real_validator import (  # noqa: E402
    InterleavedViolationSequence,
    PPSEBMRealValidationResult,
)
from carnot.pipeline.pra_eorm_beam import (  # noqa: E402
    PRABeamCandidate,
    PRABeamResult,
    PRAEBMBeamSearch,
)
from carnot.pipeline.precision_100q_v6_result import Precision100qV6Result  # noqa: E402
from carnot.pipeline.prefill_uncertainty_probe import (
    PrefillUncertaintyProbe,
    PrefillUncertaintyResult,
    compute_conjugate_bound,
    compute_input_uncertainty,
    compute_prompt_uncertainty,
)
from carnot.pipeline.property_code_verifier import (
    DerivedProperty,
    PropertyCodeVerificationResult,
    PropertyCodeVerifier,
    PropertyFailure,
    extract_official_test_examples,
    extract_prompt_examples,
)
from carnot.pipeline.saver_verifier import (
    AgentStep as SAVeRStep,
)
from carnot.pipeline.saver_verifier import (
    ConstraintState as SAVeRConstraintState,
)
from carnot.pipeline.saver_verifier import (
    SAVeRVerifier,
    build_saver_artifact,
)
from carnot.pipeline.self_learning_policy import (
    PolicyProvenance,
    PolicyQuery,
    PropertyBudgetUpdate,
    RepairPromptPatch,
    RoutingHint,
    RuntimePolicyContext,
    SelfLearningPolicy,
    SelfLearningPolicyCompiler,
    ThresholdOverride,
)
from carnot.pipeline.self_learning_relay import (
    SelfLearningBatchResult,
    SelfLearningRelay,
    build_relay_artifact,
    compute_learning_improvement,
)
from carnot.pipeline.semantic_energy_boltzmann import (  # noqa: E402
    BoltzmannSemanticEnergy,
    SemanticCluster,
)
from carnot.pipeline.semantic_energy_extractor import (
    DualEnergyGate,
    DualEnergyResult,
    SemanticEnergyExtractor,
    SemanticEnergyResult,
    compute_semantic_energy,
)
from carnot.pipeline.semantic_grounding import (
    PromptClause,
    QuestionProfile,
    SemanticClaim,
    SemanticGroundingResult,
    SemanticGroundingVerifier,
    SemanticGroundingViolation,
    verify_semantic_grounding,
)
from carnot.pipeline.semantic_verifier_v2 import (
    SemanticClaimResult,
    SemanticVerifierV2,
    SemanticVerifierV2Result,
    SemanticVerifierV2Thresholds,
    verify_semantic_verifier_v2,
)
from carnot.pipeline.session_health_check import (  # noqa: E402
    ConductorSessionHealthCheck,
    GPUHealth,
    SessionHealthResult,
    ZombieProcess,
)
from carnot.pipeline.session_memory import SessionMemory
from carnot.pipeline.session_memory_pack import (
    diff_session_memory_packs,
    export_session_memory,
    import_session_memory,
    load_session_memory_pack,
    validate_session_memory_pack,
)
from carnot.pipeline.sink_probe import (
    SinkConcentration,
    SinkProbe,
    SinkProbeResult,
    SinkTokenType,
    compute_sink_concentration,
)
from carnot.pipeline.specguard_verifier import (  # noqa: E402
    SpecGuardStepResult,
    SpecGuardVerifier,
)
from carnot.pipeline.spilled_energy import (  # noqa: E402
    SpilledEnergyDetector,
    SpilledEnergyDetectorResult,
    SpilledEnergyToken,
    compute_detector_spilled_energy,
)
from carnot.pipeline.spilled_energy_extractor import (
    SpilledEnergyExtractor,
    SpilledEnergyResult,
    compute_lookahead_energy,
    compute_spilled_energy,
)
from carnot.pipeline.structured_equation_forcer import (  # noqa: E402
    FORCER_SYSTEM_ADDENDUM,
    ForcedEquationResult,
    StructuredEquationForcer,
)
from carnot.pipeline.structured_reasoning import (
    StructuredReasoningAttempt,
    StructuredReasoningController,
    StructuredReasoningEmission,
    build_gemma_structured_reasoning_prompt,
    build_qwen_structured_reasoning_prompt,
    load_monitorability_policy,
)
from carnot.pipeline.sure_priority_replay import (  # noqa: E402
    SuRePriorityReplay,
    SuReReplayResult,
    ViolationSurprise,
)
from carnot.pipeline.symcode_verifier import (  # noqa: E402
    SymCodeVerifier,
)
from carnot.pipeline.think_probe import (  # noqa: E402
    CarnotThinkProbe,
    ThinkProbeResult,
    ThinkVerdict,
    build_think_probe_prompt,
    parse_think_probe_output,
)
from carnot.pipeline.think_probe_v2 import ThinkProbeV2, ThinkProbeV2Result  # noqa: E402
from carnot.pipeline.three_tier_pipeline import (
    ThreeTierPipeline,
    ThreeTierPipelineResult,
    build_three_tier_artifact,
)
from carnot.pipeline.typed_reasoning import (
    AtomicClaim,
    ExtractionProvenance,
    FinalAnswer,
    ReasoningStep,
    TypedReasoningExtractor,
    TypedReasoningIR,
    UserConstraint,
    extract_typed_reasoning,
)
from carnot.pipeline.verge_refiner import (
    VergeIteration,
    VergeRefiner,
    build_step_repair_prompt,
    extract_failed_assertion,
)
from carnot.pipeline.verdict_record import (
    VerdictCalibration,
    VerdictRecord,
    calibrated_confidence_from_energy,
    fit_verdict_calibration,
)
from carnot.pipeline.verify_stream import (
    VerifyStreamCandidate,
    collect_verify_stream,
    verify_stream,
)
from carnot.pipeline.verify_repair import (
    RepairResult,
    VerificationResult,
    VerifyRepairPipeline,
)
from carnot.pipeline.vg_search_scheduler import (  # noqa: E402
    VGScheduleResult,
    VGSearchScheduler,
)
from carnot.pipeline.vram_budget_ledger import (  # noqa: E402
    VRAMBudgetLedger,
    VRAMForecast,
)
from carnot.pipeline.vram_loop_eviction import (  # noqa: E402
    VRAMLoopEvictionResult,
    evict_vram_with_loop,
)
from carnot.pipeline.z3_extractor import Z3ArithmeticExtractor
from carnot.pipeline.z3_gated_repair import Z3GatedRepair, Z3GatedRepairResult, compute_skip_rate

__all__ = [
    "AgentStep",
    "ArithmeticExtractor",
    "AtomicClaim",
    "AutoExtractor",
    "CarnotError",
    "CaseEntry",
    "CaseKey",
    "CaseMatch",
    "CaseMemory",
    "CaseProvenance",
    "CaseQuery",
    "CaseRecord",
    "ConstraintMemory",
    "ConstraintState",
    "CodeVerificationTrace",
    "CodeExtractor",
    "LearningCurvePoint",
    "LearningImprovement",
    "ConstraintExtractor",
    "ConstraintResult",
    "ExtractionError",
    "ExtractionProvenance",
    "FactStatus",
    "FailureAnalyzer",
    "FailureReport",
    "FalseNegative",
    "FinalAnswer",
    "LogicExtractor",
    "LLMConstraintExtractor",
    "ModelLoadError",
    "NLExtractor",
    "PATTERN_THRESHOLD",
    "PrefillUncertaintyProbe",
    "PrefillUncertaintyResult",
    "PBTCodeVerificationResult",
    "PBTCodeVerifier",
    "PBTDerivedProperty",
    "PBTPropertyFailure",
    "PipelineTimeoutError",
    "PolicyProvenance",
    "PolicyQuery",
    "ProblemTypeScore",
    "PropertyBudgetUpdate",
    "PropertyFailureTrace",
    "PropertyRanker",
    "PropertyScore",
    "PromptClause",
    "PropertyCodeVerificationResult",
    "PropertyCodeVerifier",
    "PropertyFailure",
    "RepairPromptPatch",
    "DerivedProperty",
    "QuestionProfile",
    "RepairError",
    "RepairResult",
    "ReasoningStep",
    "RepairStrategy",
    "RoutingHint",
    "RuntimePolicyContext",
    "SemanticClaim",
    "SemanticClaimResult",
    "SemanticGroundingResult",
    "SemanticGroundingVerifier",
    "SemanticGroundingViolation",
    "SemanticVerifierV2",
    "SemanticVerifierV2Result",
    "SemanticVerifierV2Thresholds",
    "StrategyRecommendation",
    "StructuredReasoningAttempt",
    "StructuredReasoningController",
    "StructuredReasoningEmission",
    "TraceAnalysis",
    "TraceAnalyzer",
    "TrackedFact",
    "ThresholdOverride",
    "TypedReasoningExtractor",
    "TypedReasoningIR",
    "UserConstraint",
    "VerificationError",
    "VerificationResult",
    "VerificationTraceStep",
    "VerdictCalibration",
    "VerdictRecord",
    "VerifyStreamCandidate",
    "VerifyRepairPipeline",
    "collect_verify_stream",
    "verify_stream",
    "SelfLearningPolicy",
    "SelfLearningPolicyCompiler",
    "DualEnergyGate",
    "DualEnergyResult",
    "SemanticEnergyExtractor",
    "SemanticEnergyResult",
    "SpilledEnergyExtractor",
    "SpilledEnergyResult",
    "Z3ArithmeticExtractor",
    "Z3GatedRepair",
    "Z3GatedRepairResult",
    "Z3Result",
    "JepaGate",
    "NL2Z3Extractor",
    "LLMz3Formalizer",
    "Z3FormalizationResult",
    "build_z3_formalization_prompt",
    "parse_z3_snippet",
    "compute_skip_rate",
    "ConfidenceRepairResult",
    "ConfidenceWeightedRepair",
    "ViolationConfidence",
    "compute_energy_variance_confidence",
    "compute_expression_confidence",
    "DualGPUMonitor",
    "GPUProcessInfo",
    "ModelAdaptiveThresholds",
    "PerModelFPTracker",
    "SelectiveConsolidation",
    "ConstraintTemplate",
    "ConstraintTemplateLibrary",
    "DEFAULT_MANIPULABILITY_PRIORS",
    "carry_check_template",
    "comparison_direction_template",
    "manipulable_signal_dependency_template",
    "sign_check_template",
    "unit_consistency_template",
    "VergeIteration",
    "VergeRefiner",
    "build_step_repair_prompt",
    "extract_failed_assertion",
    "CoTCircuit",
    "CoTCircuitVerifier",
    "CoTStep",
    "build_circuit",
    "extract_cot_steps",
    "find_broken_links",
    "SessionMemory",
    "diff_session_memory_packs",
    "export_session_memory",
    "import_session_memory",
    "load_session_memory_pack",
    "validate_session_memory_pack",
    "SinkConcentration",
    "SinkProbe",
    "SinkProbeResult",
    "SinkTokenType",
    "SelfLearningBatchResult",
    "SelfLearningRelay",
    "ThreeTierPipeline",
    "ThreeTierPipelineResult",
    "build_relay_artifact",
    "build_saver_artifact",
    "build_three_tier_artifact",
    "SAVeRStep",
    "SAVeRConstraintState",
    "SAVeRVerifier",
    "compute_learning_improvement",
    "calibrated_confidence_from_energy",
    "fit_verdict_calibration",
    "compute_sink_concentration",
    "compute_conjugate_bound",
    "compute_input_uncertainty",
    "compute_lookahead_energy",
    "compute_prompt_uncertainty",
    "compute_semantic_energy",
    "compute_spilled_energy",
    "extract_typed_reasoning",
    "propagate",
    "build_gemma_structured_reasoning_prompt",
    "build_qwen_structured_reasoning_prompt",
    "extract_official_test_examples",
    "extract_prompt_examples",
    "load_monitorability_policy",
    "verify_semantic_grounding",
    "verify_semantic_verifier_v2",
    "verify_code",
    "LiveGPUGate",
    "build_session_startup_script",
    "check_session_startup_exists",
    "CloudGPUInstructions",
    "DeliverableContentValidator",
    "build_cloud_gpu_instructions",
    "generate_cloud_gpu_script",
    "EnvironmentAutoFix",
    "apply_env_autofix",
    "build_env_autofix_artifact",
    "ExperimentTimeoutResult",
    "ExperimentTimeoutWatchdog",
    "build_timeout_artifact",
    "get_timeout_minutes",
    "DualGPUHealthResult",
    "build_gpu_fix_artifact",
    "check_dual_gpu_health",
    "ZombieFixResult",
    "build_zombie_fix_artifact",
    "build_zombie_fix_strategy",
    "FOVERAnnotator",
    "FOVERCoTStep",
    "annotate_step_with_z3",
    "parse_cot_into_steps",
    "SpilledEnergyDetector",
    "SpilledEnergyDetectorResult",
    "SpilledEnergyToken",
    "compute_detector_spilled_energy",
    "BenchmarkBatch",
    "LongRunBenchmarkExecutor",
    "LongRunBenchmarkResult",
    "get_batch_size",
    "CarnotThinkProbe",
    "ThinkProbeResult",
    "ThinkVerdict",
    "build_think_probe_prompt",
    "parse_think_probe_output",
    "BoltzmannRepairBridge",
    "LinearSpinAdapter",
    "RepairDirection",
    "CrossSessionResult",
    "simulate_session",
    "compute_relay_verdict",
    "GemmaTransformersLoader",
    "Gemma4QuantizedLoader",
    "AtomicResultWriter",
    "ThinkProbeV2",
    "ThinkProbeV2Result",
    "ViolationPattern",
    "ConstraintAdditionFromMemory",
    "LSEBMConstraintReplayer",
    "ViolationDistribution",
    "DeliverableGuard",
    "DocOnlyClassifier",
    "DualGPUAssigner",
    "ConductorSessionHealthCheck",
    "GPUHealth",
    "SessionHealthResult",
    "ZombieProcess",
    "ConstraintDomain",
    "DomainParameterPartition",
    "PartitionIsolationScore",
    "PPSConstraintLearner",
    "GPUVRAMGate",
    "GPUVRAMGateV2",
    "GPUVRAMInsufficientError",
    "VRAMStatus",
    "GPUThermalGate",
    "GPUThermalThrottleError",
    "ThermalStatus",
    "ConductorDedupCheck",
    "PartialResultHandoff",
    "AuditFinding",
    "DualGPUHarness",
    "HarnessAudit",
    "HarnessPatchResult",
    "HarnessPatcher",
    "BatchingEnforcementAudit",
    "BatchingHookRunner",
    "BatchingViolation",
    "ContinuationEntropy",
    "NUPProbe",
    "NUPProbeResult",
    "nup_score_with_latency",
    "InterleavedViolationSequence",
    "PPSEBMRealValidationResult",
    "BayesianEntropyEstimator",
    "EntropyEstimate",
    "NUPProbeV2",
    "NUPProbeV2Result",
    "SuRePriorityReplay",
    "SuReReplayResult",
    "ViolationSurprise",
    "VRAMBudgetLedger",
    "VRAMForecast",
    "Precision100qV6Result",
    "BoltzmannSemanticEnergy",
    "SemanticCluster",
    "CLAPFeatureExtractor",
    "CLAPFeatures",
    "NUPProbeV3",
    "EnergyMagnitudeBuffer",
    "EnergyMagnitudeReplay",
    "NPUBenchmarkResult",
    "NPUEntropyProbe",
    "JITVRAMCheck",
    "JITVRAMResult",
    "ExpandedGPUReaper",
    "ExpandedGPUReaperConfig",
    "ExpandedGPUReapResult",
    "LeWorldModelJEPATrainer",
    "LeWorldModelLoss",
    "gaussian_kl_regularization",
    "BasinEstimate",
    "HallucinationBasinDetector",
    "estimate_basin_depth",
    "ContrastivePairLoss",
    "NUPProbeV4",
    "EORMAdaptiveRectifier",
    "RectifierResult",
    "EnergyGuidedConfig",
    "EnergyGuidedDecoder",
    "FOVERCorpusEntry",
    "balance_corpus",
    "compute_corpus_diversity",
    "merge_fover_sources",
    "LatentCoTCalibrationResult",
    "LatentCoTEBMCalibrator",
    "HalluFieldDetector",
    "HalluFieldResult",
    "PRABeamCandidate",
    "PRABeamResult",
    "PRAEBMBeamSearch",
    "EORMHardwareCorrelation",
    "HardwareEnergyProbe",
    "HardwareEnergyReading",
    "compute_eorm_hardware_correlation",
    "ExclusionEntry",
    "ExclusionManifest",
    "build_default_manifest",
    "DSVDAdapter",
    "DSVDLinearProbe",
    "DSVDProbeResult",
    "DSVDLiveTrainPair",
    "DSVDLiveTrainer",
    "TemporalWindowLabeler",
    "assert_live_gpu_available",
    "assert_live_or_ci_skip",
    "OTVVerificationToken",
    "OTVVerifier",
    "MISECalibrator",
    "MISETriple",
    "FLIPRepairTriple",
    "FLIPRewardCalibrator",
    "HISRViolationWeight",
    "HISRWeighter",
    "InterleavedLogicVerifier",
    "InterleavedStepResult",
    "CAPOCalibrationLoss",
    "CausalStepDependency",
    "FACTEFaithfulnessProbe",
    "SymCodeVerifier",
    "OracleCorpusBuilder",
    "OracleChain",
    "StepLabel",
    "ExtractorPolicy",
    "MetaJuLSAdapter",
    "InterWhenMonitor",
    "InterWhenViolation",
    "HermesVerificationStep",
    "HermesVerifierAdapter",
    "AdapTrackRepairer",
    "BacktrackEvent",
    "DualGPURetrain",
    "DualGPURetrainConfig",
    "HermesV2GenerationResult",
    "HermesV2LiveLoop",
    "HermesV2StepResult",
    "CausalEntailmentResult",
    "CausalReasoningVerifier",
    "FORCER_SYSTEM_ADDENDUM",
    "ForcedEquationResult",
    "StructuredEquationForcer",
    "HermesV2StructuredLoop",
    "HermesV2StructuredResult",
    "EnsembleGateV3Result",
    "EnsembleRecallGateV3",
    "SpecGuardStepResult",
    "SpecGuardVerifier",
    "LSEBMCLReplayBuffer",
    "ReplaySession",
    "HALPProbe",
    "HALPProbeResult",
    "GPUZombieResult",
    "get_gpu_memory_pids",
    "kill_gpu_zombies",
    "VRAMEvictionResult",
    "evict_gpu_vram",
    "load_gemma4_on_gpu1",
    "MARSMarginGate",
    "MARSMarginResult",
    "compute_logit_margin",
    "CPMIContrastivePairBuilder",
    "CPMITriple",
    "compute_cpmi_score",
    "generate_hard_negative",
    "ConstraintSPOTuple",
    "EmbeddingConstraintStore",
    "ConstraintInjectionResult",
    "ExternalFieldEnergyResult",
    "IsingConstraintInjector",
    "JepaWiringCheckResult",
    "check_cpmi_wiring",
    "VRAMLoopEvictionResult",
    "evict_vram_with_loop",
    "VGScheduleResult",
    "VGSearchScheduler",
    "AgentScore",
    "ArbiterResult",
    "MultiAgentArbiter",
    "ActivationJailbreakProbe",
    "ProbeMetadata",
    "GGUFCacheConfig",
    "GGUFCacheResolver",
    "GGUFModelNotFoundError",
    "resolve_gguf_path",
    "ExclusionManifestEnforcer",
]
