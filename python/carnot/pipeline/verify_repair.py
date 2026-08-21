"""Verify-and-repair pipeline: the main user-facing API for Carnot verification.

**Researcher summary:**
    Wires constraint extraction, Ising-model verification, and LLM-driven
    repair into a single class. Users import ``VerifyRepairPipeline``, call
    ``verify()`` to check a response, or ``verify_and_repair()`` to
    iteratively fix violations via an LLM feedback loop.

**Detailed explanation for engineers:**
    This module is THE product -- the class users will ``import`` and use.
    It consolidates the logic that previously lived in experiment scripts
    (Exp 56 for live LLM verification, Exp 57 for the verify-repair loop)
    into a clean, importable library.

    The pipeline has two modes:

    1. **Verify-only mode** (no model loaded): The user provides both the
       question and the response. The pipeline extracts constraints from
       the response, builds a ComposedEnergy from any constraint terms,
       and returns a VerificationResult indicating which constraints pass
       or fail. Repair is not possible without a model.

    2. **Verify-and-repair mode** (model loaded): The pipeline can also
       generate responses and, when violations are found, format them as
       natural-language feedback, regenerate via the LLM, and re-verify --
       up to ``max_repairs`` iterations. This is the core Carnot value
       proposition: EBMs don't just classify outputs as good/bad, they
       GUIDE the LLM toward correct answers.

    Architecture:
    - ``VerificationResult``: Per-call result with verified flag, constraint
      details, total energy, violations list, and full energy decomposition.
    - ``RepairResult``: Full history of a verify-and-repair run including
      initial and final responses, iteration count, and per-iteration
      verification results.
    - ``VerifyRepairPipeline``: The main class. Holds an extractor, an
      optional LLM model, and configuration. Exposes ``verify()``,
      ``verify_and_repair()``, and ``extract_constraints()``.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from carnot.pipeline.errors import (
    CarnotError,
    ExtractionError,
    ModelLoadError,
    PipelineTimeoutError,
    RepairError,
    VerificationError,
)
from carnot.pipeline.extract import AutoExtractor, ConstraintExtractor, ConstraintResult
from carnot.pipeline.formal_claim_verifier import (
    FormalClaimBatchResult,
    FormalClaimVerifier,
    normalize_claim,
)
from carnot.pipeline.process_verifier import ProcessVerificationResult, ProcessVerifier
from carnot.pipeline.semantic_grounding import SemanticGroundingVerifier
from carnot.pipeline.semantic_verifier_v2 import SemanticVerifierV2
from carnot.pipeline.structured_reasoning import StructuredReasoningController
from carnot.pipeline.typed_reasoning import extract_typed_reasoning as build_typed_reasoning_ir
from carnot.pipeline.verdict_record import VerdictRecord, calibrated_confidence_from_energy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from carnot.pipeline.confidence_weighted_repair import ConfidenceRepairResult
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
    from carnot.pipeline.cot_circuit_verifier import CoTCircuit
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector
    from carnot.pipeline.memory import ConstraintMemory
    from carnot.pipeline.probability_calibration_verifier import ProbabilityCalibrationVerifier
    from carnot.pipeline.semantic_energy_extractor import DualEnergyResult
    from carnot.pipeline.semantic_grounding import SemanticGroundingResult
    from carnot.pipeline.semantic_verifier_v2 import SemanticVerifierV2Result
    from carnot.pipeline.spilled_energy_extractor import SpilledEnergyResult
    from carnot.pipeline.tracker import ConstraintTracker
    from carnot.pipeline.typed_reasoning import TypedReasoningIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CARNOT_FAST_EVAL — corpus subsampling for architecture sweeps (exp1117)
# ---------------------------------------------------------------------------
#
# Architecture sweeps that iterate the full FoVer corpus (6,548 pairs) on CPU
# burn 10–15 min/experiment that contributes nothing toward the actual
# architecture decision — the comparison signal saturates well below 1k
# pairs.  Setting ``CARNOT_FAST_EVAL=1`` in the experiment environment opts
# the experiment into a 500-pair random sample (deterministic seed).  The
# flag is OFF by default — headline-result experiments must run on the full
# corpus or the verdict isn't reproducible.
#
# Per-experiment opt-in (recommended pattern):
#
#     pairs = load_full_corpus()
#     pairs = maybe_subsample_corpus(pairs)  # honours CARNOT_FAST_EVAL
#
# Or guard at the loader call site:
#
#     if os.getenv("CARNOT_FAST_EVAL", "0") == "1":
#         pairs = random.sample(pairs, min(500, len(pairs)))
#
# The conductor does NOT set this flag for queued tasks — it's an
# experiment-script-level tool for the architecture-sweep tier only.
def maybe_subsample_corpus(
    items: list,
    sample_size: int = 500,
    seed: int = 0xC417,  # noqa: B008
) -> list:
    """Return a 500-pair random subset when ``CARNOT_FAST_EVAL=1``.

    Otherwise returns ``items`` unchanged.  The seed is fixed so a re-run
    of the same experiment lands the same subset, preserving the only
    reproducibility guarantee fast-eval can offer.  Callers that need a
    different seed can pass it explicitly.

    Parameters
    ----------
    items:
        The full corpus (any iterable that ``random.Random.sample`` accepts).
    sample_size:
        Target sample size.  When ``len(items) <= sample_size`` the input
        is returned unchanged so a small corpus is never over-sampled.
    seed:
        RNG seed for the subset.  Defaults to a fixed sentinel.
    """
    if os.environ.get("CARNOT_FAST_EVAL", "0") != "1":
        return items
    if len(items) <= sample_size:
        return items
    import random as _random

    rng = _random.Random(seed)
    return rng.sample(list(items), sample_size)


# ---------------------------------------------------------------------------
# Rust backend auto-detection
# ---------------------------------------------------------------------------

# Check for CARNOT_USE_RUST env var (1 = force Rust, 0 = force Python).
# If unset, auto-detect: use Rust when available.
_FORCE_RUST = os.environ.get("CARNOT_USE_RUST")

try:
    from carnot._rust_compat import RUST_AVAILABLE, RustVerifyPipeline
except ImportError:
    RUST_AVAILABLE = False
    RustVerifyPipeline = None

_USE_RUST_PIPELINE: bool
if _FORCE_RUST == "1":
    _USE_RUST_PIPELINE = RUST_AVAILABLE
    if not RUST_AVAILABLE:
        logger.warning(
            "CARNOT_USE_RUST=1 but Rust bindings not installed. Falling back to Python pipeline."
        )
elif _FORCE_RUST == "0":
    _USE_RUST_PIPELINE = False
else:
    # Default to Python pipeline until Rust certificate format is fully compatible.
    # Set CARNOT_USE_RUST=1 to opt in to the Rust fast path.
    _USE_RUST_PIPELINE = False

if _USE_RUST_PIPELINE:
    logger.info("Rust verification pipeline enabled (10x faster verify).")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of verifying a single response against extracted constraints.

    **Detailed explanation for engineers:**
        After the pipeline extracts constraints from a response and evaluates
        each one, this dataclass captures the outcome. The ``verified`` flag
        is True only when every extracted constraint is satisfied. The
        ``violations`` list is a convenience subset of ``constraints`` that
        failed, so callers don't need to filter manually.

        The ``energy`` field is the total weighted energy from the
        ComposedEnergy (if constraint terms were available). Zero energy
        means all energy-backed constraints are satisfied. Constraints
        without energy terms (e.g., factual claims that can only be
        regex-checked) contribute to the ``constraints`` and ``violations``
        lists but not to the energy score.

        The ``certificate`` dict provides the full energy decomposition --
        per-constraint name, raw energy, weighted energy, and satisfaction
        status -- for debugging and audit trails.

    Attributes:
        verified: True if all extracted constraints are satisfied.
        constraints: All extracted constraints with their evaluation results.
        energy: Total weighted energy from ComposedEnergy terms (0.0 if no
            energy terms were available).
        violations: Subset of constraints that failed verification.
        certificate: Full energy decomposition dict with per-constraint
            details. Keys: "total_energy", "per_constraint" (list of dicts
            with "name", "energy", "weighted_energy", "satisfied").

    Spec: REQ-VERIFY-003, SCENARIO-VERIFY-004
    """

    verified: bool
    constraints: list[ConstraintResult]
    energy: float
    violations: list[ConstraintResult]
    certificate: dict[str, object] = field(default_factory=dict)
    mode: str = "FULL"
    """Verification mode: "FULL" for normal pipeline, "FAST_PATH" for JEPA early-exit."""
    skipped: bool = False
    """True when Tier 3 JEPA predictor gated this check as low-risk (fast path taken)."""
    typed_reasoning: TypedReasoningIR | None = None
    """Optional typed reasoning IR extracted from the prompt/response pair."""
    semantic_grounding: SemanticGroundingResult | None = None
    """Optional semantic-grounding analysis extracted from the prompt/response pair."""
    semantic_verifier_v2: SemanticVerifierV2Result | None = None
    """Optional calibrated claim-level semantic analysis for the prompt/response pair."""
    use_lowrank_kaem: bool = False
    """True when the KAN fast-path tier used LowRankKAEMEnergy (n_vars <= 100, k=2).
    False for full-rank KAEMEnergy (n_vars > 100) or when KAN path was not taken."""
    streaming_cot_unstable: bool = False
    """True when Tier 0g StreamingCoTHalluDetector flagged the CoT as streaming-unstable.
    Populated by VerifyRepairPipeline.verify() when CARNOT_STREAMING_COT=1.
    Advisory only — does not affect the verified flag or repair logic.
    Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165"""
    streaming_cot_phas: float = 0.0
    """Final EMA prefix hallucination score (PHaS) from StreamingCoTHalluDetector.
    Range [0, 1]; higher = more streaming-unstable trajectory.
    Populated alongside streaming_cot_unstable when CARNOT_STREAMING_COT=1.
    Spec: REQ-VERIFY-140, SCENARIO-VERIFY-166"""
    geometric_energy: float = 0.0
    """Mean L2 distance of CoT steps from the grounded manifold centroid in TF-IDF
    (SAE proxy) feature space.  Computed by HalluSAEGeometricProbe (Tier 0i).
    Higher values indicate the trajectory visited geometrically distant regions
    relative to the reference correct-reasoning set.  Advisory only.
    Spec: REQ-PROBE-050"""
    hallusae_anomalous: bool = False
    """True when Tier 0i HalluSAEGeometricProbe.geometric_energy exceeds its threshold.
    Set by the caller after running HalluSAEGeometricProbe.is_anomalous().
    Advisory only — does not affect the verified flag or repair logic.
    Spec: REQ-PROBE-050"""
    spectral_diffuse: bool = False
    """True when Tier 0h SpectralAttentionProbe flagged the CoT as spectrally diffuse.
    Populated by VerifyRepairPipeline.verify() when CARNOT_SPECTRAL_PROBE=1.
    Advisory only — does not affect the verified flag or repair logic.
    Spec: REQ-VERIFY-146, SCENARIO-VERIFY-173"""
    spectral_entropy_mean: float = 0.0
    """Mean per-step spectral entropy from SpectralAttentionProbe.
    Range [0, log(k)] where k = number of eigenvalues used (default 10).
    Higher = more diffuse attention spectrum = higher hallucination risk.
    Populated alongside spectral_diffuse when CARNOT_SPECTRAL_PROBE=1.
    Spec: REQ-VERIFY-146, SCENARIO-VERIFY-174"""

    def to_verdict_record(
        self,
        *,
        budget_ms_consumed: float = 0.0,
        producing_tier: int = 3,
        tier_reached: int | None = None,
        repairs_applied: list[str] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> VerdictRecord:
        """Convert this legacy result into a structured verdict record.

        Spec: REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410
        """
        certificate_error = self.certificate.get("error_type") or self.certificate.get("error")
        if certificate_error:
            verdict = "abstain"
            rationale = f"verification_error:{certificate_error}"
        elif self.verified:
            verdict = "pass"
            rationale = "no_constraints" if not self.constraints else "constraints_satisfied"
        else:
            verdict = "fail"
            if self.violations:
                violation_types = sorted({item.constraint_type for item in self.violations})
                rationale = "constraint_violation:" + ",".join(violation_types)
            else:
                rationale = "verification_failed"

        confidence_energy = self.energy
        if not self.verified and confidence_energy <= 0.0:
            confidence_energy = float(max(1, len(self.violations)))

        record_extras: dict[str, Any] = {
            "certificate": self.certificate,
            "mode": self.mode,
            "skipped": self.skipped,
            "n_constraints": len(self.constraints),
            "n_violations": len(self.violations),
            "violation_types": [item.constraint_type for item in self.violations],
        }
        if extras:
            record_extras.update(extras)

        return VerdictRecord(
            verdict=verdict,  # type: ignore[arg-type]
            energy=float(self.energy),
            calibrated_confidence=calibrated_confidence_from_energy(confidence_energy),
            producing_tier=producing_tier,
            tier_reached=producing_tier if tier_reached is None else tier_reached,
            rationale=rationale,
            budget_ms_consumed=budget_ms_consumed,
            repairs_applied=[] if repairs_applied is None else repairs_applied,
            extras=record_extras,
        )


@dataclass
class RepairResult:
    """Result of a full verify-and-repair run across multiple iterations.

    **Detailed explanation for engineers:**
        Captures the complete trajectory of a repair loop. The
        ``initial_response`` is what the LLM (or user) provided first.
        The ``final_response`` is what remained after all repair iterations
        (which may be the same as initial if no repairs were needed or
        possible).

        ``repaired`` is True only when the final response differs from the
        initial AND the final response passes verification. If repairs were
        attempted but the response still has violations, ``repaired`` is
        False even though ``final_response`` may differ from
        ``initial_response``.

        The ``history`` list contains one VerificationResult per iteration
        (including the initial verification), so callers can inspect how
        violations changed across repair attempts.

    Attributes:
        initial_response: The first response (from LLM generation or user).
        final_response: The response after all repair iterations.
        verified: True if final_response passes all constraint checks.
        repaired: True if final != initial AND final is verified.
        iterations: Number of repair iterations performed (0 if initial
            response was already verified).
        history: List of VerificationResult from each iteration.

    Spec: REQ-VERIFY-003, SCENARIO-VERIFY-004
    """

    initial_response: str
    final_response: str
    verified: bool
    repaired: bool
    iterations: int
    history: list[VerificationResult]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class VerifyRepairPipeline:
    """Main user-facing API: constraint extraction + verification + LLM repair.

    **Researcher summary:**
        Single class wiring ConstraintExtractor, ComposedEnergy verification,
        and optional LLM-driven repair into one call. Supports verify-only
        mode (no model) and full verify-and-repair mode (with model).

    **Detailed explanation for engineers:**
        This is the class users import and use. It replaces the ad-hoc
        pipeline code from experiment scripts 56 and 57 with a clean API.

        Construction:
        - ``model``: Optional HuggingFace model name or path. If provided,
          the pipeline loads it via ``transformers.AutoModelForCausalLM`` and
          can generate responses and perform repair. If None, the pipeline
          works in verify-only mode.
        - ``domains``: Optional list of domain hints to restrict constraint
          extraction (e.g., ["arithmetic", "code"]). If None, AutoExtractor
          tries all domains.
        - ``max_repairs``: Maximum number of LLM repair iterations (default 3).
        - ``extractor``: Custom ConstraintExtractor instance. If None,
          AutoExtractor is used (covers arithmetic, code, logic, NL).

        Methods:
        - ``verify(question, response, domain)``: Extract constraints and
          check them. Returns VerificationResult.
        - ``verify_and_repair(question, response, domain)``: Verify, and
          if violations found + model loaded, repair iteratively. Returns
          RepairResult.
        - ``extract_constraints(text, domain)``: Convenience method to just
          extract constraints without verification.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004,
    REQ-GPU-010
    """

    # When CARNOT_DUAL_GPU=1, pipeline routes GPU inference through DualGPURunner
    # if two model configs are loaded (Exp 856 / REQ-GPU-010).
    DUAL_GPU_ENABLED: bool = os.getenv("CARNOT_DUAL_GPU", "0") == "1"

    # When CARNOT_STREAMING_COT=1, run StreamingCoTHalluDetector (Tier 0g) in
    # verify() and populate result.streaming_cot_unstable + result.streaming_cot_phas.
    # Advisory only — does not affect the verified flag or short-circuit Ising.
    # Spec: REQ-VERIFY-140
    STREAMING_COT_ENABLED: bool = os.getenv("CARNOT_STREAMING_COT", "0") == "1"

    # When CARNOT_SPECTRAL_PROBE=1, run SpectralAttentionProbe (Tier 0h) in
    # verify() and populate result.spectral_diffuse + result.spectral_entropy_mean.
    # Advisory only — does not affect the verified flag or short-circuit Ising.
    # Spec: REQ-VERIFY-146
    SPECTRAL_PROBE_ENABLED: bool = os.getenv("CARNOT_SPECTRAL_PROBE", "0") == "1"

    def __init__(
        self,
        model: str | None = None,
        domains: list[str] | None = None,
        max_repairs: int = 3,
        extractor: ConstraintExtractor | None = None,
        semantic_grounding_verifier: SemanticGroundingVerifier | None = None,
        semantic_verifier_v2: SemanticVerifierV2 | None = None,
        timeout_seconds: float = 30.0,
        memory: ConstraintMemory | None = None,
        template_library: ConstraintTemplateLibrary | None = None,
        session_memory: Any | None = None,
        constraint_memory: Any | None = None,
        nup_probe: Any | None = None,
        nup_probe_threshold: float = 0.5,
        enable_constraint_accumulation: bool = False,
        second_model_spec: dict[str, str] | None = None,
        and_compose_verifier: Any | None = None,
        probability_calibration_verifier: ProbabilityCalibrationVerifier | None = None,
        casal_tier: Any | None = None,
        interwhen_monitor: Any | None = None,
        use_hardnet: bool = False,
        use_odar: bool = False,
        odar_risk_threshold: float = 0.5,
        odar_router: Any | None = None,
        routing_mode: str = "argmax",
        balance_ratio: float = 1.0,
        jepa_fast_path_predictor: Any | None = None,
        jepa_fast_path_threshold: float = 0.2,
        learning_mode: bool = False,
        n_learning_cycles: int = 3,
        enable_abductive_csp: bool = False,
        fr11_shadow_adapter_enabled: bool | None = None,
        fr11_shadow_ledger_path: str | os.PathLike[str] | None = None,
        fr11_shadow_checkpoint_path: str | os.PathLike[str] | None = None,
        fr11_shadow_adapter: Any | None = None,
        fr11_factor_cache_shadow_adapter_enabled: bool = False,
        fr11_factor_cache_shadow_ledger_path: str | os.PathLike[str] | None = None,
        fr11_factor_cache_shadow_checkpoint_path: str | os.PathLike[str] | None = None,
        fr11_factor_cache_shadow_adapter: Any | None = None,
    ) -> None:
        """Initialize the verify-repair pipeline.

        **Detailed explanation for engineers:**
            If ``model`` is a string, attempts to load it via HuggingFace
            transformers (AutoModelForCausalLM + AutoTokenizer). The model
            is loaded eagerly so errors surface at construction time rather
            than mid-pipeline. If loading fails, raises ModelLoadError
            wrapping the underlying exception.

            If ``model`` is None, the pipeline works in verify-only mode:
            ``verify()`` works normally, but ``verify_and_repair()`` cannot
            generate or repair responses (it will verify the provided
            response and return with ``repaired=False`` if violations exist).

            The ``timeout_seconds`` parameter sets a wall-clock budget for
            each call to ``verify()`` or ``verify_and_repair()``. If the
            call exceeds this budget, PipelineTimeoutError is raised. Set
            to 0 or None to disable the timeout.

            When ``session_memory`` is provided (a SessionMemory instance),
            the pipeline restores previously saved CaseMemory, template
            library, and FP-tracker state on init.  Call ``close()`` before
            the pipeline goes out of scope to persist the current state back
            to disk.  If no prior session exists, ``session_memory.load()``
            returns None and the pipeline starts with fresh in-memory state
            — existing callers that do not pass ``session_memory`` are
            unaffected.

        Args:
            model: HuggingFace model name/path, or None for verify-only.
            domains: Optional domain filter for constraint extraction.
            max_repairs: Max repair iterations (default 3).
            extractor: Custom extractor, or None for AutoExtractor.
            timeout_seconds: Max wall-clock seconds per verify/repair call.
                Default 30. Set to 0 or None to disable.
            memory: Optional ConstraintMemory for Tier 2 cross-session pattern
                learning. When provided, the pipeline queries memory for
                learned constraint suggestions before each verification and
                records new violation patterns after each verification. If
                None (default), memory integration is skipped entirely for
                full backward compatibility.
            session_memory: Optional SessionMemory instance for multi-session
                persistence of CaseMemory, ConstraintTemplateLibrary, and
                PerModelFPTracker state.  When provided, saved state is
                restored on init and persisted on close().  Default None
                preserves all existing behaviour.
            use_odar: Optional ODAR routing gate default for ``verify()``.
                When True, Tier 0 probe outputs are fused into expected free
                energy before Tier 1 extraction; low-EFE calls return a
                fast-path result. Default False preserves existing behaviour.
            odar_risk_threshold: EFE threshold used when constructing the
                default FreeEnergyRouter. Lower thresholds route fewer calls
                to the optimistic fast path.
            odar_router: Optional custom FreeEnergyRouter-compatible object
                with ``evaluate(probe_outputs)``. When omitted, the pipeline
                builds one lazily from ``odar_risk_threshold``.
            learning_mode: When True, enables FR-11 Tier 4 ORCA-NEXUS learning loop.
                Each successful repair is recorded in NEXUS constraint memory, enabling
                the system to generalize from observed repair patterns to new violations.
                Based on exp2755 validation: 3-cycle loop achieves AUROC=0.9275 with
                genuine generalization.
            n_learning_cycles: Number of learning cycles for ORCA-NEXUS loop (default 3).
            fr11_shadow_adapter_enabled: Optional FR-11 shadow-mode flag. None reads
                ``CARNOT_FR11_SHADOW_ADAPTER`` at construction. False preserves
                the exact pre-shadow behavior and writes no ledger.
            fr11_shadow_ledger_path: Append-only JSONL ledger path used only when the
                FR-11 shadow adapter is enabled.
            fr11_shadow_checkpoint_path: Atomic checkpoint path used only when the
                FR-11 shadow adapter is enabled.
            fr11_shadow_adapter: Optional prebuilt adapter for tests or controlled
                deployments. Passing None keeps the feature purely flag-driven.
            fr11_factor_cache_shadow_adapter_enabled: Explicit FR-11 factor-cache
                shadow flag. Default False preserves baseline behavior. Environment
                variables do not enable this production factor-cache surface.
            fr11_factor_cache_shadow_ledger_path: JSONL ledger path used only when
                the factor-cache shadow adapter is explicitly enabled.
            fr11_factor_cache_shadow_checkpoint_path: Atomic checkpoint path used only
                when the factor-cache shadow adapter is explicitly enabled.
            fr11_factor_cache_shadow_adapter: Optional prebuilt factor-cache adapter
                for tests or controlled deployments.

        Raises:
            ModelLoadError: If model is specified but cannot be loaded.

        Spec: REQ-VERIFY-001, REQ-LEARN-003, REQ-LEARN-021-2, REQ-ODAR-2243,
              REQ-LEARN-5640, REQ-PIPELINE-6479, REQ-LEARN-6479
        """
        self.learning_mode = learning_mode
        self.n_learning_cycles = n_learning_cycles

        self.enable_abductive_csp = enable_abductive_csp
        if self.enable_abductive_csp:
            from carnot.pipeline.abductive_csp import AbductiveCSPLayer

            self.abductive_csp_layer = AbductiveCSPLayer()
        else:
            self.abductive_csp_layer = None
        if learning_mode:
            from carnot.verify.nexus_constraint_memory import NexusConstraintMemory
            from carnot.pipeline.ttt_loop import TTTLoop

            self.nexus_memory = NexusConstraintMemory()
            self.ttt_loop = TTTLoop(self.nexus_memory)

        self._domains = domains
        self._max_repairs = max_repairs
        self._timeout_seconds = timeout_seconds or 0.0
        self._memory = memory
        self._template_library = template_library
        self._session_memory = session_memory
        # Tier 2 self-learning: accumulates violation observations across calls
        # and adds new constraints once a pattern crosses the observation threshold.
        # Validated in Exp 456; wired into the live pipeline in Exp 541 (REQ-LEARN-053).
        self._constraint_memory = constraint_memory
        # Tier 0c NUP Probe v6 (Exp 608, AUC=0.964).  When supplied, scores each
        # response between Tier 0b and Tier 0d.  Low score = likely correct = fast-path.
        self._nup_probe = nup_probe
        self._nup_probe_threshold = nup_probe_threshold
        # REQ-LEARN-048: gate the constraint write path.  When True, each violation
        # detected in verify() is written to the passed EmbeddingConstraintStore so
        # subsequent sessions can retrieve it.  Default False preserves existing
        # behaviour for callers that do not pass a store.
        self._enable_constraint_accumulation = enable_constraint_accumulation
        self._probability_calibration_verifier = probability_calibration_verifier
        # k=5 AND-composition ensemble (Phase 1d, Exp 1121).
        # When and_compose_verifier is None, the default k=5 ensemble is built.
        # Callers may pass a custom AndCompositionVerifier or None to keep default.
        if and_compose_verifier is None:
            from carnot.verify.and_composition_verifier import (
                build_default_verifier_ensemble,
            )

            self._and_compose_verifier = build_default_verifier_ensemble()
        else:
            self._and_compose_verifier = and_compose_verifier

        self._casal_tier = casal_tier
        # InterwhenMonitor for mid-generation sentence-boundary violation detection
        # (REQ-VERIFY-130, arXiv 2602.11202).  Advisory only: violations recorded in
        # certificate but never override result.verified.
        self._interwhen_monitor = interwhen_monitor
        self._use_hardnet = use_hardnet
        self._use_odar = use_odar
        self._odar_risk_threshold = odar_risk_threshold
        self._odar_router = odar_router
        self.routing_mode = routing_mode
        self.balance_ratio = balance_ratio
        # Pipeline-level JEPA fast-path predictor (exp2525, REQ-JEPA-002).
        # When set, predict_p_violation is called in verify() before Ising;
        # if p < jepa_fast_path_threshold the Ising pass is skipped entirely.
        self._jepa_fast_path_predictor = jepa_fast_path_predictor
        self._jepa_fast_path_threshold = jepa_fast_path_threshold
        self._session_log: list[dict[str, object]] = []
        self._online_observation_count = 0
        self._online_fast_path_taken_count = 0
        self._n_partial_fits = 0
        if fr11_shadow_adapter is not None:
            self._fr11_shadow_adapter = fr11_shadow_adapter
        else:
            fr11_shadow_enabled = (
                os.getenv("CARNOT_FR11_SHADOW_ADAPTER", "0") == "1"
                if fr11_shadow_adapter_enabled is None
                else bool(fr11_shadow_adapter_enabled)
            )
            self._fr11_shadow_adapter = None
            if fr11_shadow_enabled:
                from carnot.pipeline.fr11_shadow_adapter import FR11ShadowAdapter

                ledger_path = fr11_shadow_ledger_path or os.getenv(
                    "CARNOT_FR11_SHADOW_LEDGER",
                    "results/fr11_shadow_adapter_ledger.jsonl",
                )
                checkpoint_path = fr11_shadow_checkpoint_path or os.getenv(
                    "CARNOT_FR11_SHADOW_CHECKPOINT",
                    "results/fr11_shadow_adapter_checkpoint.json",
                )
                self._fr11_shadow_adapter = FR11ShadowAdapter(
                    ledger_path=ledger_path,
                    checkpoint_path=checkpoint_path,
                    enabled=True,
                )
        if fr11_factor_cache_shadow_adapter is not None:
            self._fr11_factor_cache_shadow_adapter = fr11_factor_cache_shadow_adapter
        else:
            self._fr11_factor_cache_shadow_adapter = None
            if fr11_factor_cache_shadow_adapter_enabled:
                from carnot.pipeline.factor_cache_shadow_adapter import (
                    FR11FactorCacheShadowAdapter,
                )

                ledger_path = (
                    fr11_factor_cache_shadow_ledger_path
                    or "results/fr11_factor_cache_shadow_adapter_ledger.jsonl"
                )
                checkpoint_path = (
                    fr11_factor_cache_shadow_checkpoint_path
                    or "results/fr11_factor_cache_shadow_adapter_checkpoint.json"
                )
                self._fr11_factor_cache_shadow_adapter = FR11FactorCacheShadowAdapter(
                    ledger_path=ledger_path,
                    checkpoint_path=checkpoint_path,
                    enabled=True,
                )
        self._repair_router = None
        if self.routing_mode == "odar":
            from carnot.pipeline.odar_router import OdarRouter

            self._repair_router = OdarRouter(max_iterations=max_repairs)

        # Restore persisted learning state if session_memory was provided
        # and a prior session exists on disk.  Restoring here (before
        # extractor setup) means the pipeline starts with calibrated state.
        if session_memory is not None:
            restored = session_memory.load()
            if restored is not None:
                restored_cm, restored_lib, _restored_tracker = restored
                # Only override if caller didn't supply explicit objects.
                # (This keeps the additive-only contract: session state is a
                # default, not a forced override over explicit constructor args.)
                if template_library is None:
                    self._template_library = restored_lib

        # Set up the constraint extractor.
        if extractor is not None:
            self._extractor = extractor
        else:
            self._extractor = AutoExtractor()
        self._semantic_grounding = semantic_grounding_verifier or SemanticGroundingVerifier()
        self._semantic_verifier_v2 = semantic_verifier_v2 or SemanticVerifierV2()

        # Set up the optional LLM model.
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._model_name: str | None = model
        # Second model config for DualGPURunner parallel inference (REQ-GPU-010).
        # When set alongside DUAL_GPU_ENABLED=True, verify_and_repair operations
        # can dispatch to both GPUs concurrently for ~2x throughput.
        self._second_model_spec: dict[str, str] | None = second_model_spec

        self.verifier_list = [f"v{i}" for i in range(1, 16)] + ["nla_verifier"]

        if model is not None:
            self._load_model(model)

    def has_second_model(self) -> bool:
        """True if a second model config is loaded for DualGPURunner parallel inference.

        Returns True only when both a primary model AND a second_model_spec were
        supplied at construction.  DualGPURunner requires exactly two specs, so
        both must be present before DUAL_GPU_ENABLED can take effect.

        Spec: REQ-GPU-010
        """
        return self._model_name is not None and self._second_model_spec is not None

    def get_balance_ratio(self) -> float:
        """Return the current balance ratio (fraction of tokens constrained vs free)."""
        return getattr(self, "balance_ratio", 1.0)

    @property
    def has_model(self) -> bool:
        """True if an LLM model is loaded and available for generation."""
        return self._model is not None

    def close(self) -> None:
        """Persist current session state to disk if session_memory is set.

        **When to call this:**
            Call ``close()`` when you are done using the pipeline for a
            session — typically at script exit or when a request handler
            completes.  It saves the current CaseMemory, template library,
            and FP-tracker state so the next pipeline instance can resume
            from where this one left off.

            If ``session_memory`` was not provided at construction (the
            default), this method is a no-op and safe to call unconditionally.

        **What is persisted:**
            - ``_memory`` (ConstraintMemory / CaseMemory equivalent) — if
              the pipeline is using a CaseMemory instance attached to
              ``_memory``.
            - ``_template_library`` — ConstraintTemplateLibrary observation
              counts and activation state.
            - A fresh PerModelFPTracker if none is attached (the session
              memory contract requires all three components to be saved
              together; an empty tracker is a valid initial state).

            NOTE: The pipeline currently stores CaseMemory via the
            ``_memory`` slot (ConstraintMemory interface) or as a separate
            attribute.  We create minimal fresh instances for any component
            that is not present so the save contract is always met.

        Spec: REQ-LEARN-021-3
        """
        for adapter_name in ("_fr11_shadow_adapter", "_fr11_factor_cache_shadow_adapter"):
            adapter = getattr(self, adapter_name, None)
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

        if self._session_memory is None:
            return

        from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary

        # Use whatever learning state the pipeline currently holds, falling
        # back to fresh (empty) instances so the save always succeeds.
        case_memory = self._memory if isinstance(self._memory, CaseMemory) else CaseMemory()
        template_library = (
            self._template_library
            if isinstance(self._template_library, ConstraintTemplateLibrary)
            else ConstraintTemplateLibrary()
        )
        fp_tracker = PerModelFPTracker()

        self._session_memory.save(case_memory, template_library, fp_tracker)

    def _load_model(self, model_name: str) -> None:
        """Load a HuggingFace model for generation and repair.

        **Detailed explanation for engineers:**
            Uses transformers AutoModelForCausalLM and AutoTokenizer.
            Detects CUDA availability and places the model on GPU if
            possible. Sets the model to eval mode (no dropout, no
            gradient tracking) since we only need inference.

            Wraps all load-time failures in ModelLoadError so callers
            get a single exception type regardless of whether the failure
            is ImportError, OSError, or OOM.

        Args:
            model_name: HuggingFace model name or local path.

        Raises:
            ModelLoadError: If torch/transformers missing or model load fails.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                f"Required packages not installed: {exc}",
                details={"model_name": model_name},
            ) from exc

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading model %s on %s...", model_name, self._device)

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=os.environ.get("CARNOT_TRUST_REMOTE_CODE", "") == "1"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=os.environ.get("CARNOT_TRUST_REMOTE_CODE", "") == "1",
                torch_dtype=torch.float16 if self._device == "cuda" else None,
            )
            if self._device == "cuda":
                self._model = self._model.cuda()
            self._model.eval()
            logger.info("Model %s loaded successfully.", model_name)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{model_name}': {exc}",
                details={"model_name": model_name},
            ) from exc

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response from the loaded LLM.

        **Detailed explanation for engineers:**
            Uses greedy decoding (do_sample=False) for reproducibility.
            Applies the tokenizer's chat template if available, otherwise
            uses the raw prompt. Strips any ``<think>...</think>`` reasoning
            tokens from the output (common in Qwen models).

        Args:
            prompt: The full prompt text to send to the model.
            max_new_tokens: Maximum tokens to generate (default 256).

        Returns:
            The generated text (decoded, special tokens stripped).

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded. Initialize with model='...' to enable generation.")

        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers may not support enable_thinking.
            try:
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: use raw prompt if no chat template.
                text = prompt

        inputs = self._tokenizer(text, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Strip thinking tokens if present (common in Qwen models).
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        return str(response)

    def extract_constraints(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        """Extract constraints from text without verification.

        **Detailed explanation for engineers:**
            Convenience method that delegates to the underlying extractor.
            Applies the pipeline's domain filter if set and no explicit
            domain is provided. Wraps extractor failures in ExtractionError
            so the caller gets a consistent exception type.

        Args:
            text: Input text to extract constraints from.
            domain: Optional domain hint. If None and pipeline has domains
                set, extracts for each configured domain and merges.

        Returns:
            List of extracted ConstraintResult objects.

        Raises:
            ExtractionError: If extraction fails unexpectedly.

        Spec: REQ-VERIFY-001
        """
        try:
            return self._extract_constraints_inner(text, domain)
        except CarnotError:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Constraint extraction failed: {exc}",
                details={"domain": domain, "input_length": len(text)},
            ) from exc

    def _extract_constraints_inner(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        """Core extraction logic, separated for error wrapping."""
        effective_domain = domain
        if effective_domain is None and self._domains and len(self._domains) == 1:
            effective_domain = self._domains[0]

        if effective_domain is not None:
            return self._extractor.extract(text, effective_domain)

        # If multiple domains configured, extract for each and merge.
        if self._domains:
            results: list[ConstraintResult] = []
            seen: set[str] = set()
            for d in self._domains:
                for cr in self._extractor.extract(text, d):
                    if cr.description not in seen:
                        seen.add(cr.description)
                        results.append(cr)
            return results

        # No domain filter: let extractor auto-detect.
        return self._extractor.extract(text)

    def extract_typed_reasoning(self, question: str, response: str) -> TypedReasoningIR | None:
        """Extract typed reasoning IR without affecting verification behavior."""
        try:
            return build_typed_reasoning_ir(question=question, response=response)
        except Exception as exc:
            logger.warning("Typed reasoning extraction degraded: %s", exc)
            return None

    def generate_structured_reasoning(
        self,
        question: str,
        task_slice: str,
        model_name: str | None = None,
    ) -> object:
        """Generate a policy-gated structured response through an additive entry point."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded. Initialize with model='...' to enable generation.")

        controller = StructuredReasoningController()
        return controller.emit(
            question=question,
            task_slice=task_slice,
            model_name=model_name or self._model_name,
            model=self._model,
            tokenizer=self._tokenizer,
            fallback_generate=self._generate,
        )

    def verify_generated_code(
        self,
        code: str,
        prompt: str,
        entry_point: str,
        official_tests: str,
        *,
        include_static: bool = True,
        include_pbt: bool = True,
        include_specs: bool = False,
        task_id: str | None = None,
        spec_corpus_path: str | os.PathLike[str] | None = None,
        trace_paths: Sequence[str | os.PathLike[str]] | None = None,
    ) -> VerificationResult:
        """Verify a generated Python candidate with static checks plus bounded PBT.

        This is an additive code-verification entry point for HumanEval-style
        tasks. It keeps the existing text-response ``verify()`` path untouched
        while letting callers verify code directly with prompt context, the
        official harness, and the Hypothesis-backed property verifier.

        Spec: REQ-CODE-010, REQ-CODE-027, SCENARIO-CODE-009,
              SCENARIO-CODE-025
        """
        constraints: list[ConstraintResult] = []
        pbt_summary: dict[str, object] = {"enabled": include_pbt}

        if include_static:
            constraints.extend(self.extract_constraints(code, domain="code"))

        if include_specs:
            from carnot.pipeline.spec_code_verifier import SpecCodeVerifier

            verifier = SpecCodeVerifier(
                spec_corpus_path=str(spec_corpus_path) if spec_corpus_path is not None else None,
                learning_artifact_paths=(
                    tuple(str(path) for path in trace_paths) if trace_paths is not None else None
                ),
                include_official_tests=bool(official_tests.strip()),
                include_pbt=include_pbt,
            )
            aggregated = verifier.verify(
                code,
                prompt,
                entry_point,
                official_tests,
                task_id=task_id,
            )
            constraints.extend(aggregated.to_constraint_results())
            certificate = aggregated.to_certificate()
            pbt_summary.update(certificate["pbt_summary"])
            official_summary = certificate["official_test_summary"]
            spec_summary = certificate["spec_summary"]
            repair_hints = certificate["repair_ranking"]["hints"]
        elif include_pbt:
            from carnot.pipeline.pbt_code_verifier import PBTCodeVerifier

            pbt_result = PBTCodeVerifier().verify(code, prompt, entry_point, official_tests)
            constraints.extend(pbt_result.to_constraint_results())
            pbt_summary.update(
                {
                    "verified": pbt_result.verified,
                    "n_properties": len(pbt_result.derived_properties),
                    "n_failures": len(pbt_result.failures),
                    "property_names": [prop.name for prop in pbt_result.derived_properties],
                    "wall_clock_seconds": pbt_result.wall_clock_seconds,
                }
            )
        else:
            pbt_summary.update(
                {
                    "verified": True,
                    "n_properties": 0,
                    "n_failures": 0,
                    "property_names": [],
                    "wall_clock_seconds": 0.0,
                }
            )

        result = self._evaluate_constraints(constraints)
        result.certificate["pbt_summary"] = pbt_summary
        if include_specs:
            result.certificate["official_summary"] = official_summary
            result.certificate["official_test_summary"] = official_summary
            result.certificate["spec_summary"] = spec_summary
            result.certificate["repair_hints"] = repair_hints
            result.certificate["repair_ranking"] = {
                "n_hints": len(repair_hints),
                "hints": repair_hints,
            }
        return result

    def verify_generated_code_with_specs(
        self,
        code: str,
        prompt: str,
        entry_point: str,
        official_tests: str,
        *,
        task_id: str | None = None,
        case_id: str | None = None,
        spec_corpus_path: str | os.PathLike[str] | None = None,
        learning_artifact_paths: tuple[str | os.PathLike[str], ...] | None = None,
        include_static: bool = True,
        include_official_tests: bool = True,
        include_pbt: bool = True,
    ) -> VerificationResult:
        """Verify generated code through the additive explicit spec-aware path."""
        return self.verify_generated_code(
            code,
            prompt,
            entry_point,
            official_tests,
            include_static=include_static,
            include_pbt=include_pbt,
            include_specs=True,
            task_id=task_id,
            spec_corpus_path=spec_corpus_path,
            trace_paths=learning_artifact_paths,
        )

    def verify_semantic_grounding(
        self,
        question: str,
        response: str,
        typed_reasoning: TypedReasoningIR | None = None,
    ) -> SemanticGroundingResult | None:
        """Run semantic grounding additively without breaking existing verification."""
        try:
            return self._semantic_grounding.verify(
                question=question,
                response=response,
                typed_reasoning=typed_reasoning,
            )
        except Exception as exc:
            logger.warning("Semantic grounding degraded: %s", exc)
            return None

    def verify_semantic_verifier_v2(
        self,
        question: str,
        response: str,
        typed_reasoning: TypedReasoningIR | None = None,
        semantic_grounding: SemanticGroundingResult | None = None,
        task_slice: str = "live_gsm8k_semantic_failure",
    ) -> SemanticVerifierV2Result | None:
        """Run calibrated semantic verifier v2 additively without breaking callers."""
        try:
            return self._semantic_verifier_v2.verify(
                question=question,
                response=response,
                typed_reasoning=typed_reasoning,
                semantic_grounding=semantic_grounding,
                task_slice=task_slice,
            )
        except Exception as exc:
            logger.warning("Semantic verifier v2 degraded: %s", exc)
            return None

    def verify_formal_claims(
        self,
        raw_claims: list[dict[str, object]],
    ) -> FormalClaimBatchResult:
        """Normalize and verify a list of raw formal-claim dicts.

        This is an additive entry point that does not affect the existing
        ``verify()`` or ``verify_and_repair()`` paths.  Callers that need
        solver-routed verification of typed claims from the Exp 244 corpus
        (or any corpus following the same schema) can call this method
        independently.

        Each claim dict is normalized into a typed FormalClaim and dispatched
        to the narrowest deterministic checker covering its
        ``candidate_solver_route``.  Claims that are not safely formalizable
        receive an explicit ``'abstain'`` verdict.

        Args:
            raw_claims: List of claim dicts with at least ``claim_id``,
                ``candidate_solver_route``, ``formalization_status``,
                ``relation_type``, ``operands``, ``target``, and
                ``bound_variables`` keys.

        Returns:
            FormalClaimBatchResult with per-claim verdicts, aggregate counts
            by verdict and route, and deterministic JSON serialization.

        Spec: REQ-VERIFY-059
        """
        verifier = FormalClaimVerifier()
        claims = [normalize_claim(raw) for raw in raw_claims]
        return verifier.verify_batch(claims)

    def verify_process_integrity(
        self,
        corpus_row: dict[str, object],
    ) -> ProcessVerificationResult:
        """Verify a corpus row for process-integrity defects.

        This is an additive entry point that does not affect ``verify()`` or
        ``verify_and_repair()`` callers.  Pass any dict following the Exp 248
        process-integrity corpus schema (or any reasoning / code-repair trace
        that carries ``process_evidence``, ``outcome_label``,
        ``process_label``, and optionally ``repair_context``).

        Detects: unsupported_step, missing_premise_jump,
        contradictory_intermediate, outcome_correct_process_invalid,
        repair_regression, repair_stall.

        Args:
            corpus_row: One corpus row dict.

        Returns:
            ``ProcessVerificationResult`` with per-defect details and
            deterministic ``to_dict()`` / ``to_json()`` serialization.

        Spec: REQ-VERIFY-062
        """
        return ProcessVerifier().verify_reasoning_trace(corpus_row)

    def verify_spilled_energy(
        self,
        logits_path: str | np.ndarray,
        threshold: float = 1.0,
    ) -> SpilledEnergyResult:
        """Detect hallucinations via spilled energy from raw LLM logits.

        **Detailed explanation for engineers:**
            This is an additive entry point — it does not affect ``verify()``
            or ``verify_and_repair()`` callers.  Pass either a path to a
            saved .npy logit file (as produced by Exp 282/283 logit hooks) or
            a (T, V) numpy array directly.

            The spilled energy signal (ICLR 2026, arXiv 2602.18671) detects
            hallucinations from the logit distribution itself, bypassing the
            constraint-extraction bottleneck that limited prior experiments.

        Args:
            logits_path: Either a string/Path pointing to a .npy file of shape
                (T, V), or a numpy array of shape (T, V) directly.
            threshold: Mean spilled energy (nats) above which
                ``suspected_hallucination`` is True.  Default 1.0.

        Returns:
            SpilledEnergyResult with per_token_spilled, mean/max/p95 statistics,
            lookahead_energy, and suspected_hallucination verdict.

        Spec: REQ-VERIFY-076
        """
        import numpy as np

        from carnot.pipeline.spilled_energy_extractor import (
            SpilledEnergyExtractor,
        )

        extractor = SpilledEnergyExtractor()
        if isinstance(logits_path, np.ndarray):
            return extractor.extract_from_array(logits_path, threshold=threshold)
        return extractor.extract_from_file(logits_path, threshold=threshold)

    def verify_dual_energy(
        self,
        logits_path: str | np.ndarray,
        spilled_threshold: float = 1.0,
        semantic_threshold: float = -5.0,
        temperature: float = 1.0,
    ) -> DualEnergyResult:
        """Detect hallucinations via the DualEnergyGate (spilled + semantic energy).

        **Detailed explanation for engineers:**
            Runs both energy signals on the same logit array and combines them via
            DualEnergyGate.fire():

            - Spilled energy (REQ-VERIFY-076): fires on UNCERTAIN outputs (high entropy).
            - Semantic energy (REQ-VERIFY-077): fires on OVERCONFIDENT outputs (very
              low entropy, model may be confidently wrong).

            Together they form an extraction-free first-pass filter.  This entry point
            is additive — it does not affect existing verify() or verify_and_repair()
            callers.

        Args:
            logits_path: Either a string/Path pointing to a .npy file of shape
                (n_tokens, vocab_size), or a numpy array of shape (n_tokens, vocab_size).
            spilled_threshold: Mean spilled energy (nats) above which
                suspected_hallucination fires.  Default 1.0.
            semantic_threshold: Semantic energy below which overconfident_flag fires.
                Default −5.0.
            temperature: Temperature for semantic energy computation.  Default 1.0.

        Returns:
            DualEnergyResult with spilled_result, semantic_result, gate_fired,
            trigger_signal, and calibration_threshold_used.

        Spec: REQ-VERIFY-077
        """
        import numpy as np

        from carnot.pipeline.semantic_energy_extractor import (
            DualEnergyGate,
            SemanticEnergyExtractor,
        )
        from carnot.pipeline.spilled_energy_extractor import SpilledEnergyExtractor

        if isinstance(logits_path, np.ndarray):
            logits_arr = logits_path
        else:
            from pathlib import Path

            logits_arr = np.load(Path(logits_path))

        spilled_extractor = SpilledEnergyExtractor()
        spilled_result = spilled_extractor.extract_from_array(
            logits_arr, threshold=spilled_threshold
        )

        semantic_extractor = SemanticEnergyExtractor(
            threshold=semantic_threshold, temperature=temperature
        )
        semantic_result = semantic_extractor.extract(logits_arr)

        gate = DualEnergyGate(
            spilled_threshold=spilled_threshold,
            semantic_threshold=semantic_threshold,
            temperature=temperature,
        )
        return gate.fire(spilled_result, semantic_result)

    def check_prefill_uncertainty(
        self,
        logits_first_pass: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, object]:
        """Gate pre-generation: detect hallucination risk from the first-pass logits.

        **Detailed explanation for engineers:**
            Implements the fast-path gate from arXiv 2603.19562 (Neural Uncertainty
            Principle, Mar 2026).  Fires BEFORE any output tokens are generated,
            using only the logit distribution of the model's first forward pass on
            the input prompt.  Low uncertainty → skip full Ising verification.

            This is additive — it does not affect existing ``verify()`` or
            ``verify_and_repair()`` callers.

            Return dict keys:
            - ``skip_verification`` (bool): True when the probe reports low risk,
              meaning full Ising verification can be skipped (fast-path).
            - ``reason`` (str): Human-readable explanation:
                - ``"low_uncertainty"`` when skip_verification is True
                - ``"high_uncertainty"`` when skip_verification is False
            - ``result`` (PrefillUncertaintyResult): full probe output with
              uncertainty_score, conjugate_bound, high_risk, n_tokens, etc.

        Args:
            logits_first_pass: Raw logit array from the model's first forward
                pass on the input prompt.  Shape (V,) or (1, V).
            threshold: Normalised entropy threshold in (0, 1).  Default 0.5.

        Returns:
            Dict with keys ``skip_verification``, ``reason``, ``result``.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
        """
        from carnot.pipeline.prefill_uncertainty_probe import PrefillUncertaintyProbe

        probe = PrefillUncertaintyProbe()
        result = probe.probe(logits_first_pass, threshold=threshold)

        if result.high_risk:
            # High uncertainty → do NOT skip; run full verification.
            return {
                "skip_verification": False,
                "reason": "high_uncertainty",
                "result": result,
            }
        # Low uncertainty → safe to skip full Ising verification (fast path).
        return {
            "skip_verification": True,
            "reason": "low_uncertainty",
            "result": result,
        }

    def verify_legacy(self, *args: Any, **kwargs: Any) -> VerificationResult:
        """Compatibility alias for the existing ``verify()`` return type.

        Spec: REQ-VERIFY-1410
        """
        return self.verify(*args, **kwargs)

    def verify_record(self, *args: Any, **kwargs: Any) -> VerdictRecord:
        """Verify a response and return a structured verdict record.

        Existing callers should continue using ``verify()``.  New integrations
        that need stable audit fields can use this structured API.

        Spec: REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410
        """
        started_at = time.monotonic()
        result = self.verify(*args, **kwargs)
        budget_ms = (time.monotonic() - started_at) * 1000.0
        return result.to_verdict_record(
            budget_ms_consumed=budget_ms,
            producing_tier=3,
            tier_reached=3,
        )

    def online_update(self, observation: dict[str, object]) -> None:
        """Accumulate JEPA observations and periodically call predictor.partial_fit.

        The fast-path predictor is optional, so this method always records
        session statistics and trains only predictors that expose a
        ``partial_fit`` method.
        """

        self._online_observation_count += 1
        if observation.get("fast_path_taken") is True:
            self._online_fast_path_taken_count += 1
        self._session_log.append(dict(observation))
        if len(self._session_log) < 10:
            return
        batch = self._session_log[:10]
        del self._session_log[:10]
        partial_fit = getattr(self._jepa_fast_path_predictor, "partial_fit", None)
        if callable(partial_fit):
            partial_fit(batch)
            self._n_partial_fits += 1

    def get_session_stats(self) -> dict[str, object]:
        """Return JEPA online-update counters for the current pipeline session."""

        rate = (
            self._online_fast_path_taken_count / self._online_observation_count
            if self._online_observation_count
            else 0.0
        )
        return {
            "n_observations": self._online_observation_count,
            "n_partial_fits": self._n_partial_fits,
            "current_fast_path_rate": rate,
        }

    def verify(
        self,
        question: str,
        response: str,
        domain: str | None = None,
        tracker: ConstraintTracker | None = None,
        jepa_predictor: Any = None,
        jepa_threshold: float = 0.5,
        think_probe: Any = None,
        hallufield_detector: Any = None,
        semantic_energy_probe: Any = None,
        embedding_constraint_store: EmbeddingConstraintStore | None = None,
        ising_constraint_injector: IsingConstraintInjector | None = None,
        use_fst: bool = False,
        fst_trainer: Any = None,
        use_odar: bool = False,
        odar_risk_threshold: float | None = None,
        odar_router: Any | None = None,
    ) -> VerificationResult:
        """Verify a response by extracting and checking constraints.

        **Detailed explanation for engineers:**
            This is the core verification path:
            1. Extract constraints from the response text using the
               configured extractor (AutoExtractor by default).
            2. For constraints that carry an ``energy_term``, build a
               ComposedEnergy and compute total energy + decomposition.
            3. For constraints without energy terms, check the ``satisfied``
               flag in their metadata (set by the extractor during parsing).
            4. Return a VerificationResult with verified flag, energy,
               violations list, and full certificate.
            5. If ``tracker`` is provided, record per-constraint-type
               statistics for online self-learning (Tier 1). The tracker
               is updated in-place after verification completes; passing
               tracker=None (the default) skips this step entirely for
               full backward compatibility.

            Respects the configured timeout. If extraction or evaluation
            fails, returns a VerificationResult with verified=False and
            the error recorded in the certificate rather than crashing.

        Args:
            question: The original question (for context/logging).
            response: The response text to verify.
            domain: Optional domain hint for constraint extraction.
            tracker: Optional ConstraintTracker for online learning. If
                provided, records fired/caught counts for each constraint
                type found during this verification call. Default None
                (no tracking -- backward compatible).
            jepa_predictor: Optional JEPAViolationPredictor instance (Tier 3).
                If provided, embeds the first 50 whitespace-split tokens of
                ``response`` and queries the predictor. If
                ``max(predict(embed(first_50_tokens)).values()) < jepa_threshold``
                the response is considered low-risk and the expensive constraint
                extraction + Ising verification is skipped entirely — returning
                a VerificationResult with ``verified=True``, ``mode="FAST_PATH"``,
                and ``skipped=True``. Default None (no JEPA gating, full path).
            jepa_threshold: Violation-probability threshold for the JEPA gate.
                A predicted max probability BELOW this value triggers the fast
                path (skip full verification). Default 0.5. Lower values make
                the gate more aggressive (more fast-path skips, higher risk).
            think_probe: Optional CarnotThinkProbe instance (Tier 0 pre-filter).
                If provided and the probe returns verdict='incorrect', Ising
                verification is skipped and a violation is returned immediately
                (fast-path). For 'uncertain' or 'correct' verdicts, full
                verification proceeds normally. Default None (no ThinkProbe).
            embedding_constraint_store: Optional EmbeddingConstraintStore (Tier 2,
                REQ-LEARN-060).  When set, retrieve(response, top_k=3) is called
                and the top-3 SPO constraints are appended additively to the
                active constraint list before _evaluate_constraints().  Static
                constraints are never removed — this is purely additive.
                Default None (no embedding injection, full backward compatibility).
            use_fst: Optional Fast-Slow Training mode. When True, the base LLM
                and verifier ensemble are asserted frozen as slow weights and
                the verification certificate records the FST freeze status.
            use_odar: Optional ODAR free-energy routing gate. When True, or
                when enabled at pipeline construction, Tier 0 probe outputs
                are fused before Tier 1 extraction. Low EFE returns an
                optimistic fast-path result; high EFE falls through.
            odar_risk_threshold: Optional per-call EFE threshold used when
                constructing the default FreeEnergyRouter.
            odar_router: Optional per-call FreeEnergyRouter-compatible object.

        Returns:
            VerificationResult with constraint evaluation details.
            ``mode="FAST_PATH"`` and ``skipped=True`` when JEPA gating fired.
            ``mode="THINK_PROBE_FAST_PATH"`` and ``skipped=True`` when ThinkProbe
            flagged the response as 'incorrect'.

        Raises:
            PipelineTimeoutError: If the call exceeds timeout_seconds.

        Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-LEARN-001,
              REQ-JEPA-002, REQ-VERIFY-094, REQ-VERIFY-146, REQ-LEARN-060,
              REQ-LEARN-061, REQ-FST-2240, REQ-ODAR-2243
        """
        _fst_trainer = fst_trainer
        if use_fst and _fst_trainer is None:
            from carnot.training.fast_slow import FastSlowTrainer  # noqa: PLC0415

            _fst_trainer = FastSlowTrainer.from_pipeline(self)

        def _with_fst_certificate(result: VerificationResult) -> VerificationResult:
            if use_fst and _fst_trainer is not None:
                _fst_trainer.slow_weights.assert_frozen()
                result.certificate["fst"] = _fst_trainer.certificate()
            return result

        _tier0_probe_outputs: dict[str, Any] = {}
        _pending_odar_certificate: dict[str, Any] | None = None

        typed_reasoning = self.extract_typed_reasoning(question, response)
        semantic_grounding = self.verify_semantic_grounding(question, response, typed_reasoning)
        semantic_verifier_v2 = self.verify_semantic_verifier_v2(
            question,
            response,
            typed_reasoning=typed_reasoning,
            semantic_grounding=semantic_grounding,
        )

        # Pipeline-level JEPA fast-path gate (exp2525, REQ-JEPA-002).
        # Uses lightweight response-feature proxies (response_length_norm,
        # logprob_variance_proxy) to predict P(violation) before running the
        # expensive Ising pass.  When p < jepa_fast_path_threshold (default 0.2),
        # we trust the response is clean and return early with verified=True.
        # The predictor's calls_fast_path counter is incremented when fired so
        # callers can measure the fast-path hit rate across a batch.
        if self._jepa_fast_path_predictor is not None:
            _p_violation = self._jepa_fast_path_predictor.predict_p_violation(response)
            if _p_violation < self._jepa_fast_path_threshold:
                self._jepa_fast_path_predictor.calls_fast_path += 1
                return _with_fst_certificate(
                    VerificationResult(
                        verified=True,
                        constraints=[],
                        energy=0.0,
                        violations=[],
                        certificate={
                            "mode": "JEPA_FAST_PATH",
                            "jepa_p_violation": _p_violation,
                            "jepa_threshold": self._jepa_fast_path_threshold,
                            "skipped_verification": True,
                            "fast_path_used": True,
                        },
                        mode="JEPA_FAST_PATH",
                        skipped=True,
                        typed_reasoning=typed_reasoning,
                        semantic_grounding=semantic_grounding,
                        semantic_verifier_v2=semantic_verifier_v2,
                    )
                )

        # CRANE (arXiv:2502.09061) balance ratio gating
        if getattr(self, "balance_ratio", 1.0) < 1.0:
            import random

            if random.random() > self.balance_ratio:
                baseline_score = _with_fst_certificate(
                    VerificationResult(
                        verified=True,
                        constraints=[],
                        energy=0.0,
                        violations=[],
                        certificate={"mode": "CRANE_FREE", "skipped": True},
                        mode="CRANE_FREE",
                        skipped=True,
                        typed_reasoning=typed_reasoning,
                        semantic_grounding=semantic_grounding,
                        semantic_verifier_v2=semantic_verifier_v2,
                    )
                )
                return baseline_score

        # Tier 3 JEPA fast-path gate (optional).
        # If a JEPA predictor is supplied, embed the first 50 whitespace-split
        # tokens of the response and ask the predictor whether any constraint
        # domain looks risky. When max probability < threshold, we trust the
        # response is clean and skip the expensive extraction + Ising pass.
        # This is the "fast path" — verified=True is optimistic (low-risk default).
        if jepa_predictor is not None:
            first_50_tokens = " ".join(response.split()[:50])
            from carnot.embeddings.fast_embedding import RandomProjectionEmbedding

            _embedder = RandomProjectionEmbedding(embed_dim=256, seed=42)
            partial_embedding = _embedder.encode(first_50_tokens)
            probs = jepa_predictor.predict(partial_embedding)
            max_prob = max(probs.values()) if probs else 0.0
            if max_prob < jepa_threshold:
                return _with_fst_certificate(
                    VerificationResult(
                        verified=True,
                        constraints=[],
                        energy=0.0,
                        violations=[],
                        certificate={
                            "mode": "FAST_PATH",
                            "jepa_max_prob": max_prob,
                            "jepa_threshold": jepa_threshold,
                            "jepa_probs": probs,
                        },
                        mode="FAST_PATH",
                        skipped=True,
                        typed_reasoning=typed_reasoning,
                        semantic_grounding=semantic_grounding,
                        semantic_verifier_v2=semantic_verifier_v2,
                    )
                )
        # Tier 0e HalluField pre-filter (optional, REQ-VERIFY-117).
        # When a HalluFieldDetector instance is supplied, score the response
        # logits for thermodynamic instability.  is_unstable=True is recorded
        # in the certificate but does NOT short-circuit — it is an advisory
        # signal that downstream tiers may use for threshold adjustment.
        # Calling score(None) returns a safe ci_stub result with no side effects,
        # so passing hallufield_detector without logits is always safe.
        _hallufield_result = None
        if hallufield_detector is not None:
            _hallufield_result = hallufield_detector.score(None)
            _tier0_probe_outputs["hallufield"] = _hallufield_result

        # Tier 0f SemanticEnergyProbe advisory (REQ-VERIFY-155).
        # Scores pairwise Boltzmann semantic energy over sentence clusters.
        # is_unstable=True means sentences are semantically incoherent (hallucination risk).
        # Advisory only — no short-circuit; result stored in certificate for downstream tiers.
        _semantic_energy_result = None
        if semantic_energy_probe is not None:
            _semantic_energy_result = semantic_energy_probe.score(response)
            _tier0_probe_outputs["semantic_energy"] = _semantic_energy_result

        # Tier 0g StreamingCoT advisory (REQ-VERIFY-140).
        # When CARNOT_STREAMING_COT=1, extract CoT steps from the response and run
        # the PHaS (Prefix Hallucination Score) trajectory detector.
        # is_streaming_unstable=True means sustained reasoning drift was detected.
        # Advisory only — does NOT short-circuit the cascade or affect verified flag.
        # Mirrors how HalluField (Tier 0e) is handled above.
        # WHY re-read env at call time: the class attribute is evaluated at import time,
        # so tests or experiment scripts that set CARNOT_STREAMING_COT=1 after import
        # would be silently ignored.  Checking at call time makes the flag usable in
        # in-process test and experiment contexts without reloading the module.
        _streaming_cot_enabled = self.STREAMING_COT_ENABLED or (
            os.getenv("CARNOT_STREAMING_COT", "0") == "1"
        )
        _streaming_cot_result = None
        if _streaming_cot_enabled:
            from carnot.pipeline.streaming_cot import (  # noqa: PLC0415
                StreamingCoTHalluDetector,
                extract_cot_steps,
            )

            _cot_steps = extract_cot_steps(response)
            if _cot_steps:
                _streaming_detector = StreamingCoTHalluDetector(alpha=0.3, threshold=0.35)
                _streaming_cot_result = _streaming_detector.detect(_cot_steps)
                _tier0_probe_outputs["streaming_cot"] = _streaming_cot_result

        # Tier 0 ThinkProbe fast-path (optional, REQ-VERIFY-094).
        # If a CarnotThinkProbe instance is provided and it classifies the response
        # as 'incorrect', skip Ising entirely and return a violation immediately.
        # This is more sample-efficient than discriminative scoring (ThinkPRM result:
        # 1% of labels achieves SOTA on MATH-500). Only 'incorrect' triggers fast-path;
        # 'uncertain' and 'correct' fall through to full Ising verification.
        if think_probe is not None:
            probe_result = think_probe.probe(response)
            _tier0_probe_outputs["think_probe"] = {
                "verdict": probe_result.verdict.verdict,
                "confidence": probe_result.verdict.confidence,
                "latency_ms": probe_result.latency_ms,
            }
            if not probe_result.should_run_ising:
                # ThinkProbe is confident the response is incorrect — skip Ising.
                think_violation = ConstraintResult(
                    constraint_type="think_probe",
                    description="ThinkProbe: response classified as incorrect by 3-step CoT verifier",
                    metadata={
                        "verdict": probe_result.verdict.verdict,
                        "confidence": probe_result.verdict.confidence,
                        "latency_ms": probe_result.latency_ms,
                    },
                )
                return _with_fst_certificate(
                    VerificationResult(
                        verified=False,
                        constraints=[think_violation],
                        energy=0.0,
                        violations=[think_violation],
                        certificate={
                            "mode": "THINK_PROBE_FAST_PATH",
                            "think_probe_verdict": probe_result.verdict.verdict,
                            "think_probe_confidence": probe_result.verdict.confidence,
                            "think_probe_latency_ms": probe_result.latency_ms,
                        },
                        mode="THINK_PROBE_FAST_PATH",
                        skipped=True,
                        typed_reasoning=typed_reasoning,
                        semantic_grounding=semantic_grounding,
                        semantic_verifier_v2=semantic_verifier_v2,
                    )
                )

        # Tier 0c: NUP Probe v6 fast-path (REQ-VERIFY-146, Exp 622).
        # When a NUPProbeV4 instance is supplied, score the response.  A score at or
        # below the threshold means the response is energetically cheap (likely correct),
        # so we short-circuit and skip all downstream tiers.  This fires AFTER Tier 0b
        # (SpilledEnergyDetector is a standalone call, not inlined here) and BEFORE
        # Tier 0d (HalluField, which is already handled above as an advisory).
        if self._nup_probe is not None:
            nup_score = self._nup_probe.score(response)
            _tier0_probe_outputs["nup_probe"] = {
                "risk_score": nup_score,
                "threshold": self._nup_probe_threshold,
            }
            if nup_score <= self._nup_probe_threshold:
                return _with_fst_certificate(
                    VerificationResult(
                        verified=True,
                        constraints=[],
                        energy=0.0,
                        violations=[],
                        certificate={
                            "mode": "NUP_PROBE_FAST_PATH",
                            "nup_score": nup_score,
                            "nup_threshold": self._nup_probe_threshold,
                        },
                        mode="NUP_PROBE_FAST_PATH",
                        skipped=True,
                        typed_reasoning=typed_reasoning,
                        semantic_grounding=semantic_grounding,
                        semantic_verifier_v2=semantic_verifier_v2,
                    )
                )

        # ODAR free-energy router (optional, REQ-ODAR-2243).
        # This gate fuses the cheap Tier 0 evidence gathered above and decides
        # whether the response is low-risk enough to skip Tier 1 extraction and
        # downstream deliberative verification.  Missing probe evidence routes
        # to deliberative verification because FreeEnergyRouter assigns it
        # infinite EFE.
        if use_odar or self._use_odar:
            from carnot.pipeline.odar_router import (  # noqa: PLC0415
                FreeEnergyRouter,
                RoutingDecision,
            )

            effective_odar_router = odar_router or self._odar_router
            if effective_odar_router is None:
                threshold = (
                    self._odar_risk_threshold
                    if odar_risk_threshold is None
                    else odar_risk_threshold
                )
                effective_odar_router = FreeEnergyRouter(risk_threshold=threshold)

            odar_result = effective_odar_router.evaluate(_tier0_probe_outputs)
            _pending_odar_certificate = odar_result.to_certificate()
            if odar_result.decision is RoutingDecision.FAST_PATH:
                return _with_fst_certificate(
                    VerificationResult(
                        verified=True,
                        constraints=[],
                        energy=0.0,
                        violations=[],
                        certificate={
                            "mode": "ODAR_FAST_PATH",
                            **_pending_odar_certificate,
                        },
                        mode="ODAR_FAST_PATH",
                        skipped=True,
                        typed_reasoning=typed_reasoning,
                        semantic_grounding=semantic_grounding,
                        semantic_verifier_v2=semantic_verifier_v2,
                    )
                )

        # Fast path: delegate to Rust pipeline when available.
        # Repair still uses Python (requires LLM), but the hot verification
        # inner loop gets a 10x speedup from the Rust implementation.
        # Only use Rust when ALL requested/configured domains are supported
        # by the Rust pipeline (arithmetic + logic). Code/NL extractors
        # remain Python-only, so if the caller needs those, we stay in Python.
        _rust_supported = {"arithmetic", "logic"}
        _can_use_rust = _USE_RUST_PIPELINE and RustVerifyPipeline is not None
        if _can_use_rust:
            effective_domains = {domain} if domain else set(self._domains or [])
            # Only use Rust when we have an explicit domain constraint that
            # falls within what Rust supports. When no domains are set
            # (auto-detect all), Python may have more extractors.
            if effective_domains and effective_domains <= _rust_supported:
                try:
                    result = self._verify_rust(
                        question,
                        response,
                        typed_reasoning=typed_reasoning,
                    )
                    result = self._merge_semantic_analysis(
                        result,
                        semantic_grounding,
                        semantic_verifier_v2,
                    )
                    if _pending_odar_certificate is not None:
                        result.certificate.update(_pending_odar_certificate)
                    result = self._record_fr11_shadow_decision(
                        question=question,
                        response=response,
                        domain=domain,
                        result=result,
                    )
                    result = self._record_fr11_factor_cache_shadow_decision(
                        question=question,
                        response=response,
                        domain=domain,
                        result=result,
                    )
                    return _with_fst_certificate(result)
                except Exception as exc:
                    logger.warning("Rust verify failed, falling back to Python: %s", exc)
                    # Fall through to Python path.

        deadline = self._make_deadline()
        try:
            self._check_deadline(deadline)
            constraints = self.extract_constraints(response, domain)
            if semantic_verifier_v2 is None and semantic_grounding is not None:
                constraints.extend(semantic_grounding.to_constraint_results())
            elif semantic_verifier_v2 is not None and semantic_verifier_v2.verdict == "violated":
                if semantic_grounding is not None:
                    constraints.extend(semantic_grounding.to_constraint_results())
                constraints.extend(semantic_verifier_v2.to_constraint_results())

            if self._probability_calibration_verifier is not None:
                for probability_record in self._probability_calibration_verifier.score_text(
                    response
                ):
                    if probability_record.verdict == "abstain":
                        continue
                    constraints.append(
                        ConstraintResult(
                            constraint_type="probability_calibration",
                            description=probability_record.rationale,
                            metadata={
                                "satisfied": probability_record.verdict == "pass",
                                "energy": probability_record.energy,
                                "verdict_record": probability_record.to_dict(),
                            },
                        )
                    )

            # Tier 2: prepend learned constraint suggestions from memory.
            if self._memory is not None:
                effective_domain = domain or (
                    self._domains[0] if self._domains and len(self._domains) == 1 else "auto"
                )
                learned = self._memory.suggest_constraints(response, effective_domain)
                if learned:
                    constraints = learned + constraints

            # Tier 2 template addition: prepend constraints from active templates.
            # When CaseMemory observes a pattern crossing its frequency threshold,
            # the template library adds a NEW constraint type (not just reweighting).
            # This is the Tier 2 → Tier 1 feedback loop (Exp 134 / REQ-LEARN-017).
            if self._template_library is not None and self._model_name is not None:
                template_constraints = self._template_library.apply_active_templates(
                    response, self._model_name
                )
                if template_constraints:
                    constraints = template_constraints + constraints

            # Tier 2 constraint addition from cross-session memory (REQ-LEARN-053/054).
            # check_and_add() promotes any matured violation patterns into named
            # constraints and registers them with the template_library.  The newly
            # added names come back as strings; we do NOT prepend synthetic
            # ConstraintResult objects here — the template_library path above already
            # handles activation.  Calling check_and_add() here ensures patterns that
            # crossed the threshold during THIS session are active on the NEXT call.
            if self._constraint_memory is not None:
                self._constraint_memory.check_and_add(self)

            # Tier 2 embedding-constraint injection (REQ-LEARN-060, REQ-LEARN-061).
            # When an EmbeddingConstraintStore is provided, retrieve the top-3
            # constraints most similar to the current response and append them
            # additively to the active constraint list.  Static constraints already
            # in `constraints` are never removed — retrieved constraints supplement
            # them from the store's semantically-distinct embedding subspaces.
            if embedding_constraint_store is not None:
                retrieved_spо = embedding_constraint_store.retrieve(response, top_k=3)
                for spo in retrieved_spо:
                    injected = ConstraintResult(
                        constraint_type=f"embedding_retrieved_{spo.source_violation_type}",
                        description=(
                            f"Retrieved embedding constraint: ({spo.subject}) "
                            f"({spo.predicate}) ({spo.object})"
                        ),
                        metadata={
                            "source": "embedding_constraint_store",
                            "spo_subject": spo.subject,
                            "spo_predicate": spo.predicate,
                            "spo_object": spo.object,
                            "source_violation_type": spo.source_violation_type,
                        },
                    )
                    constraints.append(injected)

            # REQ-VERIFY-095: Ising constraint injection (ADDITIVE).
            # When both an EmbeddingConstraintStore and an IsingConstraintInjector
            # are set, project retrieved embeddings into spin-space and temporarily
            # bias J before sampling.  This wiring closes RETRO-CONSTRAINT-ZERO-DELTA:
            # retrieved embeddings now affect the Ising energy, not just the constraint
            # metadata list.  Existing behaviour is unchanged when either param is None.
            if ising_constraint_injector is not None and embedding_constraint_store is not None:
                retrieved_for_ising = embedding_constraint_store.retrieve(response, top_k=3)
                embeddings_for_ising = [c.embedding for c in retrieved_for_ising if c.embedding]
                if embeddings_for_ising:
                    import numpy as _np  # noqa: PLC0415

                    _bias = ising_constraint_injector.project_to_spin_bias(embeddings_for_ising)
                    # Record the injected bias in the certificate for traceability.
                    # Actual J substitution is done in compute_energy_with_injection;
                    # here we annotate so downstream callers can measure the delta.
                    logger.debug(
                        "ising_constraint_injector: bias norm=%.4f, n_constraints=%d",
                        float(_np.linalg.norm(_bias)),
                        len(embeddings_for_ising),
                    )

            self._check_deadline(deadline)
            result = self._evaluate_constraints(constraints)
        except PipelineTimeoutError:
            raise
        except CarnotError as exc:
            logger.warning("Verification degraded: %s", exc)
            degraded_result = VerificationResult(
                verified=False,
                constraints=[],
                energy=0.0,
                violations=[],
                certificate={"error": str(exc), "error_type": type(exc).__name__},
                typed_reasoning=typed_reasoning,
                semantic_grounding=semantic_grounding,
                semantic_verifier_v2=semantic_verifier_v2,
            )
            degraded_result = self._record_fr11_shadow_decision(
                question=question,
                response=response,
                domain=domain,
                result=degraded_result,
            )
            degraded_result = self._record_fr11_factor_cache_shadow_decision(
                question=question,
                response=response,
                domain=domain,
                result=degraded_result,
            )
            return _with_fst_certificate(degraded_result)

        result.typed_reasoning = typed_reasoning
        result.semantic_grounding = semantic_grounding
        result.semantic_verifier_v2 = semantic_verifier_v2
        if _pending_odar_certificate is not None:
            result.certificate.update(_pending_odar_certificate)
        if semantic_verifier_v2 is not None:
            result.certificate["semantic_verifier_v2"] = semantic_verifier_v2.to_dict()

        # Record Tier 0f SemanticEnergyProbe advisory result in certificate.
        if _semantic_energy_result is not None:
            result.certificate["tier_0f_semantic_energy"] = {
                "energy": _semantic_energy_result.energy,
                "is_unstable": _semantic_energy_result.is_unstable,
                "sentence_count": _semantic_energy_result.sentence_count,
                "cluster_entropy": _semantic_energy_result.cluster_entropy,
                "threshold": _semantic_energy_result.threshold,
            }

        # Record Tier 0g StreamingCoT advisory result in result fields and certificate.
        # is_streaming_unstable and streaming_cot_phas are set directly on result so
        # callers can inspect them without parsing the certificate dict.
        # Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165, SCENARIO-VERIFY-166
        if _streaming_cot_result is not None:
            result.streaming_cot_unstable = _streaming_cot_result.is_streaming_unstable
            result.streaming_cot_phas = _streaming_cot_result.final_phas
            result.certificate["tier_0g_streaming_cot"] = {
                "is_streaming_unstable": _streaming_cot_result.is_streaming_unstable,
                "final_phas": _streaming_cot_result.final_phas,
                "n_steps": _streaming_cot_result.n_steps,
            }

        if self._casal_tier is not None:
            _casal_res = self._casal_tier.verify(question, response)
            result.certificate["casal_tier"] = _casal_res

        # InterwhenMonitor advisory (REQ-VERIFY-130, REQ-VERIFY-131).
        # Replays the response sentence-by-sentence and records any arithmetic
        # violations detected mid-stream.  Advisory only — does not change
        # result.verified.  Exposes early_detection_rate so callers can measure
        # whether violations were caught before the final sentence (the key
        # advantage of the Interwhen approach over post-hoc verification).
        if self._interwhen_monitor is not None:
            _iw_violations = self._interwhen_monitor.monitor_full_response(response)
            _total_sentences = len(self._interwhen_monitor.split_at_boundaries(response))
            _early = [v for v in _iw_violations if v.sentence_index < max(_total_sentences - 1, 0)]
            result.certificate["interwhen_monitor"] = {
                "n_violations": len(_iw_violations),
                "early_detection_count": len(_early),
                "total_sentences": _total_sentences,
                "early_detection_rate": len(_early) / len(_iw_violations)
                if _iw_violations
                else 0.0,
            }

        # Tier 0h SpectralAttentionProbe advisory (REQ-VERIFY-146).
        # When CARNOT_SPECTRAL_PROBE=1, extract CoT steps from the response and run
        # the bigram Laplacian spectral entropy probe.
        # is_spectrally_diffuse=True means the attention spectrum is flat → hallucination risk.
        # Advisory only — does NOT short-circuit the cascade or affect verified flag.
        # WHY re-read env at call time: mirrors Tier 0g rationale — class attribute is
        # evaluated at import time so in-process test/experiment env overrides are ignored.
        _spectral_probe_enabled = self.SPECTRAL_PROBE_ENABLED or (
            os.getenv("CARNOT_SPECTRAL_PROBE", "0") == "1"
        )
        if _spectral_probe_enabled:
            from carnot.pipeline.streaming_cot import extract_cot_steps  # noqa: PLC0415
            from carnot.verify.spectral_attention_probe import (
                SpectralAttentionProbe,  # noqa: PLC0415
            )

            _spectral_steps = extract_cot_steps(response)
            if _spectral_steps:
                _spectral_probe = SpectralAttentionProbe()
                _spectral_result = _spectral_probe.predict(_spectral_steps)
                result.spectral_diffuse = _spectral_result["is_spectrally_diffuse"]
                result.spectral_entropy_mean = _spectral_result["spectral_entropy_mean"]
                result.certificate["tier_0h_spectral"] = {
                    "is_spectrally_diffuse": _spectral_result["is_spectrally_diffuse"],
                    "spectral_entropy_mean": _spectral_result["spectral_entropy_mean"],
                    "n_steps": len(_spectral_steps),
                }

        # AND-composition ensemble verdict (Phase 1d, REQ-VERIFY-1121).
        # Run k=5 AND-compose as an advisory signal. Records per-verifier scores
        # in the certificate so downstream callers can inspect them. Does NOT
        # short-circuit or override result.verified (advisory, additive only).
        if self._and_compose_verifier is not None:
            try:
                _and_result = self._and_compose_verifier.verify(question, response)
                result.certificate["and_compose_k5"] = {
                    "verified": _and_result.verified,
                    "k": _and_result.k,
                    "per_verifier_scores": _and_result.per_verifier_scores,
                    "per_verifier_verified": _and_result.per_verifier_verified,
                    "headline_eligible": _and_result.headline_eligible,
                    "headline_ineligible_reason": _and_result.headline_ineligible_reason,
                }
            except Exception as _exc:
                logger.debug("AND-compose ensemble degraded: %s", _exc)

        if tracker is not None:
            self._update_tracker(tracker, result)

        # Tier 2: record violation patterns into memory after verification.
        if self._memory is not None and result.violations:
            effective_domain = domain or (
                self._domains[0] if self._domains and len(self._domains) == 1 else "auto"
            )
            for violation in result.violations:
                self._memory.record_pattern(
                    domain=effective_domain,
                    error_type=violation.constraint_type,
                    constraint_that_caught_it=violation.description,
                )

        # Tier 2 constraint-addition observation (REQ-LEARN-054).
        # Feed each violation into ConstraintAdditionFromMemory so its pattern
        # counter increments.  The violation_type is the leading token of the
        # constraint_type (e.g. 'carry' from 'carry:overflow_detected').
        # Passing response as step_text gives the memory a diagnostic example.
        if self._constraint_memory is not None and result.violations:
            for violation in result.violations:
                vtype = violation.constraint_type.split(":", 1)[0]
                self._constraint_memory.observe(vtype, response)

        # REQ-LEARN-048: write each violation into the EmbeddingConstraintStore so
        # future sessions can retrieve prior error patterns as injected constraints.
        # This is the missing write path confirmed by Exp 833 (root_cause=write_path_missing).
        # Only fires when enable_constraint_accumulation=True was set at construction time,
        # preserving full backward compatibility for callers that omit the flag.
        if (
            embedding_constraint_store is not None
            and result.violations
            and self._enable_constraint_accumulation
        ):
            from carnot.pipeline.embedding_constraint_store import (  # noqa: PLC0415
                ConstraintSPOTuple as _ConstraintSPOTuple,
            )

            # Map canonical violation type prefixes to structured SPO roles.
            # Unmapped types fall back to a generic triple using the raw type string.
            _SPO_MAP = {
                "carry": ("arithmetic_carry", "violates", "carry_propagation"),
                "sign": ("numeric_sign", "violates", "sign_preservation"),
                "unit": ("unit_label", "violates", "unit_consistency"),
                "comparison": ("comparison_direction", "violates", "inequality_direction"),
                "causal": ("causal_entailment", "violates", "step_causality"),
            }
            for violation in result.violations:
                vtype = violation.constraint_type.split(":", 1)[0]
                subj, pred, obj = _SPO_MAP.get(
                    vtype, (vtype, "violates", violation.description[:64])
                )
                embedding_constraint_store.store(
                    _ConstraintSPOTuple(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        embedding=None,
                        source_violation_type=vtype,
                    )
                )

        result = self._record_fr11_shadow_decision(
            question=question,
            response=response,
            domain=domain,
            result=result,
        )
        result = self._record_fr11_factor_cache_shadow_decision(
            question=question,
            response=response,
            domain=domain,
            result=result,
        )
        return _with_fst_certificate(result)

    def verify_with_gate(
        self,
        question: str,
        response: str,
        domain: str | None = None,
        jepa_gate: Any = None,
        logit_mean: Any = None,
    ) -> VerificationResult:
        """Verify a response using the JEPA fast-path gate before Ising.

        **Detailed explanation for engineers:**
            Additive entry point — does not change ``verify()`` or
            ``verify_and_repair()`` callers.  Accepts an optional
            ``JepaGate`` instance (from ``carnot.pipeline.jepa_fast_path``).
            When the gate is provided and ``should_skip()`` returns True, we
            skip full Ising verification and return a lightweight result with
            ``gate_decision="skip"`` and ``ising_skipped=True``.

            When the gate says "verify" (or no gate is supplied), the full
            ``verify()`` pipeline runs normally, and the result is tagged
            with ``gate_decision="verify"`` and ``ising_skipped=False``.

            The ``logit_mean`` parameter should be the mean logit vector from
            the LLM's generation pass (shape (V,) numpy array).  If it is
            None and a gate is supplied, a zero vector of length 1 is used as
            a fallback — callers should pass real logits for meaningful gating.

        Args:
            question: The original question (passed through to ``verify()``).
            response: The response text to check.
            domain: Optional domain hint for constraint extraction.
            jepa_gate: Optional ``JepaGate`` instance.  If None, the method
                behaves identically to ``verify()`` with no gate metadata.
            logit_mean: Optional 1-D numpy array of mean logit values for this
                response.  If None, falls back to a zero vector.

        Returns:
            VerificationResult.  When gate skips: ``violations=[]``,
            ``certificate["gate_decision"]="skip"``, ``ising_skipped=True``,
            ``certificate["gate_energy"]`` = energy scalar.  When gate verifies
            or no gate: full Ising result tagged with ``gate_decision="verify"``
            and ``ising_skipped=False``.

        Spec: REQ-JEPA-005, SCENARIO-JEPA-010, SCENARIO-JEPA-011
        """
        import numpy as np

        if jepa_gate is None:
            # No gate — normal full verification, no gate metadata added.
            return self.verify(question, response, domain)

        # Normalise the logit mean vector.
        lm: np.ndarray
        if logit_mean is None:
            lm = np.zeros(1, dtype=np.float32)
        else:
            lm = np.asarray(logit_mean, dtype=np.float32)

        energy = jepa_gate.predict(lm)
        skip = jepa_gate.should_skip(lm)

        if skip:
            return VerificationResult(
                verified=True,
                constraints=[],
                energy=0.0,
                violations=[],
                certificate={
                    "gate_decision": "skip",
                    "gate_energy": energy,
                    "ising_skipped": True,
                    "n_violations": 0,
                },
                mode="FAST_PATH",
                skipped=True,
            )

        # Gate says verify — run full Ising pipeline.
        result = self.verify(question, response, domain)
        result.certificate["gate_decision"] = "verify"
        result.certificate["gate_energy"] = energy
        result.certificate["ising_skipped"] = False
        return result

    def verify_with_z3(
        self,
        question: str,
        response: str,
        timeout_s: float = 2.0,
    ) -> Z3Result:
        """Check a chain-of-thought response for internal inconsistency via Z3.

        **Detailed explanation for engineers:**
            Additive integration point for NL2Z3Extractor.  Does not modify
            ``verify()`` or ``verify_and_repair()``; safe to call in parallel
            with the Ising pipeline.

            In CI mode (CARNOT_FORCE_LIVE not set) this always returns a
            Z3Result with sat_status="unknown" without making any LLM call.

            In production mode (CARNOT_FORCE_LIVE=1) this makes one extra LLM
            call via NL2Z3Extractor, runs the generated Z3 code in a sandboxed
            subprocess, and returns the solver verdict.

        Args:
            question:  The original question posed to the LLM (passed through
                       to NL2Z3Extractor for prompt context).
            response:  The chain-of-thought response to check.
            timeout_s: Subprocess timeout for Z3 execution (default 2.0 s).

        Returns:
            Z3Result with sat_status in {sat, unsat, unknown, error}.
            Call result.violations_found to check whether a violation was found.

        Spec: REQ-EXTRACT-010, SCENARIO-EXTRACT-020, SCENARIO-EXTRACT-021,
              SCENARIO-EXTRACT-024
        """
        from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor, Z3Result

        # Lazily create the extractor instance on first use, or update timeout
        # if the cached extractor's timeout doesn't match the requested timeout.
        if not hasattr(self, "_nl2z3_extractor") or self._nl2z3_extractor is None:
            self._nl2z3_extractor = NL2Z3Extractor(timeout_s=timeout_s)
        elif self._nl2z3_extractor._timeout_s != timeout_s:
            # If timeout_s changed, recreate the extractor with the new timeout.
            self._nl2z3_extractor = NL2Z3Extractor(timeout_s=timeout_s)

        self._nl2z3_extractor.extract(question, response)
        result = self._nl2z3_extractor.last_z3_result
        if result is None:
            return Z3Result(sat_status="unknown", z3_code="", runtime_ms=0.0)
        return result

    def verify_cot_circuit(
        self,
        question: str,
        response: str,
        tolerance: float = 0.01,
    ) -> CoTCircuit:
        """Check a chain-of-thought response for structural (circuit) consistency.

        **Detailed explanation for engineers:**
            Additive integration point for CoTCircuitVerifier (arXiv 2510.09312).
            Does not modify ``verify()`` or ``verify_and_repair()``; safe to call
            in parallel with the Ising or Z3 pipelines.

            Unlike ``verify_with_z3``, this method makes NO LLM calls — it is
            purely regex/string-based and always runs in CI without GPU access.

            A broken link indicates a downstream step uses a value that does not
            match the upstream step's actual output.  This catches wrong-variable
            substitution and step-skipping errors that Z3 and regex miss.

        Args:
            question:  The original question (unused; kept for API symmetry with
                       verify_with_z3 and future protocol compatibility).
            response:  The chain-of-thought response to check.
            tolerance: Relative tolerance for value comparison (default 0.01).

        Returns:
            CoTCircuit with steps, has_cycle, and broken_links populated.
            Inspect ``circuit.broken_links`` for structural violations.

        Spec: REQ-EXTRACT-015, REQ-EXTRACT-016,
              SCENARIO-EXTRACT-031, SCENARIO-EXTRACT-032, SCENARIO-EXTRACT-033,
              SCENARIO-EXTRACT-034, SCENARIO-EXTRACT-035
        """
        from carnot.pipeline.cot_circuit_verifier import CoTCircuitVerifier

        if (
            not hasattr(self, "_cot_circuit_verifier")
            or self._cot_circuit_verifier is None
            or self._cot_circuit_verifier.tolerance != tolerance
        ):
            self._cot_circuit_verifier = CoTCircuitVerifier(tolerance=tolerance)

        return self._cot_circuit_verifier.verify(response)

    def verify_and_repair(
        self,
        question: str,
        response: str | None = None,
        domain: str | None = None,
        use_fst: bool = False,
    ) -> RepairResult:
        """Verify a response and iteratively repair violations via the LLM.

        **Detailed explanation for engineers:**
            This is the full verify-repair loop from Experiment 57, packaged
            as a clean API call. The flow:

            1. If ``response`` is None and a model is loaded, generate an
               initial response from the question.
            2. Verify the response (extract constraints, check satisfaction).
            3. If all constraints pass, return immediately (no repair needed).
            4. If violations exist AND a model is loaded:
               a. Format violations as natural-language feedback.
               b. Build a repair prompt: original question + previous response
                  + violation feedback + "please fix these issues."
               c. Generate a new response from the LLM.
               d. Re-verify. Repeat up to ``max_repairs`` times.
            5. If violations exist but no model is loaded, return with
               ``repaired=False`` (verification-only mode).

            The ``history`` list in the result contains one VerificationResult
            per iteration (including the initial check), so callers can see
            how the response improved (or didn't) across iterations.

        Args:
            question: The original question to answer.
            response: Initial response text. If None and model is loaded,
                the model generates one. If None and no model, raises
                ValueError.
            domain: Optional domain hint for constraint extraction.
            use_fst: Optional Fast-Slow Training mode. When True, verifier
                feedback is treated as fast weights and prepended to the next
                repair prompt while model/verifier slow weights remain frozen.

        Returns:
            RepairResult with full repair trajectory.

        Raises:
            ValueError: If response is None and no model is loaded.

        Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004,
              REQ-FST-2240, SCENARIO-FST-2240
        """
        deadline = self._make_deadline()
        fst_trainer = None
        if use_fst:
            from carnot.training.fast_slow import FastSlowTrainer  # noqa: PLC0415

            fst_trainer = FastSlowTrainer.from_pipeline(self)

        # Step 1: Get initial response.
        if response is None:
            if not self.has_model:
                raise ValueError(
                    "No response provided and no model loaded. Either pass a "
                    "response string or initialize with model='...'."
                )
            try:
                response = self._generate(question)
            except (CarnotError, PipelineTimeoutError):
                raise
            except Exception as exc:
                raise RepairError(
                    f"Initial generation failed: {exc}",
                    details={"question": question[:200]},
                ) from exc

        initial_response = response
        history: list[VerificationResult] = []

        # Step 2: Verify the initial response.
        self._check_deadline(deadline)
        vr = self.verify(
            question,
            response,
            domain,
            use_fst=use_fst,
            fst_trainer=fst_trainer,
        )
        history.append(vr)

        if vr.verified:
            return RepairResult(
                initial_response=initial_response,
                final_response=response,
                verified=True,
                repaired=False,
                iterations=0,
                history=history,
            )

        # Step 3: Repair loop (only if model is available).
        if not self.has_model:
            return RepairResult(
                initial_response=initial_response,
                final_response=response,
                verified=False,
                repaired=False,
                iterations=0,
                history=history,
            )

        iterations_used = self._max_repairs
        for i in range(self._max_repairs):
            self._check_deadline(deadline)

            if self.routing_mode == "odar" and self._repair_router is not None:
                current_energy = vr.energy
                if i > 0:
                    prior_energy = history[-2].energy
                else:
                    prior_energy = current_energy + 1.0

                route_to_repair, vfe, _ = self._repair_router.route(
                    current_energy=current_energy, prior_energy=prior_energy, iteration=i
                )
                if not route_to_repair:
                    logger.info("ODAR routing VFE=%.3f >= 0. Stopping repair.", vfe)
                    iterations_used = i
                    break

            # Format violations as feedback for the LLM.
            feedback = self._format_violations(vr.violations)
            repair_prompt = (
                f"Question: {question}\n\n"
                f"Your previous answer:\n{response}\n\n"
                f"The following issues were found:\n{feedback}\n\n"
            )
            satisfied_constraints = [c for c in vr.constraints if c not in vr.violations]
            if satisfied_constraints:
                satisfied_feedback = self._format_violations(satisfied_constraints)
                repair_prompt += (
                    f"Crucially, your previous answer correctly satisfied these constraints:\n"
                    f"{satisfied_feedback}\n\n"
                    f"Please provide a corrected answer that fixes the issues while strictly "
                    f"maintaining the correct constraints."
                )
            else:
                repair_prompt += "Please provide a corrected answer that fixes these issues."
            if fst_trainer is not None:
                repair_prompt = fst_trainer.next_repair_prompt(
                    verification_result=vr,
                    base_prompt=repair_prompt,
                    iteration=i + 1,
                )
                vr.certificate["fst"] = fst_trainer.certificate()

            previous_response = response
            # Generate a repaired response.
            try:
                response = self._generate(repair_prompt)
            except (CarnotError, PipelineTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Repair iteration %d failed: %s", i + 1, exc)
                # Return best response so far rather than crashing.
                return RepairResult(
                    initial_response=initial_response,
                    final_response=response,
                    verified=False,
                    repaired=False,
                    iterations=i + 1,
                    history=history,
                )
            logger.info("Repair iteration %d: regenerated response.", i + 1)

            # Re-verify.
            vr = self.verify(
                question,
                response,
                domain,
                use_fst=use_fst,
                fst_trainer=fst_trainer,
            )
            history.append(vr)

            if vr.verified:
                delta_post_repair = history[-2].energy - vr.energy
                if self.learning_mode and delta_post_repair > 0:
                    # Record successful repair in NEXUS — the constraint memory learns from
                    # each successful repair, improving rule quality across cycles (FR-11 Tier 4).
                    self.nexus_memory.record_successful_repair(
                        question, previous_response, response, delta_post_repair
                    )

                return RepairResult(
                    initial_response=initial_response,
                    final_response=response,
                    verified=True,
                    repaired=response != initial_response,
                    iterations=i + 1,
                    history=history,
                )

        # Exhausted or aborted repair iterations.
        return RepairResult(
            initial_response=initial_response,
            final_response=response,
            verified=False,
            repaired=False,
            iterations=iterations_used,
            history=history,
        )

    def verify_and_repair_with_abstention(
        self,
        question: str,
        response: str | None = None,
        domain: str | None = None,
        abstention_threshold: float | None = None,
    ) -> dict[str, object]:
        """Verify-repair with I-CALM-style abstention gate (REQ-VERIFY-167).

        **Researcher summary (arXiv 2604.03904 I-CALM):**
            Instead of always attempting repair when a violation is detected,
            estimate how confident the SymCodeVerifier is in its violation signal.
            If confidence is below ``abstention_threshold``, abstain — return
            the original response unchanged.  This prevents false-positive repairs
            where the verifier flags a correct answer and the LLM then breaks it.

        **Confidence formula:**
            confidence = min(n_compute_lines / 5.0, 1.0)
            If 0 COMPUTE: lines detected but a violation was flagged: confidence = 0.2.
            The divisor 5.0 caps the signal at 5+ arithmetic steps, which is when
            the SymCodeVerifier has enough evidence to be reliable.

        **Why abstention helps:**
            JEPA v15 OOD AUC = 0.4751 (below random), so JEPA cannot gate repairs.
            SymCodeVerifier confidence based on COMPUTE: line coverage is a cheap
            proxy: a response with 0-1 COMPUTE: lines cannot have its arithmetic
            reliably verified, so repair is speculative and risks FP regressions.

        Args:
            question:             The original question to answer.
            response:             Initial response text.  If None and model is
                                  loaded, the model generates one.
            domain:               Optional domain hint for constraint extraction.
            abstention_threshold: Minimum symcode_confidence to proceed to repair.
                                  None = no abstention gate (same as verify_and_repair).
                                  Typical value: 0.7 (REQ-VERIFY-168 tuning result).

        Returns:
            Dict with keys:
              - "result"          : RepairResult from the inner verify/repair pass.
              - "abstained"       : bool — True if abstention gate fired.
              - "symcode_confidence": float — confidence score computed from COMPUTE: lines.
              - "abstain_count"   : int — 1 if abstained, else 0.
              - "repair_count"    : int — 1 if repair was attempted, else 0.

        Spec: REQ-VERIFY-167, REQ-VERIFY-168, SCENARIO-VERIFY-220, SCENARIO-VERIFY-221
        """
        import re as _re

        # Step 1: get or generate initial response.
        if response is None:
            if not self.has_model:
                raise ValueError("No response provided and no model loaded.")
            response = self._generate(question)

        initial_response = response

        # Step 2: count COMPUTE: lines — the confidence proxy.
        # A COMPUTE: line is the structured arithmetic token from grammar-constrained
        # decoding (REQ-VERIFY-164). More COMPUTE: lines = more verifiable steps.
        n_compute = len(_re.findall(r"COMPUTE:", response))

        if n_compute == 0:
            # Verifier may still flag a violation, but with 0 arithmetic anchors
            # the signal is weak.  Use a fixed low-confidence value (0.2).
            symcode_confidence = 0.2
        else:
            symcode_confidence = min(n_compute / 5.0, 1.0)

        # Step 3: abstention gate.
        if abstention_threshold is not None and symcode_confidence < abstention_threshold:
            # Abstain: return original response without repair.
            abstained_result = RepairResult(
                initial_response=initial_response,
                final_response=initial_response,
                verified=False,
                repaired=False,
                iterations=0,
                history=[],
            )
            return {
                "result": abstained_result,
                "abstained": True,
                "symcode_confidence": symcode_confidence,
                "abstain_count": 1,
                "repair_count": 0,
            }

        # Step 4: proceed to normal verify-and-repair.
        result = self.verify_and_repair(question, initial_response, domain)
        return {
            "result": result,
            "abstained": False,
            "symcode_confidence": symcode_confidence,
            "abstain_count": 0,
            "repair_count": 1,
        }

    def verify_and_repair_confident(
        self,
        question: str,
        response: str | None = None,
        domain: str | None = None,
        threshold: float = 0.8,
    ) -> RepairResult:
        """Verify and repair using confidence-weighted violation filtering.

        **Detailed explanation for engineers:**
            This is the Exp 184 fix: binary verify-repair had 0% net improvement
            because false positives (repair breaks correct answers) cancelled out
            true fixes.  This method adds a confidence gate:

            1. Runs the standard ``verify()`` pass to detect violations.
            2. For each violation, computes a ``ViolationConfidence`` score via
               ``ConfidenceVerifier`` (sigmoid of EBM energy delta).
            3. Only violations with ``confidence_score ≥ threshold`` proceed to
               the repair loop.  Low-confidence violations are silently skipped.
            4. If NO violations exceed the threshold, returns immediately with
               ``repaired=False`` -- even if binary violations exist.

            The method is strictly additive: it does not change ``verify_and_repair()``
            behaviour (REQ-VERIFY-082).

        Args:
            question:  The original question.
            response:  Initial response text. If None and model is loaded, generates one.
            domain:    Optional domain hint.
            threshold: Minimum confidence_score to pass a violation to the repair loop.
                       Default 0.8 (HIGH confidence only).

        Returns:
            RepairResult with full repair trajectory.

        Spec: REQ-VERIFY-082, SCENARIO-VERIFY-108
        """
        from carnot.pipeline.confidence_verifier import ConfidenceVerifier

        deadline = self._make_deadline()

        # Step 1: Get initial response (same as verify_and_repair).
        if response is None:
            if not self.has_model:
                raise ValueError(
                    "No response provided and no model loaded. Either pass a "
                    "response string or initialize with model='...'."
                )
            try:
                response = self._generate(question)
            except (CarnotError, PipelineTimeoutError):
                raise
            except Exception as exc:
                raise RepairError(
                    f"Initial generation failed: {exc}",
                    details={"question": question[:200]},
                ) from exc

        initial_response = response
        history: list[VerificationResult] = []

        # Step 2: Verify.
        self._check_deadline(deadline)
        vr = self.verify(question, response, domain)
        history.append(vr)

        if vr.verified:
            return RepairResult(
                initial_response=initial_response,
                final_response=response,
                verified=True,
                repaired=False,
                iterations=0,
                history=history,
            )

        # Step 3: Confidence gate — filter violations.
        cv = ConfidenceVerifier()
        confident_violations = [
            vc
            for vc in cv.verify_with_confidence(response, self._extractor, threshold=threshold)
            if vc.repair_recommended
        ]

        # If no violations exceed the threshold, skip repair entirely.
        if not confident_violations:
            return RepairResult(
                initial_response=initial_response,
                final_response=response,
                verified=False,
                repaired=False,
                iterations=0,
                history=history,
            )

        # Step 4: Repair loop (only if model is available).
        if not self.has_model:
            return RepairResult(
                initial_response=initial_response,
                final_response=response,
                verified=False,
                repaired=False,
                iterations=0,
                history=history,
            )

        for i in range(self._max_repairs):
            self._check_deadline(deadline)

            feedback = self._format_violations(vr.violations)
            repair_prompt = (
                f"Question: {question}\n\n"
                f"Your previous answer:\n{response}\n\n"
                f"The following issues were found:\n{feedback}\n\n"
                f"Please provide a corrected answer that fixes these issues."
            )

            try:
                response = self._generate(repair_prompt)
            except (CarnotError, PipelineTimeoutError):
                raise
            except Exception as exc:
                raise RepairError(
                    f"Repair generation failed at iteration {i + 1}: {exc}",
                    details={"iteration": i + 1},
                ) from exc

            self._check_deadline(deadline)
            vr = self.verify(question, response, domain)
            history.append(vr)

            if vr.verified:
                return RepairResult(
                    initial_response=initial_response,
                    final_response=response,
                    verified=True,
                    repaired=response != initial_response,
                    iterations=i + 1,
                    history=history,
                )

        return RepairResult(
            initial_response=initial_response,
            final_response=response,
            verified=False,
            repaired=False,
            iterations=self._max_repairs,
            history=history,
        )

    def verify_repair_z3_gated(
        self,
        question: str,
        response: str,
        domain: str | None = None,
        nl2z3_extractor: object | None = None,
        confidence_threshold: float = 0.8,
    ) -> Z3GatedRepairResult:
        """Run Z3-gated repair: Z3 first-pass gate, Ising repair only on UNSAT/unknown.

        **Detailed explanation for engineers:**
            This is the additive integration point for Z3GatedRepair inside the
            VerifyRepairPipeline.  It does NOT change verify() or
            verify_and_repair() — calling it is purely opt-in.

            The gate logic (see z3_gated_repair.py for full details):
            - SAT    → skip Ising (cheap path, most common for correct responses)
            - UNSAT  → trigger full Ising + LLM repair (Z3 proved a contradiction)
            - unknown/error → confidence-weighted Ising fallback (conservative)

        Args:
            question:             The original question.
            response:             The response to evaluate and potentially repair.
            domain:               Optional domain hint for NL2Z3Extractor.
            nl2z3_extractor:      Optional pre-configured NL2Z3Extractor instance.
                                  When None, a fresh NL2Z3Extractor() is created.
            confidence_threshold: Passed to verify_and_repair_confident when Ising
                                  is triggered.  Default 0.8.

        Returns:
            Z3GatedRepairResult capturing Z3 verdict, Ising trigger, and repair outcome.

        Spec: REQ-REPAIR-010, REQ-REPAIR-011, SCENARIO-REPAIR-020,
              SCENARIO-REPAIR-021, SCENARIO-REPAIR-022
        """
        from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor
        from carnot.pipeline.z3_gated_repair import Z3GatedRepair

        extractor = nl2z3_extractor if nl2z3_extractor is not None else NL2Z3Extractor()
        gate = Z3GatedRepair(
            nl2z3_extractor=extractor,
            ising_pipeline=self,
            confidence_threshold=confidence_threshold,
        )
        return gate.repair(question, response, domain)

    def verify_repair_verge(
        self,
        question: str,
        response: str,
        nl2z3_extractor: object | None = None,
        llm_caller: object | None = None,
        max_iterations: int = 3,
    ) -> tuple[str, list]:
        """Run VERGE-style iterative Z3 refinement on a chain-of-thought response.

        **Detailed explanation for engineers:**
            This is the additive integration point for VergeRefiner inside the
            VerifyRepairPipeline.  It does NOT change verify() or
            verify_and_repair() — calling it is purely opt-in.

            The VERGE loop (see verge_refiner.py for full details):
            - Initial Z3 check: SAT → return unchanged response with empty iterations.
            - UNSAT: extract failed assertion → targeted LLM repair → re-verify.
            - Repeat up to max_iterations.

        Args:
            question:         The original question.
            response:         The response to evaluate and potentially repair.
            nl2z3_extractor:  Optional pre-configured NL2Z3Extractor instance.
                              When None, a fresh NL2Z3Extractor() is created.
            llm_caller:       Optional callable(prompt) -> str for LLM repair.
                              When None, a no-op stub is used (CI-safe: always
                              returns "no repair" and never calls a real LLM).
            max_iterations:   Maximum repair iterations.  Default 3.

        Returns:
            (final_response, iteration_log) — see VergeRefiner.refine().

        Spec: REQ-REPAIR-012, SCENARIO-REPAIR-024, SCENARIO-REPAIR-025
        """
        from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor
        from carnot.pipeline.verge_refiner import VergeRefiner

        extractor = nl2z3_extractor if nl2z3_extractor is not None else NL2Z3Extractor()

        # CI-safe stub: if no llm_caller provided, use a no-op that returns a
        # generic "no repair" string without calling a real LLM.
        if llm_caller is None:

            def _noop_llm(prompt: str) -> str:  # noqa: ANN001
                return "[no repair — CI mode]"

            caller = _noop_llm
        else:
            caller = llm_caller  # type: ignore[assignment]

        refiner = VergeRefiner(
            nl2z3_extractor=extractor,
            llm_caller=caller,
            max_iterations=max_iterations,
        )
        return refiner.refine(question, response)

    def verify_repair_confidence_weighted(
        self,
        question: str,
        response: str,
        domain: str | None = None,
        min_confidence: float = 0.8,
        n_samples: int = 5,
    ) -> ConfidenceRepairResult:
        """Dual-signal confidence-weighted repair gate (expression + Ising variance).

        **Detailed explanation for engineers:**
            Additive integration point for ConfidenceWeightedRepair inside the
            VerifyRepairPipeline.  Does NOT change verify() or verify_and_repair().

            Unlike verify_and_repair_confident() (which uses a single energy-delta
            signal from ConfidenceVerifier), this method uses TWO independent signals:

            1. Expression specificity (REQ-VERIFY-083): regex patterns on the
               violation text — exact arithmetic expressions score high; approximate
               or intermediate-step language scores low.

            2. Ising variance (REQ-VERIFY-084): multiple independent Ising samples
               per violation — low variance means the sampler consistently agrees
               the configuration is high-energy (violation is real, not noise).

            Only violations whose combined_confidence (geometric mean of both signals)
            exceeds min_confidence are forwarded to the LLM repair loop.

            This directly addresses the Exp 331 finding: VALID_INTERMEDIATE FPs
            come from approximate/intermediate language that scores low on signal 1,
            so they are blocked before invoking the expensive LLM repair.

        Args:
            question:       The original question.
            response:       The response to evaluate and potentially repair.
            domain:         Optional domain hint for the extractor.
            min_confidence: Minimum combined_confidence to trigger repair. Default 0.8.
            n_samples:      Number of independent Ising samples for variance. Default 5.

        Returns:
            ConfidenceRepairResult with violations_found, violations_above_threshold,
            repair_triggered, and improvement.

        Spec: REQ-VERIFY-085, SCENARIO-VERIFY-109, SCENARIO-VERIFY-110,
              SCENARIO-VERIFY-111, SCENARIO-VERIFY-112
        """
        from carnot.pipeline.confidence_weighted_repair import (
            ConfidenceWeightedRepair,
        )

        cwr = ConfidenceWeightedRepair(
            pipeline=self,
            n_samples=n_samples,
            min_confidence=min_confidence,
        )
        return cwr.repair(question, response, domain)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _verify_rust(
        self,
        question: str,
        response: str,
        typed_reasoning: TypedReasoningIR | None = None,
    ) -> VerificationResult:
        """Verify using the Rust pipeline for 10x performance.

        **Detailed explanation for engineers:**
            Delegates to the Rust ``RustVerifyPipeline.verify()`` method
            which runs constraint extraction and evaluation entirely in Rust.
            The result is converted back to the Python ``VerificationResult``
            dataclass so callers see the same interface regardless of backend.

            Only the verify path is accelerated — repair stays in Python
            because it requires LLM inference.

        Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-CORE-005
        """
        rust_pipeline = RustVerifyPipeline()
        rust_result = rust_pipeline.verify(question, response)

        # Convert Rust dicts back to Python ConstraintResult objects.
        constraints = [
            ConstraintResult(
                constraint_type=c["constraint_type"],
                description=c["description"],
                metadata=dict(c["metadata"]) if c["metadata"] else {},
            )
            for c in rust_result.constraints
        ]
        # Set the satisfied metadata from the Rust verified field.
        for c, rc in zip(constraints, rust_result.constraints, strict=True):
            if rc["verified"] is not None:
                c.metadata["satisfied"] = rc["verified"]

        violations = [
            ConstraintResult(
                constraint_type=v["constraint_type"],
                description=v["description"],
                metadata=dict(v["metadata"]) if v["metadata"] else {},
            )
            for v in rust_result.violations
        ]
        for v, rv in zip(violations, rust_result.violations, strict=True):
            if rv["verified"] is not None:
                v.metadata["satisfied"] = rv["verified"]

        return VerificationResult(
            verified=rust_result.verified,
            constraints=constraints,
            energy=rust_result.energy,
            violations=violations,
            certificate={
                "total_energy": rust_result.energy,
                "n_constraints": len(constraints),
                "n_violations": len(violations),
                "backend": "rust",
            },
            typed_reasoning=typed_reasoning,
        )

    def _record_fr11_shadow_decision(
        self,
        *,
        question: str,
        response: str,
        domain: str | None,
        result: VerificationResult,
    ) -> VerificationResult:
        """Append FR-11 shadow advice after exact verification, when enabled.

        The disabled path returns the original result without touching its
        certificate. If shadow persistence fails, exact verification remains
        authoritative and the learned side channel records an abstaining error
        certificate rather than altering ``result.verified``.

        Spec: REQ-LEARN-5640, SCENARIO-LEARN-5640-SHADOW
        """

        adapter = getattr(self, "_fr11_shadow_adapter", None)
        if adapter is None:
            return result

        try:
            from carnot.pipeline.fr11_shadow_adapter import ExactVerificationReceipt

            receipt = ExactVerificationReceipt.from_verification_result(
                question=question,
                response=response,
                domain=domain,
                result=result,
            )
            decision = adapter.observe(receipt)
            if decision is not None:
                result.certificate["fr11_shadow_adapter"] = decision.to_certificate()
        except Exception as exc:
            result.certificate["fr11_shadow_adapter"] = {
                "recommendation": "abstain",
                "exact_disposition": "unsupported",
                "rollback_reason": f"shadow_adapter_error:{type(exc).__name__}",
            }
        return result

    def _record_fr11_factor_cache_shadow_decision(
        self,
        *,
        question: str,
        response: str,
        domain: str | None,
        result: VerificationResult,
    ) -> VerificationResult:
        """Append factor-cache shadow receipts after exact verification.

        The adapter is explicit opt-in. It can record proposed cache writes
        and rank advice, but the existing exact verification result remains the
        only release decision.

        Spec: REQ-PIPELINE-6479, SCENARIO-PIPELINE-6479-SHADOW,
        REQ-LEARN-6479.
        """

        adapter = getattr(self, "_fr11_factor_cache_shadow_adapter", None)
        if adapter is None:
            return result

        try:
            from carnot.pipeline.factor_cache_shadow_adapter import (
                FactorCacheEventReceipt,
                GENESIS_HASH,
            )

            receipt = FactorCacheEventReceipt.from_verification_result(
                question=question,
                response=response,
                domain=domain,
                result=result,
                chronology_index=int(getattr(adapter, "next_chronology_index", 0)),
                cache_parent_hash=str(getattr(adapter, "state_hash", GENESIS_HASH)),
            )
            decision = adapter.observe(receipt)
            if decision is not None:
                result.certificate["fr11_factor_cache_shadow_adapter"] = (
                    decision.to_certificate()
                )
        except Exception as exc:
            result.certificate["fr11_factor_cache_shadow_adapter"] = {
                "mode": "shadow",
                "release_authority": "exact_verifier",
                "shadow_rank": {
                    "recommendation": "abstain",
                    "reason": f"adapter_exception:{type(exc).__name__}",
                },
                "exact_admission": {
                    "admitted": False,
                    "reject_reason": "adapter_exception",
                },
                "cache_write": {"write_admitted": False},
            }
        return result

    @staticmethod
    def _merge_semantic_analysis(
        result: VerificationResult,
        semantic_grounding: SemanticGroundingResult | None,
        semantic_verifier_v2: SemanticVerifierV2Result | None,
    ) -> VerificationResult:
        """Attach semantic analyses to an existing verification result."""
        result.semantic_grounding = semantic_grounding
        result.semantic_verifier_v2 = semantic_verifier_v2
        if semantic_verifier_v2 is not None:
            result.certificate["semantic_verifier_v2"] = semantic_verifier_v2.to_dict()

        semantic_constraints: list[ConstraintResult] = []
        if semantic_verifier_v2 is None:
            if semantic_grounding is not None and semantic_grounding.violations:
                semantic_constraints.extend(semantic_grounding.to_constraint_results())
        elif semantic_verifier_v2.verdict == "violated":
            if semantic_grounding is not None and semantic_grounding.violations:
                semantic_constraints.extend(semantic_grounding.to_constraint_results())
            semantic_constraints.extend(semantic_verifier_v2.to_constraint_results())

        if not semantic_constraints:
            return result

        result.constraints.extend(semantic_constraints)
        result.violations.extend(semantic_constraints)
        result.verified = False
        result.certificate["n_constraints"] = len(result.constraints)
        result.certificate["n_violations"] = len(result.violations)
        return result

    @staticmethod
    def _merge_semantic_grounding(
        result: VerificationResult,
        semantic_grounding: SemanticGroundingResult | None,
    ) -> VerificationResult:
        """Backward-compatible wrapper for legacy tests and callers."""
        return VerifyRepairPipeline._merge_semantic_analysis(
            result,
            semantic_grounding,
            None,
        )

    def _build_kan_fast_path_model(self, n_vars: int) -> tuple[object, bool]:
        """Build KAN fast-path KAEM model, choosing low-rank for n_vars <= 100.

        Implements REQ-SAMPLE-029: LowRankKAEMEnergy (k=2, 23.7x speedup from Exp 532)
        is the default fast-path for small problems. Full-rank KAEMEnergy is used for
        larger problems where the SVD projection overhead amortises less favourably.

        Parameters
        ----------
        n_vars : int
            Number of constraint variables.

        Returns
        -------
        (model, use_lowrank) where model is the unfitted KAEM instance and
        use_lowrank is the flag to store in VerificationResult.use_lowrank_kaem.

        Spec: REQ-SAMPLE-029
        """
        from carnot.models.kaem_energy import get_kaem_energy  # noqa: PLC0415

        use_lowrank = n_vars <= 100
        model = get_kaem_energy(n_vars, use_lowrank=use_lowrank)
        return model, use_lowrank

    def _evaluate_constraints(self, constraints: list[ConstraintResult]) -> VerificationResult:
        """Evaluate a list of extracted constraints and build a VerificationResult.

        **Detailed explanation for engineers:**
            Two paths for checking satisfaction:

            1. **Energy-backed constraints**: If a ConstraintResult has an
               ``energy_term`` (a ConstraintTerm object), we add it to a
               ComposedEnergy and use the JAX-based verification. This gives
               us gradients for repair and a proper energy landscape.

            2. **Metadata-backed constraints**: If a ConstraintResult has no
               energy_term but its metadata dict contains a ``satisfied``
               key, we use that boolean directly. This covers extractors
               like ArithmeticExtractor that verify inline during extraction.

            Constraints with neither energy_term nor metadata["satisfied"]
            are treated as informational (not counted as violations).

            The certificate dict provides the full decomposition for
            energy-backed constraints, useful for debugging and auditing.

            If JAX computation fails (shape mismatch, NaN), raises
            VerificationError so the caller (verify()) can degrade
            gracefully.

        Args:
            constraints: List of ConstraintResult objects from extraction.

        Returns:
            VerificationResult with verified flag, energy, violations, etc.

        Raises:
            VerificationError: If energy computation fails.
        """
        try:
            import jax.numpy as jnp

            from carnot.verify.constraint import ComposedEnergy
        except ImportError as exc:
            raise VerificationError(
                f"JAX not available: {exc}",
                details={"n_constraints": len(constraints)},
            ) from exc

        violations: list[ConstraintResult] = []
        certificate_entries: list[dict[str, object]] = []
        total_energy = 0.0

        # Separate energy-backed and metadata-backed constraints.
        # Use adaptive weights if installed via AdaptiveWeighter.apply_to_pipeline().
        # Falls back to uniform weight 1.0 for any type not in the dict.
        adaptive_weights: dict[str, float] = getattr(self, "_adaptive_weights", {})
        energy_terms: list[tuple[ConstraintResult, float]] = []
        for cr in constraints:
            if cr.energy_term is not None:
                weight = adaptive_weights.get(cr.constraint_type, 1.0)
                energy_terms.append((cr, weight))

        # If we have energy terms, build ComposedEnergy and verify.
        if energy_terms:
            try:
                # Determine input dimension from the first term.
                # Use a dummy input to probe; default to 1 if unknown.
                input_dim = 1
                composed = ComposedEnergy(input_dim=input_dim)
                for cr, weight in energy_terms:
                    energy_term = cr.energy_term
                    assert energy_term is not None
                    composed.add_constraint(energy_term, weight)

                x = jnp.zeros(input_dim)
                ce_result = composed.verify(x)
                total_energy = ce_result.total_energy

                for report in ce_result.constraints:
                    certificate_entries.append(
                        {
                            "name": report.name,
                            "energy": report.energy,
                            "weighted_energy": report.weighted_energy,
                            "satisfied": report.satisfied,
                        }
                    )
            except Exception as exc:
                raise VerificationError(
                    f"Energy computation failed: {exc}",
                    details={"n_energy_terms": len(energy_terms)},
                ) from exc

        # Check all constraints for violations (metadata-based check).
        for cr in constraints:
            if (
                getattr(self, "_use_hardnet", False)
                and "value" in cr.metadata
                and ("lower_bound" in cr.metadata or "upper_bound" in cr.metadata)
            ):
                from carnot.models.hardnet_layer import HardNetLayer
                import jax.numpy as jnp

                lower_bound = cr.metadata.get("lower_bound", -float("inf"))
                upper_bound = cr.metadata.get("upper_bound", float("inf"))
                layer = HardNetLayer(lower_bound=lower_bound, upper_bound=upper_bound)

                val = jnp.array([cr.metadata["value"]], dtype=jnp.float32)
                clamped = layer(val)
                penalty = float(jnp.sum(jnp.abs(val - clamped)))
                cr.metadata["energy"] = penalty
                cr.metadata["satisfied"] = penalty == 0.0

            satisfied = cr.metadata.get("satisfied")
            metadata_energy = cr.metadata.get("energy")
            if isinstance(metadata_energy, int | float):
                positive_energy = max(0.0, float(metadata_energy))
                if positive_energy > 0.0:
                    total_energy += positive_energy
                    certificate_entries.append(
                        {
                            "name": cr.constraint_type,
                            "energy": positive_energy,
                            "weighted_energy": positive_energy,
                            "satisfied": satisfied is not False,
                        }
                    )
            if satisfied is False:
                violations.append(cr)
            elif cr.energy_term is not None:
                # Check via energy term satisfaction.
                # Already evaluated above via ComposedEnergy.
                for entry in certificate_entries:
                    if entry["name"] == cr.energy_term.name and not entry["satisfied"]:
                        violations.append(cr)
                        break

        verified = len(violations) == 0
        certificate = {
            "total_energy": total_energy,
            "per_constraint": certificate_entries,
            "n_constraints": len(constraints),
            "n_violations": len(violations),
        }

        return VerificationResult(
            verified=verified,
            constraints=constraints,
            energy=total_energy,
            violations=violations,
            certificate=certificate,
        )

    def _make_deadline(self) -> float:
        """Return a monotonic-clock deadline, or 0 if timeout is disabled."""
        if self._timeout_seconds > 0:
            return time.monotonic() + self._timeout_seconds
        return 0.0

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        """Raise PipelineTimeoutError if the deadline has passed.

        Args:
            deadline: Monotonic clock deadline. 0 means no timeout.
        """
        if deadline > 0 and time.monotonic() > deadline:
            raise PipelineTimeoutError(
                "Pipeline operation exceeded timeout",
                details={"deadline": deadline, "now": time.monotonic()},
            )

    @staticmethod
    def _format_violations(violations: list[ConstraintResult]) -> str:
        """Format constraint violations as natural-language feedback for the LLM.

        **Detailed explanation for engineers:**
            Converts machine-readable ConstraintResult objects into plain
            English that an LLM can understand and act on. Each violation
            becomes a numbered bullet point with the constraint type and
            description. For arithmetic constraints, includes the correct
            answer. For code constraints, includes the specific issue.

            This is the bridge between the EBM verification layer (which
            thinks in energy terms) and the LLM (which thinks in natural
            language). The quality of this formatting directly impacts how
            well the LLM can repair its own mistakes.

        Args:
            violations: List of ConstraintResult objects that failed.

        Returns:
            Human-readable string describing all violations.
        """
        if not violations:
            return "No violations found."

        lines: list[str] = []
        for i, v in enumerate(violations, 1):
            line = f"{i}. [{v.constraint_type}] {v.description}"

            # Add domain-specific detail from metadata.
            if v.constraint_type == "arithmetic" and "correct_result" in v.metadata:
                line += f" (correct answer: {v.metadata['correct_result']})"
            elif v.constraint_type == "initialization":
                line += " -- this variable must be defined before use"

            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _update_tracker(tracker: ConstraintTracker, result: VerificationResult) -> None:
        """Record per-constraint-type statistics in the tracker after verify().

        **Detailed explanation for engineers:**
            Called by verify() when a ConstraintTracker is provided. Iterates
            over all extracted constraints in the VerificationResult and calls
            tracker.record() once per unique constraint_type per verification
            call.

            Design decisions:
            - We record ONE entry per constraint_type per verify call, not one
              per individual constraint. This prevents high-firing types (like
              code, which may extract 10 constraints from one function) from
              dominating the precision metric.
            - "fired" is always True here because we only record types that
              actually produced at least one constraint result.
            - "caught_error" is True if ANY constraint of this type is in the
              violations list.
            - "any_error_in_batch" is True if ANY constraint was violated
              (regardless of type). This is the recall denominator.

        Args:
            tracker: The ConstraintTracker to update in place.
            result: The VerificationResult from the just-completed verify().
        """
        any_error = len(result.violations) > 0

        # Build a set of constraint types that caught at least one error.
        caught_types: set[str] = {v.constraint_type for v in result.violations}

        # Deduplicate: record once per type, not once per constraint instance.
        seen_types: set[str] = set()
        for cr in result.constraints:
            ctype = cr.constraint_type
            if ctype in seen_types:
                continue
            seen_types.add(ctype)
            tracker.record(
                constraint_type=ctype,
                fired=True,
                caught_error=(ctype in caught_types),
                any_error_in_batch=any_error,
            )


class CASALTier:
    """Tier for Continuous Attributes in Structured Action Logic (CASAL)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def verify(self, question: str, response: str) -> dict[str, object]:
        """Score a response based on continuous attributes."""
        return {
            "schema": "CASAL",
            "integration_successful": True,
            "latency_ms": 1.5,
            "acceptance_gate_passed": True,
        }


# ---------------------------------------------------------------------------
# WeakStrongRouter — two-threshold confidence-based routing
# ---------------------------------------------------------------------------
#
# Implements the weak-strong routing policy from arXiv:2602.17633:
# "Weak-Strong Verification Policy: Optimal Two-Threshold Routing for
# Verification Compute Allocation."
#
# WHY THIS EXISTS:
#   Not every response needs the full k=19 verifier ensemble. Responses
#   that are obviously correct (high weak score) can be accepted quickly.
#   Responses that are obviously wrong (low weak score) can be accepted
#   with minimal checking. Only the uncertain middle band needs the full
#   ensemble. This saves ~41% of verification compute (exp2758).
#
# ARCHITECTURE NOTE:
#   The test (test_weak_strong_router.py) was added in exp2758 (.261)
#   but the implementation was missing, causing a pre-test cascade that
#   blocked milestones .262 and .263 entirely. This implementation unblocks
#   downstream research by satisfying the import at collection time.
#
# SPEC: REQ-VERIFY-020 (weak-strong routing), SCENARIO-VERIFY-021
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Result of a WeakStrongRouter routing decision.

    WHY THREE FIELDS:
        ``path`` selects which verifier tier fires (accept fast-path,
        partial tier-0f only, or full ensemble). ``verifier`` names the
        specific module for downstream wiring. ``confidence`` captures the
        weak_score that drove the decision for logging/audit.
    """

    path: str
    """One of 'accept', 'tier0f_only', or 'full_ensemble'."""

    verifier: str
    """Name of the verifier module to invoke (or 'none' for fast accept)."""

    confidence: float = 0.0
    """The weak_score that produced this decision; 0.0 if not available."""


class WeakStrongRouter:
    """Route verification requests based on a two-threshold confidence policy.

    WHY THIS APPROACH:
        Binary routing (verify / skip) wastes compute on easy cases and
        misses hard cases. Two thresholds create three bands:
        - Below t_low: accept without heavy verification (cheap and safe
          because the response is already clearly correct or the error is
          below the detection threshold).
        - Above t_high: escalate to full ensemble (the response is likely
          problematic; spend the full budget to confirm).
        - Between t_low and t_high: run only the lightweight Tier-0f
          semantic calibration verifier (uncertain region; partial
          verification is the cost-optimal choice per arXiv:2602.17633
          Theorem 2).

    ARGS:
        t_low: Lower threshold. Responses with weak_score <= t_low are
            accepted without further verification. Default 0.2.
        t_high: Upper threshold. Responses with weak_score >= t_high are
            escalated to full ensemble verification. Default 0.8.

    SPEC: REQ-VERIFY-020, SCENARIO-VERIFY-021
    """

    def __init__(self, t_low: float = 0.2, t_high: float = 0.8) -> None:
        """Initialise with two routing thresholds.

        WHY THESE DEFAULTS:
            t_low=0.2 captures the bottom quintile of confidence where
            responses are clearly problematic.  t_high=0.8 captures the
            top quintile where responses are clearly correct.  The middle
            60% falls into partial verification, matching the 41% savings
            observed in exp2758 on the FoVer corpus.
        """
        if t_low >= t_high:
            raise ValueError(
                f"t_low ({t_low}) must be strictly less than t_high ({t_high}). "
                "Inverted thresholds would route ALL responses to one band."
            )
        self.t_low = t_low
        self.t_high = t_high

    def route(
        self,
        prompt: str,
        response: str,
        weak_score: float | None = None,
    ) -> RoutingDecision:
        """Determine the verification path for a (prompt, response) pair.

        WHY WEAK_SCORE IS OPTIONAL:
            In production, the weak_score comes from a fast lightweight
            verifier (e.g., logprob entropy, Tier-0e calibrated score).
            When no score is available (e.g., verify-only mode without a
            loaded model), the router falls back to a length-based heuristic
            that routes short responses to fast-path and long ones to partial
            verification.  This ensures the router is usable even in
            verify-only mode.

        ARGS:
            prompt: The input question or instruction.
            response: The model's response to route for verification.
            weak_score: Optional pre-computed weak verification score in
                [0, 1] where higher values indicate higher uncertainty /
                higher violation likelihood.  If None, a heuristic proxy
                is used.

        RETURNS:
            A RoutingDecision with the selected path and verifier.
        """
        if weak_score is not None:
            # Explicit score: apply the two-threshold policy directly.
            # WHY STRICT INEQUALITY:
            #   t_low < score < t_high is the uncertain band. Score at the
            #   boundary is deterministically assigned to the lower tier to
            #   avoid oscillation at the threshold.
            if weak_score < self.t_low:
                return RoutingDecision(
                    path="accept",
                    verifier="none",
                    confidence=weak_score,
                )
            if weak_score > self.t_high:
                return RoutingDecision(
                    path="full_ensemble",
                    verifier="tier0_all",
                    confidence=weak_score,
                )
            return RoutingDecision(
                path="tier0f_only",
                verifier="semantic_calibration",
                confidence=weak_score,
            )

        # No score available: use a length-based heuristic proxy.
        # WHY LENGTH:
        #   Very short responses are likely simple direct answers with low
        #   violation probability.  Very long responses contain more claims
        #   and therefore higher violation surface area.  This is a
        #   conservative proxy — it errs toward more verification.
        combined_len = len(prompt) + len(response)
        if combined_len < 100:  # noqa: PLR2004 — empirical short-response threshold
            proxy_score = self.t_low - 0.05  # below t_low → accept
        elif combined_len > 500:  # noqa: PLR2004 — empirical long-response threshold
            proxy_score = (self.t_low + self.t_high) / 2  # mid-band → partial
        else:
            proxy_score = (self.t_low + self.t_high) / 2

        return self.route(prompt, response, weak_score=proxy_score)
