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
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.errors import (
    CarnotError,
    ExtractionError,
    ModelLoadError,
    PipelineTimeoutError,
    RepairError,
    VerificationError,
)
from carnot.pipeline.extract import AutoExtractor, ConstraintExtractor, ConstraintResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rust backend auto-detection
# ---------------------------------------------------------------------------

# Check for CARNOT_USE_RUST env var (1 = force Rust, 0 = force Python).
# If unset, auto-detect: use Rust when available.
import os as _os

_FORCE_RUST = _os.environ.get("CARNOT_USE_RUST")

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
            "CARNOT_USE_RUST=1 but Rust bindings not installed. "
            "Falling back to Python pipeline."
        )
elif _FORCE_RUST == "0":
    _USE_RUST_PIPELINE = False
else:
    _USE_RUST_PIPELINE = RUST_AVAILABLE

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
    certificate: dict = field(default_factory=dict)


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

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004
    """

    def __init__(
        self,
        model: str | None = None,
        domains: list[str] | None = None,
        max_repairs: int = 3,
        extractor: ConstraintExtractor | None = None,
        timeout_seconds: float = 30.0,
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

        Args:
            model: HuggingFace model name/path, or None for verify-only.
            domains: Optional domain filter for constraint extraction.
            max_repairs: Max repair iterations (default 3).
            extractor: Custom extractor, or None for AutoExtractor.
            timeout_seconds: Max wall-clock seconds per verify/repair call.
                Default 30. Set to 0 or None to disable.

        Raises:
            ModelLoadError: If model is specified but cannot be loaded.

        Spec: REQ-VERIFY-001
        """
        self._domains = domains
        self._max_repairs = max_repairs
        self._timeout_seconds = timeout_seconds or 0.0

        # Set up the constraint extractor.
        if extractor is not None:
            self._extractor = extractor
        else:
            self._extractor = AutoExtractor()

        # Set up the optional LLM model.
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

        if model is not None:
            self._load_model(model)

    @property
    def has_model(self) -> bool:
        """True if an LLM model is loaded and available for generation."""
        return self._model is not None

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
                model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
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
            raise RuntimeError(
                "No model loaded. Initialize with model='...' to enable generation."
            )

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

        return response

    def extract_constraints(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
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

    def verify(
        self, question: str, response: str, domain: str | None = None
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

            Respects the configured timeout. If extraction or evaluation
            fails, returns a VerificationResult with verified=False and
            the error recorded in the certificate rather than crashing.

        Args:
            question: The original question (for context/logging).
            response: The response text to verify.
            domain: Optional domain hint for constraint extraction.

        Returns:
            VerificationResult with constraint evaluation details.

        Raises:
            PipelineTimeoutError: If the call exceeds timeout_seconds.

        Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
        """
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
                    return self._verify_rust(question, response)
                except Exception as exc:
                    logger.warning(
                        "Rust verify failed, falling back to Python: %s", exc
                    )
                    # Fall through to Python path.

        deadline = self._make_deadline()
        try:
            self._check_deadline(deadline)
            constraints = self.extract_constraints(response, domain)
            self._check_deadline(deadline)
            return self._evaluate_constraints(constraints)
        except PipelineTimeoutError:
            raise
        except CarnotError as exc:
            logger.warning("Verification degraded: %s", exc)
            return VerificationResult(
                verified=False,
                constraints=[],
                energy=0.0,
                violations=[],
                certificate={"error": str(exc), "error_type": type(exc).__name__},
            )

    def verify_and_repair(
        self,
        question: str,
        response: str | None = None,
        domain: str | None = None,
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

        Returns:
            RepairResult with full repair trajectory.

        Raises:
            ValueError: If response is None and no model is loaded.

        Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004
        """
        deadline = self._make_deadline()

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

        for i in range(self._max_repairs):
            self._check_deadline(deadline)

            # Format violations as feedback for the LLM.
            feedback = self._format_violations(vr.violations)
            repair_prompt = (
                f"Question: {question}\n\n"
                f"Your previous answer:\n{response}\n\n"
                f"The following issues were found:\n{feedback}\n\n"
                f"Please provide a corrected answer that fixes these issues."
            )

            # Generate a repaired response.
            try:
                response = self._generate(repair_prompt)
            except (CarnotError, PipelineTimeoutError):
                raise
            except Exception as exc:
                logger.warning(
                    "Repair iteration %d failed: %s", i + 1, exc
                )
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

        # Exhausted repair iterations.
        return RepairResult(
            initial_response=initial_response,
            final_response=response,
            verified=False,
            repaired=False,
            iterations=self._max_repairs,
            history=history,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _verify_rust(
        self, question: str, response: str
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
        for c, rc in zip(constraints, rust_result.constraints):
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
        for v, rv in zip(violations, rust_result.violations):
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
        )

    def _evaluate_constraints(
        self, constraints: list[ConstraintResult]
    ) -> VerificationResult:
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
        certificate_entries: list[dict] = []
        total_energy = 0.0

        # Separate energy-backed and metadata-backed constraints.
        energy_terms: list[tuple[ConstraintResult, float]] = []
        for cr in constraints:
            if cr.energy_term is not None:
                energy_terms.append((cr, 1.0))

        # If we have energy terms, build ComposedEnergy and verify.
        if energy_terms:
            try:
                # Determine input dimension from the first term.
                # Use a dummy input to probe; default to 1 if unknown.
                input_dim = 1
                composed = ComposedEnergy(input_dim=input_dim)
                for cr, weight in energy_terms:
                    composed.add_constraint(cr.energy_term, weight)

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
            satisfied = cr.metadata.get("satisfied")
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
