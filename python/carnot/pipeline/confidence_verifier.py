"""Confidence-weighted constraint verification.

**Researcher summary:**
    Exp 184 showed binary verify-repair has 0% net improvement on 3B models:
    false positives (repair breaks correct answers) cancel out true fixes.
    The root cause is treating all violations as equally urgent. This module
    converts binary violated/not-violated flags into continuous confidence
    scores so the repair gate can ignore low-confidence (likely false-positive)
    violations.

    Theoretical basis: arXiv 2602.03979 (Likelihood-Based Reward Designs)
    validates using log-probability (EBM energy score) as a CONTINUOUS signal
    for repair decisions — more informative than binary correct/incorrect, and
    correlates with actual error severity.

**Detailed explanation for engineers:**
    The pipeline has two new building blocks:

    1. ``confidence_from_energy(energy_score, temperature)``: sigmoid
       normalisation of a raw energy delta.  Large positive energy → high
       confidence the violation is real.  The temperature parameter controls
       sensitivity (lower = steeper sigmoid).  Numerically stable: clamps
       ±inf and NaN to [0, 1].

    2. ``ConfidenceVerifier.verify_with_confidence(response, extractor)``:
       runs the given extractor, finds violated constraints, computes a
       ``ViolationConfidence`` per violation, and returns the list.  Only
       violated constraints appear; satisfied ones are silently skipped.

    The ``ViolationConfidence`` dataclass carries all the information the
    repair gate needs:

    - ``confidence_score``: float in [0, 1] (sigmoid of energy delta)
    - ``confidence_class``: "HIGH" / "MEDIUM" / "LOW" bucket
    - ``repair_recommended``: True when score ≥ threshold (default 0.8)
    - ``evidence``: raw metadata from the source ConstraintResult so callers
      can log which claimed vs. correct values drove the decision

    ``repair_gate(confidence, threshold)`` is a thin helper that returns
    a bool — separating the gate logic from the dataclass for easy unit testing.

Spec: REQ-VERIFY-081, REQ-VERIFY-082,
      SCENARIO-VERIFY-105, SCENARIO-VERIFY-106, SCENARIO-VERIFY-107,
      SCENARIO-VERIFY-108
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carnot.pipeline.extract import ConstraintExtractor


# ---------------------------------------------------------------------------
# Thresholds for confidence class assignment (REQ-VERIFY-081)
# ---------------------------------------------------------------------------

_HIGH_THRESHOLD: float = 0.8
_MEDIUM_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def confidence_from_energy(energy_score: float, temperature: float = 1.0) -> float:
    """Convert an EBM energy delta to a [0, 1] confidence score.

    **Detailed explanation for engineers:**
        Uses sigmoid normalisation: ``1 / (1 + exp(-energy_score / temperature))``.
        At ``energy_score = 0`` the output is exactly 0.5 (maximum uncertainty).
        Large positive values approach 1.0 (confident violation).
        Large negative values approach 0.0 (not a violation).

        Handles edge cases safely:
        - ``+inf`` → 1.0 (clamp)
        - ``-inf`` → 0.0 (clamp)
        - ``NaN``  → 0.0 (treat unknown as non-violation)

    Args:
        energy_score: Raw energy delta from EBM constraint evaluation.
        temperature:  Scaling factor controlling sigmoid steepness.
                      Lower values produce steeper (more decisive) curves.

    Returns:
        Confidence score in [0, 1].

    Spec: REQ-VERIFY-081
    """
    if math.isnan(energy_score):
        return 0.0
    # Guard against division by zero (temperature should always be > 0)
    if temperature == 0.0:
        return 1.0 if energy_score > 0 else 0.0
    x = energy_score / temperature
    # Sigmoid, clamped to avoid overflow
    if x >= 0:
        exp_neg = math.exp(-x)
        score = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(x)
        score = exp_pos / (1.0 + exp_pos)
    # Final clamp (handles edge cases from very large floats)
    return max(0.0, min(1.0, score))


def repair_gate(confidence: float, threshold: float = 0.8) -> bool:
    """Return True only when confidence meets or exceeds the repair threshold.

    **Detailed explanation for engineers:**
        This is the core of the Exp 184 fix.  Previously every violation
        triggered a repair attempt.  Now the caller passes a ``threshold``
        (default 0.8) and only high-confidence violations proceed to the
        expensive LLM repair loop.

    Args:
        confidence: Confidence score in [0, 1].
        threshold:  Minimum confidence required to recommend repair.

    Returns:
        True if repair is recommended, False otherwise.

    Spec: REQ-VERIFY-082, SCENARIO-VERIFY-107
    """
    return confidence >= threshold


# ---------------------------------------------------------------------------
# ViolationConfidence dataclass
# ---------------------------------------------------------------------------


@dataclass
class ViolationConfidence:
    """Per-violation confidence record produced by ConfidenceVerifier.

    **Detailed explanation for engineers:**
        Replaces the binary "violated/not violated" flag with a continuous
        confidence score derived from the EBM energy delta.  Downstream
        repair logic uses ``repair_recommended`` to decide whether to invoke
        the expensive LLM repair loop for this specific violation.

    Class constants:
        HIGH:   Confidence score ≥ 0.8 — error is almost certainly real.
        MEDIUM: Score in [0.5, 0.8) — uncertain, worth logging but not repairing.
        LOW:    Score < 0.5 — likely noise or false positive, skip repair.

    Attributes:
        constraint_id:    Identifier from the source ConstraintResult description.
        energy_delta:     Raw energy contribution (positive = violation present).
        confidence_score: Sigmoid-normalised score in [0, 1].
        confidence_class: String bucket — one of HIGH / MEDIUM / LOW.
        repair_recommended: True when score ≥ repair threshold.
        evidence:         Metadata from source constraint (e.g. claimed vs. correct).

    Spec: REQ-VERIFY-081
    """

    # Class-level constants for confidence buckets
    HIGH: str = field(default="HIGH", init=False, repr=False, compare=False)
    MEDIUM: str = field(default="MEDIUM", init=False, repr=False, compare=False)
    LOW: str = field(default="LOW", init=False, repr=False, compare=False)

    constraint_id: str
    energy_delta: float
    confidence_score: float
    confidence_class: str
    repair_recommended: bool
    evidence: dict = field(default_factory=dict)

    # Override the class-level string constants so they survive dataclass
    # field processing — these need to be class attributes, not instance fields.
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        # Ensure class constants are still accessible as instance attributes
        # (dataclass does not override class attrs with default_factory)
        pass


# Attach class constants AFTER the dataclass is created so they are
# accessible on both the class and instances without being overridden
# by the dataclass machinery.
ViolationConfidence.HIGH = "HIGH"    # type: ignore[misc]
ViolationConfidence.MEDIUM = "MEDIUM"  # type: ignore[misc]
ViolationConfidence.LOW = "LOW"      # type: ignore[misc]


def _assign_class(score: float) -> str:
    """Map a confidence score to a HIGH / MEDIUM / LOW label.

    Spec: REQ-VERIFY-081
    """
    if score >= _HIGH_THRESHOLD:
        return ViolationConfidence.HIGH
    if score >= _MEDIUM_THRESHOLD:
        return ViolationConfidence.MEDIUM
    return ViolationConfidence.LOW


# ---------------------------------------------------------------------------
# ConfidenceVerifier
# ---------------------------------------------------------------------------


class ConfidenceVerifier:
    """Convert binary violations into confidence-weighted ViolationConfidence records.

    **Detailed explanation for engineers:**
        This class wraps any ConstraintExtractor and post-processes its output.
        For each violated constraint, it:

        1. Reads the ``energy_term`` (if present) to get an energy delta.
           If no energy term exists, falls back to the ``satisfied`` metadata
           flag (False → treat as medium confidence by using energy_delta=1.0).
        2. Calls ``confidence_from_energy(energy_delta, temperature)`` to get
           a [0, 1] confidence score.
        3. Assigns a confidence class (HIGH / MEDIUM / LOW).
        4. Calls ``repair_gate(score, threshold)`` to set ``repair_recommended``.
        5. Packages everything into a ``ViolationConfidence`` record.

        Satisfied constraints are silently skipped (they need no attention).

    Args:
        temperature: Sigmoid temperature for energy-to-confidence mapping.
                     Default 1.0. Lower = more decisive confidence scores.

    Spec: REQ-VERIFY-081
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self._temperature = temperature

    def verify_with_confidence(
        self,
        response: str,
        extractor: ConstraintExtractor,
        threshold: float = 0.8,
        domain: str | None = None,
    ) -> list[ViolationConfidence]:
        """Extract constraints and return confidence records for violations only.

        **Detailed explanation for engineers:**
            Calls ``extractor.extract(response, domain)`` then filters to
            violated constraints.  For each violation, derives an energy delta
            (from the ``energy_term`` if available, else a synthetic fallback)
            and converts it to a ``ViolationConfidence``.

            The invariant ``repair_recommended_count ≤ violations_count`` is
            preserved automatically: ``repair_gate`` returns False for any
            score below threshold, so it can only decrease (never increase) the
            number of recommended repairs.

        Args:
            response:   Text to extract constraints from.
            extractor:  Domain extractor (ArithmeticExtractor, AutoExtractor, etc.).
            threshold:  Minimum confidence to set ``repair_recommended=True``.
            domain:     Domain hint passed to extractor. Default ``"auto"``.

        Returns:
            List of ``ViolationConfidence`` records, one per detected violation.
            Satisfied constraints are omitted.

        Spec: REQ-VERIFY-081, SCENARIO-VERIFY-105, SCENARIO-VERIFY-106
        """
        try:
            constraints = extractor.extract(response, domain)
        except Exception:
            return []

        results: list[ViolationConfidence] = []
        for constraint in constraints:
            # Check if the constraint is violated
            is_violated = self._is_violated(constraint)
            if not is_violated:
                continue

            # Derive energy delta
            energy_delta = self._energy_delta(constraint)

            # Compute confidence
            score = confidence_from_energy(energy_delta, self._temperature)
            cls = _assign_class(score)
            recommended = repair_gate(score, threshold=threshold)

            evidence = dict(constraint.metadata) if constraint.metadata else {}
            results.append(
                ViolationConfidence(
                    constraint_id=constraint.description or f"constraint_{len(results)}",
                    energy_delta=energy_delta,
                    confidence_score=score,
                    confidence_class=cls,
                    repair_recommended=recommended,
                    evidence=evidence,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_violated(constraint: object) -> bool:
        """Return True when the constraint is violated.

        Checks ``energy_term.energy()`` if available, then falls back to
        the ``metadata["satisfied"]`` flag.
        """
        # Try energy term first (definitive source)
        energy_term = getattr(constraint, "energy_term", None)
        if energy_term is not None:
            try:
                val = energy_term.energy()
                # If energy returns a JAX / NumPy array, extract scalar
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = float(val.item())
                return val > 0.0
            except Exception:
                pass

        # Fall back to metadata["satisfied"] flag
        metadata = getattr(constraint, "metadata", {}) or {}
        satisfied = metadata.get("satisfied", True)
        return not bool(satisfied)

    @staticmethod
    def _energy_delta(constraint: object) -> float:
        """Extract raw energy delta from constraint.

        Uses ``energy_term.energy()`` when present, else derives a synthetic
        delta from arithmetic metadata (|claimed - correct|) when available,
        otherwise returns 1.0 as a neutral medium-confidence fallback.
        """
        energy_term = getattr(constraint, "energy_term", None)
        if energy_term is not None:
            try:
                val = energy_term.energy()
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return float(val.item())
            except Exception:
                pass

        # Derive from arithmetic metadata for better calibration
        metadata = getattr(constraint, "metadata", {}) or {}
        claimed = metadata.get("claimed_result")
        correct = metadata.get("correct_result")
        if claimed is not None and correct is not None:
            try:
                return abs(float(claimed) - float(correct))
            except (TypeError, ValueError):
                pass

        # Neutral fallback: treat as medium confidence
        return 1.0
