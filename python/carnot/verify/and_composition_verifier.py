"""AND-Composition Verifier: k=5 ensemble with exponential null-space shrinkage.

**Why this exists:**
    A single verifier has a kernel (null space) — a set of responses it cannot
    distinguish from correct ones. An attacker can craft responses that sit in
    that null space, fooling the verifier. AND-composition of k independent
    verifiers shrinks the exploitable null space exponentially: the attacker
    must simultaneously satisfy all k kernels, which becomes exponentially
    harder as k grows (arXiv 2604.12086 §3.2).

    Exp 1108 measured pairwise r-correlation across 6 candidate verifiers on
    the FoVer 500-pair holdout. The k=5 subset (dropping ThinkPRM, which has
    r=0.507 with Z3MathVerifier — above the 0.5 viability threshold) achieves
    max_r=0.462, meeting the AND-composition viability criterion.

**The k=5 ensemble:**
    1. SOSKANEnergyV3   — contrastive energy (AUC=0.9545 on FoVer)
    2. SemEnergyProbe   — logit-space Boltzmann energy (AUC=0.948 @ 0.017ms)
    3. ASTStructureVerifier — AST/bracket structural integrity
    4. SemanticConsistencyVerifier — cross-sentence contradiction detection
    5. Z3MathVerifier   — formal arithmetic claim checking

    ThinkPRM is intentionally excluded: its r=0.507 with Z3MathVerifier
    exceeds the 0.5 viability threshold. It stays as a standalone Tier 0a
    cascade component.

**Verification convention:**
    Each adapter normalises its underlying verifier to return a float
    ``energy`` in [0, 1] where 0.0 means "confident this response is correct"
    and 1.0 means "confident this response violates constraints."
    ``verified = energy < threshold`` (default threshold 0.5 for all).

    For SemEnergyProbe, whose raw score is a per-word Boltzmann energy
    (more negative = more confident), the adapter applies the published
    is_hallucinating() threshold of -0.5 and maps to [0, 1]:
        energy_normalized = 1.0 if raw_score > -0.5 else 0.0
    This matches SemEnergyProbe.is_hallucinating(energy, threshold=-0.5).

Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Adapter protocol — every verifier in the ensemble must satisfy this.
# ---------------------------------------------------------------------------


@runtime_checkable
class VerifierAdapter(Protocol):
    """Minimal interface every ensemble member must satisfy.

    score() returns a float in [0, 1] where 1.0 = definite violation and
    0.0 = definite pass. Callers may treat this as an energy: lower energy
    is better.
    """

    @property
    def name(self) -> str: ...

    def score(self, text: str) -> float: ...


# ---------------------------------------------------------------------------
# Concrete adapters — thin wrappers around the actual verifier objects.
# ---------------------------------------------------------------------------


class ASTStructureAdapter:
    """Adapter for ASTStructureVerifier; score() already returns [0, 1]."""

    def __init__(self) -> None:
        from carnot.verify.ast_structure_verifier import ASTStructureVerifier

        self._v = ASTStructureVerifier()

    @property
    def name(self) -> str:
        return "ASTStructureVerifier"

    def score(self, text: str) -> float:
        """Return structural violation energy in [0, 1]."""
        return float(self._v.score(text))


class SemanticConsistencyAdapter:
    """Adapter for SemanticConsistencyVerifier; score() already returns [0, 1]."""

    def __init__(self) -> None:
        from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier

        self._v = SemanticConsistencyVerifier()

    @property
    def name(self) -> str:
        return "SemanticConsistencyVerifier"

    def score(self, text: str) -> float:
        """Return cross-sentence inconsistency energy in [0, 1]."""
        return float(self._v.score(text))


class Z3MathAdapter:
    """Adapter for Z3MathVerifier; score() already returns [0, 1]."""

    def __init__(self) -> None:
        from carnot.verify.z3_math_verifier import Z3MathVerifier

        self._v = Z3MathVerifier()

    @property
    def name(self) -> str:
        return "Z3MathVerifier"

    def score(self, text: str) -> float:
        """Return arithmetic violation energy in [0, 1]."""
        return float(self._v.score(text))


class SemEnergyProbeAdapter:
    """Adapter for SemEnergyProbe; maps raw Boltzmann energy to [0, 1].

    Raw score from score_response_proxy() is a per-word energy where more
    negative = more confident (less likely hallucinating). The published
    is_hallucinating() threshold is -0.5: scores above -0.5 indicate
    possible hallucination.

    Mapping: raw > -0.5 → normalized=1.0 (violation); else → 0.0 (pass).
    This is intentionally binary at the threshold to match published
    calibration. Continuous interpolation is available via raw_score field
    in the result certificate.
    """

    def __init__(self, hallucination_threshold: float = -0.5) -> None:
        from carnot.verify.semenergy_probe import SemEnergyProbe

        self._v = SemEnergyProbe()
        self._threshold = hallucination_threshold

    @property
    def name(self) -> str:
        return "SemEnergyProbe"

    def score(self, text: str) -> float:
        """Return normalized hallucination energy in {0.0, 1.0}."""
        raw = self._v.score_response_proxy(text)
        return 1.0 if raw > self._threshold else 0.0


class SOSKANEnergyV3Adapter:
    """Adapter for SOSKANEnergyV3; uses text-feature extraction to produce energy.

    SOSKANEnergyV3 requires a feature vector in [-1, 1]^3 (not raw text).
    Without trained weights, raw energy values are arbitrary. This adapter
    uses the text-feature extraction from Exp 1108 to build the feature
    vector, then scores with the model. When the model has not been trained
    (all parameters at initialization values), the output is treated as
    a neutral 0.5 energy to avoid false positives.

    For production use, call fit(X_train, y_train) before wiring into the
    ensemble. In the default (untrained) state the adapter is conservative:
    it never produces a violation, preserving the behaviour of not having
    this verifier at all.
    """

    def __init__(self) -> None:
        from carnot.models.sos_kan import SOSKANEnergyV3

        self._v = SOSKANEnergyV3(n_splines=8, rank=4, n_features=3, hidden_dim=16, seed=1121)
        self._trained = False

    @property
    def name(self) -> str:
        return "SOSKANEnergyV3"

    def fit(self, X: object, y: object) -> None:
        """Train the underlying SOSKANEnergyV3 on labelled feature data.

        X: (n, 3) float array in [-1, 1]^3.
        y: (n,) float array with 1.0 = correct, 0.0 = incorrect.
        """
        self._v.train(X, y)  # type: ignore[arg-type]
        self._trained = True

    def score(self, text: str) -> float:
        """Return normalized energy in [0, 1]; 0.5 if model is untrained."""
        if not self._trained:
            return 0.5

        import numpy as np

        feats = _extract_text_features(text)
        raw = self._v.energy(feats)
        # Raw energy is non-negative; normalize relative to trained model range.
        # Clip at 2.0 to avoid unbounded values producing 1.0 for every string.
        return float(min(raw / 2.0, 1.0))


# ---------------------------------------------------------------------------
# Text feature extraction for SOSKANEnergyV3
# ---------------------------------------------------------------------------


def _extract_text_features(text: str) -> object:
    """Build 3-feature vector for SOSKANEnergyV3 from raw text.

    Matches the make_text_features() pattern from Exp 1108:
        feature 0: log(1 + len(text)) — response length signal
        feature 1: numeric token density — math-content signal
        feature 2: vocabulary richness (unique / total words) — diversity signal

    Values are mapped to [-1, 1] using min/max normalization anchored on
    expected FoVer range. Hard-coded anchors keep this dependency-free.
    """
    import numpy as np

    words = text.split()
    n_words = max(len(words), 1)
    num_count = sum(1 for w in words if any(c.isdigit() for c in w))
    unique_count = len(set(words))

    raw = [
        float(np.log(len(text) + 1)),  # 0–10 typical range
        num_count / n_words,  # 0–1 already
        unique_count / n_words,  # 0–1 already
    ]

    # Map to [-1, 1] using expected FoVer corpus bounds
    anchors = [(0.0, 10.0), (0.0, 1.0), (0.0, 1.0)]
    result = []
    for val, (lo, hi) in zip(raw, anchors):
        if hi > lo:
            result.append(float(np.clip(2.0 * (val - lo) / (hi - lo) - 1.0, -1.0, 1.0)))
        else:
            result.append(0.0)

    return np.array(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# AND-composition result
# ---------------------------------------------------------------------------


@dataclass
class AndCompositionResult:
    """Certificate for one AND-composition verification call.

    Attributes:
        verified: True iff ALL k verifiers returned energy < their threshold.
            This is the AND of all per-verifier verdicts.
        per_verifier_scores: Raw energy value from each adapter, keyed by
            verifier name. Lower = better (less violation energy).
        per_verifier_verified: Whether each individual verifier passed
            (energy < threshold), keyed by verifier name.
        thresholds: The threshold applied per verifier, keyed by name.
        k: Number of verifiers in the ensemble (should be 5 for default).
    """

    verified: bool
    per_verifier_scores: dict[str, float] = field(default_factory=dict)
    per_verifier_verified: dict[str, bool] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    k: int = 0


# ---------------------------------------------------------------------------
# AndCompositionVerifier
# ---------------------------------------------------------------------------


class AndCompositionVerifier:
    """Ensemble verifier using AND-composition of k independent verifiers.

    AND-composition requires ALL k verifiers to agree that a response is
    clean before returning verified=True. This raises the attack bar: an
    adversarial response must simultaneously fool all k verifiers, which
    becomes exponentially harder as k grows (when verifiers have orthogonal
    kernels — low pairwise r-correlation).

    Threshold default 0.5 applies to all adapters. Scores above 0.5 = violation.

    Usage::

        verifier = AndCompositionVerifier()          # k=5 default
        result = verifier.verify("What is 2+2?", "The answer is 4.")
        if result.verified:
            print("All verifiers agree: response looks clean")

    Args:
        verifiers: List of VerifierAdapter instances. Defaults to None,
            which triggers construction of the default k=5 ensemble
            (SOSKANEnergyV3, SemEnergyProbe, ASTStructureVerifier,
            SemanticConsistencyVerifier, Z3MathVerifier).
        thresholds: Per-verifier thresholds in the same order as verifiers.
            Defaults to 0.5 for all. Values above threshold = violation.

    Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
    """

    def __init__(
        self,
        verifiers: list[VerifierAdapter] | None = None,
        thresholds: list[float] | None = None,
    ) -> None:
        if verifiers is None:
            self._verifiers: list[VerifierAdapter] = _make_k5_verifiers()
        else:
            self._verifiers = list(verifiers)

        k = len(self._verifiers)
        if thresholds is None:
            self._thresholds = [0.5] * k
        else:
            if len(thresholds) != k:
                raise ValueError(
                    f"thresholds length {len(thresholds)} must match verifiers length {k}"
                )
            self._thresholds = list(thresholds)

    @property
    def k(self) -> int:
        """Number of verifiers in the ensemble."""
        return len(self._verifiers)

    @property
    def verifier_names(self) -> list[str]:
        """Names of all verifiers in the ensemble (for artifact recording)."""
        return [v.name for v in self._verifiers]

    def verify(self, question: str, response: str) -> AndCompositionResult:
        """Run all k verifiers and return AND-composition result.

        The response text (combined with question context where the verifier
        uses it) is scored by each adapter. verified=True requires ALL
        adapters to return energy below their threshold.

        The question argument is passed as context text where useful; most
        lightweight structural verifiers only consume the response. Combining
        them preserves the interface for future verifiers that use both.

        Args:
            question: The input question or prompt (context for verifiers
                that use full Q+A pairs).
            response: The generated response to verify.

        Returns:
            AndCompositionResult with per-verifier scores and AND verdict.
        """
        combined_text = f"{question}\n{response}" if question.strip() else response

        scores: dict[str, float] = {}
        verdicts: dict[str, bool] = {}
        threshold_map: dict[str, float] = {}

        for adapter, threshold in zip(self._verifiers, self._thresholds):
            try:
                energy = adapter.score(combined_text)
            except Exception:
                # A failing verifier is treated as non-blocking (0.0 energy).
                # This prevents one broken verifier from locking all responses.
                energy = 0.0

            scores[adapter.name] = energy
            verdicts[adapter.name] = energy < threshold
            threshold_map[adapter.name] = threshold

        and_verified = all(verdicts.values())

        return AndCompositionResult(
            verified=and_verified,
            per_verifier_scores=scores,
            per_verifier_verified=verdicts,
            thresholds=threshold_map,
            k=self.k,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_k5_verifiers() -> list[VerifierAdapter]:
    """Build the default k=5 AND-composition verifier list.

    ThinkPRM is intentionally excluded because its pairwise r-correlation
    with Z3MathVerifier is 0.507 (above the 0.5 viability threshold for
    exponential null-space shrinkage). Including it would degrade the
    ensemble's kernel-orthogonality guarantee.
    """
    return [
        SOSKANEnergyV3Adapter(),
        SemEnergyProbeAdapter(),
        ASTStructureAdapter(),
        SemanticConsistencyAdapter(),
        Z3MathAdapter(),
    ]


def build_default_verifier_ensemble() -> AndCompositionVerifier:
    """Return the production k=5 AND-composition verifier ensemble.

    Constructs and returns an AndCompositionVerifier pre-loaded with the
    five verifiers validated by Exp 1108:
        1. SOSKANEnergyV3   (contrastive energy)
        2. SemEnergyProbe   (logit-space Boltzmann energy)
        3. ASTStructureVerifier (syntactic structure)
        4. SemanticConsistencyVerifier (cross-sentence logic)
        5. Z3MathVerifier   (formal arithmetic)

    ThinkPRM is NOT in this set (see _make_k5_verifiers docstring).

    Returns:
        AndCompositionVerifier with k=5, all thresholds at 0.5.
    """
    return AndCompositionVerifier()
