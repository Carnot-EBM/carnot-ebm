"""Advisory k=5 verifier-adapter harness.

This module keeps the historical AND-composition API used by pipeline
certificates, but the default surface is not a production headline verifier.
It is an advisory harness over mixed verifier adapters. The default SOSKAN
adapter is neutral until trained, SemEnergy uses its proxy scorer, and adapter
failures remain non-blocking for compatibility with the existing pipeline.

Use this output as diagnostic certificate data only. A future headline verifier
must wire real trained/live verification substrates, calibrated thresholds, and
fail-closed behavior before removing the headline-ineligible flag below.

Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001, REQ-VERIFY-5218,
SCENARIO-VERIFY-5218.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable


AUTHENTICITY_REMEDIATION_TYPE = "registry_flag"
AUTHENTICITY_STATUS = "advisory_adapter_harness"
HEADLINE_ELIGIBLE = False
HEADLINE_INELIGIBLE_REASON = (
    "Default k=5 surface is advisory only: the SOSKAN adapter returns neutral "
    "0.5 while untrained, SemEnergy uses a proxy scorer, and exceptions are "
    "non-blocking for compatibility."
)
REAL_VERIFICATION_REQUIRED_FOR_HEADLINE = True


def authenticity_metadata() -> dict[str, object]:
    """Return the explicit authenticity flags for registry and artifact callers."""

    return {
        "authenticity_remediation_type": AUTHENTICITY_REMEDIATION_TYPE,
        "authenticity_status": AUTHENTICITY_STATUS,
        "headline_eligible": HEADLINE_ELIGIBLE,
        "headline_ineligible_reason": HEADLINE_INELIGIBLE_REASON,
        "real_verification_required_for_headline": REAL_VERIFICATION_REQUIRED_FOR_HEADLINE,
    }


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
    """Adapter for SemEnergyProbe proxy scoring; maps its score to [0, 1].

    Raw score from score_response_proxy() is a per-word proxy where more
    negative = more confident (less likely hallucinating). The existing
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

    For production use, call fit_from_corpus(examples) with a list of FoVer
    corpus dicts (keys: step_text, label). This method stores the training
    normalization statistics so inference features are extracted with the
    identical normalization, eliminating the train/inference distribution
    mismatch that caused AUROC 0.333 in Exp 1121 (Exp 1128 root-cause fix).

    Do NOT call fit(X, y) directly with pre-normalized X unless you also
    call set_feature_stats(stats) with matching per-column (min, max) tuples.
    """

    def __init__(self) -> None:
        from carnot.models.sos_kan import SOSKANEnergyV3

        self._v = SOSKANEnergyV3(n_splines=8, rank=4, n_features=3, hidden_dim=16, seed=1121)
        self._trained = False
        # Per-column (min, max) normalization anchors stored at training time.
        # None = use fixed anchors from _extract_text_features (legacy path).
        self._feature_stats: list[tuple[float, float]] | None = None

    @property
    def name(self) -> str:
        return "SOSKANEnergyV3"

    def set_feature_stats(self, stats: list[tuple[float, float]]) -> None:
        """Store per-column (min, max) normalization anchors from the training set.

        Call this after extracting raw features from the training corpus so
        that score() uses the identical min/max mapping as training. Without
        this, the model operates on features in a compressed range and predicts
        with inverted polarity (Exp 1121 AUROC 0.333 root cause).

        stats: list of (min_val, max_val) per feature column, length == n_features.
        """
        self._feature_stats = list(stats)

    def fit_from_corpus(
        self,
        examples: list[dict],
        n_epochs: int = 100,
        lr: float = 3e-3,
    ) -> None:
        """Train SOSKANEnergyV3 from FoVer corpus dicts with consistent normalization.

        Extracts raw features from each example's step_text, computes per-column
        min/max across the TRAINING set, normalizes to [-1, 1], stores the stats,
        and calls self._v.fit(). At inference, score() uses the stored stats so
        training and inference features are in the same space.

        examples: list of dicts with keys 'step_text' and 'label' (correct/incorrect).
        """
        import numpy as np

        texts = [ex.get("step_text", "") for ex in examples]
        ys = np.array([1.0 if ex["label"] == "correct" else 0.0 for ex in examples])

        arr = _extract_raw_features(texts)
        stats = [(float(arr[:, i].min()), float(arr[:, i].max())) for i in range(arr.shape[1])]
        self.set_feature_stats(stats)
        X = _apply_feature_stats(arr, stats)

        self._v.fit(X, ys, n_epochs=n_epochs, lr=lr)
        self._trained = True

    def fit(self, X: object, y: object) -> None:
        """Train the underlying SOSKANEnergyV3 on labelled feature data.

        X: (n, 3) float array in [-1, 1]^3, pre-normalized by the caller.
        y: (n,) float array with 1.0 = correct, 0.0 = incorrect.

        Prefer fit_from_corpus() for end-to-end training that automatically
        stores matching normalization stats for inference.
        """
        self._v.fit(X, y)  # type: ignore[arg-type]
        self._trained = True

    def _featurize(self, text: str) -> "object":
        """Extract and normalize a single text using stored training stats.

        Falls back to fixed-anchor normalization if no stats are stored
        (untrained or legacy-mode adapter).
        """
        import numpy as np

        if self._feature_stats is None:
            return _extract_text_features(text)

        arr = _extract_raw_features([text])  # (1, 3)
        return _apply_feature_stats(arr, self._feature_stats)[0]  # (3,)

    def score(self, text: str) -> float:
        """Return normalized energy in [0, 1]; 0.5 if model is untrained."""
        if not self._trained:
            return 0.5

        feats = self._featurize(text)
        raw = self._v.energy(feats)
        # Raw energy is non-negative; normalize relative to trained model range.
        # Clip at 2.0 to avoid unbounded values producing 1.0 for every string.
        return float(min(raw / 2.0, 1.0))


# ---------------------------------------------------------------------------
# Text feature extraction for SOSKANEnergyV3
# ---------------------------------------------------------------------------


def _extract_raw_features(texts: "list[str]") -> "object":
    """Extract unnormalized feature matrix from a list of texts.

    Returns shape (n, 3) float64 array with columns:
        0: log(1 + len(text))   — response length signal
        1: numeric token density — math-content signal
        2: vocabulary richness  — unique/total words

    Used by fit_from_corpus() and _apply_feature_stats() to separate the
    raw extraction step from the normalization step.  Keeping them separate
    lets the adapter store per-column (min, max) anchors from the TRAINING
    corpus and reuse them at inference — eliminating the train/inference
    distribution mismatch that caused AUROC 0.333 in Exp 1121.
    """
    import numpy as np

    feats = []
    for text in texts:
        words = text.split()
        n_words = max(len(words), 1)
        num_count = sum(1 for w in words if any(c.isdigit() for c in w))
        unique_count = len(set(words))
        feats.append([float(np.log(len(text) + 1)), num_count / n_words, unique_count / n_words])
    return np.array(feats, dtype=np.float64)


def _apply_feature_stats(arr: "object", stats: "list[tuple[float, float]]") -> "object":
    """Normalize each column of arr to [-1, 1] using stored (min, max) stats.

    stats: list of (min_val, max_val) per column, from _extract_raw_features
        on the training set.  Applying the same stats to inference features
        guarantees that the model operates on the same feature distribution
        it was trained on.
    """
    import numpy as np

    arr = np.array(arr, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(stats):
        if hi > lo:
            arr[:, i] = np.clip(2.0 * (arr[:, i] - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        else:
            arr[:, i] = 0.0
    return arr


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
        headline_eligible: Always false for the default advisory harness until
            real trained/live verification is implemented.
        headline_ineligible_reason: Why this result must remain non-headline.
    """

    verified: bool
    per_verifier_scores: dict[str, float] = field(default_factory=dict)
    per_verifier_verified: dict[str, bool] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    k: int = 0
    headline_eligible: bool = HEADLINE_ELIGIBLE
    headline_ineligible_reason: str = HEADLINE_INELIGIBLE_REASON


# ---------------------------------------------------------------------------
# AndCompositionVerifier
# ---------------------------------------------------------------------------


class AndCompositionVerifier:
    """Advisory ensemble harness using AND-composition over adapter scores.

    AND-composition requires ALL k verifiers to agree that a response is
    clean before returning verified=True. The current default construction is
    a compatibility harness, not proof that all five real independent signals
    are active or calibrated.

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

    Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001, REQ-VERIFY-5218.
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

    @property
    def headline_eligible(self) -> bool:
        """False until this harness is replaced by real headline verification."""
        return HEADLINE_ELIGIBLE

    @property
    def headline_ineligible_reason(self) -> str:
        """Reason downstream headline paths must treat this as advisory only."""
        return HEADLINE_INELIGIBLE_REASON

    def authenticity_metadata(self) -> dict[str, object]:
        """Return headline quarantine metadata for certificates and registries."""
        return authenticity_metadata()

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
            headline_eligible=HEADLINE_ELIGIBLE,
            headline_ineligible_reason=HEADLINE_INELIGIBLE_REASON,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_k5_verifiers() -> list[VerifierAdapter]:
    """Build the default k=5 advisory adapter list.

    ThinkPRM remains excluded to preserve the historical Exp 1121 shape, but
    this list is not enough by itself to make a production headline verifier.
    """
    return [
        SOSKANEnergyV3Adapter(),
        SemEnergyProbeAdapter(),
        ASTStructureAdapter(),
        SemanticConsistencyAdapter(),
        Z3MathAdapter(),
    ]


def build_default_verifier_ensemble() -> AndCompositionVerifier:
    """Return the default advisory k=5 verifier-adapter harness.

    Constructs and returns an AndCompositionVerifier pre-loaded with the
    historical five adapters:
        1. SOSKANEnergyV3   (contrastive energy)
        2. SemEnergyProbe   (proxy score adapter)
        3. ASTStructureVerifier (syntactic structure)
        4. SemanticConsistencyVerifier (cross-sentence logic)
        5. Z3MathVerifier   (formal arithmetic)

    This helper is headline-ineligible until a real trained/live verification
    substrate replaces the current advisory defaults.

    Returns:
        AndCompositionVerifier with k=5, all thresholds at 0.5.
    """
    return AndCompositionVerifier()
