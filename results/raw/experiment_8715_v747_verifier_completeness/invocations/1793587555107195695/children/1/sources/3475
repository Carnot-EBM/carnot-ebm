"""Semantic energy computation from LLM token log-probabilities.

**Researcher summary:**
    Implements E_semantic = mean(-logprob_i) as a hallucination detector.
    Threshold-based binary classification: energy > threshold => hallucination.

**Detailed explanation for engineers:**
    Large Language Models (LLMs) like GPT-4 or Claude produce a probability
    distribution over possible next tokens at each generation step. The
    "log-probability" (logprob) of the chosen token tells us how confident
    the model was: a logprob near 0 means high confidence (probability near 1),
    while a very negative logprob (like -8) means low confidence.

    The key insight connecting LLMs to Energy-Based Models (EBMs) is that
    -logprob IS an energy. In an EBM, low energy = likely, high energy =
    unlikely. In an LLM, high logprob = likely, low logprob = unlikely.
    So -logprob maps directly to the EBM energy concept.

    This module computes:
    1. Per-token energy: -logprob for each token (always >= 0)
    2. Semantic energy: the mean of per-token energies across the response
    3. Hallucination flag: True if semantic energy exceeds a threshold

    A high semantic energy means the LLM was uncertain about many tokens,
    which is a strong empirical signal for hallucinated (fabricated) content.

    **Why "semantic" energy?**
    We call it "semantic" to distinguish it from the physical energy in EBM
    models (like Ising or Boltzmann). Here the energy comes from the LLM's
    own confidence scores, not from a learned energy function. But the
    mathematical structure is identical: low energy = confident/correct,
    high energy = uncertain/hallucinated.

    **What is a logprob?**
    If the LLM assigns probability p to a token, the logprob is log(p).
    Since 0 < p <= 1, logprob is always <= 0. For example:
    - p = 0.95 (very confident) => logprob = -0.051
    - p = 0.5 (coin flip) => logprob = -0.693
    - p = 0.01 (very uncertain) => logprob = -4.605

    **What threshold to use?**
    The default threshold of 2.0 corresponds roughly to an average token
    probability of exp(-2) ~ 0.135, meaning the model was on average only
    ~13.5% confident per token. In practice, thresholds should be tuned per
    model and domain. Factual QA might use a lower threshold (1.5), while
    creative writing might tolerate higher energy (3.0).

Spec: REQ-INFER-008, REQ-INFER-009
"""

from __future__ import annotations

from dataclasses import dataclass, field


def compute_semantic_energy(logprobs: list[float]) -> float:
    """Compute semantic energy as the mean of negated log-probabilities.

    **Researcher summary:**
        E_semantic = (1/N) * sum(-logprob_i) for i in 1..N tokens.

    **Detailed explanation for engineers:**
        Given a list of log-probabilities from an LLM response (one per token),
        this function negates each value and takes the mean. Since logprobs are
        always <= 0, the negated values are always >= 0, and so the energy is
        always >= 0.

        - Energy near 0: the LLM was very confident about every token
        - Energy around 1: moderate confidence (average token prob ~37%)
        - Energy above 2: low confidence, likely hallucination
        - Energy above 5: very low confidence, almost certainly hallucinated

    Args:
        logprobs: A list of token log-probabilities from the LLM response.
            Each value should be <= 0 (log of a probability). An empty list
            returns 0.0 to handle edge cases gracefully.

    Returns:
        A non-negative float representing the semantic energy. Higher values
        indicate lower LLM confidence and higher hallucination risk.

    Examples:
        >>> compute_semantic_energy([-0.1, -0.2, -0.1])
        0.1333...  # Low energy, confident response
        >>> compute_semantic_energy([-5.0, -6.0, -4.5])
        5.1666...  # High energy, likely hallucination

    Spec: REQ-INFER-008
    """
    if not logprobs:
        return 0.0
    # Negate each logprob (turning negative values into positive energies)
    # then take the mean across all tokens in the response.
    token_energies = [-lp for lp in logprobs]
    return sum(token_energies) / len(token_energies)


def classify_hallucination(energy: float, threshold: float = 2.0) -> bool:
    """Classify whether a response is a hallucination based on its semantic energy.

    **Researcher summary:**
        Binary classifier: energy > threshold => hallucination.

    **Detailed explanation for engineers:**
        This is a simple threshold-based classifier. If the semantic energy
        (computed by ``compute_semantic_energy``) exceeds the threshold, the
        response is flagged as a likely hallucination.

        The threshold is configurable because different use cases have different
        tolerance levels:
        - Factual QA (low tolerance): threshold = 1.5
        - General conversation: threshold = 2.0 (default)
        - Creative writing (high tolerance): threshold = 3.0

        Note: This is a soft signal, not a guarantee. A high-energy response
        might be correct but unusual, and a low-energy response might still
        contain subtle errors. For hard guarantees, use Carnot's constraint
        verification system (REQ-VERIFY-001 through REQ-VERIFY-007).

    Args:
        energy: The semantic energy value (from ``compute_semantic_energy``).
        threshold: The energy threshold above which a response is classified
            as a hallucination. Must be positive. Default is 2.0.

    Returns:
        True if the energy exceeds the threshold (likely hallucination),
        False otherwise (likely reliable).

    Examples:
        >>> classify_hallucination(0.5)
        False  # Low energy, confident
        >>> classify_hallucination(3.0)
        True  # High energy, likely hallucination

    Spec: REQ-INFER-009, SCENARIO-INFER-008, SCENARIO-INFER-009
    """
    return energy > threshold


@dataclass
class SemanticEnergyResult:
    """Complete result of semantic energy analysis for an LLM response.

    **Researcher summary:**
        Bundles energy, hallucination flag, per-token energies, and threshold
        into a single structured result.

    **Detailed explanation for engineers:**
        After computing semantic energy and classifying hallucination, this
        dataclass holds all the results together. It also stores the per-token
        energies so you can inspect *which specific tokens* the LLM was least
        confident about -- useful for debugging and understanding where in the
        response the hallucination likely occurred.

        The ``from_logprobs`` class method is a convenience constructor that
        does the full computation pipeline in one call.

    Attributes:
        energy: The scalar semantic energy (mean of negated logprobs).
            Default 0.0 (no tokens analyzed).
        is_hallucination: Whether the energy exceeds the threshold.
            Default False.
        token_energies: Per-token energy values (-logprob for each token).
            Default empty list. Useful for finding which tokens were
            most uncertain.
        threshold: The threshold used for classification. Default 2.0.

    Examples:
        >>> result = SemanticEnergyResult.from_logprobs([-0.1, -5.0, -0.2])
        >>> result.energy  # mean of [0.1, 5.0, 0.2] = 1.766...
        >>> result.is_hallucination  # False (1.766 < 2.0)
        >>> result.token_energies  # [0.1, 5.0, 0.2]

    Spec: REQ-INFER-008, REQ-INFER-009
    """

    energy: float = 0.0
    is_hallucination: bool = False
    token_energies: list[float] = field(default_factory=list)
    threshold: float = 2.0

    @classmethod
    def from_logprobs(cls, logprobs: list[float], threshold: float = 2.0) -> SemanticEnergyResult:
        """Construct a result from raw LLM log-probabilities.

        **Researcher summary:**
            One-shot pipeline: logprobs -> token energies -> mean energy ->
            classification -> structured result.

        **Detailed explanation for engineers:**
            This is the recommended way to create a SemanticEnergyResult.
            It computes everything from the raw logprobs in one call:

            1. Negate each logprob to get per-token energies
            2. Compute mean energy (the semantic energy)
            3. Classify as hallucination or not
            4. Bundle everything into this dataclass

        Args:
            logprobs: List of token log-probabilities from the LLM.
            threshold: Classification threshold. Default 2.0.

        Returns:
            A fully populated SemanticEnergyResult.

        Spec: REQ-INFER-008, REQ-INFER-009
        """
        token_energies = [-lp for lp in logprobs]
        energy = compute_semantic_energy(logprobs)
        is_hall = classify_hallucination(energy, threshold)
        return cls(
            energy=energy,
            is_hallucination=is_hall,
            token_energies=token_energies,
            threshold=threshold,
        )
