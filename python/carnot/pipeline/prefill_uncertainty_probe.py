"""Prefill Uncertainty Probe — pre-generation hallucination risk gate.

**Researcher summary:**
    Implements a prefill-stage hallucination risk detector based on the Neural
    Uncertainty Principle (arXiv 2603.19562, Mar 2026), which proves that
    adversarial vulnerability and hallucination share a geometric origin: input
    and loss-gradient are conjugate observables with an irreducible uncertainty
    bound analogous to Heisenberg's uncertainty principle.

    Key insight: we can estimate hallucination RISK before any output tokens are
    generated, using only the logit distribution from the model's first forward
    pass on the input prompt (black-box friendly — no gradient access required).

**Detailed explanation for engineers:**
    The fundamental insight from arXiv 2603.19562 is that:

        |⟨x, ∇L⟩|² ≤ ‖x‖² · ‖∇L‖²   (Cauchy-Schwarz / uncertainty bound)

    where x is the input representation and ∇L is the loss gradient.  When this
    product is large, the model is highly sensitive to input perturbations —
    the same condition that produces hallucinations.

    Full gradient computation requires white-box model access (back-prop through
    the model).  We provide an entropy-based black-box approximation:

        uncertainty_score = H(softmax(logits)) / log(V)

    where H is Shannon entropy (in nats) and V is the vocabulary size.  This
    normalises the score to [0, 1]:
    - score ≈ 1.0: uniform distribution → maximum uncertainty → high hallucination risk
    - score ≈ 0.0: peaked distribution → model is confident → low hallucination risk

    The conjugate_bound field approximates the Cauchy-Schwarz factor using logit
    statistics as proxies:
    - input_norm proxy: RMS of the logit vector (‖logits‖ / √V)
    - gradient_norm proxy: std of the logit vector (spread = sensitivity proxy)

    Pipeline integration:
    - PrefillUncertaintyProbe.probe(): fires BEFORE generation
    - VerifyRepairPipeline.check_prefill_uncertainty(): fast-path skip when safe
    - SpilledEnergyExtractor (see spilled_energy_extractor.py): fires AFTER generation
    Together they cover the full verify cycle: pre-generation gate + post-generation check.

    Key classes / functions:
    - PrefillUncertaintyResult: dataclass with all probe outputs
    - compute_input_uncertainty(embeddings): white-box variance of embedding norms
    - compute_conjugate_bound(input_norm, gradient_norm): Cauchy-Schwarz factor
    - compute_prompt_uncertainty(logits, threshold): black-box entropy approximation
    - PrefillUncertaintyProbe: thin class wrapping compute_prompt_uncertainty

Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Core data type
# ---------------------------------------------------------------------------


@dataclass
class PrefillUncertaintyResult:
    """Result of a prefill-stage uncertainty analysis.

    **Detailed explanation for engineers:**
        All numeric fields are Python floats for JSON compatibility.
        ``high_risk`` and ``threshold_exceeded`` carry the same boolean verdict;
        both are exposed to make downstream routing code more readable.

        The score is normalised to [0, 1] so threshold comparisons are stable
        regardless of vocabulary size.  A threshold of 0.5 means: "flag when
        the distribution is more than halfway to maximum entropy."

    Attributes:
        uncertainty_score: Normalised Shannon entropy H(p) / log(V) ∈ [0, 1].
            1.0 = uniform distribution (maximum uncertainty).
            0.0 = deterministic distribution (one token has all mass).
        conjugate_bound: Cauchy-Schwarz bound proxy: rms(logits) * std(logits).
            Approximates ‖x‖ · ‖∇L‖ from the Neural Uncertainty Principle.
            Larger values indicate higher sensitivity to input perturbations.
        high_risk: True when uncertainty_score > threshold.  Use this field for
            pipeline routing decisions (e.g. skip vs. trigger full verification).
        threshold_exceeded: Alias for high_risk with an explicit name that
            communicates intent: the threshold was exceeded → take action.
        n_tokens: Number of vocabulary tokens V in the logit distribution.
        computation_method: Either ``"entropy_approximation"`` (black-box,
            derived from logit distribution) or ``"embedding_variance"``
            (white-box, derived from token embedding norms).

    Spec: REQ-VERIFY-080
    """

    uncertainty_score: float
    conjugate_bound: float
    high_risk: bool
    threshold_exceeded: bool
    n_tokens: int
    computation_method: str


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_input_uncertainty(embeddings: np.ndarray) -> float:
    """Compute uncertainty from the variance of per-token embedding L2 norms.

    **Detailed explanation for engineers:**
        White-box approximation of input uncertainty — requires access to the
        model's token embedding matrix (or hidden states at the input layer).

        The intuition: if all token embeddings have similar magnitudes, the model
        is treating all tokens "equally" (low discrimination → low uncertainty
        from embedding perspective).  When embedding norms vary widely, the model
        has learned strong distinctions between tokens, but HIGH variance of norms
        across the input sequence means the input spans a large region of
        representation space → higher uncertainty about which region the answer lies in.

        Formula:
            norms = [‖e_i‖₂ for each token i]   shape: (T,)
            uncertainty = Var(norms)             scalar

        For a single token or all-identical embeddings, variance = 0 (certain).

    Args:
        embeddings: 2-D float array of shape (T, D), where T is the number of
            input tokens and D is the embedding dimension.  Float32 or float64.

    Returns:
        Variance of per-token L2 norms as a float (≥ 0).

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    # Compute L2 norm of each token embedding: shape (T,)
    norms = np.linalg.norm(embeddings, axis=-1)
    # Variance of norms measures spread of representation magnitudes across tokens.
    return float(np.var(norms))


def compute_conjugate_bound(input_norm: float, gradient_norm: float) -> float:
    """Compute the Cauchy-Schwarz conjugate bound factor.

    **Detailed explanation for engineers:**
        From the Neural Uncertainty Principle (arXiv 2603.19562):

            |⟨x, ∇L⟩|² ≤ ‖x‖² · ‖∇L‖²

        The right-hand side is the squared Cauchy-Schwarz bound.  This function
        returns the FACTOR (not squared):

            conjugate_bound = ‖x‖ · ‖∇L‖ = input_norm * gradient_norm

        This is proportional to the maximum possible alignment between the input
        representation and the loss gradient.  Larger values → more sensitivity
        to perturbations → higher hallucination risk.

        In practice, ``compute_prompt_uncertainty`` estimates input_norm and
        gradient_norm from the logit distribution (black-box proxies), then
        calls this function to fill the ``conjugate_bound`` field.

    Args:
        input_norm: L2 norm of the input representation (‖x‖).
        gradient_norm: L2 norm of the loss gradient (‖∇L‖), or a proxy thereof.

    Returns:
        Product input_norm * gradient_norm (≥ 0.0).

    Spec: REQ-VERIFY-080
    """
    return float(input_norm) * float(gradient_norm)


def compute_prompt_uncertainty(
    logits_first_pass: np.ndarray,
    threshold: float = 0.5,
) -> PrefillUncertaintyResult:
    """Compute prefill uncertainty from the first-pass token logit distribution.

    **Detailed explanation for engineers:**
        Black-box approximation: works with ANY model that exposes logit outputs.
        Does not require gradient computation or model internals.

        The algorithm:
        1. Flatten to 1-D: accept shape (V,) or (1, V).
        2. Compute softmax probabilities: p_i = exp(logit_i) / sum(exp(logits))
           (using the log-sum-exp trick for numerical stability).
        3. Shannon entropy: H = −sum_i(p_i · log(p_i)), with 0·log(0) → 0.
        4. Normalise: uncertainty_score = H / log(V).
           - log(V) is the maximum possible entropy (uniform distribution).
           - For V=1, log(V)=0 → define uncertainty_score=0 (only one token).
        5. conjugate_bound proxies (all derived from logit statistics):
           - input_norm ≈ rms(logits) = ‖logits‖₂ / √V
           - gradient_norm ≈ std(logits) (spread in logit space ≈ sensitivity)
           - bound = input_norm * gradient_norm
        6. high_risk = uncertainty_score > threshold (strict >).

        The choice of RMS as input_norm proxy and std as gradient_norm proxy is
        motivated by the observation that logit RMS correlates with the scale of
        the input representation (both grow with model confidence), while logit
        std correlates with sensitivity to input changes (a gradient-like signal).

    Args:
        logits_first_pass: 1-D array of shape (V,) or 2-D of shape (1, V).
            Raw (pre-softmax) logit values from the model's first forward pass
            on the input prompt.
        threshold: Normalised entropy threshold in (0, 1).  ``high_risk`` is
            set to True when ``uncertainty_score > threshold``.  Default 0.5.

    Returns:
        PrefillUncertaintyResult with all fields populated.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
    """
    logits = np.asarray(logits_first_pass, dtype=np.float64)

    # Validate and flatten shape: accept (V,) or (1, V) only.
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError(
                f"logits_first_pass must be shape (V,) or (1, V); "
                f"got {logits.shape}.  For multi-token sequences use shape (V,) "
                f"on the first-token distribution only."
            )
        logits = logits.reshape(-1)
    elif logits.ndim != 1:
        raise ValueError(
            f"logits_first_pass must be 1-D (V,) or 2-D (1, V); got ndim={logits.ndim}"
        )

    vocab_size = int(logits.shape[0])

    # Handle empty vocabulary (degenerate case): return score=0, no hallucination risk.
    if vocab_size == 0:
        return PrefillUncertaintyResult(
            uncertainty_score=0.0,
            conjugate_bound=0.0,
            high_risk=False,
            threshold_exceeded=False,
            n_tokens=0,
            computation_method="entropy_approximation",
        )

    # ---------------------------------------------------------------------------
    # Numerically stable softmax via log-sum-exp trick.
    # ---------------------------------------------------------------------------
    max_logit = float(np.max(logits))
    shifted = logits - max_logit                     # prevent overflow in exp()
    exp_shifted = np.exp(shifted)
    sum_exp = float(np.sum(exp_shifted))
    log_sum_exp = math.log(sum_exp)                  # = log(sum(exp(logits - max)))
    # log_probs[i] = logit[i] - max_logit - log(sum_exp(shifted))
    log_probs = shifted - log_sum_exp                # shape (V,)
    probs = np.exp(log_probs)                        # shape (V,); sums to 1

    # ---------------------------------------------------------------------------
    # Shannon entropy: H = −sum(p * log(p)), with 0·log(0) → 0.
    # ---------------------------------------------------------------------------
    # Where probs ≈ 0, log_probs ≈ -inf, but the product probs * log_probs → 0.
    # IEEE754 handles 0 * (-inf) = NaN, so mask explicitly.
    mask = probs > 0.0
    entropy_nats = float(-np.sum(probs[mask] * log_probs[mask]))

    # Normalise by log(V): maximum entropy of a uniform over V tokens.
    if vocab_size <= 1:
        # Only one token → always deterministic, entropy = 0.
        uncertainty_score = 0.0
    else:
        max_entropy = math.log(float(vocab_size))   # log(V) in nats
        uncertainty_score = entropy_nats / max_entropy

    # Clamp to [0, 1] to guard against floating-point rounding at the boundary.
    uncertainty_score = min(1.0, max(0.0, uncertainty_score))

    # ---------------------------------------------------------------------------
    # Conjugate bound proxies from logit statistics.
    # ---------------------------------------------------------------------------
    # input_norm proxy: RMS of the logit vector (scale of model's "attention").
    rms_logits = float(np.sqrt(np.mean(logits ** 2))) if vocab_size > 0 else 0.0
    # gradient_norm proxy: std of the logit vector (spread = sensitivity signal).
    std_logits = float(np.std(logits)) if vocab_size > 0 else 0.0
    conjugate = compute_conjugate_bound(rms_logits, std_logits)

    high_risk = uncertainty_score > threshold

    return PrefillUncertaintyResult(
        uncertainty_score=uncertainty_score,
        conjugate_bound=conjugate,
        high_risk=high_risk,
        threshold_exceeded=high_risk,
        n_tokens=vocab_size,
        computation_method="entropy_approximation",
    )


# ---------------------------------------------------------------------------
# Main probe class
# ---------------------------------------------------------------------------


class PrefillUncertaintyProbe:
    """Detect hallucination risk from the first-pass logit distribution.

    **Detailed explanation for engineers:**
        This class is a thin, stateless wrapper around ``compute_prompt_uncertainty``
        that provides a clean, object-oriented entry point consistent with the
        style of ``SpilledEnergyExtractor`` (see spilled_energy_extractor.py).

        Usage pattern:
            probe = PrefillUncertaintyProbe()
            logits = model.forward(prompt_tokens)   # first-pass logits
            result = probe.probe(logits, threshold=0.5)
            if result.high_risk:
                # Trigger full Ising verification.
                ...
            else:
                # Fast-path skip — model appears confident.
                ...

        The probe fires BEFORE generation begins (pre-fill stage), so it adds
        only one forward-pass overhead.  Compare with SpilledEnergyExtractor
        which fires AFTER generation and requires the full logit sequence.

    Spec: REQ-VERIFY-080
    """

    DEFAULT_THRESHOLD: float = 0.5
    """Default normalised-entropy threshold.  Calibrate on your corpus."""

    def probe(
        self,
        logits_first_pass: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> PrefillUncertaintyResult:
        """Run the prefill uncertainty probe on a logit array.

        **Detailed explanation for engineers:**
            Delegates to ``compute_prompt_uncertainty``.  Accepts both 1-D (V,)
            and 2-D (1, V) logit arrays for convenience.

        Args:
            logits_first_pass: Raw logit array from the model's first forward
                pass.  Shape (V,) or (1, V), dtype float32 or float64.
            threshold: Normalised entropy threshold in (0, 1).  high_risk is
                True when uncertainty_score > threshold.  Default 0.5.

        Returns:
            PrefillUncertaintyResult with all fields populated.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
        """
        return compute_prompt_uncertainty(logits_first_pass, threshold=threshold)
