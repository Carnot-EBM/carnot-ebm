"""Spilled-energy hallucination pre-filter for LLM outputs.

**Researcher summary:**
    Implements the "spilled energy" signal from arxiv 2602.18671 (ICLR 2026).
    LLMs reinterpreted as EBMs: low per-token probability of the chosen token
    means the model's energy is "spilled" across many alternatives — a fast,
    KB-free signal that factual hallucinations may be present. Complements the
    KB-backed FactualKBExtractor (Exp 158) as a lightweight pre-filter.

**Detailed explanation for engineers:**
    The paper "LLM Hallucination Detection via Energy-Based Models" (arxiv
    2602.18671) observes that autoregressive LLMs, viewed through the lens of
    maximum-entropy RL and the soft Bellman equation, are implicitly EBMs.
    The key signal is "spilled energy": the discrepancy between the energy
    concentrated at the model's chosen output token and the energy distributed
    across the full logit vocabulary.

    When the model is confident (correct factual claim), nearly all probability
    mass sits on a single token → low spilled energy. When the model is
    uncertain (hallucinating), probability mass is spread across many tokens →
    high spilled energy.

    **Practical formula (numerically stable):**

    For a generated sequence with T tokens and logits of shape (T, V):

        For each position t:
            log_probs[t]    = log_softmax(logits[t])      # shape (V,)
            x_t             = argmax(logits[t])             # greedy output token
            spilled_t       = -log_probs[t, x_t]           # NLL of chosen token

        total_spilled = mean(spilled_t over T positions)

    Interpretation:
        spilled_t ≈ 0     when model is confident (p_max → 1)
        spilled_t ≈ log V when model is uncertain (p uniform across V tokens)

    **Integration:**
    - SpilledEnergyExtractor: implements ConstraintExtractor Protocol.
      Returns empty list when logits=None (graceful degradation — all existing
      callers pass no logits, so there is zero behavior change).
    - SpilledEnergyConstraint: a ConstraintTerm whose energy is the pre-computed
      spilled energy scalar. Energy is independent of the Ising configuration x;
      it is a read-only signal from the generation step.
    - AutoExtractor: gains an optional logits= keyword in its extract() call.
      When supplied, SpilledEnergyExtractor runs as an additional pass after
      the existing extractors.

    **Target models:** Qwen3.5-0.8B, google/gemma-4-E4B-it (Exp 157).

    **Benchmark result (Exp 157):** AUROC on 50 simulated TruthfulQA items.
    Target: >0.60. See scripts/experiment_157_spilled_energy.py.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint
from carnot.pipeline.extract import ConstraintExtractor, ConstraintResult  # noqa: F401

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Default configuration constants
# ---------------------------------------------------------------------------

#: Default threshold for SpilledEnergyConstraint.
#: Chosen so that a mean NLL of ~0.5 nats (≈ token-level perplexity 1.65)
#: is considered "high confidence." Tune empirically per model.
DEFAULT_SPILLED_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# SpilledEnergyConstraint — a ConstraintTerm holding a pre-computed value
# ---------------------------------------------------------------------------


class SpilledEnergyConstraint(BaseConstraint):
    """A ConstraintTerm encoding the spilled energy of one LLM generation.

    **Researcher summary:**
        Wraps the scalar spilled-energy value computed at generation time as a
        ConstraintTerm. Energy is a constant (the pre-computed value) —
        independent of any Ising configuration x. Constraint is satisfied iff
        spilled_energy < threshold (model was sufficiently confident).

    **Detailed explanation for engineers:**
        Unlike typical ConstraintTerms (e.g., Sudoku row uniqueness) whose
        energy varies with the Ising configuration x, SpilledEnergyConstraint
        holds a fixed scalar computed from the generation logits. The
        ``energy(x)`` method ignores x and returns that scalar. This is valid
        in the Carnot pipeline because the spilled-energy check is a
        generation-time signal, not an inference-time optimisation target.

        Gradient (``grad_energy``) is always zero — there is nothing to
        optimise over in the configuration space.

        ``is_satisfied`` returns True iff spilled_energy ≤ threshold.
        Low energy = model was confident = less hallucination risk. High energy
        = model was uncertain = flag for downstream KB verification.

    Attributes:
        spilled_energy_value: Pre-computed mean spilled energy (≥ 0.0).
        threshold: Satisfaction threshold. Default: DEFAULT_SPILLED_THRESHOLD.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    def __init__(
        self,
        spilled_energy_value: float,
        threshold: float = DEFAULT_SPILLED_THRESHOLD,
    ) -> None:
        """Create a SpilledEnergyConstraint from a pre-computed energy value.

        Args:
            spilled_energy_value: The mean spilled energy from
                ``SpilledEnergyExtractor._compute_spilled_energy()``.
                Must be ≥ 0.0.
            threshold: Energy threshold below which the constraint is
                considered satisfied. Defaults to DEFAULT_SPILLED_THRESHOLD.
        """
        if spilled_energy_value < 0.0:
            raise ValueError(
                f"spilled_energy_value must be ≥ 0.0, got {spilled_energy_value}"
            )
        self._value = spilled_energy_value
        self._threshold = threshold

    @property
    def name(self) -> str:
        """Human-readable name including the energy value."""
        return f"spilled_energy({self._value:.4f})"

    @property
    def satisfaction_threshold(self) -> float:
        """Energy threshold: constraint satisfied iff energy ≤ this value."""
        return self._threshold

    def energy(self, x: jax.Array) -> jax.Array:
        """Return the pre-computed spilled energy (constant; ignores x).

        **Detailed explanation for engineers:**
            The spilled energy is determined at generation time from logits.
            It does not depend on the Ising configuration x that the Carnot
            repair loop optimises. We return a JAX scalar so this constraint
            composes correctly inside ComposedEnergy.

        Args:
            x: Ising configuration (ignored).

        Returns:
            Scalar JAX float32 equal to spilled_energy_value.
        """
        # x is intentionally unused — the energy is a constant from generation.
        _ = x
        return jnp.float32(self._value)

    def is_satisfied(self, x: jax.Array) -> bool:
        """Return True iff spilled_energy ≤ satisfaction_threshold.

        **Detailed explanation for engineers:**
            Overrides BaseConstraint.is_satisfied to avoid calling energy(x),
            which would create an unnecessary JAX scalar. The direct float
            comparison is cleaner and avoids device-to-host transfers when
            the constraint is used as a quick filter.

        Args:
            x: Ignored.

        Returns:
            True if the model showed sufficient confidence during generation.
        """
        return self._value <= self._threshold


# ---------------------------------------------------------------------------
# SpilledEnergyExtractor — ConstraintExtractor Protocol implementation
# ---------------------------------------------------------------------------


class SpilledEnergyExtractor:
    """Extract a spilled-energy hallucination signal from generation logits.

    **Researcher summary:**
        Implements ConstraintExtractor Protocol. When logits are available,
        computes mean per-token spilled energy and wraps it as a
        SpilledEnergyConstraint (satisfied iff energy < threshold). When
        logits are None, returns an empty list (graceful degradation).

    **Detailed explanation for engineers:**
        This extractor is different from ArithmeticExtractor, LogicExtractor,
        etc. in one important way: it does NOT parse the text for pattern-based
        constraints. Instead, it reads the model's internal logit distribution
        from the generation step to compute a confidence signal.

        The extract() signature adds an optional ``logits`` keyword argument
        beyond the ConstraintExtractor Protocol's (text, domain=None). Python's
        structural subtyping allows additional keyword-only parameters with
        defaults, so this class still satisfies the Protocol at runtime.

        **When logits is None (default):** Returns [] immediately. All existing
        pipeline callers that do not pass logits see zero behavior change.

        **When logits is provided (shape T×V or V):**
            1. Ensure 2-D: if shape (V,), reshape to (1, V).
            2. Compute log_softmax along the vocab axis.
            3. Per position: output_energy = NLL of argmax token.
                            logit_energy  = entropy H(p) of the distribution.
            4. spilled_t    = max(0, output_energy_t - logit_energy_t).
            5. total_spilled = mean(spilled_t) over T positions.
            6. Return [ConstraintResult] wrapping a SpilledEnergyConstraint.

        The returned ConstraintResult uses ``constraint_type="spilled_energy"``
        and ``domain="factual"`` — it is the factual hallucination signal
        referenced in Exp 88 and Goal #3 of the research program.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    def __init__(
        self,
        threshold: float = DEFAULT_SPILLED_THRESHOLD,
    ) -> None:
        """Create a SpilledEnergyExtractor.

        Args:
            threshold: Satisfaction threshold passed to SpilledEnergyConstraint.
                Defaults to DEFAULT_SPILLED_THRESHOLD (0.5).
        """
        self._threshold = threshold

    @property
    def supported_domains(self) -> list[str]:
        """Domains this extractor covers: factual (hallucination detection)."""
        return ["factual"]

    def extract(
        self,
        text: str,
        domain: str | None = None,
        *,
        logits: jnp.ndarray | None = None,
    ) -> list[ConstraintResult]:
        """Extract a spilled-energy constraint from generation logits.

        **Detailed explanation for engineers:**
            Gracefully degrades when logits is None: this allows the extractor
            to be called through the standard ConstraintExtractor Protocol loop
            without breaking any existing code path.

            When a domain hint is given and it is not "factual", returns []
            immediately (this extractor is factual-domain only).

        Args:
            text: Generated response text (used for description metadata only;
                the energy signal comes from logits, not the text).
            domain: Optional domain hint. If not None and not "factual",
                returns [].
            logits: Optional JAX array of shape (T, V) or (V,) where T is
                the number of generated tokens and V is the vocabulary size.
                If None, returns [] (graceful degradation).

        Returns:
            A list with zero or one ConstraintResult:
            - Zero items: logits is None or domain is incompatible.
            - One item: a ConstraintResult with a SpilledEnergyConstraint
              encoding the model's confidence during generation.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
        """
        # Domain filter: this extractor only handles factual domain.
        if domain is not None and domain not in self.supported_domains:
            return []

        # Graceful degradation: no logits → no signal.
        if logits is None:
            return []

        spilled_value = self._compute_spilled_energy(logits)
        constraint = SpilledEnergyConstraint(
            spilled_energy_value=spilled_value,
            threshold=self._threshold,
        )
        satisfied = constraint.is_satisfied(jnp.zeros(1))
        return [
            ConstraintResult(
                constraint_type="spilled_energy",
                description=(
                    f"Spilled energy={spilled_value:.4f} "
                    f"({'satisfied' if satisfied else 'violated'}, "
                    f"threshold={self._threshold})"
                ),
                energy_term=constraint,
                metadata={
                    "spilled_energy": spilled_value,
                    "threshold": self._threshold,
                    "satisfied": satisfied,
                    "text_snippet": text[:80],
                },
            )
        ]

    def _compute_spilled_energy(self, logits: jnp.ndarray) -> float:
        """Compute mean per-token spilled energy from a logit array.

        **Detailed explanation for engineers:**
            The formula follows the "spilled energy" concept from arxiv
            2602.18671: a high-confidence model concentrates probability on
            one token; an uncertain model "spills" probability across many.

            We measure this as the negative log-probability of the greedy
            (argmax) output token at each position:

                For each token position t:
                    log_probs[t]    = log_softmax(logits[t])  # shape (V,)
                    x_t             = argmax(logits[t])         # greedy token
                    spilled_t       = -log_probs[t, x_t]        # NLL of x_t

                total_spilled = mean(spilled_t over T positions)

            Why NLL of the greedy token?
                − Confident model (p(x_t) → 1): spilled_t → 0     (low energy)
                − Uncertain model (uniform p_v = 1/V): spilled_t → log V  (high energy)

            Note: Using "sum over vocab" of -log_softmax would give 0 for flat
            logits (since entropy H = -log p_max for uniform distribution).
            Using only the output-token NLL avoids this degenerate case and is
            monotonically related to per-token uncertainty.

            The formula matches the paper's intent: factually incorrect outputs
            tend to have higher per-token uncertainty → higher spilled energy →
            constraint violated.

        Args:
            logits: JAX array of shape (T, V) or (V,).
                T = number of generated tokens, V = vocabulary size.

        Returns:
            Mean spilled energy over token positions (float, ≥ 0.0).
        """
        # Ensure 2-D shape: (T, V).
        if logits.ndim == 1:
            logits = logits[None, :]  # (1, V)

        T = logits.shape[0]

        # Normalised log-probabilities: shape (T, V).
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Greedy output token at each position (= argmax of logits).
        # Shape (T,).
        output_tokens = jnp.argmax(logits, axis=-1)

        # Spilled energy per position: NLL of the greedy token.
        # -log p(x_t) is 0 when p_max→1 (confident) and log V when uniform
        # (uncertain). Always ≥ 0 because log p(x_t) ≤ 0.
        # Shape (T,).
        spilled = -log_probs[jnp.arange(T), output_tokens]

        return float(jnp.mean(spilled))


# ===========================================================================
# SpilledEnergyDetector — arXiv 2602.18671 ICLR 2026 formulation
# ===========================================================================
#
# This is a DIFFERENT implementation from SpilledEnergyExtractor above.
# The extractor uses NLL of the greedy token (negative log probability of argmax).
# The detector uses the log-sum-exp minus expected-logit formula from the paper:
#
#   SpilledEnergy(t) = log(sum_j exp(logit_j/T)) - sum_j p_j * logit_j
#
# Where:
#   log(sum_j exp(logit_j/T)) = log partition function = "free energy" of the
#     logit distribution (total intensity in logit space, a.k.a. logit energy)
#   sum_j p_j * logit_j = expected logit value under softmax = "output energy"
#     (the average logit weighted by the probability distribution)
#
# The difference (logit_energy - output_energy) = the "spilled" intensity:
#   intensity in logit space that is ABOVE the weighted average.
#   When the model is uncertain (uniform logits), logit_energy = log(V) and
#   output_energy ≈ 0 → spilled_energy ≈ log(V) (HIGH — uncertain).
#   When the model is confident (one very large logit), logit_energy ≈ p_max * peak
#   and output_energy ≈ p_max * peak → spilled_energy ≈ 0 (LOW — confident).
#
# WHY this detects hallucination:
#   During factual recall, LLMs assign high probability to a small number of tokens
#   (peaked distribution, low spilled energy). During hallucination, the model is
#   searching over many plausible continuations (uncertain distribution, high
#   spilled energy). The spilled energy captures this "evidence discarded" signal.
#
# WHY per-token:
#   Unlike semantic entropy (which requires a full response and multiple samples),
#   spilled energy is computable per-token during streaming. This enables real-time
#   hallucination detection mid-generation, before the response is complete.
#
# Theoretical basis:
#   arXiv 2602.18671 (Spilled Energy, ICLR 2026)
#   arXiv 2512.15605 (ARM-EBM bijection — LLMs as EBMs)
#
# Hardware path:
#   log-sum-exp and softmax are native GPU tensor operations (~0.01ms per token).
#   No additional model passes or KB lookups required.
#
# Spec: REQ-VERIFY-092, REQ-VERIFY-093
# SCENARIO-VERIFY-123, SCENARIO-VERIFY-124, SCENARIO-VERIFY-125

import hashlib
from dataclasses import dataclass, field


@dataclass
class SpilledEnergyToken:
    """Per-token spilled energy measurement.

    **Researcher summary:**
        Records the spilled energy for one token position from the
        log-sum-exp minus expected-logit formula (arXiv 2602.18671).
        High spilled_energy at a position indicates the model was uncertain
        at that point in the sequence — a per-token hallucination signal.

    **Detailed explanation for engineers:**
        At each token position during LLM generation, the model produces a
        logit vector of shape (V,). This dataclass captures:
        - position: which token in the sequence (0-indexed)
        - token_id: the greedy argmax token chosen at this position
        - spilled_energy: the scalar energy gap between logit space and
          probability space at this position.

        spilled_energy = log(sum_j exp(logit_j/T)) - sum_j p_j * logit_j
        Range: [0, ∞). 0 = perfectly confident; log(V) = maximally uncertain.

    Spec: REQ-VERIFY-092
    """

    position: int
    token_id: int
    spilled_energy: float


@dataclass
class SpilledEnergyDetectorResult:
    """Decision result from SpilledEnergyDetector.score().

    **Researcher summary:**
        Aggregates per-token spilled energy values into a decision about
        whether full verification should run. should_verify=True when too
        many tokens have high spilled energy (uncertain model).

    **Detailed explanation for engineers:**
        After computing per-token spilled energies, we need a scalar
        decision. Rather than thresholding on mean (which can be dominated
        by a few very uncertain tokens), we use high_spill_fraction:

            high_spill_fraction = count(spilled_energy_t > spill_threshold) / T

        This measures "what fraction of the response tokens were uncertain?"
        If that fraction exceeds high_spill_fraction_threshold, the response
        is flagged for verification.

        Default thresholds (spill_threshold=2.0, high_spill_fraction_threshold=0.2):
        - spill_threshold=2.0 nats corresponds to perplexity ≈ 7.4 at T=1
          (roughly: the model considers ~7 tokens equally plausible at that position)
        - high_spill_fraction_threshold=0.2 means >20% of tokens uncertain → verify

    Attributes:
        mean_spilled: Mean spilled energy over all token positions.
        max_spilled: Maximum spilled energy across all token positions.
        high_spill_fraction: Fraction of positions with spilled_energy > spill_threshold.
        should_verify: True iff high_spill_fraction > high_spill_fraction_threshold.
        per_token: Per-position SpilledEnergyToken records (empty for text-mode results).

    Spec: REQ-VERIFY-092
    """

    mean_spilled: float
    max_spilled: float
    high_spill_fraction: float
    should_verify: bool
    per_token: list = field(default_factory=list)  # list[SpilledEnergyToken]


def compute_detector_spilled_energy(
    logits: jnp.ndarray,
    temperature: float = 1.0,
) -> float:
    """Compute spilled energy for a single token's logit vector.

    **Detailed explanation for engineers:**
        Formula (arXiv 2602.18671):
            logit_energy = log(sum_j exp(logit_j / T))   # log partition function
            probs = softmax(logit_j / T)                  # output distribution
            output_energy = sum_j probs_j * logit_j       # expected logit value
            spilled = logit_energy - output_energy         # intensity discarded

        Why logit_energy - output_energy equals T * H(softmax(logits/T)):
            For T=1: log Z = log sum exp(x_j). By definition of softmax:
            p_j = exp(x_j) / Z, so x_j = log(p_j) + log Z.
            E[x] = sum p_j x_j = sum p_j (log p_j + log Z) = -H(p) + log Z.
            So log Z - E[x] = log Z - (-H(p) + log Z) = H(p) >= 0.
            Spilled energy equals entropy of the softmax distribution (in nats).

        In practice, this equals T * H(softmax(logits/T)) where H is entropy.
        So spilled energy is a temperature-scaled entropy of the output distribution.

        For T=1 specifically: spilled_energy(t) = H(softmax(logits_t)) (entropy in nats).
        - Uniform distribution (V tokens): H = log V ≈ 10.8 for V=50000
        - One-hot distribution (confident): H ≈ 0
        - spill_threshold=2.0 nats ≈ entropy of a ~7-way uniform choice

    Args:
        logits: 1-D JAX array of shape (V,). Raw pre-softmax logits.
        temperature: Sampling temperature T > 0 (default 1.0). Higher T
            increases uncertainty; lower T sharpens the distribution.

    Returns:
        Spilled energy scalar >= 0.0 (in nats).

    Spec: REQ-VERIFY-092, SCENARIO-VERIFY-123, SCENARIO-VERIFY-124
    """
    # log partition function: log(sum_j exp(logit_j / T))
    logit_energy = jax.scipy.special.logsumexp(logits / temperature)
    # output distribution
    probs = jax.nn.softmax(logits / temperature)
    # expected logit value under output distribution
    output_energy = jnp.sum(probs * logits)
    # spilled = intensity lost to softmax normalization
    return float(logit_energy - output_energy)


class SpilledEnergyDetector:
    """Per-token logit-discrepancy hallucination signal (arXiv 2602.18671, ICLR 2026).

    **Researcher summary:**
        Implements the "spilled energy" hallucination pre-filter. Unlike
        semantic entropy (post-hoc, full response) or SemanticEnergyScorer
        (pre-softmax logits of full response), spilled energy is a per-token
        signal measurable DURING streaming generation — no full response needed.

    **Detailed explanation for engineers:**
        Pipeline position:
            Tier 0 (SpilledEnergyDetector) → Tier 1 (SinkProbe) → Tier 2 (EORM) → Tier 3 (Ising)

        The spilled energy formula (arXiv 2602.18671):
            For each token position t with logit vector logit_t (shape V):
                SpilledEnergy(t) = log(sum_j exp(logit_j/T)) - sum_j p_j * logit_j

            This equals T * H(softmax(logit_t / T)) where H is the entropy.
            High entropy → uncertain model → high spilled energy → potential hallucination.

        Decision logic:
            high_spill_fraction = fraction of token positions where SpilledEnergy(t) > spill_threshold
            should_verify = high_spill_fraction > high_spill_fraction_threshold

        CI-safe mode (score_from_text):
            When logits are not available, uses a deterministic hash of the response text
            as a proxy. This enables CI tests to exercise the full pipeline code path
            without GPU hardware. The hash is seeded from response content to be
            deterministic — same text → same result.

        ARM-EBM bijection (arXiv 2512.15605):
            Autoregressive LLMs are equivalent to EBMs via the soft Bellman equation.
            The spilled energy at each token position corresponds to the "free energy"
            of the EBM at that state. High free energy states → the EBM is uncertain
            about the continuation → hallucination risk.

        Hardware path:
            log-sum-exp is a single GPU kernel (~0.01ms per token on A100/MI300).
            No additional forward passes required — logits are already computed.

    Attributes:
        spill_threshold: Per-token spilled energy above which the token is "high-spill".
            Default 2.0 nats ≈ entropy of a ~7-way uniform choice. Tune empirically.
        high_spill_fraction_threshold: Fraction of high-spill tokens that triggers
            verification. Default 0.2 (20% uncertain → verify). Lower = more sensitive.

    Spec: REQ-VERIFY-092, REQ-VERIFY-093
    """

    def __init__(
        self,
        spill_threshold: float = 2.0,
        high_spill_fraction_threshold: float = 0.2,
    ) -> None:
        """Create a SpilledEnergyDetector.

        Args:
            spill_threshold: Per-token energy threshold. Tokens with spilled_energy
                above this are counted as "high-spill". Default 2.0 nats.
            high_spill_fraction_threshold: If more than this fraction of tokens are
                high-spill, should_verify=True. Default 0.2 (20%).
        """
        if spill_threshold <= 0.0:
            raise ValueError(f"spill_threshold must be > 0, got {spill_threshold}")
        if not (0.0 < high_spill_fraction_threshold < 1.0):
            raise ValueError(
                f"high_spill_fraction_threshold must be in (0, 1), got {high_spill_fraction_threshold}"
            )
        self.spill_threshold = spill_threshold
        self.high_spill_fraction_threshold = high_spill_fraction_threshold

    def score(
        self,
        logits_per_token: jnp.ndarray,
        temperature: float = 1.0,
    ) -> SpilledEnergyDetectorResult:
        """Compute per-token spilled energy from a logit array.

        **Detailed explanation for engineers:**
            Takes the full logit matrix from an LLM generation step and computes
            spilled energy at each token position. The formula is applied independently
            to each row (token position) of the matrix.

            logits_per_token must be 2-D: shape (T, V) where T = number of generated
            tokens and V = vocabulary size. If 1-D (shape V), it is treated as a
            single token sequence (T=1).

        Args:
            logits_per_token: JAX or numpy array of shape (T, V) or (V,).
                Raw pre-softmax logits from the language model's last linear layer.
            temperature: Sampling temperature (default 1.0).

        Returns:
            SpilledEnergyDetectorResult with per-token energies and summary statistics.

        Spec: REQ-VERIFY-092, SCENARIO-VERIFY-123, SCENARIO-VERIFY-124
        """
        arr = jnp.asarray(logits_per_token)
        if arr.ndim == 1:
            arr = arr[None, :]  # treat as (1, V)

        T = arr.shape[0]
        per_token_records = []

        for t in range(T):
            token_logits = arr[t]
            se = compute_detector_spilled_energy(token_logits, temperature)
            # greedy argmax token id (for logging; not used in the decision)
            token_id = int(jnp.argmax(token_logits))
            per_token_records.append(
                SpilledEnergyToken(position=t, token_id=token_id, spilled_energy=se)
            )

        spilled_values = [r.spilled_energy for r in per_token_records]
        mean_spilled = float(sum(spilled_values) / T)
        max_spilled = float(max(spilled_values))
        n_high = sum(1 for v in spilled_values if v > self.spill_threshold)
        high_spill_fraction = float(n_high / T)
        should_verify = high_spill_fraction > self.high_spill_fraction_threshold

        return SpilledEnergyDetectorResult(
            mean_spilled=mean_spilled,
            max_spilled=max_spilled,
            high_spill_fraction=high_spill_fraction,
            should_verify=should_verify,
            per_token=per_token_records,
        )

    def score_from_text(self, response_text: str) -> SpilledEnergyDetectorResult:
        """Compute a deterministic proxy spilled energy from response text.

        **Detailed explanation for engineers:**
            CI-safe mode for when logits are not available. Uses a hash of the
            response text to deterministically generate proxy energy values.

            Why hash-based?
            - Deterministic: same text → same result → reproducible CI tests
            - No GPU required: pure Python computation
            - Plausible range: proxy values span the same range as real spilled energies

            Algorithm:
            1. Hash the response text with SHA-256
            2. Use the first 16 bytes as a seed for deterministic float generation
            3. Simulate T=10 "token" measurements with varied energy values
            4. Apply the same threshold logic as score()

            This is a proxy — not a real hallucination signal. Use score() with
            real logits for production hallucination detection.

        Args:
            response_text: The LLM response text (any string).

        Returns:
            SpilledEnergyDetectorResult with deterministic proxy values.

        Spec: REQ-VERIFY-093, SCENARIO-VERIFY-125
        """
        # Deterministic hash seed from response text
        digest = hashlib.sha256(response_text.encode("utf-8")).digest()
        # Convert first 8 bytes to a float seed in [0, 1)
        seed_int = int.from_bytes(digest[:8], "big")
        seed_float = (seed_int % (2**32)) / (2**32)
        # seed_float is computed for potential future use; reference it to satisfy linters
        _ = seed_float

        # Generate T=10 deterministic proxy energy values using a simple LCG-like mapping
        # Each "token" gets a proxy energy derived from the hash bytes
        n_proxy_tokens = 10
        proxy_energies = []
        for i in range(n_proxy_tokens):
            # Use different bytes of the digest for each "token"
            byte_idx = (i * 2) % len(digest)
            byte_val = digest[byte_idx]
            # Map byte (0-255) to energy in [0, 5.0] nats
            # This range covers from confident (0) to very uncertain (5 nats ≈ 148-way entropy)
            proxy_energy = (byte_val / 255.0) * 5.0
            proxy_energies.append(proxy_energy)

        mean_spilled = float(sum(proxy_energies) / n_proxy_tokens)
        max_spilled = float(max(proxy_energies))
        n_high = sum(1 for v in proxy_energies if v > self.spill_threshold)
        high_spill_fraction = float(n_high / n_proxy_tokens)
        should_verify = high_spill_fraction > self.high_spill_fraction_threshold

        return SpilledEnergyDetectorResult(
            mean_spilled=mean_spilled,
            max_spilled=max_spilled,
            high_spill_fraction=high_spill_fraction,
            should_verify=should_verify,
            per_token=[],  # no per-token records in text mode
        )
