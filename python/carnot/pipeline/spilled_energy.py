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
