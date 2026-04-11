"""Lookahead-energy hallucination signal for LLM outputs.

**Researcher summary:**
    Implements the "lookahead energy" signal from arxiv 2512.15605 (2025).
    Autoregressive LLMs interpreted as EBMs: the lookahead energy of a response
    equals -log P(response | prompt) = sum(-log p(token_t)), which is exactly
    the negative log-likelihood computed during generation. High NLL means the
    model is "surprised" by its own output — a signal for likely hallucination
    or constraint violation. Complements spilled energy (Exp 157, arxiv
    2602.18671): spilled energy measures token-level uncertainty between logit
    and output; lookahead energy measures global response likelihood.

**Detailed explanation for engineers:**
    The paper "Autoregressive Language Models as EBMs" (arxiv 2512.15605, 2025)
    shows that the negative log-likelihood of a generated response under the
    model's own distribution defines a well-posed energy function:

        E_lookahead(response | prompt)
            = -log P(response | prompt)
            = sum_t [ -log p(token_t | token_{<t}, prompt) ]

    This is the mean per-token NLL — what language model practitioners know as
    "perplexity in nats." Under this view:

        Low NLL  (≈ 0–1.5 nats)  → model finds the response highly likely
                                   → factual, consistent, low hallucination risk
        High NLL (≈ 3.0+ nats)   → model is "surprised" by its own output
                                   → likely hallucination, incoherence, or
                                     constraint violation

    **Calibration (50 TruthfulQA pairs):**
    Empirical estimates from Exp 169 simulations:
        - Correct, factual responses:  mean NLL ≈ 1.5 nats/token
        - Hallucinated responses:      mean NLL ≈ 3.0+ nats/token
    Default threshold = 2.0 nats/token sits between these clusters.

    **Practical formula:**
    For a generated sequence with T tokens and logits of shape (T, V):

        For each position t:
            log_probs[t]    = log_softmax(logits[t])      # shape (V,)
            x_t             = argmax(logits[t])             # greedy output token
            nll_t           = -log_probs[t, x_t]           # NLL of chosen token

        lookahead_energy = mean(nll_t over T positions)

    We use argmax as a proxy for the actual sampled token because we assume
    greedy/beam-search decoding. For temperature > 0 sampling the caller
    should pass the actual token ids; the current implementation uses argmax
    as a conservative estimate (highest-probability token → lowest NLL →
    most optimistic signal).

    **Relation to spilled energy (Exp 157):**
    - Spilled energy = max(0, NLL_argmax − H(p)) where H(p) is the entropy.
      Measures how much probability was "wasted" across the vocabulary.
    - Lookahead energy = plain NLL of the argmax token.
      Measures absolute response likelihood under the model.
    Both are complementary: combining them (e.g., max) improves AUROC over
    either alone (see Exp 169 results).

    **Integration:**
    - LookaheadEnergyExtractor: implements ConstraintExtractor Protocol.
      Returns empty list when logits=None (graceful degradation — all existing
      callers pass no logits, so there is zero behavior change).
    - LookaheadEnergyConstraint: a ConstraintTerm whose energy is the
      pre-computed lookahead energy scalar. Energy is independent of the Ising
      configuration x; it is a read-only signal from the generation step.
    - AutoExtractor: when logits= is supplied, runs both SpilledEnergyExtractor
      and LookaheadEnergyExtractor, adding both constraints to the results.

    **Target models:** Qwen3.5-0.8B, google/gemma-4-E4B-it (Exp 169).

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

#: Default threshold for LookaheadEnergyConstraint.
#: Calibrated on 50 TruthfulQA pairs:
#:   - Correct (factual) responses:  mean NLL ≈ 1.5 nats/token
#:   - Hallucinated responses:       mean NLL ≈ 3.0+ nats/token
#: Threshold = 2.0 nats/token sits between these clusters.
DEFAULT_LOOKAHEAD_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# LookaheadEnergyConstraint — a ConstraintTerm holding a pre-computed value
# ---------------------------------------------------------------------------


class LookaheadEnergyConstraint(BaseConstraint):
    """A ConstraintTerm encoding the lookahead energy of one LLM generation.

    **Researcher summary:**
        Wraps the scalar lookahead-energy value (mean per-token NLL) computed
        at generation time as a ConstraintTerm. Energy is a constant (the
        pre-computed value) — independent of any Ising configuration x.
        Constraint is satisfied iff lookahead_energy < threshold (model found
        its own output likely).

    **Detailed explanation for engineers:**
        Like SpilledEnergyConstraint, this constraint holds a fixed scalar
        computed from generation logits. The ``energy(x)`` method ignores x
        and returns that scalar. The gradient is always zero — there is nothing
        to optimise in the configuration space.

        ``is_satisfied`` returns True iff lookahead_energy ≤ threshold.
        Low energy = model found its output likely = low hallucination risk.
        High energy = model was "surprised" by its own output = flag for
        downstream KB verification or re-generation.

        **Threshold interpretation:**
            ≤ 1.5 nats/token: highly confident, factual claim likely correct
            1.5–2.0 nats/token: moderate confidence (constraint satisfied)
            > 2.0 nats/token: model uncertain — lookahead constraint violated

    Attributes:
        lookahead_energy_value: Pre-computed mean lookahead energy (≥ 0.0).
        threshold: Satisfaction threshold. Default: DEFAULT_LOOKAHEAD_THRESHOLD.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    def __init__(
        self,
        lookahead_energy_value: float,
        threshold: float = DEFAULT_LOOKAHEAD_THRESHOLD,
    ) -> None:
        """Create a LookaheadEnergyConstraint from a pre-computed energy value.

        Args:
            lookahead_energy_value: The mean lookahead energy (mean per-token
                NLL) from ``LookaheadEnergyExtractor._compute_lookahead_energy()``.
                Must be ≥ 0.0.
            threshold: Energy threshold below which the constraint is
                considered satisfied. Defaults to DEFAULT_LOOKAHEAD_THRESHOLD
                (2.0 nats/token).
        """
        if lookahead_energy_value < 0.0:
            raise ValueError(
                f"lookahead_energy_value must be ≥ 0.0, got {lookahead_energy_value}"
            )
        self._value = lookahead_energy_value
        self._threshold = threshold

    @property
    def name(self) -> str:
        """Human-readable name including the energy value."""
        return f"lookahead_energy({self._value:.4f})"

    @property
    def satisfaction_threshold(self) -> float:
        """Energy threshold: constraint satisfied iff energy ≤ this value."""
        return self._threshold

    def energy(self, x: jax.Array) -> jax.Array:
        """Return the pre-computed lookahead energy (constant; ignores x).

        **Detailed explanation for engineers:**
            The lookahead energy is determined at generation time from logits.
            It does not depend on the Ising configuration x that the Carnot
            repair loop optimises. We return a JAX scalar so this constraint
            composes correctly inside ComposedEnergy.

        Args:
            x: Ising configuration (ignored).

        Returns:
            Scalar JAX float32 equal to lookahead_energy_value.
        """
        # x is intentionally unused — the energy is a constant from generation.
        _ = x
        return jnp.float32(self._value)

    def is_satisfied(self, x: jax.Array) -> bool:
        """Return True iff lookahead_energy ≤ satisfaction_threshold.

        **Detailed explanation for engineers:**
            Overrides BaseConstraint.is_satisfied to avoid calling energy(x),
            which would create an unnecessary JAX scalar. The direct float
            comparison is cleaner and avoids device-to-host transfers when
            the constraint is used as a quick filter.

        Args:
            x: Ignored.

        Returns:
            True if the model found its own output sufficiently likely.
        """
        return self._value <= self._threshold


# ---------------------------------------------------------------------------
# LookaheadEnergyExtractor — ConstraintExtractor Protocol implementation
# ---------------------------------------------------------------------------


class LookaheadEnergyExtractor:
    """Extract a lookahead-energy hallucination signal from generation logits.

    **Researcher summary:**
        Implements ConstraintExtractor Protocol. When logits are available,
        computes mean per-token NLL (= lookahead energy) and wraps it as a
        LookaheadEnergyConstraint (satisfied iff energy < threshold). When
        logits are None, returns an empty list (graceful degradation).

    **Detailed explanation for engineers:**
        Like SpilledEnergyExtractor, this extractor reads the model's internal
        logit distribution from the generation step rather than parsing text
        for pattern-based constraints.

        The key difference from SpilledEnergyExtractor:
        - SpilledEnergy = max(0, NLL_argmax − entropy). Measures dispersion.
        - LookaheadEnergy = NLL_argmax. Measures absolute surprise / likelihood.

        Both use argmax as a proxy for the generated token. The lookahead
        energy is the "simpler" of the two: it is exactly the per-token
        cross-entropy loss that the model would report during training.

        **When logits is None (default):** Returns [] immediately. All existing
        pipeline callers that do not pass logits see zero behavior change.

        **When logits is provided (shape T×V or V):**
            1. Ensure 2-D: if shape (V,), reshape to (1, V).
            2. Compute log_softmax along the vocab axis.
            3. Per position: nll_t = -log_probs[t, argmax(logits[t])].
            4. lookahead_energy = mean(nll_t) over T positions.
            5. Return [ConstraintResult] wrapping a LookaheadEnergyConstraint.

        The returned ConstraintResult uses ``constraint_type="lookahead_energy"``
        and ``domain="factual"`` — it complements the spilled_energy signal
        (Exp 157) as a second factual hallucination indicator.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    def __init__(
        self,
        threshold: float = DEFAULT_LOOKAHEAD_THRESHOLD,
    ) -> None:
        """Create a LookaheadEnergyExtractor.

        Args:
            threshold: Satisfaction threshold passed to LookaheadEnergyConstraint.
                Defaults to DEFAULT_LOOKAHEAD_THRESHOLD (2.0 nats/token).
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
        """Extract a lookahead-energy constraint from generation logits.

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
            - One item: a ConstraintResult with a LookaheadEnergyConstraint
              encoding the model's surprise level during generation.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
        """
        # Domain filter: this extractor only handles factual domain.
        if domain is not None and domain not in self.supported_domains:
            return []

        # Graceful degradation: no logits → no signal.
        if logits is None:
            return []

        lookahead_value = self._compute_lookahead_energy(logits)
        constraint = LookaheadEnergyConstraint(
            lookahead_energy_value=lookahead_value,
            threshold=self._threshold,
        )
        satisfied = constraint.is_satisfied(jnp.zeros(1))
        return [
            ConstraintResult(
                constraint_type="lookahead_energy",
                description=(
                    f"Lookahead energy={lookahead_value:.4f} nats/token "
                    f"({'satisfied' if satisfied else 'violated'}, "
                    f"threshold={self._threshold})"
                ),
                energy_term=constraint,
                metadata={
                    "lookahead_energy": lookahead_value,
                    "threshold": self._threshold,
                    "satisfied": satisfied,
                    "text_snippet": text[:80],
                },
            )
        ]

    def _compute_lookahead_energy(self, logits: jnp.ndarray) -> float:
        """Compute mean per-token lookahead energy (NLL) from a logit array.

        **Detailed explanation for engineers:**
            The lookahead energy is the mean negative log-likelihood of the
            greedy (argmax) output token at each position:

                For each token position t:
                    log_probs[t]    = log_softmax(logits[t])  # shape (V,)
                    x_t             = argmax(logits[t])         # greedy token
                    nll_t           = -log_probs[t, x_t]        # NLL of x_t

                lookahead_energy = mean(nll_t over T positions)

            This is the per-token cross-entropy loss. It equals zero only when
            every token has probability 1.0 (a degenerate peaked distribution).
            For a typical LLM:
                - Confident generation: nll ≈ 0.5–1.5 nats/token
                - Uncertain generation: nll ≈ 2.5–4.5 nats/token (log V ≈ 9.2
                  for V=10000 at maximum entropy)

            We use argmax as a proxy for the actual sampled token. This is
            exact for greedy/beam search decoding. For temperature-sampled
            outputs the true NLL would require the actual token ids (which the
            generate() API can pass separately if needed in a future version).

        Args:
            logits: JAX array of shape (T, V) or (V,).
                T = number of generated tokens, V = vocabulary size.

        Returns:
            Mean lookahead energy over token positions (float, ≥ 0.0).
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

        # NLL of the greedy token at each position.
        # -log p(x_t) ≥ 0 always (since log p ≤ 0).
        # Shape (T,).
        nll = -log_probs[jnp.arange(T), output_tokens]

        return float(jnp.mean(nll))
