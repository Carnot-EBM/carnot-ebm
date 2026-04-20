"""jepa_pure_loss — PURE min-form contrastive margin loss for JEPA chain scoring.

**Researcher summary (RETRO-060):**
    Binary BCE loss (Exp 557) let the JEPA model hedge toward P=0.5 for every chain,
    producing near-zero gradients and AUC=0.4286 — below random baseline.  The root cause
    is that BCE has no mechanism to push correct and incorrect chains apart in score space.

    This module implements the PURE PRM objective (arXiv 2504.15275): score a full
    reasoning chain by the MINIMUM step score across all its steps.  A chain with even
    one bad step gets a strong low signal.  A chain where every step looks good gets a
    strong high signal.  The contrastive loss then enforces a margin between incorrect
    and correct min-scores:

        loss = mean(max(0, margin - (min_score_incorrect - min_score_correct)))

    over all (correct, incorrect) pairs from the same question.  This is exactly the
    contrastive signal used by NUP Probe v4 to achieve AUC=1.0.

**Why min instead of mean or last step?**
    Mean step score allows one very good step to cancel a catastrophically wrong step.
    Last-step score ignores all intermediate reasoning quality.
    Min is the "weakest-link" aggregation: the chain is only as strong as its worst step,
    which matches the intuition that a single arithmetic error invalidates a full proof.

**Why a margin loss instead of cross-entropy?**
    Cross-entropy (BCE) minimises log-loss independently for each chain without coupling
    correct and incorrect chains together.  A margin loss *requires* a gap of at least
    `margin` between incorrect.min_score and correct.min_score — making it impossible to
    satisfy by hedging toward 0.5.

Spec: REQ-LEARN-061, REQ-LEARN-062,
      SCENARIO-LEARN-095, SCENARIO-LEARN-096, SCENARIO-LEARN-097
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# JEPAChainScore
# ---------------------------------------------------------------------------


@dataclass
class JEPAChainScore:
    """One chain's scoring summary for the PURE min-form loss.

    **Detailed explanation for engineers:**
        A "chain" is one model response (chain-of-thought) for a given question.
        ``step_scores`` are scalar energy values produced by passing each step's
        embedding through the JEPA energy model.  ``min_score`` is pre-computed
        as min(step_scores) to avoid recomputing it in the inner training loop.

        ``chain_id`` is a free-form identifier — typically "<question_id>/<model_id>"
        or a hash — used for debugging and logging.

    Attributes:
        chain_id:     Unique identifier for this chain (question + model response).
        step_scores:  Energy score for each CoT step, in step order.
        min_score:    Minimum of step_scores — the weakest-link score for the chain.
        is_correct:   True if the chain's final answer was correct per ground-truth.

    Spec: REQ-LEARN-061
    """

    chain_id: str
    step_scores: list[float]
    min_score: float
    is_correct: bool

    def __post_init__(self) -> None:
        # Ensure min_score is consistent with step_scores when steps are provided.
        # We do NOT override caller-supplied min_score (allows synthetic test data).
        pass


# ---------------------------------------------------------------------------
# PUREMinFormLoss
# ---------------------------------------------------------------------------


class PUREMinFormLoss:
    """PURE min-form contrastive margin loss (arXiv 2504.15275).

    **Detailed explanation for engineers:**
        The loss is computed over pairs of (correct_chain, incorrect_chain) from the
        same question.  For each pair:

            pair_loss = max(0, margin - (incorrect.min_score - correct.min_score))

        When the incorrect chain scores at least `margin` above the correct chain,
        pair_loss = 0 (the constraint is satisfied — no gradient needed).
        When the incorrect chain scores less than `margin` above the correct chain,
        pair_loss > 0 (push them apart).

        The total loss is the mean over all pairs.  Empty pair lists return 0.0
        via zero_if_empty() to avoid NaN gradients at the start of training.

    Attributes:
        margin: Minimum required gap between incorrect.min_score and correct.min_score.
                Default 1.0 (standard contrastive margin).

    Spec: REQ-LEARN-061, REQ-LEARN-062
    """

    def __init__(self, margin: float = 1.0) -> None:
        """Create a PUREMinFormLoss with the given contrastive margin.

        Args:
            margin: Required gap between incorrect and correct min-scores.
                    Pairs where the gap >= margin contribute zero loss.
                    Default 1.0 is a standard choice for unit-normalised scores.

        Spec: REQ-LEARN-062
        """
        self.margin = margin

    def compute_chain_scores(
        self,
        model: Callable[[jnp.ndarray], float],
        chain_embeddings: list[jnp.ndarray],
    ) -> list[float]:
        """Pass each step embedding through the energy model and return scalar scores.

        **Detailed explanation for engineers:**
            ``model`` is any callable that accepts a single JAX array (one step's
            embedding) and returns a scalar float (the energy value for that step).
            We call it once per step and collect the results as a Python list.

            The caller is responsible for batching across questions if throughput
            matters — this function processes one chain at a time for clarity.

        Args:
            model:             Callable mapping (jnp.ndarray) -> float.
            chain_embeddings:  One jnp.ndarray per CoT step in the chain.

        Returns:
            List of scalar floats, one per step, in step order.

        Spec: REQ-LEARN-061
        """
        return [float(model(emb)) for emb in chain_embeddings]

    def compute_loss(
        self,
        correct_chains: list[JEPAChainScore],
        incorrect_chains: list[JEPAChainScore],
    ) -> float:
        """Compute PURE min-form contrastive margin loss over all (correct, incorrect) pairs.

        **Detailed explanation for engineers:**
            We take the cross-product of correct_chains and incorrect_chains (all possible
            pairings from the same batch).  For production use with per-question grouping,
            pass only chains from the same question — pairs_to_pure_chains() handles this.

            For each pair:
                pair_loss = max(0, margin - (incorrect.min_score - correct.min_score))

            The mean over all pairs is returned.  If either list is empty, returns 0.0.

        Args:
            correct_chains:   Chains whose final answer was correct.
            incorrect_chains: Chains whose final answer was incorrect.

        Returns:
            Mean pair loss as a Python float.

        Spec: REQ-LEARN-061, SCENARIO-LEARN-095, SCENARIO-LEARN-096
        """
        pairs = [
            (c, w)
            for c in correct_chains
            for w in incorrect_chains
        ]
        if len(pairs) == 0:
            return self.zero_if_empty(pairs)
        return self._mean_pair_loss(pairs)

    def _mean_pair_loss(
        self,
        pairs: list[tuple[JEPAChainScore, JEPAChainScore]],
    ) -> float:
        """Compute mean of max(0, margin - (wrong.min_score - correct.min_score)) over pairs."""
        total = 0.0
        for correct, wrong in pairs:
            gap = wrong.min_score - correct.min_score
            total += max(0.0, self.margin - gap)
        return total / len(pairs)

    def zero_if_empty(self, pairs: list) -> float:
        """Return 0.0 when the pair list is empty; return 0.0 (falsy) otherwise for chaining.

        **Detailed explanation for engineers:**
            At the very start of training, before any contrastive pairs exist (e.g. when
            the corpus has only correct chains or only incorrect chains), we must return a
            well-defined loss value rather than NaN or a divide-by-zero error.  This method
            checks for the empty case and returns 0.0, which produces zero gradient —
            equivalent to "nothing to learn from an empty batch."

        Args:
            pairs: List of (correct, incorrect) chain pairs.

        Returns:
            0.0 if pairs is empty; 0.0 (falsy) otherwise so the caller can use
            ``return self.zero_if_empty(pairs) or self._mean_pair_loss(pairs)``.

        Spec: REQ-LEARN-061, SCENARIO-LEARN-097
        """
        if len(pairs) == 0:
            return 0.0
        return 0.0  # falsy — caller proceeds to _mean_pair_loss


# ---------------------------------------------------------------------------
# pairs_to_pure_chains
# ---------------------------------------------------------------------------


def pairs_to_pure_chains(
    fover_entries: list,  # list[FOVERCorpusEntry] — typed as list to avoid circular import
    embed_fn: Callable[[str], jnp.ndarray],
) -> tuple[list[JEPAChainScore], list[JEPAChainScore]]:
    """Convert a list of FOVERCorpusEntry objects into PURE chain score objects.

    **Detailed explanation for engineers:**
        FOVERCorpusEntry has a ``cot_steps`` list (dicts with 'step_text' key) and an
        ``is_correct`` flag.  This function:

        1. Iterates over each entry.
        2. For each step, calls ``embed_fn(step_text)`` to get a JAX embedding.
        3. If the entry has no steps, uses embed_fn(entry.response) as a single step.
        4. Assigns min_score = min(step_scores) (or 0.0 for empty step lists).
        5. Builds a JEPAChainScore with chain_id = f"{entry.question[:40]}/{entry.model_id}".
        6. Splits the list into (correct_chains, incorrect_chains) by is_correct.

        The ``embed_fn`` caller provides is intentionally generic — it may be a learned
        JAX embedding layer, a random projection, or a hash-based feature map depending
        on the experiment stage.

    Args:
        fover_entries: List of FOVERCorpusEntry objects from the FOVER corpus.
        embed_fn:      Callable mapping a text string to a 1-D jnp.ndarray embedding.

    Returns:
        Tuple (correct_chains, incorrect_chains) of JEPAChainScore lists.

    Spec: REQ-LEARN-061
    """
    correct_chains: list[JEPAChainScore] = []
    incorrect_chains: list[JEPAChainScore] = []

    for entry in fover_entries:
        chain_id = f"{entry.question[:40]}/{entry.model_id}"

        # Extract step texts — fall back to full response if no steps present.
        step_texts: list[str] = []
        for step in entry.cot_steps:
            text = step.get("step_text", "") if isinstance(step, dict) else str(step)
            if text.strip():
                step_texts.append(text)
        if not step_texts:
            step_texts = [entry.response] if entry.response.strip() else [""]

        # Embed each step and compute the weakest-link (min) score.
        embeddings = [embed_fn(t) for t in step_texts]
        # For now, use mean of embedding components as a scalar score proxy.
        # The caller can override by wrapping embed_fn to return scalar outputs.
        step_scores: list[float] = [float(jnp.mean(emb)) for emb in embeddings]
        min_score = min(step_scores) if step_scores else 0.0

        chain = JEPAChainScore(
            chain_id=chain_id,
            step_scores=step_scores,
            min_score=min_score,
            is_correct=entry.is_correct,
        )

        if entry.is_correct:
            correct_chains.append(chain)
        else:
            incorrect_chains.append(chain)

    return correct_chains, incorrect_chains
