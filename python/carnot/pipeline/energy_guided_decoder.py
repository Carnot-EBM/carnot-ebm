"""EnergyGuidedDecoder — steer token selection at each generation step using an EBM.

**Why energy-guided decoding instead of post-hoc repair:**
    Post-hoc repair (the existing VerifyRepairPipeline approach) catches constraint
    violations only AFTER the full response is generated.  It must then regenerate the
    entire response from scratch, doubling GPU time and latency.

    Energy-guided decoding intercepts violations BEFORE they happen.  At each
    generation step, rather than committing to the single most-likely next token, we
    sample K candidate continuations and score each one with the EBM energy function.
    We then commit to the candidate with the lowest energy — the one the EBM considers
    most consistent with the learned constraint distribution.

    The energy function provides a differentiable constraint score at every step
    without requiring gradients through the LLM.  This is zero-overhead at training
    time (no fine-tuning needed) and sub-linear overhead at inference time (K energy
    evaluations per step, not a full regeneration).

    Architectural lineage: COLD Decoding (arXiv 2202.11705) introduced energy-guided
    text generation using a language model energy.  Here we replace that language
    model energy with Carnot's IsingEBM, which encodes hard arithmetic and logical
    constraints rather than surface fluency.  The substitution is motivated by
    arXiv 2604.14862, which shows constrained decoding can achieve near-zero overhead
    when the constraint oracle is fast and differentiable.

    This module operates at the word level (not subword tokens) to remain independent
    of any specific tokenizer.  Word-level generation is sufficient for benchmarking
    violation rate reduction on synthetic math problems.

Spec: REQ-VERIFY-113, REQ-VERIFY-114, SCENARIO-VERIFY-149, SCENARIO-VERIFY-150
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class EnergyGuidedConfig:
    """Configuration for EnergyGuidedDecoder.

    **Why these two parameters:**
        ``k_candidates`` controls the exploration breadth at each step.  A higher
        K gives the EBM more choices to steer away from violations, at the cost of
        K energy evaluations per step instead of 1.  K=5 matches the COLD Decoding
        default and the Exp 533 benchmark design.

        ``energy_weight`` is a soft knob between pure-random (0.0) and fully
        energy-steered (1.0) selection.  At 0.0 the decoder degenerates to random
        uniform selection from candidates — the EBM score has no influence.  At 1.0
        (the default) the minimum-energy candidate is always selected.  Values
        between 0 and 1 could implement a temperature-weighted softmax over scores,
        but the current implementation treats the parameter as a binary gate:
        >0 means "use energy ranking", ==0 means "ignore energy".

    Attributes:
        k_candidates: Number of candidate tokens to score at each generation step.
        energy_weight: Weight of the energy score.  0.0 disables energy guidance;
            any positive value enables greedy minimum-energy selection.
    """

    k_candidates: int = 5
    energy_weight: float = 1.0


class EnergyGuidedDecoder:
    """Greedy word-level decoder that selects minimum-energy continuations.

    **How it works at each generation step:**
        1. Receive the accumulated prefix (everything generated so far).
        2. Receive a vocabulary list of candidate next words.
        3. For each candidate, concatenate it onto the prefix and encode the
           resulting string into a fixed-length vector (via a simple hash-based
           encoding that is independent of the vocabulary and preserves ordering).
        4. Score that vector with the EBM energy function.
        5. Return the candidate whose continuation has the lowest energy.

    **Why a simple hash encoding instead of an embedding model:**
        Experiment 533 benchmarks the steering effect of the energy function itself,
        not the quality of the text encoder.  Using a deterministic hash encoding
        ensures the experiment isolates the EBM's contribution: any violation
        reduction must come from the energy surface, not from a learned encoder.
        A real deployment would use the model's hidden states as the encoding.

    Parameters
    ----------
    energy_fn : Callable[[str], float]
        A function that takes a text string and returns a scalar float energy.
        Lower energy = more constraint-satisfying.  This is typically a wrapper
        around IsingModel.energy() that encodes the string into a JAX array.
    config : EnergyGuidedConfig
        Hyperparameters controlling K and the energy weight.

    Spec: REQ-VERIFY-113, SCENARIO-VERIFY-149
    """

    def __init__(
        self,
        energy_fn: Callable[[str], float],
        config: EnergyGuidedConfig | None = None,
    ) -> None:
        self.energy_fn = energy_fn
        self.config = config if config is not None else EnergyGuidedConfig()

    def score_candidates(self, prefix: str, candidates: List[str]) -> List[float]:
        """Score each candidate by computing energy_fn(prefix + candidate).

        **Why prefix + candidate (not just candidate):**
            The energy function must see the full context to score constraint
            satisfaction.  Scoring an isolated word ignores all prior context.
            For example, the word "seven" is only a violation if the prior
            arithmetic context requires a different value.

        Parameters
        ----------
        prefix : str
            Everything generated so far (the accumulated context).
        candidates : List[str]
            Next-token candidates to score.

        Returns
        -------
        List[float]
            Energy score for each candidate.  Same length and order as ``candidates``.

        Spec: REQ-VERIFY-113
        """
        return [float(self.energy_fn(prefix + candidate)) for candidate in candidates]

    def select_next(self, prefix: str, candidates: List[str]) -> str:
        """Return the candidate with the lowest energy continuation.

        When ``energy_weight == 0.0`` the EBM score is ignored entirely and a
        candidate is selected uniformly at random.  This degenerate mode is used
        as the unconstrained baseline in Exp 533.

        Parameters
        ----------
        prefix : str
            Accumulated context string.
        candidates : List[str]
            Next-token candidates to choose from.

        Returns
        -------
        str
            The selected candidate word.

        Spec: REQ-VERIFY-113, SCENARIO-VERIFY-149
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if self.config.energy_weight == 0.0:
            # Degenerate mode: energy has no influence — uniform random selection.
            # This is the unconstrained baseline used in the Exp 533 benchmark.
            return random.choice(candidates)
        scores = self.score_candidates(prefix, candidates)
        best_idx = scores.index(min(scores))
        return candidates[best_idx]

    def generate(
        self,
        prompt: str,
        vocab: List[str],
        max_steps: int = 20,
    ) -> str:
        """Greedily generate a sequence of ``max_steps`` words from ``vocab``.

        At each step, calls ``select_next(current_prefix, vocab)`` to pick the
        next word, appends it to the running prefix with a space separator, and
        continues until ``max_steps`` words have been appended.

        **Why word-level instead of subword tokens:**
            Word-level generation keeps the experiment independent of any specific
            tokenizer.  The violation detection rules (VPRMArithmeticVerifier) also
            operate on human-readable text, so word-level generation produces output
            that the verifier can parse without detokenization.

        Parameters
        ----------
        prompt : str
            The initial prompt / context.
        vocab : List[str]
            Vocabulary of valid next words to sample from.
        max_steps : int
            Number of generation steps (words to append).

        Returns
        -------
        str
            The generated text: ``prompt`` followed by ``max_steps`` words.

        Spec: REQ-VERIFY-113, SCENARIO-VERIFY-150
        """
        current = prompt
        for _ in range(max_steps):
            next_word = self.select_next(current, vocab)
            current = current + " " + next_word
        return current
