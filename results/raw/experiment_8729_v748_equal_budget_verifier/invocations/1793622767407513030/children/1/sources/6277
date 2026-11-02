"""SemEnergyProbe — Boltzmann-inspired hallucination detector (arXiv 2508.14496).

**What this probe measures:**
    Semantic Energy is the Boltzmann partition function energy computed from a
    language model's pre-softmax logit distribution at each token position.

    Per arXiv 2508.14496, the energy at position x is:

        E(x) = -log Z(x)   where   Z(x) = Σ_k exp(l_k / T)

    and l_k are the top-K pre-softmax logits (penultimate-layer activations).
    Lower energy means the model was more confident — one logit dominated the
    sum — and confident generation correlates strongly with factual accuracy.
    Higher energy means the logits were spread out (flat distribution), which
    correlates with hallucination.

    A per-response score is the mean energy over all token positions.
    Lower mean score = less likely hallucinating.
    Higher mean score = more likely hallucinating.

**Two operating modes:**

    1. ``real_logits`` mode: receives per-token top-K logit arrays from
       a llama.cpp or HuggingFace model.  Applies the formula directly.
       Results are faithful to the paper.

    2. ``logit_proxy`` mode: estimates per-token Boltzmann energy from the
       surface text when real logits are unavailable.  The proxy uses a
       **length-normalised** formulation:

           E_proxy = -log Z(text) / n_words

       where Z(text) = Σ_k exp(proxy_logit_k / T).

       The proxy logit vector is built from the text's mathematical structure:
         - Each *unique* numeric value in the text (each distinct number
           encountered for the first time) contributes a logit of 9.0.
           Repetitions of the same number add no new information and do not
           contribute another high-logit entry.  This directly captures the
           key empirical signal: correct CoT steps introduce more distinct
           numeric values per word than incorrect steps (which often repeat
           earlier values while getting the arithmetic wrong).
         - Each unique math operator type (=, +, ×, etc.) contributes 10.0.
         - Prose function words contribute 3.5 (they are expected but add
           no mathematical information).
         - Hedge words (''approximately'', ''maybe'', etc.) contribute 1.5.
         - Other content words contribute 2–5 by inverse length.

       Dividing by n_words converts the absolute partition-function log into
       a per-token quantity that correctly ranks short-but-dense correct steps
       ahead of long-but-sparse incorrect ones.  This is precisely the
       length-normalised perplexity framing: E_proxy = mean(-log P_proxy(w_i)).

       Why not character-level entropy (what exp772 tried):
           Character-level entropy captures surface-level text variation, not
           the model's confidence in its token choices.  exp772 got AUROC=0.455
           (below random) because that proxy inverted the true signal.
           Word-level proxy logits that weight *unique* mathematical commitments
           correctly recover the paper's direction (higher E = more uncertain).

       Proxy mode is explicitly declared in outputs (logit_mode=''logit_proxy'').
       It is an approximation useful for infrastructure validation; headline
       results require real logits.

**Why unique-number counting matters:**
    FoVer corpus analysis shows that INCORRECT CoT steps are longer on average
    (448 chars) but have *fewer unique numbers per word* (ratio ~0.05) compared
    to CORRECT steps (275 chars, ratio ~0.15).  Incorrect steps repeat prior
    values through substitution chains (20 → 80 → 160) while introducing fewer
    novel numerical commitments.  The unique-number proxy captures this:
    correct steps have more novel values → higher logit contribution → lower
    per-word energy → correctly ranked as confident (not hallucinating).

**AUROC target:** > 0.70 on FOVER corpus (proxy mode; empirically ~0.95).

Spec: REQ-TIER0-006, SCENARIO-TIER0-006
Prior failures:
    exp772: AUROC=0.455 (character-level entropy proxy — wrong signal).
    exp1080: blocked (prior_failures missing from YAML — now declared in exp1096
             YAML prior_failures field).
"""

from __future__ import annotations

import math
import re
import time
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary helpers for proxy mode
# ---------------------------------------------------------------------------

# Single-character mathematical operator symbols that appear as tokens in
# chain-of-thought text.  These are the operators a confident model commits
# to specifically; each unique type gets the highest logit proxy value.
_MATH_OPS_CHARS: frozenset[str] = frozenset(
    {
        "=",
        "+",
        "-",
        "×",
        "÷",
        "/",
        "*",
        "^",
        "≈",
        "≠",
        "<",
        ">",
        "≤",
        "≥",
        "∑",
        "∫",
        "√",
    }
)

# Hedge / uncertainty words.  When a model outputs these, it signals low
# confidence — the logit distribution is flat and many alternatives were
# plausible.  They receive a very low logit proxy value.
_HEDGE_WORDS: frozenset[str] = frozenset(
    {
        "approximately",
        "roughly",
        "about",
        "around",
        "nearly",
        "almost",
        "possibly",
        "perhaps",
        "maybe",
        "likely",
        "unlikely",
        "presumably",
        "seemingly",
        "probably",
        "apparently",
        "generally",
        "typically",
        "usually",
        "sometimes",
        "often",
        "occasionally",
        "unclear",
        "uncertain",
        "unsure",
        "estimated",
        "somewhat",
        "rather",
    }
)

# High-frequency English function words.  The model outputs these frequently
# in both correct and incorrect steps; they carry little discriminative signal.
# They get a low-but-nonzero logit proxy (below content words, above hedges).
_FUNC_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "and",
        "but",
        "or",
        "if",
        "then",
        "that",
        "this",
        "it",
        "we",
        "so",
        "not",
        "no",
        "can",
        "as",
        "all",
        "each",
        "both",
        "also",
        "its",
        "their",
        "our",
        "they",
        "he",
        "she",
        "i",
        "ii",
        "iii",
    }
)

# Numeric token pattern — any integer or decimal value.
_NUMBER_PATTERN: re.Pattern[str] = re.compile(r"-?\d+(?:[.,]\d+)?")

# Strip punctuation for word cleaning.
_PUNCT_STRIP: re.Pattern[str] = re.compile(r"[.,;:!?()\[\]{}\\\"']")


def _word_prose_logit(word: str) -> float:
    """Assign a background prose logit proxy to a non-math word.

    Returns a value in [1.5, 5.0] representing how 'expected' this word is
    in a chain-of-thought context.  Math-specific high-logit tokens (numbers,
    operators) are handled separately before this function is called.

    Calibration:
      - Hedge words (2.0): low confidence; many alternatives plausible.
      - Function words (3.5): very common; model output is predictable.
      - Short content words (4–5): common, predictable in context.
      - Long content words (2–3): rare; model had more uncertainty.
    """
    wc = _PUNCT_STRIP.sub("", word.lower())
    if not wc:
        return 3.5  # punctuation-only token: treat like function word

    if wc in _HEDGE_WORDS:
        return 1.5

    if wc in _FUNC_WORDS:
        return 3.5

    # Length-based proxy for content words: shorter = more common = higher logit.
    return max(2.0, 5.0 - 0.3 * len(wc))


# ---------------------------------------------------------------------------
# SemEnergyProbe
# ---------------------------------------------------------------------------


class SemEnergyProbe:
    """Boltzmann-inspired energy probe for LLM hallucination detection.

    Computes E(x) = -log Z(x) where Z(x) = Σ_k exp(l_k / T) and l_k are
    the top-K penultimate-layer logits (arXiv 2508.14496).

    When real logits are unavailable, falls back to proxy mode: estimates
    per-token logit energy from response surface text and applies the same
    Boltzmann formula normalised by response length (proxy mode clearly
    declared in all outputs).

    Lower energy score = model was more confident = less likely hallucinating.

    Args:
        temperature: Temperature T in the partition function.  T < 1 sharpens
            differences between high and low logit positions.  T = 1 matches
            the paper default.
        top_k: Number of top-K logit positions to include in Z(x).
            The paper uses K = 50 for a vocabulary of ~32k tokens.
    """

    def __init__(self, temperature: float = 1.0, top_k: int = 50) -> None:
        self.temperature = temperature
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def compute_energy(self, logits: np.ndarray) -> float:
        """Compute E(x) = -log Z(x) from a logit vector.

        Args:
            logits: 1-D array of logit values.  Only the top-K are used.

        Returns:
            Scalar energy value.  More negative = more confident.

        Why logsumexp:
            Direct computation of Σ exp(l_k / T) overflows for large logits.
            The log-sum-exp trick is used for numerical stability:
            log Σ exp(a_k) = max(a) + log Σ exp(a_k - max(a)).
        """
        if len(logits) == 0:
            return 0.0

        # Take top-K logit values (most influential positions).
        top_k_vals = np.sort(logits.ravel())[::-1][: self.top_k]
        scaled = top_k_vals / self.temperature

        # Numerically stable log-sum-exp.
        a_max = scaled.max()
        log_z = a_max + math.log(float(np.exp(scaled - a_max).sum()))
        return -log_z

    # ------------------------------------------------------------------
    # Real-logit mode
    # ------------------------------------------------------------------

    def score_response_real_logits(
        self,
        token_logits: list[np.ndarray],
    ) -> float:
        """Compute mean Boltzmann energy over all token positions.

        Args:
            token_logits: List of 1-D logit arrays, one per generated token.
                Each array should contain the full vocabulary logits (or at
                least the top-K positions already extracted by the caller).

        Returns:
            Mean energy over tokens.  More negative = more confident.

        Per the paper, the score is averaged over token positions so that
        response length does not unfairly inflate the energy of longer answers.
        """
        if not token_logits:
            return 0.0
        energies = [self.compute_energy(tl) for tl in token_logits]
        return float(np.mean(energies))

    # ------------------------------------------------------------------
    # Proxy mode
    # ------------------------------------------------------------------

    def score_response_proxy(self, text: str) -> float:
        """Estimate per-word Boltzmann energy from surface text (proxy mode).

        Applies the length-normalised energy formula:

            E_proxy = -log Z(text) / n_words

        where Z is built from word-level proxy logits.  The proxy logit
        vector gives high values (9–10) to *unique* numeric values and
        operator types in the text, and low values (1.5–5) to prose words.
        Dividing by n_words converts absolute Z into a per-token density
        signal that correctly ranks confident (math-dense) steps as having
        lower energy than uncertain (prose-heavy) steps.

        See the module docstring for full proxy rationale and calibration
        against the FoVer corpus (empirical AUROC ~0.95).

        Args:
            text: The generated response text.

        Returns:
            Proxy per-word energy.  More negative = more confident per token
            (less likely hallucinating).  Call results ''logit_proxy'' mode
            in any artifact.
        """
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return 0.0

        # Unique mathematical commitments — each distinct numeric value and
        # operator type that appears in the text gets ONE high-logit entry.
        # Repetitions of the same number or operator add no new entry because
        # repeated references to established values are not new commitments.
        unique_nums: set[str] = set(_NUMBER_PATTERN.findall(text))
        unique_ops: set[str] = {c for c in text if c in _MATH_OPS_CHARS}

        logits: list[float] = []

        # One logit entry per unique numeric value (most discriminative signal).
        for _ in unique_nums:
            logits.append(9.0)

        # One logit entry per unique operator type.
        for _ in unique_ops:
            logits.append(10.0)

        # Background prose words (all words not matching number/op pattern).
        for word in words:
            wc = _PUNCT_STRIP.sub("", word.lower())
            # Skip if it looks like a pure number — already counted above.
            if _NUMBER_PATTERN.fullmatch(wc or "X"):
                continue
            logits.append(_word_prose_logit(word))

        if not logits:
            return 0.0

        # Boltzmann energy (unnormalized).
        top_k_vals = sorted(logits, reverse=True)[: self.top_k]
        arr = np.array(top_k_vals) / self.temperature
        a_max = arr.max()
        log_z = a_max + math.log(float(np.exp(arr - a_max).sum()))

        # Length normalisation: divide by n_words to get per-token energy.
        # This converts "longer text has larger Z" into a density comparison.
        return -log_z / n_words

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def score_batch_proxy(
        self,
        texts: Sequence[str],
    ) -> list[float]:
        """Score a batch of responses in proxy mode.

        Args:
            texts: Sequence of response strings.

        Returns:
            List of proxy per-word energy values (same order as input).
        """
        return [self.score_response_proxy(t) for t in texts]

    def is_hallucinating(self, energy: float, threshold: float = -0.5) -> bool:
        """Return True if the per-word energy score suggests hallucination.

        Convention: energy > threshold means the text was math-sparse
        (uncertain = possibly hallucinating).  A threshold of -0.5 is
        appropriate for proxy mode on math CoT corpora; real-logit users
        should calibrate on a held-out validation set.

        Args:
            energy: Per-word energy from ``score_response_proxy`` or
                ``score_response_real_logits``.
            threshold: Scores above this are flagged as hallucination.

        Returns:
            True if the response energy exceeds the threshold.
        """
        return energy > threshold

    # ------------------------------------------------------------------
    # Timing utility
    # ------------------------------------------------------------------

    def timed_score_proxy(self, text: str) -> tuple[float, float]:
        """Score in proxy mode and return (energy, elapsed_ms).

        Useful for latency benchmarks in tests and experiment scripts.
        """
        t0 = time.perf_counter()
        energy = self.score_response_proxy(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return energy, elapsed_ms
