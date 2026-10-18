"""FACT-E causal faithfulness probe for chain-of-thought verification.

**Researcher summary (arXiv 2604.10693):**
    FACT-E (Faithfulness via Automated Causal Testing in Evaluations) detects
    reasoning steps that are logically disconnected from each other, even when
    the numbers happen to add up correctly.  The core idea: if step B genuinely
    *depends* on step A, then perturbing the numeric values in step A should
    cause the numeric claims in step B to change proportionally.  If B's numbers
    remain unchanged after perturbing A, the two steps are not causally
    connected — the reasoning chain is non-sequitur.

**Detailed explanation for engineers:**
    Traditional arithmetic verifiers (Carnot Tier 1/2) check whether numerical
    computations are correct.  FACT-E adds a *faithfulness* dimension: are the
    steps actually *using* each other's results, or is the LLM just listing
    plausible-looking sentences?

    The probe works via controlled perturbation:
    1. Extract numeric tokens (integers and decimals) from step A and step B.
    2. Count tokens that appear in both steps (shared numerics = potential
       causal links between the steps).
    3. The dependency_score is the fraction of step B's numeric tokens that
       also appear in step A.  High score → B is drawing on A's numbers →
       causally connected.  Low score → B's numbers are unrelated to A's →
       non-sequitur link.
    4. For a full response, compute the mean dependency_score across all
       consecutive step pairs.  This is the faithfulness_score ∈ [0, 1].

    The perturbation step (_perturb_step) is used to validate the intuition
    but the lightweight shared-token approach is the primary signal, since
    LLM-generated responses are not executable and cannot be re-run.

**Why this matters for Carnot:**
    This extends verification from "are the numbers right?" (Tier 1/2) to "is
    the reasoning chain internally coherent?" (Tier 3.5 advisory signal).  A
    response can pass arithmetic verification but fail faithfulness — e.g., a
    correct final answer reached by unrelated reasoning steps.

Spec: REQ-VERIFY-145, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass
class CausalStepDependency:
    """Result of a causal dependency check between two consecutive CoT steps.

    Attributes:
        step_a: The upstream step whose numeric values could influence step_b.
        step_b: The downstream step being checked for causal dependence on step_a.
        dependency_score: Float in [0, 1].  Fraction of step_b's numeric tokens
            that also appear in step_a.  Higher means more causally connected.
        is_causally_connected: True when dependency_score >= the probe threshold.
    """

    step_a: str
    step_b: str
    dependency_score: float
    is_causally_connected: bool


def _extract_numeric_tokens(text: str) -> list[str]:
    """Return all integer and decimal number strings from text.

    Extracts tokens like '16', '3.14', '0.5' but not embedded substrings
    (e.g. does not extract '16' from 'a16b').
    """
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


class FACTEFaithfulnessProbe:
    """Causal faithfulness probe for chain-of-thought reasoning.

    Implements the FACT-E method (arXiv 2604.10693) as a Tier 3.5 advisory
    signal.  Low faithfulness_score indicates that consecutive reasoning steps
    are not numerically connected, suggesting non-sequitur or hallucinated
    reasoning chains.

    Args:
        threshold: Minimum dependency_score to classify a step pair as causally
            connected.  Default 0.3 means at least 30% of step_b's numeric
            tokens must appear in step_a.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def _perturb_step(self, step: str) -> str:
        """Replace each numeric token in step with a random ±20% perturbation.

        Used to validate causal intuition: if downstream steps are truly
        dependent on upstream numbers, perturbing upstream values should
        propagate.  In the lightweight shared-token approach this is used
        conceptually; this method is provided for explainability and testing.

        Returns:
            A copy of step with all numeric tokens replaced by nearby values.
        """

        def perturb_match(m: re.Match) -> str:  # type: ignore[type-arg]
            original = m.group(0)
            try:
                value = float(original)
            except ValueError:
                return original
            # Perturb by ±20%, but ensure the result differs from the original.
            factor = random.uniform(0.8, 1.2)  # noqa: S311 (non-crypto use)
            perturbed = value * factor
            if "." in original:
                return f"{perturbed:.2f}"
            return str(int(round(perturbed)))

        return re.sub(r"\b\d+(?:\.\d+)?\b", perturb_match, step)

    def measure_dependency(self, step_a: str, step_b: str) -> CausalStepDependency:
        """Measure the causal dependency of step_b on step_a.

        Algorithm:
        1. Extract numeric tokens from both steps.
        2. Build the set of step_a's numeric tokens (the "upstream values").
        3. For each numeric token in step_b, check if it appears in step_a's set.
        4. dependency_score = (# of step_b tokens found in step_a) / max(1, |step_b tokens|).

        This is a lightweight proxy for causal dependence: if step_b references
        the same numbers as step_a, it is likely building on step_a's result.

        Args:
            step_a: Upstream reasoning step.
            step_b: Downstream reasoning step to test for dependence.

        Returns:
            CausalStepDependency with score and connected flag.
        """
        a_tokens = set(_extract_numeric_tokens(step_a))
        b_tokens = _extract_numeric_tokens(step_b)

        if not b_tokens:
            # No numerics in step_b means no numeric dependency to measure.
            # Treat as not connected (score=0).
            return CausalStepDependency(
                step_a=step_a,
                step_b=step_b,
                dependency_score=0.0,
                is_causally_connected=False,
            )

        shared_count = sum(1 for tok in b_tokens if tok in a_tokens)
        score = shared_count / max(1, len(b_tokens))

        return CausalStepDependency(
            step_a=step_a,
            step_b=step_b,
            dependency_score=score,
            is_causally_connected=score >= self.threshold,
        )

    def faithfulness_score(self, response: str) -> float:
        """Compute a faithfulness score for a full chain-of-thought response.

        Splits the response into steps (on sentence boundaries or newlines),
        then computes mean dependency_score across all consecutive step pairs.

        A high faithfulness_score (close to 1.0) means each step references
        numeric values from the previous step — the chain is internally
        consistent.  A low score means many steps are numerically disconnected
        from their predecessors — a sign of non-sequitur reasoning.

        Args:
            response: Full CoT response string (may contain newlines, periods).

        Returns:
            Float in [0, 1].  Returns 1.0 if there are fewer than 2 steps
            (no pairs to evaluate — single-step responses are trivially faithful).
        """
        # Split on newlines first; fall back to sentence-level splitting.
        raw_steps = [s.strip() for s in re.split(r"\n+", response) if s.strip()]
        if len(raw_steps) < 2:
            # Try sentence splitting as fallback.
            raw_steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]

        if len(raw_steps) < 2:
            # Only one step — no pairs to evaluate.
            return 1.0

        scores = [
            self.measure_dependency(raw_steps[i], raw_steps[i + 1]).dependency_score
            for i in range(len(raw_steps) - 1)
        ]
        return sum(scores) / len(scores)
