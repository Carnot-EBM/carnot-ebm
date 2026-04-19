"""EORM Adaptive Rectifier — best-of-K candidate selection using EORM energy as a PRM.

**Researcher summary:**
    Implements the test-time inference improvement technique from arXiv 2504.01317
    (Adaptive Rectification Sampling).  Instead of taking the first LLM response to
    a question, we generate K candidates with temperature > 0, score each with the
    EORM energy model, and return the minimum-energy candidate as the final answer.

**Why EORM as a Process Reward Model:**
    The energy function IS the process reward — low energy means the configuration
    satisfies constraints.  An EORM-scored response at low energy has passed the
    transformer encoder's learned check for CoT consistency, intermediate-step
    plausibility, and final-answer alignment.  Selecting the minimum-energy candidate
    from K samples is equivalent to best-of-K with EORM as the oracle verifier.

    This requires zero additional training.  EORM already exists (Exp 346, 55M params,
    AUROC ~0.700 on live GSM8K data).  The only new ingredient is generating K responses
    per question instead of one.

**How accuracy improves:**
    For a base model that answers correctly with probability p on any given attempt,
    the probability that at least one of K independent attempts is correct is:
        P(at least one correct | K) = 1 - (1-p)^K

    For p=0.6 and K=3: 1 - (1-0.6)^3 = 1 - 0.064 = 0.936.

    Of course, EORM does not have perfect selection accuracy (AUROC 0.700 ≠ 1.0), so
    the real improvement is between 0 and the theoretical max.  This experiment measures
    where it lands.

Spec: REQ-VERIFY-102, REQ-VERIFY-103,
      SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from carnot.models.eorm import CoTEnergyInput, EORMModel


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RectifierResult:
    """Result of a full EORMAdaptiveRectifier evaluation run.

    **For engineers:**
        Captures both the baseline (single-sample greedy) accuracy and the
        rectified (K-sample EORM-selected) accuracy so callers can compute
        the signed improvement directly.  ``honest_verdict`` is a string
        label designed for the conductor's artifact tracking: it reflects
        whether the rectifier actually helped on this run, not a promise
        about future runs.

    Attributes:
        baseline_accuracy: Fraction of questions answered correctly by the
            greedy (k=1) baseline.
        rectified_accuracy: Fraction of questions answered correctly after
            EORM selects the best of K candidates.
        k_candidates: The K value used for candidate generation.
        signed_improvement: rectified_accuracy - baseline_accuracy.  Positive
            means EORM helped; zero or negative means it did not on this run.
        honest_verdict: 'eorm_rectification_positive' if signed_improvement > 0,
            else 'no_improvement'.

    Spec: REQ-VERIFY-102
    """

    baseline_accuracy: float
    rectified_accuracy: float
    k_candidates: int
    signed_improvement: float
    honest_verdict: str


# ---------------------------------------------------------------------------
# EORMAdaptiveRectifier
# ---------------------------------------------------------------------------


class EORMAdaptiveRectifier:
    """Select the best of K candidate LLM responses using EORM energy as a PRM.

    **For engineers:**
        At evaluation time, for each question:
        1. Call ``inference_fn()`` K times (with temperature > 0) to get K candidates.
        2. Score each candidate using ``eorm_model.energy(CoTEnergyInput(q, candidate))``.
        3. Return the candidate with the minimum energy as the final answer.

        The EORM energy function acts as a process reward model (PRM): a response
        with low energy has satisfied the constraints the EORM learned during training.
        Picking the minimum-energy candidate is therefore equivalent to picking the
        response the EORM judges most likely to be correct.

    Parameters
    ----------
    eorm_model:
        A trained EORMModel instance.  Does not need to be the 55M production model;
        even a small CPU model is valid for benchmarking the selection mechanism.
    k:
        Number of candidates to generate per question.  Default 3.

    Spec: REQ-VERIFY-102, REQ-VERIFY-103
    """

    def __init__(self, eorm_model: EORMModel, k: int = 3) -> None:
        self.eorm_model = eorm_model
        self.k = k

    def select_candidate(self, question: str, candidates: list[str]) -> str:
        """Return the candidate with the lowest EORM energy for this question.

        **For engineers:**
            Each candidate is scored independently via the EORM forward pass.
            The energy function scores the (question, response) pair as a unit
            so context-sensitivity is preserved — the same response text gets a
            different score for different questions.

        Args:
            question: The question that produced the candidates.
            candidates: List of candidate response strings (length >= 1).

        Returns:
            The string from ``candidates`` with the lowest EORM energy.

        Spec: REQ-VERIFY-102, SCENARIO-VERIFY-138
        """
        energies = [
            self.eorm_model.energy(CoTEnergyInput(question_text=question, response_text=c))
            for c in candidates
        ]
        best_idx = energies.index(min(energies))
        return candidates[best_idx]

    def evaluate(
        self,
        questions: list[dict],
        inference_fn: Callable[[str], str],
        *,
        k: int | None = None,
        is_correct_fn: Callable[[str, str], bool] | None = None,
    ) -> RectifierResult:
        """Run greedy baseline and EORM-rectified evaluation over a question set.

        **For engineers:**
            For each question, this method:
            1. Generates a single greedy response (baseline).
            2. Generates K temperature-sampled candidates (including the greedy one
               if the caller's inference_fn produces varied output when called repeatedly).
            3. Scores all K candidates with EORM and selects the minimum-energy one.
            4. Checks both the baseline and the rectified answer against the gold answer.

            The ``is_correct_fn`` defaults to an exact string-match check
            (gold answer appears in response) so the method works without any
            additional dependencies.

        Args:
            questions: List of dicts with at least ``"question"`` (str) and
                ``"answer"`` (str) keys.  The answer is the gold correct answer.
            inference_fn: Callable that takes a question string and returns a
                response string.  Called K+1 times per question (1 baseline + K
                for selection).
            k: Override the instance-level K for this evaluation run.  Defaults
                to ``self.k``.
            is_correct_fn: Optional callable(response, gold_answer) -> bool.
                Defaults to checking whether the gold answer string appears in
                the response.

        Returns:
            RectifierResult with baseline_accuracy, rectified_accuracy,
            k_candidates, signed_improvement, honest_verdict.

        Spec: REQ-VERIFY-102, REQ-VERIFY-103, SCENARIO-VERIFY-139
        """
        effective_k = k if k is not None else self.k

        if is_correct_fn is None:
            def is_correct_fn(response: str, gold: str) -> bool:  # type: ignore[misc]
                return gold.strip() in response

        baseline_correct = 0
        rectified_correct = 0

        for item in questions:
            question = item["question"]
            gold = item["answer"]

            # Greedy baseline: single call
            baseline_response = inference_fn(question)
            if is_correct_fn(baseline_response, gold):
                baseline_correct += 1

            # K-candidate generation for rectification
            candidates = [inference_fn(question) for _ in range(effective_k)]
            best = self.select_candidate(question, candidates)
            if is_correct_fn(best, gold):
                rectified_correct += 1

        n = len(questions)
        baseline_acc = baseline_correct / n if n > 0 else 0.0
        rectified_acc = rectified_correct / n if n > 0 else 0.0
        signed_improvement = rectified_acc - baseline_acc

        verdict = (
            "eorm_rectification_positive" if signed_improvement > 0 else "no_improvement"
        )

        return RectifierResult(
            baseline_accuracy=baseline_acc,
            rectified_accuracy=rectified_acc,
            k_candidates=effective_k,
            signed_improvement=signed_improvement,
            honest_verdict=verdict,
        )
