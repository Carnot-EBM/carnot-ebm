"""Composite energy scorer: combine logprob confidence with structural tests.

**Researcher summary:**
    Experiment 14 proved that composite scoring (logprob + structural test
    failures) is never worse than either signal alone. For code tasks, greedy
    selection achieves 0% accuracy — composite rejection sampling pushes it
    to 30%. For QA tasks, logprobs dominate. The composite handles both
    domains because it weights the two signals independently.

**Detailed explanation for engineers:**
    The composite energy for a candidate response is:

        score = -logprob_weight * mean_logprob
              + structural_weight * test_failure_penalty * n_failures

    Lower score is better (energy minimization convention). A candidate with
    zero test failures and high logprob (large negative score) beats one with
    failures. Among equal-failure candidates, the one with the highest
    mean logprob wins because -logprob_weight * (larger negative number) is
    more negative.

    ``select_best()`` takes a list of candidates — each described by
    (code_text, mean_logprob, n_failures) — scores them all, and returns
    the index and score of the best (lowest energy) candidate.

    This module is designed to be wired into ``iterative_refine_code`` so that
    at each refinement iteration, N candidates are generated and the composite
    scorer picks the best one before deciding whether to continue refining.

Spec: REQ-INFER-013
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompositeEnergyConfig:
    """Configuration for composite energy scoring.

    **Researcher summary:**
        Three knobs: how much to trust the model's own confidence (logprob_weight),
        how much to penalize structural failures (structural_weight), and the
        per-failure penalty magnitude (test_failure_penalty).

    **Detailed explanation for engineers:**
        Default weights are tuned from Experiment 14 results:
        - logprob_weight=1.0: mean logprob is typically in [-5, 0], so this
          contributes 0-5 to the score.
        - structural_weight=1.0 and test_failure_penalty=10.0: each test
          failure adds 10 to the score, dominating logprob differences.
          This ensures that a candidate passing all tests always beats one
          with failures, unless logprob differences are extreme.

    Spec: REQ-INFER-013
    """

    logprob_weight: float = 1.0
    structural_weight: float = 1.0
    test_failure_penalty: float = 10.0


class CompositeEnergyScorer:
    """Score candidates by combining logprob confidence with test results.

    **Researcher summary:**
        Implements the composite energy function from Experiment 14.
        Lower score = better candidate. The scorer is stateless — all
        state lives in the config.

    **Detailed explanation for engineers:**
        Instantiate with a ``CompositeEnergyConfig`` (or use defaults), then
        call ``score_candidate()`` for individual scores or ``select_best()``
        to pick the winner from a batch.

    Spec: REQ-INFER-013
    """

    def __init__(self, config: CompositeEnergyConfig | None = None) -> None:
        self.config = config or CompositeEnergyConfig()

    def score_candidate(
        self,
        code: str,
        mean_logprob: float,
        n_failures: int,
    ) -> float:
        """Compute composite energy for a single candidate.

        **Researcher summary:**
            score = -logprob_weight * mean_logprob
                  + structural_weight * test_failure_penalty * n_failures

            Lower is better. The ``code`` argument is accepted for interface
            consistency (future scorers may inspect the code itself) but is
            not used in the current formula.

        **Detailed explanation for engineers:**
            mean_logprob is typically negative (log-probabilities), so
            ``-logprob_weight * mean_logprob`` is positive for low-confidence
            responses and near-zero for high-confidence ones. Adding the
            failure penalty on top ensures structural correctness dominates.

        Args:
            code: The candidate code text (reserved for future use).
            mean_logprob: Mean per-token log-probability from the LLM.
            n_failures: Number of test cases that failed.

        Returns:
            Composite energy score (lower is better).

        Spec: REQ-INFER-013
        """
        cfg = self.config
        return (
            -cfg.logprob_weight * mean_logprob
            + cfg.structural_weight * cfg.test_failure_penalty * n_failures
        )

    def select_best(
        self,
        candidates: list[tuple[str, float, int]],
    ) -> tuple[int, float]:
        """Select the best candidate from a list.

        **Researcher summary:**
            Scores all candidates and returns the index and score of the one
            with the lowest composite energy.

        **Detailed explanation for engineers:**
            Each candidate is a tuple of (code_text, mean_logprob, n_failures).
            We score each one and return (best_index, best_score). If the list
            is empty, raises ValueError.

        Args:
            candidates: List of (code, mean_logprob, n_failures) tuples.

        Returns:
            Tuple of (best_index, best_score).

        Raises:
            ValueError: If candidates list is empty.

        Spec: REQ-INFER-013
        """
        if not candidates:
            msg = "candidates list must not be empty"
            raise ValueError(msg)

        best_idx = 0
        best_score = self.score_candidate(*candidates[0])

        for i in range(1, len(candidates)):
            score = self.score_candidate(*candidates[i])
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score
