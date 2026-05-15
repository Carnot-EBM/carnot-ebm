"""NLA-gated FR-11 policy retention gate for continuous self-learning.

Integrates the NLA (Natural Language Autoencoder) confidence score as a
feedback signal in the FR-11 policy retention gate.  Any policy update
whose NLA confidence score is <= 0.7 is rejected before retention — this
prevents overfit or mode-collapsed policies from entering the active pool.

WHY NLA gating matters:
  The NLA confidence score measures how faithfully the SAE can reconstruct
  model activations for the candidate repair.  A low score (<=0.7) indicates
  the repair activates unusual feature directions that the SAE has not seen
  during normal correct-reasoning examples — a strong proxy for mode collapse
  or hallucination.  Rejecting these candidates keeps the policy pool clean
  without requiring a full re-verification pass.

Spec: REQ-LEARN-2151, SCENARIO-LEARN-2151, SCENARIO-LEARN-2152
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NLA_CONFIDENCE_THRESHOLD = 0.7
MAX_ITERATIONS = 10


@dataclass
class PolicyCandidate:
    """One candidate policy update with its NLA confidence score.

    Fields:
        policy_id:       Unique identifier for this candidate.
        nla_confidence:  NLA SAE reconstruction confidence in [0, 1].
                         High value (>0.7) means the candidate activates
                         well-understood feature directions — safe to retain.
        is_correct:      Ground-truth label: True when the candidate produces
                         a correct repair output, False otherwise.  Used to
                         detect soundness mistakes.
    """

    policy_id: str
    nla_confidence: float
    is_correct: bool


@dataclass
class IterationResult:
    """Outcome of one NLA-gated self-learning iteration.

    Fields:
        iteration:          0-based iteration index.
        n_candidates:       Total candidates evaluated this iteration.
        n_retained:         Candidates that passed the NLA gate.
        n_rejected_by_nla:  Candidates rejected because nla_confidence <= threshold.
        soundness_mistakes: Count of retained-but-incorrect candidates — must stay 0.
    """

    iteration: int
    n_candidates: int
    n_retained: int
    n_rejected_by_nla: int
    soundness_mistakes: int


@dataclass
class NLAGatedLoopResult:
    """Aggregate result of the full NLA-gated self-learning loop.

    Fields:
        iterations_run:     Number of iterations that actually ran (<=MAX_ITERATIONS).
        total_retained:     Cumulative count of retained policy candidates.
        total_rejected:     Cumulative count of NLA-rejected candidates.
        soundness_mistakes: MUST be 0.  Any nonzero value indicates a retained
                            candidate that produced an incorrect result — a
                            soundness failure.
        per_iteration:      Detailed breakdown per iteration.
    """

    iterations_run: int
    total_retained: int
    total_rejected: int
    soundness_mistakes: int
    per_iteration: list[IterationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_run": self.iterations_run,
            "total_retained": self.total_retained,
            "total_rejected": self.total_rejected,
            "soundness_mistakes": self.soundness_mistakes,
            "per_iteration": [
                {
                    "iteration": r.iteration,
                    "n_candidates": r.n_candidates,
                    "n_retained": r.n_retained,
                    "n_rejected_by_nla": r.n_rejected_by_nla,
                    "soundness_mistakes": r.soundness_mistakes,
                }
                for r in self.per_iteration
            ],
        }


class NLAGatedSelfLearner:
    """FR-11 policy retention gate extended with NLA confidence filtering.

    How it works:
        1. For each candidate in an iteration, check nla_confidence > threshold.
        2. Only candidates that pass the threshold are forwarded to the policy pool.
        3. A soundness_mistake is counted whenever a retained candidate is incorrect
           (is_correct == False).  This must never happen in production because the
           upstream verifier must be correct; tracking it here surfaces any regression.

    Args:
        nla_threshold: Confidence cut-off.  Default 0.7 per REQ-LEARN-2151.
        max_iterations: Hard cap on loop iterations.  Default 10.
    """

    def __init__(
        self,
        nla_threshold: float = NLA_CONFIDENCE_THRESHOLD,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.nla_threshold = nla_threshold
        self.max_iterations = max_iterations

    def _gate_candidate(self, candidate: PolicyCandidate) -> tuple[bool, bool]:
        """Apply the NLA gate to a single candidate.

        Returns:
            (retained, soundness_mistake):
                retained           – True if nla_confidence > threshold
                soundness_mistake  – True if retained AND incorrect
        """
        retained = candidate.nla_confidence > self.nla_threshold
        soundness_mistake = retained and not candidate.is_correct
        return retained, soundness_mistake

    def run_iteration(
        self,
        iteration: int,
        candidates: list[PolicyCandidate],
    ) -> IterationResult:
        """Gate one batch of policy candidates with the NLA confidence filter.

        Spec: REQ-LEARN-2151-2

        Args:
            iteration:  0-based iteration index (informational).
            candidates: List of candidate policies for this iteration.

        Returns:
            IterationResult with per-iteration stats.
        """
        n_retained = 0
        n_rejected = 0
        soundness_mistakes = 0

        for candidate in candidates:
            retained, mistake = self._gate_candidate(candidate)
            if retained:
                n_retained += 1
            else:
                n_rejected += 1
            if mistake:
                soundness_mistakes += 1

        return IterationResult(
            iteration=iteration,
            n_candidates=len(candidates),
            n_retained=n_retained,
            n_rejected_by_nla=n_rejected,
            soundness_mistakes=soundness_mistakes,
        )

    def run_loop(
        self,
        batches: list[list[PolicyCandidate]],
    ) -> NLAGatedLoopResult:
        """Run the NLA-gated self-learning loop for at most max_iterations.

        Iterates over ``batches``, stopping early at max_iterations.  Each
        batch is one self-learning round; a batch contains the policy candidates
        the FR-11 relay produced for that round.

        Spec: REQ-LEARN-2151-1, REQ-LEARN-2151-3

        Args:
            batches: One list of PolicyCandidates per iteration.  May have more
                     entries than max_iterations; surplus is silently ignored.

        Returns:
            NLAGatedLoopResult aggregated across all executed iterations.
        """
        per_iteration: list[IterationResult] = []
        total_retained = 0
        total_rejected = 0
        total_soundness = 0

        for idx, batch in enumerate(batches[: self.max_iterations]):
            result = self.run_iteration(iteration=idx, candidates=batch)
            per_iteration.append(result)
            total_retained += result.n_retained
            total_rejected += result.n_rejected_by_nla
            total_soundness += result.soundness_mistakes

        return NLAGatedLoopResult(
            iterations_run=len(per_iteration),
            total_retained=total_retained,
            total_rejected=total_rejected,
            soundness_mistakes=total_soundness,
            per_iteration=per_iteration,
        )


__all__ = [
    "NLA_CONFIDENCE_THRESHOLD",
    "MAX_ITERATIONS",
    "PolicyCandidate",
    "IterationResult",
    "NLAGatedLoopResult",
    "NLAGatedSelfLearner",
]
