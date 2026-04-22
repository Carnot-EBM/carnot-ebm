"""PSV Self-Play Loop — Propose, Solve, Verify for autonomous constraint weight learning.

**Researcher summary (Exp 688):**
    arXiv 2512.18160 (PSV: Propose, Solve, Verify) shows that self-play with formal
    verification labels enables autonomous improvement without human supervision.

    The PSV loop for Carnot works as follows:
      1. PROPOSE  — select N questions from a problem bank (caller's responsibility).
      2. SOLVE    — call inference_fn(question) -> response for each question.
      3. VERIFY   — call verify_fn(response) -> bool (True = correct) for each response.
      4. LEARN    — pass (violation pairs, correct pairs) to JitRLConstraintMemory so
                    the constraint weights adapt from the binary verification labels.

    After K iterations, we compute the linear regression slope of fp_rate across
    iterations.  A negative slope means the PSV loop is working: the constraint weights
    learned from previous iterations reduce the false-positive rate in later ones.

**What this module provides:**
    ``PSVIteration`` — dataclass capturing per-iteration statistics.
    ``PSVSelfPlayLoop`` — orchestrates the Propose-Solve-Verify-Learn cycle and
        delegates weight updates to a ``JitRLConstraintMemory`` instance.

**Honest constraints:**
    - inference_fn and verify_fn are supplied by the caller; this module is agnostic
      to whether they use a live GPU or synthetic pre-generated data.
    - JitRLConstraintMemory.record() is called with violation_energy=0.6 (a fixed
      proxy) because PSV does not produce a raw energy value — only a binary label.
      This is sufficient for Tier 1 threshold adaptation.
    - Thread-unsafe: designed for single-process experiments.

Spec: REQ-LEARN-076, REQ-LEARN-077,
      SCENARIO-LEARN-078, SCENARIO-LEARN-079, SCENARIO-LEARN-080
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from carnot.pipeline.jitrl_memory import JitRLConstraintMemory

# Fixed proxy energy used when PSV produces only a binary label, not a raw energy.
# 0.6 is above the default base_threshold (0.5), ensuring that true positives
# actually trigger the "lower threshold" branch in JitRLConstraintMemory.record().
_PROXY_VIOLATION_ENERGY: float = 0.6


# ---------------------------------------------------------------------------
# PSVIteration
# ---------------------------------------------------------------------------


@dataclass
class PSVIteration:
    """Per-iteration statistics from one PSV self-play cycle.

    Why a dataclass instead of a dict: typed access prevents key-typo bugs when
    downstream code reads n_correct vs n_violations, which are easy to confuse.

    Attributes:
        iteration:               Zero-based iteration index.
        n_questions:             Number of questions processed this iteration.
        n_correct:               Count of responses where verify_fn returned True.
        n_violations:            Count of responses where verify_fn returned False.
        fp_count:                Alias for n_violations (false-positive candidates
                                 that triggered constraint weight updates).
        constraint_weight_delta: Mean absolute threshold change across all domains
                                 in the constraint memory after this iteration's
                                 update_from_pairs call.  Zero if no domains updated.
    """

    iteration: int
    n_questions: int
    n_correct: int
    n_violations: int
    fp_count: int
    constraint_weight_delta: float


# ---------------------------------------------------------------------------
# PSVSelfPlayLoop
# ---------------------------------------------------------------------------


class PSVSelfPlayLoop:
    """Orchestrates a fixed number of Propose-Solve-Verify-Learn iterations.

    **Detailed explanation for engineers:**
        Each call to ``run_iteration`` processes a batch of questions through three
        stages:
          - Solve:  ``inference_fn(q)`` is called for each question q.
          - Verify: ``verify_fn(r)`` is called for each response r.  Returns True
                    if the response is correct, False if it contains an error.
          - Learn:  ``update_from_pairs`` records each (question, response) pair into
                    the constraint memory.  Incorrect responses (verify_fn=False) are
                    treated as constraint violations (was_fp=False in the memory,
                    meaning they ARE real violations we want to catch).  Correct
                    responses are treated as false-positive candidates (was_fp=True,
                    meaning "this fired but shouldn't have"), which nudges thresholds
                    upward to reduce over-sensitivity.

        Why "fp_count" for violations: from the verifier's perspective, a fired
        constraint that the PSV oracle says is CORRECT is a false positive.
        Conversely, violations confirmed by PSV are true positives.  The JitRL
        memory's "was_fp" convention is: was_fp=True raises the threshold (fewer
        future alerts), was_fp=False lowers the threshold (more aggressive catching).

    Args:
        n_iterations:          Total number of PSV iterations to run.
        n_questions_per_iter:  Questions per iteration (informational; the caller
                               supplies the actual question list to run_iteration).
        constraint_memory:     JitRLConstraintMemory instance to update each iteration.
    """

    def __init__(
        self,
        n_iterations: int,
        n_questions_per_iter: int,
        constraint_memory: JitRLConstraintMemory,
    ) -> None:
        self.n_iterations = n_iterations
        self.n_questions_per_iter = n_questions_per_iter
        self._memory = constraint_memory

    def run_iteration(
        self,
        questions: List[str],
        inference_fn: Callable[[str], str],
        verify_fn: Callable[[str], bool],
        *,
        iteration: int = 0,
    ) -> PSVIteration:
        """Run one PSV iteration: Solve -> Verify -> Learn.

        **Detailed explanation for engineers:**
            - inference_fn: str -> str.  Any callable mapping a question to a response.
              In live mode this calls a real LLM; in synthetic mode it returns a
              pre-generated response from a lookup table.
            - verify_fn: str -> bool.  Any callable mapping a response to a correctness
              label.  In live mode this calls SymCodeVerifier; in synthetic mode it
              checks a pre-known label.
            - Correct responses (verify_fn=True) are used to raise the constraint
              memory threshold ("this looked like a violation but was actually fine").
            - Incorrect responses (verify_fn=False) are used to lower the threshold
              ("the verifier caught a real error; be more aggressive").

        Args:
            questions:    List of question strings to process this iteration.
            inference_fn: Maps question -> response string.
            verify_fn:    Maps response -> True (correct) / False (violation).
            iteration:    Zero-based iteration index for the returned dataclass.

        Returns:
            PSVIteration with counts and constraint weight delta for this iteration.
        """
        responses = [inference_fn(q) for q in questions]
        labels = [verify_fn(r) for r in responses]

        violations: List[Tuple[str, str]] = [
            (q, r) for q, r, lbl in zip(questions, responses, labels) if not lbl
        ]
        correct: List[Tuple[str, str]] = [
            (q, r) for q, r, lbl in zip(questions, responses, labels) if lbl
        ]

        thresholds_before = dict(self._memory._thresholds)
        self._update_from_pairs(violations, correct)
        thresholds_after = dict(self._memory._thresholds)

        delta = self._mean_threshold_delta(thresholds_before, thresholds_after)

        return PSVIteration(
            iteration=iteration,
            n_questions=len(questions),
            n_correct=len(correct),
            n_violations=len(violations),
            fp_count=len(violations),
            constraint_weight_delta=delta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_from_pairs(
        self,
        violations: List[Tuple[str, str]],
        correct: List[Tuple[str, str]],
    ) -> None:
        """Record violation and correct pairs into the constraint memory.

        Why domain="psv_gsm8k" for all pairs: PSV operates on GSM8K arithmetic
        questions.  A single domain label is fine for Tier 1 because threshold
        adaptation is per-domain and we want a unified GSM8K threshold rather than
        splitting by sub-category.

        Violations (verify_fn=False) are recorded with was_fp=False: they are real
        errors that the constraint system should keep catching.
        Correct responses (verify_fn=True) are recorded with was_fp=True: the
        constraint *would* have flagged them, but the oracle says they're fine,
        so we tell the memory to raise the threshold (fewer false alarms).
        """
        domain = "psv_gsm8k"
        for _q, _r in violations:
            self._memory.record(domain, _PROXY_VIOLATION_ENERGY, was_fp=False)
        for _q, _r in correct:
            self._memory.record(domain, _PROXY_VIOLATION_ENERGY, was_fp=True)

    def _mean_threshold_delta(
        self,
        before: dict,
        after: dict,
    ) -> float:
        """Compute mean absolute threshold change across all domains touched this iter.

        Returns 0.0 if no domains were updated (no pairs recorded).
        """
        all_keys = set(before) | set(after)
        if not all_keys:
            return 0.0
        base = self._memory._base_threshold
        deltas = [abs(after.get(k, base) - before.get(k, base)) for k in all_keys]
        return sum(deltas) / len(deltas)
