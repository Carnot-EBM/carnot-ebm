"""PSV-PaCoRe (Parallel Chain with Reward-shaped Constraints) K=2 runner.

**Why this module exists (Exp 709 hypothesis):**
    Exp 697 showed that single-chain PSV self-play DEGRADED over 10 iterations
    (fp_rate_trend_slope=+0.004242).  The root cause: the same 10 questions
    answered by the same model at the same temperature produce nearly identical
    responses each iteration.  The constraint pool fills with redundant FP noise
    rather than genuinely diverse violation patterns.

    PaCoRe-style K=2 chains (arXiv 2601.05593) fix this by running two instances
    of the same model at different temperatures (temp_A=0.7, temp_B=1.0).  A
    greedy model (temp_A=0.7) produces near-deterministic responses; a stochastic
    model (temp_B=1.0) explores more of the error manifold.  The union of their
    violations gives the constraint pool a richer training signal.

**Energy-merge:**
    For each question we have two candidate responses.  We prefer the one with
    LOWER violation energy (fewer detected constraint violations).  This is a
    conservative selection policy: we keep the response the verifier trusts more.
    Both responses still contribute to the violation pool regardless of which is
    selected — we want to LEARN from all violations, not just the selected one.

**Binary energy proxy:**
    PSV verify_fn returns a boolean (True = no violations, False = violation).
    We map True -> energy=0.0 and False -> energy=1.0.  This binary proxy is
    sufficient for the merge step; a future extension could use raw logit energies
    from the EBM, but that requires EBM scoring of every inference response.

Spec: REQ-LEARN-020, REQ-LEARN-021,
      SCENARIO-LEARN-020, SCENARIO-LEARN-021
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# IterationResult
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    """Statistics from one PaCoRe PSV iteration (both chains combined).

    Attributes:
        iteration:          Zero-based iteration index.
        best_responses:     One response per question, selected by energy-merge
                            (lower violation energy wins; chain A on tie).
        all_violations:     All (question, response) pairs from BOTH chains where
                            verify_fn returned False.  Both chains contribute to
                            the pool regardless of which response was selected.
        fp_rate_estimate:   Fraction of questions where BOTH chain A AND chain B
                            responses were flagged as violations.  This is the
                            conservative (strict) FP rate: a question only counts
                            as an FP if neither chain could answer it correctly.
        n_questions:        Number of questions processed this iteration.
        n_chain_a_violations: Count of chain-A violations (verify_fn=False).
        n_chain_b_violations: Count of chain-B violations (verify_fn=False).
    """

    iteration: int
    best_responses: List[str]
    all_violations: List[Tuple[str, str]]
    fp_rate_estimate: float
    n_questions: int
    n_chain_a_violations: int
    n_chain_b_violations: int


# ---------------------------------------------------------------------------
# PSVPaCoReRunner
# ---------------------------------------------------------------------------


class PSVPaCoReRunner:
    """Run K=2 parallel PSV chains at different temperatures and merge violations.

    **Why different temperatures instead of different models:**
        Loading two different models costs 2× VRAM.  Running the same model at
        temp_A=0.7 (near-greedy) and temp_B=1.0 (stochastic) costs only marginal
        VRAM overhead for the extra forward pass.  The two temperatures produce
        qualitatively different error patterns from the same underlying model,
        which is sufficient to diversify the constraint pool.

    **Thread model:**
        In GPU environments, chain A and chain B CAN be dispatched to different
        CUDA devices (model_a_device / model_b_device).  This is the DualGPU path
        confirmed in Exp 685 (2.0175× speedup).  When only one GPU is available,
        the caller should pass the same device for both chains; the runner will
        run them sequentially (no parallelism, but still diverse via temperature).

    **inference_fn contract:**
        The caller provides a SINGLE inference_fn with signature:
            inference_fn(question: str, temperature: float, device: str) -> str
        This is intentionally different from the vanilla PSV inference_fn (str->str)
        because PaCoRe needs per-call temperature control.

    Args:
        inference_fn:  Maps (question, temperature, device) -> response string.
        verify_fn:     Maps response -> True (correct) / False (violation).
        n_iterations:  Number of PSV iterations to run (default 10).
        n_questions:   Number of questions per iteration (default 10).
    """

    def __init__(
        self,
        inference_fn: Callable[[str, float, str], str],
        verify_fn: Callable[[str], bool],
        n_iterations: int = 10,
        n_questions: int = 10,
    ) -> None:
        self._inference_fn = inference_fn
        self._verify_fn = verify_fn
        self.n_iterations = n_iterations
        self.n_questions = n_questions
        # Shared constraint pool: accumulated (question, response) violation pairs
        # from both chains across all iterations.
        self._constraint_pool: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_iteration(
        self,
        questions: List[str],
        model_a_device: str = "cpu",
        model_b_device: str = "cpu",
        temp_a: float = 0.7,
        temp_b: float = 1.0,
        *,
        iteration: int = 0,
    ) -> IterationResult:
        """Run one PaCoRe PSV iteration: generate, verify, energy-merge, collect.

        **Step-by-step walkthrough for engineers:**
        1. Chain A: call inference_fn(q, temp_a, model_a_device) for each question.
        2. Chain B: call inference_fn(q, temp_b, model_b_device) for each question.
        3. Verify both chains: call verify_fn(response) for each (question, chain).
        4. Energy-merge: for each question, select the response with lower energy
           (verify_fn=True -> energy=0.0; verify_fn=False -> energy=1.0).
           Tie: chain A wins (deterministic, lower temperature is preferred).
        5. Collect violations: add ALL (question, response) pairs where
           verify_fn=False from EITHER chain into self._constraint_pool.
        6. Return IterationResult with all statistics.

        Args:
            questions:        List of question strings to process.
            model_a_device:   CUDA device string for chain A (e.g. "cuda:0" or "cpu").
            model_b_device:   CUDA device string for chain B (e.g. "cuda:1" or "cpu").
            temp_a:           Temperature for chain A (low = near-greedy).
            temp_b:           Temperature for chain B (high = exploratory).
            iteration:        Zero-based iteration index for the returned result.

        Returns:
            IterationResult with best_responses, all_violations, fp_rate_estimate,
            and per-chain violation counts.
        """
        responses_a = [self._inference_fn(q, temp_a, model_a_device) for q in questions]
        responses_b = [self._inference_fn(q, temp_b, model_b_device) for q in questions]

        labels_a = [self._verify_fn(r) for r in responses_a]
        labels_b = [self._verify_fn(r) for r in responses_b]

        best_responses: List[str] = []
        new_violations: List[Tuple[str, str]] = []
        both_violated_count = 0

        for i, q in enumerate(questions):
            r_a, r_b = responses_a[i], responses_b[i]
            ok_a, ok_b = labels_a[i], labels_b[i]

            # Energy proxy: True -> 0.0, False -> 1.0
            energy_a = 0.0 if ok_a else 1.0
            energy_b = 0.0 if ok_b else 1.0

            # Energy-merge: prefer lower energy; chain A wins on tie.
            # This is the conservative selection policy from REQ-LEARN-021.
            if energy_b < energy_a:
                best_responses.append(r_b)
            else:
                best_responses.append(r_a)

            # Collect ALL violations from both chains into the constraint pool.
            # Both chains contribute to learning regardless of which was "selected".
            if not ok_a:
                new_violations.append((q, r_a))
            if not ok_b:
                new_violations.append((q, r_b))

            # Conservative FP rate: only count questions where BOTH chains failed.
            if not ok_a and not ok_b:
                both_violated_count += 1

        self._constraint_pool.extend(new_violations)

        fp_rate = both_violated_count / max(len(questions), 1)
        n_violations_a = sum(1 for lbl in labels_a if not lbl)
        n_violations_b = sum(1 for lbl in labels_b if not lbl)

        return IterationResult(
            iteration=iteration,
            best_responses=best_responses,
            all_violations=new_violations,
            fp_rate_estimate=fp_rate,
            n_questions=len(questions),
            n_chain_a_violations=n_violations_a,
            n_chain_b_violations=n_violations_b,
        )

    def run_10_iterations(
        self,
        questions: List[str],
        model_a_device: str = "cpu",
        model_b_device: str = "cpu",
        temp_a: float = 0.7,
        temp_b: float = 1.0,
    ) -> List[IterationResult]:
        """Run n_iterations PSV iterations with both chains, accumulating the pool.

        Each call re-uses the same question list for all iterations.  This mirrors
        the single-chain PSV pattern from Exp 697 (same 10 questions per iteration)
        so the fp_rate_trend_slope comparison is apples-to-apples.

        The key difference from single-chain: the constraint pool grows from BOTH
        chains at different temperatures, so later iterations benefit from a richer
        violation pool even though the question set is fixed.

        Args:
            questions:        Question strings (must have at least 1 question).
            model_a_device:   Device for chain A (e.g. "cuda:0" or "cpu").
            model_b_device:   Device for chain B (e.g. "cuda:1" or "cpu").
            temp_a:           Temperature for chain A.
            temp_b:           Temperature for chain B.

        Returns:
            List of IterationResult, one per iteration, in order.
        """
        results: List[IterationResult] = []
        for it in range(self.n_iterations):
            result = self.run_iteration(
                questions,
                model_a_device=model_a_device,
                model_b_device=model_b_device,
                temp_a=temp_a,
                temp_b=temp_b,
                iteration=it,
            )
            results.append(result)
        return results

    @property
    def constraint_pool(self) -> List[Tuple[str, str]]:
        """Read-only view of the accumulated constraint pool from all iterations."""
        return list(self._constraint_pool)
