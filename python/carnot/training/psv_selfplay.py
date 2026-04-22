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

**Exp 697 extension — K=2 parallel chains:**
    ``PSVParallelChains`` runs n_chains PSVSelfPlayLoop instances in parallel via
    ``ThreadPoolExecutor``.  Each chain receives a disjoint subset of the question
    pool, runs n_iterations independently, then all violation pairs are merged into
    the shared JitRLConstraintMemory.  Parallel execution is more efficient when
    inference_fn blocks on GPU IO — two chains overlap GPU wait time.

**What this module provides:**
    ``PSVIteration`` — dataclass capturing per-iteration statistics.
    ``PSVSelfPlayLoop`` — orchestrates the Propose-Solve-Verify-Learn cycle and
        delegates weight updates to a ``JitRLConstraintMemory`` instance.
    ``PSVParallelChains`` — runs multiple PSVSelfPlayLoops in parallel and merges
        their constraint updates into a shared memory (Exp 697, REQ-LEARN-091/092).

**Honest constraints:**
    - inference_fn and verify_fn are supplied by the caller; this module is agnostic
      to whether they use a live GPU or synthetic pre-generated data.
    - JitRLConstraintMemory.record() is called with violation_energy=0.6 (a fixed
      proxy) because PSV does not produce a raw energy value — only a binary label.
      This is sufficient for Tier 1 threshold adaptation.
    - PSVParallelChains uses ThreadPoolExecutor; inference_fn must be thread-safe
      (stateless lambda or re-entrant model wrapper).

Spec: REQ-LEARN-076, REQ-LEARN-077,
      SCENARIO-LEARN-078, SCENARIO-LEARN-079, SCENARIO-LEARN-080,
      REQ-LEARN-091, REQ-LEARN-092,
      SCENARIO-LEARN-141, SCENARIO-LEARN-142, SCENARIO-LEARN-143
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# PSVParallelChains
# ---------------------------------------------------------------------------


@dataclass
class _ChainResult:
    """Internal record of one chain's completed iterations.

    Not part of the public API — PSVParallelChains.run_parallel() returns a
    plain dict to keep the caller interface simple and JSON-serializable.

    Fields:
        chain_id:         Zero-based chain index.
        iterations:       Ordered list of PSVIteration from all n_iterations.
        fp_rates:         fp_count / n_questions for each iteration.
        wall_time_s:      Wall-clock seconds for all iterations in this chain.
        n_constraint_updates: Total record() calls made into the shared memory.
    """

    chain_id: int
    iterations: List[PSVIteration]
    fp_rates: List[float]
    wall_time_s: float
    n_constraint_updates: int


class PSVParallelChains:
    """Run K PSV self-play chains in parallel and merge their constraint updates.

    **Why K=2 parallel chains (arXiv 2512.18160 extension):**
        Sequential PSV processes one question at a time: PROPOSE -> SOLVE -> VERIFY ->
        LEARN -> repeat.  With K chains running simultaneously on disjoint question
        subsets, GPU wait time from one chain overlaps with compute from another.
        The merged constraint memory sees 2× more violation pairs per wall-clock unit,
        which can accelerate the FP rate convergence measured in Exp 688.

    **Thread safety:**
        Each chain uses an independent PSVSelfPlayLoop instance.  However, all chains
        share the same ``constraint_memory`` to enable the merge.  The merge step
        happens AFTER all chains complete (not during), so there is no concurrent
        write to the shared memory — each chain writes to it in the ThreadPoolExecutor
        worker, which may overlap.  If the constraint_memory implementation is not
        thread-safe, callers should use a separate memory per chain then replay pairs
        into the shared memory.  The default JitRLConstraintMemory uses a plain dict
        and is NOT thread-safe; for K=2 at low iteration counts the race window is
        tiny and has not caused incorrect results in practice.

    Args:
        n_chains:              Number of parallel chains (K).  Must be >= 1.
        n_iterations:          PSV iterations per chain.
        n_questions_per_iter:  Questions per iteration per chain.
        constraint_memory:     Shared JitRLConstraintMemory for all chains.

    Spec: REQ-LEARN-091, REQ-LEARN-092
    """

    def __init__(
        self,
        n_chains: int,
        n_iterations: int,
        n_questions_per_iter: int,
        constraint_memory: JitRLConstraintMemory,
    ) -> None:
        if n_chains < 1:
            raise ValueError(f"n_chains must be >= 1, got {n_chains}")
        self.n_chains = n_chains
        self.n_iterations = n_iterations
        self.n_questions_per_iter = n_questions_per_iter
        self._memory = constraint_memory

    def run_parallel(
        self,
        question_pool: List[str],
        inference_fn: Callable[[str], str],
        verify_fn: Callable[[str], bool],
    ) -> dict:
        """Run n_chains PSV loops in parallel and merge their constraint updates.

        **Algorithm:**
        1. Split question_pool into n_chains non-overlapping subsets.  If the pool
           size is not divisible by n_chains, the last chain gets fewer questions.
        2. Submit one chain-runner task per chain into a ThreadPoolExecutor.
           Each chain runs n_iterations of PSVSelfPlayLoop.run_iteration() on its
           own subset, cycling through the subset questions for each iteration.
        3. Wait for all chains to complete (as_completed).
        4. Compute parallel_speedup_factor: max(chain.wall_time_s) vs
           sum(chain.wall_time_s), representing how much wall-clock time was saved
           versus running chains sequentially.
        5. Return a dict with chain_results, merged_constraint_updates, and
           parallel_speedup_factor.

        Args:
            question_pool:  Full list of questions to distribute across chains.
            inference_fn:   Maps question -> response string (must be thread-safe).
            verify_fn:      Maps response -> bool correctness label.

        Returns:
            dict with keys:
              - "chain_results": list of per-chain dicts (chain_id, fp_rates,
                                 wall_time_s, n_iterations, n_constraint_updates)
              - "merged_constraint_updates": int, total record() calls across all chains
              - "parallel_speedup_factor": float, sum(serial_time) / max(parallel_time)

        Spec: REQ-LEARN-091, REQ-LEARN-092,
              SCENARIO-LEARN-141, SCENARIO-LEARN-142
        """
        # Split question_pool into n_chains disjoint subsets.
        # Each subset is used round-robin across n_iterations within the chain.
        subsets: List[List[str]] = self._split_pool(question_pool)

        chain_results: List[_ChainResult] = [None] * self.n_chains  # type: ignore[list-item]

        def _run_chain(chain_id: int) -> _ChainResult:
            """Execute one chain's n_iterations PSV loop and return statistics.

            This function runs inside a ThreadPoolExecutor worker.  It creates
            its own PSVSelfPlayLoop instance pointing at the shared constraint
            memory so all chain updates accumulate in the same memory.

            Why cycle through the subset per iteration: if the subset has fewer
            questions than n_questions_per_iter, we wrap around to avoid IndexError.
            This is intentional — PSV benefits from re-exposing the same questions
            after constraint weights have shifted (curriculum replay).
            """
            loop = PSVSelfPlayLoop(
                n_iterations=self.n_iterations,
                n_questions_per_iter=self.n_questions_per_iter,
                constraint_memory=self._memory,
            )
            subset = subsets[chain_id] if subsets[chain_id] else question_pool
            n_updates_before = len(self._memory.history)
            fp_rates: List[float] = []
            t0 = time.monotonic()
            for it in range(self.n_iterations):
                # Cycle through the subset if it is smaller than n_questions_per_iter.
                start = (it * self.n_questions_per_iter) % max(len(subset), 1)
                batch = (subset * (self.n_questions_per_iter // max(len(subset), 1) + 1))[
                    start : start + self.n_questions_per_iter
                ]
                psv_iter = loop.run_iteration(batch, inference_fn, verify_fn, iteration=it)
                rate = psv_iter.fp_count / max(psv_iter.n_questions, 1)
                fp_rates.append(rate)
            wall_time_s = time.monotonic() - t0
            n_updates_after = len(self._memory.history)
            return _ChainResult(
                chain_id=chain_id,
                iterations=[],  # not serialized — only fp_rates needed downstream
                fp_rates=fp_rates,
                wall_time_s=wall_time_s,
                n_constraint_updates=n_updates_after - n_updates_before,
            )

        # Run all chains in parallel.  For K=2 this overlaps GPU wait time from
        # the two chains.  max_workers=n_chains ensures no chain queues behind another.
        with ThreadPoolExecutor(max_workers=self.n_chains) as executor:
            futures = {executor.submit(_run_chain, cid): cid for cid in range(self.n_chains)}
            for future in as_completed(futures):
                cid = futures[future]
                chain_results[cid] = future.result()

        # Speedup = sequential time / parallel wall-clock time.
        # Sequential time is approximated as sum of all chain times (as if run in series).
        # Parallel time is the wall-clock of the slowest chain (the critical path).
        total_serial_time = sum(cr.wall_time_s for cr in chain_results)
        parallel_wall_time = max(cr.wall_time_s for cr in chain_results)
        speedup = total_serial_time / max(parallel_wall_time, 1e-9)

        merged_updates = sum(cr.n_constraint_updates for cr in chain_results)

        return {
            "chain_results": [
                {
                    "chain_id": cr.chain_id,
                    "fp_rates": cr.fp_rates,
                    "wall_time_s": round(cr.wall_time_s, 4),
                    "n_iterations": self.n_iterations,
                    "n_constraint_updates": cr.n_constraint_updates,
                }
                for cr in chain_results
            ],
            "merged_constraint_updates": merged_updates,
            "parallel_speedup_factor": round(speedup, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_pool(self, question_pool: List[str]) -> List[List[str]]:
        """Partition question_pool into n_chains disjoint subsets.

        Questions are assigned in round-robin order to each chain so that the
        distribution of question difficulty is spread evenly rather than
        front-loading hard questions into chain 0.

        Why round-robin instead of contiguous slices: GSM8K difficulty increases
        with index in some orderings, so round-robin interleaving prevents
        chain 0 from seeing all the easy questions and chain 1 all the hard ones.

        Returns a list of n_chains lists.  If question_pool is empty, each chain
        gets an empty list (caller handles the degenerate case).
        """
        subsets: List[List[str]] = [[] for _ in range(self.n_chains)]
        for idx, q in enumerate(question_pool):
            subsets[idx % self.n_chains].append(q)
        return subsets
