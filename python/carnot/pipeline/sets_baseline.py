"""SETS baseline: Self-Evaluation-Then-Self-Correction (arXiv 2501.19306).

**Researcher summary:**
    SETS is the closest published system to Carnot as of January 2026.  It uses
    Best-of-N sampling (generate N candidates), zero-shot LLM self-verification
    to pick the best candidate, and a self-correction prompt if no candidate
    passes.  The key difference from Carnot: SETS uses LLM calls as the oracle
    (expensive, high-latency); Carnot uses energy evaluation (cheap, hardware-
    acceleratable).  This module implements SETS so Exp 773 can run a direct
    head-to-head comparison on the same GSM8K question set.

**Detailed explanation for engineers:**
    The SETS pipeline from the paper has three stages:
    1. Generate N responses in parallel using N distinct prompt prefixes.  Each
       prefix represents a slightly different instruction-following nudge, which
       induces diversity in the candidate pool without requiring sampling with
       high temperature (which can degrade small-model quality).
    2. For each candidate: ask the LLM "Is this correct? Answer Yes or No."
       The first candidate for which the LLM says "Yes" is selected.  If none
       say "Yes", pick the first candidate by default (fallback).
    3. If the selected candidate still looks wrong (or if we want to attempt
       improvement regardless), run one self-correction pass: "Your solution
       may have errors. Correct it: ..." and return the corrected response.

    Oracle call accounting:
    - Generation phase: N LLM calls (one per candidate prefix).
    - Verification phase: up to N LLM calls (one per candidate until "Yes").
    - Correction phase: 1 LLM call if applied.
    Total worst case: 2N + 1 calls per question.

    For comparison, Carnot's verify-repair loop uses:
    - 1 generation call + K energy evaluations (much cheaper than LLM calls).

Spec: REQ-COMPARE-001, REQ-COMPARE-002
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Prompt prefixes for candidate generation
# ---------------------------------------------------------------------------

CANDIDATE_PREFIXES: list[str] = [
    "Solve step by step:",
    "Think carefully then solve:",
    "Let me work through this:",
    "Using arithmetic:",
]
"""Four prompt prefixes used to induce diversity in the candidate pool.

Each prefix nudges the LLM toward a slightly different reasoning style,
which is the standard diversity trick for Best-of-N sampling without raising
temperature.  Four prefixes is the default from the SETS paper (N=4).
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SETSConfig:
    """Configuration for the SETS baseline.

    Attributes:
        n_candidates: Number of candidate solutions to generate (default 4).
            Must match len(CANDIDATE_PREFIXES) when using the default prefix list.
        max_correction_rounds: Maximum number of self-correction passes to apply
            after selection (default 2).  In practice Exp 773 uses 1 because
            multi-round correction does not measurably improve GSM8K pass rate
            for small CPU models.

    Spec: REQ-COMPARE-001
    """

    n_candidates: int = 4
    max_correction_rounds: int = 2


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SETSResult:
    """Result of running the SETS pipeline on a single question.

    Attributes:
        answer: The final answer string after generation, selection, and
            optional correction.
        pass_flag: True if the answer was judged correct (by ground-truth
            comparison outside this class — this field is set by the caller
            after comparing answer to expected_answer).
        n_oracle_calls: Total number of LLM calls made for this question
            (generation + verification + correction).
        wall_time_s: Wall-clock seconds to run the full SETS pipeline.
        candidates: All N generated candidates (for debugging / ablation).
        selected_index: Index into candidates of the chosen response before
            correction.
        correction_applied: True if a self-correction pass was run.

    Spec: REQ-COMPARE-002
    """

    answer: str
    pass_flag: bool
    n_oracle_calls: int
    wall_time_s: float
    candidates: list[str] = field(default_factory=list)
    selected_index: int = 0
    correction_applied: bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SETSBaseline:
    """SETS pipeline: generate N candidates, self-verify, self-correct.

    **Researcher summary:**
        Drop-in SETS implementation from arXiv 2501.19306.  Pass any callable
        that takes a string prompt and returns a string response as llm_fn.
        The class counts oracle calls automatically and records wall time.

    **Detailed explanation for engineers:**
        llm_fn is called for EVERY oracle interaction: candidate generation,
        self-verification, and self-correction.  This makes oracle call counting
        accurate regardless of what the underlying LLM is (mock, local, API).

        The implementation is intentionally simple and faithful to the paper.
        No batching, no caching, no parallelism — we want to isolate the
        architectural overhead of SETS vs. Carnot, not conflate it with
        implementation differences.

    Args:
        llm_fn: Callable[[str], str] — takes a prompt string, returns a
            response string.  May be a mock or a real LLM.
        config: SETSConfig controlling n_candidates and max_correction_rounds.
        prefixes: Optional override for candidate generation prefixes.  Defaults
            to CANDIDATE_PREFIXES (4 entries).

    Spec: REQ-COMPARE-001, REQ-COMPARE-002
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        config: SETSConfig | None = None,
        prefixes: list[str] | None = None,
    ) -> None:
        self._llm_fn = llm_fn
        self._config = config or SETSConfig()
        self._prefixes = prefixes if prefixes is not None else CANDIDATE_PREFIXES[: self._config.n_candidates]

    def generate_candidates(self, question: str) -> list[str]:
        """Generate N candidate responses using distinct prompt prefixes.

        Why N prefixes instead of N identical prompts: identical prompts with
        a deterministic LLM produce identical outputs.  Distinct prefixes
        induce lexical and structural variation, making the verification step
        meaningful (at least one candidate may be more correct than others).

        Oracle calls: exactly N (one per prefix).

        Args:
            question: The question to answer.

        Returns:
            List of N response strings, one per prefix.

        Spec: REQ-COMPARE-001 (a)
        """
        candidates: list[str] = []
        for prefix in self._prefixes:
            prompt = f"{prefix} {question}"
            candidates.append(self._llm_fn(prompt))
        return candidates

    def self_verify(self, question: str, candidate: str) -> bool:
        """Ask the LLM whether a candidate answer is correct.

        The verification prompt is intentionally minimal — exactly as in the
        SETS paper — to avoid giving the verifier-LLM hints that could inflate
        the apparent pass rate.  We parse only the first token of the response:
        "Yes" → True, anything else → False.

        Oracle calls: 1.

        Args:
            question: The original question.
            candidate: The candidate response to verify.

        Returns:
            True if the LLM says "Yes", False otherwise.

        Spec: REQ-COMPARE-001 (b)
        """
        prompt = f"Question: {question}\nAnswer: {candidate}\nIs this correct? Answer Yes or No."
        response = self._llm_fn(prompt)
        # Parse: strip whitespace, lowercase first word.
        first_word = response.strip().split()[0].lower().rstrip(".,!") if response.strip() else ""
        return first_word == "yes"

    def self_correct(self, question: str, candidate: str) -> str:
        """Ask the LLM to correct a potentially wrong candidate.

        This is the self-correction stage from the SETS paper.  It is run on
        the selected candidate (best by self-verification, or fallback to first)
        when the pipeline has not found a verified-correct answer.

        Oracle calls: 1.

        Args:
            question: The original question.
            candidate: The selected response to correct.

        Returns:
            Corrected response string from the LLM.

        Spec: REQ-COMPARE-001 (c)
        """
        prompt = f"Your solution may have errors. Correct it: {question}\nCurrent: {candidate}"
        return self._llm_fn(prompt)

    def run(self, question: str) -> SETSResult:
        """Run the full SETS pipeline on a single question.

        Pipeline stages and oracle call accounting:
        1. Generate N candidates: N oracle calls.
        2. Self-verify each candidate in order: up to N oracle calls.
           Stop when the first "Yes" is found.
        3. Self-correct the selected candidate: up to max_correction_rounds
           oracle calls.

        pass_flag is initialised to False and must be set by the caller after
        comparing answer to the expected ground-truth answer.

        Args:
            question: The question to answer.

        Returns:
            SETSResult with all fields populated except pass_flag (set to False).

        Spec: REQ-COMPARE-001, REQ-COMPARE-002
        """
        t0 = time.perf_counter()
        n_oracle_calls = 0

        # Stage 1: Generate N candidates.
        candidates = self.generate_candidates(question)
        n_oracle_calls += len(candidates)

        # Stage 2: Self-verify each candidate in order.
        selected_index = 0
        for idx, candidate in enumerate(candidates):
            is_correct = self.self_verify(question, candidate)
            n_oracle_calls += 1
            if is_correct:
                selected_index = idx
                break
        # Fallback: if none verified, use the first candidate.
        selected = candidates[selected_index]

        # Stage 3: Self-correct the selected candidate.
        correction_applied = False
        for _ in range(self._config.max_correction_rounds):
            corrected = self.self_correct(question, selected)
            n_oracle_calls += 1
            correction_applied = True
            selected = corrected

        wall_time_s = time.perf_counter() - t0

        return SETSResult(
            answer=selected,
            pass_flag=False,  # Caller must set this after ground-truth comparison.
            n_oracle_calls=n_oracle_calls,
            wall_time_s=wall_time_s,
            candidates=candidates,
            selected_index=selected_index,
            correction_applied=correction_applied,
        )
