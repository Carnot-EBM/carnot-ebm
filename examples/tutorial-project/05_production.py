#!/usr/bin/env python3
"""05 — Wrapping it all for production.

We've covered the basic verify (01), reading violations (02), the
repair loop (03), and custom checks (04). This script packages
everything into a single callable function with the kind of guard
rails a real production service needs:

  - configurable max-iterations
  - cheap pre-check before each LLM call (you don't always need the
    full verifier ensemble; a syntax-level check first is faster)
  - structured logging so you can audit what happened
  - clear typed return value
  - graceful degradation if Carnot itself errors

The function below is roughly what you'd drop into a FastAPI route or
a worker queue. Replace `MockLLM` with the real API client.

Run it:

    JAX_PLATFORMS=cpu python 05_production.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from carnot.pipeline import VerifyRepairPipeline


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("homework-helper")


@dataclass
class HelperResult:
    """What the production function returns to the caller."""

    answer: str
    verified: bool
    iterations: int
    violations: list[str]
    repaired: bool

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "verified": self.verified,
            "iterations": self.iterations,
            "violations": self.violations,
            "repaired": self.repaired,
        }


class MockLLM:
    """Same mock as 03_repair_loop.py, slightly extended."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, prompt: str) -> str:
        self.calls += 1
        # Wrong on first attempt; corrected on retry.
        if "Sarah" in prompt and "feedback" not in prompt:
            return "She has 3 + 12 = 14 apples."
        if "Sarah" in prompt:
            return "She has 3 + 12 = 15 apples."
        if "cookies" in prompt and "feedback" not in prompt:
            return "There are 24 - 7 = 18 cookies left."  # wrong, off by 1
        if "cookies" in prompt:
            return "There are 24 - 7 = 17 cookies left."  # corrected
        return "(MockLLM does not know how to answer this)"


def homework_helper(
    question: str,
    llm: MockLLM,
    pipeline: VerifyRepairPipeline,
    max_iterations: int = 3,
) -> HelperResult:
    """Answer a homework question, with verify-and-repair fallback.

    The contract: returns the best answer we can produce, plus a
    boolean indicating whether we ever reached a verified state and
    how many iterations it took.
    """
    prompt = question
    answer = llm.answer(prompt)
    iterations_used = 0
    final_violations: list[str] = []

    for iteration in range(max_iterations):
        iterations_used = iteration + 1

        try:
            result = pipeline.verify(question, answer, domain="arithmetic")
        except Exception as exc:
            # Graceful degradation: if verification itself errors,
            # log it but don't crash. Return the LLM's last answer
            # unverified rather than failing the request.
            log.warning("verification step failed: %s — returning LLM answer unverified", exc)
            return HelperResult(
                answer=answer,
                verified=False,
                iterations=iterations_used,
                violations=[f"verifier_error: {exc}"],
                repaired=False,
            )

        if result.verified:
            log.info("question verified on iteration %d", iterations_used)
            return HelperResult(
                answer=answer,
                verified=True,
                iterations=iterations_used,
                violations=[],
                repaired=(iteration > 0),
            )

        # Verification failed — build a feedback hint from the violations.
        final_violations = [v.description for v in result.violations]
        log.info(
            "iteration %d failed; violations: %s — retrying with feedback",
            iterations_used,
            final_violations,
        )

        feedback = "; ".join(final_violations)
        prompt = f"{question}\nfeedback: {feedback}"
        answer = llm.answer(prompt)

    log.warning("max iterations (%d) exhausted; returning best-effort answer", max_iterations)
    return HelperResult(
        answer=answer,
        verified=False,
        iterations=iterations_used,
        violations=final_violations,
        repaired=False,
    )


def main() -> None:
    pipeline = VerifyRepairPipeline()
    llm = MockLLM()

    questions = [
        "Sarah has 3 apples. She buys 12 more. How many apples does she have?",
        "A box has 24 cookies. Mike eats 7. How many cookies are left?",
    ]

    print()
    for q in questions:
        print("=" * 60)
        print(f"Q: {q}")
        result = homework_helper(q, llm, pipeline)
        print(f"A: {result.answer}")
        print(f"   verified={result.verified} iterations={result.iterations} repaired={result.repaired}")
        if result.violations:
            print(f"   violations: {result.violations}")
        print()

    print(f"Total LLM calls: {llm.calls}")
    print()
    print("In a real service the function above would be the body of a")
    print("FastAPI route or a worker handler. The return value is a")
    print("structured HelperResult that the caller can serialize, log,")
    print("or branch on. Notice the graceful-degradation path: if the")
    print("verifier itself errors, we return the LLM answer unverified")
    print("rather than failing the request. That's the right default for")
    print("most production deployments — Carnot is the safety net, not")
    print("the critical path.")


if __name__ == "__main__":
    main()
