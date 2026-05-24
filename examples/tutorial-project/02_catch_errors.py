#!/usr/bin/env python3
"""02 — Catching wrong answers.

Now we feed the pipeline an answer that's actually wrong. The
point of this script is to show what a violation looks like and
how to read it.

Three examples below: one correct answer (passes), one with bad
arithmetic (fails on the math), and one with both bad arithmetic
AND a logic error (fails on multiple constraints).

Run it:

    JAX_PLATFORMS=cpu python 02_catch_errors.py
"""

from carnot.pipeline import VerifyRepairPipeline


def show(label: str, question: str, answer: str, pipeline: VerifyRepairPipeline) -> None:
    """Run the pipeline and print the result in a readable form."""
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Q: {question}")
    print(f"  A: {answer}")

    result = pipeline.verify(question, answer, domain="arithmetic")

    print(f"  verified: {result.verified}")
    print(f"  constraints found: {len(result.constraints)}")
    print(f"  violations: {len(result.violations)}")

    # If anything failed, look at the violation list. Each violation
    # carries a description plus a metadata dict that often includes
    # what the correct value should have been.
    for v in result.violations:
        print(f"     FAIL: {v.description}")
        correct = v.metadata.get("correct_result")
        if correct is not None:
            print(f"           correct answer: {correct}")
    print()


def main() -> None:
    pipeline = VerifyRepairPipeline()

    # --- Example A: correct answer, baseline ---
    show(
        "A: Correct answer",
        question="What is 47 + 28?",
        answer="47 + 28 = 75. Adding the tens place gives 60, then 60 + 15 = 75.",
        pipeline=pipeline,
    )

    # --- Example B: wrong arithmetic ---
    # The LLM "computed" 47 + 28 = 73. Off by 2. Carnot catches it.
    show(
        "B: Wrong arithmetic",
        question="What is 47 + 28?",
        answer="47 + 28 = 73. Simple addition.",
        pipeline=pipeline,
    )

    # --- Example C: multiple errors in one answer ---
    # Two arithmetic claims, both wrong. Carnot returns both violations
    # so you can show them all to the user (or use them all as repair
    # signal in the next script).
    show(
        "C: Multiple arithmetic errors",
        question="If you have 8 boxes with 12 items each, plus 5 extra items, how many in total?",
        answer="8 boxes * 12 items = 95 items. Plus 5 extra is 8 + 12 + 5 = 90 items.",
        pipeline=pipeline,
    )

    print("Reading the output: each violation has a .description telling you")
    print("which claim failed, plus a .metadata dict that often contains the")
    print("correct value. That metadata is the repair signal — see 03_repair_loop.py.")


if __name__ == "__main__":
    main()
