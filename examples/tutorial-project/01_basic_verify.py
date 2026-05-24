#!/usr/bin/env python3
"""01 — The 5-line "hello, world" of Carnot verification.

You have an answer to a math problem. You want to know if the
arithmetic in it is right. That's it. Five lines.

This is the simplest possible Carnot program. The next script
(02_catch_errors.py) shows what happens when the answer is wrong.

Run it:

    JAX_PLATFORMS=cpu python 01_basic_verify.py
"""

from carnot.pipeline import VerifyRepairPipeline


def main() -> None:
    # Step 1: construct the pipeline. By default it knows about
    # arithmetic, logic, and a handful of other "domains" — categories
    # of constraint it can extract from natural language.
    pipeline = VerifyRepairPipeline()

    # Step 2: ask it to verify an answer to a question. The pipeline
    # reads the answer text, pulls out the arithmetic claims it can
    # find ("3 + 12 = 15"), and checks each one.
    question = "Sarah has 3 apples. She buys 12 more. How many apples does she have?"
    answer = "She has 3 + 12 = 15 apples in total."

    result = pipeline.verify(question, answer, domain="arithmetic")

    # Step 3: read the result.
    print(f"Question:   {question}")
    print(f"Answer:     {answer}")
    print(f"Verified:   {result.verified}")
    print(f"Found {len(result.constraints)} constraint(s) in the answer:")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", "n/a")
        print(f"   [{satisfied}] {c.description}")


if __name__ == "__main__":
    main()
