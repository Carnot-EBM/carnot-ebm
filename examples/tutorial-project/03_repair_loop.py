#!/usr/bin/env python3
"""03 — The verify-and-repair loop.

When verification fails, the next step is to fix it. Two options:

  (a) Call pipeline.verify_and_repair() and let Carnot do it if you've
      wired up a model. We'll show that API surface.

  (b) Drive your own loop using the violation metadata as feedback to
      whatever LLM you're already using (OpenAI, Anthropic, local
      llama.cpp, anything). This is the more common production pattern
      because it doesn't require Carnot to know about your model.

We'll demo (b) end-to-end using a tiny mock LLM that imitates the
hallucinate-then-correct behavior real LLMs exhibit when you re-prompt
them with the specific failing constraint.

Run it:

    JAX_PLATFORMS=cpu python 03_repair_loop.py
"""

from carnot.pipeline import VerifyRepairPipeline


# A stand-in for a real LLM. In production, replace this with the
# OpenAI / Anthropic / local-model call you're already making.
class MockLLM:
    """Returns the WRONG answer on the first call, the right one on retry.

    Real LLMs aren't this consistent, but the pattern is the same:
    feed the failing constraint back as a hint and ask again.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def answer(self, prompt: str) -> str:
        self.call_count += 1
        if "Sarah" in prompt and "feedback" not in prompt:
            return "Sarah has 3 + 12 = 14 apples in total."  # wrong (off by 1)
        if "Sarah" in prompt and "feedback" in prompt:
            return "Sarah has 3 + 12 = 15 apples in total."  # corrected on retry
        return f"(MockLLM does not know how to answer: {prompt[:40]})"


def repair_with_carnot(
    question: str,
    llm: MockLLM,
    pipeline: VerifyRepairPipeline,
    max_iterations: int = 3,
) -> str:
    """Call the LLM, verify with Carnot, retry with feedback if wrong.

    Returns the final (verified-or-best-effort) answer. The loop is
    short because the strategy is so simple: each retry adds the
    violation description as a feedback hint to the prompt.
    """
    prompt = question
    answer = llm.answer(prompt)

    for iteration in range(max_iterations):
        result = pipeline.verify(question, answer, domain="arithmetic")

        if result.verified:
            print(f"  iter {iteration}: verified on attempt {iteration + 1}")
            return answer

        # Build the feedback hint from the violation metadata.
        feedback_lines = []
        for v in result.violations:
            correct = v.metadata.get("correct_result")
            if correct is not None:
                feedback_lines.append(
                    f"Your previous answer claimed {v.description}, but the correct value is {correct}."
                )
            else:
                feedback_lines.append(f"Your previous answer was wrong: {v.description}")

        feedback = "\n".join(feedback_lines)
        print(f"  iter {iteration}: violation caught — feeding back: {feedback[:80]}...")

        # Re-prompt the LLM with the failing constraint as context.
        # In production this is just `prompt += f"\nfeedback: {feedback}"` and
        # another API call.
        prompt = f"{question}\nfeedback: {feedback}"
        answer = llm.answer(prompt)

    return answer  # max iterations exhausted, return best effort


def main() -> None:
    pipeline = VerifyRepairPipeline()
    llm = MockLLM()

    question = "Sarah has 3 apples. She buys 12 more. How many apples does she have?"

    print(f"Question: {question}")
    print()
    print("Running verify-repair loop:")
    final = repair_with_carnot(question, llm, pipeline)
    print()
    print(f"Final answer: {final}")
    print(f"LLM was called {llm.call_count} times.")
    print()
    print("Notice: on the first call the LLM said 14 (wrong). Carnot caught the")
    print("violation, the loop fed back the correct value as a hint, and the")
    print("second call returned 15. Total: 2 LLM calls + 2 cheap verification")
    print("passes to land on a correct answer.")


if __name__ == "__main__":
    main()
