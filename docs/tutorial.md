# Tutorial: Building a Hallucination-Resistant Math Homework Helper with Carnot

**Time required:** ~30 minutes
**You will need:** Python 3.11+, a terminal, the ability to run `pip`. No GPU.
**You will build:** a small Python function that takes a math problem and an LLM-generated answer, catches wrong arithmetic, and repairs it by re-prompting the LLM with feedback.

This tutorial walks you through five short Python scripts in order. By the end you will have used the Carnot verification pipeline at every level — from a five-line "hello, world" up to a production-shaped function ready to drop into a FastAPI route. The full code lives at [`examples/tutorial-project/`](../examples/tutorial-project/). You can read along here and run the scripts from that directory as you go.

## Why this scenario

LLMs are very good at *talking* about math and very bad at *doing* math. Ask any production engineer who has shipped an LLM-backed feature: confidently wrong answers are the most common failure mode, and they're worse than no answers because the model's confidence makes them hard to catch automatically. Carnot exists to give you a programmatic catch-and-repair layer for exactly this problem — extract verifiable constraints from the model's output (arithmetic, logic, code, units, schemas — whatever you can specify), check them, and either reject the response or repair it.

We focus the tutorial on arithmetic because it's the most universally recognizable failure mode and the easiest to demonstrate without a GPU. The same API works for every other domain Carnot supports.

## Install

```bash
pip install carnot-ebm
```

Python 3.11+. CPU-only by default (JAX will pick GPU if you have one, but it isn't required). If the install fails, see the troubleshooting note at the end.

## Step 1 — The five-line "hello, world"

Open [`01_basic_verify.py`](../examples/tutorial-project/01_basic_verify.py). The whole file is about forty lines of which five do the work:

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()
result = pipeline.verify(
    question="Sarah has 3 apples. She buys 12 more. How many apples does she have?",
    response="She has 3 + 12 = 15 apples in total.",
    domain="arithmetic",
)
print(result.verified)  # True
```

Run it:

```bash
JAX_PLATFORMS=cpu python 01_basic_verify.py
```

You should see `verified: True` and a single extracted constraint `[True] 3 + 12 = 15`. What happened: the pipeline read the response text, found one arithmetic claim, evaluated it (3 + 12 = 15 ✓), and reported success. No GPU, no model download, no API key.

This is the simplest possible Carnot program. Everything else in this tutorial is variations on this shape.

## Step 2 — What violations look like

Now the question that brought you to Carnot: what happens when the answer is wrong?

Open [`02_catch_errors.py`](../examples/tutorial-project/02_catch_errors.py) and run it. You'll see three examples:

- **Correct answer** — passes, no violations.
- **Wrong arithmetic** — `47 + 28 = 73` (off by 2). Result: `verified: False`, one violation, and the violation's `metadata["correct_result"]` field tells you the answer should have been 75.
- **Multiple errors** — two wrong claims in one response. Carnot returns both violations so you can show them all to a user, or use them all as repair signal.

The output for the wrong-arithmetic case looks like:

```
verified: False
constraints found: 1
violations: 1
   FAIL: 47 + 28 = 73 (correct: 75)
         correct answer: 75
```

**The shape of a violation:** every violation is a structured object with a `.description` (human-readable) and a `.metadata` dict (machine-readable, often including the correct value). The metadata is what makes the repair loop possible — you can feed the correct value back to the LLM as a hint on the next call.

## Step 3 — The repair loop

Open [`03_repair_loop.py`](../examples/tutorial-project/03_repair_loop.py). Here we introduce a `MockLLM` — a placeholder for whatever LLM you're using in production (OpenAI, Anthropic, a local llama.cpp, anything). It returns the wrong answer on the first call and the right answer on the retry, mimicking the way real LLMs often get a question right when you re-prompt them with the specific failure.

The loop is short:

```python
prompt = question
answer = llm.answer(prompt)

for iteration in range(max_iterations):
    result = pipeline.verify(question, answer, domain="arithmetic")
    if result.verified:
        return answer

    # Build the feedback hint from the violation metadata.
    feedback = "; ".join(
        f"Your previous answer claimed {v.description}, but the correct value is {v.metadata.get('correct_result')}"
        for v in result.violations
    )

    prompt = f"{question}\nfeedback: {feedback}"
    answer = llm.answer(prompt)

return answer  # max iterations exhausted, return best effort
```

That's it. Three or four ideas wrapped in a `for` loop:

1. Ask the LLM.
2. Verify with Carnot.
3. If verified, return.
4. If not, build a feedback hint from the violation metadata and try again.

In production you'd replace `MockLLM.answer()` with your actual API call. Everything else stays the same.

When you run the script you'll see:

```
iter 0: violation caught — feeding back: Your previous answer claimed 3 + 12 = 14 (correct: 15)...
iter 1: verified on attempt 2
Final answer: Sarah has 3 + 12 = 15 apples in total.
LLM was called 2 times.
```

Two LLM calls and one extra (cheap, sub-millisecond) verification pass to land on a correct answer. That's the win.

## Step 4 — Adding a domain-specific check

Carnot ships with built-in extractors for arithmetic, logic, and several other domains. Real applications usually have rules the built-ins don't cover. The pattern for extending the pipeline is to implement the `ConstraintExtractor` protocol and register your extractor with the pipeline's `AutoExtractor`.

Open [`04_custom_check.py`](../examples/tutorial-project/04_custom_check.py). The example custom extractor is intentionally silly — it flags answers shorter than 20 characters as "not showing work." The shape is what matters, not the specific rule.

The protocol is small:

```python
class HomeworkFormatExtractor:
    @property
    def supported_domains(self) -> list[str]:
        return ["homework_format"]

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []

        stripped = text.strip()
        satisfied = len(stripped) >= 20
        return [
            ConstraintResult(
                constraint_type="homework_format",
                description=f"answer length >= 20 chars (got {len(stripped)})",
                metadata={"satisfied": satisfied, "length": len(stripped)},
            )
        ]
```

Two methods: `supported_domains` tells the pipeline which domain labels you respond to, and `extract` parses the response and emits a list of `ConstraintResult` objects. Each result carries a `constraint_type`, a `description`, and a `metadata` dict with at least `satisfied: bool`.

Register it with the pipeline:

```python
extractor = AutoExtractor()
extractor.add_extractor(HomeworkFormatExtractor())

pipeline = VerifyRepairPipeline(
    extractor=extractor,
    domains=["arithmetic", "homework_format"],
)
```

Now the pipeline checks BOTH arithmetic and homework_format on every response. The script's output shows a terse-but-correct answer (`75.`) failing on the format check while passing arithmetic, and a full worked-out answer passing both.

This is the extension point. For your own application: schema compliance, units-of-measure consistency, attribution requirements, style guidelines, domain ontologies — anything you can write a Python function to check fits this protocol. See [`../examples/custom_extractor.py`](../examples/custom_extractor.py) for a richer example (units checking with cross-system conflict detection).

## Step 5 — Wrapping it all for production

Open [`05_production.py`](../examples/tutorial-project/05_production.py). This is what the previous four scripts add up to — a single callable function with the kind of guard rails real production services need:

- Configurable `max_iterations` ceiling
- Structured logging on every iteration
- Typed return value via a `HelperResult` dataclass
- **Graceful degradation** if Carnot itself errors — return the LLM's answer marked unverified rather than crashing the request

The function body is about sixty lines and reads top-to-bottom. The key shape:

```python
def homework_helper(
    question: str,
    llm: LLMClient,
    pipeline: VerifyRepairPipeline,
    max_iterations: int = 3,
) -> HelperResult:
    prompt = question
    answer = llm.answer(prompt)

    for iteration in range(max_iterations):
        try:
            result = pipeline.verify(question, answer, domain="arithmetic")
        except Exception as exc:
            # Carnot is the safety net, not the critical path.
            log.warning("verification step failed: %s", exc)
            return HelperResult(answer=answer, verified=False, ...)

        if result.verified:
            return HelperResult(answer=answer, verified=True, ...)

        # Build feedback and retry.
        feedback = "; ".join(v.description for v in result.violations)
        prompt = f"{question}\nfeedback: {feedback}"
        answer = llm.answer(prompt)

    return HelperResult(answer=answer, verified=False, ...)  # max iters exhausted
```

Run the script to see two questions go through this pipeline — one that converges on the second attempt, one that converges on the first.

**This is the function you'd drop into a FastAPI route or a worker handler.** Replace `MockLLM` with your real API client and you have a hallucination-resistant LLM endpoint. The `HelperResult` is JSON-serializable so the caller can log it, branch on it, or surface the violation list to the user.

## What you've learned

| Capability | Where it's introduced |
|---|---|
| Basic verification | Script 01 |
| Reading violations | Script 02 |
| The verify-and-repair pattern | Script 03 |
| Custom domain checks | Script 04 |
| Production wrapping | Script 05 |

The whole API surface for the common case is three classes (`VerifyRepairPipeline`, `AutoExtractor`, `ConstraintResult`) and one protocol (`ConstraintExtractor`). The pipeline is substrate-agnostic — your LLM choice, your serving stack, your monitoring tool are all interchangeable. Carnot is the verification layer that sits between the LLM's output and whatever you do with it.

## What's intentionally not in this tutorial

A few things were left out because they're either premature optimizations or covered better elsewhere:

- **The energy-based verifier ensemble.** Carnot's deeper machinery is a k=15 verifier ensemble with energy-grounded scoring. For the homework-helper scenario, the built-in arithmetic extractor is enough. The full ensemble matters for harder tasks (long-context reasoning, code repair, structured-output validation at scale). See [the technical report](technical-report.md) for the architecture.
- **GPU acceleration.** The CPU path is fast enough for most verification workloads (verification is sub-millisecond per claim; the LLM call dominates). When you need GPU, install with `pip install carnot-ebm[cuda]` and the pipeline picks it up automatically.
- **The MCP server.** If you're integrating Carnot with Claude Code or another MCP-compatible client, see [`mcp-server.md`](mcp-server.md). The Python API is what most production deployments use; the MCP path is for agentic tool-use scenarios.

## Where to go next

- **[The full reference guide](usage-guide.md)** — every public API, with parameters and return values.
- **[CLI usage](cli-usage.md)** — `carnot verify` as a command-line tool, useful for quick checks and scripting.
- **[`examples/code_review_pipeline.py`](../examples/code_review_pipeline.py)** — same pattern applied to Python code: type errors, undefined variables, structural bugs.
- **[`examples/batch_verify.py`](../examples/batch_verify.py)** — verifying a corpus of question/answer pairs from a JSON file with a summary report.
- **[`examples/custom_extractor.py`](../examples/custom_extractor.py)** — a richer custom extractor: units-of-measure consistency with cross-system conflict detection.
- **[The blog series](blog/)** — design rationale and lessons from running the verifier ensemble on real data, including how we caught our own pipeline cheating, why we report two AUROCs now, and what a hostile audit of our paper draft caught.

## If something doesn't work

**`ImportError: No module named 'carnot'`** — `pip install carnot-ebm` again, then `pip show carnot-ebm` to confirm version `0.1.0b1` or later.

**Verification produces unexpected output** — run with `CARNOT_LOG=debug python 0X_script.py` to see what the extractor is parsing and why. The inline comments in each tutorial script explain expected behavior.

**Multiplication or division claims aren't being caught** — the built-in arithmetic extractor focuses on addition and subtraction. For multiplicative reasoning, add a custom extractor (see [`examples/custom_extractor.py`](../examples/custom_extractor.py) for the pattern) or use the `code` domain to evaluate the expression directly.

**Anything else** — file an issue at [github.com/Carnot-EBM/carnot-ebm](https://github.com/Carnot-EBM/carnot-ebm). Reproducer feedback at this stage of the project is genuinely valuable.
