# Tutorial Project: Hallucination-Resistant Math Homework Helper

A complete walk-through showing how to use Carnot to catch and repair LLM
hallucinations in a real (small) application. By the end of this
~30-minute tutorial you will have built a math homework helper that:

1. Takes a math problem and an LLM-generated answer
2. Verifies the arithmetic and logic in the answer
3. Catches wrong answers automatically
4. Repairs them by re-asking the LLM with feedback
5. Wraps the whole thing as a callable function ready for production

No GPU required. No model download. CPU-only Python.

## Prerequisites

```bash
# Install Carnot from PyPI (or `pip install -e ".[dev]"` from a clone)
pip install carnot-ebm
```

Python 3.11+ is required. Everything in this tutorial runs CPU-only.

## What you will read, in order

| Script | What it teaches | Lines |
|---|---|---|
| [`01_basic_verify.py`](01_basic_verify.py) | The 5-line "hello, world" of Carnot verification | ~40 |
| [`02_catch_errors.py`](02_catch_errors.py) | What violations look like, and how to read them | ~70 |
| [`03_repair_loop.py`](03_repair_loop.py) | The verify-and-repair iteration in 30 lines | ~80 |
| [`04_custom_check.py`](04_custom_check.py) | Adding a domain-specific constraint check | ~80 |
| [`05_production.py`](05_production.py) | Wrapping the whole pipeline as a callable function | ~100 |

Each script is independently runnable. They build on each other in
sequence, but you can also jump to any one if you already know what
came before it.

## Run them in order

```bash
# Force CPU-only JAX (reproducible, no GPU needed)
export JAX_PLATFORMS=cpu

python 01_basic_verify.py
python 02_catch_errors.py
python 03_repair_loop.py
python 04_custom_check.py
python 05_production.py
```

## The scenario

You are building a math homework helper. Students type in a problem
("Sarah has 3 apples and buys 12 more. How many does she have?") and an
LLM gives an answer. **Sometimes the LLM gets the arithmetic wrong.**
You want a programmatic way to catch the wrong answers before showing
them to the student.

That's what Carnot does. The verifier ensemble extracts the
mathematical claims from the answer text ("3 + 12 = 15") and checks
them. If the LLM claims `3 + 12 = 14`, Carnot flags it. The repair
loop then re-asks the LLM with that feedback and gets the right answer
on the next try.

## Why this matters

LLMs are great at *talking* about math and terrible at *doing* math.
For a homework helper that's a deal-breaker — confidently wrong
answers are worse than no answers. Carnot's verifier ensemble gives
you a programmatic catch-and-repair layer that doesn't require
fine-tuning, doesn't require a GPU, and doesn't require trusting the
LLM's self-reported confidence.

The same pattern works for anything where you can write down what
"correct" means as a check: code review (does it parse, does it
type-check, does property-based testing pass?), structured-output
extraction (does the JSON match the schema?), factual claims (do
the dates and quantities check out?). This tutorial focuses on the
arithmetic case because it's the most universally recognizable, but
the API is the same.

## Where to go next

After this tutorial:

- [`../verify_api_responses.py`](../verify_api_responses.py) — same pattern at slightly larger scale, with multi-domain verification
- [`../code_review_pipeline.py`](../code_review_pipeline.py) — Carnot applied to Python code review (type errors, undefined variables, structural bugs)
- [`../custom_extractor.py`](../custom_extractor.py) — building a custom constraint extractor (units-of-measure checking, plugin pattern)
- [`../batch_verify.py`](../batch_verify.py) — batch-verifying a corpus of question/answer pairs with a summary report
- [`../mcp_integration.py`](../mcp_integration.py) — exposing Carnot via the Model Context Protocol so Claude Code can call it as a tool
- [`../../docs/cli-usage.md`](../../docs/cli-usage.md) — using the `carnot` CLI directly
- [`../../docs/usage-guide.md`](../../docs/usage-guide.md) — full reference guide
- [`../../docs/blog/`](../../docs/blog/) — design rationale + lessons from running the verifier ensemble on real data

## If something doesn't work

Run with the diagnostic flag to see what Carnot is doing internally:

```bash
CARNOT_LOG=debug python 01_basic_verify.py
```

If the import fails (`ImportError: No module named 'carnot'`), make
sure the install picked up correctly: `pip show carnot-ebm`. If you
see `0.1.0b1` or later you're good. If not, run
`pip install -U carnot-ebm`.

If a step produces output you don't understand, the inline comments
in each script explain what every line does. Read the script before
running it — it's part of the tutorial.
