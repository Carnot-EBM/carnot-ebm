# Concepts

This page explains what Carnot does and why, in plain English. No math required.

## The Problem

Large language models (LLMs) generate text one word at a time, moving forward without looking back. They have no built-in way to check whether the full response is correct. This means an LLM can confidently say "15 + 27 = 43" without ever verifying the arithmetic.

This is called a **hallucination**: the model produces something that sounds right but isn't.

## What Carnot Does

Carnot sits between the LLM and the user (or downstream system). It reads the LLM's response, finds claims that can be checked, and verifies each one. If something is wrong, Carnot reports exactly what failed and why.

Think of it like a spell-checker, but for factual correctness instead of spelling.

### The pipeline in three steps

1. **Extract** -- Carnot scans the response and pulls out verifiable claims. For example, `"47 + 28 = 75"` becomes an arithmetic constraint. A Python code block becomes type-check and initialization constraints.

2. **Verify** -- Each constraint is checked. Arithmetic is computed directly. Code is parsed and analyzed. Logical statements are checked for internal consistency. Each constraint gets a pass/fail result.

3. **Repair** (optional) -- If violations are found and an LLM is connected, Carnot formats the violations as plain-English feedback and asks the LLM to try again. This loop can repeat up to a configurable number of times.

## Constraints

A **constraint** is any claim in the response that Carnot can check. There are four built-in types:

### Arithmetic

Carnot finds patterns like `X + Y = Z` in text and checks the math.

- `"15 + 27 = 42"` -- Carnot computes 15 + 27 = 42. Pass.
- `"15 + 27 = 43"` -- Carnot computes 15 + 27 = 42, not 43. Fail.

### Code

When the response contains Python code (in triple-backtick blocks), Carnot parses it and checks:

- **Type annotations** -- Are parameters and return types declared?
- **Return type consistency** -- Does a literal return value match the declared type?
- **Undefined variables** -- Is any variable used before it's assigned?
- **Loop bounds** -- Are loop variables properly bounded?

### Logic

Carnot finds logical patterns in natural language:

- **Implications** -- "If it rains, then the ground gets wet."
- **Exclusions** -- "Cats but not dogs."
- **Disjunctions** -- "Either A or B."
- **Negations** -- "Birds cannot swim."
- **Universals** -- "All mammals are warm-blooded."

These are extracted for consistency checking. If a response says "If X then Y" and later contradicts it, Carnot can flag the inconsistency.

### Natural Language

Carnot extracts factual claims from prose:

- **"Paris is the capital of France"** -- extracted as a factual relation.
- **"There are 50 states"** -- extracted as a quantity claim.

These are primarily used for structured extraction rather than truth-checking (Carnot doesn't have a knowledge base). They become useful when combined with other verification methods or when checking internal consistency.

## Energy

Behind the scenes, Carnot uses **Energy-Based Models (EBMs)** to score responses. The core idea is simple:

- Every response gets an **energy score**. Low energy means the response looks correct. High energy means something is off.
- Constraints translate into energy terms. A satisfied constraint contributes zero energy. A violated constraint adds energy.
- The total energy across all constraints gives you one number summarizing how "correct" the response is.

You don't need to think about energy to use Carnot. The `verified` flag on every result tells you pass/fail. But energy scores are useful for:

- **Ranking** multiple candidate responses (pick the lowest energy)
- **Thresholding** (accept responses below a certain energy)
- **Debugging** (see which constraint contributes the most energy)

## Verification vs. Repair

Carnot has two modes:

### Verify-only mode (default)

You provide a question and a response. Carnot checks the response and tells you what's wrong. No LLM needed.

```python
pipeline = VerifyRepairPipeline()
result = pipeline.verify(question, response)
# result.verified, result.violations, result.energy
```

### Verify-and-repair mode

You provide a question and optionally a response. Carnot checks it, and if there are violations, sends feedback to a connected LLM to generate a corrected response. This loop repeats up to `max_repairs` times.

```python
pipeline = VerifyRepairPipeline(model="Qwen/Qwen2.5-0.5B-Instruct")
result = pipeline.verify_and_repair(question)
# result.final_response, result.repaired, result.history
```

This is Carnot's core value proposition: EBMs don't just classify outputs as good or bad -- they **guide** the LLM toward correct answers.

## Custom Extractors

The built-in extractors cover arithmetic, code, logic, and natural language. But you can add your own. Any class that implements the `ConstraintExtractor` protocol (a `supported_domains` property and an `extract` method) can be registered with `AutoExtractor` and plugged into the pipeline.

See the [custom extractor example](https://github.com/Carnot-EBM/carnot-ebm/tree/main/examples/custom_extractor.py) for a units-of-measure checker that flags mixed metric/imperial without conversion.

## The Three Model Tiers

Carnot includes three tiers of energy-based models, each with different capacity and speed:

| Tier | Name | Architecture | Use case |
|------|------|-------------|----------|
| Small | Ising | Quadratic energy (like a spin glass) | Fast, simple constraints |
| Medium | Gibbs | MLP-based energy | Moderate complexity |
| Large | Boltzmann | Deep residual network | High-dimensional verification |

Each tier is implemented in both Rust (for production speed) and Python/JAX (for research flexibility). Models trained in one language can be loaded in the other via the safetensors format.

## Next Steps

- [Getting Started](getting-started.md) -- install and run your first verification
- [API Reference](api-reference.md) -- all public classes and methods
- [Technical Writeup](technical-writeup.html) -- the research behind Carnot
