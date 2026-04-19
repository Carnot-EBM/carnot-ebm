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

## The Four Model Tiers

Carnot includes four tiers of energy-based models, each with different capacity and speed:

| Tier | Name | Architecture | Use case |
|------|------|-------------|----------|
| Hardware | **Ising** | Quadratic energy (pairwise spin couplings) | Real-time sampling, maps to FPGA/TSU hardware |
| Efficient | **KAN** | Learnable B-spline edges (Kolmogorov-Arnold Networks) | **Default for verification** -- best accuracy per parameter (0.994 AUROC with 2.3K params, 8.7x fewer than Ising at same accuracy) |
| Medium | **Gibbs** | Multi-layer MLP energy | Complex pattern learning |
| Large | **Boltzmann** | Deep residual network with attention | Research frontiers, large-scale generation |

**When to use which:** KAN is the default for constraint verification -- it achieves the highest accuracy with the fewest parameters. Ising is for real-time guided decoding and hardware deployment (fastest sampling, maps directly to physical p-bits on FPGA/TSU hardware). They complement each other: KAN for accuracy, Ising for speed.

Each tier implements the same `EnergyFunction` trait (Rust) / protocol (Python), so algorithms written against the interface work with any tier. Models trained in one language can be loaded in the other via the safetensors format.

## Why Ising? Learned and Writable

A common question, especially when comparing Carnot to pure learned architectures like JEPA (Joint Embedding Predictive Architecture), is: if you're going to train an energy model on data anyway, why bother with an Ising at all? Isn't a learned embedding space strictly more expressive?

The answer is that Ising is not just a learned function -- it is a **learned and writable** function, and no other standard architecture offers this cleanly.

**Learned:** Ising couplings (`J_ij`) and biases (`h_i`) are trained from data the same way any neural net is trained -- typical recipes include contrastive divergence, pseudo-likelihood, and score matching. In that mode Ising is a perfectly ordinary learned EBM. Carnot's EORM (Energy-based Outcome Reward Model) uses exactly this path to learn CoT step quality from labelled examples.

**Writable:** But Ising couplings can also be **written down directly** from symbolic reasoning. If a theorem prover (Z3) establishes an invariant like "these two facts cannot both be true", that invariant compiles into a specific `J_ij` entry -- a large negative coupling penalising the disallowed joint state. The resulting energy surface is well-defined whether the entries came from training or from theorem-prover output, and you can mix both in the same model. Hand-coded arithmetic invariants, unit-of-measure constraints, monotonicity requirements -- all of them drop into the same `J` matrix alongside the learned entries.

This matters because the verification pipeline has two kinds of knowledge to represent. Some things are learned from data ("CoT steps that say X usually hallucinate"). Other things are *known* by symbolic reasoning ("if the question asks about dollars and the answer is in euros without a conversion, the answer is wrong"). A pure learned architecture like JEPA handles the first kind beautifully but has no natural interface for the second -- you cannot `add_constraint("a + b = c")` to a JEPA's embedding space, because its energy lives in latent coordinates that symbolic reasoning does not speak. Ising's discrete pairwise form *is* the interface symbolic reasoning already speaks.

The practical architecture in Carnot reflects this. JEPA-style detectors sit in the fast-detection tier -- they are excellent at flagging statistically anomalous steps. Ising sits underneath, accepting the same flags plus whatever Z3-compiled invariants apply to the domain, and producing the continuous energy gradient that the repair step descends to actually fix the answer. JEPA flags; Ising fixes.

Put differently: in a learned-only architecture, every piece of knowledge has to be paid for in training data. In a learned-plus-writable architecture, knowledge that is already provable symbolically does not have to be re-learned from examples. That is a significant efficiency gain for any domain where some rules are genuinely known.

This also maps cleanly onto the hardware roadmap. An Ising's `J` matrix is the natural representation for physical p-bits, so FPGA bitfiles and Extropic-style TSU primitives speak the same format. The writable half of the model becomes a hardware contract, not just a software convenience.

## Next Steps

- [Getting Started](getting-started.md) -- install and run your first verification
- [API Reference](api-reference.md) -- all public classes and methods
- [Technical Report](technical-report.html) -- the full research report (300+ experiments)
