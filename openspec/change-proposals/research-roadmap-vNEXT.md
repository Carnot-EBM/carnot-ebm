# Carnot Research Roadmap v20: Semantic Verification + Typed Constraints + Live Trace Learning

**Created:** 2026-04-12
**Milestone:** 2026.04.15
**Status:** Planned (activates when milestone 2026.04.14 completes)
**Supersedes:** research-roadmap-v19.md (milestone 2026.04.14)
**Informed by:** Exp 203 (live extraction autopsy), Exp 206 (Z3 live benchmark), Exp 207 (LLM extractor live benchmark), Exp 208 (live HumanEval repair), Exp 209 (provenance cleanup), Exp 210 (research scan)
**External inputs:** ConstraintBench (2602.22465), CRANE (2502.09061), SynCode (OpenReview TMLR 2025), BEAVER (2025), MARCH (2603.24579), Project Aletheia (2601.14290), property-based testing for code validation (2506.18315)

## What 2026.04.14 Proved

| Approach | Experiments | Finding |
|----------|-------------|---------|
| Live arithmetic extraction autopsy | 203 | Gemma4-E4B-it's wrong GSM8K answers were mostly semantic or question-grounding failures, not arithmetic slips. |
| Formal vs LLM arithmetic extraction | 204-207 | Z3 and LLM extractors both reduced false positives relative to regex, but neither detected any of the 9 live wrong answers in the shared 100-question benchmark. |
| Live code verification | 208 | Execution-based verification still works on instruction-tuned models: Gemma4-E4B-it improved from 16.7% to 20.0% on official HumanEval. |
| Honest reporting pass | 209 | Simulated gains cannot drive milestone design anymore; roadmap decisions now need live evidence and explicit provenance. |
| Research scan | 210 | The natural next sequence is `Exp 211 -> Exp 213 -> Exp 212`: define a prompt-side constraint IR, measure monitorability, then implement typed extraction. |

**The milestone-level conclusion:** Carnot's current bottleneck is no longer arithmetic normalization. It is monitorable semantic verification for instruction-tuned models. The next milestone must make live outputs structurally legible, then test whether semantic verification and trace learning can produce real gains on Qwen3.5-0.8B and google/gemma-4-E4B-it.

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: No Prompt-Side Constraint IR for Instruction-Tuned Models

Carnot can verify what it can formalize, but live instruction-tuned outputs still arrive as loosely structured prose. That means the system has no stable intermediate representation for user constraints, reasoning steps, or question-grounded claims. This blocks semantic verification, guided decoding, and any robust multi-turn reasoning loop.

### Gap 2: Live Verification Still Cannot See the Dominant Error Class

Exp 203, 206, and 207 showed the live GSM8K failure mode is semantic: omitted premises, wrong entity binding, or incorrect interpretation of the question. Arithmetic checkers do not help if the model solves the wrong problem correctly. Until Carnot has a semantic/question-grounding verifier, it cannot close the core PRD gap for real reasoning verification.

### Gap 3: Continuous Self-Learning Exists in Pieces but Not from Live Traces

Tier 1 and Tier 2 infrastructure already exists in-tree (`ConstraintTracker`, `ConstraintMemory`), but it is not being fed with high-quality live traces from the current verification bottlenecks. The PRD vision requires Carnot to improve from use. That means the next milestone must convert live verifier outcomes into reusable memory, not just produce one-off benchmark numbers.

## v20 Architecture: Typed Constraint IR + Semantic Verifier + Trace Memory

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Prompt-Side Constraint Layer                                               │
│  Exp 211: Constraint IR benchmark + schema                                 │
│  Exp 213: Monitorability policy (free-form vs structured vs no-trace)      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Structured Reasoning Layer                                                  │
│  Exp 212: TypedStepGraph / reasoning IR                                     │
│  Exp 216: CRANE/SynCode-style structured reasoning emission                 │
│  Output: {constraints, claims, steps, final answer, provenance}            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌─────────────────────────────┐   ┌──────────────────────────────────────────┐
│ Semantic Grounding Verifier │   │ Code Verification Path                   │
│  Exp 214: failure corpus    │   │  Exp 217: property-generated tests       │
│  Exp 215: claim decomposition│  │  + existing runtime instrumentation      │
│  + question-entity alignment│   │  + official HumanEval execution          │
└──────────────┬──────────────┘   └──────────────────┬───────────────────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Verify-Repair Pipeline                                                      │
│  Baseline / verify-only / verify-repair                                     │
│  Shared live harness (Exp 218)                                              │
│  Benchmarks: GSM8K semantic (Exp 219), HumanEval property (Exp 220),       │
│  instruction/constraint benchmark (Exp 221)                                 │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Continuous Self-Learning                                                    │
│  Exp 222: live trace -> ConstraintMemory / repair snippets / policy updates │
│  Exp 223: chronological replay benchmark on held-out live traces            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 65: Make Instruction-Tuned Outputs Monitorable (Experiments 211, 213, 212)

This phase directly follows Exp 210's recommendation order.

### Exp 211: Instruction-to-Constraint IR Benchmark

**Deliverable:** `data/research/constraint_ir_benchmark_211.jsonl`

Build a benchmark that turns Carnot's current blocker into a measurable artifact. The benchmark should combine:

- Live GSM8K failures from Exp 203/206/207
- Constraint-following prompts inspired by VIFBench / ConstraintBench / CFBench
- Code prompts where requirements can be rendered as typed properties

Each item should specify the prompt, the expected atomic constraints, their types, the expected verifier path, and whether a free-form trace is monitorable enough to trust. This benchmark becomes the contract for typed reasoning extraction.

### Exp 213: Chain-of-Thought Monitorability Audit + Fallback Policy

**Deliverable:** `results/experiment_213_results.json`

Measure when free-form reasoning traces are useful and when they are actively misleading. For both target models, compare:

- Free-form chain-of-thought
- Answer-only output
- Structured reasoning scaffold / JSON output

Track parseability, constraint coverage, semantic visibility, answer quality, and token cost. The result should be a policy that tells the pipeline when to request structured reasoning, when to accept terse outputs, and when to distrust free-form traces.

### Exp 212: Typed Step Graph / Reasoning IR

**Deliverable:** `python/carnot/pipeline/typed_reasoning.py`

Implement a typed intermediate representation that can hold:

- User constraints
- Reasoning steps
- Claims and references
- Final answers
- Provenance about how the structure was obtained

It must support a dual path: directly parse structured model output when available, and fall back to post-hoc parsing when not. This is the core abstraction the next semantic verifier will consume.

## Phase 66: Build Semantic and Property-Based Verifiers (Experiments 214-217)

This phase turns the live autopsy findings into actual verifier code.

### Exp 214: Semantic Failure Corpus from Live Traces

**Deliverable:** `data/research/semantic_failure_corpus_214.jsonl`

Create a labeled corpus of live failure cases with a taxonomy that separates:

- Question-grounding failures
- Omitted premises
- Entity/quantity binding errors
- Unit/aggregation errors
- Genuine arithmetic mistakes
- Code-specific oracle or property misses

Every example should include the prompt, response, gold diagnosis, and the verifier signal that should have fired.

### Exp 215: Semantic Grounding Verifier

**Deliverable:** `python/carnot/pipeline/semantic_grounding.py`

Implement a semantic verifier that decomposes an answer into atomic claims, aligns those claims against the question, and returns structured violations. The fast path should be deterministic where possible: missing entities, missing required quantities, answer-target mismatch, unsupported claim references. An optional model-assisted checker can refine the verdict, but the first layer should not depend on hidden chain-of-thought.

### Exp 216: Structured Reasoning Emission Path

**Deliverable:** `python/carnot/pipeline/structured_reasoning.py`

Add a structured reasoning emission path for Qwen3.5-0.8B and Gemma4-E4B-it. The key requirement is not "hard grammar everywhere." It is "structured enough to verify without crushing reasoning quality." Use a minimal JSON / typed schema and validate or retry when outputs drift.

### Exp 217: Property-Generated Code Verifier

**Deliverable:** `python/carnot/pipeline/property_code_verifier.py`

Extend the code path from "run official tests" to "derive and check additional properties." Use prompt intent, function signatures, docstrings, and existing tests to synthesize invariants and extra checks. This is the most credible next step after Exp 208 because code verification already has a live positive signal.

## Phase 67: Rebuild the Live Evidence Base (Experiments 218-221)

These experiments convert the new verifier path into live benchmark evidence on the two target small models.

### Exp 218: Shared Dual-Model Live Benchmark Harness

**Deliverable:** `scripts/experiment_218_live_dual_model_suite.py`

Create one checkpointed harness for both target models and all milestone benchmarks. It should guarantee paired prompts, stable seeds, and reusable checkpoint files so that Exp 219-221 are directly comparable instead of ad hoc one-off scripts.

### Exp 219: Live GSM8K Semantic Benchmark (N=200/model)

**Deliverable:** `results/experiment_219_results.json`

Run Qwen3.5-0.8B and Gemma4-E4B-it on 200 GSM8K questions each with:

- Baseline
- Verify-only
- Verify-repair

Use Exp 213's policy to decide when to request structured reasoning. The key metrics are semantic violations detected, false positives, parse coverage, accuracy delta, and repair yield.

### Exp 220: Live HumanEval Property Benchmark (N=50/model)

**Deliverable:** `results/experiment_220_results.json`

Re-run the code path with the stronger verifier. This should extend Exp 208 from execution-only checking to execution plus property-generated verification on both target models.

### Exp 221: Live Constraint IR Benchmark

**Deliverable:** `results/experiment_221_results.json`

Run a curated subset of Exp 211's prompt-side benchmark live. The goal is to measure whether Carnot can actually extract and verify multi-constraint instructions from instruction-tuned models, not just math answers. This is the direct bridge from the semantic verifier work to the broader PRD vision.

## Phase 68: Continuous Self-Learning from Live Traces (Experiments 222-223)

This phase activates the `Continuous Self-Learning` section of `research-program.md` using live data rather than old simulated traces.

### Exp 222: Live Trace -> Memory / Repair Snippet Builder

**Deliverable:** `results/experiment_222_results.json`

Take the outputs of Exp 219-221 and convert them into:

- `ConstraintMemory` entries
- Repair snippets / prompt patches
- Monitorability-policy updates
- Model/domain-specific verifier reliability stats

This is Tier 1 + Tier 2 grounded in current live behavior.

### Exp 223: Chronological Self-Learning Replay Benchmark

**Deliverable:** `results/experiment_223_results.json`

Replay held-out live traces in chronological order and compare:

- No learning
- Tracker-only learning
- Tracker + memory learning

Success is not a simulated "accuracy jumps over time" chart. Success is a small but real gain on held-out traces with a tight false-positive budget and clean provenance from live verifier outcomes.

## Dependency Graph

```
Exp 211 (constraint IR benchmark) ──▶ Exp 213 (monitorability audit) ──▶ Exp 212 (typed reasoning IR)
          │                                         │                           │
          │                                         └───────────────▶ Exp 216 ──┤
          │                                                                     │
          └────────────────────────────────────────────────────────────▶ Exp 221 │

Exp 214 (semantic failure corpus) ────────────────────────────────────▶ Exp 215 ─┼──▶ Exp 218 ─▶ Exp 219
Exp 212 (typed reasoning IR) ─────────────────────────────────────────▶ Exp 215 ─┘        │
Exp 216 (structured reasoning path) ───────────────────────────────────────────────▶ Exp 218 ├▶ Exp 221

Exp 217 (property code verifier) ─────────────────────────────────────────────────▶ Exp 218 ─▶ Exp 220

Exp 219 ─┐
Exp 220 ─┼──────────────────────────────────────────────────────────────▶ Exp 222 ─▶ Exp 223
Exp 221 ─┘
```

## Execution Order

```
1. exp211  -- Define the benchmark contract for prompt-side constraints
2. exp213  -- Measure monitorability before trusting free-form reasoning
3. exp212  -- Implement typed reasoning IR
4. exp214  -- Build the semantic failure corpus
5. exp215  -- Implement semantic grounding verifier
6. exp216  -- Add structured reasoning emission path
7. exp217  -- Strengthen code verification with property checks
8. exp218  -- Build shared live benchmark harness
9. exp219  -- Run live GSM8K semantic benchmark
10. exp220 -- Run live HumanEval property benchmark
11. exp221 -- Run live prompt-side constraint benchmark
12. exp222 -- Turn live traces into reusable memory
13. exp223 -- Validate continuous self-learning on held-out traces
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. | Notes |
|-----------|---------|--------|-----------|-------|
| 211, 214 | CPU | 4-8GB | 15-30 min | Dataset assembly + analysis only |
| 212, 215-217 | CPU | 4-8GB | 30-60 min each | Python implementation + tests |
| 213 | 1x CUDA GPU | 12GB+ VRAM | 30-90 min | Short live audit on both target models |
| 218 | CPU + 1x CUDA GPU | 12GB+ VRAM | 30 min | Harness build + smoke test |
| 219 | 1-2x CUDA GPUs | 12-24GB VRAM each | 2-4 hours | 200 GSM8K/model; paired live runs |
| 220 | 1-2x CUDA GPUs + CPU subprocess execution | 12-24GB VRAM each | 2-4 hours | 50 HumanEval/model with code execution |
| 221 | 1x CUDA GPU | 12GB+ VRAM | 1-2 hours | Curated live prompt-side benchmark |
| 222, 223 | CPU + optional 1x CUDA GPU | 8-16GB | 1-2 hours | Trace ingestion, replay, and held-out evaluation |

**Current hardware fit:** The available dual RTX 3090 setup is enough for this milestone. The Kria KV260 and the TSU / XTR-0 path remain relevant, but neither should sit on the critical path for 2026.04.15.

## Success Criteria

- Carnot can extract a stable typed reasoning / constraint IR from instruction-tuned outputs with measured coverage and fallback behavior.
- The semantic verifier detects real live GSM8K error modes that arithmetic extractors missed, with a materially better detection rate than Exp 206/207 and without regressing into high false positives.
- The code path improves beyond Exp 208 by using property-generated verification, not just official tests.
- The next strongest live evidence base is built on Qwen3.5-0.8B and Gemma4-E4B-it only.
- At least one continuous self-learning experiment shows a real held-out benefit from live trace memory or tracker updates.

## Explicitly Deferred

- JEPA / Tier 3 predictive verification refresh: still valuable, but should consume higher-quality semantic traces after this milestone.
- Tier 4 adaptive energy structure work: still strategically important, but not before Carnot can reliably observe live semantic failures.
- FPGA / TSU acceleration experiments: keep the interface and references current, but do not let hardware novelty distract from the monitorability bottleneck.
