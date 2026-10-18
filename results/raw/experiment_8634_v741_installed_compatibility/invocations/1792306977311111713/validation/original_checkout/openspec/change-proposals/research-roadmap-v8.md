# Carnot Research Roadmap v8: Production Ising — Benchmark, Harden, Scale

**Created:** 2026-04-09
**Milestone:** 2026.04.3
**Status:** Planned (activates when milestone 2026.04.2 completes)
**Supersedes:** research-roadmap-v7.md (milestone 2026.04.2, experiments 53-64)
**Informed by:** Experiments 53-64, live LLM pipeline, continuous relaxation results

## What v7 Proved

| Approach | Experiments | Finding |
|----------|------------|---------|
| Runtime instrumentation | 53 | Dynamic AST rewriting catches bugs static analysis misses |
| Live LLM pipeline | 56 | End-to-end constraint verification with real Qwen3.5 |
| Verify-repair loop | 57 | LLM + Ising iterative repair improves accuracy |
| Constraint-aware prompting | 59 | Preventive constraints reduce hallucinations |
| Scale CD to 100+ vars | 60 | Learned couplings generalize at 10K parameter scale |
| Sparse Ising at 500+ vars | 61 | Clause-graph sparsity enables large-scale learning |
| Continuous relaxation | 64 | Gradient descent on relaxed Ising as alternative to sampling |

**The gap:** The pipeline works end-to-end but only with Qwen, only on synthetic questions, only in Python, and only with hand-coded constraints. v8 makes it credible (real benchmarks), general (multi-model), scalable (learned constraints), and production-ready (Rust port, hardware abstraction).

## v8 Architecture: Production Constraint Pipeline

```
User Question
     │
     ▼
┌──────────────────┐
│  LLM Backend      │ → Qwen3.5-0.8B or Gemma4-E4B (local)
│  (model-agnostic) │ → SamplerBackend abstraction for hardware
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  Constraint       │ → Learned from traces (Exp 55) and data (Exp 62)
│  Extractor        │ → Hierarchical structure (Exp 63)
│  (learned + coded)│ → Coverage metric (Exp 73)
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  Ising Verifier   │ → Hierarchical Ising (1000+ vars)
│  + Repair Loop    │ → Continuous relaxation (Exp 64) or Gibbs
│  (Rust or Python) │ → Embedding-space joint verification (Exp 65)
└──────────────────┘
     │
     ├── PASS → Return verified answer
     │
     ▼ FAIL
┌──────────────────┐
│  Repair + Reprompt│ → Constraint-aware feedback to LLM
│  (verify-repair)  │ → Re-verify (max K iterations)
└──────────────────┘
     │
     ▼
Verified + Repaired Answer
     │
     ▼
┌──────────────────┐
│  Benchmark        │ → GSM8K, HumanEval (real published benchmarks)
│  Evaluation       │ → Multi-domain (arithmetic, code, logic, factual, scheduling)
└──────────────────┘
```

## Phase 9: External Benchmark Validation (experiments 58, 67, 68)

Validate the verify-repair pipeline against published ML benchmarks.

### Exp 58: Multi-domain live pipeline benchmark
Run the verify-repair loop (Exp 57) across 5 domains with 100 questions each:
- Arithmetic word problems (GSM8K-style)
- Code generation (HumanEval-style)
- Logical reasoning (syllogisms, entailment)
- Factual QA (TruthfulQA subset)
- Scheduling/planning problems
- **Metrics per domain:** accuracy, hallucination rate, repair success rate,
  average Ising energy, constraint coverage
- **Comparison:** LLM alone vs LLM + Ising verify vs LLM + Ising verify-repair
- **Models:** Qwen3.5-0.8B (primary)

### Exp 67: GSM8K subset verification
First contact with a real published benchmark:
- Load 200 GSM8K questions from the dataset
- Run the full verify-repair pipeline (Exp 57)
- Extract arithmetic constraints, verify via Ising
- **Metric:** Absolute accuracy on GSM8K subset, improvement over baseline Qwen3.5
- **Comparison:** Baseline LLM vs LLM + verify-repair

### Exp 68: HumanEval subset verification + Ising-guided fuzzing
Code generation benchmark with full constraint toolchain:
- Load 50 HumanEval problems
- LLM generates code, constraint extractor finds types/bounds/returns (Exp 48)
- Runtime instrumentation verifies (Exp 53)
- Ising-guided fuzzing tests edge cases (Exp 54)
- Verify-repair loop re-prompts on failures
- **Metric:** pass@1 improvement with and without Ising verification

## Phase 10: Constraint Learning at Scale (experiments 55, 62, 63, 73)

Move from hand-coded constraints to learned constraint structures.

### Exp 55: Dynamic constraint learning from execution traces
Learn constraints from execution behavior instead of hand-coding:
- Generate 1000 correct + 1000 buggy execution traces
- Extract per-line features: variable types, values, branch decisions
- Train discriminative CD Ising model (like Exp 51)
- **Metric:** Bug detection rate vs static-only analysis

### Exp 62: Domain-specific constraint learning at scale
Scale discriminative learning to 10K triples:
- Collect 10,000 (question, correct_answer, wrong_answer) triples using Qwen3.5
- Binary feature encoding (200+ features per answer)
- Train discriminative Ising (200 vars, sparse)
- **Metric:** AUROC for correct/wrong discrimination on held-out data
- **Comparison:** Learned vs hand-coded vs logprob baseline

### Exp 63: Hierarchical Ising for compositional reasoning
Decompose large problems into hierarchical blocks:
- Group spins into blocks (per-clause for SAT, per-bit for arithmetic)
- Learn inter-block sparse + intra-block dense couplings
- Multi-scale Gibbs sampling (blocks first, then refine within)
- **Metric:** Quality and speed vs flat Ising at 1000+ vars

### Exp 73: Constraint coverage metric
Quantify what the pipeline can and cannot verify:
- For each answer in Exp 58 benchmark, compute fraction of verifiable claims
  covered by extracted constraints
- Identify "dark matter" — claims that escape verification
- Categorize uncovered claims by type
- **Metric:** Mean constraint coverage, correlation between coverage and accuracy

## Phase 11: Model Generality & Hardening (experiments 54, 69, 70)

### Exp 54: Ising-guided fuzzing
Use the Ising energy landscape to generate adversarial test inputs:
- Encode input space as spins, constraint violations as energy
- ParallelIsingSampler generates low-energy inputs (likely to expose bugs)
- Compare outputs of LLM function vs reference on Ising-selected inputs
- **Metric:** Bug detection rate vs random fuzzing

### Exp 69: Multi-model constraint verification (Qwen3.5 + Gemma4)
Validate that the constraint pipeline transfers across model families:
- Run the live verify-repair pipeline on Qwen3.5-0.8B and google/gemma-4-E4B-it
- Same 20 questions from Exp 56
- **Key question:** Does the Ising constraint pipeline transfer across models
  without retraining?
- **Metric:** Cross-model accuracy, constraint satisfaction rates
- **Models:** Qwen3.5-0.8B, google/gemma-4-E4B-it

### Exp 70: Rust constraint extraction and verification pipeline
Port the core constraint pipeline to Rust:
- Create `carnot-constraints` crate
- Port ComposedEnergy and Ising verification from `python/carnot/verify/constraint.py`
- Port AST-based constraint extraction using `tree-sitter` for Python parsing
- Cross-language conformance: same inputs → same verification results
- **Metric:** Feature parity with Python, performance comparison

## Phase 12: Continuous Reasoning & Hardware Prep (experiments 65, 71, 72)

### Exp 65: Embedding-space constraint verification
Bridge discrete Ising constraints to continuous embedding space:
- Encode LLM answer as sentence embedding (all-MiniLM-L6-v2, 384-dim)
- Concatenate with constraint satisfaction vector from Ising verifier
- Train small Gibbs EBM on joint (embedding + constraint) space
- Gradient descent repair in joint space
- **Metric:** AUROC improvement over Ising-only, repair success rate
- **Prerequisite for:** Exp 66 (end-to-end differentiable, deferred to v9)

### Exp 71: Extropic TSU abstraction layer
Create hardware-agnostic sampling interface:
- Define `SamplerBackend` protocol/trait
- Implementations: `CpuBackend` (ParallelIsingSampler), `TsuBackend` (stub)
- Interface: `minimize_energy(biases, couplings, config) -> spins`
- All existing experiments swap backends via config
- **Metric:** All existing tests pass with CpuBackend, TsuBackend exercises interface

### Exp 72: Autoresearch self-verification via Ising
Dog-food the constraint pipeline on the autoresearch loop:
- When orchestrator evaluates a hypothesis, extract verifiable claims
  from hypothesis code (Exp 48) and results (Exp 49)
- Verify via Ising as a fourth gate in the evaluator
- **Metric:** False acceptance rate with vs without Ising gate

## Dependencies

```
2026.04.2 outputs (assumed done):
  Exp 53 ✓, Exp 56 ✓, Exp 57 ✓, Exp 59 ✓, Exp 60 ✓, Exp 61 ✓, Exp 64 ✓

Phase 9 (benchmarks):
  Exp 58 ← Exp 57
  Exp 67 ← Exp 58
  Exp 68 ← Exp 58 + Exp 53 + Exp 54

Phase 10 (constraint learning):
  Exp 55 ← Exp 53 + Exp 51
  Exp 62 ← Exp 60 + Exp 51
  Exp 63 ← Exp 61
  Exp 73 ← Exp 58

Phase 11 (hardening):
  Exp 54 ← Exp 53
  Exp 69 ← Exp 56
  Exp 70 ← Exp 48 + Exp 47

Phase 12 (continuous + hardware):
  Exp 65 ← Exp 64
  Exp 71 ← parallel Ising sampler
  Exp 72 ← Exp 57 + autoresearch orchestrator
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 54, 55 | CPU | 2GB | 10-15 min each |
| 58 | CPU + Qwen3.5 | 4GB | 1-2 hours |
| 62 | CPU + Qwen3.5 | 4GB | 1-2 hours |
| 63 | CPU | 4GB | 30 min |
| 65 | CPU + sentence-transformers | 4GB | 30 min |
| 67, 68 | CPU + Qwen3.5 | 4GB | 30-60 min each |
| 69 | CPU + Qwen3.5 + Gemma4 | 8GB | 30 min |
| 70 | CPU (Rust build) | 2GB | 30-60 min |
| 71 | CPU | 2GB | 15 min |
| 72 | CPU | 2GB | 15 min |
| 73 | CPU | 2GB | 15 min |

## Explicitly Deferred to 2026.04.4

- **Exp 66**: End-to-end differentiable constraint reasoning (needs Exp 65 results)
- **Kona continuous latent reasoning**: needs the full Exp 64+65+66 arc to complete
- **TSU actual integration**: hardware not yet available; v8 builds the abstraction
- **TruthfulQA benchmark**: needs Exp 65 embedding work for factual domain
- **Older model families** (Llama, Phi): only if Qwen3.5+Gemma4 results demand architectural diversity
