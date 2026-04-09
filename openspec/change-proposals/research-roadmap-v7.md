# Carnot Research Roadmap v7: Toward Kona — Live LLM + Ising End-to-End

**Created:** 2026-04-09
**Status:** Active
**Supersedes:** research-roadmap-v6.md (all experiments completed)
**Informed by:** Experiments 42-52, parallel Ising sampler, Kona architecture

## What v6 Proved

| Approach | Experiments | Finding |
|----------|------------|---------|
| Arithmetic via QUBO | 42b, 42c | Exact QUBO works; deterministic propagation for verification (16/16) |
| SAT at scale | 46b | Parallel sampler: 5000 vars in 0.7s, +5.5% vs random |
| Logical consistency | 45 | 8/8 contradiction detection |
| Code → constraints | 48 | AST extraction works for types, bounds, returns, init |
| NL → constraints | 49 | Pattern-based claim extraction + Ising verification |
| LLM self-constraints | 47 | 10/10 perfect: all hallucinations caught, all correct verified |
| Scheduling | 44 | Practical constraint domain validated |
| Learn Ising from data | 50 | CD training: 89/100 perfect, generalizes to unseen |
| Learn from LLM errors | 51 | Discriminative CD: correct ↔ wrong separation learned |
| Transfer across domains | 52 | Structure-dependent transfer validated |
| Parallel sampler | infra | 183x faster than thrml (572x at 500 vars) |

**The gap:** Every experiment used **simulated LLM outputs** and **small instances** (10-15 vars for training). The pipeline is validated in pieces but never run end-to-end with a live LLM at realistic scale. Kona-like reasoning requires closing this gap.

## v7 Architecture: Live LLM + Ising Reasoning Engine

```
User Question
     │
     ▼
┌──────────────────┐
│  Live LLM        │ → Generate answer (Qwen3.5-0.8B local)
│  (Qwen/Llama)    │
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  Constraint       │ → Static: AST types, bounds, logic (Exp 48)
│  Extractor        │ → Dynamic: instrument + execute (NEW)
│  (code + NL)      │ → NL: claim patterns + KB lookup (Exp 49)
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  Ising Verifier   │ → Learned couplings (Exp 50-52, scaled up)
│  (parallel Gibbs) │ → Energy decomposition per constraint
│                    │ → Error localization (which constraint fails)
└──────────────────┘
     │
     ├── PASS → Return verified answer
     │
     ▼ FAIL
┌──────────────────┐
│  Repair Loop      │ → Feed violations back to LLM
│  (LLM + Ising)    │ → LLM regenerates with constraint feedback
│                    │ → Re-verify (max K iterations)
└──────────────────┘
     │
     ▼
Verified + Repaired Answer
```

## Phase 5: Runtime Verification (experiments 53-55)

Exp 48 extracted constraints statically from code AST. The next step: actually
run the code and verify constraints hold dynamically.

### Exp 53: Runtime constraint instrumentation
Instrument LLM-generated Python code to verify constraints during execution.
- **Input:** Python function + extracted constraints from Exp 48
- **Instrumentation:** Insert assertion checks at constraint points:
  - Type checks at function boundaries (`isinstance` guards)
  - Bound checks inside loops (`assert 0 <= i < n`)
  - Return type checks before each `return`
  - Variable initialization tracking
- **Execution:** Run instrumented code in Firecracker sandbox
- **Output:** Per-constraint pass/fail with execution trace
- **Comparison:** Static (Exp 48) vs dynamic — which catches more bugs?

### Exp 54: Differential testing via Ising-guided fuzzing
Use the Ising energy landscape to generate adversarial test inputs.
- **Input:** LLM-generated function + reference implementation
- **Ising encoding:** Input space as spins, constraint violations as energy
- **Sampling:** ParallelIsingSampler generates low-energy inputs (likely to expose bugs)
- **Verification:** Compare outputs of LLM function vs reference on Ising-selected inputs
- **Metric:** Bug detection rate vs random fuzzing vs AFL-style coverage fuzzing

### Exp 55: Dynamic constraint learning from execution traces
Instead of hand-coded constraint types, learn constraints from execution traces.
- **Input:** 1000 execution traces of correct code + 1000 traces of buggy code
- **Feature extraction:** Per-line variable types, values, branch decisions
- **Training:** Discriminative CD (like Exp 51) on trace features
- **Output:** Learned Ising model that distinguishes correct from buggy execution
- **Validation:** Does the learned model catch bugs that static analysis misses?

## Phase 6: Live LLM End-to-End Pipeline (experiments 56-59)

Connect a real LLM to the constraint pipeline. No more simulated outputs.

### Exp 56: Live LLM → constraint extraction → Ising verification
Full pipeline with Qwen3.5-0.8B (or similar small model):
- LLM answers 50 questions across domains: arithmetic, logic, code, factual
- Constraint extractor (Exp 47-49 combined) processes each answer
- Ising verifier (parallel sampler) checks constraints
- **Metric:** Detection rate on LLM's actual hallucinations vs ground truth
- **Baseline:** Compare to logprob rejection sampling (Exp 13, +10%)

### Exp 57: Live LLM verify-and-repair loop
Extend Exp 56 with iterative repair:
- When Ising finds violations, format them as natural language feedback
- Feed violations back to LLM: "Your answer violates constraint X because Y"
- LLM regenerates with constraint context in prompt
- Re-verify, up to 3 iterations
- **Metric:** Answer quality improvement after repair vs one-shot
- **Key insight from Kona:** The repair loop is where EBMs add value — not as
  classifiers (which failed, Exp 1-38) but as reasoning constraints that guide
  the LLM toward correct answers

### Exp 58: Multi-domain live pipeline benchmark
Run Exp 57 across 5 domains with 100 questions each:
- Arithmetic word problems (GSM8K-style)
- Code generation (HumanEval-style)
- Logical reasoning (syllogisms, entailment)
- Factual QA (TruthfulQA subset)
- Scheduling/planning problems
- **Metrics per domain:** accuracy, hallucination rate, repair success rate,
  average Ising energy, constraint coverage
- **Comparison:** LLM alone vs LLM + Ising verify vs LLM + Ising verify-repair

### Exp 59: Constraint-aware prompting
Instead of verify-then-repair (post-hoc), inject constraints INTO the prompt:
- Extract constraint templates for each domain
- Include constraints in the system prompt: "Your answer must satisfy: ..."
- Compare: constraint-aware prompting vs post-hoc verification
- Hypothesis: prompting prevents some hallucinations; verification catches the rest
- **Combined pipeline:** Constraint-aware prompt → generate → verify → repair

## Phase 7: Scale Learned Ising Models (experiments 60-63)

Exp 50-52 used toy instances (10-15 vars). Scale to real problem sizes.

### Exp 60: Scale CD training to 100+ variables
- Train on 3-SAT with 100 vars (4260 clauses at phase transition)
- Generate training data: 10,000 satisfying assignments via parallel sampler
- Train CD for 500 epochs on GPU (if ROCm works) or CPU
- **Key challenge:** Coupling matrix is 100×100 = 10K parameters. Regularization
  (L1 sparsity, spectral norm) may be needed to prevent overfitting
- **Metric:** Generalization to unseen 100-var instances

### Exp 61: Scale to 500+ variables with sparse Ising
Full coupling matrix for 500 vars is 500×500 = 250K parameters — too many to
learn from limited data. Use sparse Ising:
- **Approach 1:** Only learn couplings for edges in the SAT clause graph
  (sparse, typically 3 edges per clause × ~2130 clauses = ~6390 parameters)
- **Approach 2:** Low-rank factorization J = V V^T where V is (500 × k)
- **Approach 3:** Graph neural network predicts J from clause structure
- **Metric:** Solution quality vs hand-coded SAT-to-Ising encoding (Exp 46b)

### Exp 62: Learn domain-specific constraint structure at scale
Extend Exp 51 to learn from large corpora of (correct, wrong) LLM outputs:
- Collect 10,000 (question, correct_answer, wrong_answer) triples
  from arithmetic, code, and logic domains
- Binary feature encoding: 200+ features per answer
- Train discriminative Ising model (200 vars, sparse)
- **Metric:** AUROC for correct/wrong discrimination on held-out data
- **Comparison:** Learned Ising vs hand-coded constraints vs logprob baseline

### Exp 63: Hierarchical Ising for compositional reasoning
Kona's key insight: reasoning is compositional. A 1000-variable problem is
not a flat energy landscape — it decomposes into sub-problems.
- **Hierarchical encoding:** Group spins into blocks (e.g., per-clause for SAT,
  per-bit-position for arithmetic, per-variable for code constraints)
- **Block coupling:** Learn inter-block couplings (sparse) + intra-block
  couplings (dense but small)
- **Hierarchical sampling:** Sample blocks first, then refine within blocks
  (multi-scale Gibbs, similar to thrml's SuperBlock concept)
- **Metric:** Quality and speed vs flat Ising at 1000+ vars

## Phase 8: Toward Continuous Latent Reasoning (experiments 64-66)

Bridge from discrete Ising spins to continuous latent space (the Kona direction).

### Exp 64: Continuous relaxation of Ising constraints
Replace binary spins {0,1} with continuous variables [0,1]:
- Ising energy E(s) = -β(b^T s + s^T J s) works for s ∈ [0,1]^n
- Gradient descent on continuous relaxation → round to binary
- Compare: continuous relaxation vs parallel Gibbs on same problems
- **Why this matters:** Continuous space enables gradient-based reasoning
  (Kona-style) instead of sampling-based (thrml-style)

### Exp 65: Embedding-space constraint verification
Combine v6 Ising constraints with v5 embedding-space reasoning:
- Encode LLM answer as sentence embedding (all-MiniLM-L6-v2, 384-dim)
- Concatenate with constraint satisfaction vector (from Ising verifier)
- Train a small Gibbs EBM on the joint space (embedding + constraints)
- The EBM learns: "low energy = correct answer that satisfies constraints"
- **Repair:** Gradient descent in joint space optimizes both meaning and constraints

### Exp 66: End-to-end differentiable constraint reasoning
The full Kona-like pipeline, differentiable end-to-end:
- LLM generates logits → soft token probabilities → embedding
- Embedding → continuous Ising constraints (Exp 64) → energy
- Backpropagate energy gradient through constraints to logits
- Adjust LLM sampling distribution toward constraint-satisfying tokens
- **This is the holy grail:** Energy-guided decoding where constraints
  steer generation in real-time, not post-hoc

## Implementation Priority

For overnight research-loop runs:

1. **Exp 56: Live LLM pipeline** — validates everything end-to-end
2. **Exp 57: Verify-repair loop** — the core Kona value proposition
3. **Exp 53: Runtime instrumentation** — catches bugs static can't
4. **Exp 60: Scale CD to 100 vars** — proves learning scales
5. **Exp 58: Multi-domain benchmark** — publishable results
6. **Exp 61: Sparse Ising at 500+ vars** — proves learning at scale
7. **Exp 64: Continuous relaxation** — bridge to Kona
8. **Exp 59: Constraint-aware prompting** — easy win
9. **Exp 54: Ising-guided fuzzing** — novel application
10. **Exp 55: Learn from traces** — ambitious but high upside

Experiments 56-57 are the critical path. Everything else can run in parallel.

## Dependencies

```
Exp 53 (runtime) ← Exp 48 (static)
Exp 54 (fuzzing) ← Exp 53 (runtime)
Exp 55 (traces) ← Exp 53 (runtime) + Exp 51 (discriminative CD)
Exp 56 (live LLM) ← Exp 47-49 (constraint extraction)
Exp 57 (repair) ← Exp 56 (live pipeline)
Exp 58 (benchmark) ← Exp 57 (repair loop)
Exp 59 (prompting) ← Exp 56 (live pipeline)
Exp 60 (scale 100) ← Exp 50 (CD training)
Exp 61 (scale 500) ← Exp 60 (scale 100)
Exp 62 (learn domain) ← Exp 51 (discriminative CD) + Exp 60 (scale)
Exp 63 (hierarchical) ← Exp 61 (sparse) + thrml SuperBlock
Exp 64 (continuous) ← Exp 60 (scale)
Exp 65 (embedding) ← Exp 64 (continuous) + embedding infrastructure
Exp 66 (differentiable) ← Exp 64 + Exp 65 + live LLM (Exp 56)
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 53-55 | CPU | 2GB | 5-15 min each |
| 56-57 | CPU + Qwen3.5 (ROCm) | 4GB | 15-30 min |
| 58 | CPU + Qwen3.5 | 4GB | 1-2 hours |
| 59 | CPU + Qwen3.5 | 4GB | 30 min |
| 60 | CPU (GPU if ROCm fixed) | 2GB | 30 min |
| 61 | GPU recommended | 4GB | 1 hour |
| 62 | CPU + Qwen3.5 | 8GB | 2 hours |
| 63 | GPU recommended | 4GB | 1 hour |
| 64-66 | GPU + Qwen3.5 | 8GB | 2-4 hours |
