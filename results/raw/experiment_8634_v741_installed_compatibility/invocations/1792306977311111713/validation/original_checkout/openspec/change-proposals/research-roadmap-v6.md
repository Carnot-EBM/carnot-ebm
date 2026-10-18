# Carnot Research Roadmap v6: Constraint-Based Reasoning via Ising/thrml

**Created:** 2026-04-08
**Status:** Active
**Supersedes:** research-roadmap-v5.md (proven wrong by experiments 36-41)
**Informed by:** Experiments 36-41, Extropic thrml integration, Kona architecture

## What We Proved Doesn't Work (experiments 1-38)

| Approach | Experiments | Finding |
|----------|------------|---------|
| Activation-based EBMs | 1-31 | Detects confidence, not correctness (50% practical) |
| Logit lens dynamics | 36 | 50.6% = chance. Dynamics identical for correct/wrong |
| Sentence embeddings | 37 | Embed topic similarity, not factual truth |
| NLI embeddings | 38 | Embed text consistency, not factual knowledge |
| Cross-model transfer | 26 | 50% = chance. Representations are model-specific |
| Cross-domain transfer | 31 | Mixing domains hurts (70.8% < 75.5%) |
| Normalization | 35 | Destroys signal without enabling transfer |

**The definitive finding:** You cannot detect factual hallucination from model internals alone. No internal signal — activations, dynamics, embeddings, confidence — distinguishes "Neil Armstrong walked on Mars" from "Neil Armstrong walked on the Moon." You need external verification.

## What Works (experiments 39-41)

| Approach | Result | Hardware |
|----------|--------|----------|
| SAT → Ising → thrml sampling | Beats random at 50+ vars | Extropic TSU |
| Graph coloring → Ising → thrml | Perfect solutions on 3/6 problems | Extropic TSU |
| LLM → Ising verify → repair | 2/6 hallucinations caught and fixed (0%→100%) | Extropic TSU |
| Logprob rejection sampling | +10% accuracy, zero training | CPU/GPU |
| Composite scoring (logprob + tests) | 0% → 30% for code | CPU/GPU |

## The Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  LLM        │ → Natural language response + constraint extraction
│  (language)  │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Constraint  │ → Formal constraints (SAT, graph coloring, types, arithmetic)
│  Parser      │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Ising       │ → Energy function over binary spins
│  Encoder     │   E(s) = -β(Σ b_i s_i + Σ J_ij s_i s_j)
└─────────────┘
     │
     ▼
┌─────────────┐
│  thrml /     │ → Sample low-energy configurations
│  Extropic TSU│   (Block Gibbs / thermodynamic hardware)
└─────────────┘
     │
     ▼
┌─────────────┐
│  Decoder     │ → Verified/repaired answer in natural language
│  (solution)  │
└─────────────┘
```

The LLM handles language. The Ising model handles reasoning. No hallucination is possible in the constraint layer — it's pure math. On Extropic hardware, the sampling runs in nanoseconds.

## Phase 1: Scale Constraint Types (experiments 42-46)

Expand beyond SAT and graph coloring to cover more constraint domains.

### Exp 42: Arithmetic constraint verification
Encode arithmetic claims as Ising constraints. "7 × 8 = 56" → binary multiplication circuit → Ising spins. Verify or find the correct answer.
- **Input:** LLM's arithmetic answer
- **Ising encoding:** binary adder/multiplier circuits (well-known QUBO encodings)
- **Validation:** does thrml sampling find the correct arithmetic result?

### Exp 43: Type constraint verification
Encode type-checking constraints as Ising model. Function signature + call site → type compatibility as binary constraints.
- **Input:** LLM-generated code with type annotations
- **Ising encoding:** type compatibility matrix → spin couplings
- **Validation:** does thrml find type violations the LLM introduced?

### Exp 44: Scheduling constraint verification
Encode scheduling problems (no conflicts, resource limits) as Ising model.
- **Input:** LLM-generated schedule
- **Ising encoding:** time slots as spins, conflicts as anti-couplings
- **Validation:** does thrml find valid schedules when LLM fails?

### Exp 45: Logical consistency verification
Encode logical implications as Ising constraints. If LLM says "A implies B" and "A is true" and "B is false" → contradiction detectable as high energy.
- **Input:** LLM's chain of logical claims
- **Ising encoding:** propositions as spins, implications as couplings
- **Validation:** does thrml detect logical contradictions?

### Exp 46: Scale test — 1000+ variable SAT
Push thrml SAT solver to large instances where the advantage over random is dramatic.
- Test at 100, 200, 500, 1000 variables
- Measure: time to solution, quality vs random, quality vs dpll/cdcl solvers
- Profile: where does thrml's block Gibbs sampling bottleneck?

## Phase 2: LLM Constraint Extraction (experiments 47-49)

Automate the constraint extraction — have the LLM itself identify what needs verifying.

### Exp 47: LLM self-constraint extraction
Prompt the LLM: "What verifiable constraints does your answer satisfy?"
The LLM generates both the answer AND the constraints. Ising verifies.
- Tests whether LLMs can identify their own verification criteria
- If yes: fully automated verify/repair pipeline

### Exp 48: Code → constraint extraction
Parse LLM-generated code into constraint graphs automatically.
- Variable types → type constraints
- Loop bounds → arithmetic constraints
- Function contracts → logical constraints
- Test coverage → SAT constraints (which inputs trigger which branches)

### Exp 49: Natural language → constraint extraction
Use NLI or entailment to extract verifiable claims from free text.
"The capital of France is Paris" → lookup constraint against knowledge base.
This bridges the gap between structured reasoning (Ising) and free-form text.

## Phase 3: thrml Training (experiments 50-52)

Learn Ising parameters from data instead of hand-coding constraints.

### Exp 50: Learn SAT couplings from satisfying assignments
Use thrml's `estimate_kl_grad` to learn Ising parameters that reproduce a distribution of satisfying assignments. The learned couplings should generalize to unseen instances.

### Exp 51: Learn constraint structure from LLM outputs
Given a corpus of (correct answer, wrong answer) pairs for a domain, learn Ising couplings that assign low energy to correct and high energy to wrong. This is training, but on Ising-compatible (binary) features rather than continuous activations.

### Exp 52: Transfer learned Ising models across domains
Do Ising models learned on one domain transfer to another? (Unlike activation EBMs which showed 50% transfer, Ising constraints might transfer because they encode structural rules, not statistical patterns.)

## Phase 4: Extropic Hardware Integration

### When TSU is available:
- Run all thrml code natively on hardware
- Benchmark: sampling speed, energy consumption, quality vs software simulation
- Scale to problems impossible in software (10,000+ spin Ising models)

### Hardware compilation:
- Compile learned Ising parameters to TSU configuration
- Map constraint types to hardware-optimized Ising circuits
- Build the LLM → TSU → solution pipeline end-to-end

## Autoresearch Tasks (for tonight's run)

Priority order for unattended execution:

1. **Exp 42: Arithmetic constraints** — small, fast, validates a new constraint type
2. **Exp 46: Scale SAT to 500+ vars** — shows thrml's advantage at scale
3. **Exp 45: Logical consistency** — most useful for LLM output verification
4. **Exp 47: LLM self-constraint extraction** — tests if pipeline can be automated
5. **Exp 44: Scheduling constraints** — practical use case

Each experiment should:
- Run independently (no dependencies between experiments)
- Save results to `data/` and commit to git
- Push to both origin and github remotes
- Take <30 minutes each on current hardware
