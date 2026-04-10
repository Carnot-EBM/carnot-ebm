# Carnot Research Roadmap v11: Agentic Verification — Multi-Turn Reasoning, Factual Grounding, Energy-Guided Decoding

**Created:** 2026-04-10
**Milestone:** 2026.04.6
**Status:** Planned (activates when milestone 2026.04.5 completes)
**Supersedes:** research-roadmap-v10.md (milestone 2026.04.5)
**Informed by:** Experiments 66–95, GSM8K +14-15%, multi-model +10.2%, 93% false negative rate, gradient repair (40%), self-bootstrap (0.788 AUROC)
**Target Models:** Qwen3.5-0.8B, google/gemma-4-E4B-it

## What v10 Proved

| Approach | Experiments | Finding |
|----------|------------|---------|
| Differentiable constraint reasoning | 66 | Joint model 1.0 test AUROC; Ising adds discriminative power beyond embeddings |
| Learned energy weights | 86 | Marginal (+1.1%), not statistically significant; arithmetic weight dominant (1.19) |
| Gradient repair | 87 | 40% success; arithmetic/scheduling 100%, factual/code/logic 0% |
| Failure-driven mining | 88 | 93% false negative rate; implicit_logic (74), comparison (40), arithmetic_chain (23) top gaps |
| Self-bootstrapped training | 89 | 0.788 combined AUROC; arithmetic/logic perfect (1.0), factual/scheduling ~0.5 |
| Autoresearch improvement | 90 | 17/20 accepted (85%); coverage improves but AUROC plateau — needs richer signal |
| GSM8K live benchmark | 91 | Qwen3.5: +15%, Gemma4: +14%; 100% precision, zero false positives |
| MATH step-level | 92 | Step-by-step verification catches intermediate errors |
| Multi-model comparison | 93 | +10.2% avg improvement (p<0.001); scheduling +30%, code +14% |
| Rust pipeline | 94 | VerifyPipeline + AutoExtractor in Rust; 10x verification target |
| PyO3 bridge | 95 | Partial — REVERT on first attempt, deliverable exists but needs validation |

**The 3 gaps this milestone addresses:**

### Gap 1: Multi-Turn Agentic Verification (research-program.md Goal #2 — BIGGEST UNTAPPED OPPORTUNITY)

Verification is single-turn: one question, one answer, one verify call. But real AI agents execute multi-step workflows: plan → act → observe → reason → act again. Each step's output becomes the next step's input. A constraint violation in step 2 may invalidate steps 3-5. No one else is building constraint propagation across agent reasoning chains. This is Carnot's most differentiating potential feature.

**What's needed:**
- Constraint state that accumulates across turns (verified facts vs assumptions)
- Constraint propagation: step N's output constraints become step N+1's input constraints
- Rollback semantics: when step K fails, identify which earlier step to repair
- Integration with the existing VerifyRepairPipeline API

### Gap 2: Factual Constraint Extraction (research-program.md Goal #3 — CLOSES BIGGEST COVERAGE GAP)

Exp 88 revealed a 93% false negative rate. The top 3 uncovered claim types — implicit_logic (74 instances), comparison (40), arithmetic_chain (23) — account for the vast majority of missed hallucinations. Factual and scheduling domains have near-zero constraint coverage. The pipeline can't verify what it can't extract. The `intermediate_result` extractor alone would catch an estimated 44.8% of current false negatives.

**What's needed:**
- Intermediate result extractor (arithmetic chains like "first compute X, then Y")
- Comparison extractor ("X is greater/less/equal to Y")
- Knowledge-base-backed factual claim verifier (Wikidata/Wikipedia lookups)
- Integration with AutoExtractor's pluggable protocol

### Gap 3: Guided Decoding Viability (research-program.md Goals #4, #7, #9 — QUICK WIN + NEW ARCHITECTURE)

The differentiable pipeline works (Exp 66, 1.0 AUROC) but we don't know if constraint checks are fast enough for per-token use during generation. A single latency benchmark answers this. If <1ms → energy-guided decoding is viable (the Kona path). If >10ms → stick with post-hoc verify-repair. Additionally, the KAN energy tier (Goal #7) addresses the constraint learning ceiling from Exp 62/88 where linear Ising features can't capture nonlinear relationships.

**What's needed:**
- Constraint check latency microbenchmark (Exp 66 forward pass timing)
- KAN energy tier prototype (spline-based energy between Ising and Gibbs)
- Energy-guided token sampling prototype (if latency permits)

## v11 Architecture: Agentic Verification Pipeline

```
                         Agent Workflow
                     ┌─────────────────┐
                     │  Step 1: Plan    │
                     │  constraints: C₁ │
                     └────────┬────────┘
                              │ propagate C₁
                              ▼
                     ┌─────────────────┐
                     │  Step 2: Act     │
                     │  verify: C₁ ∧ C₂│ ← accumulated constraints
                     └────────┬────────┘
                              │ propagate C₁ ∧ C₂
                              ▼
                     ┌─────────────────┐
                     │  Step 3: Observe │
                     │  verify: C₁∧C₂∧C₃│
                     └────────┬────────┘
                              │ violation detected at C₂!
                              ▼
                     ┌─────────────────┐
                     │  Rollback to     │ ← identify root cause step
                     │  Step 2, repair  │ ← re-verify chain from step 2
                     └─────────────────┘

    New Extractors                    New Energy Tier
┌──────────────────┐             ┌──────────────────┐
│ IntermediateResult│             │  KAN Energy       │
│ Comparison        │             │  (spline-based)   │
│ FactualKB         │             │  Ising < KAN < MLP│
└────────┬─────────┘             └────────┬─────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐             ┌──────────────────┐
│ AutoExtractor     │             │ Differentiable    │
│ (expanded)        │             │ Pipeline (Exp 66) │
└────────┬─────────┘             └────────┬─────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
             ┌──────────────────┐
             │ Latency Benchmark │ ← Is per-token viable?
             │ → Guided Decoding │ ← If yes: energy-steered generation
             └──────────────────┘
```

## Phase 21: Expand Constraint Coverage (experiments 96, 97, 98)

Close the 93% false negative gap by adding the three most impactful missing extractors identified by Exp 88's failure analysis.

### Exp 96: Intermediate result extractor
The single highest-impact addition. Exp 88 showed `intermediate_result` patterns (e.g., "first compute 3×5=15, then add 7") would catch 44.8% of current false negatives. These are arithmetic chains where the final answer depends on intermediate computations that can each be independently verified.

- Add `IntermediateResultExtractor` to `carnot.pipeline.extract`
- Pattern: detect "first/then/next/so/therefore" + arithmetic expression chains
- Verify each intermediate step independently
- Chain verification: if step K is wrong, all subsequent steps are suspect
- **Metric:** False negative reduction on Exp 88 failure set
- **Deliverable:** `python/carnot/pipeline/extract.py` (extended)

### Exp 97: Comparison constraint extractor
Second highest gap (40 instances in Exp 88). Handles claims like "X is greater than Y", "A has more than B", "temperature increased from X to Y".

- Add `ComparisonExtractor` to `carnot.pipeline.extract`
- Patterns: "greater/less/more/fewer/higher/lower than", "increased/decreased", "X vs Y"
- Extract numeric values and comparison operator
- Encode as Ising constraint: spin configuration satisfying the inequality
- **Metric:** Comparison claim detection rate on held-out test set
- **Deliverable:** `python/carnot/pipeline/extract.py` (extended)

### Exp 98: Knowledge-base factual claim verifier
Fundamentally different from pattern-based extractors. Verifies factual claims against external knowledge. This addresses the "definitive finding" that you cannot detect factual hallucination without access to factual knowledge.

- Create `FactualKBExtractor` in `carnot.pipeline.extract`
- Approach: extract entity-relation-value triples from text via regex/spaCy patterns
- Verify against a local knowledge base (precomputed JSON of common facts)
- Start small: 1000 common facts (capitals, populations, dates, scientific constants)
- For unknown facts: return "unverifiable" (don't hallucinate verification!)
- Encode verified/unverified/contradicted as constraint energy terms
- **Metric:** Factual claim detection precision/recall on TruthfulQA-style questions
- **Deliverable:** `python/carnot/pipeline/extract.py` (extended) + `python/carnot/pipeline/knowledge_base.py`

## Phase 22: Multi-Turn Agentic Verification (experiments 99, 100, 101)

Build constraint propagation across multi-step agent reasoning chains. This is the most differentiating feature Carnot could offer — no existing tool does this.

### Exp 99: Constraint state propagation
Core primitive: a `ConstraintState` that accumulates verified facts across turns and propagates output constraints as input constraints to the next step.

- Create `python/carnot/pipeline/agentic.py` with:
  - `ConstraintState`: accumulator for verified/assumed/violated facts
  - `AgentStep`: (input, output, constraints, verification_result)
  - `propagate(state, step) -> ConstraintState`: merge new step's constraints into state
- State tracks: which facts are verified (low energy), assumed (not yet checked), violated (high energy)
- Output constraints of step N automatically become input constraints of step N+1
- **Metric:** Correct constraint accumulation across 5-step synthetic agent traces
- **Deliverable:** `python/carnot/pipeline/agentic.py`

### Exp 100: Multi-step verification with rollback
When a constraint violation is detected at step K, trace back through the chain to find the root cause step and repair from there.

- Extend `ConstraintState` with causal dependency tracking:
  - Each constraint records which step introduced it
  - When a violation is found, trace the dependency chain backward
  - Identify the earliest step whose output constraint is violated
- `AgentVerifier` class:
  - `verify_chain(steps: list[AgentStep]) -> ChainVerificationResult`
  - `rollback_to(step_index) -> ConstraintState` (reset state to before the bad step)
  - `repair_from(step_index, chain) -> list[AgentStep]` (re-run from the bad step)
- Test on 20 synthetic 5-step reasoning chains (10 correct, 10 with planted errors at various steps)
- **Metric:** Root cause identification accuracy, repair success rate
- **Deliverable:** `python/carnot/pipeline/agentic.py` (extended)

### Exp 101: Agent workflow verification end-to-end
Integration test: verify a realistic agent workflow (plan → search → reason → answer) with full constraint propagation.

- Define 3 agent workflow templates:
  1. Math tutor: read problem → identify approach → compute steps → verify answer
  2. Code assistant: understand spec → write code → test → fix bugs
  3. Research assistant: formulate query → gather facts → synthesize → cite sources
- For each template, generate 10 workflows (5 correct, 5 with planted errors)
- Run through AgentVerifier with full constraint propagation
- Compare: per-step verification vs final-answer-only verification
- Measure: error detection latency (which step first flags the problem?)
- **Metric:** Error detection rate, detection step vs actual error step, false positive rate
- **Deliverable:** `scripts/experiment_101_agent_verification.py`

## Phase 23: KAN Energy Tier + Guided Decoding (experiments 102, 103, 104)

New energy architecture and the critical latency benchmark that determines whether energy-guided decoding is viable.

### Exp 102: Constraint check latency microbenchmark
The quick experiment that answers Goal #4. Time the Exp 66 differentiable forward pass at various scales to determine if per-token constraint checking is viable during generation.

- Benchmark the full differentiable pipeline forward pass:
  - Embedding computation (sentence-transformers)
  - Constraint extraction (AutoExtractor)
  - Continuous Ising energy computation (JAX)
  - Joint MLP scoring
- Measure at different input sizes: 10, 50, 100, 500, 1000 tokens
- Measure with different constraint counts: 1, 5, 10, 50 constraints
- Compare: Python pipeline vs Rust pipeline (Exp 94) vs JAX JIT-compiled
- **Key thresholds:** <1ms → viable for guided decoding, 1-10ms → batch-viable, >10ms → post-hoc only
- **Deliverable:** `scripts/experiment_102_latency_benchmark.py`

### Exp 103: KAN energy tier prototype
Kolmogorov-Arnold Network as a new energy function tier. Spline-based learnable edge activations — strictly more expressive than Ising (quadratic) while remaining interpretable. Addresses the constraint learning ceiling from Exp 62/88.

- Create `python/carnot/models/kan.py`:
  - `KANEnergyFunction`: implements EnergyFunction protocol
  - Edge activations: B-spline basis functions (learnable control points)
  - Energy: sum of spline(s_i, s_j) over edges (generalizes -J_ij × s_i × s_j)
  - Training: CD with spline parameter gradients via JAX
  - Interpretability: plot learned spline shapes to visualize constraint relationships
- Create `crates/carnot-kan/` Rust crate (scaffold only):
  - `KanEnergyFunction` struct implementing `EnergyFunction` trait
  - B-spline evaluation in Rust (lookup table for FPGA path)
- Train KAN on Exp 62's domain-specific constraint data (10K triples)
- Compare: KAN AUROC vs Ising AUROC vs Gibbs MLP AUROC
- **Metric:** Verification AUROC improvement over Ising, parameter count, interpretability
- **Deliverable:** `python/carnot/models/kan.py` + `crates/carnot-kan/src/lib.rs`

### Exp 104: Energy-guided token sampling prototype
If Exp 102 shows <10ms latency: build the first prototype of energy-steered LLM generation. Constraint energy modifies token probabilities during generation, steering the LLM away from constraint-violating continuations.

- Create `python/carnot/inference/guided_decoding.py`:
  - `EnergyGuidedSampler`: wraps a HuggingFace model's generate()
  - At each token position: compute partial constraint energy on tokens so far
  - Modify logits: subtract α × ∂E/∂logit (energy gradient steers away from violations)
  - α (guidance strength) as tunable hyperparameter
  - Fallback: if constraint check too slow, batch every K tokens instead of every token
- Test on 50 GSM8K questions with Qwen3.5-0.8B:
  - Compare: greedy vs energy-guided vs post-hoc verify-repair
  - Measure: accuracy, latency overhead, constraint satisfaction rate
- If Exp 102 shows >10ms: document findings and recommend post-hoc approach
- **Metric:** Accuracy improvement from guided decoding, latency overhead per token
- **Deliverable:** `python/carnot/inference/guided_decoding.py`

## Phase 24: Scale Benchmarks + Integration (experiments 105, 106, 107)

With improved extractors and new energy tier, re-run benchmarks at full scale.

### Exp 105: Full-scale benchmark with improved extractors
Re-run the multi-model comparison (Exp 93) with the new extractors from Phase 21. Measure the actual false negative reduction.

- Same setup as Exp 93: 250 questions × 2 models × 3 modes = 1500 evals
- But with expanded AutoExtractor (intermediate result + comparison + factual KB)
- Compare v10 extractors vs v11 extractors: false negative reduction
- Per-domain improvement: expect biggest gains in factual and scheduling
- Statistical significance: bootstrap 95% CI on accuracy difference
- **Metric:** False negative reduction, overall accuracy improvement, per-domain breakdown
- **Deliverable:** `scripts/experiment_105_improved_benchmark.py`

### Exp 106: TruthfulQA benchmark with factual extractor
First benchmark specifically targeting factual hallucination detection — the hardest domain. Uses the new FactualKBExtractor from Exp 98.

- Load TruthfulQA dataset (or create 100 synthetic TruthfulQA-style questions)
- Generate answers with Qwen3.5-0.8B and Gemma4-E4B-it
- Run through pipeline with factual KB extractor
- Compare: baseline vs verify-only vs verify-repair
- Breakdown: which factual categories are best/worst covered?
- **Metric:** Factual hallucination detection rate, false positive rate, knowledge base hit rate
- **Deliverable:** `scripts/experiment_106_truthfulqa.py`

### Exp 107: HuggingFace model card + Exp 66 joint model publish
Publish the Exp 66 differentiable constraint model to HuggingFace as a proof-of-concept artifact. This is the first publishable HuggingFace artifact from the constraint-verification era (vs the earlier activation-based models).

- Package the Exp 66 joint model (embedding + constraints + Ising + MLP):
  - Save weights via safetensors
  - Create model card with: approach description, AUROC numbers, usage example
  - Clear disclaimer: "research proof-of-concept, not production quality"
- Create `scripts/publish_constraint_model.py` for reproducible publishing
- Update existing HuggingFace model READMEs to clarify Phase 1 vs Phase 5+ distinction
- **Deliverable:** `scripts/publish_constraint_model.py` + model card template

## Dependencies

```
2026.04.5 outputs (done):
  Exp 66 ✓ (differentiable pipeline, 1.0 AUROC)
  Exp 87 ✓ (gradient repair, 40%)
  Exp 88 ✓ (failure mining, 93% FN rate)
  Exp 89 ✓ (self-bootstrap, 0.788 AUROC)
  Exp 93 ✓ (multi-model comparison, +10.2%)
  Exp 94 ✓ (Rust pipeline)

Phase 21 (extractors):
  Exp 96 ← Exp 88 (failure analysis: intermediate_result is top gap)
  Exp 97 ← Exp 88 (failure analysis: comparison is second gap)
  Exp 98 ← Exp 88 (failure analysis: world_knowledge gap) + new KB approach

Phase 22 (agentic):
  Exp 99 ← Exp 75 (VerifyRepairPipeline) + Exp 92 (step-level verification)
  Exp 100 ← Exp 99 (constraint state)
  Exp 101 ← Exp 100 (rollback) + Phase 21 extractors

Phase 23 (KAN + guided decoding):
  Exp 102 ← Exp 66 (differentiable pipeline) + Exp 94 (Rust pipeline)
  Exp 103 ← carnot-core traits + Exp 62 (domain constraint data)
  Exp 104 ← Exp 102 (latency result) + Exp 103 (KAN energy, optional)

Phase 24 (scale):
  Exp 105 ← Phase 21 extractors + Exp 93 (benchmark setup)
  Exp 106 ← Exp 98 (factual KB extractor)
  Exp 107 ← Exp 66 (joint model) + HuggingFace publishing
```

## Execution Order

```
1.  exp96   -- Intermediate result extractor (highest impact, no deps)
2.  exp97   -- Comparison constraint extractor (second highest impact)
3.  exp98   -- Knowledge-base factual claim verifier (new approach)
4.  exp99   -- Constraint state propagation (core agentic primitive)
5.  exp100  -- Multi-step verification with rollback (builds on exp99)
6.  exp102  -- Constraint check latency benchmark (quick, high signal)
7.  exp103  -- KAN energy tier prototype (new architecture)
8.  exp101  -- Agent workflow verification E2E (needs exp100 + Phase 21)
9.  exp104  -- Energy-guided token sampling (depends on exp102 result)
10. exp105  -- Full-scale benchmark with improved extractors
11. exp106  -- TruthfulQA benchmark with factual extractor
12. exp107  -- HuggingFace model card + publish
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 96 | CPU | 2GB | 15-30 min |
| 97 | CPU | 2GB | 15-30 min |
| 98 | CPU | 2GB | 30-60 min |
| 99 | CPU + JAX | 2GB | 30 min |
| 100 | CPU + JAX | 2GB | 30 min |
| 101 | CPU + JAX | 4GB | 1 hour |
| 102 | CPU + JAX + Rust | 4GB | 15 min |
| 103 | CPU + JAX | 4GB | 1-2 hours |
| 104 | CPU + Qwen3.5-0.8B | 4GB | 1-2 hours |
| 105 | CPU + Qwen3.5 + Gemma4 | 8GB | 2-3 hours |
| 106 | CPU + Qwen3.5 + Gemma4 | 8GB | 1-2 hours |
| 107 | CPU | 2GB | 30 min |

## Success Criteria

- Intermediate result + comparison extractors reduce false negatives by ≥30% (from 93% baseline)
- Factual KB extractor achieves ≥60% precision on factual claims (better than 0% current)
- Multi-step constraint propagation correctly accumulates state across ≥5 steps
- Rollback identifies root cause step in ≥70% of planted-error chains
- Constraint check latency measured and reported (threshold: <1ms for guided, <10ms for batch)
- KAN AUROC ≥ Ising AUROC on Exp 62 constraint data (demonstrating expressiveness gain)
- If guided decoding viable: ≥3% accuracy improvement on GSM8K over post-hoc only
- Full-scale benchmark with v11 extractors shows measurable improvement over v10 numbers
- TruthfulQA factual detection rate ≥40% (vs current ~0%)
- Exp 66 joint model published to HuggingFace with model card

## What's Explicitly NOT in Scope

- **LNN (Liquid Neural Networks)** — pairs with agentic verification but adds too much complexity this milestone; consider for v12 once agentic primitives are proven
- **FPGA Ising machine** — no FPGA hardware available; SamplerBackend abstraction is ready when hardware arrives
- **Full Kona implementation** — guided decoding prototype is a stepping stone, not the full vision
- **Activation-based EBMs** — proven insufficient (experiments 8-38), permanently retired
- **PyPI publish** — beta release is ready (Exp 85), actual publish is a release ops task
- **Mamba/RWKV constraint propagation** — consider after agentic primitives work with simpler state
