# Carnot Research Roadmap v12: Production Energy-Guided Reasoning + Factual Grounding

**Created:** 2026-04-10
**Milestone:** 2026.04.7
**Status:** Planned (activates when milestone 2026.04.6 completes)
**Supersedes:** research-roadmap-v11.md (milestone 2026.04.6)
**Informed by:** Experiments 66-107, 60% agentic detection, 0.008ms JAX JIT latency, +10.2% multi-model improvement, 93% FN rate, Exp 103/104 failed (rate limits), Exp 98/105/106 skipped (rate limits)
**Target Models:** Qwen3.5-0.8B, google/gemma-4-E4B-it

## What v11 Proved / Left Unfinished

| Approach | Experiments | Finding |
|----------|------------|---------|
| Agentic workflow verification | 101 | 60% detection; agentic chain catches 67% more errors than final-only; math 80%, code 100%, research 0% |
| Latency benchmark | 102 | JAX JIT 0.008ms mean (per-token viable); MiniLM embedding is bottleneck (7.6ms); extraction linear 0.04-2.6ms |
| Constraint state propagation | 99 | Correctly accumulates state across steps |
| Multi-step rollback | 100 | Root cause identification, rollback semantics |
| KAN energy tier | 103 | FAILED — rate limit hit during execution |
| Guided decoding prototype | 104 | FAILED — rate limit hit during execution |
| Factual KB extractor | 98 | SKIPPED — pre-tests failing, self-heal failed repeatedly |
| Full-scale benchmark | 105 | SKIPPED — rate limit hit |
| TruthfulQA benchmark | 106 | SKIPPED — rate limit hit |
| HuggingFace publish | 107 | SKIPPED — rate limit hit |

**Rate Limit Impact:** 5 of 12 experiments failed or skipped due to API rate limits. The conductor hit `claude -p` rate limits repeatedly while trying to implement KAN and guided decoding. These are complex experiments that need dedicated execution time without rate limit interruptions.

**The 3 gaps this milestone addresses:**

### Gap 1: KAN Energy Tier (Research-Program Goal #7 — COMPLETE THE FAILED EXPERIMENT)

Exp 103 failed due to rate limits. KAN (Kolmogorov-Arnold Networks) addresses the constraint learning ceiling where linear Ising features cannot capture nonlinear constraint relationships (e.g., "if X > 10 AND Y < 5 then Z must be positive"). KAN edges have learnable spline activations — strictly more expressive than Ising while remaining interpretable.

**What's needed:**
- Complete `python/carnot/models/kan.py` — KANEnergyFunction with B-spline edge activations
- Train KAN on Exp 62's domain constraint data (10K triples)
- Compare: KAN AUROC vs Ising AUROC vs Gibbs MLP AUROC
- Create `crates/carnot-kan/` Rust scaffold

### Gap 2: Factual Constraint Extraction (Research-Program Goal #3 — CLOSES BIGGEST COVERAGE GAP)

Exp 98 (FactualKBExtractor) was skipped multiple times due to pre-test failures. Self-bootstrap showed factual at 0.55 AUROC and scheduling at 0.52 AUROC — near zero. The knowledge-base-backed approach is the right direction but needs to be more robust.

**What's needed:**
- Robust FactualKBExtractor with expanded knowledge base (5000+ facts)
- Entity linking and coreference resolution for better claim extraction
- WebSearch fallback for claims not in the local KB
- Integration with AutoExtractor

### Gap 3: Production Energy-Guided Decoding + LNN Adaptive Constraints

Exp 102 proved per-token constraint checking is viable (0.008ms JAX JIT). But MiniLM embedding is the bottleneck at 7.6ms. Need: (1) complete the guided decoding prototype, (2) productionize it into the VerifyRepairPipeline, (3) add LNN-based adaptive constraints for multi-turn agent workflows where static Ising cannot adapt to new facts.

**What's needed:**
- Complete guided decoding prototype (Exp 104 — failed due to rate limits)
- Productionize into VerifyRepairPipeline with guided generation option
- LNN-based constraint model for adaptive multi-turn verification
- Full-scale benchmark with all v11+v12 extractors

## v12 Architecture: Production Energy-Guided Reasoning

```
                    Guided Decoding Pipeline
              ┌────────────────────────────────┐
              │    LLM Token Generation         │
              │  (Qwen3.5-0.8B / Gemma4)       │
              └──────────────┬─────────────────┘
                              │ partial text (every K tokens)
                              ▼
              ┌────────────────────────────────┐
              │  Constraint Extraction          │
              │  AutoExtractor (v11+v12)        │
              │  IntermediateResult            │
              │  Comparison                    │
              │  FactualKB + WebSearch         │
              └──────────────┬─────────────────┘
                              │ constraint features
                              ▼
              ┌────────────────────────────────┐
              │  Energy Computation            │
              │  KAN (spline-based) OR        │
              │  Ising (if KAN unavailable)    │
              │  JAX JIT: 0.008ms              │
              └──────────────┬─────────────────┘
                              │ energy score
                              ▼
              ┌────────────────────────────────┐
              │  Logit Modification            │
              │  α × gradient descent          │
              │  steer away from violations   │
              └──────────────┬─────────────────┘
                              │ modified logits
                              ▼
              ┌────────────────────────────────┐
              │  Token Sampling                │
              │  guided next-token selection   │
              └───────────────────────────────┘

    Factual Grounding              LNN Adaptive Constraints
┌──────────────────────────┐   ┌──────────────────────────┐
│ FactualKBExtractor       │   │ Liquid Neural Net        │
│ + WebSearch fallback     │   │ ConstraintModel          │
│ 5000+ facts, entity     │   │ adapts couplings during  │
│ linking, coreference     │   │ multi-turn execution     │
└────────────┬─────────────┘   └────────────┬─────────────┘
             │                              │
             ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ TruthfulQA ≥40% detect   │   │ Agentic workflow ≥75%   │
│ Factual AUROC ≥0.70      │   │ detection (up from 60%) │
└──────────────────────────┘   └──────────────────────────┘
```

## Phase 31: KAN Energy Tier Completion (experiments 108, 109)

Complete the failed Exp 103 — KAN energy tier prototype.

### Exp 108: KAN Energy Function Implementation

Create the KAN (Kolmogorov-Arnold Network) energy function tier — a new energy tier between Ising (quadratic) and Gibbs (MLP).

- Create `python/carnot/models/kan.py`:
  - `BSpline`: B-spline basis with learnable control points via JAX
  - `KANEnergyFunction`: implements EnergyFunction protocol
    - Edge activations: one B-spline per edge (learnable)
    - Bias splines: one B-spline per node (replaces linear bias)
    - Energy: sum of spline_ij(s_i × s_j) over edges
    - Gradients: JAX autodiff through splines
    - `n_params`: total learnable parameters
    - `from_ising()`: initialize KAN from trained Ising model (each spline starts as linear)
  - `KANModel`: wraps energy + training state, CD training via spline parameter updates
  - `interpret_edge()`: return spline shape for visualization
- Create `crates/carnot-kan/Cargo.toml` and `crates/carnot-kan/src/lib.rs`:
  - Minimal scaffold: `KanEnergyFunction` implementing `EnergyFunction` trait
  - B-spline evaluation via lookup table (for FPGA path)
  - Add to workspace Cargo.toml members
- Create `tests/python/test_kan.py` (comprehensive):
  - B-spline evaluation at knot points
  - KAN energy finite and gradient computable
  - `from_ising()` produces similar energy values
  - KAN has more parameters than Ising but fewer than Gibbs MLP
  - Every test references REQ-CORE-001 or SCENARIO-CORE-001
- **Deliverable:** `python/carnot/models/kan.py` + `crates/carnot-kan/src/lib.rs`

### Exp 109: KAN vs Ising vs Gibbs Comparison

Train KAN on the domain constraint data and compare energy tiers.

- Load Exp 62's domain constraint data (10K triples across arithmetic/logic/code)
- Train three energy models with identical data/splits:
  1. **Ising**: discriminative CD on binary features
  2. **KAN**: discriminative CD with spline edge activations
  3. **Gibbs MLP**: discriminative CD with fixed nonlinear activations
- Compare per-domain AUROC, parameter count, training time
- Interpretability: plot top-5 most important edge splines from KAN
- Ablation: initialize KAN from Ising (via `from_ising`) and fine-tune — does it converge faster?
- **Metric:** Verification AUROC, parameter efficiency, interpretability score
- **Deliverable:** `scripts/experiment_109_kan_comparison.py`

## Phase 32: Production Energy-Guided Decoding (experiments 110, 111, 112)

Complete the failed Exp 104 and productionize guided decoding into the pipeline.

### Exp 110: Guided Decoding Prototype Completion

Complete the energy-guided token sampling prototype (failed in v11 due to rate limits).

- Create `python/carnot/inference/guided_decoding.py`:
  - `EnergyGuidedSampler`:
    - `__init__(pipeline, alpha=0.5, check_every_k=1)`:
      - alpha = guidance strength (energy influence on logits)
      - check_every_k = frequency of constraint checks (1 = every token)
    - `compute_energy_penalty(text_so_far)`:
      - Run AutoExtractor + energy computation on partial text
      - Return constraint energy (higher = more violations)
    - `modify_logits(logits, text_so_far)`:
      - If energy > threshold: reduce probability of high-energy continuations
      - Use energy gradient if differentiable pipeline available
      - Return modified logits
    - `generate(prompt, model, tokenizer, max_tokens, temperature)`:
      - Token-by-token generation with energy guidance
      - At each step: get logits → modify via energy penalty → sample
  - Handle both real model inference and simulated fallback
- Create `scripts/experiment_110_guided_decoding.py`:
  - Test on 50 GSM8K-style arithmetic problems
  - Three modes: greedy baseline, guided (check_every_k=1), guided (check_every_k=5)
  - Alpha sweep: [0.1, 0.3, 0.5, 1.0, 2.0]
  - Metrics: accuracy, constraint satisfaction rate, latency overhead
- Create `tests/python/test_guided_decoding.py`:
  - Sampler initialization, energy penalty computation, logit modification
  - Every test references REQ-VERIFY-001 or SCENARIO-VERIFY-004
- **Deliverable:** `python/carnot/inference/guided_decoding.py`

### Exp 111: Productionize Guided Decoding

Integrate energy-guided generation into the VerifyRepairPipeline API.

- Extend `python/carnot/pipeline/verify_repair.py`:
  - Add `verify_and_generate()` method to VerifyRepairPipeline:
    - Input: question/prompt
    - Output: generated answer with guided decoding
    - Integrates EnergyGuidedSampler with the existing verify/repair mechanism
    - Options: guided_only, guided_then_verify, verify_then_guided
  - Add `GuidedGenerationConfig`:
    - `mode`: enum (guided_only, guided_then_verify, verify_then_guided)
    - `alpha`: guidance strength
    - `check_every_k`: constraint check frequency
    - `max_tokens`: max generation length
    - `temperature`: sampling temperature
  - Backward compatible: existing `verify()` and `verify_and_repair()` unchanged
- Create `tests/python/test_guided_generation.py`:
  - Test all three modes
  - Test config validation
  - Test backward compatibility
  - Every test references REQ-VERIFY-001 or SCENARIO-VERIFY-004
- Create `scripts/experiment_111_guided_production.py`:
  - Compare: verify_and_repair (post-hoc) vs verify_and_generate (guided) vs combined
  - 100 GSM8K problems × 3 modes
  - Measure: accuracy, repair count, latency, constraint satisfaction
  - Answer: is guided generation better than post-hoc repair?
- **Deliverable:** `python/carnot/pipeline/verify_repair.py` (extended)

### Exp 112: Embedding Bottleneck Resolution

Exp 102 showed MiniLM embedding is the bottleneck at 7.6ms. Explore faster alternatives.

- Profile embedding options:
  - ONNX runtime with MiniLM-ONNX: expect 2-5x speedup
  - Quantized MiniLM (int8/fp16): memory + speedup
  - LLM-native embeddings (Qwen/Gemma hidden states): no extra model, but task-specific
  - Hash embedding (no neural net): ultra-fast for exact-match constraints
- Implement and benchmark each option:
  - Measure: latency (p50/p95/p99), memory, accuracy (AUROC impact)
  - Compare to current MiniLM-ONNX baseline if available
- Choose best option based on: latency <1ms at p99 AND no AUROC regression
- Integrate chosen embedding into the guided decoding pipeline
- **Metric:** Embedding latency reduction, p99 constraint check latency
- **Deliverable:** `python/carnot/embeddings/fast_embedding.py` + `scripts/experiment_112_embedding_benchmark.py`

## Phase 33: Factual Grounding + Knowledge Base Expansion (experiments 113, 114, 115)

Complete the failed Exp 98 — robust factual claim verification.

### Exp 113: Factual Knowledge Base Extractor v2

Robust implementation of the factual KB extractor (failed in v11 due to pre-test failures).

- Create `python/carnot/pipeline/knowledge_base.py`:
  - `KnowledgeBase`:
    - `__init__(facts_path=None)` — load from JSON or use embedded defaults
    - `lookup(entity, relation)` → value or None
    - `verify_claim(entity, relation, claimed_value)` → VerifyResult
    - Built-in: 5000 facts as embedded dict:
      - 1000 country capitals, populations (with tolerance ranges)
      - 500 historical dates, scientific constants
      - 500 geographic facts, element properties
      - 500 person facts, company founders
      - 500 movie/book/invention facts
      - 1000 misc common knowledge
  - `FactualKBExtractor` implementing ConstraintExtractor Protocol:
    - Entity-relation-value triple extraction via regex + spaCy patterns
    - Pattern examples:
      - "X is the capital of Y" → (Y, capital, X)
      - "X was born in Y" → (X, birth_year, Y)
      - "The population of X is Y" → (X, population, Y)
    - KB lookup: verified/contradicted/unknown
    - Energy encoding: verified=0.0, contradicted=1.0, unknown=skip
  - Entity linking: normalize "USA", "U.S.", "United States" → same entity
  - Coreference resolution: "The capital of France is Paris. It is..." → link "It" to Paris
- Create `tests/python/test_factual_kb.py`:
  - KB lookup for known/unknown facts
  - Claim extraction from various sentence patterns
  - Entity linking and coreference
  - Integration with AutoExtractor and VerifyRepairPipeline
  - Every test references REQ-VERIFY-001 or SCENARIO-VERIFY-002
- Register FactualKBExtractor in AutoExtractor
- **Deliverable:** `python/carnot/pipeline/knowledge_base.py`

### Exp 114: WebSearch Fallback for Uncovered Claims

For facts not in the local KB, use web search as a fallback.

- Add `WebSearchExtractor` to `python/carnot/pipeline/extract.py`:
  - Triggered when FactualKBExtractor returns "unknown"
  - Uses DuckDuckGo API (no key required) or simulated fallback
  - Extract factual claims from search snippets
  - Verify extracted claims against retrieved facts
  - Return confidence score based on source consistency
- Create `scripts/experiment_114_websearch_fallback.py`:
  - 100 factual questions (50 correct, 50 false) not in local KB
  - Measure: KB hit rate (what fraction covered by local KB)
  - Measure: WebSearch fallback coverage
  - Compare: KB-only vs KB+WebSearch accuracy
  - Analyze: which fact categories benefit most from WebSearch?
- Graceful degradation: if search unavailable, fall back to KB-only
- **Metric:** KB coverage increase from WebSearch fallback, accuracy improvement
- **Deliverable:** `python/carnot/pipeline/extract.py` (extended)

### Exp 115: TruthfulQA Benchmark with v12 Extractors

Run the TruthfulQA benchmark with all v11+v12 extractors.

- Create `scripts/experiment_115_truthfulqa.py`:
  - Load TruthfulQA dataset (or 100 synthetic TruthfulQA-style questions):
    - 25 factual (capitals, dates, scientific facts)
    - 25 common misconceptions
    - 25 tricky phrasing
    - 25 adversarial
  - Generate answers with Qwen3.5-0.8B and Gemma4-E4B-it (simulated if unavailable)
  - Run through VerifyRepairPipeline with all extractors (v11 + v12):
    - ArithmeticExtractor, CodeExtractor, LogicExtractor, NLExtractor
    - IntermediateResultExtractor, ComparisonExtractor
    - FactualKBExtractor, WebSearchExtractor
  - Three modes: baseline, verify-only, verify-repair
- Metrics:
  - Factual hallucination detection rate
  - False positive rate
  - Knowledge base hit rate by category
  - Per-model comparison
- Error analysis:
  - Which fact categories does KB catch best/worst?
  - What types of factual claims still slip through?
  - Recommendations for KB expansion
- **Deliverable:** `scripts/experiment_115_truthfulqa.py` + `ops/truthfulqa-v12-results.md`

## Phase 34: LNN Adaptive Constraints + Full-Scale Benchmark (experiments 116, 117, 118)

LNN-based adaptive constraints and full-scale benchmark to validate all v12 work.

### Exp 116: LNN-Based Adaptive Constraint Model

Liquid Neural Networks for constraint models that adapt during multi-turn agent workflows (Research-Program Goal #8).

- Create `python/carnot/models/lnn_constraint.py`:
  - `LNNConstraintModel` implementing EnergyFunction protocol:
    - Uses continuous-time differential equations for parameter adaptation
    - Coupling strengths evolve in response to observations during agent execution
    - More robust to noise than static Ising
  - Based on liquid time-constant networks (LTCN):
    - Gate: input-dependent time constant τ(x) = τ_base / (1 + W_gx · x)
    - Memory: hidden state h(t) evolves as dh/dt = (1/τ(x)) × (-h + W · x)
    - Output: energy computed from evolved hidden state
  - `adapt(observation)` — update coupling strengths based on new observation
  - `reset()` — reset to initial coupling strengths
  - Training: contrastive divergence with LTC forward pass
- Create `tests/python/test_lnn_constraint.py`:
  - LNN forward pass produces finite energy
  - Adaptation changes energy values meaningfully
  - Training via CD converges
  - Every test references REQ-CORE-001 or SCENARIO-CORE-001
- Compare LNN vs Ising on multi-step verification:
  - 20 synthetic 5-step reasoning chains (10 correct, 10 with errors at step 2-4)
  - Both models process the full chain
  - LNN adapts between steps, Ising does not
  - Measure: which model detects errors faster/better?
- **Deliverable:** `python/carnot/models/lnn_constraint.py`

### Exp 117: Full-Scale Multi-Model Benchmark with v12 Extractors

Re-run the full multi-model comparison with all v11+v12 extractors.

- Create `scripts/experiment_117_full_benchmark.py`:
  - Same setup as Exp 93: 250 questions × 2 models × 3 modes = 1500 evals
  - But with expanded AutoExtractor (all v11 + v12 extractors)
  - Additional test: guided generation mode (from Exp 111)
- Four modes now:
  1. Baseline (LLM alone)
  2. Verify-only (post-hoc constraint verification)
  3. Verify-and-repair (post-hoc with regeneration)
  4. Guided generation (energy-steered during generation)
- Metrics per cell (model × domain × mode):
  - Accuracy, hallucination rate, repair success rate
  - Constraint coverage, false negative rate
  - Per-extractor contribution
  - Latency (total time per question)
- Comparison with v10 (Exp 93):
  - Side-by-side accuracy table: v10 vs v11 vs v12 extractors
  - False negative reduction over time
  - Statistical significance: bootstrap 95% CI on accuracy difference
  - Per-domain: which domains improved most with each phase?
- **Deliverable:** `scripts/experiment_117_full_benchmark.py` + `ops/full-benchmark-v12.md`

### Exp 118: HuggingFace Publish v12 Artifacts

Publish the KAN model and update existing model cards.

- Create `scripts/publish_v12_models.py`:
  - Publish the KAN constraint model (Exp 108-109):
    - Save via safetensors with config JSON
    - Model card: approach, AUROC comparison table, usage example
    - Clear disclaimer: research prototype
  - Publish the guided decoding adapter:
    - Small module that wraps any HuggingFace LLM
    - Attaches constraint verification during generation
    - Model-agnostic, novel artifact
  - Update existing model cards:
    - Clarify Phase 1 (activation EBMs) vs Phase 5+ (constraint EBMs)
    - Point to `pip install carnot` for production use
- Create `models/constraint-verifier-v2/README.md`:
  - KAN energy tier documentation
  - Comparison: Ising vs KAN vs Gibbs
  - Usage example
  - Performance table
- Print HuggingFace upload instructions (do NOT upload automatically)
- **Deliverable:** `scripts/publish_v12_models.py` + `models/constraint-verifier-v2/README.md`

## Dependencies

```
2026.04.6 outputs (done):
  Exp 101 ✓ (agentic workflow: 60% detection)
  Exp 102 ✓ (latency: JAX JIT 0.008ms)
  Exp 99 ✓ (constraint state propagation)
  Exp 100 ✓ (multi-step rollback)
  Exp 96 ✓ (intermediate result extractor)
  Exp 97 ✓ (comparison extractor)

Phase 31 (KAN):
  Exp 108 ← Exp 62 (domain constraint data) + carnot-core traits
  Exp 109 ← Exp 108 (KAN implementation) + Exp 62 (comparison data)

Phase 32 (guided decoding):
  Exp 110 ← Exp 102 (latency data) + Exp 108 (KAN energy, optional)
  Exp 111 ← Exp 110 (guided prototype) + VerifyRepairPipeline
  Exp 112 ← Exp 102 (embedding bottleneck) + Exp 110

Phase 33 (factual grounding):
  Exp 113 ← Exp 88 (world_knowledge gap) + knowledge base approach
  Exp 114 ← Exp 113 (KB extractor) + WebSearch fallback
  Exp 115 ← Exp 114 (WebSearch) + Exp 98 context (TruthfulQA)

Phase 34 (LNN + benchmark):
  Exp 116 ← carnot-core + Exp 62 (constraint data) + Exp 99 (agentic)
  Exp 117 ← Phase 31 + Phase 32 + Phase 33 (all extractors)
  Exp 118 ← Exp 108/109 (KAN model) + Exp 110/111 (guided)
```

## Execution Order

```
1.  exp108  -- KAN energy function implementation (high priority, no deps)
2.  exp109  -- KAN vs Ising vs Gibbs comparison (needs exp108)
3.  exp110  -- Guided decoding prototype completion (needs exp108/109)
4.  exp113  -- Factual KB extractor v2 (high priority, no deps)
5.  exp111  -- Productionize guided decoding (needs exp110)
6.  exp114  -- WebSearch fallback (needs exp113)
7.  exp116  -- LNN adaptive constraint model (needs exp99)
8.  exp112  -- Embedding bottleneck resolution (needs exp102)
9.  exp115  -- TruthfulQA benchmark with v12 extractors (needs exp114)
10. exp117  -- Full-scale multi-model benchmark (needs all extractors)
11. exp118  -- HuggingFace publish v12 artifacts (needs exp108/109/110/111)
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 108 | CPU + JAX | 4GB | 1-2 hours |
| 109 | CPU + JAX | 4GB | 1-2 hours |
| 110 | CPU + JAX | 4GB | 1-2 hours |
| 111 | CPU + JAX | 4GB | 1 hour |
| 112 | CPU + JAX | 4GB | 30 min |
| 113 | CPU | 2GB | 30 min |
| 114 | CPU + Network | 2GB | 30 min |
| 115 | CPU + Qwen3.5 + Gemma4 | 8GB | 1-2 hours |
| 116 | CPU + JAX | 4GB | 1-2 hours |
| 117 | CPU + Qwen3.5 + Gemma4 | 8GB | 2-3 hours |
| 118 | CPU | 2GB | 30 min |

## Success Criteria

- KAN achieves ≥ Ising AUROC on all domains (demonstrating expressiveness gain)
- KAN parameter count < Gibbs MLP (demonstrating efficiency)
- Guided decoding latency ≤ 1ms per constraint check at p99
- Guided generation accuracy ≥ verify-and-repair accuracy (proves generation-time steering works)
- Factual KB extractor achieves ≥70% precision on factual claims (up from 0% v10)
- WebSearch fallback increases factual coverage by ≥20%
- TruthfulQA detection rate ≥40% (up from ~0%)
- LNN adapts meaningfully between agent steps (energy changes >10% after adaptation)
- Full-scale benchmark shows ≥15% improvement over baseline (up from +10.2%)
- All 11 experiments produce results files
- HuggingFace publish script ready (not auto-uploaded)

## What's Explicitly NOT in Scope

- **FPGA Ising machine** — no FPGA hardware available; SamplerBackend abstraction ready for hardware
- **Full Kona implementation** — guided decoding is a stepping stone, not the full vision
- **Activation-based EBMs** — proven insufficient, permanently retired
- **PyPI publish** — beta release ready, actual publish is release ops
- **Mamba/RWKV for constraint state** — LNN addresses this more directly
- **External LLM API calls in production** — all computation is local (JAX/Rust)
