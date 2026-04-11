# Carnot Research Roadmap v17: JEPA Logic Repair + Lookahead Energy + Live Inference + Tier-4 Structure

**Created:** 2026-04-11
**Milestone:** 2026.04.12
**Status:** Planned (activates when milestone 2026.04.11 completes)
**Supersedes:** research-roadmap-vNEXT.md (milestone 2026.04.11 / v16)
**Informed by:** Exp 155 (JEPA v2 logic AUROC=0.479), Exp 156 (fast-path target_met=false),
Exp 157 (spilled energy AUROC=1.000 simulated), Exp 158 (factual extractor 96% coverage),
Exp 162 (adversarial p=0.017 z-test, permutation non-significant), Exp 165 (ArXiv scan)
**ArXiv inputs:** 2512.15605 (AR-EBM bijection/lookahead energy), s41467-025-59011-x
(thermodynamic FPGA SPU), 2601.13600 (global consistency checking noisy oracles),
2505.07179 (LagONN Lagrange constraint satisfaction), 2602.18671 (spilled energy ICLR 2026)

## What 2026.04.11 Proved

- **JEPA v2 logic domain failed** — Multi-domain retraining (Exp 155) improved code (0.776)
  but logic AUROC=0.479 (chance level, down from no-domain baseline). Root cause: RandomProjection
  byte-histogram embeddings are domain-agnostic — they cannot distinguish valid from invalid
  logical arguments at the byte level. Logic needs structured symbolic features (truth tables,
  entailment checks, contradiction flags), not character-level projections.

- **JEPA fast-path target not met** — Exp 156: at threshold=0.5, 52.8% fast-path but 10.2%
  accuracy degradation (target: <2%). Code domain causes all errors at t≤0.5 because the predictor
  fast-paths all code questions (AUROC=0.776 still produces too many false negatives). Logic errors
  dominate at t=0.7. Architecture is proven correct; feature quality is the bottleneck.

- **Spilled energy validated on simulated data only** — Exp 157: AUROC=1.000 on 50 synthetic
  TruthfulQA pairs with hand-crafted logit tensors. The AUROC result is mathematically correct
  but potentially over-optimistic. Needs validation with real Qwen3.5-0.8B logits on actual
  TruthfulQA questions where the model may be uncertain for different reasons than simulated.

- **Factual extractor works at coverage=96%, accuracy=83.3%** — Exp 158: Wikidata SPARQL-based
  FactualExtractor closes the factual blind spot (was 100% false negative rate per Exp 88).
  Coverage exceeds target (>30%) and accuracy is actionable. The remaining gap: it needs
  integration into the live pipeline with real model outputs, not just offline TruthfulQA pairs.

- **Full benchmarks confirmed with simulation** — Exp 161 (full GSM8K N=1319): Qwen +13.8%
  [±1.8pp 95% CI], Gemma +10.7% [±1.6pp]. Exp 162 (adversarial): z-test p=0.017 significant,
  1.41× improvement ratio on adversarial vs control. Permutation test p=0.429 (non-significant).
  Statistical significance requires BOTH tests; currently only z-test passes. Need larger N
  (permutation test requires ≥4 conditions/variant for power) or live inference with real variance.

- **HuggingFace publishing complete** — Exp 164: guided-decoding-adapter, constraint-propagation
  models (arithmetic AUROC=0.997, logic=1.000, code=0.867), JEPA predictor v2 (macro AUROC=0.659),
  all published. 16 per-token EBM READMEs updated with `pip install carnot` pointer.

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: JEPA Logic Fast-Path Still Broken (Tier 3 Self-Learning, FR-11)

Logic domain AUROC=0.479 means the fast-path gate is actively harmful for logic queries —
it skips verification on 75%+ of logic questions that need it. The root cause is clear:
RandomProjection on byte histograms cannot represent logical structure. Fix requires
domain-specific features for logic: truth table consistency, propositional formula parse trees,
entailment/contradiction flags. This is the blocking issue for Tier 3 self-learning
(predictive fast-path gate) which has been the target for three consecutive milestones.

### Gap 2: All Strong Results Remain Simulated (Goals #5, #6, FR-12)

Every benchmark showing >10% improvement uses CARNOT_SKIP_LLM=1 simulation with
Apple-calibrated error rates. The adversarial significance is directional only (one of two
required tests is significant). The eGPU (RX 7900 XTX, 24GB VRAM) is owned hardware with
confirmed availability but remains unconnected. Once connected, gfx1100 (unlike gfx1150 iGPU)
has full ROCm + JAX support, enabling full-scale live inference benchmarks. This is the single
highest-leverage hardware action in the project.

### Gap 3: New ArXiv Signals Not Integrated (FR-12, research-program.md Goal #10)

Three new papers from the Exp 165 scan provide concrete pipeline improvements not yet implemented:

1. **2512.15605 (lookahead energy)** — AR-EBM bijection gives a training-free hallucination
   signal from LLM logits. Stronger than spilled energy for multi-token constraint reasoning.
2. **2601.13600 (consistency checking)** — Framework for detecting globally inconsistent outputs
   across multi-turn chains when each step passes locally. Directly extends ConstraintStateMachine.
3. **2505.07179 (LagONN)** — Lagrange oscillatory networks escape infeasible Ising states.
   Addresses the failure mode where constraint problems have no solution in feasible region.

## Architecture (2026.04.12)

```
                 ┌────────────────────────────────────────────────────────────┐
                 │               Carnot Pipeline (2026.04.12)                 │
                 │                                                            │
LLM generates ──▶│  ┌──────────────────────────────────────────────────────┐ │
tokens + logits  │  │ Pre-Filter Layer (parallel, fast)                     │ │
                 │  │  ├─ SpilledEnergyExtractor (Exp 157, logits required) │ │
                 │  │  └─ LookaheadEnergyExtractor (NEW Exp 169, AR-EBM)   │ │
                 │  └──────────────────────┬───────────────────────────────┘ │
                 │                         │ combined pre-filter score        │
                 │  ┌──────────────────────▼───────────────────────────────┐ │
                 │  │ JEPA Fast-Path Gate v3 (NEW Exp 166-168)             │ │
                 │  │  Logic: structured symbolic features (NEW)           │ │
                 │  │  Code: AST-hash features (improved)                  │ │
                 │  │  Arithmetic: RandomProjection (working)              │ │
                 │  │  Target: <2% degradation @ ≥40% fast-path           │ │
                 │  └──────┬───────────────────────────────────────────────┘ │
                 │         │ slow-path (needs full verification)              │
                 │         ▼                                                  │
                 │  ┌─────────────────────────────────────────────────────┐  │
                 │  │ AutoExtractor + ConstraintGenerator v2 (NEW Exp 173)│  │
                 │  │  + NegationConstraint                               │  │
                 │  │  + CarryChainConstraint (improved)                  │  │
                 │  │  + FactualExtractor (Exp 158, Wikidata SPARQL)      │  │
                 │  └──────────────────────┬──────────────────────────────┘  │
                 │                         │                                  │
                 │  ┌──────────────────────▼──────────────────────────────┐  │
                 │  │ ComposedEnergy + KAN Tier-4 adaptive (NEW Exp 175) │  │
                 │  │  KAN AMR self-restructures after N verifications    │  │
                 │  │  LagONN fallback for infeasible states (NEW Exp 174)│  │
                 │  └──────────────────────┬──────────────────────────────┘  │
                 │                         │                                  │
                 │  ┌──────────────────────▼──────────────────────────────┐  │
                 │  │ GlobalConsistencyChecker (NEW Exp 176, arxiv         │  │
                 │  │ 2601.13600) wrapping ConstraintStateMachine         │  │
                 │  │ Detects cross-step inconsistencies missed locally    │  │
                 │  └─────────────────────────────────────────────────────┘  │
                 └────────────────────────────────────────────────────────────┘
                                      │
                 ┌────────────────────▼────────────────────────────────────────┐
                 │            Hardware Path (2026.04.12)                        │
                 │  eGPU (RX 7900 XTX): live LLM — CONNECT VIA THUNDERBOLT    │
                 │  CPU: AutoExtractor, ConstraintMemory, KAN verify            │
                 │  NPU (XDNA): JEPA predictor v3 — needs VitisAI EP build     │
                 │  FPGA (future): LagONN + sparse Ising (arxiv 2602.15985)    │
                 └─────────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
Exp 166 (logic symbolic features) ──▶ Exp 167 (JEPA v3 retrain) ──▶ Exp 168 (fast-path v3)
                                                                            │
Exp 169 (lookahead energy extractor) ──▶ Exp 170 (real logits benchmark)   │
                 │                              │                           │
                 └──────────────────┐           │                           │
                                    ▼           ▼                           ▼
                           Exp 171 (combined signal benchmark) ◀─────────────┘

Exp 172 (consistency checker, 2601.13600) [depends on Exp 176 multi-turn]

Exp 173 (constraint generation v2: negation + carry) [independent]

Exp 174 (LagONN, 2505.07179) [independent]

Exp 175 (Tier-4 KAN adaptive loop) [depends on Exp 153 AMR - done]

Exp 176 (multi-turn factual + consistency) [depends on Exp 158, Exp 172]

Exp 177 (eGPU setup) ──▶ Exp 178 (live adversarial GSM8K, N≥400/variant)

Exp 179 (NPU VitisAI activation) [independent, hardware-gated]
```

## Phase 45: Fix JEPA Logic Domain (Experiments 166–168)

Logic AUROC=0.479 is a blocking failure for Tier 3 self-learning. Three experiments fix the
data → embedding → model → validation pipeline.

### Exp 166: Logic-Aware JEPA Training Data with Symbolic Features

**Deliverable:** `results/jepa_training_pairs_logic_v3.json`

RandomProjection on byte histograms cannot represent logical structure. This experiment generates
a richer feature vector for logic domains: truth table consistency (can premises be simultaneously
true?), propositional parse completeness, presence of contradiction indicators ("not", "but",
"however"), syllogism structure (all/some/no quantifiers), conclusion validity (does it follow?).

Generate 500 logic pairs (250 valid arguments, 250 with deliberate logical errors):
- Syllogisms (all A are B; some C are A; therefore? — varied with/without valid conclusions)
- Propositional chains (if P then Q; P; therefore Q vs if P then Q; Q; therefore P [fallacy])
- 3-step reasoning (premise1 + premise2 + premise3 → claim, with one premise sometimes false)
- Negation scope (not all A are B ≠ all A are not B)
- Quantifier scope (some vs all vs no — generate violations of each)

Feature extraction: 40-dimensional logic feature vector (not byte histogram):
- `has_contradiction_word` (not, but, however, except, unless)
- `quantifier_type` (all=1, some=0.5, no=0, unknown=-1)  
- `n_premises`, `n_conclusions`, `conclusion_length_ratio`
- `has_therefore_therefore` (double therefore → likely redundant)
- `negation_count` / `assertion_count`
- `conditional_chain_depth` (if...then nesting)
- `vocabulary_overlap` (premises vs conclusion)
- Pad to 256 dims with zeros for compatibility with existing JEPA predictor input

Label with LogicExtractor violations. Save schema identical to Exp 143.

### Exp 167: JEPA Predictor v3 — Domain-Specific Embedding Heads

**Deliverable:** `results/jepa_predictor_v3.safetensors`

Train JEPAViolationPredictor with domain-specific embedding pre-processing:
- Logic: 40-dim symbolic features → Linear(40, 256) before MLP
- Code: 256-dim AST-hash features (character n-gram hashes of AST node types)
- Arithmetic: keep RandomProjection (AUROC=0.721, working)

Multi-head input: concat(domain_embedding[256], domain_one_hot[3]) → 259-dim input
→ Linear(259, 256) → existing MLP(256→64→32→3)

Train with balanced domain sampling (equal batch weight regardless of domain size).
Class-weighted BCE per domain. 150 epochs, early stop on val macro AUROC.
Target: logic AUROC >0.70, macro AUROC >0.75.
Compare v3 vs v2 (v2: arithmetic=0.721, code=0.776, logic=0.479).

### Exp 168: JEPA Fast-Path Benchmark v3

**Deliverable:** `results/experiment_168_results.json`

Re-run Exp 156 benchmark with v3 predictor. Same 500 questions (200 arithmetic, 200 code,
100 logic). Test thresholds: 0.3, 0.5, 0.7. Target: ≥40% fast-path AND <2% accuracy
degradation at some threshold. Report per-domain error breakdown.

If target met: update VerifyRepairPipeline to load jepa_predictor_v3.safetensors by default
when present. Tier 3 self-learning is then complete.

## Phase 46: Lookahead Energy + Real Logits Validation (Experiments 169–172)

The spilled energy signal (AUROC=1.000) was validated only on simulated logits. Two new arxiv
papers provide a stronger theoretical foundation and real-data validation path.

### Exp 169: Lookahead Energy Extractor (arxiv 2512.15605)

**Deliverable:** `python/carnot/pipeline/lookahead_energy.py`

Implement the AR-EBM bijection from arxiv 2512.15605. The "lookahead energy" of a response
prefix is: E_lookahead = -log P(response | prompt) under the LLM = sum(-log p(token_t))
over all response tokens. This is exactly the negative log-likelihood of the generation.

Key insight: this energy is already computed during inference (it's the sum of logit values
used for token sampling). High lookahead energy = LLM is "surprised" by its own output =
likely hallucination or constraint violation.

`LookaheadEnergyExtractor`:
- `extract(text, logits=None)` → list[ConstraintTerm]
- If logits present: compute mean NLL over response tokens as LookaheadEnergyConstraint
- If no logits: graceful degradation (return empty list)
- `LookaheadEnergyConstraint.energy` = mean(-log p_token) = NLL per token
- Satisfied iff NLL < threshold (default: 2.0, calibrate on TruthfulQA)

Benchmark vs SpilledEnergyExtractor (Exp 157):
- Run both on 50 arithmetic Q&A pairs (25 correct, 25 wrong) with synthetic logits
- Compare AUROC: which signal better separates correct from incorrect responses?
- Expected: lookahead NLL < spilled energy for arithmetic (arithmetic errors produce
  locally high-confidence but globally inconsistent tokens)

### Exp 170: Real LLM Logits Benchmark for Both Energy Signals

**Deliverable:** `results/experiment_170_real_logits_results.json`

Validate SpilledEnergyExtractor and LookaheadEnergyExtractor on real Qwen3.5-0.8B logits.

Generate 100 real TruthfulQA-style factual QA pairs using local model inference:
- 50 factual questions where Qwen3.5-0.8B is likely correct (well-known facts)
- 50 factual questions where it is likely to hallucinate (obscure/adversarial)

Use `generate()` from `carnot.inference.model_loader` with `return_logits=True` if available.
If model loading fails: use Exp 162/163 simulation logs to extract proxy logit distributions.

Compute for each response:
- `spilled_energy` (Exp 157 method: max(0, logit_energy - output_energy) per token)
- `lookahead_energy` (Exp 169 method: mean NLL per token)

Report:
- AUROC for each signal separately on factual questions
- AUROC when combined (max, sum, weighted combination)
- Optimal combination weights (grid search 0.1 increments)
- Compare to FactualExtractor (Exp 158: 83.3% accuracy) on the same questions

### Exp 171: Combined Signal Pipeline Benchmark

**Deliverable:** `results/experiment_171_combined_results.json`

Run the full VerifyRepairPipeline with ALL signals active: SpilledEnergy + LookaheadEnergy +
ConstraintGenerator + FactualExtractor on 200 mixed-domain questions.

Compare 5 configurations:
1. Baseline (no verification)
2. Ising constraints only (Exp 141 production pipeline)
3. Spilled energy + Ising (Exp 157 + existing)
4. Lookahead energy + Ising (Exp 169 + existing)
5. All signals combined (pre-filter → gate → Ising → factual)

Metric: accuracy improvement vs baseline per domain (arithmetic, code, logic, factual).
Also report: latency per signal (target: combined pre-filter <0.5ms added overhead).

### Exp 172: Global Consistency Checker (arxiv 2601.13600)

**Deliverable:** `python/carnot/pipeline/consistency_checker.py`,
               `results/experiment_172_results.json`

Implement `GlobalConsistencyChecker` based on the consistency checking framework from
arxiv 2601.13600. The ConstraintStateMachine (Exp 125) checks each step locally — a step
passes if its constraints are satisfied. But step 3's output may be globally inconsistent
with step 1's verified facts even if each step passed independently.

`GlobalConsistencyChecker.check(state_machine)` → `GlobalConsistencyReport`:
- Pairwise consistency: for each pair of verified steps (i, j), extract claims from both
  and check if any claim in step i contradicts any claim in step j
- Contradiction types: factual contradictions (claim reversal), numeric contradictions
  (different values for same quantity), logical contradictions (A and not-A)
- Returns: list of inconsistent pairs, severity scores, recommended rollback target

Test on 20 multi-step reasoning chains (4 steps each, deliberate cross-step contradictions
in 10 chains). Measure: detection rate vs local-only checking.

## Phase 47: Constraint Generation v2 + LagONN + Tier-4 Adaptive KAN (Exp 173–176)

### Exp 173: Constraint Generation v2 — NegationConstraint + CarryChain v2

**Deliverable:** `python/carnot/pipeline/generation.py` (update),
               `results/experiment_173_results.json`

Exp 141 ConstraintGenerator caught `comparison_boundary` (recall 0%→100%) but left two
patterns unimplemented: `negation_scope` and `arithmetic_carry`.

Implement in `ConstraintGenerator`:
- `NegationConstraint` for `negation_scope` memory pattern:
  - Detect "X is not Y", "not all A are B", "neither A nor B"
  - Energy = 1 if LLM response asserts the negated claim; 0 otherwise
  - Pattern matching: extract (subject, negated_predicate, object) and verify
  - Test: "The capital of France is NOT Rome" → should catch "The capital is Rome"

- `CarryChainConstraint` v2 (improve Exp 141's _count_carries):
  - Handle multi-step carry chains: 999+1, 99999+1, N-digit cascades
  - Also: borrowing chains in subtraction (1000-1 = 999, not 899)
  - Feature: detect when answer has wrong number of digits (e.g., 99+1=99 instead of 100)

Benchmark on 300 targeted questions (100 negation, 100 multi-carry, 100 mixed from Exp 141):
- Compare: Exp 141 accuracy vs Exp 173 (v2) accuracy on same test set
- Target: +5pp over Exp 141's +11% memory augmentation

### Exp 174: LagONN — Lagrange Oscillatory Constraint Solver (arxiv 2505.07179)

**Deliverable:** `python/carnot/models/lagoon.py`,
               `results/experiment_174_results.json`

Implement `LagrangeONN` from arxiv 2505.07179. The paper shows that standard Ising machines
get trapped in infeasible states on constrained problems. LagONN uses Lagrange multipliers
to dynamically penalize infeasibility: the energy landscape is augmented with constraint
violation penalties that grow over time, pushing the trajectory toward feasible regions.

Architecture:
```python
class LagONN(EnergyFunction):
    def energy(self, x, lambda_: jnp.ndarray) -> jnp.ndarray:
        # Base Ising energy
        e_ising = -0.5 * x @ self.J @ x - self.bias @ x
        # Constraint violation penalty (Lagrange term)
        violation = jnp.maximum(0, self.A @ x - self.b)  # Ax ≤ b constraints
        e_lagrange = lambda_ @ violation
        return e_ising + e_lagrange
    
    def update_lambda(self, lambda_, violation, lr=0.01):
        # Dual ascent: increase penalty for violated constraints
        return lambda_ + lr * violation
```

Benchmark vs ParallelIsingSampler on:
- 200-variable Max-3-SAT problems (Exp 61 used 500-var; start smaller)
- 10 constrained satisfaction problems from Exp 44 (scheduling domain)
- Metric: fraction of problems where LagONN finds feasible solution vs Ising alone
- Expected: LagONN should solve problems where Ising gets stuck in infeasible states

Also compare: does LagONN + standard Ising (run LagONN first to find feasible region,
then Ising for energy minimization) beat either alone?

### Exp 175: Tier-4 KAN Adaptive Structure Self-Learning Loop

**Deliverable:** `python/carnot/models/adaptive_kan.py`,
               `results/experiment_175_results.json`

Exp 153 proved KAN AMR maintains accuracy while reducing params by 1.3%. This experiment
wires AMR into the verification tracking loop — creating Tier-4 adaptive structure behavior.

`AdaptiveKAN` extends `KANConstraintModel` (Exp 108-109):
- Tracks: verification count, per-edge curvature running average (updated after each batch)
- `maybe_restructure(n_verifications)`: if n_verifications is a multiple of 500:
  - Run `compute_edge_curvature()` on last 100 verification inputs
  - Call `refine(threshold_multiplier=1.5)` — add knots where curvature is high
  - Log: edges added/removed, param count delta, before/after AUROC
- `checkpoint(path)`: save current knot structure + weights

Simulation: run 1500 synthetic verifications (arithmetic domain) in 3 batches of 500.
Between each batch: trigger restructuring. Measure:
- AUROC after batch 1, 2, 3 — does it improve monotonically?
- Parameter count delta — does the model learn to allocate capacity where needed?
- Compare: AdaptiveKAN vs static KAN (no AMR) on 200-question held-out test set

Self-learning loop: this is the hardware-acceleratable path to Tier-4 (research-program.md):
static within chains → online weights (Tier 1) → constraint memory (Tier 2) → JEPA predictor
(Tier 3) → adaptive structure (Tier 4). Each tier shown to work. Now close the loop.

### Exp 176: Multi-Turn Factual Verification with Global Consistency

**Deliverable:** `scripts/experiment_176_multiturn_factual.py`,
               `results/experiment_176_results.json`

Combine three components: FactualExtractor (Exp 158) + ConstraintStateMachine (Exp 125) +
GlobalConsistencyChecker (Exp 172) for multi-turn factual reasoning verification.

Test on 20 multi-step factual reasoning chains:
- Each chain: 4 steps (claim A → derive B → derive C → reach conclusion D)
- Ground truth: 10 chains where claims are internally consistent, 10 with contradictions
- Contradiction types: factual (different birth year in step 1 vs step 3),
  temporal (event before its cause in step 2), spatial (location inconsistency)

Measure:
- Local-only detection (ConstraintStateMachine, step-by-step): precision, recall
- Global detection (+GlobalConsistencyChecker): precision, recall
- Expected: global consistency adds detection for chains where each step passes locally

Hardware note: This requires FactualExtractor's Wikidata SPARQL calls (Exp 158).
Set timeout to 5s per call; 20 chains × 4 steps = 80 max API calls. Run with network access.

## Phase 48: Live Inference + NPU Activation (Experiments 177–179)

### Exp 177: eGPU Setup + Live Inference Baseline

**Deliverable:** `scripts/experiment_177_egpu_setup.py`,
               `results/experiment_177_results.json`

**HARDWARE ACTION REQUIRED:** Connect RX 7900 XTX (gfx1100) via Thunderbolt chassis.

Script runs hardware detection + validation:
1. Detect eGPU: `rocminfo | grep gfx` (expect gfx1100)
2. Test PyTorch: `import torch; torch.cuda.get_device_name(0)`
3. Test JAX on GPU: `jax.devices()` (if gfx1100, JAX should work — unlike gfx1150)
4. If JAX works: remove `JAX_PLATFORMS=cpu` requirement for inference experiments
5. Benchmark Qwen3.5-0.8B inference: CPU p50, eGPU p50, speedup ratio
6. Benchmark Gemma4-E4B-it: CPU p50, eGPU p50
7. Run 10 GSM8K questions live: compare accuracy (simulated vs live)

If Thunderbolt unavailable:
- Document blocker with error details
- Run simulation fallback: CARNOT_SKIP_LLM=1 with Apple-calibrated rates
- Proceed with subsequent experiments using simulation

Results inform Exp 178's statistical design. Hardware note: see research-hardware-wishlist.md
"Connect RX 7900 XTX via Thunderbolt" action item.

### Exp 178: Live Adversarial GSM8K — Definitive Statistical Test

**Deliverable:** `scripts/experiment_178_adversarial_definitive.py`,
               `results/experiment_178_results.json`

Exp 162 achieved z-test p=0.017 but permutation test p=0.429 (underpowered: only 2 control +
6 adversarial delta points). Statistical significance requires BOTH tests. Fix: scale to
N=400/variant (1,600 total questions × 2 models = 3,200 inference calls) and use a paired
design that gives permutation test enough within-variant variance.

Statistical design:
- Control: 400 standard GSM8K questions (random sample from full 1,319)
- Number-swapped: 400 adversarial (same problems, numbers swapped)
- Irrelevant-injected: 400 adversarial (irrelevant sentence added)
- Combined: 400 adversarial (both perturbations)
- Modes: baseline, verify-repair (max 3 iters)
- Models: Qwen3.5-0.8B, Gemma4-E4B-it (if eGPU available)

Primary test: paired permutation test on (adversarial_improvement - control_improvement)
across N=400 question-level pairs. With N=400, achieve 80% power at 5pp true effect.
Secondary test: two-proportion z-test (same as Exp 162).
Report: adversarial/control improvement ratio, 95% CIs, both p-values.

This is Goal #5 (Apple GSM8K adversarial) definitive experiment.

### Exp 179: AMD XDNA NPU Activation

**Deliverable:** `scripts/experiment_179_npu_activation.py`,
               `results/experiment_179_npu_results.json`

The AMD XDNA NPU has been detected since Exp 146 but blocked by missing VitisAI onnxruntime.
Three paths to unlock:
1. **Path A (conda):** `conda create -n npu python=3.11; conda install -c amd onnxruntime-vitisai`
2. **Path B (AMD wheel):** Download from ryzenai.docs.amd.com (requires AMD account)
3. **Path C (build from source):** Clone onnxruntime, cmake with `-Donnxruntime_USE_VITISAI=ON`

Script tries all three paths in order, stops at first success.
If successful:
- Load `results/jepa_predictor_146.onnx` with AMDXDNAExecutionProvider
- Benchmark: 1000 inferences, measure p50/p99 latency (vs CPU: p50=0.005ms)
- Target: NPU p50 ≤ CPU p50 (prove NPU not slower for tiny model)
- Document VitisAI provider configuration for future experiments

If all paths fail: document specific error, log hardware/software state, propose remediation.

Update `python/carnot/samplers/npu_backend.py` with working configuration.
This unblocks Tier 3 self-learning on-device: JEPA predictor on NPU = <1ms gate at 0W cost.

## Hardware Requirements

| Experiment | Hardware | Available |
|------------|----------|-----------|
| Exp 166–176 | CPU (JAX, no GPU needed) | Yes |
| Exp 177 | RX 7900 XTX + Thunderbolt chassis | **Owned, needs connection** |
| Exp 178 | eGPU (or simulation fallback) | Conditional on Exp 177 |
| Exp 179 | AMD XDNA NPU + VitisAI software | Hardware available, software blocked |

**Priority hardware action:** Connect RX 7900 XTX via Thunderbolt before Exp 177.
This is the single highest-ROI action in the project — live benchmarks are the path to
credible publishable results.

**Priority software action:** Install VitisAI onnxruntime (conda path A is fastest).
Hardware is there; 30 minutes of setup unlocks NPU inference.

## Self-Learning Tier Progress

| Tier | Previous | This Milestone |
|------|----------|----------------|
| 1: Online weights | Exp 134 complete (reweighting works) | No new work |
| 2: Constraint memory | Exp 141 +11% on comparison_boundary | Exp 173: +negation + carry_chain |
| **3: Predictive verify** | **JEPA logic AUROC=0.479 FAILING** | **Exp 166-168: fix with symbolic features** |
| **4: Adaptive structure** | Exp 153 AMR standalone | **Exp 175: wire into live loop** |

Tier 4 closure is new this milestone: AdaptiveKAN restructures after N verifications without
human intervention. When FPGA hardware arrives, structural updates happen via partial
reconfiguration — the hardware path is validated by software first.

## Success Criteria

| Experiment | Target | Why It Matters |
|------------|--------|----------------|
| Exp 167 (JEPA v3) | logic AUROC >0.70, macro >0.75 | Tier 3 self-learning unblocked |
| Exp 168 (fast-path v3) | ≥40% fast-path, <2% degradation | Tier 3 milestone complete |
| Exp 170 (real logits) | spilled + lookahead AUROC >0.70 on real data | Validates Exp 157 claim |
| Exp 173 (constraint gen v2) | +5pp over Exp 141 | Tier 2 broader coverage |
| Exp 174 (LagONN) | finds feasible solutions where Ising fails | Escape infeasible states |
| Exp 175 (adaptive KAN) | AUROC improves after each restructure | Tier 4 loop demonstrated |
| Exp 178 (live adversarial) | both permutation + z-test p<0.05 | Goal #5 definitive proof |
| Exp 179 (NPU) | VitisAI EP working, NPU inference confirmed | Hardware Tier 3 path |
