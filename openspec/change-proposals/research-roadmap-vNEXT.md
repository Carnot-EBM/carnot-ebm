# Carnot Research Roadmap v15: JEPA Predictive Verification + Constraint Generation + Credibility Proof

**Created:** 2026-04-11
**Milestone:** 2026.04.10
**Status:** Planned (activates when milestone 2026.04.9 completes)
**Supersedes:** research-roadmap-v14.md (milestone 2026.04.9)
**Informed by:** Exp 134 (reweighting failure), Exp 135-136 (memory built), Exp 137-138 (guided decoding packaged), Exp 139 (arxiv scan), research-program.md §"Next Milestone Focus (2026.04.10)"

## What 2026.04.9 Proved

- **Tier 1 reweighting doesn't work** — Exp 134: precision-based AdaptiveWeighter produced
  fixed=0.53 vs adaptive=0.55 (no meaningful improvement). The infrastructure (ConstraintTracker,
  AdaptiveWeighter) is built and works mechanically, but adjusting _weights_ on existing constraints
  is the wrong approach. The error patterns that matter aren't captured by the existing constraint
  types at all — they need NEW constraint types generated from memory.
- **Tier 2 memory infrastructure is ready** — Exp 135-136: ConstraintMemory persists error
  patterns across sessions. Session 2 accuracy improved when session 1 memory was loaded.
  The pattern data is there; it just isn't being converted into new constraints at extraction time.
- **Guided decoding is packaged but not published** — Exp 137 produced
  `exports/guided-decoding-adapter/` with README, config.json, constraint_weights.safetensors,
  and example.py. Exp 138 benchmarked it. Neither pushed to HuggingFace.
- **ArXiv scan identified three concrete opportunities** — Exp 139 proposed:
  - Exp 140: Constraint-projection guided decoding latency benchmark
  - Exp 141: Apple GSM8K adversarial credibility experiment
  - Exp 143 (renumbered 152): LoRA-style continual learning for constraint retention

## The Gaps (3 Biggest vs PRD Vision)

### Gap 1: Self-Learning Doesn't Actually Improve Accuracy Yet (vs FR-11)

FR-11 requires an "autonomous self-improvement" loop. We have tracking (Tier 1) and memory
(Tier 2) infrastructure, but neither demonstrably improves accuracy. The root cause is architectural:
accumulating statistics about _existing_ constraints and reweighting them cannot close gaps that
exist because the constraints themselves don't cover the error modes. The fix is constraint
_generation_: when ConstraintMemory detects "arithmetic carry errors appear 5+ times," it should
produce a `CarryChainConstraint` and inject it into AutoExtractor for future queries — not just
reweight the existing `ArithmeticConstraint`.

### Gap 2: No Predictive/Proactive Verification (vs FR-12, research-program.md Tier 3)

The current pipeline is purely post-hoc: generate → verify → repair. This is provably slower than
it needs to be. After accumulating (partial_response, final_violation) pairs from Exp 130-136's
verify-repair logs, we have the training signal to predict constraint violations _before_ generation
completes. A small JEPA-style predictor that takes the first 50-100 tokens of a response and
predicts violation probability can gate the expensive Ising check: run full verification only when
the predictor says "high energy likely." This creates a fast-path/slow-path architecture that could
halve average verification latency.

### Gap 3: No Published Credibility Numbers (vs PRD §Success Metrics, research-program.md Goal #5)

Carnot has strong internal results (+27% accuracy on adversarial math, 100% hallucination detection)
but they're not on any external benchmark that the community trusts. The Apple GSM8K adversarial
variant (arxiv 2410.05229) is the ideal vehicle: it's a published methodology that exposes exactly
the weakness Carnot's architecture fixes (LLMs pattern-match irrelevant context; Ising verifies
independently). Running Carnot on the Apple adversarial variants and showing the improvement is
_larger_ on adversarial than standard GSM8K is the single most compelling result we can publish.
Additionally, the HuggingFace guided decoding adapter (Exp 137) is packaged but not uploaded —
no external user can try it yet.

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              Carnot Pipeline (2026.04.10)           │
                    │                                                     │
  LLM generates ──▶ │  ┌──────────────┐     ┌──────────────────────────┐ │
  tokens (partial)  │  │ JEPA Predictor│────▶│ fast-path: skip Ising   │ │
                    │  │ (Tier 3, NEW) │     └──────────────────────────┘ │
                    │  │ MLP on partial│     ┌──────────────────────────┐ │
                    │  │ embeddings    │────▶│ slow-path: full Ising    │ │
                    │  └──────────────┘     └──────────┬───────────────┘ │
                    │         ▲                         │                 │
                    │         │                         ▼                 │
                    │  ┌──────────────┐     ┌──────────────────────────┐ │
                    │  │ConstraintMem │     │ ConstraintStateMachine   │ │
                    │  │ory (Tier 2)  │────▶│ + rollback               │ │
                    │  │ patterns →   │     │ (Exp 125-127, working)   │ │
                    │  │ NEW constraints│    └──────────────────────────┘ │
                    │  └──────────────┘                                   │
                    │         ▲                                           │
                    │         │ ADD constraints                           │
                    │  ┌──────────────┐                                   │
                    │  │AutoExtractor │ ◀── memory-augmented extraction   │
                    │  │ (Exp 141 fix)│     (new in 2026.04.10)          │
                    │  └──────────────┘                                   │
                    └─────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────────────────┐
                    │            Hardware Acceleration Path               │
                    │  CPU: AutoExtractor, ConstraintMemory (Tier 1-2)   │
                    │  GPU/NPU: JEPA predictor inference (Tier 3, NEW)   │
                    │  FPGA/TSU: Ising sampling (future)                 │
                    └─────────────────────────────────────────────────────┘
```

## Phase 42: Fix Self-Learning — Constraint Generation (Experiments 140–142)

### Exp 140: Constraint-Projection Guided Decoding Latency Benchmark

The per-token constraint-projection operator measures whether the KAN energy gradient can project
logits onto the constraint-satisfying subspace fast enough for real-time guided decoding.

**Deliverable:** `scripts/experiment_140_guided_latency.py`

Implement a projection operator in `EnergyGuidedSampler` that:
1. Computes KAN energy gradient w.r.t. embedding at current generation state
2. Projects token logits away from high-energy directions
3. Measures wall-clock overhead at batch sizes 1, 8, 32 on CPU

Success criterion: <1ms per token at batch=1 (Exp 102 budget was 0.006ms for check-only;
projection adds gradient computation overhead). Compare to Exp 138's alpha-penalty approach.

### Exp 141: Memory-Augmented Constraint Generation

Wire `ConstraintMemory` (Exp 135) into `AutoExtractor` to _add_ new constraints when memory
patterns are found. This fixes the root cause of Exp 134's failure.

**Deliverable:** `python/carnot/pipeline/generation.py`

Design:
- `ConstraintGenerator` class: reads ConstraintMemory, translates patterns into new
  `ConstraintTerm` instances, injects them alongside AutoExtractor's static constraints
- Pattern→constraint mapping:
  - "arithmetic carry errors" → `CarryChainConstraint` for all arithmetic questions
  - "comparison boundary errors" → `BoundaryConstraint` for questions with inequalities
  - "negation scope errors" → `NegationConstraint` for questions with "not" / "never"
- Integration: `AutoExtractor.extract(text, memory=None)` accepts optional ConstraintMemory

Benchmark: compare AutoExtractor alone vs memory-augmented on 200 GSM8K questions where
Exp 134 warmup data exists. Target: measurable improvement over fixed weights.

### Exp 142: Combined Tier 1+2 Learning Benchmark

Compare all four configurations on a 500-question benchmark:

| Configuration | Tier 1 | Tier 2 |
|---------------|--------|--------|
| Baseline      | off    | off    |
| Tier 1 only   | on     | off    |
| Tier 2 only   | off    | on     |
| Combined      | on     | on     |

**Deliverable:** `scripts/experiment_142_combined_learning.py`

Key questions:
- Does Tier 2 (generation) beat Tier 1 (reweighting)? (Hypothesis: yes, reweighting is wrong)
- Does combined beat either alone?
- Which domains benefit most from memory-augmented generation?

## Phase 43: JEPA Predictive Verification — Tier 3 (Experiments 143–146)

### Exp 143: Collect (Partial Response, Final Violation) Training Pairs

Mine existing verify-repair logs and the Exp 130-138 results to build the JEPA training set.

**Deliverable:** `results/jepa_training_pairs.json` + `scripts/experiment_143_collect_pairs.py`

For each logged verification event:
1. Retrieve the full LLM response
2. Create N prefixes at 10%, 25%, 50%, 75% of token length
3. Label each prefix with: `final_violated` (bool) + per-constraint-type violation flags
4. Embed each prefix with RandomProjection (0.026ms, best from Exp 112)

Target: ≥1,000 (prefix, violation_label) pairs across arithmetic, code, logic domains.

### Exp 144: Train JEPA Violation Predictor

A small MLP trained to predict "will this response violate constraints?" from the first 50 tokens.

**Deliverable:** `python/carnot/pipeline/jepa_predictor.py` + `results/jepa_predictor.safetensors`

Architecture:
- Input: 256-dim RandomProjection embedding of first 50 tokens (0.026ms)
- Hidden: 64-dim ReLU layer
- Output: per-constraint-type violation probability (sigmoid)
- Training: binary cross-entropy on (partial_embedding, final_violated) pairs
- Train/val split: 80/20 from Exp 143 dataset

Evaluation: AUROC, precision@0.5, recall@0.5 per constraint type.
Target: AUROC >0.65 (above chance, below KAN's 0.994 — this is a predictor, not a verifier).

### Exp 145: JEPA Fast-Path / Slow-Path Integration and Benchmark

Integrate the JEPA predictor into `VerifyRepairPipeline` as an optional early-exit gate.

**Deliverable:** `scripts/experiment_145_jepa_fastpath.py`

Protocol:
- After N=50 tokens generated: run JEPA predictor (target: <1ms)
- If max(violation_probs) < threshold: skip full Ising, mark as FAST_PATH
- If max(violation_probs) >= threshold: run full constraint extraction + Ising

Benchmark on 500 questions (200 arithmetic, 200 code, 100 logic):
- Metrics: % fast-path taken, accuracy on fast-path, accuracy on slow-path, total wall-clock vs
  no-gating baseline
- Target: ≥40% fast-path taken with <2% accuracy degradation

### Exp 146: AMD XDNA NPU Experimentation

The machine has an AMD XDNA NPU (Ryzen AI / Strix Point). Install the AMDXDNA kernel driver and
Ryzen AI Software SDK, then test whether the JEPA predictor (Exp 144) runs on the NPU.

**Deliverable:** `scripts/experiment_146_npu.py` + `results/experiment_146_npu_results.json`

Steps:
1. Check NPU availability: `lspci | grep -i xdna` / `ls /dev/accel`
2. Install: `pip install amdxdna-sdk` (or follow AMD Ryzen AI SDK setup)
3. Run JEPA MLP (128 input, 64 hidden, 12 output) on NPU, benchmark latency
4. Compare: NPU vs CPU on 1000 inference calls
5. If NPU unavailable, document the setup attempt and environment constraints

Success: NPU inference working with latency measurement. If driver unavailable, document
environment state (useful for future sessions when AMDXDNA driver is in-tree).

## Phase 44: Credibility Benchmarks (Experiments 147–149)

### Exp 147: Apple GSM8K Adversarial Benchmark — THE Credibility Experiment

Directly prove Carnot's value proposition on the Apple (arxiv 2410.05229) adversarial benchmark.

**Deliverable:** `scripts/experiment_147_apple_gsm8k.py` + `results/experiment_147_results.json`

Methodology (reproducing Apple's approach):
1. Take 200 GSM8K test questions
2. Generate 3 adversarial variants per question:
   - **Number-swapped**: same logic, all numeric values replaced with different numbers
   - **Irrelevant-sentence**: one semantically irrelevant sentence injected with a number
   - **Combined**: both perturbations applied
3. Run **Qwen3.5-0.8B** and **google/gemma-4-E4B-it** on all 4 variants (baseline)
4. Run Carnot **verify-repair** pipeline on all 4 variants

Key hypothesis:
- LLM accuracy drops on adversarial variants (reproduce Apple's finding)
- Carnot verify-repair maintains accuracy (Ising ignores irrelevant context)
- **Improvement delta is LARGER on adversarial than standard** (more errors to catch)

Report: accuracy table per model × variant × mode (baseline/verify/verify-repair),
improvement delta ± 95% bootstrap CI, error type breakdown.

### Exp 148: Full GSM8K (1,319 questions) with Live Inference + CIs

The full benchmark with Exp 123's robust model loader guaranteeing live inference.

**Deliverable:** `scripts/experiment_148_gsm8k_full.py` + `results/experiment_148_results.json`

Protocol:
- Use `CARNOT_FORCE_LIVE=1` (fail rather than fall back to simulated)
- All 1,319 GSM8K test questions
- Both models: Qwen3.5-0.8B, google/gemma-4-E4B-it
- Three modes: baseline, verify-only, verify-repair
- Report: accuracy ± 95% bootstrap CI
- Compare to published baselines (Qwen3.5 GSM8K ~60-65%)

Target: verify-repair ≥ +10% over baseline on live inference, consistent with simulated results.

### Exp 149: TruthfulQA at Scale with Factual Constraint Coverage Analysis

Benchmark the factual domain — the known weak spot (Exp 88: near-zero coverage on factual claims).

**Deliverable:** `scripts/experiment_149_truthfulqa.py` + `results/experiment_149_results.json`

Protocol:
- 200 TruthfulQA questions (full categories: misconceptions, history, science, health)
- Measure factual constraint coverage (what % of questions yield extractable constraints)
- Run knowledge-base verifier (Exp 98) where coverage exists
- Run AutoExtractor elsewhere

Output:
- Per-category coverage rate and accuracy
- Which TruthfulQA categories are covered vs blind spots
- Estimated improvement from memory-augmented extraction (Exp 141 applied)
- Concrete list of 5 constraint types that would close the coverage gap

## Phase 45: HuggingFace Publishing + Continual Learning (Experiments 150–153)

### Exp 150: Push Guided Decoding Adapter to HuggingFace + Update Model READMEs

Exp 137 packaged `exports/guided-decoding-adapter/` but never uploaded it. This experiment
actually publishes to HuggingFace.

**Deliverable:** `scripts/experiment_150_hf_push.py` + updated HuggingFace model cards

Steps:
1. Run `huggingface-cli upload Carnot-EBM/guided-decoding-adapter exports/guided-decoding-adapter/`
2. Verify upload at huggingface.co/Carnot-EBM/guided-decoding-adapter
3. Update all 16 existing Carnot-EBM model READMEs:
   - Clarify: "These are Phase 1 research artifacts (detect confidence, not correctness)"
   - Add: "For the production verify-repair pipeline: `pip install carnot`"
   - Link to guided-decoding-adapter for guided decoding use
4. Log upload confirmation to `results/experiment_150_results.json`

### Exp 151: Publish Constraint Propagation Models to HuggingFace

Package the domain-specific trained Ising models (arithmetic, logic, code) as community artifacts.

**Deliverable:** `exports/constraint-propagation-models/` + HuggingFace upload

For each domain (arithmetic, logic, code):
1. Export trained Ising coupling matrix J as safetensors
2. Write model card: training methodology, benchmark results, usage example
3. Create `ConstraintPropagationModel.from_pretrained()` API
4. Upload to `Carnot-EBM/constraint-propagation-{domain}`

These are novel artifacts — no other HuggingFace repo has learned Ising constraint models.

### Exp 152: Continual Learning for Constraint Retention Across Agent Steps

Based on Exp 139 arxiv proposals (LoRA continual learning, Ferret framework): test whether
orthogonal LoRA-style updates to Gibbs parameters can retain constraints across a 5-step
reasoning chain without catastrophic forgetting — fixing the Exp 116 failure mode (LNN 10% vs
static Ising 100%).

**Deliverable:** `python/carnot/models/continual_gibbs.py` + `scripts/experiment_152_continual.py`

Design:
- `ContinualGibbsModel` wraps Gibbs MLP with orthogonal projection buffer
- At each agent step: compute gradient of new constraint → project onto null space of prior
  constraint gradients → update only the orthogonal component
- Compare: static Ising (100%), LNN (10%, Exp 116), ContinualGibbs (target: >80%)

Benchmark: 5-step math reasoning chains, 20 problems, accuracy on step 5 given constraints
accumulated across steps 1-4.

### Exp 153: KAN Adaptive Mesh Refinement (Tier 4 Foundation)

Implement knot insertion/pruning for carnot-kan based on local energy landscape complexity.
This is the Tier 4 "adaptive structure" mechanism from research-program.md.

**Deliverable:** `python/carnot/models/kan.py` (updated) + `scripts/experiment_153_kan_refinement.py`

Design:
- After training: compute per-edge curvature (second derivative of spline)
- High curvature edges get more knots (add at inflection points)
- Near-linear edges get fewer knots (prune)
- Refinement criterion: curvature > threshold × mean_curvature
- Measure: AUROC before vs after refinement on constraint verification task

Why now: carnot-kan exists (Exp 108-109, 0.994 AUROC). Adaptive mesh refinement is cheap
to add (pure Python/JAX, no new architecture). This plants the flag for Tier 4 adaptive structure.

## Dependencies

```
Phase 42 (constraint generation):
  Exp 140 ← exports/guided-decoding-adapter/ (Exp 137), EnergyGuidedSampler (Exp 110)
  Exp 141 ← python/carnot/pipeline/memory.py (Exp 135), AutoExtractor (Exp 74)
  Exp 142 ← Exp 141 + tracker.py (Exp 132) + adaptive.py (Exp 133)

Phase 43 (JEPA):
  Exp 143 ← verify-repair logs from Exp 130-138
  Exp 144 ← Exp 143 (training data) + fast_embedding.py (Exp 112)
  Exp 145 ← Exp 144 (predictor) + VerifyRepairPipeline (Exp 75)
  Exp 146 ← Exp 144 (model to deploy) + AMD hardware

Phase 44 (credibility):
  Exp 147 ← results/adversarial_gsm8k_data.json (Exp 119), verify-repair pipeline
  Exp 148 ← model_loader.py (Exp 123), VerifyRepairPipeline
  Exp 149 ← knowledge_base.py (Exp 98), AutoExtractor, memory.py (Exp 135)

Phase 45 (publishing + continual learning):
  Exp 150 ← exports/guided-decoding-adapter/ (Exp 137), HuggingFace credentials
  Exp 151 ← domain-specific Ising models (Exp 62), carnot-constraints crate
  Exp 152 ← LNN baseline (Exp 116, 128), Gibbs model, ConstraintStateMachine (Exp 125)
  Exp 153 ← carnot-kan (Exp 108-109), constraint benchmark data
```

## Execution Order

```
1.  exp140  — Guided decoding latency (quick, validates constraint projection)
2.  exp143  — Collect JEPA training pairs (independent, mines existing logs)
3.  exp141  — Memory-augmented constraint generation (fixes Exp 134 root cause)
4.  exp144  — Train JEPA predictor (depends on exp143 dataset)
5.  exp142  — Combined Tier 1+2 benchmark (depends on exp141)
6.  exp145  — JEPA fast-path integration (depends on exp144)
7.  exp146  — AMD XDNA NPU experiment (depends on exp144 model)
8.  exp147  — Apple GSM8K adversarial (can use existing adversarial data from Exp 119)
9.  exp148  — Full GSM8K 1319 with live inference
10. exp149  — TruthfulQA at scale + coverage analysis
11. exp150  — Push to HuggingFace (requires internet/credentials)
12. exp151  — Publish constraint propagation models
13. exp152  — Continual learning for constraint retention
14. exp153  — KAN adaptive mesh refinement
```

## Hardware Notes

- **AMD XDNA NPU (Exp 146):** Ryzen AI / Strix Point NPU. Check via `lspci | grep -i xdna`.
  Driver: AMDXDNA kernel module (in-tree on recent kernels ≥6.12). SDK: `pip install amdxdna-sdk`
  or AMD Ryzen AI Software. If unavailable in kernel, document for future session.
- **ROCm:** `JAX_PLATFORMS=cpu` mandatory — ROCm JAX backend crashes on gfx1150. PyTorch ROCm
  works (3.3x speedup) but JAX must stay on CPU for all research experiments.
- **RAM budget:** Qwen3.5-0.8B requires ~2.5GB RAM. Gemma4-E4B-it requires ~8GB. Use
  `CARNOT_FORCE_LIVE=1` with `torch_dtype=float32` via Exp 123 model loader.

## Success Criteria

| Experiment | Success |
|------------|---------|
| Exp 140    | Constraint projection <5ms per token on CPU (publishable latency number) |
| Exp 141    | Memory-augmented extraction improves over AutoExtractor alone on ≥1 domain |
| Exp 142    | Tier 2 (generation) > Tier 1 (reweighting) on arithmetic |
| Exp 143    | ≥1,000 (partial_response, violation) training pairs collected |
| Exp 144    | JEPA predictor AUROC >0.65 on held-out test set |
| Exp 145    | ≥40% fast-path rate with <2% accuracy degradation |
| Exp 146    | NPU benchmarked (pass/fail + latency, or documented environment state) |
| Exp 147    | LLM accuracy drops on adversarial; Carnot verify-repair maintains; delta LARGER on adversarial |
| Exp 148    | Full 1319 GSM8K with live inference, ±95% CI reported |
| Exp 149    | TruthfulQA coverage analysis and top-5 gap constraint types identified |
| Exp 150    | Guided decoding adapter live on HuggingFace; 16 model READMEs updated |
| Exp 151    | 3 domain-specific Ising models published |
| Exp 152    | ContinualGibbs retention >80% (vs LNN 10%) on 5-step chains |
| Exp 153    | KAN refinement implemented; AUROC maintained or improved post-refinement |
