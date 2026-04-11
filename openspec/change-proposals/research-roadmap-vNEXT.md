# Carnot Research Roadmap v16: Multi-Domain JEPA + Factual Extraction + Live Model Benchmarks

**Created:** 2026-04-11
**Milestone:** 2026.04.11
**Status:** Planned (activates when milestone 2026.04.10 completes)
**Supersedes:** research-roadmap-vNEXT.md (milestone 2026.04.10 / v15)
**Informed by:** Exp 141 (+0.11 accuracy from memory-augmented generation), Exp 144 (JEPA arithmetic AUROC=0.71), Exp 145 (fast-path architecture works, predictor insufficient), Exp 147 (adversarial results not statistically significant — p=0.463), Exp 151 (HF models published), Exp 152 (ContinualGibbs 100% accuracy), Exp 153 (KAN AMR -1.3% params)
**ArXiv inputs:** 2602.18671 (EBM hallucination ICLR 2026), 2602.03417 (FactNet KB), 2507.02092 (EBTs), 2604.04606 (FPGA Ising 6x), 2508.10480 (Πnet hard constraints)

## What 2026.04.10 Proved

- **Memory-augmented constraint generation works** — Exp 141: `ConstraintGenerator` wiring
  `ConstraintMemory` into `AutoExtractor` boosted accuracy +0.11 (0.85 → 0.96) on 200 GSM8K
  questions. `comparison_boundary` recall jumped 0% → 100%. Constraint ADDITION beats
  reweighting (Exp 134), confirming the research-program.md diagnosis.
- **JEPA architecture is correct, predictor quality is the bottleneck** — Exp 144 achieved
  arithmetic AUROC=0.7126 (target met). But code/logic AUROC=0.5 (chance) because Exp 143
  only had arithmetic training data. Exp 145: fast-path gate fires and runs correctly, but
  19.8% accuracy degradation (vs <2% target) — entirely caused by missing code/logic signal.
  Fix is clear: collect multi-domain training pairs and retrain.
- **AMD XDNA NPU hardware detected, software blocked** — Exp 146: `/dev/accel0` and
  `amdxdna` kernel module confirmed. Blocker: `onnxruntime-vitisai` requires AMD conda channel,
  not PyPI. ONNX model exported, CPU baseline p50=0.005ms. NPU would help at sustained load.
- **Adversarial results positive but underpowered** — Exp 147: number-swapped recovery
  +27pp (Qwen) / +24.5pp (Gemma) vs +10pp / +13pp on standard GSM8K — hypothesis direction
  confirmed. But p=0.463, N=6 adversarial vs N=2 control data points. Need live models +
  full dataset (N≥100/variant) for statistical significance.
- **ContinualGibbs achieves 100% step-5 accuracy** — Exp 152: orthogonal Gram-Schmidt
  projection prevents catastrophic forgetting in multi-step agent chains. Beats LNN (90%)
  and matches static Ising (100%) while learning across steps.
- **KAN AMR maintains accuracy with fewer params** — Exp 153: adaptive mesh refinement
  -1.3% params at same AUROC. High-curvature edges on domain×numeric cross-interactions
  (the interesting nonlinear regions).

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: JEPA Fast-Path Not Yet Working (Tier 3 Self-Learning, FR-11)

The fast-path gate architecture is complete and integrated into `VerifyRepairPipeline`. The
bottleneck is purely data: Exp 143 collected only arithmetic training pairs, so code/logic
AUROC=0.5. The fix requires three steps: (1) collect multi-domain (partial_response, violation)
pairs for code and logic domains, (2) retrain the JEPA predictor with balanced domain coverage,
(3) benchmark the improved predictor to confirm <2% degradation at ≥40% fast-path rate.
This is the highest-priority incomplete item in the self-learning roadmap.

### Gap 2: Factual Domain Still Blind (Goal #3, REQ-VERIFY-001)

Exp 88 measured 100% false negative rate on factual claims. Exp 122 confirmed: 66.9% of
adversarial errors are structurally uncatchable by arithmetic constraint verification. The
factual domain requires a fundamentally different approach — external knowledge grounding.
Two recent papers provide the path: FactNet (2602.03417, 1.7B assertions) as the KB, and
"spilled energy" detection (2602.18671, ICLR 2026) as a fast pre-filter. A factual claim
extractor that maps LLM outputs to FactNet triples would close the biggest coverage gap in
the entire pipeline.

### Gap 3: No Statistically Significant Live Benchmark (Goals #5, #6)

All benchmarks with strong results use simulated inference (CARNOT_SKIP_LLM=1). The
adversarial GSM8K result had p=0.463. The eGPU (RX 7900 XTX, 24GB VRAM) is available
in hardware but not connected. Once connected, gfx1100 (unlike gfx1150) has full ROCm + JAX
support, enabling: full GSM8K (1,319 questions), full adversarial battery (200/variant × 4
variants), HumanEval 164 problems — all with live inference and N large enough for significance.

## Architecture (2026.04.11)

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                Carnot Pipeline (2026.04.11)              │
                    │                                                          │
  LLM generates ──▶ │  ┌─────────────────────────────────────────────────┐    │
  tokens (partial)  │  │ SpilledEnergyExtractor (NEW, arxiv 2602.18671)  │    │
                    │  │ Fast pre-filter: logit vs output energy delta    │    │
                    │  └───────────────────┬─────────────────────────────┘    │
                    │                      │ high spilled energy?              │
                    │  ┌─────────────┐     ▼     ┌────────────────────────┐   │
                    │  │JEPA Predictor│ (FIXED)  │ FactualExtractor (NEW) │   │
                    │  │v2: multi-domain│────────▶│ FactNet triples →     │   │
                    │  │AUROC >0.70  │           │ IsingConstraints       │   │
                    │  └──────┬──────┘           └────────────────────────┘   │
                    │         │ fast-path / slow-path                         │
                    │         ▼                                               │
                    │  ┌─────────────────────────────────────────────────┐    │
                    │  │ AutoExtractor + ConstraintGenerator (Exp 141)  │    │
                    │  │ memory-augmented: +0.11 accuracy proven         │    │
                    │  └──────────────────┬──────────────────────────────┘   │
                    │                     │                                   │
                    │                     ▼                                   │
                    │  ┌─────────────────────────────────────────────────┐    │
                    │  │ ComposedEnergy + KAN verifier (Exp 108)        │    │
                    │  │ Ising MCMC (183x vs thrml, Exp 71)             │    │
                    │  └──────────────────┬──────────────────────────────┘   │
                    │                     │                                   │
                    │                     ▼                                   │
                    │  ┌─────────────────────────────────────────────────┐    │
                    │  │ ConstraintStateMachine + rollback (Exp 125-127)│    │
                    │  │ ContinualGibbs for multi-step chains (Exp 152) │    │
                    │  └─────────────────────────────────────────────────┘    │
                    └──────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────────────────┐
                    │              Hardware Path (2026.04.11)                 │
                    │  eGPU (RX 7900 XTX): live LLM inference — CONNECT NOW  │
                    │  CPU: AutoExtractor, ConstraintMemory, KAN verify       │
                    │  NPU (XDNA, blocked): JEPA predictor (conda AMD stack) │
                    │  FPGA (Kria KV260, future): sparse Ising (arxiv 2604.04606) │
                    └──────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
Exp 154 (multi-domain pairs) ──▶ Exp 155 (retrain JEPA v2) ──▶ Exp 156 (fast-path v2)
                                                                      │
Exp 157 (spilled energy extractor) ──▶ Exp 158 (FactNet KB extractor)  │
                │                         │                            │
                └──────────────────────┐  │                            │
                                       ▼  ▼                            ▼
                              Exp 159 (5-domain benchmark) ──▶ Exp 161 (full live GSM8K)
                                                                       ▲
Exp 160 (eGPU setup) ─────────────────────────────────────────────────┘
                           │
                           ├──▶ Exp 162 (adversarial live, N≥200/variant)
                           └──▶ Exp 163 (HumanEval live + repair)

Exp 164 (HuggingFace publishing) [independent]
Exp 165 (ArXiv scan, next milestone prep) [independent]
```

## Phase 44: Fix JEPA to Multi-Domain (Experiments 154–156)

The JEPA fast-path architecture (Exp 145) works correctly. The predictor quality is bottlenecked
by single-domain training data. These three experiments fix the data → model → integration path.

### Exp 154: Multi-Domain JEPA Training Data Collection

Collect (partial_response, violation) pairs for code and logic domains to supplement the
arithmetic-only Exp 143 dataset.

**Deliverable:** `results/jepa_training_pairs_v2.json`

Generate 200 code verification pairs (Python functions + constraints) and 200 logic pairs
(NL reasoning + logic constraints). Apply prefix-ratio extraction (10%, 25%, 50%, 75%) with
RandomProjection (same as Exp 143: 256-dim, seed=42). Target ≥800 multi-domain pairs total.
Label with per-domain violation flags.

### Exp 155: JEPA Predictor v2 — Retrain with Multi-Domain Data

Retrain `JEPAViolationPredictor` on the combined Exp 143 (arithmetic) + Exp 154 (code+logic)
dataset.

**Deliverable:** `results/jepa_predictor_v2.safetensors`

Target: macro AUROC >0.70 across all three domains (arithmetic, code, logic).
Use same architecture (256→64→32→3) but ensure balanced domain sampling during training.
Report per-domain AUROC improvement vs Exp 144 (arithmetic=0.71, code=0.50, logic=0.50).

### Exp 156: JEPA Fast-Path Benchmark v2

Re-run the Exp 145 benchmark with the improved predictor.

**Deliverable:** `scripts/experiment_156_jepa_fastpath_v2.py`, `results/experiment_156_results.json`

Same 500-question benchmark (200 arithmetic, 200 code, 100 logic). Target: ≥40% fast-path
taken with <2% accuracy degradation. Compare threshold=0.3 / 0.5 / 0.7.
Update `VerifyRepairPipeline` to load `jepa_predictor_v2.safetensors` by default when present.

## Phase 45: Factual Constraint Extraction (Experiments 157–159)

The factual domain has 100% false negative rate (Exp 88). These experiments build the knowledge-
grounded extractor and measure its impact. Incorporates two new arxiv papers.

### Exp 157: Spilled Energy Pre-Filter

Implement the "spilled energy" signal from arxiv 2602.18671 (ICLR 2026) as a fast factual
plausibility detector. No external KB required.

**Deliverable:** `python/carnot/pipeline/spilled_energy.py`

"Spilled energy" = sum of max(0, logit_energy - output_energy) across token positions, where
logit_energy = -log(softmax(logits)) at the output token and output_energy = -log(p_output).
High spilled energy → model is "uncertain" about its generation → factual hallucination risk.
Measure AUROC on TruthfulQA (factual domain ground truth).
This is orthogonal to Ising constraint verification — acts as a fast pre-filter.

### Exp 158: Wikidata/FactNet-Backed Factual Claim Extractor

Build the `FactualExtractor` that maps LLM claims to knowledge-base triples for Ising encoding.

**Deliverable:** `python/carnot/pipeline/factual_extractor.py`

Design:
- Entity extraction: spaCy NER to identify people, places, dates, organizations
- Claim decomposition: subject-predicate-object triples from NL text
- KB lookup: Wikidata SPARQL API for entity relations (use cached/offline subset for speed)
- Encoding: each (subject, predicate, object) triple becomes a binary ConstraintTerm:
  satisfied if LLM's claimed value matches KB value, violated otherwise
- Test on TruthfulQA subset (50 factual questions): measure constraint coverage (was ~0%, target >30%)

### Exp 159: Full 5-Domain Benchmark with Factual Extractor

Run the full pipeline benchmark including the new factual extractor.

**Deliverable:** `scripts/experiment_159_five_domain.py`, `results/experiment_159_results.json`

Extend the Exp 93 benchmark (250 questions × 2 models × 3 modes) to include factual domain
with `FactualExtractor`. Target: factual domain accuracy improvement >0pp (even small is a win
vs the current 0% constraint coverage). Report constraint coverage per domain.

## Phase 46: eGPU Setup + Live Model Benchmarks at Scale (Experiments 160–163)

The RX 7900 XTX eGPU is owned hardware but not connected. These experiments establish live
inference and run the high-N benchmarks that produce publishable statistics.

### Exp 160: eGPU Hardware Setup and ROCm/JAX Validation

Connect the RX 7900 XTX via Thunderbolt and validate the full ML stack.

**Deliverable:** `scripts/experiment_160_egpu_setup.py`, `results/experiment_160_results.json`

Steps: `rocminfo | grep gfx` (expect gfx1100, not gfx1150), JAX GPU test, PyTorch GPU test,
Qwen3.5-0.8B inference speed comparison (CPU vs eGPU). Update `research-program.md` constraints
if JAX works (remove `JAX_PLATFORMS=cpu` requirement). Baseline: CPU Qwen3.5-0.8B inference
latency (p50, p99 per token). Target: >5x speedup on eGPU vs CPU.

**Hardware note:** This experiment requires physical hardware connection (Thunderbolt chassis).
If the Thunderbolt connection is unavailable, document the blocker and proceed with CPU-only
inference for subsequent experiments using `CARNOT_SKIP_LLM=1` simulation as fallback.

### Exp 161: Full GSM8K Live Benchmark at Scale

Run the full 1,319-question GSM8K test set with live model inference.

**Deliverable:** `scripts/experiment_161_gsm8k_full.py`, `results/experiment_161_results.json`

Modes: baseline / verify-only / verify-repair × Qwen3.5-0.8B × Gemma4-E4B-it.
Report: accuracy, 95% bootstrap CI (N=1319 → CI < ±3pp), comparison to published baselines
(GPT-4 87.1%, Llama2-70B 56.8%). This produces the first Carnot result with rigorous CIs.
Requires live inference from Exp 160 (or simulation fallback with `CARNOT_SKIP_LLM=1`).

### Exp 162: Apple Adversarial GSM8K with Statistical Power

Re-run the Exp 147 adversarial experiment with live models and N=200/variant.

**Deliverable:** `scripts/experiment_162_adversarial_live.py`, `results/experiment_162_results.json`

Use Exp 119's adversarial dataset (800 items: 200/variant × 4 variants). Run all 4 variants
(control, number-swapped, irrelevant-injected, combined) × 3 modes × 2 models = 4,800
inference calls. Target: p<0.05 (two-sided permutation test) for "adversarial improvement >
control improvement." With N=200/variant this should be achievable if the effect is real.
This addresses the p=0.463 statistical weakness in Exp 147.

### Exp 163: HumanEval Full 164 Problems with Live Repair

Run the full HumanEval benchmark (164 problems) with live code generation and repair.

**Deliverable:** `scripts/experiment_163_humaneval_full.py`, `results/experiment_163_results.json`

Modes: pass@1 baseline / pass@1+verify / pass@1+verify+repair × Qwen3.5-0.8B.
Use CodeExtractor + runtime instrumentation (Exp 53). Target: pass@1+repair > 90% (Exp 68
established 96% on 50-problem subset with simulated inference).
Report: comparison to published HumanEval baselines (GPT-4 86.5%, Llama2-70B 29.9%).

## Phase 47: Publication + ArXiv Scan (Experiments 164–165)

### Exp 164: HuggingFace Publishing Sprint

Push all packaged-but-unpublished artifacts to HuggingFace.

**Deliverable:** Published model cards at huggingface.co/Carnot-EBM

Artifacts to publish:
1. `exports/guided-decoding-adapter/` (Exp 137 — packaged, not pushed)
2. `exports/constraint-propagation-models/{arithmetic,logic,code}/` (Exp 151 — packaged, not pushed)
3. Update existing 16 per-token EBM READMEs to point to `pip install carnot`
4. New model card: JEPA violation predictor v2 (Exp 155 result)

Use `huggingface-cli` or `huggingface_hub` Python API. Verify each upload with `hf_hub_download`.

### Exp 165: ArXiv Research Scan (Next Milestone Prep)

Scan arxiv for new papers to inform Milestone 2026.04.12.

**Deliverable:** Updated `research-references.md`, `results/experiment_165_arxiv_scan.json`

Queries: ebm_reasoning_2026, factual_verification_kg, jepa_prediction, kan_energy_new,
fpga_ising_2026, constrained_generation_llm, orthogonal_projection_constraint,
spilled_energy_ebm, continual_learning_constraint, thermodynamic_ebm_hardware
Update `research-references.md` with top 10 findings and proposed next experiments.

## Hardware Requirements

| Experiment | Hardware | Available |
|------------|----------|-----------|
| Exp 154-159 | CPU (JAX) | Yes |
| Exp 160 | RX 7900 XTX + Thunderbolt chassis | **Owned, needs connection** |
| Exp 161-163 | eGPU (or simulation fallback) | Conditional on Exp 160 |
| Exp 164 | CPU + internet | Yes |
| Exp 165 | CPU + internet | Yes |

**Priority action:** Connect RX 7900 XTX via Thunderbolt before Exp 160. This unblocks
Goals #1, #5, #6 and is the single highest-leverage hardware action available.

## Self-Learning Advancement

This milestone advances all four self-learning tiers:

| Tier | This Milestone |
|------|----------------|
| 1: Online weights | No new work (Exp 134 complete) |
| 2: Constraint memory | Exp 159 benchmarks memory-augmented generation at 5-domain scale |
| **3: Predictive verify** | **Exp 154-156: fix JEPA to multi-domain → unblock fast-path** |
| 4: Adaptive structure | No new work (Exp 153 KAN AMR complete) |

The Tier 3 fix (JEPA multi-domain) is the core self-learning advance for this milestone.
Once the fast-path gate works with <2% degradation, Carnot routes easy queries through
a sub-1ms path — inference gets faster with every example seen.

## Success Criteria

| Experiment | Target | Why It Matters |
|------------|--------|----------------|
| Exp 155 (JEPA v2) | macro AUROC >0.70 | Proves multi-domain training fixes the gap |
| Exp 156 (fast-path v2) | ≥40% fast-path, <2% degradation | Tier 3 self-learning complete |
| Exp 158 (factual) | factual coverage >30% | Closes biggest blind spot |
| Exp 161 (full GSM8K) | CI < ±3pp, beats simulated baseline | Credible external number |
| Exp 162 (adversarial) | p<0.05 adversarial > control | Apple thesis validated |
| Exp 163 (HumanEval) | pass@1+repair > 90% | Code domain credibility |
