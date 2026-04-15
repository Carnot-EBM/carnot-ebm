# Carnot Research Roadmap v31: Live E2E Precision Pipeline, Constraint Addition from Memory, and EORM Predictive Verification

**Created:** 2026-04-15
**Milestone:** 2026.05.13
**Status:** Planned (activates when milestone 2026.05.06 completes)
**Supersedes:** Milestone 2026.05.06 — "Live GPU Benchmarks, Constraint Precision Analysis, Hardware Unblock, and Conductor Hardening"
**Informed by:** Exps 325-337, operational retrospective 2026.05.06, v30 carry-forwards
**External inputs (new in v31):**
- EORM (2505.14999) — 55M-param energy reward model for CoT ranking, near-perfect accuracy
- SinkProbe (2604.10697) — attention sink analysis as fast hallucination pre-filter
- Eidoku (2512.20664) — neuro-symbolic CSP verification gate for structured LLM reasoning
- LLM-guided SMT (2601.04675) — LLM guidance improves Z3 performance 80%
- Energy-guided decoding for VLMs (2507.07731) — energy layer selection reduces hallucination
- Scalable Ising connectivity (2503.01177) — dense-to-sparse FPGA Ising connectivity analysis
- Online CoT verifier learnability bounds (2603.03538) — soundness/completeness trade-offs

---

## What 2026.05.06 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| Conductor timeout wrapper + test-first stub | 325 | **COMPLETE** | RETRO-001 + NEW-001 implemented; 27% speedup est |
| DualGPUMonitor + GPU zombie detection | 326 | **COMPLETE** | RETRO-002 + RETRO-003 implemented |
| Pre-experiment dependency audit | 327 | **COMPLETE** | NEW-002 implemented; dependency audit CLI ready |
| Live GPU full-scale benchmark (Exp 315 script) | 328 | **COMPLETE** | inference_mode=live_gpu; first real numbers |
| Four-tier self-learning relay on live GPU | 329 | **COMPLETE** | Live relay benchmark established |
| HuggingFace live publish | 330 | **COMPLETE** | 16 EBM READMEs updated; live GPU numbers embedded |
| FP autopsy — broken verify-repair case analysis | 331 | **COMPLETE** | VALID_INTERMEDIATE as dominant FP category |
| Confidence-weighted repair | 332 | **COMPLETE** | 86.7% FP avoided, 100% TP preserved (GATE_EFFECTIVE) |
| Model-adaptive constraint thresholds + selective CaseMemory | 333 | **COMPLETE** | ADAPTIVE_PASS_ATLAS_PARTIAL; range_check auto-disabled |
| VERGE-style iterative Z3 refinement | 334 | **COMPLETE** | Targeted step repair implemented; n_resolved benchmark |
| AMD XDNA NPU build (attempt 5) | 335 | **BLOCKED** | ninja + openblas still missing (4 consecutive milestones) |
| CoTCircuitVerifier — CRV structural verification | 336 | **COMPLETE** | TP/FP benchmark vs Exp 311 complete |
| Operational retrospective 2026.05.06 | 337 | **COMPLETE** | 6 carry-forwards (RETRO-003–008); 38% savings estimate |

**Milestone-level conclusion:**
2026.05.06 delivered the full precision pipeline stack: confidence-weighted repair (86.7% FP
reduction), model-adaptive thresholds, VERGE iterative Z3 refinement, and CoT circuit verification.
DualGPURunner and dependency audit are implemented. First live GPU benchmark numbers exist.

However, the new precision components (Exps 332-336) were benchmarked SYNTHETICALLY. They have
not been tested end-to-end on live RTX 3090 output. We do not know whether the combined precision
pipeline makes verify-repair helpful vs harmful at 1B-3B scale under live conditions.

Additionally, the self-learning loop still does not ADD constraints from memory — it only
reweights existing ones, which Exp 134 proved ineffective. This is research-program.md's #1
priority and it has not been addressed.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: New precision components unverified on live GPU data

The confidence-weighted repair (Exp 332), model-adaptive thresholds (Exp 333), VERGE (Exp 334),
and CoT circuit verifier (Exp 336) were all implemented and benchmarked against synthetic/simulated
responses. The live GPU benchmark (Exp 328) ran against the OLD pipeline (no VERGE, no CRV,
no confidence weighting). We have no measurement of whether the combined precision stack moves
verify-repair from "harmful" to "helpful" on live IT model output.

The comparison we need: baseline (ArithmeticExtractor only) vs precision stack (confidence-weighted
+ adaptive thresholds + VERGE + CRV) on the SAME 200 live GSM8K questions with Gemma4-E4B-it.
If the precision stack shows net positive improvement on live GPU, that is Carnot's first credible
positive result on instruction-tuned models.

**This milestone's priority #1: Run live E2E precision pipeline benchmark with both models,
all new components active. Measure signed improvement vs baseline. If positive, file as
headline result.**

### Gap 2: Self-learning loop adds no new constraints — only reweights existing ones

Research-program.md item #1 (highest priority): "constraint ADDITION from memory patterns, not
just weight changes." Exp 134 proved precision-based reweighting does not improve accuracy.
The Tier 2 CaseMemory has pattern data (which error types are common per model, per domain).
The ConstraintStateMachine (Exp 125) can update which constraint types are active.

The missing piece: a ConstraintTemplateLibrary that CaseMemory can populate with NEW constraint
types based on observed error patterns. When memory detects "carry errors appear in 40% of
Qwen3.5-0.8B arithmetic responses," the library should generate a carry-check constraint template
and ADD it to the active constraint set. This is fundamentally different from upweighting an
existing carry constraint.

**This milestone's priority #2: Implement ConstraintTemplateLibrary, wire CaseMemory into
it, and benchmark constraint addition vs constraint reweighting on 200 questions.**

### Gap 3: JEPA/EORM predictor trained on simulated data — real training pairs now available

The JEPA predictor (Tier 3) was trained on synthetic error patterns from simulated inference.
The live GPU benchmarks (Exps 328-330) now provide real (response, correctness) pairs. EORM
(arXiv 2505.14999) provides a proven 55M-parameter architecture for energy-based CoT ranking
that directly matches Tier 3's goal. SinkProbe (arXiv 2604.10697) provides a fast pre-filter
signal that could replace expensive Ising verification on the fast path.

The combination: SinkProbe (fast, zero-cost signal) → EORM (55M energy ranking) → Ising (slow,
high-accuracy) creates a three-tier fast/medium/slow verification chain that is both faster and
more accurate than any single-tier approach.

**This milestone's priority #3: Train EORM on real live benchmark data, retrain JEPA on
real violation pairs, implement SinkProbe as fast pre-filter.**

---

## Architecture: v31 Additions

```
[Input Query]
    │
    ├─[SinkProbe]──────────────────── attention sink signal   (NEW: Exp 348)
    │  arXiv 2604.10697               fast pre-filter; if sink score < threshold, skip Ising
    │
    ▼
[LLM Generation — RTX 3090 x2, CARNOT_FORCE_LIVE=1]         (ESTABLISHED Exp 328)
    │
    ├─[EORM Energy Ranker]──────────── CoT ranking score       (NEW: Exp 346)
    │  arXiv 2505.14999               55M-param; ranks multiple responses by energy
    │
    ├─[JEPA Fast-Path Gate]──────────── violation predicted?   (RETRAINED: Exp 347)
    │  arXiv 2509.14252               now trained on real violation pairs
    │       ↓ if high energy
    │
    ├─[NL2Z3Extractor]──────────────── Z3 SMT assertions       (EXISTING, Exp 310)
    │       ↓ UNSAT
    │  [VERGE Iterative Loop] ──────── targeted step repair    (EXISTING, Exp 334)
    │       repair only broken step → re-verify
    │
    ├─[CoTCircuitVerifier] ──────────── structural CoT graph   (EXISTING, Exp 336)
    │  arXiv 2510.09312               broken circuit → flag
    │
    ▼
[Ising Verification — CPU fast path]
    │
    ├─[ConfidenceWeightedRepair] ──── score × confidence       (EXISTING, Exp 332)
    │   only repair violations ≥ 0.8
    │
    └─[ModelAdaptiveThresholds] ──── per-model FP tracking     (EXISTING, Exp 333)
    │
    ▼
[Tier 1: Online constraint weight updates]                     (EXISTING)
[Tier 2: CaseMemory → ConstraintTemplateLibrary]              (NEW: Exps 343-345)
    │   error pattern detected → ADD new constraint type
[Tier 3: EORM + JEPA gate + ThresholdAdapter]                 (UPGRADED: Exps 346-347)
    │   real-data trained; 3-tier fast/medium/slow chain
[Tier 4: Adaptive structure — KAN splines]                    (FUTURE)
    │
    ▼
[Hardware backends]
    ├─[FpgaBackend] ─── KV260 open-source bitfile synthesis    (NEW: Exp 349)
    │   yosys + nextpnr synthesis of Exp 291 Verilog RTL
    └─[AMD XDNA NPU] ── still blocked (ninja + openblas)       (CARRY-FORWARD)
```

**New infrastructure (this milestone):**
```
[ops/host-prereqs.md] ────────────── system package registry   (NEW: Exp 338)
    ninja, openblas, CARNOT_FORCE_LIVE entries catalogued
[DualGPURunner default wiring] ───── RETRO-004 fix             (NEW: Exp 338)
    two-model benchmarks auto-assign to separate GPUs
[scripts/session_startup.sh] ─────── pre-session health check  (NEW: Exp 339)
    GPU zombie cleanup, health summary before conductor starts
```

---

## Phase Breakdown

### Phase 1: Infrastructure Carry-Forwards (Exps 338-339)

**Goal:** Close RETRO-003 through RETRO-008 from the 2026.05.06 retrospective.
These have been carried forward; the retro estimates 38% savings if implemented.

- **Exp 338:** Host prereqs registry + DualGPURunner as default (RETRO-004 + RETRO-006)
  - Create `ops/host-prereqs.md` cataloguing all required system packages per experiment class
  - Wire `DualGPURunner` as default for any experiment with two LLM models
  - Add CARNOT_FORCE_LIVE=1 to host-prereqs as a required env var for live benchmark class

- **Exp 339:** Pre-session GPU health + zombie cleanup automation (RETRO-007 + RETRO-008)
  - Write `scripts/session_startup.sh` — zombie GPU process detection and cleanup
  - Integrate into ExperimentTemplate pre-flight check (additive, warn-only)

**Expected outcome:** 38% estimated reduction in milestone wall time (retro estimate).

### Phase 2: Live E2E Precision Benchmark (Exps 340-342)

**Goal:** Benchmark the FULL new precision stack (VERGE + CRV + confidence-weighted + adaptive
thresholds) on live GPU. First credible measurement of whether the combined stack helps.

- **Exp 340:** Live full precision pipeline benchmark
  - Two models on two GPUs (DualGPURunner): Gemma4-E4B-it + Qwen3.5-0.8B
  - 200 GSM8K questions per model (live RTX 3090, CARNOT_FORCE_LIVE=1)
  - Compare: (1) baseline ArithmeticExtractor, (2) +confidence-weighted, (3) +adaptive thresholds,
    (4) +VERGE, (5) full stack (VERGE + CRV + confidence + adaptive)
  - Primary metric: signed net improvement (positive = helpful, negative = harmful)
  - Deliverable: results/experiment_340_live_precision_benchmark.json

- **Exp 341:** Live HumanEval code verification
  - CodeExtractor + runtime execution (most likely domain to still work — no regex extraction)
  - 50 HumanEval questions with Gemma4-E4B-it on live RTX 3090
  - Primary metric: pass@1 improvement with vs without Ising-guided repair
  - Deliverable: results/experiment_341_live_humaneval.json

- **Exp 342:** Live extractor comparison on same responses
  - ArithmeticExtractor vs NL2Z3 vs VERGE vs CoTCircuit on 50 live Gemma4-E4B-it responses
  - Measure per-extractor: violation rate, FP rate, precision (vs ground truth from Exp 331 taxonomy)
  - Deliverable: results/experiment_342_live_extractor_comparison.json

**Expected outcome:** First credible data on precision stack effectiveness. If combined stack
shows positive signed improvement, report as first headline-quality result on live IT models.

### Phase 3: Constraint Addition from Memory (Tier 1/2 Fusion) (Exps 343-345)

**Goal:** Address research-program.md priority #1. Move from constraint reweighting to
constraint addition based on observed error patterns.

- **Exp 343:** ConstraintTemplateLibrary
  - New module: maps observed error patterns to NEW constraint type templates
  - Template API: `add_template(pattern_key, constraint_fn)` → adds to active constraint set
  - Key templates: carry-check, range-check, sign-check, unit-consistency, comparison-direction
  - Addresses Exp 134 finding: reweighting failed; addition must succeed
  - Deliverable: python/carnot/pipeline/constraint_template_library.py

- **Exp 344:** CaseMemory-to-ConstraintTemplateLibrary wiring
  - Wire CaseMemory into ConstraintTemplateLibrary: when pattern threshold met → add template
  - Benchmark: run 200 simulated questions; compare accuracy (reweighting only vs addition)
  - Hypothesis: constraint addition shows improvement where reweighting showed 0% (Exp 134)
  - Deliverable: results/experiment_344_constraint_addition_benchmark.json

- **Exp 345:** Multi-session memory persistence
  - CaseMemory + ConstraintTemplateLibrary persist to disk and reload across sessions
  - Session fingerprinting: track which model was active to load per-model templates
  - Deliverable: python/carnot/pipeline/session_memory.py

**Expected outcome:** Constraint addition from memory showing measurable improvement on simulated
benchmark. Self-learning loop advances from Tier 1 (reweighting) to Tier 1+2 (addition).

### Phase 4: EORM + SinkProbe Predictive Verification (Exps 346-348)

**Goal:** Upgrade Tier 3 with real-data training and a fast pre-filter signal.
EORM (arXiv 2505.14999) + SinkProbe (arXiv 2604.10697) form the fast/medium/slow chain.

- **Exp 346:** EORM-style CoT energy reward model (arXiv 2505.14999)
  - Train 55M-param energy model on (CoT, correctness) pairs from Exp 340 live benchmark
  - Architecture: small transformer encoder → scalar energy → ranking loss
  - Compare vs JEPA gate accuracy on same held-out data
  - Deliverable: python/carnot/models/eorm.py + results/experiment_346_eorm_training.json

- **Exp 347:** JEPA predictor retraining on real violation pairs
  - Collect (partial_response[first N tokens], final_violation_flag) pairs from Exp 340
  - Retrain JEPA on real pairs; compare gate accuracy vs Exp 307/308 simulated baseline
  - Primary metric: AUC-ROC on held-out live benchmark responses
  - Deliverable: results/experiment_347_jepa_real_retrain.json

- **Exp 348:** SinkProbe attention-sink hallucination pre-filter (arXiv 2604.10697)
  - Implement attention sink analysis: compute per-head sink token attention concentration
  - Threshold: if SinkProbe score < τ, skip expensive Ising verification (fast path)
  - Benchmark: SinkProbe FPR/FNR on 50 live responses from Exp 340
  - Multi-signal ensemble: SinkProbe + SpilledEnergy + EORM + Ising
  - Deliverable: python/carnot/pipeline/sink_probe.py + results/experiment_348_sink_probe.json

**Expected outcome:** EORM trained on first real (CoT, correctness) pairs. JEPA retrained on
real violation pairs. SinkProbe provides fast-path skip for low-risk responses.

### Phase 5: Hardware + Operational (Exps 349-350)

**Goal:** Advance KV260 FPGA path and capture operational lessons.

- **Exp 349:** KV260 FPGA bitfile synthesis via open-source toolchain
  - Attempt yosys + nextpnr synthesis of Ising sampler Verilog (Exp 291 RTL)
  - Use sparsification strategy from arXiv 2503.01177 (Scalable Ising connectivity)
  - Incorporate Mpemba-effect annealing from arXiv 2603.24183
  - Success criteria: synthesis script runs, reports LUT/FF utilization (even if not placed)
  - Deliverable: hardware/kv260/carnot_ising_synthesis.sh + results/experiment_349_kv260_synthesis.json

- **Exp 350:** Operational retrospective for milestone 2026.05.13
  - Wall time analysis, per-experiment duration, bottleneck identification
  - Action items for next milestone
  - Deliverable: results/operational_retro_2026_05_13.json

---

## Dependency Graph

```
Exp 338 (host prereqs + DualGPURunner default)  ──────────────────────────────┐
Exp 339 (session startup + zombie cleanup)       ──────────────────────────────┤
                                                                                ↓
Exp 340 (live full precision benchmark)  ──[uses Exps 332-336 pipeline]───────┐
         [DualGPURunner: Gemma4 on GPU0, Qwen3.5 on GPU1]                     │
                                                                                │
Exp 341 (live HumanEval code verification) ─[uses CodeExtractor]──────────────┤
                                                                                │
Exp 342 (live extractor comparison) ─[uses Exp 340 responses]──────────────────┤
                                                                                ↓
Exp 343 (ConstraintTemplateLibrary) ─[new module]──────────────────────────────┐
                                                                                │
Exp 344 (CaseMemory → template wiring) ─[uses Exp 343 + CaseMemory]───────────┤
                                                                                │
Exp 345 (session memory persistence) ─[uses Exp 343 + 344]─────────────────────┤
                                                                                │
Exp 346 (EORM training) ─[uses Exp 340 live pairs]─────────────────────────────┤
                                                                                │
Exp 347 (JEPA real retrain) ─[uses Exp 340 partial responses]──────────────────┤
                                                                                │
Exp 348 (SinkProbe) ─[uses Exp 340 attention tensors]──────────────────────────┘

Exp 349 (KV260 synthesis) ─[independent; uses hardware/kv260/ising_sampler_v1.v]
Exp 350 (retro) ─[uses all Exp 338-349 results]
```

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 340 (live precision benchmark) | 2x RTX 3090 | CARNOT_FORCE_LIVE=1; DualGPURunner; Gemma4 on GPU0, Qwen3.5 on GPU1 |
| Exp 341 (live HumanEval) | 1x RTX 3090 | Single model, CodeExtractor + execution |
| Exp 342 (extractor comparison) | 1x RTX 3090 | Reuses live responses from Exp 340 |
| Exps 338-339 (infra) | CPU | No GPU needed |
| Exps 343-345 (constraint addition) | CPU | Simulation; no live GPU needed |
| Exp 346 (EORM training) | 1x RTX 3090 | 55M-param model training; JAX on GPU |
| Exp 347 (JEPA retrain) | 1x RTX 3090 | JEPA predictor fine-tuning |
| Exp 348 (SinkProbe) | 1x RTX 3090 | Attention tensor analysis on live responses |
| Exp 349 (KV260 synthesis) | CPU | yosys + nextpnr synthesis; no GPU needed |
| Exp 350 (retro) | CPU | Analysis only |

**Hardware actions needed before this milestone:**
1. Ensure both RTX 3090s are visible: `nvidia-smi -L` should show two GPU entries
2. Install ninja+openblas for NPU: `sudo pacman -S ninja openblas` (or apt equivalent)
   — unblocks eventual NPU experiment but does NOT block this milestone

---

## Self-Learning Progress Tracker

| Tier | Status | Experiments |
|------|--------|-------------|
| Tier 1: Online weight updates | IMPLEMENTED (Exp 132-134) | Reweighting proven ineffective |
| Tier 1+2 Fusion: Constraint addition | **NEW THIS MILESTONE** | Exps 343-345 |
| Tier 2: Constraint memory | IMPLEMENTED (Exp 135) | CaseMemory, selective consolidation |
| Tier 3: Predictive verification | UPGRADED THIS MILESTONE | EORM (346) + JEPA real-retrain (347) |
| Tier 4: Adaptive structure | FUTURE | KAN adaptive mesh; depends on Tier 3 |

**Tier 3 fast/medium/slow chain (this milestone):**
```
SinkProbe (fast, 0ms overhead) → EORM ranker (medium, 55M GPU) → Ising (slow, 0.006ms/check)
Each layer gates the next. SinkProbe reduces Ising calls by estimated 40-60%.
```

---

## New Papers to Incorporate

| Paper | Exp | What it enables |
|-------|-----|-----------------|
| EORM (2505.14999) | 346 | Energy reward model for CoT ranking trained on real data |
| SinkProbe (2604.10697) | 348 | Fast attention-based pre-filter replacing expensive Ising on easy queries |
| Eidoku (2512.20664) | 343 | Constraint type taxonomy for ConstraintTemplateLibrary |
| LLM-guided SMT (2601.04675) | 342 | Z3 performance improvement via LLM-guided quantifier elimination |
| Energy-guided decoding (2507.07731) | 348 (future) | Layer selection insight for token probability adjustment |
| Scalable Ising connectivity (2503.01177) | 349 | Sparsification strategy for KV260 FPGA bitfile |
| CoT verifier learnability (2603.03538) | 343-344 | Soundness/completeness bounds for constraint addition |
