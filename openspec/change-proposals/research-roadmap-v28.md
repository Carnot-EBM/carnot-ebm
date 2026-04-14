# Carnot Research Roadmap v28: Apple Adversarial Completion, Dual-Energy Gate, Constraint Addition, and NPU Hardware

**Created:** 2026-04-14
**Milestone:** 2026.04.22
**Status:** Planned (activates when milestone 2026.04.21 completes)
**Supersedes:** Milestone 2026.04.21 — "Apple Adversarial GSM8K, Extraction-Free Detection, and FPGA Hardware Bring-up"
**Informed by:** Exps 281-293, operational retrospective 2026-04-14, v27 carry-forwards
**External inputs (new in v28):**
- Neural Uncertainty Principle (2603.19562) — prefill-stage hallucination probe
- LogitScope/Varentropy (2603.24929) — entropy variance signal
- SciDC (2604.06603) — multi-layer formal decoding constraints, +12% on science tasks
- Talking with Verifiers (2603.02235) — NL→Z3 auto-spec generation
- Digitally Optimized Thermodynamic Init (2603.24183) — Mpemba-effect Ising thermalization
- Ising Sampling Advantage Predictor (2504.18359) — hardware routing decision metric
- CoT Verifier Online Learnability (2603.03538) — soundness/completeness bounds
- Likelihood-Based Reward Designs (2602.03979) — EBM energy as continuous RL signal
- Hardware Acceleration Frustrated Lattice RBM (2511.20911) — 10^3-10^5x FPGA speedup

---

## What 2026.04.21 Proved (Apple Adversarial + FPGA + Extraction-Free)

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| Apple adversarial dataset generator | 281 | **COMPLETE** | 400 adversarial rows, 100% variant coverage |
| Apple adversarial GPU baseline | 282 | **SCRIPTS READY, GPU STALLED** | Logits not produced (conductor GPU stall) |
| Apple adversarial verify-repair | 283 | **SCRIPTS READY, GPU STALLED** | Logits not produced (conductor GPU stall) |
| Apple adversarial analysis | 284 | **INCONCLUSIVE** | 2nd consecutive milestone stalled |
| SpilledEnergyExtractor | 285/REQ-VERIFY-076 | **IMPLEMENTED** | Module complete, awaits real logits |
| FpgaBackend quantum-inspired sparse Ising | 289 | **IMPLEMENTED** | quantum_annealing_schedule, LagONN, AXI |
| FpgaBackend vs CPU benchmark | 290 | **CONFIRMED** | Geometric β wins 3/3 problem sizes |
| JEPA Apple adversarial retrain | 291 | **TARGETS_MET (synthetic)** | TP=1.0, FP=0.0 on synthetic; needs real logits |
| AMD XDNA NPU VitisAI | 292 | **BLOCKED** | Missing: ninja, openblas |
| HuggingFace publish | 293 | **SCRIPT READY** | Blocked by huggingface-cli login credentials |

**Milestone-level conclusion:** The primary credibility experiment (Apple adversarial benchmark, Exps
282/283) failed to produce results for the SECOND consecutive milestone due to GPU inference stall.
The GPU stall root cause must be diagnosed and fixed before any further live-GPU experiments.
Scripts are complete and correct — the only blocker is runtime GPU stall during conductor runs.
SpilledEnergyExtractor and FpgaBackend were successfully implemented; the JEPA predictor has correct
architecture but was trained on synthetic data and requires real logits.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Apple adversarial benchmark still INCONCLUSIVE after 2 milestones

The Apple adversarial GSM8K benchmark has been in the roadmap for 4 consecutive milestones. The GPU
stall issue is the root cause. The diagnosis: during conductor runs (non-interactive `claude -p`
subprocesses), the RTX 3090 inference stalls at the 60s timeout boundary. Both RTX 3090s show
0% utilization and 2MB residual VRAM at milestone end — they are IDLE, not busy.

This milestone's priority #1 is diagnosing the GPU stall, then re-running Exps 282/283. Without
the Apple adversarial result, Carnot has no credible positive result on a real benchmark (after
280+ experiments). This is the mission-critical gap.

**The stall hypothesis:** `DualGPURunner` initializes model loading lazily in the subprocess, and
the model load itself stalls (not the inference). The fix: add explicit model pre-warm before
benchmarking, with a separate health-check that proves the model is responding before timing starts.

### Gap 2: Extraction-free detection is implemented but untested on real data

SpilledEnergyExtractor (Exp 285) is implemented and awaits logit files from Exps 282/283. The
Semantic Energy signal (2508.14496 — catches confident-wrong outputs) is NOT YET implemented.
The Neural Uncertainty Principle (2603.19562) provides a complementary pre-generation probe.
LogitScope/Varentropy (2603.24929) adds a third signal. Together, these form a three-signal
extraction-free detection system that requires ZERO constraint extraction and NO knowledge base.

The dual-energy gate design (spilled + semantic → fast filter → trigger expensive Ising) is
designed but not built. Without the gate, every query runs the full Ising pipeline regardless
of whether it's likely to have violations — this is 10-100x more expensive than necessary.

### Gap 3: Self-learning Tier 1+2 still doesn't improve accuracy

The research program identified the fix: constraint ADDITION (not reweighting). Exp 134 proved
that precision-based reweighting of existing constraints gives 0% improvement. Exp 135/136/141
showed that constraint memory accumulates patterns but never generates new constraint types.
The fix is explicit: when memory shows "arithmetic carry errors are common," ADD a carry-check
constraint to the IsingModel, not just upweight the existing addition constraints.

The theoretical bounds from CoT Verifier Online Learnability (2603.03538) tell us WHICH constraint
types are online-learnable without soundness violations — this prevents the constraint addition from
breaking correct answers. Implement memory-to-constraint generation with these soundness bounds.

---

## Architecture: v28 Additions

```
[Input Query]
    │
    ├─[PrefillUncertaintyProbe]─── prefill uncertainty (pre-generation gate)
    │      arXiv 2603.19562       ↓
    │                          [SKIP SLOW PATH if low uncertainty]
    │
    ▼
[LLM Generation]
    │
    ├─[SpilledEnergyExtractor]─── spilled energy (high entropy = uncertain)
    │      arXiv 2602.18671       │
    ├─[SemanticEnergyExtractor]── semantic energy (low entropy, wrong = confident-wrong)
    │      arXiv 2508.14496       │
    ├─[VarEntropyProbe]────────── varentropy (entropy variance = uncertainty-about-uncertainty)
    │      arXiv 2603.24929       │
    │                          [DUAL-ENERGY GATE: any signal fired?]
    │                             │ NO: fast path (no verification)
    │                             │ YES: route to slow path
    │                             ▼
    ├─[FormalClaimVerifier]────── Z3/SMT hard constraints
    ├─[IsingVerifier]──────────── Ising energy constraint check
    │      FpgaBackend (KV260)    │
    │                          [Violation detected?]
    │                             │ NO: pass through
    │                             │ YES: trigger repair
    │                             ▼
    └─[VerifyRepairPipeline]───── constrained repair + self-learning update
           [ConstraintGenerator]── memory pattern → NEW constraint type (Tier 2→1)
           [TrackerUpdate]──────── online weight update (Tier 1)
```

New components in v28:
- **PrefillUncertaintyProbe** (Exp 298): Pre-generation hallucination risk assessment
- **SemanticEnergyExtractor** (Exp 297): Boltzmann logit energy for confident-wrong detection
- **VarEntropyProbe** (Exp 297): Varentropy baseline signal, model-agnostic
- **ConstraintGenerator** (Exp 301): Memory patterns → new IsingModel constraint types
- **GPU Stall Diagnosis** (Exp 294 prerequisite): root cause analysis for conductor GPU stall

---

## Phase Plan

### Phase 1: Apple Adversarial Completion (Exps 294-296)
**Goal:** Produce the credibility result that has been INCONCLUSIVE for 2 milestones.
**Critical dependency:** GPU stall must be diagnosed BEFORE Exps 295/296.

| Exp | Title | Deliverable | Key technique |
|-----|-------|-------------|---------------|
| 294 | GPU stall diagnosis + Apple baseline re-run | results/experiment_294_results.json | Model pre-warm, explicit health-check, stall root cause analysis |
| 295 | Apple adversarial verify-repair re-run | results/experiment_295_results.json | Three modes × two variants × two models |
| 296 | Apple adversarial analysis + docs | results/experiment_296_results.json | CONFIRMED/PARTIAL/RULED_OUT classification |

**Expected outcome:** CONFIRMED (semantic grounding detects stale answers from number-swap with
100% recall, same as Exp 279). If still INCONCLUSIVE due to GPU stall, document root cause.

### Phase 2: Extraction-Free Detection Expansion (Exps 297-299)
**Goal:** Implement the full three-signal extraction-free detection system.
**Dependency:** Exps 294/295 must produce logit files for JEPA retraining.

| Exp | Title | Deliverable | Key technique |
|-----|-------|-------------|---------------|
| 297 | SemanticEnergyExtractor + VarEntropyProbe | python/carnot/pipeline/semantic_energy_extractor.py | Boltzmann logit energy (2508.14496) + varentropy (2603.24929) |
| 298 | PrefillUncertaintyProbe | python/carnot/pipeline/prefill_uncertainty_probe.py | Conjugate uncertainty bound (2603.19562) |
| 299 | JEPA retrain on real Apple adversarial logits | results/experiment_299_results.json + results/jepa_predictor_299.onnx | 8-feature real logit training, isotonic calibration |

### Phase 3: Constraint Addition from Memory (Exps 300-302)
**Goal:** Fix Tier 1 self-learning by adding new constraint types from memory patterns.
**Key fix:** Memory detects error pattern → generate new IsingModel constraint → add to pipeline.

| Exp | Title | Deliverable | Key technique |
|-----|-------|-------------|---------------|
| 300 | Memory-to-Constraint Generator | python/carnot/pipeline/constraint_generator.py | Pattern → constraint type, soundness bounds (2603.03538) |
| 301 | Confidence-weighted constraints | python/carnot/pipeline/confidence_verifier.py | Continuous energy as violation confidence; only repair high-confidence (2602.03979) |
| 302 | Self-learning integrated benchmark | results/experiment_302_results.json | Constraint addition + confidence weighting on live Gemma4 |

### Phase 4: Hardware + Publishing (Exps 303-306)
**Goal:** Unblock NPU, publish to HuggingFace, synthesize KV260 Verilog.

| Exp | Title | Deliverable | Key technique |
|-----|-------|-------------|---------------|
| 303 | AMD XDNA NPU unblock | results/experiment_303_npu_results.json | sudo pacman -S ninja openblas, ORT source build |
| 304 | HuggingFace actual publish | results/experiment_304_hf_results.json | huggingface-cli login + Exp 293 script run |
| 305 | KV260 Verilog Ising sampler first synthesis | hardware/kv260/ising_sampler_v1.v | Mpemba-effect init (2603.24183), quantum schedule, sparse coupling |
| 306 | Operational efficiency improvements | scripts/experiment_template.py | Scaffolding template, DualGPU from Exp 1, inference batching |

---

## Dependency Graph

```
Exp294 (GPU stall diagnosis + Apple baseline)
    └─→ Exp295 (verify-repair re-run)
            └─→ Exp296 (analysis + docs)
            └─→ Exp299 (JEPA retrain on real logits)

Exp297 (SemanticEnergyExtractor + VarEntropy)
    └─→ [standalone, no dependencies]

Exp298 (PrefillUncertaintyProbe)
    └─→ [standalone, no dependencies]

Exp300 (Memory-to-Constraint Generator)
    └─→ Exp301 (ConfidenceVerifier)
            └─→ Exp302 (Self-learning benchmark)

Exp303 (NPU unblock) ─── [standalone]
Exp304 (HF publish) ──── [standalone, needs HF credentials]
Exp305 (KV260 Verilog) ─ [standalone]
Exp306 (Scaffolding) ─── [standalone, no blocking deps]
```

---

## Hardware Requirements

| Exp | Hardware | Requirement |
|-----|----------|-------------|
| 294-296 | 2x RTX 3090 | GPU stall diagnosis; both GPUs required for DualGPURunner |
| 299 | 2x RTX 3090 | JEPA training on real logits from Exp 294/295 |
| 302 | 2x RTX 3090 | Live Gemma4 benchmark for self-learning |
| 303 | AMD XDNA NPU | Source build ORT after ninja+openblas install |
| 305 | KV260 FPGA | Synthesize Verilog (can be done on host CPU via Vivado) |

**GPU stall note:** The research-hardware-wishlist.md confirms 2x RTX 3090 (48GB VRAM total, CUDA)
are connected and have been working. The stall is a software issue (conductor subprocess model
loading), not hardware. Pre-warm + health-check pattern should fix it.

---

## Success Criteria for This Milestone

1. **Apple adversarial classified (not INCONCLUSIVE)** — produces CONFIRMED, PARTIAL, or RULED_OUT
2. **SemanticEnergyExtractor + VarEntropyProbe implemented** with 100% test coverage
3. **PrefillUncertaintyProbe implemented** with 100% test coverage
4. **Constraint addition (ConstraintGenerator) implemented** — memory pattern → new constraint type
5. **Self-learning benchmark shows improvement** vs baseline (even +1pp is a first-ever positive)
6. **AMD NPU unblocked** — VitisAI EP loaded in at least one successful ONNX inference
7. **HuggingFace models published** — at least one model card live at Carnot-EBM
8. **KV260 Verilog synthesis attempted** — bitfile generated or synthesis failure documented

---

## What This Milestone Does NOT Include

- Z3/SMT auto-spec generation from NL (Talking with Verifiers, 2603.02235) — needs dedicated milestone
- SciDC multi-layer decoding constraints — needs guided decoding milestone
- Full Apple adversarial with 1,319 questions (scale benchmark) — after credibility result confirmed
- KAN adaptive mesh refinement — after KV260 baseline proven
- JEPA guided decoding (Tier 3 full pipeline) — after JEPA predictor trained on real data
- Exp 53 re-run (slowest experiment, 418 min, flagged 5 milestones) — isolated infrastructure milestone

---

## Carry-Forwards from Previous Milestones

| Item | Origin | Status | This Milestone |
|------|--------|--------|----------------|
| Apple adversarial credibility result | v25/v26/v27 | INCONCLUSIVE x2 | Primary focus (Phase 1) |
| GPU stall root cause | 2026.04.21 retro | Unresolved | Exp 294 Phase 1 |
| AMD NPU source build | Exp 292 | Blocked (ninja+openblas) | Exp 303 Phase 4 |
| HuggingFace credentials | Exp 293 | Blocked (login) | Exp 304 Phase 4 |
| Real logits for JEPA | Exps 282/283 | Not produced | Exp 299 (after Exp 294/295) |
| SemanticEnergyExtractor | v27 Gap 2 | Not implemented | Exp 297 Phase 2 |
| Memory constraint addition | research-program.md | Not implemented | Exp 300-302 Phase 3 |
| KV260 Verilog synthesis | Exp 228, Exp 288 | Design only | Exp 305 Phase 4 |
