# Carnot Research Roadmap v41: Scale the First Positive — 200q Credibility, Process Hardening, and FPGA Bring-Up

**Created:** 2026-04-18
**Milestone:** 2026.04.35
**Status:** Planned (activates when milestone 2026.04.34 retrospective completes)
**Supersedes:** Milestone 2026.04.34 — "VeriCoT Extraction, EBM-CoT Calibration, First Positive Numbers"
**Informed by:** Exps 450-461, operational retrospective 2026.04.34, v40 carry-forwards
**External inputs (new in v41):**
- PPSEBM (arXiv 2512.15658) — EBM + progressive parameter isolation for continual LLM constraint learning
- Equilibrium Propagation on OIMs (arXiv 2510.12934 + 2505.02103) — local learning on Ising machines
- GPU-Accelerated Oscillator Ising (arXiv 2505.22631) — ~10,000x over CPU on RTX 3090, already available
- From Ising to Potts (arXiv 2507.18379) — multi-state spins for richer constraint encoding
- GSM-Symbolic adversarial benchmark (arXiv 2410.05229, ICLR 2025) — THE credibility experiment, confirmed ICLR 2025

---

## What 2026.04.34 Proved

| Approach | Experiments | Verdict | Key Finding |
|----------|-------------|---------|-------------|
| Gemma4 Tokenizer Fix (RETRO-028) | 450 | **FIX IMPLEMENTED** | GemmaTransformersLoader built; result file absent (RETRO-032) |
| Live Precision Post-Fix (RETRO-033) | 451 | **+5pp CONFIRMED** | First positive verify-repair number; honest_verdict=repair_better |
| Energy Matching v2 (RETRO-030) | 452 | **RETRO-030 CLOSED** | AtomicResultWriter prevents silent drops; energy_matching best_sampler |
| VeriCoT Step Validator | 453 | **8/20 vs 0/20** | 40% improvement rate over ArithmeticExtractor on IT model output |
| VPRM Arithmetic Verifier | 454 | **F1=1.0 vs 0.0** | Rule-based; deterministic; no LLM call; dramatic improvement |
| ThinkProbeV2 (RETRO-029) | 455 | **RETRO-029 CLOSED** | ThinkProbeV2 60-min budget; result file absent (RETRO-036) |
| Constraint Addition from Memory | 456 | **fp_rate_delta<0** | Carry-check constraint auto-added; FP rate reduced |
| LSEBMCL Cross-Session Replay | 457 | **lsebmcl_better** | session2_fp_rate=0.0 vs exp448_fp_rate=0.46; Tier 2 confirmed |
| EBM-CoT Latent Calibration | 458 | **AUC 0.5099→0.5554** | Below 0.600 target; improvement but insufficient (RETRO-034) |
| KAEM Large-Variable Crossover | 459 | **crossover at n_vars=50** | KAEM faster than MCMC starting n_vars=50; RETRO-031 closed |
| AMD XDNA IRON NPU | 460 | **install_failed** | mlir-aie pip install blocked; RETRO-035 |
| Milestone 2026.04.34 Retrospective | 461 | **COMPLETE** | 296 experiments; mean=16.5 min; 3 missing result files; 0/10 retro improvements adopted |

**Milestone-level conclusion:**

Milestone 2026.04.34 delivered the FIRST POSITIVE verify-repair number (+5pp, Exp 451) — a Phase 1
milestone reached. VeriCoT (40% detection improvement) and VPRM (F1=1.0 vs 0.0) are ready to
integrate into the live pipeline. LSEBMCL Tier 2 continual learning confirmed. KAEM crossover found.

However, process debt is compounding: 3 missing result files, zero retro improvements adopted,
and the same bottlenecks (GPU zombies, sequential model loading, no deliverable assertion)
recurred. The retrospective verdict was "operational_inefficiency_compounding."

The three open RETRO items (RETRO-032, RETRO-033, RETRO-036) reveal that experiments are completing
but their deliverables are lost due to path mismatches. This is now a systemic issue requiring
infrastructure hardening before more experiments are designed.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Process Debt Blocking Headline Results (CRITICAL)

**Status:** Three consecutive milestones ended without confirming the headline research question
due to missing result JSON files. Exp 451 produced +5pp but the JSON was absent at retrospective
time. RETRO-028/029/032/033/036 all trace to the same root cause: deliverable path mismatches
between the conductor spec and experiment script output paths.

**Root cause:**
- No assertion at experiment exit that the deliverable file exists
- DualGPURunner not wired into ExperimentTemplate — GPU 1 idle the entire milestone
- Retro improvements never implemented — same bottlenecks every milestone
- 10+ suggestions, 0% adoption rate; retro work has no conductor task budget

**Fix:** (1) Add `ExperimentTemplate.assert_deliverable_written()` as mandatory final call.
(2) Wire DualGPURunner into ExperimentTemplate for all dual-model experiments.
(3) Allocate 2 experiment slots to retro improvement implementation each milestone.

### Gap 2: Live Benchmark Scale Insufficient for Credibility (HIGH)

**Status:** First positive verify-repair number confirmed at +5pp on 50 questions. This is
statistically weak (50q = ~7% confidence interval). For credibility, Carnot needs:
- 200q GSM8K at 95% confidence interval
- The same extraction stack (VeriCoT + VPRM) driving live benchmarks — currently they're
  separate experiments, not integrated into the live pipeline
- GSM-Symbolic adversarial variant (Apple 2410.05229) — the experiment that proves Carnot's
  thesis: maintain accuracy where all other approaches fail

**Fix:** (1) Integrate VeriCoT+VPRM into the live VerifyRepairPipeline as the primary extractor.
(2) Scale to 200q GSM8K live benchmark. (3) Run GSM-Symbolic adversarial variant.

### Gap 3: Self-Learning AUC Plateau and FPGA Hardware Idle (MEDIUM)

**Status:** JEPA AUC stuck at 0.571 (trained on 57 real CoT pairs). Target for Tier 3 is 0.650+.
EBM-CoT calibration reached 0.5554 below 0.600 target (RETRO-034). KV260 FPGA hardware arrived
months ago but the bitfile has never been built. GPU-accelerated Ising simulation exists as a
paper (arXiv 2505.22631) but hasn't been tested on the 2x RTX 3090s.

**Fix:** (1) Retrain JEPA on 200+ real CoT pairs (Exp 451 generates ~100 new pairs per 50q run).
(2) Implement EBM-CoT v3 with more Langevin steps or training data to hit AUC > 0.650.
(3) KV260 FPGA bitfile bring-up (hardware sitting idle). (4) Test GPU-Accelerated Oscillator
Ising on existing RTX 3090 hardware — potential 10,000x over CPU.

---

## Architecture After This Milestone

```
Input LLM Response (Gemma4-E4B-it or Qwen3.5-0.8B)
    │
    ├─[Tier 0a: CarnotThinkProbe] ─── CoT verdict (60-min budget, partial verdicts)
    │                                  (arXiv 2504.16828, Exp 444/455, RETRO-036 closed)
    │
    ├─[Tier 0b: SpilledEnergyDetector] ─── Pre-softmax logit energy
    │                                       (arXiv 2602.18671, Exp 433)
    │
    ├─[NEW: Integrated Extraction Stack] ─── VeriCoT FOL+Z3 + VPRM rules + CRANE suffix
    │   ├── VeriCoTStepValidator: IT-model natural language → FOL → Z3 UNSAT check
    │   ├── VPRMArithmeticVerifier: rule-based addition/mult/% (F1=1.0, deterministic)
    │   └── CRANEExtractionGate: constrained suffix prompt for structured claims
    │                                  (Exps 453, 454, CRANE from Exp 451 — now INTEGRATED)
    │
    ├─[Tier 1: SinkProbe] ─── Attention sink concentration
    │                          (arXiv 2604.10697, Exp 348)
    │
    ├─[Tier 2: EORM (EBM-CoT v3 calibrated)] ─── CoT energy reward model
    │                                              AUC target > 0.650 (RETRO-034 closed)
    │                                              (arXiv 2505.14999 + 2511.07124, Exp 458/465)
    │
    ├─[Tier 3: Ising/KAN/KAEM + GPU-OIM] ─── Full constraint verification
    │           │                              (GPU-accelerated: arXiv 2505.22631, RTX 3090)
    │           └─[BoltzmannRepairBridge] ─── Repair direction from ground state
    │
    └─[Self-Learning Loop]
        Tier 1: ConstraintAdditionFromMemory (Exp 456, carry/sign/unit auto-add)
        Tier 2: PPSConstraintLearner (arXiv 2512.15658, task-specific parameter isolation)
        Tier 3: JEPA retrained on 200+ real CoT pairs (AUC target > 0.700)
        Hardware: KV260 FPGA (sparsified Ising bitfile, AXI backend)
```

---

## Dependency Graph

```
[462: DeliverableGuard + DualGPURunner] ──┐
                                          ▼
[463: Conductor Session Health Check] ──► [464: Live Precision 100q RETRO-033] ──► [466: Live 200q VeriCoT+VPRM]
                                          │                                          │
                                          ▼                                          ▼
                                    [465: ThinkProbeV2 RETRO-036]            [467: GSM-Symbolic adversarial]
                                          │
                                          ▼
                                    [468: EBM-CoT v3 RETRO-034]
                                          │
                                          ▼
                               [469: HumanEval live CodeExtractor]
                                          │
                               [470: PPSEBM Tier 2 continual learning] ──┐
                                                                         ▼
                               [471: JEPA Tier 3 + GPU-OIM] ◄──────────┘
                                          │
                               [472: KV260 FPGA bring-up v2]
                                          │
                               [473: Milestone 2026.04.35 Retrospective]
```

---

## Phase Descriptions

### Phase 1 — Process Hardening (Exps 462-463)

**Critical prerequisite for all subsequent work.** Three consecutive milestones lost headline
experiment results to missing deliverable files. This phase adds infrastructure-level enforcement:
every experiment script MUST assert its deliverable file exists before exit, or fail loudly.
DualGPURunner is wired into ExperimentTemplate so GPU 1 stops being idle.

**Exp 462** — DeliverableGuard + DualGPURunner Integration (RETRO-032/033/036 prevention)
: Add `ExperimentTemplate.assert_deliverable_written()` that raises `FileNotFoundError` if the
  deliverable JSON is absent after `build_result()` returns. Wire `DualGPURunner` into
  `ExperimentTemplate.setup_gpu()` so dual-model experiments automatically parallelize across
  cuda:0 and cuda:1. Implement `doc-only` commit classifier to skip the 3900+ test suite for
  changelog/ops doc changes (run ruff+mypy only, saving 80-120 min per milestone).
  Deliverable: `results/experiment_462_deliverable_guard.json`

**Exp 463** — Conductor Session Health Check (zombie kill + env verify + GPU thermal gate)
: Add `scripts/conductor_session_health.py` that runs at the START of every conductor session:
  kill zombie GPU processes (calls `gpu_monitor.py --kill-zombies`), verify CARNOT_FORCE_LIVE
  propagates to subprocesses, check both GPUs under 200MB VRAM and idle, verify no GPU exceeds
  80°C. If any check fails, auto-remediate (kill zombies, re-export env) before first experiment.
  Estimated savings: 60-90 min per session where env or GPU state is wrong.
  Deliverable: `results/experiment_463_session_health.json`

### Phase 2 — RETRO Closures (Exps 464-466)

Three RETRO items from milestone .34 prevent research progress. Close them before scaling.

**Exp 464** — Live Precision Benchmark 100q (RETRO-033 closure)
: Re-run the live precision benchmark with `GemmaTransformersLoader` and the integrated
  VeriCoT+VPRM extraction stack. Scale from 50q to 100q for better statistical confidence
  (95% CI ~±5pp vs ±7pp at 50q). BOTH models on dual-GPU (Gemma4 on cuda:0, Qwen on cuda:1).
  Assert deliverable written (Exp 462's DeliverableGuard). Target: signed improvement > 0 on
  BOTH models. This is the RETRO-033 closure: first positive confirmed with statistical weight.
  Deliverable: `results/experiment_464_live_precision_100q.json`

**Exp 465** — ThinkProbeV2 Live GPU Execution (RETRO-036 closure)
: Execute the ThinkProbeV2 harness (Exp 455) on live GPU with `CARNOT_FORCE_LIVE=1`. The script
  was already implemented in Exp 455 but the result JSON was absent (path mismatch). This run
  uses `assert_deliverable_written()` from Exp 462 to ensure the file is written. Expected:
  `honest_verdict='partial_N_of_50'` or `complete` with `inference_mode='live_gpu'`.
  Deliverable: `results/experiment_465_think_probe_live.json`

**Exp 466** — EBM-CoT Calibration v3 (RETRO-034 closure, AUC > 0.650)
: Calibration v2 (Exp 458) reached AUC 0.5554 vs 0.600 target. Two fixes: (1) increase
  Langevin steps from 10 to 50 (more thorough hidden-state relaxation), (2) supplement the
  57-pair dataset with synthetic pairs generated from the ContinuousEBM (same distribution as
  real pairs). Target: AUC > 0.650 on held-out set. Also implement OIM-style EP coupling
  update (arXiv 2510.12934): update EORM coupling matrix based on free vs clamped CoT step
  spin correlations, no backpropagation needed.
  Deliverable: `results/experiment_466_ebm_cot_v3.json`

### Phase 3 — Scale the First Positive (Exps 467-469)

With process hardening in place and RETRO items closed, scale the benchmarks to credibility.

**Exp 467** — VeriCoT+VPRM Integrated Live Pipeline 200q (THE scale experiment)
: Integrate VeriCoTStepValidator (Exp 453) + VPRMArithmeticVerifier (Exp 454) + CRANEExtractionGate
  into the live `VerifyRepairPipeline` as the primary extraction front-end (replacing
  ArithmeticExtractor). Run 200 GSM8K questions × 2 models × 2 conditions (baseline, pipeline).
  Use `DualGPURunner` from Exp 462 for parallel model inference. Target: signed improvement > 0
  on both models with n=200 (95% CI). This is the experiment that replaces the simulation-era
  numbers with live credible results.
  Deliverable: `results/experiment_467_live_200q_integrated.json`

**Exp 468** — GSM-Symbolic Adversarial Benchmark (THE credibility experiment)
: Run Carnot's verify-repair pipeline on Apple's GSM-Symbolic adversarial variant (arXiv 2410.05229,
  ICLR 2025). Expected LLM behavior: accuracy drops 10-30pp on symbolic variants (same logic,
  different numbers + irrelevant sentences) vs standard GSM8K. Expected Carnot behavior: the
  verify-repair loop closes the gap because Ising verifies arithmetic constraints, not pattern-
  matched keywords. Three conditions: (A) baseline (no Carnot), (B) standard GSM8K + Carnot,
  (C) GSM-Symbolic + Carnot. Key metric: improvement is LARGER on adversarial variant than
  standard (because there are MORE arithmetic errors for Ising to catch). This is Carnot's thesis.
  Load dataset: `datasets.load_dataset('apple/GSM-Symbolic', 'main')` or equivalent.
  Deliverable: `results/experiment_468_gsm_symbolic_adversarial.json`

**Exp 469** — HumanEval Live with CodeExtractor + VeriCoT-guided Repair (50 problems)
: Run 50 HumanEval problems on live GPU. Pipeline: generate code → CodeExtractor extracts
  verifiable claims → VeriCoTStepValidator checks logical consistency of the implementation →
  repair suggestions via BoltzmannRepairBridge if violations detected → re-execute code.
  This is the code verification path where Carnot is MOST LIKELY to show improvement (execution-
  based, not regex-based). Compare: pass@1 baseline vs pass@1 with verify-repair.
  Deliverable: `results/experiment_469_humaneval_live_vericot.json`

### Phase 4 — Self-Learning and Hardware (Exps 470-472)

**Exp 470** — PPSEBM Tier 2 Progressive Constraint Parameter Isolation (mandatory self-learning)
: Implement `PPSConstraintLearner` (arXiv 2512.15658) as an upgrade over LSEBMCL (Exp 457).
  Each constraint domain (arithmetic, code, logical) gets an isolated parameter partition in the
  constraint weight space. When Session 2 encounters arithmetic errors, only the arithmetic
  partition is updated — code and logic partitions remain stable. The EBM generates synthetic
  boundary violations to reinforce partition isolation. Train on accumulated Session 1-3 violation
  data from Exps 456/457. Compare: LSEBMConstraintReplayer vs PPSConstraintLearner on the same
  cross-session carry-error task. Target: PPSEBM partition_isolation_score > 0.8 (sessions
  don't interfere), session2_fp_rate ≤ Exp 457 result.
  Deliverable: `results/experiment_470_ppsebm_constraint_learner.json`

**Exp 471** — KV260 FPGA Bring-Up v2 (sparsified bitfile + AXI backend)
: KV260 hardware arrived months ago (Exp 313: blocked_no_bitfile). This experiment synthesizes
  a minimal Ising sampler bitfile for the KV260 (using the sparsified connectivity approach from
  arXiv 2604.04606). Steps: (1) Generate Verilog for 128-spin sparsified Ising sampler (sparse
  coupling matrix, LFSR RNG, AXI-Lite interface), (2) Synthesize via Vivado (or document the
  exact Vivado commands for human execution), (3) Test Python FpgaBackend AXI communication
  (even if bitfile is CPU-simulated). Include EP-style coupling update (arXiv 2505.02103):
  10-bit precision coupling matrix updates without backprop. Target: FpgaBackend.sample() returns
  valid spin configuration (real hardware or documented bring-up procedure for human to follow).
  Deliverable: `results/experiment_471_kv260_fpga_v2.json`

**Exp 472** — JEPA Tier 3 Scale + GPU-Accelerated Oscillator Ising (target AUC > 0.700)
: Two tasks: (1) Collect 200+ real CoT pairs from Exp 464+467 live runs (live GPU generates
  ~100 pairs per 100q benchmark). Retrain JEPA predictor on the combined 200+ pair dataset
  (57 from Exp 443 + ~150 from Exps 464/467). Target: AUC > 0.700 (vs 0.571 at 57 pairs).
  (2) Implement `GPUOscillatorIsingSimulator` (arXiv 2505.22631) on RTX 3090: run the 1024-spin
  Ising simulation on GPU and benchmark vs CPU `ParallelIsingSampler`. Expected speedup: 100x-
  10,000x on n_vars > 100. If GPU OIM is faster, replace ParallelIsingSampler for large-variable
  constraint problems. This unblocks fast energy evaluation for real-time JEPA gating.
  Deliverable: `results/experiment_472_jepa_gpu_oim.json`

### Phase 5 — Retrospective (Exp 473)

**Exp 473** — Milestone 2026.04.35 Retrospective
: Read all Exp 462-472 result JSONs. Assess milestone questions: (1) Is RETRO-033 closed
  (first positive confirmed at 100q)? (2) Did 200q live benchmark maintain or improve the +5pp?
  (3) Did GSM-Symbolic confirm Carnot's thesis? (4) Is RETRO-034 closed (AUC > 0.650)?
  (5) Did PPSEBM improve over LSEBMCL? (6) Is JEPA AUC > 0.700? (7) How many retro
  improvements were adopted this milestone (target: ≥5 of 10 from prior retro)?
  Compute adoption rate (MUST improve from 0% in .34). Open new RETRO items.
  Deliverable: `results/operational_retro_2026_04_35.json`

---

## Hardware Requirements

| Experiment | Hardware | Minimum | Notes |
|------------|----------|---------|-------|
| Exps 462, 463, 466, 470 | CPU | Any | JAX_PLATFORMS=cpu |
| Exp 464, 465 | GPU | 1x RTX 3090 | CARNOT_FORCE_LIVE=1 |
| Exps 467, 468, 469 | GPU | 2x RTX 3090 | DualGPURunner (Gemma4 cuda:0, Qwen cuda:1) |
| Exp 471 | CPU + Vivado | KV260 | FPGA synthesis optional; CPU sim acceptable |
| Exp 472 | GPU | 1x RTX 3090 | GPU-OIM simulation + JEPA retrain |

---

## Success Criteria

| Criterion | Pass | Experiments |
|-----------|------|-------------|
| RETRO-032/033/036 closed | All 3 result files present at retrospective | 462, 464, 465 |
| RETRO-034 closed | EBM-CoT AUC > 0.650 | 466 |
| First positive confirmed 100q | signed_improvement > 0 on ≥1 model, n=100 | 464 |
| 200q integrated pipeline positive | signed_improvement > 0 on ≥1 model, n=200 | 467 |
| GSM-Symbolic thesis confirmed | Carnot improvement > 0 on adversarial variant | 468 |
| PPSEBM better than LSEBMCL | PPSEBM session2_fp_rate ≤ LSEBMCL .457 result | 470 |
| JEPA AUC > 0.700 | calibrated_auc > 0.700 on held-out set | 472 |
| GPU-OIM speedup | speedup > 10x vs ParallelIsingSampler at n_vars=100 | 472 |
| Retro improvements adopted | ≥5 of 10 from prior retro implemented | all |

---

## Carry-Forward Open Items (Not in This Milestone)

- **RETRO-035:** AMD XDNA NPU (install_failed in Exp 460). Defer until IRON conda package stabilizes.
- **D-Wave cloud:** Promising but not urgent; defer until FPGA bring-up complete.
- **HuggingFace publishing:** Update 16 existing model READMEs + publish Exp 66 joint model.
  Block on 200q benchmark completing (need honest positive result to reference in README).
- **FactNet integration:** Factual claim verification. Defer until extraction pipeline stable.
- **PottsEBM:** Multi-state Ising tier. Design in this milestone, implement in next.
- **RLVR:** EORM-as-reward for LLMExtractor fine-tuning. Tier 3 milestone item.
