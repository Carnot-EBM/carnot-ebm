# Research Roadmap — Milestone 2026.04.66

**Title:** Permanent LIVE-ENV Fix + DualGPU Production + Inertia Ising + Streaming CoT + iCE40 N=16

**CalVer:** 2026.04.66 (sequence increment from 2026.04.65)
**Planned Experiments:** Exps 855-867 (13 experiments)
**Date Designed:** 2026-04-25
**Prerequisite:** Milestone 2026.04.65 retro complete (Exp 854)

---

## What Milestone 2026.04.65 Proved

Milestone .65 (Exps 843-854) targeted 12 success criteria. Results from Exp 854 retro:

**Wins:**
- Exp 843: governance_ready — RETRO audit + retirement plan written; manifest_enforcement_patch.txt provided
- Exp 846: arbiter_calibrated — Gibbs warm-start fixed accuracy_standard from 0.0 → 1.0; RETRO-ARBITER-FLAT-ENERGY CLOSED
- Exp 849: gguf_cache_implemented — GGUFCacheResolver implemented; RETRO-GGUF-CACHE-IMPORT CLOSED
- Exp 852: SemanticEnergyProbe Tier 0f deployed (AUC_synthetic result in artifact)

**Failures and new diagnostics:**
- Exp 853: RETRO-LIVE-ENV-NOT-PROPAGATED OPENED — CARNOT_FORCE_LIVE not propagated (RETRO-015 recurrence); live benchmark v4 fell back to simulation for ninth consecutive code repair block
- Exp 851: RETRO-ICE40-N16-UNEXPECTED-EXPANSION — N=16 expanded from 2 LCs at synthesis to 12258 LCs at P&R; flip-flop register proliferation root cause (synchronous spin state regs inferred by nextpnr from sequential Verilog)
- Exp 850: RETRO-SOTA-MODEL-DOWNLOAD — model file absent despite GGUFCacheResolver; download path not implemented in Exp 849 (resolve() only checks existence, doesn't download)
- Exp 854 retro: Wall time REGRESSION SIXTH CONSECUTIVE (+78 min, +2.0% vs .64). manifest_fix_patch.txt unapplied SEVEN consecutive milestones. DualGPURunner validated (1.96x throughput, Exp 685) but NEVER deployed in production path. Experiment count 772 — new historic high, 72 over 700 cap. GPU close clean (47C/47C, 0C differential). 10 open RETROs.

**10 RETROs open going into .66:**
- RETRO-MANIFEST-FULL-SCOPE — manifest_fix_patch.txt provided in Exp 843 but not applied (human action needed)
- RETRO-JEPA-OOD — result of Exp 844 (if min_domain_auc < 0.50, still open)
- RETRO-CONSTRAINT-ZERO-DELTA — result of Exps 847/848 (if retrieval still broken)
- RETRO-XILINX-TOOLS-UNAVAILABLE — Vivado not installed; KV260 native synthesis blocked
- RETRO-ISING-INJECTION-NO-DISCRIMINATION — energy delta identical for error/clean code
- RETRO-SVAMP-ZERO-AUC — result of Exp 844 (if auc_svamp < 0.40)
- RETRO-ICE40-PNR-LUT-OVERFLOW — N=16 LUT overflow (resolved by .66 Exp 859 if combinational fix works)
- RETRO-SOTA-MODEL-DOWNLOAD — model absent despite resolver; download needed
- RETRO-ICE40-N16-UNEXPECTED-EXPANSION — 12258 LCs from registered spin state registers
- RETRO-LIVE-ENV-NOT-PROPAGATED — permanent fix needed, not just workaround

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: LIVE-ENV Never Permanently Fixed — Code Repair Blocked 9 Consecutive Milestones

The RETRO-015 pattern (CARNOT_FORCE_LIVE not propagating to subprocesses) has now recurred
SIX times across different implementation attempts. The apply_env_autofix() approach is not
sufficient: it sets the env var in the current process but does not ensure subprocess inheritance
in all execution paths. Every GPU experiment that depends on this is silently falling back to
simulation.

**Root cause (to confirm in Exp 855):** Python subprocess.Popen() with `env=None` inherits the
calling process's environment. But the conductor launches experiments via `claude -p` which
creates a NEW process tree that does not inherit the outer shell's environment. The fix:
(1) Make apply_env_autofix() write the env var to a session file that is sourced at experiment
startup, AND (2) add a hard assert that blocks the experiment if CARNOT_FORCE_LIVE is not set
AND GPU is detected.

**Fix for .66 (Exp 855):**
1. Implement `EnvPropagationGuard` that writes CARNOT_FORCE_LIVE=1 to `~/.carnot_session_env`
   at the start of any experiment that needs live GPU.
2. All subsequent experiments source `~/.carnot_session_env` at startup (one-line addition to
   ExperimentTemplate.__init__).
3. Exp 857 (code repair) and Exp 858 (benchmark) are gated on Exp 855's `live_env_fixed=True`.
4. Add this to ExperimentTemplate.apply_env_autofix() permanently.

### Gap 2: DualGPURunner Validated But Never Deployed — 60 min/milestone Lost

Exp 685 validated DualGPURunner at 1.96x throughput. Every milestone since .57 has noted
"DualGPURunner NEVER deployed in production path" in the retro. This is the single highest-
impact unimplemented improvement: deploying it would eliminate the need to serialize GPU inference
across experiments, potentially recovering 60+ min/milestone.

**Fix for .66 (Exp 856):**
1. Wire DualGPURunner into VerifyRepairPipeline.verify() for dual-model experiments.
2. Wire into ThreeTierPipeline for benchmarks that run both models simultaneously.
3. Validate with 25 synthetic questions: throughput_ratio > 1.5x vs serial execution.
4. Gate Exps 857/858 on dual_gpu_deployed=True.

### Gap 3: iCE40 N=16 Register Expansion — 12258 LCs from Sequential Verilog

N=16 synthesis was clean (2 LUTs) but P&R expanded to 12258 LCs because the spin state
registers were inferred as flip-flops (sequential logic). This is a fundamental mismatch
between the synchronous Verilog design and the combinational FPGA synthesis target.

**Root cause:** The spin state `s[15:0]` is declared as `reg` updated in an `always @(posedge clk)`
block. Nextpnr-ice40 correctly infers 16 flip-flops for 16 spin bits, plus the combinational logic
for the Gibbs conditional energy computation. But the combinational logic for a 16x16 coupling
matrix expands dramatically: each spin update requires `sum_j J_ij * s_j`, which is 16 multiply-
accumulate operations. In fixed-point arithmetic, each MAC is ~50 LUTs. Total: 16 spins × 16 MACs
× 50 LUTs ≈ 12,800 LUTs. That IS the expansion.

**Fix for .66 (Exp 859):**
Reduce N to 8 (N=8 design) with combinational energy readout only:
- Remove the sequential Gibbs sweep logic entirely (too expensive for iCE40)
- Implement ONLY the energy computation: E = -sum_i sum_j J_ij * s_i * s_j + sum_i h_i * s_i
- Input: spin configuration as external input (not internally updated)
- Output: energy value for that configuration
- Expected LUT count: N^2 multipliers + N adder tree = 64 * 2 + 8 * 2 = 144 LUTs (within budget)
- This is an "energy oracle" not a full sampler — but it validates the FPGA path and enables
  the PYNQ overlay dispatch for Carnot's FpgaBackend

---

## Architecture Diagram

```
Query (live LLM response)
  |
  v
[Tier 0a] CarnotThinkProbe (generative CoT verdict, ThinkPRM arXiv 2504.16828)
  |  fast-path on "incorrect" verdict
  v
[Tier 0b] SpilledEnergyDetector (logit-discrepancy, arXiv 2602.18671)
  |
  v
[Tier 0c] NUP Probe v4 (contrastive energy, AUC=1.0, Exp 523)
  |
  v
[Tier 0d] HallucinationBasinDetector (latent-space basin depth, arXiv 2604.04743)
  |
  v
[Tier 0e] HalluField (token-path thermodynamic instability, arXiv 2509.10753) [advisory]
  |
  v
[Tier 0f] SemanticEnergyProbe (pairwise Boltzmann semantic energy, Exp 852) [advisory]
  |
  v
[Tier 0g] *** NEW: StreamingCoTHalluDetector (prefix-level cumulative PHaS, arXiv 2601.02170) [advisory]
  |
  v
[Tier 0h] JailbreakDetectionKAN (AUC=1.0, Exp 775)
  |
  v
[Tier 0i] *** NEW: HalluSAEGeometricProbe (SAE feature geometry energy, arXiv 2604.16430) [advisory]
  |
  v
[Tier 1] SinkProbe (attention sink concentration, arXiv 2604.10697)
  |
  v
[Tier 2] EORM (CoT energy reward model, 55M params)
  |
  v
[Tier 2.5] SymCodeVerifier (executable Python arithmetic, Exp 619)
  |
  v
[Tier 2.6] HermesVerifierAdapter (step-boundary feedback, arXiv 2511.18760)
  |
  v
[Tier 2.7] CausalReasoningVerifier (causal entailment, arXiv 2601.21210)
  |
  v
[Tier 3] IsingEBM via InertiaIsingSampler (arXiv 2604.17109, Exp 860) ***NEW***
       + LagrangeAdaptiveConstraints (arXiv 2501.04971, Exp 862) ***NEW***
  |
  v
[Tier 3.5] JEPA v24b Predictive Verifier (Exp 845, if deployed)

[DualGPURunner] *** NEW DEPLOYMENT (Exp 856): parallelizes GPU inference across tiers ***

[FPGA Energy Oracle] *** iCE40 N=8 combinational (Exp 859, gates KV260 deploy) ***
```

---

## Phase Descriptions

### Phase 0: Governance Fixes (Exps 855-856, CPU)

Break the six-milestone LIVE-ENV and DualGPU loops with permanent implementations.

**Exp 855: Pre-flight v15 — Permanent LIVE-ENV Fix**
- Implement `EnvPropagationGuard` (writes to `~/.carnot_session_env`, sourced at ExperimentTemplate.__init__)
- Audit all 10 open RETROs, update MILESTONE_PREREQS.md
- Gate: live_env_fixed=True before GPU experiments can run

**Exp 856: DualGPURunner Production Deployment**
- Wire DualGPURunner into VerifyRepairPipeline and ThreeTierPipeline
- Validate: throughput_ratio >= 1.5x on 25 synthetic questions
- Gate: dual_gpu_deployed=True for benchmark experiments

### Phase 1: GPU Critical Path (Exps 857-858, GPU)

First positive live code repair result (GGUF cache + SOTA model now possible).

**Exp 857: SOTA GGUF Model Download + Code Repair v6**
- Extend GGUFCacheResolver with download capability (huggingface-hub pull)
- Run 25 HumanEval problems with Qwen3.6-35B-A3B-GGUF
- Gated on Exp 855 live_env_fixed=True + Exp 856 dual_gpu_deployed=True

**Exp 858: Live Full Precision Benchmark v5 — DualGPU + All Cascade Tiers**
- DualGPURunner active (Exp 856)
- All cascade tiers: Tier 0a through 3.5 (if deployed)
- 50 GSM8K + 25 HumanEval
- Gated on Exp 856 dual_gpu_deployed=True

### Phase 2: FPGA Hardware (Exps 859-860, CPU + iCE40 tools)

Fix the N=16 register expansion and benchmark the inertia sampler.

**Exp 859: iCE40 N=8 Combinational Energy Oracle**
- Pure combinational Verilog: energy computation only (no sequential spin state)
- Expected LUT count: ~144 (vs 12258 for sequential N=16)
- Generates .bin bitstream for PYNQ overlay dispatch
- Closes RETRO-ICE40-N16-UNEXPECTED-EXPANSION and RETRO-ICE40-PNR-LUT-OVERFLOW

**Exp 860: Inertia Ising Sampler Benchmark**
- arXiv 2604.17109: EMA per-spin inertia term (alpha=0.5)
- Benchmark: discrimination_delta between correct/incorrect constraint configs
- Mpemba initialization (arXiv 2603.24183): spectral-optimal starting magnetization
- Targets: inertia reduces mixing sweeps by 5x+; improves energy discrimination

### Phase 3: New Detection Probes + Self-Learning (Exps 861-864, CPU)

Wire new probe tiers and advance FR-11 self-learning.

**Exp 861: StreamingCoTHalluDetector (Tier 0i)**
- arXiv 2601.02170: prefix-level cumulative PHaS signal
- Per-step EORM scores → running state estimate phas_t = alpha * score_t + (1-alpha) * phas_{t-1}
- Advisory: is_streaming_unstable flag in VerificationCertificate

**Exp 862: LagrangeAdaptiveIsingConstraints (FR-11 Tier 1, Self-Learning)**
- arXiv 2501.04971: iterative Lagrange relaxation of constraint weights
- Violation-driven coupling weight increase (adaptive self-learning)
- 5-session relay: compare delta_s1_to_s5 vs non-adaptive baseline
- Mandatory FR-11 experiment for this milestone

**Exp 863: HalluSAEGeometricProbe (Tier 0j)**
- arXiv 2604.16430: SAE feature geometry energy over CoT trajectory
- Lightweight bigram SAE dictionary (no GPU training needed)
- AUC on 50 synthetic CoT pairs; advisory tier deployment

**Exp 864: FR-11 Tier 2 Integration v5 — Wire All New Probes**
- Integrate Exp 861 (streaming PHaS) + Exp 862 (Lagrange adaptive) + Exp 863 (SAE geometry)
- Full 5-session self-learning relay on live or synthetic data
- FR-11 mandatory relay experiment; report tier1_relay_works status

### Phase 4: Infrastructure (Exps 865-866, CPU)

Compress constraint memory and analyze KAN hardware suitability.

**Exp 865: Constraint Memory Bank Compression**
- arXiv 2601.00756: online kmeans clustering of EmbeddingConstraintStore
- K=32 centroid embeddings; compress 10-session accumulated constraints
- Compare retrieval AUROC before/after compression

**Exp 866: KAN Hardware Complexity Analysis**
- arXiv 2604.03345: per-knot LUT estimates for KAEMEnergy
- 8 knots, piecewise-linear; simulate iCE40 LUT budget
- Determine KAN vs Ising synthesis priority for KV260

### Phase 5: Retrospective (Exp 867)

**Exp 867: Milestone 2026.04.66 Operational Retrospective**
- Standard retro format (schema=carnot.operational_retro.v41)
- 13 success criteria
- RETRO audit: close those resolved, open any new

---

## Dependency Graph

```
Exp 855 (LIVE-ENV fix) ──┐
                          ├── Exp 857 (SOTA code repair, GPU)
Exp 856 (DualGPU) ───────┤
                          └── Exp 858 (live benchmark, GPU)

Exp 859 (iCE40 N=8) ── independent (CPU, iCE40 tools)
Exp 860 (Inertia Ising) ── independent (CPU)

Exp 861 (Streaming CoT) ──┐
Exp 862 (Lagrange Ising) ──┼── Exp 864 (FR-11 Tier 2 relay)
Exp 863 (HalluSAE) ───────┘

Exp 865 (Memory compression) ── independent (CPU)
Exp 866 (KAN hardware) ── independent (CPU)

All Exps 855-866 ── Exp 867 (retrospective)
```

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|------------|----------|-------|
| Exp 855-856 | CPU only | Governance + wiring |
| Exp 857 | 2x RTX 3090, 48GB VRAM | Qwen3.6-35B needs ~20GB; GPU 1 for model |
| Exp 858 | 2x RTX 3090 via DualGPURunner | Both GPUs active |
| Exp 859 | CPU + iCE40 tools | OSS-CAD-Suite at ~/tools/oss-cad-suite |
| Exp 860 | CPU | Pure Python simulation |
| Exps 861-864 | CPU | No GPU needed |
| Exp 865-866 | CPU | Analysis experiments |
| Exp 867 | CPU | Retrospective |

---

## Success Criteria for Milestone 2026.04.66

| # | Criterion | Target Experiment |
|---|-----------|------------------|
| 1 | live_env_permanently_fixed | Exp 855: live_env_fixed=True AND EnvPropagationGuard deployed |
| 2 | dual_gpu_deployed | Exp 856: throughput_ratio >= 1.5x in production path |
| 3 | code_repair_positive | Exp 857: signed_improvement > 0 AND inference_mode=live_gpu |
| 4 | live_benchmark_improvement | Exp 858: pipeline_improvement AND inference_mode=live_gpu |
| 5 | ice40_n8_bitstream | Exp 859: bitstream_generated=True AND lut_count < 500 |
| 6 | inertia_discrimination | Exp 860: discrimination_delta > 0 AND mixing_sweeps_reduction >= 5x |
| 7 | streaming_cot_viable | Exp 861: AUC_streaming > 0.65 on synthetic CoT |
| 8 | lagrange_adaptive_works | Exp 862: delta_s1_to_s5 > 0 (FR-11 Tier 1 mandatory) |
| 9 | hallusae_viable | Exp 863: AUC_geometric > 0.65 on synthetic CoT |
| 10 | fr11_tier2_relay_confirmed | Exp 864: tier2_relay_confirmed=True |
| 11 | memory_compression_viable | Exp 865: retrieval_auroc post-compression > 0.75 |
| 12 | kan_fpga_roadmap_clear | Exp 866: KAN synthesis LUT estimate < 2000 AND priority_determined |
| 13 | wall_time_improvement | Exp 867: wall_time_delta_vs_65 < 0 (first positive delta in 6 milestones) |

---

## Key References

- arXiv 2604.17109 — Fully Parallel Inertia Ising Machine (FPGA, Exp 860)
- arXiv 2501.04971 — Self-Adaptive Ising for Constrained Optimization (Lagrange, Exp 862)
- arXiv 2601.02170 — Streaming Hallucination Detection in Long CoT (Exp 861)
- arXiv 2604.16430 — HalluSAE Geometric Energy (SAE geometry, Exp 863)
- arXiv 2601.00756 — Memory Bank Compression for Continual Adaptation (Exp 865)
- arXiv 2604.03345 — Hardware-Oriented KAN Inference Complexity (Exp 866)
- arXiv 2603.24183 — Mpemba Initialization for Fast Thermodynamic Computing (Exp 860)
