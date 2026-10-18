# Research Roadmap v38 — Milestone 2026.04.38

**Status:** Proposed
**Milestone:** 2026.04.38
**Title:** Break the Credibility Ceiling — Gemma4 Quantized, 100q+ Live, GPU 1 Activated
**Planned Experiments:** 500–512 (13 experiments)
**Planned Date:** 2026-04-19 onwards

---

## What Milestone 2026.04.37 Proved

Milestone .37 achieved **100% retro improvement adoption** for the first time, closing RETRO-040
(JEPA AUC recovered 0.281 → 0.967 via curriculum training), RETRO-044 (GPUVRAMGateV2 kills
zombies before checking VRAM), RETRO-045 (batching pre-commit hook), and RETRO-046 (thermal gate).

But the adoption of all enforcement items yielded only 1.8% wall-time reduction. The dominant
constraint shifted from **zombie VRAM** to **conductor process VRAM**: the conductor itself holds
~15.7 GiB GPU 0, leaving only ~8.9 GiB free — insufficient for Gemma4-E4B-it (14.89 GiB).
Three key experiments (488, 489, 490) deferred for the **fifth consecutive milestone** because
GPUVRAMGateV2 can kill zombie processes but cannot kill the conductor.

Root cause analysis from ops/retro .37:

- **RETRO-048 is the new master blocker:** Conductor's own VRAM footprint exceeds the headroom
  needed for Gemma4. Fix: quantize Gemma4 to INT4/GGUF (~8-10 GiB) so both fit within 24 GiB
  GPU 0 VRAM with ~6 GiB headroom. Alternative: route conductor to CPU-only and reserve all
  GPU VRAM for inference.
- **GPU 1 returned to 0% at milestone close** despite DualGPURunner existing and .37 harness_patch
  adoption. The active process at close (PID 107270) is the conductor itself with device_map='auto',
  yielding 0% compute on GPU 1. The harness_patch only covered newly-written experiments; all
  prior dual-model scripts remain unpatched.
- **JEPA recovered to 0.967 AUC** — headline achievement of .37. FR-11 Tier 3 self-learning is
  the strongest research signal. The real-data training loop is closed; the next milestone should
  retrain on live CoT pairs from the credibility benchmarks.
- **RETRO-049/050 opened:** NUP Probe v2 Bayesian SE features produced AUC delta ~1e-16 vs v1
  (feature aggregation too coarse). SuRe surprise replay showed no isolation improvement
  (isolation_improvement = -0.1172) — LLM-surprise signal doesn't correlate with EBM energy.

**Open RETRO items carried into .38:**

| RETRO | Priority | Description |
|-------|----------|-------------|
| RETRO-031 | Low | KAEM no crossover — switch axis from n_vars to distribution_family. |
| RETRO-033 | Critical | Live 100q positive — FIFTH consecutive milestone miss. Blocked by RETRO-048. |
| RETRO-038 | Critical | 200q VeriCoT+VPRM — FOURTH consecutive attempt needed. Blocked by RETRO-048. |
| RETRO-039 | High | GSM-Symbolic adversarial — FOURTH attempt. Blocked by RETRO-048. |
| RETRO-047 | Medium | NUP Probe AUC = 0.600, below 0.700 Tier 0c threshold. Bayesian SE failed. |
| RETRO-048 | Critical | Gemma4 OOM from conductor VRAM — requires quantized model or CPU routing. |
| RETRO-049 | Medium | NUP Probe v2 Bayesian SE delta ~1e-16. Requires per-token entropy + attention variance. |
| RETRO-050 | Medium | SuRe surprise replay: isolation_improvement = -0.1172. Requires energy-magnitude priority. |

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: Gemma4 OOM blocks all credibility work (FR-12 — Verifiable Reasoning)

The conductor process holds ~15.7 GiB of GPU 0 VRAM from its JAX-compiled computation graph.
GPUVRAMGateV2 kills external zombie processes before checking VRAM — but cannot kill the
conductor itself. The remaining free VRAM (~8.9 GiB) is insufficient for Gemma4-E4B-it
at FP16 (14.89 GiB). This single architectural constraint has blocked RETRO-033, RETRO-038,
and RETRO-039 for five consecutive milestones.

**The fix is model quantization.** Gemma4 quantized to INT4/GGUF (Q4_K_M format, consistent
with Carnot's SOTA model guidance for unsloth GGUFs) targets ~8-10 GiB VRAM. With quantized
Gemma4 (~9 GiB) + conductor process (~9 GiB) = ~18 GiB, fitting within 24 GiB GPU 0 with
~6 GiB headroom. The complementary fix — routing the conductor to CPU-only (JAX_PLATFORMS=cpu
for the conductor process, reserving all GPU VRAM for inference) — eliminates the competition
entirely.

**Evidence of urgency:** Five consecutive milestones (.33/.34/.35/.36/.37) ended with
RETRO-033 open. The blocking chain: zombie VRAM (.33-.35) → GPUVRAMGate needed (.33) →
GPUVRAMGateV2 implemented (.37) → conductor VRAM exposed (.37) → quantized model needed (.38).
.38 is the terminal link.

### Gap 2: GPU 1 at 0% utilization — throughput wall (Hardware Efficiency)

GPU 1 (24 GiB RTX 3090) has contributed 0% forward-pass compute across all milestones.
The .37 harness_patch adoption addressed NEW experiments but left all prior dual-model scripts
with device_map='auto', which routes weight storage to GPU 1 but runs all forward-pass compute
on GPU 0. The result: GPU 1 consumes ~1.8 GiB for layer offload while GPU 0 handles all
inference at 100% utilization.

A retroactive sweep of all existing dual-model scripts — replacing device_map='auto' with
explicit {'': 'cuda:0'} for model A and {'': 'cuda:1'} for model B — would unlock true
parallel inference. For dual-model benchmarks (Exps 502-504), this approximately halves
wall time because the two models run simultaneously rather than sequentially.

**Expected impact:** If both Gemma4 and Qwen3.5-0.8B load simultaneously on separate GPUs,
a 100q benchmark with two models completes in the time of one model's 100q run instead of
two model × 100q sequential passes. Estimated 40-50% throughput improvement on benchmark
experiments.

### Gap 3: No publishable credibility claim after 500 experiments (PRD Vision)

Carnot has completed 499 experiments (Exps 1-499). The PRD vision requires verifiable
improvement on real LLM inference. The research-program.md CRITICAL FINDING section documents
that all prior positive results were simulation artifacts. Exps 439/440/441 provided honest
negatives. Exp 451 showed +5pp on 50 questions (too small for statistical significance).

After 500 experiments, there is no statistically significant positive result to claim publicly.
The 200q Wilson 95% CI lower bound > 0 (Exp 503 target) would be the first such claim. The
GSM-Symbolic adversarial result (Exp 504) would be the headline credibility experiment — the
same benchmark Apple used to show all LLMs degrade under adversarial prompting, where Carnot's
thesis is that EBM verification improves ROBUSTNESS (not just accuracy) because the Ising
sampler verifies constraints independently of surface form.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MILESTONE 2026.04.38 PIPELINE                      │
│                                                                        │
│  PHASE 1: RETRO-048 Fix (Gemma4 OOM — the new master blocker)        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 500: Gemma4 INT4 Quantization (GGUF Q4_K_M, ~8-10 GiB)     │ │
│  │ Exp 501: Conductor CPU Routing + VRAM Budget Ledger              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 2: Credibility (RETRO-033/038/039 — 6th/4th/4th attempts,    │
│                         unblocked by quantized Gemma4)                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 502: Live 100q Precision v6 (RETRO-033 — 6th attempt)       │ │
│  │ Exp 503: Live 200q VeriCoT+VPRM v4 (RETRO-038 — 4th attempt)   │ │
│  │ Exp 504: GSM-Symbolic Adversarial v4 (RETRO-039 — 4th attempt)  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 3: GPU 1 Activation (throughput recovery)                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 505: Retroactive DualGPU Harness Sweep (all prior scripts)  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 4: New Research (arxiv 2025-2026 findings)                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 506: Semantic Energy Tier 0d (arXiv 2508.14496)             │ │
│  │ Exp 507: NUP Probe v3 (CLAP features, RETRO-049)               │ │
│  │ Exp 508: KAEM Distribution Family (RETRO-031 new axis)          │ │
│  │ Exp 509: PPSEBM Energy-Magnitude Replay (RETRO-050)             │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 5: Self-Learning (FR-11 mandatory)                             │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 510: JEPA Live Retraining v4 (on Exps 502-503 CoT pairs)    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 6: Hardware Frontier (AMD XDNA NPU)                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 511: AMD XDNA NPU NUP Probe Inference (arXiv 2504.03083)    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 7: Retrospective                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 512: Milestone 2026.04.38 Retrospective                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

```
Exp 500 (Gemma4 GGUF)  ──┬──► Exp 502 (100q v6) ──► Exp 503 (200q v4)
                          │                         ──► Exp 504 (adversarial v4)
Exp 501 (CPU routing) ───┘                         ──► Exp 510 (JEPA retrain v4)

Exps 502/503 (live CoT pairs) ──► Exp 507 (NUP Probe v3)
                               ──► Exp 510 (JEPA retrain v4)

Exp 507 (NUP Probe v3) ──► Exp 511 (AMD NPU)

Exp 505 (GPU sweep) ──► (throughput improvement for all subsequent GPU experiments)

Exp 506 (Semantic Energy) ── independent (CPU-only)
Exp 508 (KAEM dist) ──────── independent (CPU-only)
Exp 509 (PPSEBM replay) ──── independent (CPU-only)

Exp 512 (Retro) ──► requires all prior experiments complete
```

---

## Phase Descriptions

### Phase 1: RETRO-048 Fix (Experiments 500–501)

The root cause of five consecutive milestone misses is the conductor process holding ~15.7 GiB
GPU 0 VRAM, leaving insufficient headroom for Gemma4. Two complementary fixes are needed:

**Exp 500 — Gemma4 INT4 Quantization:** Quantize `google/gemma-4-E4B-it` to GGUF Q4_K_M
format targeting 8-10 GiB VRAM. Use unsloth's quantization pipeline (consistent with SOTA
model guidance in research-program.md) or llama-cpp-python with Q4_K_M. Verify the quantized
model: (a) loads within VRAM budget alongside conductor, (b) achieves >70% GSM8K accuracy
(verifying quality is not catastrophically degraded by quantization), (c) generates valid
CoT steps parseable by VeriCoT+VPRM extractors. Write `Gemma4QuantizedLoader` as a new loader
class (sibling to GemmaTransformersLoader) that loads the GGUF checkpoint via llama-cpp-python
with n_gpu_layers=-1 for full GPU offload.

**Exp 501 — Conductor CPU Routing + VRAM Budget Ledger:** Establish the pattern for running
the conductor process on CPU while reserving GPU VRAM for inference. Add a `VRAMBudgetLedger`
class that reads a YAML manifest of per-experiment VRAM requirements and pre-checks feasibility
before each experiment launch. The ledger should produce a VRAM forecast for the full milestone
at planning time, flagging any experiment sequence where cumulative VRAM commitment exceeds
free VRAM. This converts the reactive "deferred_to_gpu" pattern into proactive scheduling.

### Phase 2: Credibility Benchmarks (Experiments 502–504)

These three experiments are the credibility goals of the entire research program. They have been
deferred for 5 consecutive milestones by the VRAM deadlock. Phase 1 removes the blocker.

**Exp 502 — Live 100q Precision v6 (RETRO-033):** 100 GSM8K questions, 2 models
(Gemma4-INT4 on cuda:0, Qwen3.5-0.8B on cuda:1), VeriCoT+VPRM+CRANE extraction, baseline vs
pipeline comparison. Wilson 95% CI reported. An `is_positive=True` result closes RETRO-033
after 5 milestones. Write 100 CoT pairs to `results/exp502_cot_pairs.json` for Exp 510.

**Exp 503 — Live 200q VeriCoT+VPRM v4 (RETRO-038):** 200 GSM8K questions with full extraction
stack. `is_statistically_positive=True` (Wilson 95% CI lower bound > 0) is the first
publishable claim. Write 200 CoT pairs to `results/exp503_cot_pairs.json` for Exp 510.

**Exp 504 — GSM-Symbolic Adversarial v4 (RETRO-039):** Apple arXiv 2410.05229 benchmark with
symbolic adversarial variants. The thesis: Carnot's improvement should be LARGER on adversarial
variants because the Ising sampler verifies arithmetic constraints independent of surface form.
Expected result: standard accuracy drops less with Carnot than without; adversarial accuracy
drops less with Carnot. This is the headline credibility experiment.

### Phase 3: GPU 1 Activation (Experiment 505)

**Exp 505 — Retroactive DualGPU Harness Sweep:** Audit all experiment scripts for device_map
usage. For each script that loads two models with device_map='auto', replace with explicit
device assignment: model A on cuda:0, model B on cuda:1. Use the `DualGPUHarness` class (Exp
480) for uniform assignment. Run the GPU 1 utilization benchmark (from Exp 495) before and
after the sweep to confirm improvement. Target: GPU 1 utilization > 30% during dual-model
benchmark experiments (from 0% baseline). The sweep itself is a migration experiment —
write a migration script that patches all affected files in a single commit.

### Phase 4: New Research (Experiments 506–509)

**Exp 506 — Semantic Energy Tier 0d (arXiv 2508.14496):** Implement `BoltzmannSemanticEnergy`
as a new Tier 0d in the verification cascade (positioned between SpilledEnergy Tier 0b and
SinkProbe Tier 1). The approach: (1) cluster semantically similar tokens in the penultimate
layer using k-means (k=10), (2) compute Boltzmann-weighted energy per cluster from logits,
(3) return a scalar hallucination score. Benchmark AUC vs SpilledEnergy baseline on 200
synthetic + 200 real (from Exp 502) CoT responses. CPU-only experiment.

**Exp 507 — NUP Probe v3 (RETRO-049, arXiv 2509.09700 CLAP):** Implement `CLAPFeatureExtractor`
that constructs a (n_layers, n_tokens, hidden_dim) activation tensor from multiple residual
stream layers (last 4 layers), applies multi-head attention over the cross-layer sequence, and
returns a per-token hallucination score. Features: (a) per-token softmax entropy over vocabulary,
(b) top-k probability concentration (ratio of top-1 to top-5 probability mass), (c) cross-layer
attention head variance. Retrain NUP Probe v3 on real CoT pairs from Exps 502-503 (target: 200
pairs). Target AUC > 0.700 for Tier 0c promotion. CPU-only training; GPU needed for feature
extraction from live model.

**Exp 508 — KAEM Distribution Family (RETRO-031):** Switch the KAEM experimental axis from
n_vars to distribution_family. Test KAEM vs ParallelIsingSampler on: (a) Gaussian mixture
(2 modes, σ=0.5, separation=3.0 — multimodal), (b) Student-t with ν=2 (heavy-tail), (c)
piecewise uniform (5 pieces, non-smooth). For each distribution: compute mean_l2 between
KAEM samples and ground truth samples from the target distribution, compare vs MCMC baseline.
If KAEM shows advantage on any distribution family, this is an actionable result (more
informative than continuing to extend n_vars). CPU-only experiment.

**Exp 509 — PPSEBM Energy-Magnitude Replay (RETRO-050):** Replace the SuRe LLM-surprise
priority signal with EBM energy magnitude priority in the PPSEBM replay buffer. The new
replay strategy: rank all constraint violations by |energy(x) - expected_energy|, where
expected_energy is the running mean energy of previously seen violations. Replay the top-k
highest-energy violations (the domain boundary cases the model most needs to learn from).
Compare isolation_score before/after vs SuRe baseline from Exp 497 (isolation_improvement
= -0.1172). Target: isolation_score improvement > 0. CPU-only experiment.

### Phase 5: Self-Learning (Experiment 510)

**Exp 510 — JEPA Live Retraining v4 (FR-11 Tier 3):** Retrain the JEPA predictor on live
CoT pairs accumulated from Exps 502-503. Incorporate quasimetric regularization from
arXiv 2602.12245: add L_quasimetric = λ * max(0, d(conclusion, premise) - d(premise, conclusion))
to the training objective, penalizing symmetry in embedding distance (premise → conclusion
should be harder than conclusion → premise in a reasoning chain). Use the curriculum training
approach from Exp 492 (high-confidence pairs first, then all pairs). Target: AUC >= 0.800
on held-out live pairs. Write retrained checkpoint to `results/jepa_predictor_510_live.safetensors`.

This is the mandatory self-learning experiment for every milestone per research-program.md.
FR-11 relay status: JEPA AUC recovered to 0.967 in .37 (Exp 492 synthetic curriculum).
.38 closes the loop on live data.

### Phase 6: Hardware Frontier (Experiment 511)

**Exp 511 — AMD XDNA NPU NUP Probe Inference (arXiv 2504.03083):** Deploy the NUP Probe v3
entropy computation on the AMD Ryzen AI NPU using the IRON tool-flow and ONNX Runtime VitisAI
backend. The probe's per-token entropy computation (softmax over vocabulary at each generation
step) is embarrassingly parallel and NPU-native. If the NPU entropy computation latency is
< 5ms/token, it can pipeline with LLM generation on GPU — enabling zero-overhead Tier 0c
filtering at inference speed.

The experiment is self-contained: (1) export the per-token entropy layer to ONNX, (2) load
via VitisAI, (3) benchmark latency on synthetic activation tensors of shape (1, seq_len,
hidden_dim). If the NPU is unavailable (VitisAI not installed), emit `honest_verdict='npu_not_available'`
with setup instructions — do NOT fail silently.

### Phase 7: Retrospective (Experiment 512)

**Exp 512 — Milestone 2026.04.38 Retrospective:** Analyze execution efficiency, retro closure
rates, GPU utilization, and wall-time improvement across Exps 500-511. Identify the top 3
bottlenecks and 3 most actionable improvements for .39. Report headline results from the
credibility benchmarks (Exps 502-504) — these must be in the retro as the milestone's primary
research contribution.

---

## Hardware Requirements

| Experiment | Hardware | Minimum VRAM | Duration Estimate |
|-----------|----------|-------------|-------------------|
| Exp 500 | CUDA GPU | 24 GiB (GPU 0) | 30 min |
| Exp 501 | CPU only | — | 15 min |
| Exp 502 | 2x RTX 3090 | 18 GiB (quantized Gemma4 + Qwen) | 120 min |
| Exp 503 | 2x RTX 3090 | 18 GiB | 150 min |
| Exp 504 | 2x RTX 3090 | 18 GiB | 120 min |
| Exp 505 | CPU only | — | 20 min |
| Exp 506 | CPU only (GPU optional) | — | 20 min |
| Exp 507 | GPU recommended | 8 GiB | 30 min |
| Exp 508 | CPU only | — | 20 min |
| Exp 509 | CPU only | — | 20 min |
| Exp 510 | GPU recommended | 8 GiB | 30 min |
| Exp 511 | AMD NPU (CPU fallback) | — | 25 min |
| Exp 512 | CPU only | — | 20 min |

**VRAM constraint:** After Exp 500 (Gemma4 INT4 quantization), the conductor + Gemma4-INT4
+ Qwen3.5-0.8B should fit within 24 GiB GPU 0:
- Conductor process: ~9 GiB
- Gemma4-INT4 (Q4_K_M): ~8-10 GiB
- Qwen3.5-0.8B: ~1.5 GiB
- Total: ~18.5-20.5 GiB (fits with ~3-5.5 GiB headroom)

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| retro_048_resolved | Exp 500+501 | Gemma4-INT4 loads within VRAM budget |
| retro_033_closed | Exp 502 | is_positive=True on 100q |
| retro_038_closed | Exp 503 | is_statistically_positive=True (Wilson 95% CI lower > 0) |
| retro_039_confirmed | Exp 504 | Carnot degrades less than baseline on adversarial variants |
| gpu1_utilized | Exp 505 | GPU 1 utilization > 30% during dual-model benchmarks |
| semantic_energy_viable | Exp 506 | AUC > SpilledEnergy baseline |
| nup_probe_promoted | Exp 507 | AUC > 0.700 (Tier 0c threshold) |
| kaem_advantage_found | Exp 508 | Any distribution_family where KAEM beats MCMC |
| ppsebm_isolation_positive | Exp 509 | isolation_improvement > 0 |
| fr11_live_relay | Exp 510 | JEPA AUC >= 0.800 on live data |
| npu_inference_measured | Exp 511 | Latency measured (pass/fail for viability) |

---

## Arxiv Findings Incorporated

| Finding | Paper | Experiment |
|---------|-------|-----------|
| Semantic Energy Tier 0d (Boltzmann logit energy) | arXiv 2508.14496 | Exp 506 |
| CLAP cross-layer attention probing | arXiv 2509.09700 | Exp 507 (NUP Probe v3) |
| AMD XDNA NPU ML inference | arXiv 2504.03083 | Exp 511 |
| Intrinsic-Energy JEPA quasimetric regularization | arXiv 2602.12245 | Exp 510 |
| KAEM distribution family axis | (from retro .37 RETRO-031) | Exp 508 |

Filed for future milestones (not .38):
- Semantic Token Clustering (arXiv 2603.20161) — score_candidates MCP optimization (.39+)
- EB-JEPA Library (arXiv 2602.03604) — JEPA Phase 3 refactor
- CIKAN KAN + hard constraints (arXiv 2412.03710) — KAN tier enhancement (.39+)
- Stochastic Ising Machine sampling advantage (arXiv 2504.18359) — KV260 bring-up acceptance criterion
