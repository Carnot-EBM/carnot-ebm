# Carnot Roadmap

*Auto-updated by the research conductor as experiments complete.*

## Current Milestone

**2026.04.33: "First Live Results, ThinkPRM Bridge, Boltzmann-GPT Repair"**

| # | Experiment | Status |
|---|-----------|--------|
| 437 | LongRunBenchmarkExecutor (RETRO-026 fix) | Complete |
| 438 | DualGPURunner GPU1 zombie fix (RETRO-025) | Complete — fix_applied_unverified |
| 439 | Live Precision Micro-Benchmark (50q × 3 × 2) | In Progress — first credible live number |
| 440-449 | Live HumanEval, Adversarial, FR-11 chain, ThinkPRM, Phase 3 seed v2, retro | Queued |

## Completed Milestones

| Milestone | Theme | Experiments | Key Breakthrough |
|-----------|-------|------------|-----------------|
| 2026.04.15 | Semantic Grounding | 211-223 | +4.9pp typed IR constraints, 86% FP reduction via self-learning |
| 2026.04.16 | Scale What Works | 224-231 | +3.0pp on full 164-problem HumanEval (statistically significant) |
| 2026.04.17 | Calibrated Verification | 232-243 | Spec-grounded code repair equalizes cross-model performance |
| 2026.04.18 | Formal Claims + Predictive | 244-257 | Formal claim verifier, predictive verification gate |
| 2026.04.19 | GPU + Calibration + Publish | 258-270 | GPU acceleration stack, HuggingFace publish |
| 2026.04.20 | Revalidation Sweep | 271-280 | 6 approaches confirmed live (consistency, rollback, factual, KAN, Z3+LLM, memory) |
| 2026.04.21 | Apple Adversarial + FPGA | 281-294 | Apple adversarial benchmark, SpilledEnergy + SemanticEnergy extractors, FPGA Verilog |
| 2026.04.22 | Adversarial Completion | 294-306 | Confidence-weighted repair (86.7% FP avoidance), experiment template |
| 2026.04.23 | JEPA + Z3 + D-Wave | 307-324 | D-Wave quantum sampler, NL-to-Z3 extractor, reward hacking detection, conductor constitution |
| 2026.04.24 | GPU Benchmarks + Hardening | 325-337 | SinkProbe pre-filter (60% skip, 0% FN), CoT circuit verifier, EORM energy reward model |
| 2026.04.25 | E2E Precision + EORM | 338-350 | Three-tier pipeline (SinkProbe, EORM, Ising), constraint template library |
| 2026.04.26 | Apple Adversarial + Z3 | 351-364 | LLM-guided Z3 formalization, GPU acceleration end-to-end |
| 2026.04.27 | LLMExtractor + Self-Learning | 365-376 | CIKAN energy tier, live adversarial GSM8K |
| 2026.04.28 | Break Simulated Barrier | 377-389 | JitRL self-learning, live precision pipeline |
| 2026.04.29 | Live Results At Last | 390-403 | GPU confirmed, CIKAN, FR-11 closed |
| 2026.04.30 | Purge + First Credible Live | 404-417 | DeliverableContentValidator, env auto-fix, GPU preflight v2 |
| 2026.04.31 | EnvironmentAutoFix + VPRM | 418-424 | Env propagation workaround, VPRM architecture |
| 2026.04.32 | Live Numbers Confirmed (infrastructure) | 425-436 | Conductor timeout watchdog, DualGPU detector, FOVER annotation, Kona Phase 3 seed, provenance audit |

## Breakthrough Results

Results are labeled with provenance: **LIVE** (real model inference on GPU with `CARNOT_FORCE_LIVE=1`), **SIMULATED** (synthetic benchmark or canned CI cases), **DERIVED** (post-hoc analysis of prior live artifacts), or **PLACEHOLDER** (fast-path deliverable without actual inference). Audit performed 2026-04-16 after RETRO-022 root cause was identified in the conductor's env propagation — several results that were previously unlabeled turned out to be simulated.

| Result | Value | Experiment | Provenance | Significance |
|--------|-------|------------|------------|-------------|
| HumanEval code verification | +3.0pp [+0.6, +6.1] CI | Exp 226 | LIVE | Statistically significant on 164 official problems (gemma-4-E4B-it, 1574s runtime) |
| PBT bug detection rate | 99.3% (144/145) | Exp 220 | LIVE | Property-based testing catches nearly all wrong code (Qwen+Gemma, 816s) |
| Typed IR constraints | +4.9pp (Gemma4) | Exp 221 | LIVE | Prompt-side constraint extraction works (81 cases, 459s) |
| Self-learning FP reduction | 86% (7 to 1) | Exp 223 | DERIVED | Post-hoc analysis of Exps 220/221 held-out cohorts (inherits live inputs, no new inference) |
| Global consistency checker | 100% detection, 0% FP | Exp 271 | SIMULATED | Hand-crafted consistent/contradicted chains; ~1ms latency, no model inference |
| Agent rollback | 100% success | Exp 273 | SIMULATED | 10 hand-crafted workflows, `live_mode=false` |
| Z3+LLM on GSM8K arithmetic | 80% detection, 0% FP | Exp 276 | SIMULATED | Canned CI cohort (10 cases), `live_mode=false`, 2.54s runtime |
| Adversarial semantic grounding | +40pp lift | Exp 279 | SIMULATED | Model field explicitly `"Gemma4-E4B-it (simulated)"` |
| Confidence-weighted repair | 86.7% FP avoidance | Exp 332 | PLACEHOLDER | Fast-path deliverable: `duration_s=0.0`, constant confidence scores, no inference |
| SinkProbe pre-filter | 60% skip rate, 0% FN | Exp 348 | SIMULATED | `inference_mode="simulated"`, 50 synthetic samples, 1.5s total |

**Headline claim (honest):** Three live GPU results — HumanEval +3.0pp, PBT 99.3%, Typed IR +4.9pp — with statistical confidence intervals and multi-minute GPU runtimes. The remaining six entries motivated the research but need live re-runs before they can be cited externally. Re-verification is queued as an explicit milestone goal.

## Product Roadmap

| Tier | Products | Status |
|------|----------|--------|
| A: Ship Now | LLM output verification, code quality scorer, candidate ranker | Built |
| B: Build Next | Safety classifier (from gpt-oss-safeguard), compliance checker, multi-agent arbiter | Planned |
| C: Needs Hardware | Factual grounding gate, anomaly detector, prompt quality scorer | Phase 2 |
| D: Foundation Model | Data quality filter, synthetic data validator, test oracle | Phase 3 |

## Hardware Acceleration

| Hardware | Type | Status |
|----------|------|--------|
| D-Wave Advantage | Quantum annealing | Sampler built (Exp 320) |
| Extropic Z1 | Thermodynamic sampling | Early access 2026 |
| KV260 FPGA | Digital Ising (4K spins) | Software sim done, needs bitfile |
| RTX 3090 x2 | CUDA GPU | Working |
| Vulkan compute | Universal GPU | Planned for Phase 2 |
| Intel Loihi 2 | Neuromorphic | Need INRC access |
| NTT CIM | Coherent optical (100K+ spins) | Monitor |

## Phase 3: EBM/EBT Foundation Model

The long-term vision: an open-source foundation model based on hardware-acceleratable Energy-Based Models, with functional parity to Logical Intelligence's Kona.

- Continuous energy landscapes (bridge from discrete Ising/Z3)
- Non-autoregressive reasoning (generate via energy minimization)
- Language-free verification (learn constraint structure directly)
- Open-source (Apache 2.0) and hardware-portable (Vulkan/FPGA/D-Wave/TSU)
