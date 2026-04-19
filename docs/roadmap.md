# Carnot Roadmap

*Auto-updated by the research conductor as experiments complete.*

## Current Milestone

**2026.04.38: "Break the Credibility Ceiling — Gemma4 Quantized, 100q+ Live, GPU 1 Activated"**

| # | Experiment | Status |
|---|-----------|--------|
| 500 | Gemma4 INT4 Quantization — RETRO-048 root cause fix (~9 GiB GGUF Q4_K_M) | Complete — unblocks RETRO-033/038/039 live re-runs |
| 501 | Conductor CPU Routing + VRAM Budget Ledger | Complete — diagnostic checkpoint |
| 502 | Live 100q Precision v6 — RETRO-033 sixth attempt with quantized Gemma4 | In Progress |
| 503-512 | 200q VeriCoT+VPRM v4, GSM-Symbolic v4, Retroactive DualGPU sweep, Semantic Energy Tier 0d, NUP v3 with CLAP, KAEM distribution family, PPSEBM energy-replay, JEPA Live Retrain v4, AMD XDNA NPU inference, retro | Queued |

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
| 2026.04.33 | First Live Results, ThinkPRM Bridge, Boltzmann-GPT Repair | 437-449 | LongRunBenchmarkExecutor (RETRO-026 closed), Tier 2 cross-session constraint memory relay, BoltzmannRepairBridge, operational retro v7 |
| 2026.04.34 | VeriCoT Extraction, EBM-CoT Calibration, First Positive Numbers | 450-461 | **First positive verify-repair number** (Exp 451, +5pp, LIVE); Gemma4 tokenizer bug closed (RETRO-028); EBM-CoT Langevin calibration; VeriCoT/VPRM step validators |
| 2026.04.35 | Scale the First Positive — 200q Credibility, Process Hardening, FPGA Bring-Up | 462-473 | DeliverableGuard + DualGPURunner harness, session health check, live 100q precision statistical scale-up, KAEM 3.4x speedup at n=50, PPSEBM Tier 2 isolation, KV260 RTL + AXI backend (bitfile pending hardware) |
| 2026.04.36 | Fix the Root Cause — GPU VRAM Gate, Live 200q Credibility, JEPA Recovery | 474-486 | GPUVRAMGate, conductor dedup + partial-result handoff, DualGPU harness enforcement (53 scripts), batching enforcement audit, NUP Probe v1, PPSEBM real-data validation; honest negatives on JEPA quality-gate and KAEM at n=1000 |
| 2026.04.37 | Break the VRAM Deadlock — Credibility at Last, JEPA Recovery, Surprise-Driven Replay | 487-499 | **JEPA AUC recovered 0.281→0.967 via curriculum training** (Exp 492, pending live validation in Exp 510); GPUVRAMGateV2 (kill-before-check); batching pre-commit hook; GPU thermal gate; 100% retro adoption rate (first ever); KAEM at n=5000 definitively slower than MCMC across the range, FPGA-only path confirmed |

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
| First positive verify-repair | +5pp (LIVE) | Exp 451 | LIVE | Gemma4-E4B-it live GSM8K, 50 questions, Wilson CI — first time the verify-repair loop produced a signed positive signed_improvement on live data |
| JEPA step-quality discriminator | AUC 0.967 (pending live validation) | Exp 492 | DERIVED | Curriculum training (high→low confidence ordering) on Exp 442's Z3-annotated CoT pairs recovered from an AUC 0.281 regression.  Real mechanistic fix (prevents majority-class collapse), but the eval set may share structure with the training capture — **Exp 510 (milestone 2026.04.38) re-runs the discriminator on genuinely fresh live CoT pairs**.  If AUC holds near 0.967 there, the breakthrough is real; if it collapses to 0.5-0.7, the number was leakage.  Do not cite externally until Exp 510 lands. |

**Headline claim (honest):** Live-validated signed improvements are now three-plus-one: HumanEval +3.0pp, PBT 99.3%, Typed IR +4.9pp, and Exp 451's first positive verify-repair number on GSM8K (+5pp, 50q).  The JEPA 0.967 AUC and several simulated entries remain motivating but need live re-runs before they can be cited externally; these re-runs are explicitly scheduled into the active roadmap (Exps 502/503/504/510).

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
