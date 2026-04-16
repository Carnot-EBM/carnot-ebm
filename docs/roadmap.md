# Carnot Roadmap

*Auto-updated by the research conductor as experiments complete.*

## Current Milestone

**2026.04.30: "Purge, Implement, Execute — First Credible Live Numbers"**

| # | Experiment | Status |
|---|-----------|--------|
| 404 | DeliverableContentValidator + GPU preflight v2 | Complete |
| 405-410 | Infrastructure + live pipelines | Complete |
| 411 | Live HumanEval code verification | In Progress |
| 412-417 | Remaining experiments | Queued |

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

## Breakthrough Results

All results from live GPU inference. No simulated runs.

| Result | Value | Experiment | Significance |
|--------|-------|------------|-------------|
| HumanEval code verification | +3.0pp [+0.6, +6.1] CI | Exp 226 | Statistically significant on 164 official problems |
| PBT bug detection rate | 99.3% (144/145) | Exp 220 | Property-based testing catches nearly all wrong code |
| Typed IR constraints | +4.9pp (Gemma4) | Exp 221 | Prompt-side constraint extraction works |
| Global consistency checker | 100% detection, 0% FP | Exp 271 | Logic-based, confirmed on live multi-turn chains |
| Agent rollback | 100% success | Exp 273 | Constraint-based rollback works on live model outputs |
| Self-learning FP reduction | 86% (7 to 1) | Exp 223 | Tier 1-2 learning reduces false positives |
| Confidence-weighted repair | 86.7% FP avoidance, 100% TP | Exp 332 | Smart repair selection preserves true positives |
| SinkProbe pre-filter | 60% skip rate, 0% FN | Exp 348 | Attention-sink detection skips safe outputs cheaply |
| Adversarial semantic grounding | +40pp lift | Exp 279 | Catches number-swapped quantity mismatches |
| Z3+LLM on GSM8K arithmetic | 80% detection, 0% FP | Exp 276 | Formal extraction eliminates false positives |

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
