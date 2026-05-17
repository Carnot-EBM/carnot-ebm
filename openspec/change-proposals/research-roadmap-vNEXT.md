# Milestone 2026.05.212: Hardware-Assisted DAB, Substrate-Aware CSL, and Continuous Latent Verification

**Status:** Proposed
**Date:** 2026-05-16

## 1. Context and Motivation

Milestone `2026.05.211` achieved Constraint-Aware Retrieval Module (CARM) integrations, Discrete Auto-Regressive Biasing (DAB) in software, and Substrate-Aware KAN formulations. However, to achieve the hardware efficiency mandated by the PRD and escape continual learning mode collapse, we must map our decoding enhancements physically and dynamically. Recent 2026 literature highlights **Hardware-Assisted Constrained Decoding using Energy Landscapes** (arXiv:2605.10112) and **Mode-Collapse Recovery via Substrate Shifting** (arXiv:2605.12304) as the next natural steps.

This milestone addresses the following major gaps:
1. **Hardware-Accelerated Decoding:** We have DAB in software, but we need Energy-Guided Decoding with Hardware-Assisted Verification to evaluate landscapes on simulated FPGA/LUTs during generation.
2. **Continual Learning Mode-Collapse:** Dynamic Resolution CSL (from .211) can collapse. Substrate Shifting allows the underlying KAN LUT grids to physically translate to push the EBM out of local minima.
3. **Continuous Latent Validation:** Bridging the gap between Kona-style continuous latent traces and our discrete logic verifiers to formally verify reasoning.

## 2. Architecture Impact

- **Phase 1 (HW-DAB):** DAB operations will be offloaded to a LUT representation, significantly dropping theoretical decoding latency.
- **Phase 2 (Substrate Shifting CSL):** The Continuous Self-Learning loop will explicitly detect mode concentration and translate the KAN energy boundaries, ensuring zero forgetting.
- **Phase 3 (CLR Bridge):** A new latent verification bridge will directly map continuous EBM representations back into verifiable discrete formats.

## 3. Phase Descriptions

### Phase 0: Activation
- **Exp 2132:** Archive .211 and activate .212.

### Phase 1: Hardware-Assisted DAB Module
- **Exp 2133:** Implement HW-DAB mapping energy updates onto LUTs.
- **Exp 2134:** No-synthesis hardware accounting for HW-DAB vs CPU baselines.
- **Exp 2135:** Evaluate HW-DAB semantic preservation on GSM8K using mandated SOTA GGUF models.

### Phase 2: Substrate-Aware Continual Learning
- **Exp 2136:** Introduce Substrate Shifting grid parameters to the CSL loop.
- **Exp 2137:** Explicit mode-collapse detection and substrate pushing.
- **Exp 2138:** Zero-forgetting evaluation via multi-task logic boundaries.

### Phase 3: Continuous Latent Reasoning Validation
- **Exp 2139:** Build the CLR Verifier Bridge.
- **Exp 2140:** Benchmark the CLR bridge alongside `unsloth/gemma-4-31B-it-GGUF`.

### Phase 4: Integration and Retrospective
- **Exp 2141:** E2E Benchmark combining HW-DAB, Substrate Shifting, and CLR.
- **Exp 2142:** Full Milestone 2026.05.212 Retrospective.

## 4. Hardware Requirements
- **Local SOTA GGUFs:** `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` required for headline tests.
- **GPU Cluster:** Standard local dual-GPU workspace environment.
- **FPGA/KV260:** Source-level RTL simulation only; no physical board claims.

## 5. Dependency Graph
```text
Exp 2132 (Activation)
  |-- Phase 1 (HW-DAB)
  |      |-- Exp 2133 --> Exp 2134 --> Exp 2135
  |-- Phase 2 (CSL Mode Collapse)
  |      |-- Exp 2136 --> Exp 2137 --> Exp 2138
  |-- Phase 3 (CLR Validation)
  |      |-- Exp 2139 --> Exp 2140
  |-- Phase 4 (Integration)
         |-- Exp 2141 --> Exp 2142
```
