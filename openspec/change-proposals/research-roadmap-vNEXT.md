# Research Roadmap: Milestone 2026.05.214

**Title:** Thermodynamic Generation, Hardware KANs, and Dynamic CSL
**Status:** Planned

## 1. Context & Motivation

Milestone 2026.05.213 successfully established the Process-Reward Energy Model (PREM), Ising ALPS sampling, and the Dynamic Test-Time Compute (TTC) Controller. However, significant gaps remain between the current state and our PRD vision:
1.  **Hardware Acceleration:** We have not yet moved constraint sampling logic onto physical FPGA (KV260) infrastructure, despite architectural blueprints.
2.  **Continuous Self-Learning (CSL) Stability:** Prior milestones observed mode collapse during sequential task learning in EBMs.
3.  **Generative Decoding:** While we have discrete biasing, our generation loop lacks true continuous energy-landscape constraints (Thermodynamic Generation).

Recent literature (arxiv:2605.02104, arxiv:2605.08412, OpenReview 2026, and arxiv:2605.14558) provides concrete paths to address these gaps.

## 2. Architecture Updates

This milestone introduces three major architectural components:
-   **Thermodynamic Generation & ActFocus:** Replaces standard autoregressive decoding with a generation loop that incorporates thermodynamic continuous penalties and redistributes token-level energy gradients to prioritize action tokens.
-   **Substrate-Aware KANs:** A multiply-free Kolmogorov-Arnold Network topology designed strictly for LUT/BRAM allocation on FPGAs.
-   **Dynamic Resolution EBMs:** Introduces resolution scaling to the core EBMs to prevent catastrophic forgetting in the CSL loop.

## 3. Phase Descriptions

### Phase 0: Activation
Archive the completed `.213` milestone and activate `.214` in the conductor framework.

### Phase 1: Generative Decoding Enhancements
-   Implement ActFocus token reweighting (arxiv:2605.14558) into the PREM trainer to improve credit assignment.
-   Implement Thermodynamically Constrained Neural Generation (arxiv:2605.02104) to strictly guide autoregressive decoding.
-   Benchmark using `unsloth/Qwen3.6-35B-A3B-GGUF`.

### Phase 2: Hardware-Aware KAN Verification
-   Design Substrate-Aware KAN architectures tailored for KV260 FPGAs (arxiv:2605.08412).
-   Create LUT/BRAM allocation scripts to estimate hardware resources.
-   Execute an end-to-end KV260 simulation to identify bottleneck latencies in the Rust/Python bridge.

### Phase 3: Continuous Self-Learning Resilience
-   Introduce Dynamic Resolution for Continual EBM Learning (OpenReview 2026).
-   Run retention benchmarks using `unsloth/gemma-4-26B-A4B-it-GGUF` to ensure sequential task learning avoids mode collapse.

### Phase 4: Capstone & Retrospective
-   Full end-to-end integration and verification of all new modules.
-   Synthesize performance and reliability insights.

## 4. Hardware Requirements
-   **Required:** 1x GPU with 24GB+ VRAM for SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`).
-   **Simulated:** KV260 FPGA board (simulated via Python latency mock, physical integration planned for subsequent milestones).

## 5. Dependency Graph
-   Phase 1 depends on baseline PREM training from .213.
-   Phase 2 depends on core KAN architecture.
-   Phase 3 is independent but must be integrated into Phase 4.
-   Phase 4 depends on the successful completion of Phases 1, 2, and 3.