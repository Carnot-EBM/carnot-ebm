# Carnot Research Roadmap: vNEXT (Milestone 2026.05.174)

**Title:** Phase 2 Continuous Latent Trace Editing and Verification-Compute Routing
**Target:** 2026-05-174
**Status:** DRAFT

## 1. What the Previous Milestone Proved

Milestone `.173` successfully closed out the Phase 1 Ship-Track Density tasks. Key outcomes included scaling the zero-false-accept constraint stack to multi-turn drift, turning FR-11 query-time updates into verified memory growth without soundness mistakes, and stabilizing the local SOTA GGUF runtime. However, we identified three critical gaps to close before Phase 3 (Globally Scored Reasoning) can become a reality:
1. **Verification-Compute Routing:** Generating a full sequence before validating is too costly and leads to high residual drift.
2. **Kona-Parity Continuous Latent Trace Editing:** Traditional LLM reasoning is autoregressive and discrete. EBMs like Kona 1.0 succeed by reasoning in continuous latent space where energy-minimization enables non-autoregressive generation and local editing.
3. **Hardware Execution Target Migration:** Extropic's Z1 chip is confirmed for Early Access 2026 as a mass-manufacturable CMOS TSU. Carnot must transition its THRML simulation stubs to support DTM (Denoising Thermodynamic Model) architectures.

## 2. Architecture Diagram

```mermaid
graph TD
    subgraph Generation Layer
        LLM[Mandated SOTA GGUF] --> CCTU[CCTU Benchmark]
        LLM -.->|Trace generation| Monitor(Interwhen Monitor)
        Monitor -.->|Interrupt| SP(HoVer Safe-Prefix Continuation)
    end

    subgraph Kona-Parity Latent Reasoning
        NAG[Non-Autoregressive Latent Generator]
        CASAL[CASAL Primal-Dual Sampler]
        NAG -->|Continuous Trace| CASAL
        CASAL -->|Global Energy Score| TraceEditor[Gradient-Based Trace Editor]
    end

    subgraph Hardware Stub Layer
        DTM[DTM Architecture Interface] --> THRML[THRML TSU Simulator]
        THRML --> Z1[Z1 Accounting / Readiness]
    end

    Generation Layer --> Kona-Parity Latent Reasoning
    Kona-Parity Latent Reasoning --> Hardware Stub Layer
```

## 3. Phase Descriptions

### Phase 1: Verification-Compute Routing and Telemetry (Exps 2101-2104)
Implement test-time monitoring to interrupt generation when intermediate constraints fail, reducing wasted compute. Implement HoVer-style safe-prefix continuation so we do not discard entire traces when only the suffix drifts. Automate prompt-to-constraint validation (ConstrainPrompt) to expand verification coverage without manual coding.

### Phase 2: Continuous Trace Editing (Kona Parity) (Exps 2105-2108)
Transition from step-by-step token prediction to trace-level non-autoregressive generation in a continuous latent space. Incorporate gradient-based refinement and global energy scoring to emulate Kona 1.0 capabilities, benchmarking on structured tasks like Sudoku.

### Phase 3: Differentiable Constraint Layers (Exps 2109-2111)
Rescue the PiNet Douglas-Rachford splitting prototype. Use it alongside the CASAL primal-dual sampler to guarantee zero-violation safety in the continuous domain. Transition the continuous self-learning loop to Energy-Based Fine-Tuning (EBFT) using feature matching rather than token-level CE loss.

### Phase 4: Hardware Readiness (Z1) (Exps 2112-2114)
Align the THRML simulator stack with the expected Extropic Z1 SDK interfaces, specifically for Denoising Thermodynamic Models (DTMs). Perform a no-synthesis hardware resource accounting for probabilistic sampling on Z1, ensuring we remain within bounds. Perform the retrospective.

## 4. Hardware Requirements
- **Local GPUs:** Dual RTX 3090 (or similar) mandated for running local SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`).
- **Hardware Claims:** No synthesis or authenticated execution claims are permitted for Extropic Z1, XTR-0, or Kona hardware in this milestone. Use CPU-based THRML simulations only.

## 5. Dependency Graph

- Phase 1 (2101-2104) is foundational for dynamic routing.
- Phase 2 (2105-2108) builds continuous space reasoning.
- Phase 3 (2109-2111) depends on Phase 2 continuous traces.
- Phase 4 (2112-2114) depends on Phase 2 models to run through THRML stubs.
