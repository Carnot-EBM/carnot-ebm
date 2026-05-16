# Research Roadmap: Milestone 2026.05.197

**Milestone Title:** Thermodynamic EBT Verification + Formal Proof Scaling
**Date:** 2026-05-16

## 1. Context and Outcomes of .196
The previous milestone (.196) successfully completed Fast-Slow codification, PyPI workflow checks, HF mirroring, and comprehensive audits. However, significant gaps remain between the current state and the PRD vision:
1.  **Thermodynamic Hardware Scaling:** We need to natively integrate thermodynamic hardware abstractions (like THRML) to bypass CPU Gibbs sampling bottlenecks.
2.  **EBT System-2 Verification:** Energy-Based Transformers (EBT) system-2 inference scaling is required to optimize energy dynamically across multi-step logic.
3.  **Formal Mathematical Proofs:** Bridging the verifiable solver with rigorous formal proofs (e.g., Lean 4, via Kona/Aleph concepts) to achieve zero-false-accept logic extraction.

## 2. Phase Descriptions

### Phase 1: Thermodynamic Hardware Abstractions & THRML Integration
Focus on adopting `thrml` for hybrid digital-thermodynamic sampling abstractions and perform parity tests against our local baselines, laying the groundwork for hardware deployment. We will also perform a KAN hardware complexity audit.

### Phase 2: EBT-Driven System 2 Verification
Implement an Energy-Based Transformer decoding strategy on local SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`), optimizing energy at test-time to achieve System-2 level logic verification.

### Phase 3: Formal Verification Loop (Lean 4 constraint bridge)
Develop zero-false-accept logic extraction inspired by Logical Intelligence's Kona/Aleph, synthesizing logic constraints into machine-checkable proofs (Z3/Lean compatible).

### Phase 4: Multi-Agent EBM-CoT Self-Learning
Enhance FR-11 continuous self-learning loops by implementing verifier-governed memory promotion to ensure non-forgetting and soundness without completeness degradation.

## 3. Dependency Graph
- Phase 1 must succeed for Phase 4 energy accounting.
- Phase 2 sets the runtime bounds for Phase 3 constraint elicitation.
- Phase 4 depends on the formal verification loop (Phase 3) for valid skill promotion.

## 4. Hardware Requirements
- Local SOTA GGUFs: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`
- Local simulator execution for THRML
