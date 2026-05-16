# Research Roadmap: Milestone 2026.05.199

**Milestone Title:** Phase 4 CLaRa Integration, Self-Learning Strict Epsilon Constraints, and Hardware Unblocking
**Date:** 2026-05-16

## 1. Context and Outcomes of .198
The previous milestone (.198) successfully completed the Phase 1 ship completion: MCP docs + reproducer + CoT2-Meta integration + .197 audit. While the structural foundations of verification routing (CoT2-Meta) are solidified, three primary gaps remain between our current capability and the PRD vision:

1. **Continuous Self-Learning Forgetting:** The FR-11 requirement for continuous self-learning currently suffers from catastrophic forgetting in the policy buffer without rigorous parameter-level protection.
2. **Continuous Latent Generation:** The Phase 4 CASAL integration is still reliant on discrete token generation. Recent advances in Continuous Latent Reasoning (CLaRa-V) provide the pathway to bridge Carnot's verification from discrete token generation to fully continuous latent space evaluation.
3. **Hardware Acceleration Bottlenecks:** Both the KV260 FPGA bitfile synthesis and the eGPU ROCm paths are stalled, preventing actual hardware latency measurements for live continuous sampling.

This milestone introduces Gradient-Guided Epsilon Constraint (GEC) to mathematically enforce non-forgetting in self-learning, and CLaRa-V to push the verification mechanism into the continuous domain, alongside strategic hardware unblockers.

## 2. Phase Descriptions

### Phase 1: Hardware Paths & Math Foundations
Attempt to finally unblock the Thunderbolt RX 7900 XTX eGPU path via ROCm/JAX, and structure the KV260 v4 RTL parameters. Simultaneously, implement the core Rust mathematical primitives for the Gradient-Guided Epsilon Constraint (GEC) projection.

### Phase 2: Verifier & Continuous Latent Constraints
Develop the CLaRa-V continuous latent representation schema in Python and interface it with a PiNet-inspired differentiable projection layer to enforce hard constraints natively in the continuous space, bypassing symbolic synthesis overheads.

### Phase 3: E2E Self-Learning & Reasoning Generation
Integrate GEC into the SEAL continuous self-learning loop. Execute the CLaRa-V continuous sampling tests using the flagship local GGUF models (`unsloth/gemma-4-31B-it-GGUF` and `unsloth/Qwen3.6-35B-A3B-GGUF`) to realize Phase 4 EBM-driven reasoning.

### Phase 4: Retro & Audit
Perform standard E2E pipeline verification of the new Phase 4 continuous sampling path, audit the .198 findings, and conclude with the operational retro.

## 3. Dependency Graph
- Phase 1 (GEC Math) unblocks Phase 3 (GEC SEAL Loop).
- Phase 2 (CLaRa-V Schema & PiNet) unblocks Phase 3 (Continuous Reasoning Generation).
- Phase 4 depends on all prior phases successfully producing artifacts or explicitly failing via gate constraints.

## 4. Hardware Requirements
- **Mandated Models:** `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE), `unsloth/gemma-4-31B-it-GGUF` (flagship dense), `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE).
- **Physical Targets:** Thunderbolt RX 7900 XTX eGPU for ROCm testing.
