# Milestone 2026.05.220: Dynamic EBCN Extraction, Hardware-in-the-Loop Decoding, and Continuous Semantic Pruning

**Status:** Proposed
**Date:** {date}

## Context & Vision Gap

Milestone `2026.05.219` successfully integrated Primal-Dual Guided Decoding, formalized KAN synthesis using SMT barrier certificates, and implemented LipKAN robustness constraints. However, three critical gaps remain between the current state and the PRD vision:

1. **Constraint Extraction Bottleneck:** While the EBCN State-Space Model verifies structural coherence, we lack a real-time capability to *extract* these formal constraints dynamically from unstructured user prompts on our mandated SOTA GGUF models.
2. **Hardware Execution:** The KANELÉ synthesis was validated via OSS-CAD-Suite, but the energy scores and constraints are still evaluated in software. We must close the loop with Hardware-in-the-Loop Energy Decoding (HILED) on physical or simulated KV260 hardware interfaces.
3. **Continuous Learning Saturation:** FR-11 established dynamic Lipschitz boundaries, but as the constraint memory grows, continuous self-learning faces replay saturation. We need semantic pruning to retain high-value, non-redundant structural constraints without mode collapse.

## Phase 0: Activation & Archival
Archive the results of milestone `.219` and prepare the operational logs for the `.220` execution run.

## Phase 1: Dynamic Structural Constraint Elicitation
We will leverage the Joint Latent Energy and Dynamic Constraint Elicitation (JLE-DCE) paradigm to parse unstructured text into formal EBCN matrices using our mandated SOTA MoE models (`unsloth/Qwen3.6-35B-A3B-GGUF`). This completely deprecates the legacy regex-based extraction layers.

## Phase 2: Hardware-in-the-Loop Energy Decoding (HILED)
Transitioning the KANELÉ synthesizable blocks into a realistic runtime environment. We will build a PCIe/AXI software bridge that mocks or integrates with physical KV260 boards, allowing the Energy-Guided Decoding loop to offload energy scoring to the hardware tier.

## Phase 3: Continual EBM Semantic Pruning
To prevent catastrophic forgetting and replay buffer bloat, we implement Semantic Pruning. This unsupervised neural process will evaluate the similarity of stored EBCN constraints and prune redundant entries, ensuring FR-11 policies remain tight and responsive.

## Phase 4: Capstone E2E Evaluation
A live E2E GPU integration test wrapping the new dynamic elicitation, hardware-delegated scoring, and pruned continuous learning cache into a single benchmark evaluated by the local SOTA MoE model.

## Task Graph & Routing

- Phase 0: exp2217
- Phase 1: exp2218 -> exp2219 -> exp2220
- Phase 2: exp2221 -> exp2222 (Opus Hardware Routing) -> exp2223
- Phase 3: exp2224 -> exp2225 -> exp2226
- Phase 4: [2220, 2223, 2226] -> exp2227 -> exp2228 (Retro)
