# Carnot Research Roadmap: vNEXT (Milestone 2026.05.218)

## 1. What Previous Milestones Proved
Milestone 2026.05.217 successfully integrated Muon-OGD spectral orthogonal gradient projection for CSL stability, digitally optimized ALPS initializations for fast sampling, and the Wahkon RKHS KAN verifier. While these resolved catastrophic forgetting in isolated tests and provided theoretical rigor, our hardware scaling for KANs remained constrained by pure CPU evaluation, and constrained decoding still faces a "reasoning gap" where strict formal boundaries degrade general reasoning capability.

## 2. The 3 Biggest Gaps to PRD Vision
1. **Inference-Time Energy Verification:** The PRD demands verifiable reasoning (FR-12). Recent arXiv (2025/2026) highlights Energy-Based Transformers (EBT) and Energy Outcome Reward Models (EORM). Carnot currently checks constraints after-the-fact; we need iterative energy minimization during inference to achieve genuine "System 2" thinking.
2. **Hardware KAN Deployability:** We lack a direct translation of Kolmogorov-Arnold Networks to FPGA. The recent KANELÉ framework (arxiv:2512.12850) maps learnable 1D splines directly to FPGA LUTs, promising a 2700x speedup. We must implement this to hit the 10x throughput NFR-01.
3. **Safe Online Constraint Learning:** While Muon-OGD stabilized forgetting, continuous self-learning (FR-11) lacks safety guarantees. New online regression oracles (pessimistic constraints) and Crosscoder feature-readout constraints are needed to guarantee safe online adaptation.

## 3. Phase Descriptions

### Phase 0: Activation
Archive 2026.05.217 and initialize 2026.05.218.

### Phase 1: EORM & EBT Inference-Time Verification
Implement a 55M parameter Energy Outcome Reward Model (EORM) to provide scalar energy scores for CoT solutions, and integrate Energy-Based Transformer (EBT) iterative energy minimization to allow the model to refine its predictions based on energy gradients. We also integrate CRANE (arxiv:2502.09061) reasoning-augmented grammar to balance formal constraints with natural reasoning.

### Phase 2: KANELÉ Hardware Evolution
Implement the KANELÉ (arxiv:2512.12850) framework, translating KAN splines directly to FPGA Lookup Tables (LUTs). Draft and synthesize the KV260 bitstream to bypass the k_max=5 bottleneck via efficient LUT usage.

### Phase 3: Pessimistic CSL & Feature Memory
Advance Tier 3 Continuous Self-Learning (FR-11) by introducing an online regression oracle that imposes pessimistic safety constraints, ensuring zero catastrophic violations during online updates. Additionally, incorporate Schema-Constrained Generative Memory (SCG-MEM) to safely manage agentic memory retrieval.

### Phase 4: Capstone Evaluation
Live GPU evaluation using mandated SOTA models (`unsloth/gemma-4-31B-it-GGUF` and `unsloth/Qwen3.6-35B-A3B-GGUF`), running the complete EORM+EBT pipeline with KANELÉ hardware simulation.

## 4. Hardware Requirements
- **2x NVIDIA RTX 3090 (CUDA):** Required for Phase 4 Capstone live inference.
- **KV260 / OSS-CAD-Suite:** Required for synthesizing the KANELÉ LUT RTL.

## 5. Dependency Graph
- Phase 1 (EORM/EBT) -> Phase 4 (Capstone)
- Phase 2 (KANELÉ RTL) -> Phase 2 (KV260 Synthesis) -> Phase 4 (Capstone)
- Phase 3 (Pessimistic CSL) -> Phase 4 (Capstone)