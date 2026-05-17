# Milestone 2026.05.221: ActFocus RL, KAN-CL Continual Learning, Wahkon RKHS, and AdamFLIP Hard Constraints

## 1. Goal
Address three massive gaps between the Carnot PRD vision and current state:
1. Overcoming the action bottleneck in token-level RL attribution.
2. Achieving FR-11 continuous self-learning for KANs without buffer bloat via KAN-CL.
3. Enforcing strict equality constraints through AdamFLIP instead of soft penalties. 

Additionally, we explore Wahkon RKHS networks for superior finite-sample statistical guarantees over standard Spline-KANs and implement thermodynamic initializations.

## 2. Context & Previous Milestone Proofs
Milestone `.220` proved that Dynamic EBCN Extraction (JLE-DCE) allows latent constraint extraction and Primal-Dual decoding seamlessly modifies logits. However, we found that buffer-based semantic pruning scales poorly, soft penalties leave edge-case violations, and token-level PPO misallocates gradients.

### The 3 Biggest Gaps Identified
1. **Continuous Self-Learning (FR-11) Limitations:** Buffer pruning retains semantic knowledge but suffers boundary degradation. KAN-CL fixes this by shifting memory retention to the per-knot topological level.
2. **Hard Constraints Drift:** Existing constraint training is soft and subject to drift. AdamFLIP treats physics/constraint-informed training as a feedback-linearized dynamical system, ensuring hard mathematical bounds with zero false accepts.
3. **Statistical Scaling Bottlenecks:** While KANs offer high capacity, they lack strict MAP guarantees. Wahkon deep RKHS architectures provide these exact statistical guarantees for high-dimensional inference.

## 3. Phase Plan

### Phase 0: Archive & Activate
- Gracefully archive milestone `.220` and set up the foundation for `.221`.

### Phase 1: ActFocus RL & Token-Level Energy Bottleneck
- Implement ActFocus to track variance across rollouts, increasing gradients on critical action tokens and suppressing redundant reasoning tokens.
- Evaluate heavily on SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`).

### Phase 2: KAN-CL Continual Learning (FR-11)
- Map the FR-11 mandate directly to the KAN splines via per-knot importance regularization.
- Execute an FR-11 self-learning loop measuring non-forgetting rates across multiple disparate domains.

### Phase 3: Wahkon RKHS Integration & Benchmarking
- Implement Wahkon as an alternative to KAEMEnergy for statistically principled deep RKHS superposition.
- Compare Wahkon and KAEMEnergy inverse-transform sampling fast paths.

### Phase 4: AdamFLIP Constraints & Thermodynamic Capstone
- Shift Verifier training from soft-penalties to AdamFLIP hard constraint enforcement.
- Introduce Mpemba-effect inspired initializations to accelerate Langevin sampling for thermodynamic computation scaling.
- Conclude with a rigorous E2E GPU validation via mandated SOTA endpoints.

## 4. Hardware and Dependencies
- **Hardware Requirement:** Dual GPU ROCm/CUDA for live model baselines.
- **Model Specs Mandate:** `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`.
