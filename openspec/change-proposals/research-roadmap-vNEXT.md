# Research Roadmap vNEXT (Milestone 2026.05.179)
## Phase 4 Bijection Integration, NLA Integrity Audit, and CEM Substrate

### 1. Previous Milestone Recap (.178)
Milestone .178 successfully validated the Phase 4 alpha_t replacement via maximum-caliber and finalized the NLA 16th verifier Task 4. The codification of the bijection-invariance findings (exp1719) hit a locatable error and is carried forward.

### 2. Strategic Objectives for .179
- **Integrate the LM-EBM Bijection:** Connect Carnot's verifier-as-free-energy theory to the formal autoregressive-EBM bijection discovered in recent arXiv literature (arXiv:2512.15605v3).
- **NLA Verifier Integrity:** Perform descriptive collision audits to prevent auto-interpretability inflation, and reweight the expanded k=16 ensemble to avoid behavioral entanglement.
- **Compositional Energy Minimization (CEM):** Build the CEM substrate for Phase 4, allowing complex constraint solving by summing smaller independent energy landscapes.
- **Continuous Self-Learning Feedback:** Route the NLA white-box signals into the continuous self-learning loop (FR-11) as a high-quality feedback mechanism.
- **Mandated Hardware & SOTA Inference:** Benchmark the Qwen3.6-35B-A3B and Gemma-4 31B models on the full k=16 ensemble via DualGPURunner, and calculate KANELÉ LUT-based hardware accounting for CEM.

### 3. Phases
**Phase 1: Carry-Forward & Theoretical Grounding**
- Codify the bijection-invariance finding into paper-v6.
- Empirical δ Calculation for Verify-Repair Convergence.
- Theoretical LM-EBM Bijection Re-derivation.

**Phase 2: NLA Integrity & Continuous Learning**
- NLA-Class Verifier Descriptive Collision Audit.
- Continuous Self-Learning Loop with NLA Signal Integration.
- Behavioral Entanglement Reweighting for the k=16 Ensemble.

**Phase 3: CEM Substrate & Verification**
- CEM Substrate Prototype.
- Scale CEM to n=64 with KANELÉ Hardware Accounting.
- Formal Verification of NLA Abstractions via PWA/MILP.

**Phase 4: SOTA Dual-GPU Parity & Retrospective**
- SOTA GGUF Parity Benchmark with k=16 Ensemble.
- Milestone .179 Retrospective.

### 4. Hardware Requirements
- Local Dual RTX 3090s for the SOTA GGUF Parity (DualGPURunner mandatory).
- CPU-only for CEM prototype and theoretical formalizations.
- KANELÉ hardware accounting runs entirely in CPU simulation/heuristic scripts (no Vivado synthesis required).