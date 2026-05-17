# Carnot Research Roadmap: vNEXT (Milestone 2026.05.217)

## 1. What Previous Milestones Proved
Milestone 2026.05.216 focused on KAN Continual Learning (KAN-CL), ActFocus token-level reweighting, and Energy-Guided Decoding. While these mechanisms laid the groundwork for Tier 3 continuous self-learning, our telemetry indicates that catastrophic forgetting remains a severe issue during online adaptation. Furthermore, our hardware acceleration paths (ALPS Langevin sampling and KV260 k_max=5 RTL) face challenges: ALPS thermalization is too slow for real-time decoding, and the k_max=5 FPGA limit restricts the complexity of transpilable constraints.

## 2. The 3 Biggest Gaps to PRD Vision
1. **Tier 3 Continuous Self-Learning (CSL) Instability:** The PRD (FR-11) demands autonomous directed self-learning. Currently, when the system adapts its constraints online, it suffers from catastrophic forgetting. We must implement spectral-norm orthogonal projection (Muon-OGD) to constrain updates safely.
2. **Slow Thermodynamic Sampling:** Langevin sampling is the backbone of our energy-guided decoding, but its thermalization time is too high. We need digitally optimized initializations (Mpemba effect) to suppress slow relaxation modes.
3. **Statistical Guarantees for Verifiers:** Our KAN-based verifiers lack finite-sample performance bounds. Integrating RKHS superposition (Wahkon) will provide the necessary theoretical rigor and out-of-distribution robustness.

## 3. Phase Descriptions

### Phase 0: Activation
Archive 2026.05.216 and initialize 2026.05.217.

### Phase 1: Tier 3 Continuous Self-Learning Stability
Integrate the Muon-OGD spectral orthogonal gradient projection (arxiv:2605.08949) into the CSL pipeline. This replaces standard Frobenius-norm updates with spectral-norm-aware orthogonal projections, ensuring new constraints do not overwrite previously learned ones.

### Phase 2: Fast Thermodynamic Sampling & Hardware Evolution
Implement digitally optimized initializations (arxiv:2603.24183) for the ALPS module, shifting some initialization compute to a digital pre-processor to exponentially speed up Langevin thermalization. Additionally, draft the NeuroRing (arxiv:2604.28059) RTL to bypass the KV260 k_max=5 bottleneck via a stream-dataflow ring architecture.

### Phase 3: KAN Verifier Rigor
Adopt the Wahkon architecture (arxiv:2605.14041), uniting Kolmogorov-Arnold Networks with Reproducing Kernel Hilbert Space (RKHS) regularization. This provides the finite-sample guarantees missing from standard KANs and acts as a robust verifier against null-space mimicry attacks.

### Phase 4: Capstone Evaluation
Perform live GPU evaluation using the mandated SOTA models (`unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`). The pipeline must run end-to-end with Muon-OGD CSL and optimized ALPS sampling, measuring the reduction in hallucination rates.

## 4. Hardware Requirements
- **2x NVIDIA RTX 3090 (CUDA):** Required for the Phase 4 Capstone live inference.
- **KV260 / OSS-CAD-Suite:** Required for synthesizing the NeuroRing RTL concept.

## 5. Dependency Graph
- Phase 1 (Muon-OGD) -> Phase 4 (Capstone)
- Phase 2 (Optimized ALPS) -> Phase 4 (Capstone)
- Phase 2 (NeuroRing RTL) -> Phase 2 (NeuroRing Synthesis)
- Phase 3 (Wahkon) -> Phase 4 (Capstone)
