import sys

refs = """
## 2026-05-11 Post-.138 Planning Sweep (Milestone 2026.05.139)

This sweep was run after milestone `.138` completed. The literature search revealed advances in verification of Kolmogorov-Arnold Networks, Energy-Based Transformers (EBTs), and hardware-accelerated thermodynamic sampling.

### Optimal Abstractions for Verifying Properties of KANs
- **Paper:** arXiv:2602.06737, "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks" (Feb 2026).
- **What:** Introduces a framework for verifying KANs by replacing nonlinear units with piecewise affine (PWA) abstractions and encoding the verification problem as a Mixed Integer Linear Program (MILP).
- **Relevance to Carnot:** Carnot relies on deterministic bounds for its verifier layer. This provides a direct path for the KAN energy tier to be formally verified using mature MILP solvers.

### Energy-Based Transformers and Continuous Latent Reasoning
- **Papers:** Energy-Based Transformers (ICLR 2026); Kona 1.0 architecture updates.
- **What:** EBTs assign energy values to candidate predictions, acting as a System 2 reasoner that explicitly minimizes conflict. Kona uses continuous latent space for gradient-based edits.
- **Relevance to Carnot:** Validates the shift toward energy-guided reasoning. Enables using local partial-trace energy to select and repair reasoning traces without just relying on autoregressive next-token prediction.

### Denoising Thermodynamic Models (DTM) and EDDP
- **Papers:** arXiv:2510.23972, "An efficient probabilistic hardware architecture for diffusion-like models"; arXiv:2601.04358, "Energy-Time-Accuracy Tradeoffs in Thermodynamic Computing".
- **What:** DTMs repurpose EBMs as denoising steps for hardware, allowing diffusion-like generation. EDDP introduces an engineering metric for thermodynamic computers balancing energy, delay, and deficiency.
- **Relevance to Carnot:** Provides the algorithmic foundation to evaluate Carnot's EBMs on continuous or diffusion-like processes via `thrml` simulation, bridging toward Extropic hardware paths.
"""

with open("research-references.md", "a") as f:
    f.write(refs)
