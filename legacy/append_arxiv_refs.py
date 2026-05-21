import os

new_findings = """
## Recent 2025-2026 ArXiv Findings

### Energy-Based Models and Reasoning
- **Energy-Based Transformers (EBTs) (Gladstone et al., 2025):** Frames training as an optimization problem, scaling 35-57% faster than AR models. Energy provides a dense verification signal.
- **Kona (Bodnia & Hanin, 2026):** Non-autoregressive latent reasoning model using gradient edits for constraint satisfaction.
- **Energy-Based Self-Learning Engine (Ghosh, 2025):** Integrates EBT with a Recursive Logic Subsystem (RLS) for autoformalization.

### Constraint Satisfaction
- **HardNet++ (2026):** Differentiable layers for enforcing linear and nonlinear equality/inequality constraints via damped local linearizations.
- **LagONN (2025):** Lagrange Oscillatory Neural Networks for hard constraint combinatorial optimization.

### Kolmogorov-Arnold Networks (KANs)
- **Symbolic-KAN (2026):** Embeds discrete symbolic structures directly into the network to discover closed-form constraints.
- **KANELÉ (2026):** Efficient LUT-based evaluation of KANs for FPGA deployment, yielding 2700x speedups.
- **KANO (2026):** Dual-domain neural operator overcoming spectral bottlenecks.

### Constrained Generation
- **Interleaved Gibbs Diffusion (IGD) (2025):** Hybrid sampling for constrained generation involving mixed continuous-discrete data (e.g., 3-SAT).
"""

with open('research-references.md', 'a') as f:
    f.write(new_findings)
