import os

new_refs = """

## 2026.05.124 Milestone Additions (Planning Phase)
- **Energy-Based Dynamical Models for Neurocomputation, Learning, and Optimization** (arXiv:2604.05042, April 2026): Establishes EDMs as a unifying framework for reasoning. Justifies continuous latent optimization.
- **Kona 1.0 / Energy-Based Reasoning Models (EBRMs)** (Jan 2026): Demonstrates reasoning as optimization in continuous latent space with compositional energy functions. Reached 96.2% on Sudoku.
- **Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks** (arXiv:2602.06737, Feb 2026): Framework to replace KAN units with piecewise affine (PWA) functions for MILP-based formal verification.
- **KANELÉ: Kolmogorov-Arnold Networks for Efficient LUT-based Evaluation** (FPGA '26, Feb 2026): Direct instantiation of KANs into FPGA LUT logic, avoiding DSPs.
- **Energy-Guided Decoding for Object Hallucination Mitigation** (arXiv:2507.07731, AAAI 2026): Energy-guided test-time decoding using logit lens and internal energy scores.
- **Energy-Guided Test-Time Scaling (ETS)** (arXiv Jan 2026): Sample from optimal RL policy using online Monte Carlo estimation of energy terms during inference.
- **nabla-Reasoner**: Iterative generation with differentiable textual optimization via EBMs.
"""

with open("research-references.md", "a") as f:
    f.write(new_refs)
