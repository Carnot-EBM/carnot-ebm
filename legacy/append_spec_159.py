import sys

new_content = """
## 2026-05-13 Post-.158 Planning Sweep (Milestone 2026.05.159)

This sweep was run after milestone `.158` completed. The literature search revealed advances in Compositional Energy Minimization, formal verification of KANs via MILP, and Gradient-Guided Epsilon Constraint methods for continual learning.

### Compositional Energy Minimization and Continuous Latent EBRMs
- **Papers:** "Generalizable Reasoning through Compositional Energy Minimization" (NeurIPS 2025 Spotlight) and Kona EBM architecture (2026).
- **What:** Decomposes complex constraints into sub-energies and performs parallel energy minimization (PEM) in a continuous latent space. Reasoning becomes trajectory optimization via energy gradients rather than discrete autoregressive sampling.
- **Relevance to Carnot:** Directly targets the PRD's goal of escaping AR models. Continuous latent reasoning with gradient-based local edits provides a path to Kona-parity constraint satisfaction.
- **Concrete experiment hook:** Prototype a continuous latent EBRM that uses energy gradients to refine reasoning traces, gated by deterministic zero-false-accept verifiers.

### Formal Verification of KANs via MILP Abstractions
- **Paper:** "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs)" (arXiv:2602.06737).
- **What:** Replaces nonlinear KAN splines with Piecewise Affine (PWA) abstractions, allowing network safety properties to be encoded and formally verified using Mixed Integer Linear Programming (MILP).
- **Relevance to Carnot:** Carnot relies on zero-false-accept bounds. Adding MILP verification allows our KAN tiers to be mathematically guaranteed, fulfilling Phase 3 PRD goals.
- **Concrete experiment hook:** Implement PWA abstractions for the existing Carnot KAN models and verify bounds using a MILP solver.

### Gradient-Guided Epsilon Constraints for Continual Learning (GEC)
- **Paper:** "Gradient-Guided Epsilon Constraint Method for Online Continual Learning" (NeurIPS 2025).
- **What:** Prevents catastrophic forgetting by formulating the continual learning update as an $\epsilon$-constraint optimization problem, projecting gradients to strictly maintain performance on previous tasks.
- **Relevance to Carnot:** `.158` introduced the SEAL loop but FR-11 requires rigorous parameter-level protection against drift. GEC gives a principled mathematical foundation for updating the policy buffer.
- **Concrete experiment hook:** Implement GEC for FR-11 policy updates, requiring `utility_delta > 0` and strict `nonforgetting_rate = 1.0` via epsilon constraints.
"""

with open("research-references.md", "a") as f:
    f.write(new_content)
print("Appended new findings to research-references.md")
