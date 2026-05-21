import os

refs = """
- **[ActFocus](http://arxiv.org/abs/2605.14558v1):** "Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy" - Introduces token-level energy weighting to focus GRPO updates on critical reasoning and action tokens.
- **[KAN-CL](http://arxiv.org/abs/2605.11181v1):** "Per-Knot Importance Regularization for Continual Learning with KANs" - Mitigates catastrophic forgetting in Kolmogorov-Arnold Networks.
- **[Muon-OGD](http://arxiv.org/abs/2604.14818v1):** "Muon-OGD: A Muon-based Spectral Orthogonal Gradient Projection for LLM Continual Learning" - Addresses catastrophic forgetting in LLMs using spectral orthogonal gradient projections.
- **[Mpemba-Thermo](http://arxiv.org/abs/2605.13883v1):** "Digitally Optimized Initializations for Fast Thermodynamic Computing" - Hybrid digital-thermodynamic initialization inspired by the Mpemba effect for accelerating stochastic samplers.
- **[Langevin-Clock](http://arxiv.org/abs/2605.12782v1):** "Adding noise and scaling forces to speed up the Langevin clock" - Accelerates Langevin dynamics for faster sampling.
- **[EBM-RLVR](http://arxiv.org/abs/2605.11059v1):** "A Theoretical Lens for Reinforcement Learning-Tuned Language Models via Energy-Based Models" - Formalizes the equivalence between KL-regularized RL policies and EBM structures.
"""

with open("research-references.md", "a") as f:
    f.write(refs)
