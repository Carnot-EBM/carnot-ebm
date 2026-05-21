content = """
## 2026-05-14 Post-.166 Planning Sweep (Milestone 2026.05.167)

This sweep was run after milestone `.166` completed. The literature search revealed advances in differentiable hard constraint enforcement, formal verification and hardware metrics for KANs, and test-time energy-guided scaling.

### HardNet++ and PiNet: Differentiable Constraint Projection
- **Papers:** "HardNet++: Nonlinear Constraint Enforcement in Neural Networks" (arXiv:2604.19669) and "PiNet: Optimizing Hard-Constrained Neural Networks" (arXiv:2508.10480).
- **What:** Introduces differentiable projection layers (damped local linearizations and Douglas-Rachford splitting) that guarantee hard constraint satisfaction during neural inference, bypassing slow symbolic synthesis.
- **Relevance to Carnot:** Directly targets the "synthesis tasks bottleneck" observed in .166 by providing a differentiable, neural alternative to symbolic constraint resolution.

### KAN Hardware Inference Complexity and Optimal Abstractions
- **Papers:** "Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks" (arXiv:2604.03345) and "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks" (arXiv:2602.06737).
- **What:** Derives platform-independent hardware complexity metrics (RM, BOP, NABS) and uses piecewise affine (PWA) MILP formulations to formally verify KAN robustness.
- **Relevance to Carnot:** Unblocks the KV260 deployment prep for KANs without requiring a full Vivado synthesis run, offering a rigorous accounting framework.

### Energy-Guided Test-Time Scaling and Schema-Constrained Memory
- **Papers:** "ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment" and "Schema-Constrained Generation for Agent Memory" (SCG-MEM) (arXiv:2604.20117).
- **What:** Replaces RLHF with test-time energy-guided probabilities, and enforces strict schema constraints on long-term memory generation.
- **Relevance to Carnot:** Maps directly to Carnot's Continuous Self-Learning (FR-11), ensuring policy updates and memory traces are structurally valid and energy-optimal.
"""
with open("research-references.md", "a") as f:
    f.write(content)
