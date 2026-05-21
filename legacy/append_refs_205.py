import os

content = """
## 2026-05-16 Post-.204 Planning Sweep (Milestone 2026.05.205)

This sweep was run after milestone `.204` completed. The literature search revealed major advances in continuous latent trajectory optimization, test-time gradient descent, agentic RL energy redistribution, and PWA-based formal verification for KANs.

### Energy-Based Reasoning via Structured Latent Planning (EBRM) and $\\nabla$-Reasoner
- **Papers:** "Energy-Based Reasoning via Structured Latent Planning" (arXiv:2603.28248) and "$\\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent" (arXiv:2603.04948).
- **What:** Shifts from discrete token-level CoT to continuous latent trajectory optimization, treating reasoning as an energy minimization problem over latent representations.
- **Relevance to Carnot:** Directly aligns with Carnot's Phase 3 goals of continuous latent reasoning and Kona parity.

### ActFocus: Agentic RL Informed by Token-Level Energy
- **Paper:** "Resolving Action Bottleneck: Agentic RL Informed by Token-Level Energy" (arXiv:2605.14558).
- **What:** An energy-based redistribution mechanism that reweights gradients to improve credit assignment in multi-turn reasoning tasks.
- **Relevance to Carnot:** Directly applicable to our Continuous Self-Learning (CSL) loops (FR-11) for updating policies without forgetting.

### Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs)
- **Paper:** "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs)" (arXiv:2602.06737).
- **What:** Proposes replacing KAN units with Piecewise Affine (PWA) approximations and translating the verification problem into a Mixed Integer Linear Program (MILP).
- **Relevance to Carnot:** Provides the definitive formal verification methodology for our KAN energy tiers.

### Energy-Guided Decoding for Object Hallucination Mitigation
- **Paper:** "Energy-Guided Decoding for Object Hallucination Mitigation" (ICLR 2026).
- **What:** Uses energy scores to dynamically select hidden states from the layer with the minimal energy score to reduce hallucination bias during generation.
- **Relevance to Carnot:** Enhances the generation phase for local SOTA GGUF models.
"""

with open("research-references.md", "a") as f:
    f.write(content)
