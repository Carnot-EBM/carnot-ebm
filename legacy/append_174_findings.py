import os

findings = """
## 2026-05-14 Post-.173 Planning Sweep (Milestone 2026.05.174)

This sweep was run after milestone `.173` completed. The literature search revealed architectural specifics of Kona 1.0, the scale of Extropic Z1, and formalization of operator splitting for convex constraints.

### Kona 1.0 and Energy-Based Reasoning Models (EBRMs)
- **Source:** Logical Intelligence Blog (Jan 2026).
- **What:** Kona 1.0 is a non-autoregressive EBRM that generates complete reasoning traces simultaneously in a continuous latent space. It uses a "global energy score" and gradient-based refinement for continuous trace editing to minimize logical and structural constraint violations.
- **Relevance to Carnot:** Validates the shift away from autoregressive token generation. Provides a blueprint for integrating continuous latent trace editing with Phase 4's CASAL primal-dual samplers.

### Extropic Z1 Production-Scale Hardware
- **Source:** Extropic Writing / Z1 Early Access (2026).
- **What:** The Z1 is a production-scale Thermodynamic Sampling Unit (TSU) using CMOS-manufactured probabilistic circuits (P-bits) to sample from probability distributions via thermal noise, targeting 10,000x efficiency gains.
- **Relevance to Carnot:** Extropic is moving from XTR-0 to Z1. Carnot must prepare the DTM (Denoising Thermodynamic Model) and DTCA architectures in software (via THRML) to interface with the Z1 SDK upon early access.

### $\Pi$Net: Ensuring Satisfaction of Convex Constraints
- **Paper:** "$\Pi$Net: Ensuring Satisfaction of Convex Constraints in Neural Networks" (ICLR 2026 Oral).
- **What:** A layer for convex programs leveraging Douglas-Rachford operator splitting for fast projections and the implicit function theorem for backpropagation.
- **Relevance to Carnot:** While PiNet encountered Gemini CLI routing errors in Exp 2091, its theoretical grounding for convex constraints remains a critical target to revisit for zero-violation safety in generative steps.

### Energy-Based Dynamical Models (EDM)
- **Paper:** "Energy-Based Dynamical Models for Neurocomputation, Learning, and Optimization" (arXiv:2604.05042).
- **What:** Bridges classical Hopfield networks with modern proximal-descent dynamics, framing computation as a relaxation process toward equilibria in an energy landscape.
- **Relevance to Carnot:** Offers a control-theoretic formulation that unifies Phase 3's continuous energy samplers and constraint adherence.
"""

with open('research-references.md', 'a') as f:
    f.write(findings)

print("Findings appended successfully.")
