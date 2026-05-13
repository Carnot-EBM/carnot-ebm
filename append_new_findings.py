import sys

new_content = """
## 2026-05-13 Post-.157 Planning Sweep (Milestone 2026.05.158)

This sweep was run after milestone `.157` completed. The literature search revealed advances in self-adaptive continuous learning, spatio-temporal KANs, and uncertainty quantification for hallucination detection.

### SEAL: Self-Adaptive Learning for Continuous Improvement
- **Paper:** "SEAL: Self-Adaptive Learning" (NeurIPS 2025).
- **What:** A framework that enables models to generate their own synthetic data post-deployment, allowing continuous self-learning, knowledge repair, and adaptation to real-time data distributions without manual labeling.
- **Relevance to Carnot:** Directly targets the PRD's Continuous Self-Learning (FR-11) requirement. By treating Carnot's deterministic verifier stack as the reward/filter mechanism for SEAL-generated synthetic data, we can achieve safe continuous learning.
- **Concrete experiment hook:** Implement a SEAL-style self-adaptive learning loop where local SOTA models generate reasoning traces, the verifier stack filters them (zero false accepts), and the valid traces are added to the continuous learning buffer.

### STKAN: Spatio-Temporal Decomposition Learning
- **Paper:** "Spatio-Temporal Decomposition Learning with Kolmogorov-Arnold Networks" (ICLR 2026).
- **What:** Extends KANs to spatio-temporal data by modeling dependencies separately, achieving state-of-the-art forecasting accuracy with better interpretability than MLPs.
- **Relevance to Carnot:** Provides a new axis for Tier 4 Adaptive Energy Landscapes, especially for multi-turn reasoning traces which have an inherent temporal (step-by-step) structure.
- **Concrete experiment hook:** Design an STKAN-inspired energy model for scoring sequential constraint reasoning traces.

### Uncertainty Quantification for Hallucination Detection
- **Paper:** "Uncertainty Quantification for Hallucination Detection in LLMs" (2025).
- **What:** Uses latent correctness signals and explicit confidence verbalization to reliably detect hallucinations.
- **Relevance to Carnot:** Complements the "Spilled Energy" telemetry. We can combine latent uncertainty signals with explicit energy scores to route failing generations to the deterministic verifier stack for repair.
"""

with open("research-references.md", "a") as f:
    f.write(new_content)
print("Appended new findings to research-references.md")
