import sys
import datetime

new_content = """## 2026-05-17 Post-.218 Planning Sweep (Milestone 2026.05.219)

This sweep was run after milestone `.218` completed. The literature search revealed advances in energy-guided decoding optimization, state-space energy models, and formal KAN verification.

### Energy-Based Constraint Networks
- **Paper:** "Energy-Based Constraint Networks: Learning Structural Coherence Across Modalities" (arXiv:2605.00960).
- **What:** Introduces a modality-agnostic state-space model that produces scalar energy scores for global consistency, explicitly localizing constraint violations.
- **Relevance to Carnot:** Upgrades the Phase 1 verify-repair constraint extractor by directly providing energy-based structural coherence instead of relying on regex.

### Primal-Dual Guided Decoding
- **Paper:** "Primal-Dual Guided Decoding for Constrained Generation" (arXiv:2605.10).
- **What:** Inference-time method modifying token logits directly via adaptive Lagrangian multipliers, avoiding the overhead of inner Gumbel-softmax loops.
- **Relevance to Carnot:** Drastically accelerates Carnot's Energy-Guided Decoding pipeline in discrete domains without requiring auxiliary model training.

### KAN4CBC and LipKAN
- **Papers:** "Formal Synthesis of Safe KAN Controllers with Barrier Certificates" (IJCAI 2025) and "LipKANs: Lipschitz-Regularized Kolmogorov-Arnold Networks" (NeurIPS 2025/2026).
- **What:** KAN4CBC uses SMT solvers for formal verification of KAN control barrier certificates. LipKAN introduces L1.5-regularization to bound model variance.
- **Relevance to Carnot:** Addresses the Phase 2 goals of formal zero-false-accept guarantees for the Carnot KAN tiers and improves robustness for Continuous Self-Learning (FR-11).

"""

with open("research-references.md", "r") as f:
    old_content = f.read()

with open("research-references.md", "w") as f:
    f.write(new_content + old_content)
