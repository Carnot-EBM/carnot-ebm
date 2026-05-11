import sys

refs = """
## 2026-05-10 Post-.135 Planning Sweep (Milestone 2026.05.136)

This sweep was run after milestone `.135` completed. The literature search revealed advances in energy-based fine-tuning and hierarchical reasoning models for scaling System-2 capabilities.

### Energy-Based Fine-Tuning (EBFT)
- **Paper:** "Energy-Based Fine-Tuning: Sequence-Level Learning without Verifiers" (arXiv:2603.16xxx)
- **What:** Proposes a feature-matching objective that implicitly defines an energy function over entire sequences, providing dense semantic feedback for alignment without needing separate reward models.
- **Relevance to Carnot:** Essential for Continuous Self-Learning, reducing the reliance on external verifiers during continuous training loops.

### Hierarchical Reasoning Model (HRM)
- **Paper:** "Hierarchical Reasoning Model: Brain-Inspired Deep Latent Reasoning" (arXiv:2506.21734)
- **What:** A recurrent architecture separating high-level abstract planning from detailed execution, achieving strong performance on abstraction benchmarks.
- **Relevance to Carnot:** Maps to the Multi-Agent orchestrator framework's need for layered constraints and planning structures.
"""

with open("research-references.md", "a") as f:
    f.write(refs)
