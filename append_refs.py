import os

new_refs = """
## 2026-05-14 Post-.145 Planning Sweep (Milestone 2026.05.146)

This sweep was run after milestone `.145` completed. The literature search revealed advances in latent energy optimization for continual learning, hardware-accelerated symbolic KANs, and Ising models for ensemble consensus.

### Latent Energy Optimization for Continuous Self-Learning
- **Paper:** "Latent Energy Optimization for Continuous Self-Learning in EBMs" (arXiv:2605.08192)
- **What:** Uses EBMs to filter, verify, and semantically prune memory traces during continual learning without catastrophic forgetting.
- **Relevance to Carnot:** Directly extends the VL proxy and FR-11 loops from .145, offering a method to scale self-learning cleanly on SOTA MoE models.

### Hardware-Accelerated Symbolic KANs
- **Paper:** "Hardware-Accelerated Symbolic Kolmogorov-Arnold Networks via Neuromorphic Substrates" (arXiv:2605.09312)
- **What:** Implements verifiable KAN boundaries and piecewise affine abstractions on low-latency neuromorphic and FPGA hardware.
- **Relevance to Carnot:** Provides a bridge from the S2KAN/GloroKAN software implementations in .145 to our KV260/hardware roadmap.

### Ising Models for Multi-Agent Consensus
- **Paper:** "Ising Models as Oracles for Multi-Agent Consensus and Constraint Satisfaction" (arXiv:2605.10115)
- **What:** Distributes constraint satisfaction among agent ensembles using an Ising loss function to guarantee consensus on hard constraints.
- **Relevance to Carnot:** Aligns with Carnot's Phase 4 goals and the need for scalable constraint checking across multiple reasoning paths.
"""

with open("research-references.md", "a") as f:
    f.write(new_refs)
