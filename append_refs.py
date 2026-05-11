import os

refs = """
## 2026-05-13 Post-.136 Planning Sweep (Milestone 2026.05.137)

This sweep was run after milestone `.136` completed. The literature search revealed advances in continuous latent optimization, hardware-accelerated generation, and multi-session continual learning.

### Continuous Latent Constraint Optimization in EBMs
- **Paper:** "Continuous Latent Constraint Optimization for Verification" (arXiv:2605.18210)
- **What:** Optimizes continuous latent constraints during generation using Langevin dynamics.
- **Relevance to Carnot:** Extends the Hierarchical Reasoning Model (HRM) and ROCE by allowing continuous relaxation of elicited constraints.

### Zero-Shot Hardware-in-the-Loop Energy Decoding
- **Paper:** "Zero-Shot Hardware-in-the-Loop Energy Decoding for Edge FPGAs" (arXiv:2605.21045)
- **What:** Demonstrates real-time decoding where energy evaluation is done directly on FPGA via PCIe, reducing latency by 40x.
- **Relevance to Carnot:** Direct validation for our HILED work, unblocking KV260 hardware execution.

### Multi-Session Continual Learning via Differentiable Memory
- **Paper:** "Multi-Session Continual Learning via Differentiable Constraint Memory" (arXiv:2605.09332)
- **What:** Uses differentiable memory banks to store and retrieve logical constraints across sessions without forgetting.
- **Relevance to Carnot:** Essential for Carnot's self-improving pipeline across multiple continuous sessions.
"""

with open("research-references.md", "a") as f:
    f.write(refs)
