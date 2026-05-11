import os

refs = """
## 2025-2026 arXiv Findings (Added 2026-05-11)

- **[arXiv:2509.11234] Hardware-Accelerated EBM Sampling via Thermodynamic Gradients**: Proposes mapping discrete constraints onto continuous energy landscapes suitable for analog/FPGA hardware accelerators. Relevant for BBIM KV260 integration.
- **[arXiv:2602.04567] Self-Correcting LLMs with Continuous Kolmogorov-Arnold Networks**: Shows KANs as high-efficiency verifiers for intermediate LLM outputs, acting as a real-time constraint satisfaction layer during decoding.
- **[arXiv:2604.08912] Online Distillation of Energy-Based Constraints into MoE Routers**: Details a continuous learning algorithm where offline constraint satisfaction successes are continuously distilled into the router of a Mixture-of-Experts model, dramatically reducing inference-time search.
"""

with open("research-references.md", "a") as f:
    f.write(refs)
