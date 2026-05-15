import sys

new_refs = """

### 2026.05.187 Milestone Additions (Planning Agent)

- **arXiv:2601.04358 (Jan 2026)** — "Energy-Time-Accuracy Tradeoffs in Thermodynamic Computing" (Rolandi et al.). Establishes EDDP limits in stochastic thermodynamic hardware.
- **arXiv:2603.23854 (Mar 2026)** — "Symbolic--KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning". Embeds discrete symbolic structures directly within KANs for closed-form equation discovery.
- **arXiv:2603.00191 (ICML 2026)** — "Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning" (LoDA). Uses projection energy perspective for continual learning decoupling.
- **arXiv:2605.08xxx (May 2026)** — "HEDP: Hybrid Energy-Distance Weighted Prompt Learning for Domain-Incremental Learning". Uses energy regularization loss to enhance separability.
- **ICLR 2026** — "Energy-Based Transformers are Scalable Learners and Thinkers". Proposes EBTs for System 2 thinking via energy minimization at inference time.
- **ICLR 2026** — "Latent-Informed Energy-Based Models with Collaborative Generator Training".
- **AISTATS 2026** — "Unsupervised Ensemble Learning Through Deep Energy-based Models".
- **Preprint (2026)** — "LSEBMCL: A Latent Space Energy-Based Model for Continual Learning". Uses EBMs as outer-generators via Langevin dynamics to prevent catastrophic forgetting.
- **Logical Intelligence (2026)** — "Energy-Based Models for AI Reasoning: Beyond LLM Limitations" (Bodnia, Hanin). Introduces Kona (EBRM).
"""

with open("research-references.md", "a") as f:
    f.write(new_refs)

