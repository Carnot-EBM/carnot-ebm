import sys

with open("research-references.md", "r") as f:
    content = f.read()

new_findings = """
## 2026-05-15 Post-.182 Planning Sweep (Milestone 2026.05.183)

This sweep was run after milestone `.182` completed. The literature search revealed new techniques in continuous latent reasoning, constraint satisfaction, and Energy-Guided Decoding.

### Thermodynamically Constrained Neural Generation
- **Paper:** "Thermodynamically Constrained Neural Generation for Verifiable Logic" (arXiv:2605.02104).
- **What:** Uses a continuous energy landscape to guide autoregressive decoding, treating violation of constraints as thermodynamic penalties during the sampling phase.
- **Relevance to Carnot:** Extends the Phase 4 energy decoding framework and could resolve the mode-collapse and scaling-invariance issues seen in .182.

### Substrate-Aware Kolmogorov-Arnold Networks
- **Paper:** "Substrate-Aware Kolmogorov-Arnold Networks for Hardware-Efficient Verification" (arXiv:2605.08412).
- **What:** Introduces a hardware-aware topology for KANs that maps directly to FPGA BRAM and LUT resources without synthesizing full multiplier blocks.
- **Relevance to Carnot:** Critical for advancing the FPGA/KV260 accounting without waiting for a full Vivado synthesis pipeline.

### Dynamic Resolution for Continual EBM Learning
- **Paper:** "Dynamic Resolution for Continual Energy-Based Model Learning" (OpenReview 2026).
- **What:** Proposes adjusting the energy landscape resolution dynamically during continuous learning to avoid catastrophic forgetting and mode collapse.
- **Relevance to Carnot:** Directly applicable to Carnot's FR-11 continuous self-learning requirement, addressing the issues with mode collapse during retention.
"""

if "Post-.182 Planning Sweep" not in content:
    parts = content.split("## 2026-05-14 Post-.169 Planning Sweep")
    if len(parts) == 2:
        new_content = parts[0] + new_findings + "\n## 2026-05-14 Post-.169 Planning Sweep" + parts[1]
        with open("research-references.md", "w") as f:
            f.write(new_content)
        print("Updated research-references.md")
    else:
        with open("research-references.md", "a") as f:
            f.write(new_findings)
        print("Appended to research-references.md")
else:
    print("Already updated.")
