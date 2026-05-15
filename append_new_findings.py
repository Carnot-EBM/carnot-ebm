import os

with open('research-references.md', 'r') as f:
    content = f.read()

new_content = """## 2026-05-15 Post-.176 Planning Sweep (Milestone 2026.05.177)

This sweep was run after milestone `.176` completed. The literature search revealed major advances in compositional energy minimization, constraint-aware retrieval, discrete auto-regressive biasing, and KAN hardware acceleration.

### Compositional Energy Minimization (CEM)
- **Paper:** "Generalizable Reasoning through Compositional Energy Minimization" (arXiv:2510.20607).
- **What:** Proposes learning energy landscapes for small, tractable subproblems, which are summed at inference to create a global energy landscape for complex tasks (e.g., 3-SAT) optimized via Parallel Energy Minimization.
- **Relevance to Carnot:** Directly aligns with Carnot's Phase 4 compositional constraint checking.

### ConstraintLLM and Constraint-Aware Retrieval
- **Paper:** "ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming" (arXiv:2510.05774).
- **What:** Fine-tunes LLMs for Constraint Programming using a Constraint-Aware Retrieval Module (CARM) and Tree-of-Thoughts to generate executable CP models.
- **Relevance to Carnot:** Essential for the constraint extraction gap. Incorporating logic-aware retrieval into Carnot's parser pipeline.

### Discrete Auto-Regressive Biasing (DAB)
- **Paper:** "Controlled LLM Decoding via Discrete Auto-regressive Biasing" (arXiv:2502.03685).
- **What:** Identifies that energy-based decoding struggles in continuous space and proposes leveraging gradients in the discrete token domain to improve constraint satisfaction.
- **Relevance to Carnot:** Enhances the energy-guided decoding loop by addressing continuous space limitations during generation.

### KAN Hardware Evaluation (BiKA and KANELÉ)
- **Papers:** BiKA (arXiv:2602.23455) and KANELÉ (arXiv:2512.12850).
- **What:** BiKA proposes a multiply-free KAN architecture using binary learnable thresholds. KANELÉ uses LUTs for high clock-frequency evaluation.
- **Relevance to Carnot:** Gives concrete hardware blueprints to synthesize Carnot's KAN tiers on KV260 FPGAs.

""" + content

with open('research-references.md', 'w') as f:
    f.write(new_content)
