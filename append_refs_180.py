import os

ref_file = "research-references.md"
new_refs = """
### Recent Additions (2026-05-15 Sweep)
- **Energy-Based Transformers are Scalable Learners and Thinkers** (arXiv:2507.02092): Replaces softmax with an energy function for gradient-descent based iterative refinement. Showed a 29% improvement in reasoning tasks.
- **Learning to Discover at Test Time (TTT-Discover)** (arXiv:2601.16175): Uses an entropic utility objective for inference-time verification and exploration to satisfy hard constraints.
- **Kona & Aleph (Logical Intelligence)**: Commercial EBRM paired with Aleph (Lean 4 orchestration) demonstrating 96.2% on expert Sudoku via continuous latent space reasoning rather than autoregressive guessing.
- **A Conceptual and Mathematical Account of a Novel Self-Learning Engine... (Ananta)**: Introduces EB-SLE integrating EBMs with symbolic verification to prevent reward hacking.
- **Self-Distillation Enables Continual Learning** (arXiv:2601.19897): Uses energy-based objectives to prevent catastrophic forgetting in reasoning models during continuous learning.
"""

with open(ref_file, "a") as f:
    f.write(new_refs)

print("Appended references successfully.")
