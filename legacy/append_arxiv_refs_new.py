import os

arxiv_findings = """

## Recent ArXiv Findings (2026-05) Added for Milestone 2026.05.215 Planning

- **A Theoretical Lens for RL-Tuned Language Models via Energy-Based Models** (arxiv:2512.18730): Provides a unified variational analysis showing RLVR (verifiable rewards) is equivalent to expected KL minimization toward an optimal reasoning distribution. Important for tuning Carnot's EBM extraction on real models.
- **Adaptive Data Harvesting for Efficient Neural Network Learning with Universal Constraints** (arxiv:2605.09707): Demonstrates learning an adaptive policy for sample selection to enforce continuous constraints. Highly relevant to Continuous Self-Learning (Tier 3).
"""

with open("research-references.md", "a") as f:
    f.write(arxiv_findings)
print("Added references to research-references.md")
