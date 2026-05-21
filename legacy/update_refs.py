import re

with open("research-references.md", "r") as f:
    content = f.read()

new_section = """## 2026-05-11 Post-.142 Planning Sweep (Milestone 2026.05.143)

This sweep was run after milestone `.142` completed. The literature search revealed advances in formal KAN verification, hardware-oriented metrics, gradient-guided online learning, and reasoning-time constraint elicitation.

### Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks
- **Paper:** "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs)" (arXiv:2602.06737)
- **What:** Replaces nonlinear KAN units with piecewise affine (PWA) abstractions. Verification of safety properties is then encoded as a Mixed Integer Linear Program (MILP). Dynamic programming minimizes the linear pieces while staying within error bounds.
- **Relevance to Carnot:** Directly addresses the PRD goal for verifiable EBM tiers. Integrating this gives Carnot a path to formally verify the KAN-guided constraint layers.

### Hardware-Oriented Inference Complexity of KANs
- **Paper:** "Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks" (arXiv:2604.03345)
- **What:** Introduces platform-independent inference complexity metrics for KANs: Real Multiplications (RM), Bit Operations (BOP), and Number of Additions and Bit-Shifts (NABS).
- **Relevance to Carnot:** Essential for estimating hardware requirements before pushing KANs to the KV260 or future FPGA targets.

### Gradient-Guided Epsilon Constraint Method for Online Continual Learning
- **Paper:** "Gradient-Guided Epsilon Constraint Method for Online Continual Learning" (NeurIPS 2025/2026)
- **What:** Treats the preservation of prior knowledge as a hard epsilon constraint rather than a soft penalty in the loss function, solving it iteratively during online learning.
- **Relevance to Carnot:** Addresses the non-forgetting failures seen in FR-11 continuous self-learning loops. Ensures zero-violation policy updates.

### Reasoning-Time Open Constraint Elicitation (ROCE)
- **Paper:** "Reasoning-Time Open Constraint Elicitation for Verifiable LLMs" (arXiv:2605.01124)
- **What:** Extracts dynamic, verifiable logical constraints from unstructured user prompts on-the-fly and enforces them during generation.
- **Relevance to Carnot:** A crucial step for bridging the gap between natural language user instructions and Carnot's strict internal verifiable solvers.

"""

content = re.sub(r'(## \d{4}-\d{2}-\d{2})', new_section + r'\1', content, count=1)

with open("research-references.md", "w") as f:
    f.write(content)
