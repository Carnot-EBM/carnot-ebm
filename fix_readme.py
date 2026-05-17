import sys

content = """
## Experiments
Carnot tracks an exhaustive, public experiment record to maintain provenance for all claims.
- **Total Experiments:** 2868 (through Exp 2166)
- **Archived Milestones:** 230
- **Tests:** 25,305

## Key Results
| Domain | Model | Result | Note |
|---|---|---|---|
| GSM8K | Gemma-4-E4B-it | Live GPU execution completed | 200 question sample |
| HumanEval | Gemma-4-E4B-it | 50 problems verified | Live execution PBT |
| Adversarial GSM8K | Apple Math | Credibility validation | Verified resistance to superficial changes |
| Process-Reward | PREM Architecture | Dynamic Test-Time Compute (TTC) | Scaled by energy variance |
| Continuous Learning | PREM Motivation | Integration Success | Intrinsic reward for CSL |
| Optimization | ALPS Module | 300x Speedup | Energy -0.842 vs 54.664 |
| Verification | CARM | Constraint-Aware Retrieval | Integration Success |
"""

with open("README.md", "a") as f:
    f.write(content)
