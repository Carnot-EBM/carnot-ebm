import re

with open("docs/technical-report.md", "r") as f:
    tr = f.read()

new_finding = """
### 4.33 Recent Additions (Milestone .209 to .213)

**Process-Reward Energy Model Architecture (PREM)**
Experiment 2144 successfully implemented the PREM architecture, establishing the foundational Phase 1 framework for subsequent process-reward tasks.

**Dynamic Test-Time Compute (TTC) Controller**
Experiment 2150 successfully implemented a dynamic budget controller capable of scaling Test-Time Compute (TTC) based on PREM energy variance, verifying the Phase 3 capability.

**Continuous Self-Learning with PREM Intrinsic Motivation**
Experiment 2152 evaluated Continuous Self-Learning with PREM intrinsic motivation. The integration was a success, laying groundwork for future test-time adaptations driven by intrinsic energy rewards.
"""

# Try to insert it before "## 5. Operations" or at the end
if "## 5." in tr:
    tr = tr.replace("## 5.", new_finding + "\n## 5.")
elif "## Milestones 192–205" in tr:
    tr = tr + "\n" + new_finding
else:
    tr = tr + "\n" + new_finding

with open("docs/technical-report.md", "w") as f:
    f.write(tr)
