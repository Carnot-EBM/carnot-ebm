import re
import os

def update_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w') as f:
        f.write(content)

# Define replacements
replacements = [
    ("2097 Experiments Across", "2111 Experiments Across"),
    ("2,097 tracked experiment records", "2,111 tracked experiment records"),
    ("2,097 experiment records", "2,111 experiment records"),
    ("2,097\nexperiment records", "2,111\nexperiment records"),
    ("2097 tracked experiment records", "2111 tracked experiment records"),
    ("150 Archived Milestone Records", "151 Archived Milestone Records"),
    ("150 Archived completed milestone records", "151 Archived completed milestone records"),
    ("150\nArchived completed milestone records", "151\nArchived completed milestone records"),
    ("through Exp 1770", "through Exp 1784"),
    ("24,113 Python Test Items", "23,946 Python Test Items"),
    ("24,113** Python", "23,946** Python"),
    ("148 archived records through .136", "149 archived records through .137"),
    ("148\nartifact-backed completed milestone records archived through .136", "149\nartifact-backed completed milestone records archived through .137"),
    ("extending through .136", "extending through .137"),
    ("through 2026.05.136", "through 2026.05.137"),
    ("archive currently stops at .136", "archive currently stops at .137"),
    ("Milestone .136 closeout | .133 | Phase 4 synthesis | Analyzed wall time / experiments for .136 with phase_4_synthesis_complete | Exp 1770", "Milestone .137 closeout | .136 | Phase 4 operations aggregated | Analyzed wall time / experiments for .137 with phase_4_operations_aggregated | Exp 1784"),
    ("| Milestone .136 closeout | Phase 4 operations aggregated; .136 retrospective complete | Exp 1770 |", "| Milestone .137 closeout | Phase 4 operations aggregated; .137 retrospective complete | Exp 1784 |")
]

update_file('docs/index.html', replacements)
update_file('README.md', replacements)
update_file('docs/technical-report.md', replacements)

# Append new section to docs/technical-report.md
with open('docs/technical-report.md', 'r') as f:
    tr_content = f.read()

new_section = """
### 4.13 Recent Additions (Milestone .137)
Experiments 1771-1784 advanced the framework through Capstone E2E pipelines with Qwen3.6-35B-A3B and Gemma4-31B-it (Exps 1782, 1783). We implemented Continuous Latent Constraint Modeling (Exp 1771) and evaluated a Differentiable Constraint Memory Bank (Exp 1774). Additionally, the self-learning pipeline was scaled up on LTLZinc (Exp 1777), and we conducted a comprehensive Hardware vs Software Latency and Energy convergence benchmark (Exp 1781), culminating in the .137 operational retrospective.
"""

if "### 4.13 Recent Additions (Milestone .137)" not in tr_content:
    tr_content = tr_content.replace(
        "from the activation-based phase of a research program that now spans",
        new_section + "\nFrom the activation-based phase of a research program that now spans"
    ).replace(
        "From the activation-based phase of a research program that now spans 2,111",
        new_section + "\nFrom the activation-based phase of a research program that now spans 2,111"
    )
    with open('docs/technical-report.md', 'w') as f:
        f.write(tr_content)

print("Updates applied.")
