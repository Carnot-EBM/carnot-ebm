import re
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('2,686', '2,864')
    content = content.replace('Exp 2154', 'Exp 2166')
    content = content.replace('.212', '.214')
    content = content.replace('226', '227')

    # Fix the paragraph specifically in docs/index.html and docs/technical-report.md
    content = content.replace('Milestone 2026.05.212', 'Milestone 2026.05.214')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.25 Recent Additions (Milestones .213 to .214)

**Process-Reward Energy Model Architecture**  
Experiment 2144 implemented the PREM architecture, and Experiment 2150 added a Dynamic Test-Time Compute (TTC) Controller that successfully scaled TTC based on PREM energy variance.

**Continuous Self-Learning with PREM Intrinsic Motivation**  
Experiment 2152 successfully integrated PREM intrinsic reward signals for continuous self-learning.

**Discrete-to-Ising Translation**  
Experiment 2147 successfully mapped basic AND/OR/NOT clauses to quadratic energy penalties, enabling translation of discrete constraints to Ising.
"""

if '### 4.25 Recent Additions (Milestones .213 to .214)' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running render_html.py")
subprocess.run(['python3', 'render_html.py'])
print("Done")
