import re
import os
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('1,979', '1,991')
    content = content.replace('1979', '1991')
    content = content.replace('Exp 1664', 'Exp 1676')
    content = content.replace('1664**', '1676**')
    
    # Update archived records text specifically
    content = content.replace('140</div><div class="stat-label">archived records through .126', '142</div><div class="stat-label">archived records through .128')
    content = content.replace('140 Archived', '142 Archived')
    
    # Update README table line
    content = content.replace(
        '| Milestone .126 closeout | Analyzed 151 experiments in 711 mins. Both RTX 3090s idle; 40% savings possible via DualGPURunner parallelization | Exp 1664 |',
        '| Milestone .128 closeout | Analyzed latest experiments (Exp 1666 to 1676); 9 of 10 criteria complete | Exp 1676 |'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.7 Recent Additions (Milestones .127 and .128)

**Energy-Guided Decoding (EGD) for Hallucination Mitigation**  
Experiment 1670 verified Energy-Guided Decoding logic against benchmark hallucination triggers. Resulted in a `pass` for targeted mitigation constraints.

**Parallel Inertial Probabilistic Ising Machines (PIPIM) Simulation**  
Experiment 1674 implemented a software simulation of PIPIM logic. Currently CPU-simulator only; no improvement observed over baseline simulated sampling yet.

**Energy-Based Constraint Networks (EBCN) Coherence Score**  
Experiment 1667 integrated EBCN scoring to grade logical traces and state coherence, though the absolute performance improvement metrics remain unspecified pending live scaling.
"""

# Append just before ## 5. Operations and Retrospectives if it exists, else end of file
if '## 5. Operations and' in tr_content:
    tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
else:
    tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running update_html.py")
subprocess.run(['python', 'update_html.py'])
print("Done")