import sys

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('3,218', '3,234')
    content = content.replace('209 archived records', '211 archived records')
    content = content.replace('209 archived', '211 archived')
    content = content.replace('209 artifact-backed', '211 artifact-backed')
    content = content.replace('25,017', '25,061')
    content = content.replace('2,512', '2,528')
    content = content.replace('archived through .194', 'archived through .197')
    content = content.replace('archived through 2026.05.194', 'archived through 2026.05.197')
    content = content.replace('209 milestones up to .194', '211 milestones up to .197')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.28 Recent Additions (Milestones .195 to .197)

**Dynamic Resolution Continual EBM Learning Prototype & FR-11**  
Experiments 1915-1916 implemented and evaluated the Dynamic Resolution Continual EBM Learning Prototype with Live Data Evaluation for FR-11, and Experiments 1978-1979 later performed a Continuous Self-Learning Retention Audit on the FR-11 loop.

**Compositional Energy Minimization (CEM) Architecture**  
Experiments 1922-1923 introduced the Compositional Energy Minimization (CEM) Architecture Design, along with a Proof of Concept on 3-SAT using a Local SOTA.

**THRML Hybrid Thermodynamic Abstraction & EBT System-2 Decoding**  
Experiments 1970-1973 linked the Phase 1 THRML Hybrid Thermodynamic Abstraction Hookup and performed a THRML vs CPU Gibbs Latency Audit, as well as a Phase 2 EBT System-2 Energy Decoding Baseline and Inference Scaling on GSM8K Subset.
"""

if '### 4.28 Recent Additions' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Done")
