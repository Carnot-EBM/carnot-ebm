import re
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update total experiment counts
    content = content.replace('2,315', '2,354')
    content = content.replace('2315', '2354')
    content = content.replace('Exp 1956', 'Exp 1995')
    content = content.replace('1956**', '1995**')
    
    # Update archived records
    content = content.replace('166</div><div class="stat-label">archived records through .153', '167</div><div class="stat-label">archived records through .154')
    content = content.replace('166 Archived', '167 Archived')
    content = content.replace('166**', '167**')
    content = content.replace('165**', '167**')
    content = content.replace('through 2026.05.147', 'through 2026.05.154')
    content = content.replace('through milestone .148', 'through milestone .154')
    content = content.replace('Milestone .148 completed', 'Milestone .154 completed')
    content = content.replace('milestone 2026.05.152 on 2026-05-12', 'milestone 2026.05.154 on 2026-05-13')

    # Update Python test counts
    content = content.replace('24,330', '24,316')
    content = content.replace('24330', '24316')

    # Update index.html specific stats
    content = content.replace('36/36</div><div class="stat-label">experiments completed in .153', '29/29</div><div class="stat-label">experiments completed in .154')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.13 Recent Additions (Milestones .153 to .155)

**Milestone .154 Operational Retrospective**  
Milestone 2026.05.154 completed 29 experiments in 64.6 minutes. All experiments were synthesis-only (0 GPU usage), resulting in an average duration of 2 minutes per experiment.

**Milestone .155 Planning and Retrospective**  
Experiment 1995 executed the Milestone .155 retrospective. The milestone concluded with a blocked status, recording 6 completed, 4 blocked, and 3 failed tasks, primarily due to contract gaps and the third consecutive failure in the FR-11 Curie-Weiss shipping gate.
"""

if '### 4.13 Recent Additions' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running update_html.py")
subprocess.run(['python3', 'update_html.py'])
print("Done")
