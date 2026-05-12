import re
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('2,241', '2,279')
    content = content.replace('161 archived', '165 archived')
    content = content.replace('161 Archived', '165 Archived')
    content = content.replace('161 artifact-backed', '165 artifact-backed')
    content = content.replace('**161** completed', '**165** completed')
    content = content.replace('**161**\ncompleted', '**165**\ncompleted')
    content = content.replace('161-record', '165-record')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

update_file('docs/technical-report.md')

# Read technical report to append new findings from 152
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.13 Recent Additions (Milestones .145 to .152)

**Ontology NN Topological Constraints**  
Experiment 1955 implemented Ontology NN topological constraints, effectively capturing constraint graph structures for continuous energy landscapes.

**GNN vs Z3 on 3-SAT**  
Experiment 1955 demonstrated that GNN struggles on 3-SAT problem instances when compared directly to the Z3 theorem prover's exact symbolic matching.

**Tri-SOTA E2E v6 Success**  
Experiment 1955 achieved a successful Tri-SOTA E2E v6 milestone, further integrating energy-guided verification with SOTA reasoning paths.
"""

if '### 4.13 Recent Additions' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running scripts/build_technical_report.py")
subprocess.run(['python3', 'scripts/build_technical_report.py'])
print("Done")