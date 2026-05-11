import re
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('2,153', '2,177')
    content = content.replace('2153', '2177')
    content = content.replace('Exp 1824', 'Exp 1853')
    content = content.replace('1824**', '1853**')
    
    # Update archived records text specifically
    content = content.replace('155</div><div class="stat-label">archived records through .141', '157</div><div class="stat-label">archived records through .144')
    content = content.replace('155 Archived', '157 Archived')
    
    # Update Python test counts
    content = content.replace('24,024', '24,109')

    # Update index.html experiment count text
    content = content.replace('17/17</div><div class="stat-label">experiments completed in .141', '4/4</div><div class="stat-label">experiments completed in .144')
    content = content.replace('completed 17 experiments in 46.0 minutes', 'completed 4 experiments with 2 of 4 gates passed')
    content = content.replace('Exp 1824', 'Exp 1853')
    
    # Update README table line if applicable
    content = content.replace(
        '| Milestone .141 closeout |',
        '| Milestone .144 closeout |'
    )
    content = re.sub(
        r'\| Milestone .141 closeout \|.*?\|',
        r'| Milestone .144 closeout | Analyzed latest experiments; 2 of 4 gates passed | Exp 1853 |',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.12 Recent Additions (Milestones .142 to .144)

**Semantic Pruning in Continual Energy-Based Models**  
Experiment 1849 implemented COCOM pruning, demonstrating that continual EBMs can maintain capacity by selectively pruning semantically redundant constraint connections.

**NLA-Class 16th Verifier Prototype**  
Experiment 1851 deployed a white-box SAE probe achieving a True Positive Rate lift of 0.98 and orthogonal coverage of 10.

**Research Findings Audit**  
Experiment 1852 audited artifacts from .130 through .143, surfacing 80 previously underclaimed results and verifying continuous self-learning constraints.
"""

if '### 4.12 Recent Additions (Milestones .142 to .144)' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running update_html.py")
subprocess.run(['python', 'update_html.py'])
print("Done")