import os
import subprocess

def replace_in_file(path, old, new):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {path}: replaced '{old}' with '{new}'")

files_to_update = ['README.md', 'docs/technical-report.md', 'docs/index.html']

for filepath in files_to_update:
    replace_in_file(filepath, '2,241 Experiments', '2,279 Experiments')
    replace_in_file(filepath, '2,241 tracked experiment records', '2,279 tracked experiment records')
    replace_in_file(filepath, 'Through Exp 1917', 'Through Exp 1955')
    replace_in_file(filepath, 'Exp 1917', 'Exp 1955')
    replace_in_file(filepath, '1917**', '1955**')
    replace_in_file(filepath, '163 Archived', '165 Archived')
    replace_in_file(filepath, '163 archived', '165 archived')
    replace_in_file(filepath, '24,268', '24,330')
    replace_in_file(filepath, '24268', '24330')

with open('README.md', 'r') as f:
    readme_content = f.read()
    
# After the previous replaces, the line has 'Exp 1955'
old_line = "| Milestone .148 closeout | **1** non-retro task completed, **2** blocked artifacts written, **6** gate-skipped scopes retired, and **4** unexpected missing-artifact failures; SOTA cache/runtime gap unresolved and .147's **11%** speedup target not proven | Exp 1955 |"
new_line = "| Milestone .152 closeout | Tri-SOTA E2E v6 successful, Ontology NN topological constraints implemented, GNN struggles on 3-SAT compared to Z3 | Exp 1955 |"

if old_line in readme_content:
    readme_content = readme_content.replace(old_line, new_line)
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("Updated README.md table")
else:
    print("Warning: old line not found in README.md table!")

# Append new findings to technical-report.md
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.10 Recent Additions (Milestones .150 to .152)

**Ontology NN Topological Constraints**  
Experiment 1946 successfully implemented Forman-Ricci curvature and Deep Delta Learning to enforce ontology-level constraints.

**Integrated Tri-SOTA E2E v6**  
Experiment 1954 successfully completed the integrated Tri-SOTA E2E v6.

**GNN vs. Classical Benchmarking Audit**  
Experiment 1952 demonstrated that the Carnot continuous solver struggles on hard random 3-SAT instances compared to the classical Z3 solver.
"""

if '### 4.10 Recent Additions' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)
    print("Appended findings to docs/technical-report.md")

print("Running build_technical_report.py")
subprocess.run(['python3', 'scripts/build_technical_report.py'])
print("Done")
