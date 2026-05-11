import os
import re

files_to_update = ['README.md', 'docs/technical-report.md', 'docs/index.html']

for filepath in files_to_update:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update experiment IDs and counts
    content = content.replace('Exp 1745', 'Exp 1770')
    content = content.replace('1745**', '1770**')
    content = content.replace('2,072', '2,097')
    content = content.replace('2072', '2097')
    content = content.replace('148 Archived Milestone Records', '150 Archived Milestone Records')
    content = content.replace('148 archived completed milestone records', '150 archived completed milestone records')
    content = content.replace('145 Archived', '147 Archived')
    content = content.replace('145 archived', '147 archived')
    
    # Milestone updates
    content = content.replace('.134 closeout', '.136 closeout')
    content = content.replace('.134', '.136')
    content = content.replace('Experiments 1736-1745', 'Experiments 1736-1770')
    
    # Table updates in README
    if 'Analyzed wall time' in content:
        content = re.sub(
            r'\| Milestone \.136 closeout \| Analyzed wall time [^\|]+ \| Exp 1770 \|',
            r'| Milestone .136 closeout | Phase 4 operations aggregated; .136 retrospective complete | Exp 1770 |',
            content
        )
    
    # Specific to technical-report
    if filepath == 'docs/technical-report.md':
        new_findings = """
### 4.14 Recent Additions (Milestones .135 and .136)

**EBFT Sequence-Level Objective**
Experiment 1759 implemented the EBFT sequence-level objective.

**Reasoning-Time Open Constraint Elicitation (ROCE)**
Experiment 1763 targeted constraints elicited at reasoning time with baseline models remaining unstable.

**Hardware-In-The-Loop Energy Decoding (HILED)**
Experiment 1766 implemented the HILED decoder logic successfully.

**Symbolic-KAN and LTLZinc Spatial Extensions**
Experiments 1749-1753 integrated Symbolic-KAN structure mappings and expanded the LTLZinc benchmark to include spatial reasoning tasks, showing successful end-to-end integration and continual self-learning stability.
"""
        if '## 5. Operations and Retrospectives' in content:
            content = content.replace('## 5. Operations and Retrospectives', new_findings + '\n## 5. Operations and Retrospectives')
        elif '## 5. Operations and' in content:
            content = content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Files updated.")
