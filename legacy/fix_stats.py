import re
import os

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix 2097 -> 2111
    content = re.sub(r'2,097', '2,111', content)
    content = re.sub(r'2097', '2111', content)
    
    # Fix Exp 1770 -> Exp 1784
    content = re.sub(r'Exp(\s+)1770', r'Exp\g<1>1784', content)
    
    # Fix 150 -> 151 archived milestone records
    content = re.sub(r'150(\s+)Archived', r'151\g<1>Archived', content)
    content = re.sub(r'150(\s*)</div', r'151\g<1></div', content)
    content = re.sub(r'148(\s+)archived records', r'149\g<1>archived records', content)
    content = re.sub(r'148(\s+)artifact-backed', r'149\g<1>artifact-backed', content)

    # Fix .136 to .137 in texts related to current state
    content = re.sub(r'through 2026\.05\.136', 'through 2026.05.137', content)
    content = re.sub(r'milestone 2026\.05\.136', 'milestone 2026.05.137', content)
    content = re.sub(r'extending through \.136', 'extending through .137', content)
    content = re.sub(r'archive currently stops at \.136', 'archive currently stops at .137', content)

    # Markdown table for README.md
    content = content.replace(
        "| Milestone .136 closeout | .133 | Phase 4 synthesis | Analyzed wall time / experiments for .136 with phase_4_synthesis_complete | Exp 1770 |", 
        "| Milestone .137 closeout | .136 | Phase 4 operations aggregated | Analyzed wall time / experiments for .137 with phase_4_operations_aggregated | Exp 1784 |"
    )
    content = content.replace(
        "| Milestone .136 closeout | Phase 4 operations aggregated; .136 retrospective complete | Exp 1770 |",
        "| Milestone .137 closeout | Phase 4 operations aggregated; .137 retrospective complete | Exp 1784 |"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    fix_file(filepath)
print("Fixes applied.")
