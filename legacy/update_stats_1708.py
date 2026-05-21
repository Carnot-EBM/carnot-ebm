import os
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Base replacements
    content = content.replace('2,010', '2,023')
    content = content.replace('2010', '2023')
    content = content.replace('Exp 1695', 'Exp 1708')
    content = content.replace('1695**', '1708**')
    
    # Task records text
    content = content.replace('1,832 task records', '1,846 task records')
    
    # Update README table line
    old_row = '| Milestone .130 closeout | Analyzed latest experiments (Exp 1682 to 1695); 8 of 13 criteria complete | Exp 1695 |'
    new_row = '| Milestone .131 closeout | Analyzed latest experiments (Exp 1696 to 1708); 1 of 6 criteria complete | Exp 1708 |'
    content = content.replace(old_row, new_row)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.9 Recent Additions (Milestone .131)

**Full Pipeline SOTA integration**  
Experiment 1707 successfully verified the full pipeline SOTA integration combining GloroKAN, Eidoku, and FR11.

**KV260 Hardware Execution Blocked**  
Experiment 1704 attempted to synthesize and execute Potts q=3 on KV260 hardware, but was blocked due to Vivado not being installed.
"""

# Append just before ## 5. Operations and Retrospectives if it exists, else end of file
if '## 5. Operations and' in tr_content:
    tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
else:
    tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Updates applied. Running update_html.py")
subprocess.run(['python3', 'update_html.py'])
print("Done")