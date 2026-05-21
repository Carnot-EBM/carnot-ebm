import re
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('2,227', '2,241')
    content = content.replace('Exp 1903', 'Exp 1917')
    content = content.replace('1903**', '1917**')
    
    # Update archived records text
    content = content.replace('161 archived records through .147', '163 archived records through .149')
    content = content.replace('161 Archived', '163 Archived')
    content = content.replace('161 archived', '163 archived')
    
    # Update README table line
    content = content.replace(
        '| Milestone .148 closeout | **1** non-retro task completed, **2** blocked artifacts written, **6** gate-skipped scopes retired, and **4** unexpected missing-artifact failures; SOTA cache/runtime gap unresolved and .147\'s **11%** speedup target not proven | Exp 1903 |',
        '| Milestone .149 closeout | SOTA caching failures blocked terminal artifact recovery in .149 | Exp 1917 |'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.9 Recent Additions (Milestones .148 and .149)

**Non-Autoregressive Constraint Interface Audit**  
Experiment 1912 confirmed that existing validators can be safely wrapped with DummyEnergyExtractionProxy to yield Glauber/Diffusion loop metadata, demonstrating complete compatibility with continuous latent scoring.

**Probability Calibration Verifier**  
Experiment 1414 implemented an opt-in verifier that scores explicit probability claims against simple reference-class evidence.
"""

if '## 5. Operations and' in tr_content:
    tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
else:
    tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running build_technical_report.py")
subprocess.run(['python', 'scripts/build_technical_report.py'])
print("Done")
