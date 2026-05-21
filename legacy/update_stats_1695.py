import os
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Base replacements
    content = content.replace('1,991', '2,010')
    content = content.replace('1991', '2010')
    content = content.replace('Exp 1676', 'Exp 1695')
    content = content.replace('1676**', '1695**')
    
    # Milestone records text specifically
    content = content.replace('142</div><div class="stat-label">archived records through .128', '144</div><div class="stat-label">archived records through .130')
    content = content.replace('142 Archived', '144 Archived')
    
    content = content.replace('1,827 task records', '1,832 task records')
    content = content.replace('142 artifact-backed', '144 artifact-backed')
    content = content.replace('2026.05.125', '2026.05.129')
    content = content.replace('2026.05.128 on 2026-05-09', '2026.05.130 on 2026-05-10')
    content = content.replace('through .128', 'through .130')
    content = content.replace('through 2026.05.128', 'through 2026.05.130')
    
    # Update README table line
    old_row = '| Milestone .126 closeout | Analyzed 151 experiments in 711 mins. Both RTX 3090s idle; 40% savings possible via DualGPURunner parallelization | Exp 1695 |'
    new_row = '| Milestone .130 closeout | Analyzed latest experiments (Exp 1682 to 1695); 8 of 13 criteria complete | Exp 1695 |'
    content = content.replace(old_row, new_row)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.8 Recent Additions (Milestones .129 and .130)

**Kolmogorov-Arnold Attention (KArAt)**  
Experiment 1679 successfully implemented and verified the KArAt attention block prototype, and Experiment 1686 implemented Piecewise Affine (PWA) abstractions for KArAt.

**Deep Energy-Guided Test-Time Scaling**  
Experiment 1690 implemented deep energy-guided test-time scaling integrated with Nabla-Reasoner, leveraging continuous latent optimization dynamics to steer generations.

**Cycle-Accurate Potts Simulation on KV260**  
Experiments 1692 and 1693 successfully completed the Vivado synthesizable Verilog export for a q=3 Potts machine and validated it with cycle-accurate simulations, establishing the hardware pathway for multi-state energy models.
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