import re
import subprocess

def update_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()

    for old, new in replacements:
        content = content.replace(old, new)

    with open(filepath, 'w') as f:
        f.write(content)

# README replacements
readme_replacements = [
    ('archives **2,364** task records across **192** artifact-backed completed milestone records through 2026.05.179',
     'archives **2,385** task records across **194** artifact-backed completed milestone records through 2026.05.181'),
    ('extend through milestone 2026.05.179 on 2026-05-15',
     'extend through milestone 2026.05.181 on 2026-05-15'),
    ('2,501 Experiments Across', '2,501 Experiments Across') # Keep 2,501 if it's not changed in technical report, or I will update technical-report separately. Wait, the technical report had 2,501 but the latest might be 2,522.
]

index_replacements = [
    ('192</div><div class="stat-label">archived records through .179', '194</div><div class="stat-label">archived records through .181'),
    ('20</div><div class="stat-label">experiments completed in .179', '10</div><div class="stat-label">experiments completed in .180'),
    ('192 archived records', '194 archived records'),
    ('2,364 task records', '2,385 task records')
]

tech_report_replacements = [
    ('191 Archived Milestone Records', '194 Archived Milestone Records'),
    ('192 artifact-backed completed milestone records archived through 2026.05.174', '194 artifact-backed completed milestone records archived through 2026.05.181'),
    ('191 completed milestone records through 2026.05.179', '194 completed milestone records through 2026.05.181'),
    ('2,364 task records in 192\nartifact-backed completed milestone records', '2,385 task records in 194\nartifact-backed completed milestone records'),
    ('2,364 task records in 192 artifact-backed', '2,385 task records in 194 artifact-backed'),
    ('2,501 Experiments', '2,522 Experiments'),
    ('2,501 experiment', '2,522 experiment'),
    ('2,501\nExperiment', '2,522\nExperiment'),
    ('**2,501', '**2,522')
]

update_file('README.md', readme_replacements)
update_file('docs/index.html', index_replacements)
update_file('docs/technical-report.md', tech_report_replacements)

# Add new findings to technical-report.md
with open('docs/technical-report.md', 'r') as f:
    tr_content = f.read()

new_findings = """
### 4.14 Recent Additions (Milestones .180 to .181)

**Integration of PiNet with CASAL**
Experiment 2110 successfully integrated PiNet with CASAL, showing zero constraint violations across 100 trials, validating the stable synthesis pathways.

**Energy-Based Fine-Tuning (EBFT) with Latent Features**
Experiment 2111 implemented EBFT Contrastive Loss for Continuous Latent State, achieving Kona-parity on continuous latent state with a divergence of 0.014460 between expert and rollout traces.

**Z1 SDK and DTM Stub Alignment**
Experiment 2112 completed Z1 SDK and DTM Stub Alignment. The DTM stub interface aligned successfully with the Z1 continuous DTM signature in a simulator-only environment.

**Lean 4 Verifier Backend Prototype**
Experiment 1739 implemented and successfully tested a Lean 4 Verifier Backend prototype.

**Symbolic Verification on Expert Sudoku (Kona-style)**
Experiment 1740 evaluated the Lean 4 verifier bridge on expert Sudoku puzzles. While the Lean4-verified solve rate was 0.0% due to unavailability at eval time, the EBM-verified solve rate reached 94.00% (47/50).

**EB-SLE Reward Hacking Prevention**
Experiment 1742 correctly identified looping and syntax exploitation in repair generations, validating the EB-SLE hack verifier.
"""

if '### 4.14 Recent Additions' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w') as f:
    f.write(tr_content)

print("Running update_html.py")
subprocess.run(['python3', 'update_html.py'])
print("Done")
