import sys
import re
import subprocess

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_replacements = [
    ('2,501 experiment records tracked', '2,959 experiment records tracked'),
    ('2,359', '2,364'),
    ('191', '192'),
    ('through 2026.05.176', 'through 2026.05.179'),
    ('milestone 2026.05.176', 'milestone 2026.05.179'),
]
update_file('README.md', readme_replacements)

index_replacements = [
    ('2,501</div>', '2,959</div>'),
    ('191</div><div class="stat-label">archived records through .176', '192</div><div class="stat-label">archived records through .179'),
    ('24,761</div>', '24,316</div>'),
    ('24,761 Python test items', '24,316 Python test items'),
    ('10</div><div class="stat-label">experiments completed in .176', '20</div><div class="stat-label">experiments completed in .179'),
    ('Milestone 2026.05.176 Operational', 'Milestone 2026.05.179 Operational'),
    ('Milestone .176 completed 10 experiments in 19.3 minutes.', 'Milestone .179 completed with Phase 3 and Phase 4 findings.')
]
update_file('docs/index.html', index_replacements)

tr_replacements = [
    ('2,501\\nexperiment records tracked', '2,959\\nexperiment records tracked'),
    ('2,501\\n', '2,959\\n'),
    ('2,501 experiment records tracked', '2,959 experiment records tracked'),
    ('2,359', '2,364'),
    ('191** artifact-backed', '192** artifact-backed'),
    ('191 artifact-backed', '192 artifact-backed'),
    ('in 191', 'in 192'),
    ('in **191**', 'in **192**'),
    ('through 2026.05.176', 'through 2026.05.179'),
    ('milestone 2026.05.176', 'milestone 2026.05.179'),
    ('24,761', '24,316'),
]
update_file('docs/technical-report.md', tr_replacements)

# Add new findings
new_findings = """
## Milestones 177–179 — Continuous Self-Learning and KAN Abstractions (Exps 1720–1782, May 2026)

**Continuous Self-Learning Non-Forgetting**
Experiment 1779 implemented non-forgetting soundness checks for continuous learning, and Experiment 1780 ran the FR-11 continuous self-learning loop with rigorous checks.

**NLA-Class Verifier Integration**
Experiment 1720 successfully integrated the ensemble as production verifier #16.

**KANELÉ LUT Abstractions**
Experiments 1781 and 1782 drafted and benchmarked Python LUT abstractions for KANs based on KANELÉ against baselines, demonstrating new Phase 4 hardware-accounting capabilities.

**Phase 4 Alpha Replacement**
Experiment 1721 successfully derived the alpha_t replacement from the maximum-caliber FEP<->IIT bridge, confirming monotonic decay and breaking the bijection-invariance artifact.
"""

with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

if "Milestones 177–179" not in tr_content:
    tr_content += "\\n" + new_findings
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Updated text files. Re-rendering HTML...")
try:
    subprocess.run(['python3', 'update_html.py'], check=True)
    print("Rendered docs/technical-report.html successfully.")
except Exception as e:
    print(f"Error running update_html.py: {e}")
