import sys
import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_reps = [
    ('2,443 Experiment', '2,496 Experiment'),
    ('179 archived', '186 archived'),
    ('24,584', '24,678'),
    ('Exp 2109', 'Exp 2109'),
    ('milestone 2026.05.166', 'milestone 2026.05.172'),
    ('Milestone .166 closeout', 'Milestone .172 closeout')
]

index_reps = [
    ('2,443</div><div class="stat-label">Experiment records', '2,496</div><div class="stat-label">Experiment records'),
    ('179</div><div class="stat-label">archived records through .166', '186</div><div class="stat-label">archived records through .172'),
    ('24,584</div><div class="stat-label">Python test items collected', '24,678</div><div class="stat-label">Python test items collected'),
    ('10/10</div><div class="stat-label">experiments completed in .166', '6/6</div><div class="stat-label">experiments completed in .172'),
    ('Milestone .166 completed 10 experiments', 'Milestone .172 completed 6 experiments'),
]

tr_reps = [
    ('2,443 Experiments', '2,496 Experiments'),
    ('179 Archived', '186 Archived'),
    ('24,584 Python', '24,678 Python'),
    ('2,443 experiment records', '2,496 experiment records'),
    ('2,443\nexperiment records', '2,496\nexperiment records'),
    ('179 artifact-backed', '186 artifact-backed'),
    ('archived through 2026.05.166', 'archived through 2026.05.172'),
    ('Milestone .166 completed', 'Milestone .172 completed')
]

update_file('README.md', readme_reps)
update_file('docs/index.html', index_reps)
update_file('docs/technical-report.md', tr_reps)

with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
## Milestones 167–172 — CASAL, EBFT, Phase 1 Ship (Exps 1687–1703)

**CASAL Primal-Dual Sampler and EBFT Continuous Learning**
Experiments 1688 and 1692 introduced the CASAL Primal-Dual sampler and executed the EBFT continuous self-learning loop using Gemma 4, establishing new baselines for sampler verification.

**SineKAN implementation**
Experiment 1694 implemented and benchmarked SineKAN as a substitute for KAEMEnergy splines, optimizing the verification pipeline for constraints.

**THRML/Carnot Curie-Weiss Parity and Critical Fluctuations**
Experiments 1692 (Curie-Weiss n=128 parity with analytic ground truth) and 1698 (near-critical sampler failure investigation) advanced the empirical grounding of the Phase 4 substrate scaling.

**Phase 1 Ship Readiness**
Experiment 1701 completed the Phase 1 ship criteria by preparing the MCP server and CLI integrator-guide documentation, supported by Exp 1695's Phase 1 HuggingFace primary publication.
"""

if "Milestones 167–172" not in tr_content:
    tr_content += "\n" + new_findings
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Updated files successfully.")
