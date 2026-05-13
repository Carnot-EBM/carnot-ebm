import sys

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_replacements = [
    ('Exp 2027', 'Exp 2052'),
    ('2,386 Experiment', '2,411 Experiment'),
    ('171 archived', '173 archived'),
    ('2,160 task', '2,200 task'),
    ('2026.05.158', '2026.05.160'),
    ('| Milestone .158 closeout | Analyzed .158 retro. SEAL and STKAN failed, proving negative empirical bounds. | Exp 2027 |', '| Milestone .160 closeout | Continuous execution audits and AIA hardware simulators tested. | Exp 2052 |\n| Milestone .158 closeout | Analyzed .158 retro. SEAL and STKAN failed, proving negative empirical bounds. | Exp 2027 |')
]
update_file('README.md', readme_replacements)

index_replacements = [
    ('2,386</div><div class="stat-label">Experiment records through Exp 2027', '2,411</div><div class="stat-label">Experiment records through Exp 2052'),
    ('171</div><div class="stat-label">archived records through .158', '173</div><div class="stat-label">archived records through .160'),
    ('24,472</div><div class="stat-label">Python test items collected', '24,535</div><div class="stat-label">Python test items collected'),
]
update_file('docs/index.html', index_replacements)

tr_replacements = [
    ('2,386 Experiments', '2,411 Experiments'),
    ('171 Archived', '173 Archived'),
    ('24,472 Python', '24,535 Python'),
    ('Exp 2027', 'Exp 2052'),
    ('171 artifact-backed', '173 artifact-backed'),
    ('2026.05.158', '2026.05.160'),
    ('2,160 task', '2,200 task'),
    ('milestone .158', 'milestone .160'),
    ('Milestone .158 completed', 'Milestone .160 completed')
]
update_file('docs/technical-report.md', tr_replacements)

with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
## Milestones 159–160 — Continuous Execution and Architecture Audits (Exps 2028–2052, May 2026)

**Equilibrium Matching (EqM) Gradient Probing**
Experiment 2041 probed Equilibrium Matching (EqM) gradient landscapes.

**AIA Hardware and Sampler Simulators**
Experiments 2043 and 2044 simulated AIA Knuth-Yao hardware and Gumbel sampling, yielding results favorable to hardware implementation.

**Semantic Compression and Continuous Introspection**
Experiments 2046 and 2048 explored CLaRa Semantic Compression and InEx-style Continuous Introspection prototypes.

**Architectural Coherence Audit**
Experiment 2051 performed an Architectural Coherence Audit for Continuous Execution.
"""

if "Milestones 159–160" not in tr_content:
    tr_content += "\n" + new_findings
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Updated text files successfully.")
