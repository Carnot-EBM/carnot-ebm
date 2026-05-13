import sys

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_replacements = [
    ('Exp 2017', 'Exp 2027'),
    ('170 archived', '171 archived'),
    ('| Milestone .147 closeout |', '| Milestone .158 closeout | Analyzed .158 retro. SEAL and STKAN failed, proving negative empirical bounds. | Exp 2027 |\n| Milestone .147 closeout |')
]
update_file('README.md', readme_replacements)

index_replacements = [
    ('2,376</div><div class="stat-label">Experiment records through Exp 2017', '2,386</div><div class="stat-label">Experiment records through Exp 2027'),
    ('170</div><div class="stat-label">archived records through .156', '171</div><div class="stat-label">archived records through .158'),
]
update_file('docs/index.html', index_replacements)

tr_replacements = [
    ('2,376 Experiments Across the Public Record, 170 Archived Milestone Records, 24,472 Python Test Items Collected (Results and Ops Retros Through Exp 2017)', '2,386 Experiments Across the Public Record, 171 Archived Milestone Records, 24,472 Python Test Items Collected (Results and Ops Retros Through Exp 2027)'),
    ('Exp 2017', 'Exp 2027'),
    ('170 Archived', '171 Archived')
]
update_file('docs/technical-report.md', tr_replacements)

with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
## Milestones 156–158 — Structured Verification, Portable Pacts, and Constraints (Exps 2000–2027, May 2026)

**DeepSaDe Guaranteed Constraints**
Experiment 2000 implemented DeepSaDe constraints. Implementation is complete and verified against local baselines.

**EBM Transformer Reasoning Trace Evaluation**
Experiment 2004 delivered an initial audit of reasoning traces for EBM Transformer structures.

**Adaptive Energy Landscapes KAN**
Experiment 2005 deployed a Tier 4 Adaptive Energy Landscapes KAN spline topology updated with +1/-1 knots.

**Milestone .156 / .157 / .158 Operational Findings**
Milestone .156 identified NSVIF/Z3 SMT Constraint Extractor execution time as a primary bottleneck. Milestone .157 proved that Doomed-Rerun blocks successfully saved significant wall-time. Milestone .158 resulted in SEAL and STKAN failures, establishing negative empirical bounds to be addressed in upcoming cycles.
"""

if "DeepSaDe Guaranteed Constraints" not in tr_content:
    tr_content += "\n" + new_findings
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Updated text files successfully.")
