import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        if old not in content:
            print(f"WARNING: '{old}' not found in {filepath}")
        content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

index_reps = [
    ('219</div><div class="stat-label">archived records through .205', '220</div><div class="stat-label">archived records through .206'),
    ('27</div><div class="stat-label">experiments completed in .205', '0</div><div class="stat-label">experiments completed in .206'),
    ('Milestone 2026.05.205 Operational Retrospective', 'Milestone 2026.05.206 Operational Retrospective'),
    ('Milestone 2026.05.205 operational retrospective complete. Analyzed 80.4 min wall time / 27 experiments. All tasks were synthesis-only, so GPUs correctly idled at 0% utilization throughout. Synthesis tasks (Exp 2058) remain the primary bottleneck for optimization.', 'Milestone 2026.05.206 operational retrospective complete. Analyzed 0 min wall time / 0 experiments. No experiment commits found since activation of 2026.05.206. There were no compute-bound experiments to analyze, and GPUs were correctly idle. No new bottlenecks were identified as no data was available this milestone.'),
    ('80.4 min wall time</span>', '0 min wall time</span>')
]
update_file('docs/index.html', index_reps)

readme_reps = [
    ('across **219** milestone records (latest 2026.05.205)', 'across **220** milestone records (latest 2026.05.206)'),
    ('| .205 |', '| .206 | Complete | 0 experiments, 0 min wall time. GPUs idle. |\n| .205 |')
]
update_file('README.md', readme_reps)

tr_reps = [
    ('219 Archived Milestone Records', '220 Archived Milestone Records'),
    ('219 milestone records archived through 2026.05.205', '220 milestone records archived through 2026.05.206'),
    ('217 completed milestone records through 2026.05.202', '219 completed milestone records through 2026.05.205')
]
update_file('docs/technical-report.md', tr_reps)

# Append new findings to technical-report.md
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.31 Recent Additions (Milestone .206)

**Milestone 2026.05.206 Operational Retrospective**  
Milestone 2026.05.206 operational retrospective complete. Analyzed 0 min wall time / 0 experiments. No experiment commits found since activation of 2026.05.206. There were no compute-bound experiments to analyze, and GPUs were correctly idle. No new bottlenecks were identified as no data was available this milestone.
"""

if '### 4.31 Recent Additions (Milestone .206)' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Updates applied.")
