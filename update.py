import re
import subprocess

def update_file(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            # Try regex if simple replace fails, though we won't strictly need it if careful
            pass

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# Update index.html
index_replacements = [
    ('3,260</div><div class="stat-label">Experiment records through Exp 2152', '3,295</div><div class="stat-label">Experiment records through Exp 2187'),
    ('213</div><div class="stat-label">archived records through .200', '217</div><div class="stat-label">archived records through .202'),
    ('0</div><div class="stat-label">experiments completed in .192', '10</div><div class="stat-label">experiments completed in .202'),
    ('Milestone 2026.05.200 Operational Retrospective', 'Milestone 2026.05.202 Operational Retrospective'),
    ('Milestone 2026.05.200 operational retrospective complete. Analyzed 16.2 min wall time / 10 experiments. Execution was stable with 1 compute-bound task and 9 synthesis-only tasks. No anomalous GPU idling was flagged on the compute-bound task. Synthesis tasks remain the primary bottleneck for optimization.', 
     'Milestone 2026.05.202 operational retrospective complete. Analyzed 17.1 min wall time / 10 experiments. Execution was stable with 0 compute-bound tasks and 10 synthesis-only tasks. GPUs correctly idled at 0% utilization throughout. Synthesis tasks remain the primary bottleneck for optimization.'),
    ('16.2 min wall time</span> <span class="r-after">Exp 2152', '17.1 min wall time</span> <span class="r-after">Exp 2187')
]
update_file('docs/index.html', index_replacements)

# Update README.md
readme_content = """---
license: apache-2.0
---
# Carnot EBM Framework

This project tracks **3,295** experiment records through Exp 2187 across **217** milestone records (latest 2026.05.202).

## Key Results Table
| Milestone | Status | Description |
|---|---|---|
| .202 | Complete | 10 experiments, 17.1 min wall time. GPUs idle. |
| .200 | Complete | 10 experiments, 16.2 min wall time. GPUs efficient. |
| .194 | Complete | 12 experiments, 19.8 min wall time. GPUs idle. |
| .192 | Complete | 0 experiments, 0 min wall time. GPUs idle. |
"""
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

# Update technical-report.md
tr_replacements = [
    ('3,260 Experiments Across the Public Record, 213 Archived Milestone Records', '3,295 Experiments Across the Public Record, 217 Archived Milestone Records'),
    ('(Results and Ops Retros Through Exp 2152)', '(Results and Ops Retros Through Exp 2187)'),
    ('This report summarizes 3,255 experiments across 213 milestones up to .200', 'This report summarizes 3,295 experiments across 217 milestones up to .202'),
    ('**3,260 experiment records tracked through Exp 2152, with 2,548 task records in 213 artifact-backed completed milestone records archived through 2026.05.200**', '**3,295 experiment records tracked through Exp 2187, with 2,583 task records in 217 artifact-backed completed milestone records archived through 2026.05.202**'),
    ('archives **209** completed milestone records through 2026.05.200', 'archives **217** completed milestone records through 2026.05.202'),
    ('Milestone 2026.05.194 completed **12** experiments in **19.8** minutes', 'Milestone 2026.05.202 completed **10** experiments in **17.1** minutes')
]
update_file('docs/technical-report.md', tr_replacements)

# Now, we are asked to "Add new sections for any major new findings not yet documented" in docs/technical-report.md.
# Let's add a small section to the end or before the conclusion. We'll append it for now if we can't find a better place.
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_finding = """
### Milestone 2026.05.202 Synthesis Bottlenecks and Execution Stability
Recent retrospective analysis (Milestones .200 to .202) confirmed that synthesis-only tasks remain the primary bottleneck for orchestration speed. The framework execution was highly stable across the latest 10 experiments (17.1 min wall time) with GPUs correctly idling at 0% utilization throughout the synthesis phase. This identifies a clear opportunity for optimization in the reporting and artifact generation pipelines, rather than the compute-bound paths.
"""
if "### Milestone 2026.05.202 Synthesis Bottlenecks" not in tr_content:
    # Just append it to the document if we don't have a specific place. It fits well as a new subsection.
    with open('docs/technical-report.md', 'a', encoding='utf-8') as f:
        f.write("\n" + new_finding + "\n")

print("Files updated.")
