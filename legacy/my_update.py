import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

index_reps = [
    ('3,295</div><div class="stat-label">Experiment records through Exp 2187', '3,330</div><div class="stat-label">Experiment records through Exp 2214'),
    ('217</div><div class="stat-label">archived records through .202', '219</div><div class="stat-label">archived records through .205'),
    ('25,136</div><div class="stat-label">Python test items collected', '25,193</div><div class="stat-label">Python test items collected'),
    ('10</div><div class="stat-label">experiments completed in .202', '27</div><div class="stat-label">experiments completed in .205'),
    ('Milestone 2026.05.202 Operational Retrospective', 'Milestone 2026.05.205 Operational Retrospective'),
    ('Milestone 2026.05.202 operational retrospective complete. Analyzed 17.1 min wall time / 10 experiments. Execution was stable with 0 compute-bound tasks and 10 synthesis-only tasks. GPUs correctly idled at 0% utilization throughout. Synthesis tasks remain the primary bottleneck for optimization.', 'Milestone 2026.05.205 operational retrospective complete. Analyzed 80.4 min wall time / 27 experiments. All tasks were synthesis-only, so GPUs correctly idled at 0% utilization throughout. Synthesis tasks (Exp 2058) remain the primary bottleneck for optimization.'),
    ('17.1 min wall time</span> <span class="r-after">Exp 2187', '80.4 min wall time</span> <span class="r-after">Exp 2214')
]

readme_reps = [
    ('3,295 Experiment', '3,330 Experiment'),
    ('217 archived', '219 archived'),
    ('25,136', '25,193'),
    ('Exp 2187', 'Exp 2214'),
    ('milestone 2026.05.202', 'milestone 2026.05.205'),
    ('Milestone .202 closeout', 'Milestone .205 closeout')
]

tr_reps = [
    ('3,295 Experiments', '3,330 Experiments'),
    ('217 Archived', '219 Archived'),
    ('25,136 Python', '25,193 Python'),
    ('Through Exp 2187', 'Through Exp 2214'),
    ('3,295 experiments across 217 milestones up to .202', '3,330 experiments across 219 milestones up to .205'),
    ('3,295 experiment records tracked through Exp 2187', '3,330 experiment records tracked through Exp 2214'),
    ('in 217 artifact-backed', 'in 219 artifact-backed'),
    ('archived through 2026.05.202', 'archived through 2026.05.205'),
    ('Milestone 2026.05.202 completed **10** experiments in **17.1** minutes', 'Milestone 2026.05.205 completed **27** experiments in **80.4** minutes')
]

update_file('docs/index.html', index_reps)
update_file('README.md', readme_reps)
update_file('docs/technical-report.md', tr_reps)

with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
## Milestones 192–205 — Synthesis Bottlenecks and Operational Scaling (Exps 2115–2214, May 2026)

**Synthesis-Only Orchestration Optimization**
Across milestones 192 through 205, the pipeline analyzed numerous experiments heavily weighted toward synthesis-only tasks. Operations such as Exp 1970, Exp 1993, and Exp 2058 demonstrated that orchestration and synthesis remain the primary bottlenecks for scaling. Execution stability was confirmed, with GPUs correctly idling at 0% utilization during these synthesis-bound intervals.

**Live Artifact Provenance Tracking**
Routine tracking of live GPU execution confirmed expected behavior without anomalous idling flags. Ongoing updates have maintained strict documentation of the provenance and integrity of hardware acceleration traces.
"""

if "Milestones 192–205" not in tr_content:
    tr_content += "\n" + new_findings
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Updated markdown and index.html")
