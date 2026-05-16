import re

with open("docs/index.html", "r") as f:
    html = f.read()

html = html.replace('3,330</div><div class="stat-label">Experiment records through Exp 2214', '2,815</div><div class="stat-label">Experiment records through Exp 2114')
html = html.replace('25,193</div><div class="stat-label">Python test items collected', '25,215</div><div class="stat-label">Python test items collected')
html = html.replace('<span class="r-after">Exp 2214</span>', '<span class="r-after">Exp 2114</span>')

with open("docs/index.html", "w") as f:
    f.write(html)

with open("docs/technical-report.md", "r") as f:
    tr = f.read()

tr = tr.replace('3,330 Experiments Across the Public Record', '2,815 Experiments Across the Public Record')
tr = tr.replace('25,193 Python Test Items Collected (Results and Ops Retros Through Exp 2214)', '25,215 Python Test Items Collected (Results and Ops Retros Through Exp 2114)')
tr = tr.replace('3,330 experiments across 220 milestones', '2,815 experiments across 220 milestones')
tr = tr.replace('3,330 experiment records tracked through Exp 2114', '2,815 experiment records tracked through Exp 2114')
tr = tr.replace('2,583 task records in 220 artifact-backed', '2,584 task records in 220 artifact-backed')

# The prompt also asks to "Add new sections for any major new findings not yet documented."
# The changelog mentions ".206 Operational Retrospective complete". So I'll append a section to docs/technical-report.md if not there.

new_section = """
### Milestone 2026.05.206 Positive Updates
In milestone .206, the operational retrospective completed, analyzing 0 minutes of wall time and 0 experiments. No experiment commits were found since activation, leaving GPUs correctly idle. No new bottlenecks were identified.
"""
if "Milestone 2026.05.206 Positive Updates" not in tr:
    tr += new_section

with open("docs/technical-report.md", "w") as f:
    f.write(tr)
