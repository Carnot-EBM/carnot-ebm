import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

replacements = [
    # Update index.html
    ('188</div><div class="stat-label">archived records through .174', '191</div><div class="stat-label">archived records through .176'),
    ('4</div><div class="stat-label">experiments completed in .174', '10</div><div class="stat-label">experiments completed in .176'),

    # Update README.md
    ('**181** artifact-backed completed milestone records through 2026.05.174', '**191** artifact-backed completed milestone records through 2026.05.176'),
    ('**2,263** task records', '**2,359** task records'),
    ('extend through milestone 2026.05.174 on 2026-05-15', 'extend through milestone 2026.05.176 on 2026-05-15'),

    # Update technical-report.md
    ('188 Archived Milestone Records', '191 Archived Milestone Records'),
    ('181\nartifact-backed completed milestone records archived through 2026.05.172 and\nchecked-in result artifacts extending through milestone .174', '191\nartifact-backed completed milestone records archived through 2026.05.174 and\nchecked-in result artifacts extending through milestone .176'),
    ('archives **171**\ncompleted milestone records through 2026.05.174', 'archives **191**\ncompleted milestone records through 2026.05.176'),
    ('181 artifact-backed completed milestone records archived through 2026.05.172 and checked-in result artifacts extending through milestone .174', '191 artifact-backed completed milestone records archived through 2026.05.174 and checked-in result artifacts extending through milestone .176'),
    ('archives **171** completed milestone records through 2026.05.174', 'archives **191** completed milestone records through 2026.05.176')
]

update_file('docs/index.html', replacements)
update_file('README.md', replacements)
update_file('docs/technical-report.md', replacements)

# Handle cases with weird newlines in technical-report
with open('docs/technical-report.md', 'r') as f:
    tr = f.read()

tr = re.sub(r'\b188 Archived Milestone Records\b', '191 Archived Milestone Records', tr)
tr = re.sub(r'181\s+artifact-backed completed milestone records archived through 2026\.05\.172 and\s+checked-in result artifacts extending through milestone \.174', '191 artifact-backed completed milestone records archived through 2026.05.174 and checked-in result artifacts extending through milestone .176', tr)
tr = re.sub(r'archives \*\*171\*\*\s+completed milestone records through 2026\.05\.174', 'archives **191** completed milestone records through 2026.05.176', tr)

# README table update
with open('README.md', 'r') as f:
    readme = f.read()
    
# check if .176 is in the table
if "| Milestone .176 closeout |" not in readme:
    readme = readme.replace('| Milestone .152 closeout | Tri-SOTA E2E v6 successful | Exp 1995 |', '| Milestone .176 closeout | Analyzed .176 retro. Synthesis-only tasks remain the primary bottleneck. | Exp 2114 |\n| Milestone .152 closeout | Tri-SOTA E2E v6 successful | Exp 1995 |')

with open('README.md', 'w') as f:
    f.write(readme)

# Append to technical-report
findings = """
## Milestones 166–176 — Operational Retrospectives and Synthesis Bottlenecks (Exps 2105–2114, May 2026)

**Milestone 166 Operational Retrospective**
Analyzed 41 min wall time across 10 experiments. GPUs correctly idled at 0% utilization throughout since all 10 tasks were synthesis-only. The slowest paths were purely synthesis tasks, with Exp 2105 taking 14 minutes.

**Milestone 169 Operational Retrospective**
Analyzed 20.1 min wall time across 11 experiments (avg ~2 min). GPUs correctly idled at 0% utilization throughout. Synthesis tasks remained the primary bottleneck for optimization.

**Milestone 176 Operational Retrospective**
Analyzed 19.3 min wall time across 10 experiments. GPU utilization on the single compute-bound task was efficient, and no anomalous idling was flagged. The slowest path was Exp 1716 (8.7 min, synthesis-only).
"""

if "Milestones 166–176" not in tr:
    tr += "\n" + findings

with open('docs/technical-report.md', 'w') as f:
    f.write(tr)
    
print("Updated basic numbers.")
