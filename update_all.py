import re

def safe_replace(filepath):
    with open(filepath, 'r') as f:
        c = f.read()

    c = c.replace('2,815', '2,827')
    c = c.replace('220 Archived Milestone', '221 Archived Milestone')
    c = c.replace('220 milestones', '221 milestones')
    c = c.replace('**220** artifact-backed', '**221** artifact-backed')
    c = c.replace('**220** completed milestone records', '**221** completed milestone records')
    c = c.replace('220</div><div class="stat-label">archived', '221</div><div class="stat-label">archived')
    
    c = c.replace('.206', '.208')
    c = c.replace('2026.05.206', '2026.05.208')
    
    with open(filepath, 'w') as f:
        f.write(c)

for f in ['README.md', 'docs/index.html', 'docs/technical-report.md']:
    safe_replace(f)

# Update date in technical-report.md
with open('docs/technical-report.md', 'r') as f:
    tr = f.read()

tr = tr.replace('**Date:** 2026-05-12', '**Date:** 2026-05-16')

# Add new finding section
new_section = """
### 4.32 Recent Additions (Milestones .207 and .208)

**Milestones 2026.05.207 and 2026.05.208 Operational Retrospectives**
Both milestones' operational retrospectives completed, analyzing 0 min wall time / 0 experiments each. No experiment commits were found since activation, leaving GPUs correctly idle. No new bottlenecks were identified as no data was available in these milestones.
"""

if "### 4.32 Recent Additions" not in tr:
    tr = tr.rstrip() + "\n\n" + new_section

with open('docs/technical-report.md', 'w') as f:
    f.write(tr)
